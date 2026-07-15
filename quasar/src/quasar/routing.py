"""Density-aware shortest-path routing.

Dijkstra over the venue graph with an edge cost of traversal *time*, not
distance: an edge's cost is its length divided by the crowd speed its current
density implies (:func:`quasar.crowd.safe_speed`), plus profile-specific
penalties. Because the speed term is strictly positive and bounded below by
``V_MIN_M_S``, every cost is positive and finite and Dijkstra's optimality
argument holds.

Three kinds of constraint are distinguished, and the distinction is the whole
safety story:

* **hard exclusions** -- a cordoned edge, a staff-only edge for a fan, a
  stepped edge for a wheelchair user. These are removed from the graph. If no
  path remains, the router returns ``None`` and the caller must escalate. It
  never silently relaxes a hard constraint.
* **density exclusions** -- an edge above the profile's ``max_density``. Fans
  are excluded from these. Responders are not: a medic must be able to reach a
  casualty *through* the crowd that created the emergency, so for them a dense
  edge is priced with a counterflow penalty rather than removed.
* **soft penalties** -- stairs for an elderly fan, counterflow for a responder.
  These shape the route without forbidding it.

The generative plane never calls into this module's internals. It proposes
*which* route request to make; the answer is computed here, deterministically.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Mapping

from quasar.crowd import CRITICAL_DENSITY, level_of_service, safe_speed
from quasar.types import EdgeId, EdgeKind, LOS, NodeId
from quasar.venue import Venue


@dataclass(frozen=True, slots=True)
class Profile:
    """Who is walking, and what they can and cannot traverse."""

    name: str
    step_free_required: bool = False
    allow_staff_only: bool = False
    # Fans are excluded from edges above this density. None = no exclusion
    # (responders), in which case counterflow_penalty_s_per_m applies instead.
    max_density: float | None = CRITICAL_DENSITY
    # Multiplies free-walking speed: 1.0 = unimpeded adult, 0.6 = assisted.
    speed_factor: float = 1.0
    # Extra seconds per metre when pushing against a dense crowd (responders).
    counterflow_penalty_s_per_m: float = 0.0
    # Flat seconds added per stair edge.
    stair_penalty_s: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.speed_factor <= 1.5:
            raise ValueError("speed_factor must be in (0, 1.5]")
        if self.max_density is not None and self.max_density <= 0.0:
            raise ValueError("max_density must be positive or None")


# The four profiles the venue actually operates with.
FAN = Profile(name="fan")
ACCESSIBLE = Profile(
    name="accessible",
    step_free_required=True,
    # An elderly or wheelchair-using fan should not be sent into a LOS-E crush
    # that an unimpeded adult would tolerate. The lower bound is the point of
    # the profile, not a nicety.
    max_density=1.075,
    speed_factor=0.6,
)
STAFF = Profile(name="staff", allow_staff_only=True, max_density=CRITICAL_DENSITY)
RESPONDER = Profile(
    name="responder",
    allow_staff_only=True,
    max_density=None,  # may push through any density
    speed_factor=1.0,
    counterflow_penalty_s_per_m=0.35,
    stair_penalty_s=5.0,
)


@dataclass(frozen=True, slots=True)
class Route:
    origin: NodeId
    destination: NodeId
    nodes: tuple[NodeId, ...]
    edges: tuple[EdgeId, ...]
    distance_m: float
    eta_s: float
    worst_density: float
    profile: str

    @property
    def worst_los(self) -> LOS:
        return level_of_service(self.worst_density)

    def crosses(self, edge_id: EdgeId) -> bool:
        return edge_id in self.edges


@dataclass(frozen=True, slots=True)
class RouteRequest:
    origin: NodeId
    destination: NodeId
    profile: Profile = FAN
    # Edges closed to everyone -- a cordon, a structural failure, a fire door.
    # A cordon binds responders too; they reach the far side via the service ring.
    cordoned_edges: frozenset[EdgeId] = frozenset()
    # Nodes that must not appear on the path (e.g. the casualty's own footprint).
    avoid_nodes: frozenset[NodeId] = frozenset()


class NoRouteError(Exception):
    """No path satisfies the request's hard constraints.

    Raised rather than returned so a caller cannot accidentally treat "no safe
    route exists" as "route of length zero". Callers must handle it explicitly;
    the governance layer escalates it to a human.
    """

    def __init__(self, request: RouteRequest) -> None:
        super().__init__(
            f"no {request.profile.name} route from {request.origin} to "
            f"{request.destination} satisfies the hard constraints"
        )
        self.request = request


class Router:
    def __init__(self, venue: Venue) -> None:
        self._venue = venue

    def edge_cost(
        self, edge_id: EdgeId, density: float, profile: Profile
    ) -> float | None:
        """Traversal time in seconds, or None if the edge is excluded.

        Kept public so the audit log can record *why* an edge was or was not
        used, and so tests can assert the pricing directly.
        """
        edge = self._venue.edge(edge_id)

        if edge.staff_only and not profile.allow_staff_only:
            return None
        if profile.step_free_required and not edge.step_free:
            return None
        if profile.max_density is not None and density > profile.max_density:
            return None

        speed = safe_speed(density) * profile.speed_factor
        cost = edge.length_m / speed

        if profile.counterflow_penalty_s_per_m > 0.0 and density > 1.0:
            # Penalty scales with how far past comfortable density we are, so a
            # responder prefers the service ring but will still cross a busy
            # concourse if that is genuinely the fastest way to the casualty.
            excess = min(density, 5.4) - 1.0
            cost += edge.length_m * profile.counterflow_penalty_s_per_m * excess

        if edge.kind is EdgeKind.STAIR:
            cost += profile.stair_penalty_s

        return cost

    def route(
        self, request: RouteRequest, density: Mapping[EdgeId, float]
    ) -> Route:
        """Least-time path under the request's constraints. Raises NoRouteError."""
        venue = self._venue
        if request.origin not in venue.nodes:
            raise KeyError(f"unknown origin node {request.origin!r}")
        if request.destination not in venue.nodes:
            raise KeyError(f"unknown destination node {request.destination!r}")
        if request.origin in request.avoid_nodes:
            raise ValueError("origin is in avoid_nodes")

        dist: dict[NodeId, float] = {request.origin: 0.0}
        prev: dict[NodeId, tuple[NodeId, EdgeId]] = {}
        visited: set[NodeId] = set()
        # (cost, node) -- ties broken by node id for a deterministic route.
        heap: list[tuple[float, NodeId]] = [(0.0, request.origin)]

        while heap:
            cost, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            if node == request.destination:
                break

            for neighbour, edge_id in venue.neighbours(node):
                if neighbour in visited or neighbour in request.avoid_nodes:
                    continue
                if edge_id in request.cordoned_edges:
                    continue
                step = self.edge_cost(edge_id, density.get(edge_id, 0.0), request.profile)
                if step is None:
                    continue
                candidate = cost + step
                if candidate < dist.get(neighbour, math.inf):
                    dist[neighbour] = candidate
                    prev[neighbour] = (node, edge_id)
                    heapq.heappush(heap, (candidate, neighbour))

        if request.destination not in dist:
            raise NoRouteError(request)

        nodes: list[NodeId] = [request.destination]
        edges: list[EdgeId] = []
        cursor = request.destination
        while cursor != request.origin:
            parent, edge_id = prev[cursor]
            edges.append(edge_id)
            nodes.append(parent)
            cursor = parent
        nodes.reverse()
        edges.reverse()

        distance = sum(venue.edge(e).length_m for e in edges)
        worst = max((density.get(e, 0.0) for e in edges), default=0.0)

        return Route(
            origin=request.origin,
            destination=request.destination,
            nodes=tuple(nodes),
            edges=tuple(edges),
            distance_m=distance,
            eta_s=dist[request.destination],
            worst_density=worst,
            profile=request.profile.name,
        )

    def nearest_tagged(
        self,
        origin: NodeId,
        tag: str,
        density: Mapping[EdgeId, float],
        *,
        profile: Profile = FAN,
        cordoned_edges: frozenset[EdgeId] = frozenset(),
    ) -> Route:
        """Least-time route to the nearest node carrying ``tag``.

        This is what backs "take me to the nearest accessible washroom": the
        concierge resolves the intent to a tag, the router resolves the tag to
        the reachable node that is actually closest *right now*, given crowd
        density and the fan's mobility profile. Under the ``ACCESSIBLE``
        profile it will never return a node that is only reachable by stairs,
        even if that node is physically nearer.
        """
        best: Route | None = None
        for candidate in self._venue.nodes_tagged(tag):
            try:
                route = self.route(
                    RouteRequest(
                        origin=origin,
                        destination=candidate.id,
                        profile=profile,
                        cordoned_edges=cordoned_edges,
                    ),
                    density,
                )
            except NoRouteError:
                continue
            if best is None or route.eta_s < best.eta_s:
                best = route
        if best is None:
            raise NoRouteError(
                RouteRequest(origin=origin, destination=f"<tag:{tag}>", profile=profile)
            )
        return best
