"""The deterministic plane: everything a life-safety decision is allowed to depend on.

This module owns the numbers. It reads telemetry and produces an
:class:`Assessment` -- which corridors are at which level of service, which gates
have breached the utilisation trigger, which edges must be cordoned. It also owns
the *executors*: the code that actually computes a medic's route, actually
reallocates lanes, actually verifies that a route does not cross a cordon.

No model output reaches this module. Agents propose an :class:`Assessment`-shaped
opinion and a plan; governance corroborates both against what this module already
computed, and only then are these executors called with parameters the model
selected but did not compute.

The asymmetry is deliberate and is the whole architecture in one sentence: the
model may choose *which* of the venue's medical posts to dispatch from, but the
ETA, the path, and the guarantee that the path does not cross the cordon are
computed here, every time, from the graph.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from quasar.crowd import ADVISORY_DENSITY, CRITICAL_DENSITY, level_of_service
from quasar.queueing import QueueMetrics, analyse_gate, lanes_required
from quasar.routing import (
    FAN,
    RESPONDER,
    NoRouteError,
    Profile,
    Route,
    RouteRequest,
    Router,
)
from quasar.types import (
    EdgeId,
    GateId,
    GateTelemetry,
    LOS,
    NodeId,
    Severity,
    TelemetrySnapshot,
    ZoneId,
)
from quasar.venue import Venue


@dataclass(frozen=True, slots=True)
class Hotspot:
    edge_id: EdgeId
    zone: ZoneId
    density: float
    los: LOS
    trend: str  # rising | steady | falling

    @property
    def critical(self) -> bool:
        return self.density >= CRITICAL_DENSITY


@dataclass(frozen=True, slots=True)
class Assessment:
    """The ground truth for one sensing frame."""

    t: float
    hotspots: tuple[Hotspot, ...]  # density >= ADVISORY_DENSITY, worst first
    gates: Mapping[GateId, QueueMetrics]
    edge_density: Mapping[EdgeId, float]

    @property
    def critical_edges(self) -> tuple[EdgeId, ...]:
        return tuple(h.edge_id for h in self.hotspots if h.critical)

    @property
    def breaching_gates(self) -> tuple[GateId, ...]:
        return tuple(
            g for g, m in sorted(self.gates.items()) if m.breaches_trigger
        )

    def density(self, edge_id: EdgeId) -> float:
        return self.edge_density.get(edge_id, 0.0)

    def hotspot(self, edge_id: EdgeId) -> Hotspot | None:
        return next((h for h in self.hotspots if h.edge_id == edge_id), None)


class DeterministicPlane:
    def __init__(self, venue: Venue) -> None:
        self._venue = venue
        self.router = Router(venue)

    @property
    def venue(self) -> Venue:
        return self._venue

    # -- sensing ----------------------------------------------------------

    def assess(
        self, snapshot: TelemetrySnapshot, *, previous: TelemetrySnapshot | None = None
    ) -> Assessment:
        hotspots: list[Hotspot] = []
        for edge_id, density in snapshot.edge_density.items():
            if edge_id not in self._venue.edges:
                raise KeyError(f"telemetry references unknown edge {edge_id!r}")
            if density < ADVISORY_DENSITY:
                continue
            trend = "steady"
            if previous is not None:
                delta = density - previous.density(edge_id)
                if delta > 0.15:
                    trend = "rising"
                elif delta < -0.15:
                    trend = "falling"
            hotspots.append(
                Hotspot(
                    edge_id=edge_id,
                    zone=self._venue.node(self._venue.edge(edge_id).u).zone,
                    density=density,
                    los=level_of_service(density),
                    trend=trend,
                )
            )
        hotspots.sort(key=lambda h: (-h.density, h.edge_id))

        gates = {gid: analyse_gate(t) for gid, t in snapshot.gates.items()}

        return Assessment(
            t=snapshot.t,
            hotspots=tuple(hotspots),
            gates=gates,
            edge_density=dict(snapshot.edge_density),
        )

    # -- severity floor ---------------------------------------------------

    def severity_floor(self, assessment: Assessment, at_node: NodeId, category: str) -> Severity:
        """The *minimum* severity an incident of this category may be graded at.

        Derived from SOP, not from the model. An agent may grade an incident more
        severely than this; governance rejects any brief that grades it less
        severely, because under-grading is the failure mode that kills people and
        it is precisely the one a fluent model is prone to.
        """
        if category in ("fire", "security"):
            return Severity.P0
        if category == "medical":
            # SOP-MED-03#1: P1 by default, P0 if crowd pressure at the scene
            # exceeds LOS E.
            incident_edges = [
                eid for _, eid in self._venue.neighbours(at_node)
            ]
            worst = max((assessment.density(e) for e in incident_edges), default=0.0)
            return Severity.P0 if worst >= CRITICAL_DENSITY else Severity.P1
        if category in ("crush", "weather"):
            return Severity.P1
        if category == "infrastructure":
            return Severity.P2
        return Severity.P3

    # -- executors --------------------------------------------------------

    def dispatch_route(
        self,
        assessment: Assessment,
        *,
        from_node: NodeId,
        to_node: NodeId,
        cordoned: frozenset[EdgeId] = frozenset(),
    ) -> Route:
        """Compute a responder's route. Raises NoRouteError -- never guesses."""
        return self.router.route(
            RouteRequest(
                origin=from_node,
                destination=to_node,
                profile=RESPONDER,
                cordoned_edges=cordoned,
            ),
            assessment.edge_density,
        )

    def fan_route(
        self,
        assessment: Assessment,
        *,
        from_node: NodeId,
        to_node: NodeId,
        profile: Profile = FAN,
        cordoned: frozenset[EdgeId] = frozenset(),
    ) -> Route:
        return self.router.route(
            RouteRequest(
                origin=from_node,
                destination=to_node,
                profile=profile,
                cordoned_edges=cordoned,
            ),
            assessment.edge_density,
        )

    def nearest_amenity(
        self,
        assessment: Assessment,
        *,
        from_node: NodeId,
        tag: str,
        profile: Profile = FAN,
        cordoned: frozenset[EdgeId] = frozenset(),
    ) -> Route:
        return self.router.nearest_tagged(
            from_node, tag, assessment.edge_density, profile=profile, cordoned_edges=cordoned
        )

    def reallocate_lanes(
        self, telemetry: GateTelemetry, lanes: int
    ) -> QueueMetrics:
        """Re-analyse a gate under a proposed lane count. Raises if infeasible."""
        if lanes > telemetry.installed_lanes:
            raise ValueError(
                f"{telemetry.gate_id}: cannot open {lanes} lanes; only "
                f"{telemetry.installed_lanes} are installed"
            )
        return analyse_gate(
            GateTelemetry(
                gate_id=telemetry.gate_id,
                arrival_rate_per_s=telemetry.arrival_rate_per_s,
                service_rate_per_s=telemetry.service_rate_per_s,
                open_lanes=lanes,
                installed_lanes=telemetry.installed_lanes,
            )
        )

    def lanes_needed(self, telemetry: GateTelemetry) -> int:
        return lanes_required(telemetry)

    def divert(
        self, source: GateTelemetry, target: GateTelemetry, share: float
    ) -> tuple[QueueMetrics, QueueMetrics]:
        """Move ``share`` of the source gate's arrivals to the target gate.

        Returns the post-diversion metrics for both. SOP-QUEUE-02#2 requires the
        target to stay below 0.85 after the move; the caller checks that, and the
        numbers it checks come from here.
        """
        if not 0.0 < share <= 1.0:
            raise ValueError("share must be in (0, 1]")
        moved = source.arrival_rate_per_s * share
        after_source = analyse_gate(
            GateTelemetry(
                source.gate_id,
                source.arrival_rate_per_s - moved,
                source.service_rate_per_s,
                source.open_lanes,
                source.installed_lanes,
            )
        )
        after_target = analyse_gate(
            GateTelemetry(
                target.gate_id,
                target.arrival_rate_per_s + moved,
                target.service_rate_per_s,
                target.open_lanes,
                target.installed_lanes,
            )
        )
        return after_source, after_target

    # -- verification -----------------------------------------------------

    def verify_route(
        self,
        route: Route,
        *,
        cordoned: frozenset[EdgeId],
        step_free: bool = False,
    ) -> None:
        """Post-hoc assertion that an actuated route is safe.

        Belt and braces: the router already guarantees these properties. We check
        them again at the moment of actuation, because the cordon set can change
        between planning and dispatch, and a stale route is exactly as dangerous
        as a wrong one.
        """
        for edge_id in route.edges:
            if edge_id in cordoned:
                raise ValueError(
                    f"route {route.origin}->{route.destination} crosses cordoned edge {edge_id}"
                )
            edge = self._venue.edge(edge_id)
            if step_free and not edge.step_free:
                raise ValueError(
                    f"step-free route {route.origin}->{route.destination} crosses "
                    f"stepped edge {edge_id}"
                )

    def egress_estimate_s(self, assessment: Assessment, from_node: NodeId) -> float:
        """Time for a spectator at ``from_node`` to reach the fastest open gate."""
        best = float("inf")
        for gate in self._venue.nodes_tagged("gate"):
            try:
                route = self.fan_route(
                    assessment, from_node=from_node, to_node=gate.id
                )
            except NoRouteError:
                continue
            best = min(best, route.eta_s)
        return best
