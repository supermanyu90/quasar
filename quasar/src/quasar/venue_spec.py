"""Venues as data. The system is configured by a venue, not written for one.

A venue operating system that only works for the venue it was written for is a
demo. Quasar reads a **venue spec** -- a JSON document describing the graph, the
zones, the languages its crowd actually speaks, and the fixture it is about to
host -- and configures itself from that. Adding a stadium is a data change.

The spec goes through the same published-schema validator as every model payload
(:mod:`quasar.schemas`), because a venue config is untrusted input too: a typo in
a corridor width is a wrong evacuation time, and a typo in a node id is a route to
a place that does not exist. It is then checked for *referential* integrity --
every edge names real nodes, every gate names a real gate, the graph is connected --
because a schema can tell you a string is well-formed and cannot tell you it points
anywhere.

The three things that vary between venues, and that the system therefore must not
hardcode:

* **the graph** -- an oval stadium and a rectangular indoor arena have completely
  different failure modes, and the second one is not the first one with fewer nodes;
* **the languages the crowd actually speaks** -- which is what makes the Tier-1
  translation gate a *venue readiness* question rather than a code question (see
  :mod:`quasar.readiness`);
* **the fixture** -- the incident, the crowd state, and the fan who is lost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from quasar import schemas
from quasar.types import (
    Beacon,
    Edge,
    EdgeKind,
    GateTelemetry,
    IncidentReport,
    LangCode,
    Node,
    NodeId,
    Role,
    TelemetrySnapshot,
)
from quasar.venue import Venue, build_adjacency


class VenueSpecError(Exception):
    """The venue config is malformed, or it points at things that do not exist."""


@dataclass(frozen=True, slots=True)
class Fixture:
    """The match the venue is about to host, and what goes wrong during it."""

    name: str
    casualty_node: NodeId
    category: str
    background_density: float
    edge_density: Mapping[str, float]
    gates: Mapping[str, GateTelemetry]
    report: IncidentReport
    fan_at_node: NodeId
    fan_seat: NodeId
    fan_language: LangCode
    fan_accessible: bool
    fan_utterance: str

    def snapshot(self, venue: Venue, *, t: float = 1000.0) -> TelemetrySnapshot:
        """Every edge gets a density: the sensed ones, and a quiet floor elsewhere."""
        density = {
            e: self.edge_density.get(e, self.background_density) for e in venue.edges
        }
        return TelemetrySnapshot(
            t=t,
            edge_density=density,
            gates=dict(self.gates),
            source_ids=("cv", "turnstile-log", "ble-gw"),
        )


@dataclass(frozen=True, slots=True)
class VenueProfile:
    """A venue, fully configured: the graph plus everything that varies with it."""

    id: str
    name: str
    fifa_name: str
    city: str
    country: str
    capacity: int
    venue: Venue
    # Languages this venue's crowd actually speaks. Drives the Tier-1 readiness
    # audit: a venue that cannot make a safety announcement in the language of its
    # own majority is not ready, however good its software is.
    languages: tuple[LangCode, ...]
    service_rate_per_s: float
    # "surveyed" (real floor-plan survey) or "representative" (parametric model from
    # public capacity + gate count). Defaults to representative -- the conservative
    # assumption when provenance is unknown. The readiness audit and UI surface it.
    topology: str
    labels: Mapping[str, Mapping[LangCode, str]]
    fixture: Fixture

    @property
    def surveyed(self) -> bool:
        return self.topology == "surveyed"

    @property
    def zones(self) -> frozenset[str]:
        return frozenset(n.zone for n in self.venue.nodes.values())

    @property
    def gate_ids(self) -> frozenset[str]:
        return frozenset(n.id for n in self.venue.nodes_tagged("gate"))


def _labels(spec: Mapping[str, Any], venue: Venue) -> dict[str, dict[str, str]]:
    """id -> {language: public name}. Announcements name places the way the SIGNAGE
    does; the audit log keeps the internal ids."""
    labels: dict[str, dict[str, str]] = {
        zone: dict(by_lang) for zone, by_lang in spec["zones"].items()
    }
    for edge in spec["edges"]:
        labels[edge["id"]] = {"en": edge.get("label", edge["id"])}
    for node in venue.nodes_tagged("gate"):
        # Gates render as their bare number: the template already supplies the word
        # "Gate" in each language, so the noun is translated and the numeral is not.
        labels[node.id] = {"en": "".join(c for c in node.id if c.isdigit()) or node.id}
    return labels


def load_spec(spec: Mapping[str, Any]) -> VenueProfile:
    """Validate a venue spec and build everything downstream from it."""
    schemas.validate(spec, schemas.VENUE_SPEC)

    nodes: dict[NodeId, Node] = {}
    for n in spec["nodes"]:
        if n["id"] in nodes:
            raise VenueSpecError(f"duplicate node id {n['id']!r}")
        nodes[n["id"]] = Node(
            id=n["id"], name=n["name"], x=float(n["x"]), y=float(n["y"]),
            level=int(n["level"]), zone=n["zone"], tags=frozenset(n["tags"]),
            info=n.get("info", ""),
        )

    zones = set(spec["zones"])
    for n in nodes.values():
        if n.zone not in zones:
            raise VenueSpecError(
                f"node {n.id} is in zone {n.zone!r}, which the spec does not define"
            )

    edges: dict[str, Edge] = {}
    for e in spec["edges"]:
        if e["id"] in edges:
            raise VenueSpecError(f"duplicate edge id {e['id']!r}")
        for end in ("u", "v"):
            if e[end] not in nodes:
                raise VenueSpecError(f"edge {e['id']} names unknown node {e[end]!r}")
        if e["u"] == e["v"]:
            raise VenueSpecError(f"edge {e['id']} is a self-loop")
        edges[e["id"]] = Edge(
            id=e["id"], u=e["u"], v=e["v"],
            length_m=float(e["length_m"]), width_m=float(e["width_m"]),
            kind=EdgeKind(e["kind"]), step_free=bool(e["step_free"]),
            staff_only=bool(e.get("staff_only", False)),
        )

    beacons = {}
    for nid in spec["beacons"]:
        if nid not in nodes:
            raise VenueSpecError(f"beacon is mounted on unknown node {nid!r}")
        n = nodes[nid]
        beacons[f"BLE-{nid}"] = Beacon(id=f"BLE-{nid}", node=nid, x=n.x, y=n.y, level=n.level)

    venue = Venue(
        name=spec["name"],
        nodes=nodes,
        edges=edges,
        beacons=beacons,
        _adjacency=build_adjacency(nodes, edges),
    )

    _assert_connected(venue)
    fixture = _fixture(spec["scenario"], venue, float(spec["service_rate_per_s"]))

    return VenueProfile(
        id=spec["id"],
        name=spec["name"],
        fifa_name=spec.get("fifa_name", spec["name"]),
        city=spec["city"],
        country=spec.get("country", ""),
        capacity=int(spec["capacity"]),
        venue=venue,
        languages=tuple(spec["languages"]),
        service_rate_per_s=float(spec["service_rate_per_s"]),
        topology=spec.get("topology", "representative"),
        labels=_labels(spec, venue),
        fixture=fixture,
    )


def _assert_connected(venue: Venue) -> None:
    """Every node must be reachable. A disconnected venue is a stranded crowd, and
    it is far better to find that here than at 19:40."""
    start = next(iter(venue.nodes))
    seen = {start}
    stack = [start]
    while stack:
        for neighbour, _edge in venue.neighbours(stack.pop()):
            if neighbour not in seen:
                seen.add(neighbour)
                stack.append(neighbour)
    if orphaned := set(venue.nodes) - seen:
        raise VenueSpecError(
            f"the venue graph is disconnected; unreachable: {sorted(orphaned)}"
        )


def _fixture(s: Mapping[str, Any], venue: Venue, mu: float) -> Fixture:
    if s["casualty_node"] not in venue.nodes:
        raise VenueSpecError(f"casualty is at unknown node {s['casualty_node']!r}")
    for edge_id in s["edge_density"]:
        if edge_id not in venue.edges:
            raise VenueSpecError(f"scenario sets density on unknown corridor {edge_id!r}")

    gate_ids = {n.id for n in venue.nodes_tagged("gate")}
    gates: dict[str, GateTelemetry] = {}
    for g in s["gates"]:
        if g["gate_id"] not in gate_ids:
            raise VenueSpecError(f"scenario names unknown gate {g['gate_id']!r}")
        if g["open_lanes"] > g["installed_lanes"]:
            raise VenueSpecError(
                f"{g['gate_id']}: cannot open {g['open_lanes']} of "
                f"{g['installed_lanes']} installed lanes"
            )
        gates[g["gate_id"]] = GateTelemetry(
            gate_id=g["gate_id"],
            arrival_rate_per_s=float(g["arrival_rate_per_s"]),
            service_rate_per_s=mu,
            open_lanes=int(g["open_lanes"]),
            installed_lanes=int(g["installed_lanes"]),
        )

    fan = s["fan"]
    for key in ("at_node", "seat"):
        if fan[key] not in venue.nodes:
            raise VenueSpecError(f"fan {key} names unknown node {fan[key]!r}")

    r = s["report"]
    return Fixture(
        name=s["name"],
        casualty_node=s["casualty_node"],
        category=s["category"],
        background_density=float(s["background_density"]),
        edge_density=dict(s["edge_density"]),
        gates=gates,
        report=IncidentReport(
            id=r["id"],
            reporter_role=Role(r["reporter_role"]),
            at_node=s["casualty_node"],
            text=r["text"],
            language=r["language"],
            t=1001.0,
        ),
        fan_at_node=fan["at_node"],
        fan_seat=fan["seat"],
        fan_language=fan["language"],
        fan_accessible=bool(fan["accessible"]),
        fan_utterance=fan["utterance"],
    )


def load_file(path: Path) -> VenueProfile:
    try:
        spec = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise VenueSpecError(f"{path.name}: not valid JSON: {exc}") from exc
    return load_spec(spec)


VENUES_DIR = Path(__file__).resolve().parent.parent.parent / "venues"

# The id of the hand-authored reference stadium, used by tests and the CLI demo.
REFERENCE_VENUE_ID = "national-stadium"


def reference_venue() -> Venue:
    """The hand-authored reference stadium, loaded from its spec.

    There is one definition of it -- ``venues/national-stadium.json`` -- and this is
    how the tests and the CLI demo reach it. It used to be a second, hardcoded copy
    in :mod:`quasar.venue`; that copy drifted from the spec (it never grew the
    amenity nodes the spec has), which is exactly why a single source matters.
    """
    return reference_profile().venue


def reference_profile() -> VenueProfile:
    return load_file(VENUES_DIR / f"{REFERENCE_VENUE_ID}.json")


def discover(directory: Path | None = None) -> dict[str, VenueProfile]:
    """Load every venue spec on disk. A malformed venue fails loudly at startup --
    it does not silently drop out of the list, because an operator who cannot see
    their venue will assume the software is broken rather than their config."""
    directory = directory or VENUES_DIR
    profiles: dict[str, VenueProfile] = {}
    for path in sorted(directory.glob("*.json")):
        profile = load_file(path)
        if profile.id in profiles:
            raise VenueSpecError(f"duplicate venue id {profile.id!r} in {path.name}")
        profiles[profile.id] = profile
    if not profiles:
        raise VenueSpecError(f"no venue specs found in {directory}")
    return profiles
