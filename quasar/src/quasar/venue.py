"""The venue model: a real, navigable graph of a 60,000-seat stadium.

Levels: 0 = pitch/service, 1 = main concourse ring, 2 = upper seating bowl.
Coordinates are metres on a venue-local grid with the centre circle at (0, 0);
they are used by the router for tie-breaking and by the BLE positioning filter
for trilateration, so they have to be geometrically consistent, not decorative.

Two structural details carry the safety design:

* ``staff_only`` service corridors form an inner ring that fans are never routed
  through and responders always can be. This is what lets a medic reach a
  casualty behind a cordon without threading the crowd that caused it.
* every level change exists twice -- as a stair and, where the venue is
  compliant, as a ramp or lift. ``step_free`` is a hard constraint, not a
  preference, so an accessibility-profiled route either exists or the router
  says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

from quasar.types import Beacon, Edge, EdgeId, EdgeKind, Node, NodeId, ZoneId


@dataclass(frozen=True, slots=True)
class Venue:
    name: str
    nodes: Mapping[NodeId, Node]
    edges: Mapping[EdgeId, Edge]
    beacons: Mapping[str, Beacon]
    _adjacency: Mapping[NodeId, tuple[tuple[NodeId, EdgeId], ...]]

    def neighbours(self, node: NodeId) -> tuple[tuple[NodeId, EdgeId], ...]:
        return self._adjacency.get(node, ())

    def edge(self, edge_id: EdgeId) -> Edge:
        return self.edges[edge_id]

    def node(self, node_id: NodeId) -> Node:
        return self.nodes[node_id]

    def nodes_tagged(self, tag: str) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes.values() if tag in n.tags)

    def edges_in_zone(self, zone: ZoneId) -> tuple[Edge, ...]:
        return tuple(
            e
            for e in self.edges.values()
            if self.nodes[e.u].zone == zone or self.nodes[e.v].zone == zone
        )

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes.values())


def _build_adjacency(
    nodes: Mapping[NodeId, Node], edges: Mapping[EdgeId, Edge]
) -> Mapping[NodeId, tuple[tuple[NodeId, EdgeId], ...]]:
    adj: dict[NodeId, list[tuple[NodeId, EdgeId]]] = {n: [] for n in nodes}
    for e in edges.values():
        if e.u not in nodes or e.v not in nodes:
            raise ValueError(f"edge {e.id} references an unknown node")
        adj[e.u].append((e.v, e.id))
        adj[e.v].append((e.u, e.id))  # all pedestrian edges are bidirectional
    return {k: tuple(v) for k, v in adj.items()}


# --------------------------------------------------------------------------
# Node table
# --------------------------------------------------------------------------

_NODES: Sequence[Node] = (
    # Entry gates (level 1, outside the concourse ring)
    Node("G1", "Gate 1 (North-West)", -80, 130, 1, "GATE", frozenset({"gate"})),
    Node("G2", "Gate 2 (North)", 0, 145, 1, "GATE", frozenset({"gate"})),
    Node("G3", "Gate 3 (North-East)", 80, 130, 1, "GATE", frozenset({"gate"})),
    Node("G4", "Gate 4 (East)", 140, -20, 1, "GATE", frozenset({"gate"})),
    Node("G5", "Gate 5 (South)", 0, -145, 1, "GATE", frozenset({"gate"})),
    Node("G6", "Gate 6 (West)", -140, 0, 1, "GATE", frozenset({"gate"})),
    # Main concourse ring (level 1)
    Node("C-N1", "North-West Concourse", -60, 90, 1, "NORTH", frozenset({"concourse"})),
    Node("C-N2", "North Concourse", 0, 100, 1, "NORTH", frozenset({"concourse"})),
    Node("C-N3", "North-East Concourse", 60, 90, 1, "NORTH", frozenset({"concourse"})),
    Node("C-E1", "East Concourse (Upper)", 100, 30, 1, "EAST", frozenset({"concourse"})),
    Node("C-E2", "East Concourse (Lower)", 100, -30, 1, "EAST", frozenset({"concourse"})),
    Node("C-S3", "South-East Concourse", 60, -90, 1, "SOUTH", frozenset({"concourse"})),
    Node("C-S2", "South Concourse", 0, -100, 1, "SOUTH", frozenset({"concourse"})),
    Node("C-S1", "South-West Concourse", -60, -90, 1, "SOUTH", frozenset({"concourse"})),
    Node("C-W2", "West Concourse (Lower)", -100, -30, 1, "WEST", frozenset({"concourse"})),
    Node("C-W1", "West Concourse (Upper)", -100, 30, 1, "WEST", frozenset({"concourse"})),
    # Seating bowl (level 2)
    Node("SEAT-N", "North Stand", 0, 72, 2, "NORTH", frozenset({"seating"})),
    Node("SEAT-E", "East Stand", 78, 0, 2, "EAST", frozenset({"seating"})),
    Node("SEAT-S", "South Stand", 0, -72, 2, "SOUTH", frozenset({"seating"})),
    Node("SEAT-W", "West Stand", -78, 0, 2, "WEST", frozenset({"seating"})),
    # Amenities
    Node("WC-N-ACC", "North Accessible Washroom", -50, 80, 1, "NORTH",
         frozenset({"washroom", "accessible"})),
    Node("WC-N", "North Washroom", 45, 82, 1, "NORTH", frozenset({"washroom"})),
    Node("WC-E-ACC", "East Accessible Washroom", 92, -8, 1, "EAST",
         frozenset({"washroom", "accessible"})),
    Node("WC-S", "South Washroom", -18, -92, 1, "SOUTH", frozenset({"washroom"})),
    Node("FNB-N", "North Food Court", 18, 92, 1, "NORTH", frozenset({"fnb"})),
    Node("FNB-S", "South Food Court", 20, -92, 1, "SOUTH", frozenset({"fnb"})),
    Node("LOST", "Lost & Found", -40, -80, 1, "SOUTH", frozenset({"lost_and_found"})),
    # Medical posts
    Node("MED-1", "Medical Post 1 (North)", -70, 78, 1, "NORTH", frozenset({"medical"})),
    Node("MED-2", "Medical Post 2 (East)", 95, 0, 1, "EAST", frozenset({"medical"})),
    Node("MED-3", "Medical Post 3 (South)", 0, -82, 1, "SOUTH", frozenset({"medical"})),
    # Command and VIP
    Node("CONTROL", "Command Centre", -95, 45, 1, "WEST", frozenset({"control"})),
    Node("VIP", "VIP Box", 85, 60, 2, "EAST", frozenset({"vip"})),
    # Inner service ring (level 0, staff only)
    Node("SVC-NE", "Service Corridor NE", 62, 55, 0, "EAST", frozenset({"service"})),
    Node("SVC-SE", "Service Corridor SE", 62, -55, 0, "EAST", frozenset({"service"})),
    Node("SVC-W", "Service Corridor W", -62, 0, 0, "WEST", frozenset({"service"})),
)

# --------------------------------------------------------------------------
# Edge table. Widths are the constraint that matters: capacity is flow x width.
# --------------------------------------------------------------------------

_E = EdgeKind

_EDGES: Sequence[Edge] = (
    # Gate halls -> concourse
    Edge("E-G1", "G1", "C-N1", 46.0, 8.0, _E.VOMITORY, True),
    Edge("E-G2", "G2", "C-N2", 45.0, 10.0, _E.VOMITORY, True),
    Edge("E-G3", "G3", "C-N3", 45.0, 6.0, _E.VOMITORY, True),
    Edge("E-G4", "G4", "C-E2", 41.0, 8.0, _E.VOMITORY, True),
    Edge("E-G5", "G5", "C-S2", 45.0, 10.0, _E.VOMITORY, True),
    Edge("E-G6", "G6", "C-W2", 50.0, 8.0, _E.VOMITORY, True),
    # Concourse ring. CORR-NE is the pinch point: narrow, and the only fan-side
    # link between the north and east concourses.
    Edge("CORR-N1", "C-N1", "C-N2", 61.0, 9.0, _E.CONCOURSE, True),
    Edge("CORR-N2", "C-N2", "C-N3", 61.0, 9.0, _E.CONCOURSE, True),
    Edge("CORR-NE", "C-N3", "C-E1", 72.0, 4.0, _E.CORRIDOR, True),
    Edge("CORR-E", "C-E1", "C-E2", 60.0, 7.0, _E.CONCOURSE, True),
    Edge("CORR-SE", "C-E2", "C-S3", 72.0, 5.0, _E.CORRIDOR, True),
    Edge("CORR-S2", "C-S3", "C-S2", 61.0, 9.0, _E.CONCOURSE, True),
    Edge("CORR-S1", "C-S2", "C-S1", 61.0, 9.0, _E.CONCOURSE, True),
    Edge("CORR-SW", "C-S1", "C-W2", 72.0, 5.0, _E.CORRIDOR, True),
    Edge("CORR-W", "C-W2", "C-W1", 60.0, 7.0, _E.CONCOURSE, True),
    Edge("CORR-NW", "C-W1", "C-N1", 72.0, 5.0, _E.CORRIDOR, True),
    # Concourse -> seating bowl. Every stand has a stair; only N, E and S have a
    # step-free ramp. The West stand's lift is out of service in this fixture --
    # a real constraint the accessibility router must respect rather than paper over.
    Edge("STAIR-N", "C-N2", "SEAT-N", 32.0, 5.0, _E.STAIR, False),
    Edge("RAMP-N", "C-N1", "SEAT-N", 68.0, 3.0, _E.RAMP, True),
    Edge("STAIR-E", "C-E1", "SEAT-E", 38.0, 5.0, _E.STAIR, False),
    Edge("RAMP-E", "C-E2", "SEAT-E", 72.0, 3.0, _E.RAMP, True),
    Edge("STAIR-S", "C-S2", "SEAT-S", 32.0, 5.0, _E.STAIR, False),
    Edge("RAMP-S", "C-S1", "SEAT-S", 68.0, 3.0, _E.RAMP, True),
    Edge("STAIR-W", "C-W1", "SEAT-W", 38.0, 5.0, _E.STAIR, False),
    # Amenity spurs
    Edge("SP-WCNA", "C-N1", "WC-N-ACC", 14.0, 3.0, _E.CORRIDOR, True),
    Edge("SP-WCN", "C-N3", "WC-N", 17.0, 3.0, _E.STAIR, False),
    Edge("SP-WCEA", "C-E2", "WC-E-ACC", 23.0, 3.0, _E.CORRIDOR, True),
    Edge("SP-WCS", "C-S2", "WC-S", 20.0, 3.0, _E.CORRIDOR, True),
    Edge("SP-FNBN", "C-N2", "FNB-N", 20.0, 4.0, _E.CORRIDOR, True),
    Edge("SP-FNBS", "C-S2", "FNB-S", 22.0, 4.0, _E.CORRIDOR, True),
    Edge("SP-LOST", "C-S1", "LOST", 22.0, 3.0, _E.CORRIDOR, True),
    Edge("SP-MED1", "C-N1", "MED-1", 16.0, 3.0, _E.CORRIDOR, True),
    Edge("SP-MED2", "C-E1", "MED-2", 31.0, 3.0, _E.CORRIDOR, True),
    Edge("SP-MED3", "C-S2", "MED-3", 18.0, 3.0, _E.CORRIDOR, True),
    Edge("SP-CTRL", "C-W1", "CONTROL", 16.0, 3.0, _E.CORRIDOR, True),
    Edge("SP-VIP", "C-E1", "VIP", 40.0, 3.0, _E.RAMP, True),
    # Inner service ring (staff only). Bypasses the fan-side concourse entirely.
    # These run at level 0 beneath the bowl and double back around plant rooms, so
    # their walking length materially exceeds the straight-line distance between
    # their endpoints. That is the trade the router has to get right: the service
    # ring is the LONGER path and, whenever the concourse is crowded, the FASTER one.
    Edge("SVC-1", "MED-2", "SVC-NE", 85.0, 3.0, _E.SERVICE, True, staff_only=True),
    Edge("SVC-2", "SVC-NE", "C-N3", 60.0, 3.0, _E.SERVICE, True, staff_only=True),
    Edge("SVC-3", "MED-2", "SVC-SE", 85.0, 3.0, _E.SERVICE, True, staff_only=True),
    Edge("SVC-4", "SVC-SE", "C-S3", 60.0, 3.0, _E.SERVICE, True, staff_only=True),
    Edge("SVC-5", "SVC-W", "CONTROL", 70.0, 3.0, _E.SERVICE, True, staff_only=True),
    Edge("SVC-6", "SVC-W", "MED-1", 95.0, 3.0, _E.SERVICE, True, staff_only=True),
)

# BLE beacons / Wi-Fi RTT responders, one per concourse and amenity anchor.
_BEACON_NODES: Sequence[NodeId] = (
    "C-N1", "C-N2", "C-N3", "C-E1", "C-E2", "C-S3", "C-S2", "C-S1", "C-W2", "C-W1",
    "G2", "G3", "FNB-N", "MED-2", "WC-N-ACC",
)


# --------------------------------------------------------------------------
# Public labels.
#
# An announcement that says "Corridor CORR-NE is closed" is useless: no spectator
# has ever seen that string. Internal identifiers are for the audit log; the PA
# system gets the name that is painted on the wall the fan is looking at.
#
# Corridor and gate names stay in Latin script deliberately -- they must match the
# physical signage, which is bilingual-with-Latin-numerals, so a fan following the
# announcement and a fan following the signs end up in the same place. Zone names,
# which are words rather than labels, are localised.
# --------------------------------------------------------------------------

ZONE_LABELS: Mapping[ZoneId, Mapping[str, str]] = {
    "NORTH": {"en": "North Stand", "hi": "उत्तर स्टैंड", "mr": "उत्तर स्टँड"},
    "SOUTH": {"en": "South Stand", "hi": "दक्षिण स्टैंड", "mr": "दक्षिण स्टँड"},
    "EAST": {"en": "East Stand", "hi": "पूर्व स्टैंड", "mr": "पूर्व स्टँड"},
    "WEST": {"en": "West Stand", "hi": "पश्चिम स्टैंड", "mr": "पश्चिम स्टँड"},
    "GATE": {"en": "the entry gates", "hi": "प्रवेश द्वार", "mr": "प्रवेश द्वार"},
}

EDGE_LABELS: Mapping[EdgeId, str] = {
    "CORR-NE": "North-East",
    "CORR-NW": "North-West",
    "CORR-SE": "South-East",
    "CORR-SW": "South-West",
    "CORR-N1": "North 1",
    "CORR-N2": "North 2",
    "CORR-S1": "South 1",
    "CORR-S2": "South 2",
    "CORR-E": "East",
    "CORR-W": "West",
}


def public_labels(venue: "Venue") -> Mapping[str, Mapping[str, str]]:
    """id -> {language: public label}, for the message catalogue's slot renderer.

    Gates render as their bare number, because the templates already supply the
    word ("Gate {gate}") in each language -- so the noun gets translated and the
    numeral does not, which is exactly the invariant the entity check enforces.
    """
    labels: dict[str, dict[str, str]] = {}

    for zone, by_lang in ZONE_LABELS.items():
        labels[zone] = dict(by_lang)

    for edge_id in venue.edges:
        labels[edge_id] = {"en": EDGE_LABELS.get(edge_id, edge_id)}

    for node in venue.nodes_tagged("gate"):
        labels[node.id] = {"en": node.id.removeprefix("G")}

    return labels


def build_stadium() -> Venue:
    """Construct the reference venue. Raises if the graph is malformed."""
    nodes = {n.id: n for n in _NODES}
    edges = {e.id: e for e in _EDGES}
    if len(nodes) != len(_NODES):
        raise ValueError("duplicate node id in venue definition")
    if len(edges) != len(_EDGES):
        raise ValueError("duplicate edge id in venue definition")

    beacons = {
        f"BLE-{nid}": Beacon(
            id=f"BLE-{nid}",
            node=nid,
            x=nodes[nid].x,
            y=nodes[nid].y,
            level=nodes[nid].level,
        )
        for nid in _BEACON_NODES
    }

    return Venue(
        name="National Stadium (60,000)",
        nodes=nodes,
        edges=edges,
        beacons=beacons,
        _adjacency=_build_adjacency(nodes, edges),
    )
