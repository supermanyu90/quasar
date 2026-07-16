"""Venue graph primitives and public signage labels.

A venue is a navigable graph. Levels: 0 = pitch/service, 1 = main concourse ring,
2 = upper seating bowl. Coordinates are metres on a venue-local grid with the
centre circle at (0, 0); they are used by the router for tie-breaking and by the
BLE positioning filter for trilateration, so they must be geometrically
consistent, not decorative.

Two structural details carry the safety design and every venue honours them:

* ``staff_only`` service corridors form an inner ring that fans are never routed
  through and responders always can be -- this is what lets a medic reach a
  casualty behind a cordon without threading the crowd that caused it;
* every level change exists as a stair and, where the venue is compliant, a ramp
  or lift. ``step_free`` is a hard constraint, not a preference, so an
  accessibility-profiled route either exists or the router says so.

This module holds only the graph *primitives* -- the :class:`Venue` container, the
adjacency builder, and the public-label mapping. The venues themselves are data,
loaded and validated by :mod:`quasar.venue_spec` from the specs in ``venues/``.
There is exactly one definition of each venue, and it lives on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping

from quasar.types import Beacon, Edge, EdgeId, Node, NodeId, ZoneId


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


def build_adjacency(
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
#
# This is the *venue-only* label fallback, used when a MessageCatalogue is built
# from a bare Venue (the governance default). When a full venue spec is available,
# ``quasar.venue_spec`` derives richer per-venue labels from the spec instead.
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
