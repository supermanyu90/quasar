"""Venues as configuration, and the readiness audit that only a venue-aware system can run."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from quasar import schemas
from quasar.readiness import audit
from quasar.venue_spec import VENUES_DIR, VenueSpecError, discover, load_spec

SPEC = json.loads((VENUES_DIR / "national-stadium.json").read_text())


def spec(**overrides) -> dict:
    s = copy.deepcopy(SPEC)
    s.update(overrides)
    return s


class TestSpecLoading(unittest.TestCase):
    def test_every_shipped_venue_loads(self) -> None:
        profiles = discover()
        self.assertGreaterEqual(len(profiles), 2)
        for p in profiles.values():
            self.assertTrue(p.venue.nodes)
            self.assertTrue(p.venue.edges)
            self.assertTrue(p.languages)

    def test_a_venue_spec_is_validated_like_any_other_payload(self) -> None:
        """A venue config is untrusted input. A typo in a corridor width is a wrong
        evacuation time."""
        broken = spec()
        broken["edges"][0]["width_m"] = 0.2  # narrower than a human
        with self.assertRaises(schemas.SchemaError):
            load_spec(broken)

    def test_an_edge_naming_an_unknown_node_is_rejected(self) -> None:
        broken = spec()
        broken["edges"][0]["u"] = "NOWHERE"
        with self.assertRaises(VenueSpecError) as ctx:
            load_spec(broken)
        self.assertIn("unknown node", str(ctx.exception))

    def test_a_disconnected_venue_is_rejected(self) -> None:
        """A stranded stand is far better found here than at 19:40."""
        broken = spec()
        broken["nodes"].append(
            {"id": "ORPHAN", "name": "Orphan", "x": 300, "y": 300,
             "level": 1, "zone": "NORTH", "tags": []}
        )
        with self.assertRaises(VenueSpecError) as ctx:
            load_spec(broken)
        self.assertIn("disconnected", str(ctx.exception))

    def test_a_gate_opening_more_lanes_than_it_has_is_rejected(self) -> None:
        """A schema can tell you 20 is a plausible lane count. Only the referential
        check knows this gate has 12."""
        broken = spec()
        broken["scenario"]["gates"][0]["open_lanes"] = 20  # schema-legal, physically false
        broken["scenario"]["gates"][0]["installed_lanes"] = 12
        with self.assertRaises(VenueSpecError) as ctx:
            load_spec(broken)
        self.assertIn("installed lanes", str(ctx.exception))

    def test_a_scenario_naming_an_unknown_corridor_is_rejected(self) -> None:
        broken = spec()
        broken["scenario"]["edge_density"]["CORR-IMAGINARY"] = 3.0
        with self.assertRaises(VenueSpecError) as ctx:
            load_spec(broken)
        self.assertIn("unknown corridor", str(ctx.exception))


class TestVenuesAreIndependent(unittest.TestCase):
    def test_two_venues_coexist(self) -> None:
        """The zone lookup used to be a module-level global, which quietly asserted
        that only one venue exists in the world. A venue operating system that
        cannot hold two venues in memory at once is not one."""
        venues = discover()
        stadium = venues["national-stadium"]
        arena = venues["coastal-arena"]

        self.assertNotEqual(stadium.venue.nodes.keys(), arena.venue.nodes.keys())
        self.assertNotEqual(stadium.languages, arena.languages)
        self.assertEqual(stadium.venue.node("C-N3").zone, "NORTH")
        self.assertEqual(arena.venue.node("C-NE").zone, "EAST")
        # ...and asking one does not change the other.
        self.assertEqual(stadium.fixture.casualty_node, "C-N3")
        self.assertEqual(arena.fixture.casualty_node, "C-NE")


class TestReadiness(unittest.TestCase):
    def setUp(self) -> None:
        self.venues = discover()

    def test_the_stadium_is_blocked_by_its_ARCHITECTURE(self) -> None:
        """The west stand has a staircase and no ramp. No software fixes that."""
        r = audit(self.venues["national-stadium"])
        blockers = r.by_severity("blocker")

        self.assertFalse(r.ready)
        self.assertTrue(any(b.id.startswith("access.egress") for b in blockers))
        self.assertTrue(any("SEAT-W" in b.detail for b in blockers))

    def test_the_arena_is_blocked_by_its_LANGUAGE(self) -> None:
        """Chennai. The crowd speaks Tamil. Tier-1 Tamil is a machine draft, and
        Quasar will not machine-translate an evacuation order — so tonight, most of
        the people in that building cannot be told to leave in a language they read.

        This is the finding a venue-blind system cannot produce, and it is not an
        engineering defect: it is a translation procurement.
        """
        r = audit(self.venues["coastal-arena"])
        blockers = r.by_severity("blocker")

        self.assertFalse(r.ready)
        self.assertTrue(any(b.id == "lang.tier1.ta" for b in blockers))
        self.assertIn("majority language", blockers[0].detail)
        self.assertIn("procurement", blockers[0].remedy)

    def test_the_arena_architecture_is_sound(self) -> None:
        """Same code, opposite verdicts. The arena is fully step-free -- every stand
        has a ramp -- so it has no accessibility blocker at all, while the stadium
        does."""
        r = audit(self.venues["coastal-arena"])
        self.assertFalse([c for c in r.checks if c.id.startswith("access.egress")])

    def test_the_stadium_can_address_its_own_majority(self) -> None:
        """Mumbai: Hindi and Marathi are human-validated, so the majority language is
        covered. Tamil is present but a minority, so it is critical, not blocking:
        the pictogram and the steward carry it."""
        r = audit(self.venues["national-stadium"])
        lang_blockers = [b for b in r.by_severity("blocker") if b.id.startswith("lang.")]
        self.assertFalse(lang_blockers)
        self.assertTrue(any(c.id == "lang.tier1.ta" for c in r.by_severity("critical")))


if __name__ == "__main__":
    unittest.main()
