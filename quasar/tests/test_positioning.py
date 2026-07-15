"""The graph-constrained particle filter."""

from __future__ import annotations

import random
import unittest

from quasar.positioning import (
    GraphParticleFilter,
    RssiReading,
    expected_rssi,
    simulate_readings,
)
from quasar.venue import build_stadium


class TestPathLoss(unittest.TestCase):
    def test_rssi_falls_with_distance(self) -> None:
        near = expected_rssi(1.0, -45.0)
        far = expected_rssi(30.0, -45.0)
        self.assertAlmostEqual(near, -45.0)
        self.assertLess(far, near)

    def test_reference_distance_floors_the_model(self) -> None:
        """A pedestrian standing on top of a beacon must not produce +inf dBm."""
        self.assertEqual(expected_rssi(0.0, -45.0), expected_rssi(1.0, -45.0))


class TestFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.venue = build_stadium()

    def test_it_converges_on_a_walk_with_no_prior(self) -> None:
        """Seeded uniformly over the whole venue -- the filter does not know where
        the fan is -- it must find them from BLE alone as they walk, and then
        track them.

        The route walked is G2 -> C-N2 -> C-N3 (the north concourse).
        """
        rng = random.Random(11)
        pf = GraphParticleFilter(self.venue, n_particles=1200, seed=3)
        pf.seed_uniform()

        path = [("E-G2", 45.0), ("CORR-N2", 61.0)]
        truth: tuple[float, float, int] = (0.0, 0.0, 1)

        for edge_id, length in path:
            edge = self.venue.edge(edge_id)
            u, v = self.venue.node(edge.u), self.venue.node(edge.v)
            steps = 10
            for i in range(1, steps + 1):
                frac = i / steps
                truth = (
                    u.x + (v.x - u.x) * frac,
                    u.y + (v.y - u.y) * frac,
                    v.level,
                )
                pf.predict(length / steps)
                pf.update(simulate_readings(self.venue, truth[0], truth[1], truth[2], rng=rng))

        fix = pf.estimate()
        error = fix.distance_to(truth[0], truth[1])
        # Raw RSSI trilateration in a stadium is worth about +/- 8 m. Constraining
        # the estimate to the walkable graph buys back most of that.
        self.assertLess(error, 8.0, f"fix error {error:.1f} m at {fix}")
        self.assertEqual(fix.edge_id, "CORR-N2")
        self.assertEqual(fix.nearest_node, "C-N3")
        self.assertGreater(fix.confidence, 0.5)

    def test_a_strong_prior_converges_immediately(self) -> None:
        """The fan tapped in at Gate 3 two seconds ago. We know where they are."""
        rng = random.Random(5)
        pf = GraphParticleFilter(self.venue, n_particles=600, seed=1)
        pf.seed_at_node("G3")

        g3 = self.venue.node("G3")
        pf.update(simulate_readings(self.venue, g3.x, g3.y, g3.level, rng=rng))
        fix = pf.estimate()

        self.assertLess(fix.distance_to(g3.x, g3.y), 6.0)
        self.assertEqual(fix.edge_id, "E-G3")

    def test_the_estimate_is_always_on_a_walkable_edge(self) -> None:
        """The invariant that makes this worth doing: a fix can never land inside
        a wall, because 'inside a wall' is not a representable state."""
        rng = random.Random(2)
        pf = GraphParticleFilter(self.venue, n_particles=400, seed=9)
        pf.seed_uniform()
        for _ in range(12):
            pf.predict(9.0)
            pf.update(simulate_readings(self.venue, 60.0, 90.0, 1, rng=rng))
            fix = pf.estimate()
            self.assertIn(fix.edge_id, self.venue.edges)
            self.assertGreaterEqual(fix.offset, 0.0)
            self.assertLessEqual(fix.offset, 1.0)

    def test_fans_are_never_localised_onto_a_service_corridor(self) -> None:
        pf = GraphParticleFilter(self.venue, n_particles=400, seed=4)
        pf.seed_uniform(walkable_only=True)
        for _ in range(6):
            pf.predict(12.0)
            self.assertFalse(self.venue.edge(pf.estimate().edge_id).staff_only)

    def test_impossible_measurements_reset_rather_than_lie(self) -> None:
        """If every particle is impossible under the measurement, the prior was
        wrong. Reseeding is honest; renormalising zero weights would produce a
        confidently wrong fix."""
        pf = GraphParticleFilter(self.venue, n_particles=200, seed=6)
        pf.seed_at_node("G3")
        # An RSSI vector that no position in the venue can explain.
        absurd = [RssiReading(b, -180.0) for b in list(self.venue.beacons)[:4]]
        pf.update(absurd)
        fix = pf.estimate()  # does not raise, does not divide by zero
        self.assertIn(fix.edge_id, self.venue.edges)

    def test_unknown_beacons_are_ignored_not_guessed(self) -> None:
        pf = GraphParticleFilter(self.venue, n_particles=200, seed=8)
        pf.seed_at_node("G2")
        pf.update([RssiReading("BLE-NOT-A-BEACON", -60.0)])
        self.assertEqual(pf.estimate().edge_id, "E-G2")

    def test_too_few_particles_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            GraphParticleFilter(self.venue, n_particles=4)

    def test_estimating_before_seeding_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            GraphParticleFilter(self.venue).estimate()


if __name__ == "__main__":
    unittest.main()
