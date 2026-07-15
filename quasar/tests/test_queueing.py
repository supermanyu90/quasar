"""M/M/c turnstile queueing."""

from __future__ import annotations

import math
import unittest

from quasar.queueing import (
    REROUTE_TRIGGER,
    analyse_gate,
    divertible_arrivals,
    erlang_b,
    erlang_c,
    lanes_required,
)
from quasar.types import GateTelemetry


def gate(lam: float, mu: float = 0.55, c: int = 10, installed: int = 12) -> GateTelemetry:
    return GateTelemetry("G3", lam, mu, c, installed)


class TestErlang(unittest.TestCase):
    def test_erlang_c_reduces_to_utilisation_for_a_single_server(self) -> None:
        """M/M/1: the probability an arrival waits is exactly rho. A closed-form
        check that the recursion is right."""
        for a in (0.1, 0.35, 0.5, 0.9, 0.99):
            self.assertAlmostEqual(erlang_c(1, a), a, places=9)

    def test_erlang_b_matches_the_factorial_form(self) -> None:
        # B(c, a) = (a^c / c!) / sum_{k=0..c} a^k / k!
        for c, a in ((3, 2.0), (6, 4.5), (10, 9.0)):
            numerator = a**c / math.factorial(c)
            denominator = sum(a**k / math.factorial(k) for k in range(c + 1))
            self.assertAlmostEqual(erlang_b(c, a), numerator / denominator, places=10)

    def test_saturated_queue_always_waits(self) -> None:
        self.assertEqual(erlang_c(4, 4.0), 1.0)
        self.assertEqual(erlang_c(4, 9.0), 1.0)

    def test_zero_lanes_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            erlang_c(0, 1.0)


class TestGateAnalysis(unittest.TestCase):
    def test_utilisation_is_lambda_over_c_mu(self) -> None:
        m = analyse_gate(gate(5.4))
        self.assertAlmostEqual(m.utilisation, 5.4 / (10 * 0.55), places=9)
        self.assertTrue(m.stable)

    def test_trigger_fires_at_0_90(self) -> None:
        below = analyse_gate(gate(0.89 * 10 * 0.55))
        at = analyse_gate(gate(0.90 * 10 * 0.55))
        self.assertFalse(below.breaches_trigger)
        self.assertTrue(at.breaches_trigger)
        self.assertAlmostEqual(at.utilisation, REROUTE_TRIGGER, places=9)

    def test_unstable_queue_is_flagged_not_silently_infinite(self) -> None:
        m = analyse_gate(gate(7.0, c=10))  # lambda > c*mu
        self.assertFalse(m.stable)
        self.assertTrue(m.breaches_trigger)
        self.assertEqual(m.wait_s, math.inf)

    def test_wait_grows_sharply_as_utilisation_approaches_one(self) -> None:
        """The justification for a 0.90 trigger rather than a 0.99 one."""
        w90 = analyse_gate(gate(0.90 * 10 * 0.55)).wait_s
        w98 = analyse_gate(gate(0.98 * 10 * 0.55)).wait_s
        self.assertGreater(w98, 4 * w90)

    def test_cannot_open_more_lanes_than_installed(self) -> None:
        with self.assertRaises(ValueError):
            analyse_gate(GateTelemetry("G3", 3.0, 0.55, 14, 12))

    def test_zero_service_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            analyse_gate(GateTelemetry("G3", 3.0, 0.0, 4, 12))


class TestMitigation(unittest.TestCase):
    def test_lanes_required_actually_clears_the_trigger(self) -> None:
        saturated = gate(5.4, c=10)
        self.assertTrue(analyse_gate(saturated).breaches_trigger)

        needed = lanes_required(saturated)
        self.assertGreater(needed, saturated.open_lanes)

        fixed = analyse_gate(
            GateTelemetry("G3", 5.4, 0.55, needed, max(needed, 12))
        )
        self.assertFalse(fixed.breaches_trigger)
        self.assertLess(fixed.utilisation, REROUTE_TRIGGER)

        # ...and it is the *smallest* such lane count.
        one_fewer = analyse_gate(
            GateTelemetry("G3", 5.4, 0.55, needed - 1, max(needed, 12))
        )
        self.assertTrue(one_fewer.breaches_trigger)

    def test_divertible_arrivals_is_zero_when_the_gate_can_cope(self) -> None:
        self.assertEqual(divertible_arrivals(gate(2.0)), 0.0)

    def test_divertible_arrivals_is_positive_when_every_lane_is_not_enough(self) -> None:
        # 12 installed lanes at mu=0.55 gives 6.6/s of capacity; 0.90 of that is 5.94.
        excess = divertible_arrivals(gate(7.0, installed=12))
        self.assertAlmostEqual(excess, 7.0 - 0.90 * 12 * 0.55, places=9)


if __name__ == "__main__":
    unittest.main()
