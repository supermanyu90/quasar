"""Crowd fluid dynamics."""

from __future__ import annotations

import unittest

from quasar.crowd import (
    ADVISORY_DENSITY,
    CRITICAL_DENSITY,
    GAMMA_MANDATED,
    RHO_JAM_PED_M2,
    V_FREE_M_S,
    V_MIN_M_S,
    corridor_capacity,
    fit_gamma,
    level_of_service,
    safe_speed,
    specific_flow,
    weidmann_canonical_speed,
    weidmann_speed,
)
from quasar.types import LOS


class TestWeidmann(unittest.TestCase):
    def test_shipped_gamma_is_the_least_squares_optimum(self) -> None:
        """The constant in the source is the fit, not a hand-picked number."""
        gamma, rmse = fit_gamma()
        self.assertAlmostEqual(gamma, GAMMA_MANDATED, places=3)
        # Documented in the module docstring and the submission's Limitations.
        self.assertAlmostEqual(rmse, 0.207, places=2)

    def test_speed_is_monotone_decreasing_in_density(self) -> None:
        for model in (weidmann_speed, weidmann_canonical_speed, safe_speed):
            previous = model(0.0)
            for i in range(1, 55):
                rho = i * 0.1
                current = model(rho)
                self.assertLessEqual(current, previous + 1e-9, f"{model.__name__} at {rho}")
                previous = current

    def test_boundary_conditions(self) -> None:
        self.assertLess(weidmann_speed(0.0), V_FREE_M_S)
        self.assertEqual(weidmann_speed(RHO_JAM_PED_M2), V_MIN_M_S)
        self.assertEqual(weidmann_canonical_speed(RHO_JAM_PED_M2), V_MIN_M_S)
        self.assertAlmostEqual(weidmann_canonical_speed(0.0), V_FREE_M_S)

    def test_negative_density_is_rejected(self) -> None:
        for model in (weidmann_speed, weidmann_canonical_speed, level_of_service):
            with self.assertRaises(ValueError):
                model(-0.1)

    def test_safe_speed_is_never_optimistic(self) -> None:
        """The whole point of the envelope: it never exceeds either model.

        The mandated form over-predicts speed under congestion. If safe_speed
        ever agreed with it there, the router would price a jammed corridor as
        walkable -- which is the bug this function exists to prevent.
        """
        for i in range(0, 55):
            rho = i * 0.1
            self.assertLessEqual(safe_speed(rho), weidmann_speed(rho) + 1e-12)
            self.assertLessEqual(safe_speed(rho), weidmann_canonical_speed(rho) + 1e-12)

    def test_mandated_form_over_predicts_under_congestion(self) -> None:
        """Documents the known defect rather than hiding it.

        If a future recalibration removes this divergence, this test fails and
        somebody has to consciously decide that safe_speed is still needed.
        """
        self.assertGreater(weidmann_speed(3.0), weidmann_canonical_speed(3.0) + 0.15)
        self.assertAlmostEqual(safe_speed(3.0), weidmann_canonical_speed(3.0))


class TestLevelOfService(unittest.TestCase):
    def test_fruin_boundaries(self) -> None:
        cases = [
            (0.10, LOS.A), (0.35, LOS.B), (0.50, LOS.C),
            (0.90, LOS.D), (1.50, LOS.E), (3.00, LOS.F),
        ]
        for rho, expected in cases:
            self.assertIs(level_of_service(rho), expected, f"at rho={rho}")

    def test_trigger_thresholds_sit_on_the_los_boundaries(self) -> None:
        self.assertIs(level_of_service(ADVISORY_DENSITY), LOS.E)
        self.assertIs(level_of_service(ADVISORY_DENSITY - 1e-6), LOS.D)
        self.assertIs(level_of_service(CRITICAL_DENSITY), LOS.F)
        self.assertIs(level_of_service(CRITICAL_DENSITY - 1e-6), LOS.E)


class TestFlow(unittest.TestCase):
    def test_flow_vanishes_at_both_ends_and_peaks_between(self) -> None:
        """The fundamental diagram is unimodal: nobody moves in an empty corridor
        and nobody moves in a jammed one."""
        self.assertAlmostEqual(specific_flow(0.0), 0.0)
        self.assertLess(specific_flow(RHO_JAM_PED_M2), 0.3)
        peak = max((specific_flow(i * 0.05) for i in range(1, 108)))
        self.assertGreater(peak, specific_flow(0.05))
        self.assertGreater(peak, specific_flow(5.0))

    def test_capacity_scales_with_width(self) -> None:
        self.assertAlmostEqual(
            corridor_capacity(1.0, 8.0), 2.0 * corridor_capacity(1.0, 4.0)
        )
        with self.assertRaises(ValueError):
            corridor_capacity(1.0, 0.0)


if __name__ == "__main__":
    unittest.main()
