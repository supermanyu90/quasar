"""Density-aware routing and the accessibility constraints."""

from __future__ import annotations

import unittest

from quasar.crowd import CRITICAL_DENSITY
from quasar.routing import (
    ACCESSIBLE,
    FAN,
    RESPONDER,
    STAFF,
    NoRouteError,
    Profile,
    RouteRequest,
    Router,
)
from quasar.venue import build_stadium


class RoutingCase(unittest.TestCase):
    def setUp(self) -> None:
        self.venue = build_stadium()
        self.router = Router(self.venue)
        self.quiet = {e: 0.3 for e in self.venue.edges}
        # The north-east pinch point in a crush; the north concourse busy.
        self.busy = dict(self.quiet, **{"CORR-NE": 3.4, "CORR-N2": 1.6})


class TestDensityAwareness(RoutingCase):
    def test_density_aware_route_beats_the_shortest_path_when_it_matters(self) -> None:
        """The headline claim, checked rather than asserted.

        A distance-shortest router sends the medic down the 103 m direct corridor.
        That corridor is at LOS F. Walking it actually takes ~380 s. The
        density-aware router takes the 145 m service ring instead and gets there
        in ~165 s -- more than twice as fast, by choosing the longer path.
        """
        blind_density = {e: 0.0 for e in self.venue.edges}
        blind = self.router.route(
            RouteRequest("MED-2", "C-N3", RESPONDER), blind_density
        )
        aware = self.router.route(RouteRequest("MED-2", "C-N3", RESPONDER), self.busy)

        self.assertIn("CORR-NE", blind.edges)  # straight through the crush
        self.assertNotIn("CORR-NE", aware.edges)
        self.assertLess(aware.distance_m, blind.distance_m + 1e-9 + 60)
        self.assertGreater(aware.distance_m, blind.distance_m)  # the LONGER path...

        true_cost_of_blind = sum(
            self.router.edge_cost(e, self.busy[e], RESPONDER) for e in blind.edges
        )
        self.assertGreater(true_cost_of_blind, 2.0 * aware.eta_s)

    def test_fans_are_excluded_from_a_los_f_corridor(self) -> None:
        route = self.router.route(RouteRequest("C-N3", "C-E1", FAN), self.busy)
        self.assertNotIn("CORR-NE", route.edges)
        self.assertLess(route.worst_density, CRITICAL_DENSITY)

    def test_responders_may_push_through_where_fans_may_not(self) -> None:
        """A medic must be able to reach a casualty through the crowd that caused
        the emergency. If every dense edge were removed for them too, the system
        would refuse to send help exactly when help is needed."""
        cost = self.router.edge_cost("CORR-NE", 3.4, RESPONDER)
        self.assertIsNotNone(cost)
        self.assertIsNone(self.router.edge_cost("CORR-NE", 3.4, FAN))

    def test_counterflow_penalty_makes_a_dense_edge_more_expensive_than_length_alone(self) -> None:
        quiet_cost = self.router.edge_cost("CORR-NE", 0.3, RESPONDER)
        dense_cost = self.router.edge_cost("CORR-NE", 3.4, RESPONDER)
        assert quiet_cost is not None and dense_cost is not None
        self.assertGreater(dense_cost, 3.0 * quiet_cost)


class TestHardConstraints(RoutingCase):
    def test_fans_are_never_routed_through_a_service_corridor(self) -> None:
        for edge_id in ("SVC-1", "SVC-2", "SVC-3", "SVC-4", "SVC-5", "SVC-6"):
            self.assertIsNone(self.router.edge_cost(edge_id, 0.0, FAN))
            self.assertIsNotNone(self.router.edge_cost(edge_id, 0.0, STAFF))

        route = self.router.route(RouteRequest("MED-2", "C-N3", FAN), self.quiet)
        self.assertFalse(any(self.venue.edge(e).staff_only for e in route.edges))

    def test_a_cordon_binds_responders_too(self) -> None:
        """SOP-MED-03#3. The responder does not get an exemption from the cordon;
        they get the service ring."""
        route = self.router.route(
            RouteRequest("MED-2", "C-N3", RESPONDER, cordoned_edges=frozenset({"CORR-NE"})),
            self.busy,
        )
        self.assertNotIn("CORR-NE", route.edges)

    def test_step_free_route_never_crosses_a_stair(self) -> None:
        route = self.router.route(RouteRequest("G5", "SEAT-N", ACCESSIBLE), self.quiet)
        for edge_id in route.edges:
            self.assertTrue(
                self.venue.edge(edge_id).step_free,
                f"accessible route crossed stepped edge {edge_id}",
            )
        self.assertIn("RAMP-N", route.edges)

    def test_no_route_raises_rather_than_returning_a_degraded_one(self) -> None:
        """The west stand has a stair and no ramp. There is no step-free way out.

        This is a real gap in the reference venue, and the correct behaviour is a
        loud refusal that forces a staffed refuge point -- not a route that
        quietly sends a wheelchair user down a staircase.
        """
        with self.assertRaises(NoRouteError):
            self.router.route(RouteRequest("SEAT-W", "G6", ACCESSIBLE), self.quiet)

    def test_an_accessible_fan_is_not_sent_into_a_los_e_crush(self) -> None:
        """ACCESSIBLE caps density well below the LOS-F limit an unimpeded adult
        tolerates. At C-N3 in this state, every step-free exit is over that cap,
        so the router refuses instead of improvising."""
        with self.assertRaises(NoRouteError):
            self.router.nearest_tagged("C-N3", "washroom", self.busy, profile=ACCESSIBLE)
        # ...while an unimpeded fan is routed normally.
        route = self.router.nearest_tagged("C-N3", "washroom", self.busy, profile=FAN)
        self.assertTrue(route.edges)


class TestNearestTagged(RoutingCase):
    def test_nearest_accessible_washroom_skips_the_nearer_stepped_one(self) -> None:
        """'Take me to the nearest accessible washroom.' WC-N is 17 m from C-N3
        but up a staircase; WC-N-ACC is much further and step-free. The step-free
        one is the answer, and no amount of proximity changes that."""
        step_free = self.router.nearest_tagged(
            "C-N1", "washroom", self.quiet, profile=ACCESSIBLE
        )
        self.assertEqual(step_free.destination, "WC-N-ACC")
        for edge_id in step_free.edges:
            self.assertTrue(self.venue.edge(edge_id).step_free)

    def test_nearest_medical_post_is_reachable_from_every_seat(self) -> None:
        for seat in self.venue.nodes_tagged("seating"):
            route = self.router.nearest_tagged(seat.id, "medical", self.quiet, profile=FAN)
            self.assertTrue(route.edges)

    def test_unroutable_tag_raises(self) -> None:
        with self.assertRaises(NoRouteError):
            self.router.nearest_tagged("C-N1", "no_such_tag", self.quiet)


class TestProfileValidation(unittest.TestCase):
    def test_invalid_profiles_are_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            Profile(name="bad", speed_factor=0.0)
        with self.assertRaises(ValueError):
            Profile(name="bad", max_density=-1.0)


if __name__ == "__main__":
    unittest.main()
