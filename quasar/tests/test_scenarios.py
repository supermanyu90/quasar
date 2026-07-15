"""Synthetic scenario generation and the pre-match stress harness."""

from __future__ import annotations

import json
import unittest

from quasar import schemas
from quasar.llm import ModelUnavailable
from quasar.scenarios import (
    SCENARIO_KINDS,
    ScenarioGenerator,
    SeededSampler,
    StressHarness,
)

from tests.fixtures import SequenceModel, plane


class TestSeededSampler(unittest.TestCase):
    def setUp(self) -> None:
        self.plane = plane()
        self.sampler = SeededSampler(self.plane, seed=3)

    def test_every_sampled_scenario_satisfies_the_schema(self) -> None:
        for kind in SCENARIO_KINDS:
            for scenario in self.sampler.sample(kind, n=3):
                schemas.validate(scenario, schemas.SCENARIO)

    def test_sampling_is_reproducible(self) -> None:
        a = list(SeededSampler(self.plane, seed=99).sample("gate_failure", n=2))
        b = list(SeededSampler(self.plane, seed=99).sample("gate_failure", n=2))
        self.assertEqual(a, b)


class TestGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.plane = plane()

    def valid_scenario(self) -> dict:
        return {
            "schema": schemas.SCENARIO,
            "scenario_id": "GEN-001",
            "name": "gate 3 turnstile failure during a weather hold",
            "kind": "gate_failure",
            "edge_density": {"CORR-NE": 4.1, "CORR-N2": 3.2},
            "closed_edges": ["E-G3"],
            "gate_overrides": [
                {"gate_id": "G2", "arrival_rate_per_s": 11.0, "open_lanes": 8}
            ],
        }

    def test_a_generated_scenario_is_schema_validated(self) -> None:
        model = SequenceModel(json.dumps(self.valid_scenario()))
        scenario = ScenarioGenerator(model, self.plane).propose("gate_failure")
        self.assertEqual(scenario["kind"], "gate_failure")

    def test_a_scenario_naming_a_corridor_that_does_not_exist_is_rejected(self) -> None:
        """The model is proposing situations, not inventing geography."""
        bad = self.valid_scenario()
        bad["edge_density"]["CORR-IMAGINARY"] = 2.0

        model = SequenceModel(json.dumps(bad))
        with self.assertRaises(ValueError) as ctx:
            ScenarioGenerator(model, self.plane).propose("gate_failure")
        self.assertIn("unknown corridor", str(ctx.exception))

    def test_a_physically_impossible_density_is_rejected_by_the_schema(self) -> None:
        bad = self.valid_scenario()
        bad["edge_density"]["CORR-NE"] = 40.0  # more people than fit in the space
        model = SequenceModel(json.dumps(bad))
        with self.assertRaises(schemas.SchemaError):
            ScenarioGenerator(model, self.plane).propose("gate_failure")

    def test_an_unknown_kind_is_rejected_before_the_model_is_called(self) -> None:
        model = SequenceModel()
        with self.assertRaises(ValueError):
            ScenarioGenerator(model, self.plane).propose("alien_invasion")

    def test_an_unavailable_model_propagates(self) -> None:
        """Scenario generation runs the week before the fixture, not during it.
        If the model is down, the right answer is to fail the pre-match check --
        not to silently run fewer scenarios and declare the venue safe."""
        model = SequenceModel()  # empty queue -> ModelUnavailable
        with self.assertRaises(ModelUnavailable):
            ScenarioGenerator(model, self.plane).propose("gate_failure")


class TestStressHarness(unittest.TestCase):
    def setUp(self) -> None:
        self.plane = plane()
        self.harness = StressHarness(self.plane)

    def base_scenario(self, **overrides) -> dict:
        scenario = {
            "schema": schemas.SCENARIO,
            "scenario_id": "STRESS-001",
            "name": "baseline",
            "kind": "medical_surge",
            "edge_density": {},
            "closed_edges": [],
            "gate_overrides": [],
        }
        scenario.update(overrides)
        return scenario

    def test_the_harness_finds_a_real_accessibility_gap_in_the_reference_venue(self) -> None:
        """The point of the harness, demonstrated.

        The west stand has a staircase and no ramp or lift. Under *any* scenario,
        a wheelchair user in the west stand cannot reach a gate. That is a defect
        in the venue, not in the code, and it is exactly the kind of thing that is
        invisible on a floor plan and lethal on the night. The harness finds it
        before the gates open, and SOP-EVAC-01#3 tells you what to do about it:
        staff a refuge point.
        """
        result = self.harness.run(self.base_scenario())
        gaps = [f for f in result.findings if f.invariant == "step-free-egress-exists"]

        self.assertTrue(gaps, "the harness failed to find the west stand's missing ramp")
        self.assertEqual(len(gaps), 1)
        self.assertIn("SEAT-W", gaps[0].detail)
        self.assertIn("refuge", gaps[0].detail)
        self.assertEqual(gaps[0].severity, "critical")

    def test_the_three_causes_of_lost_step_free_egress_are_told_apart(self) -> None:
        """Same symptom, three owners. A harness that cannot tell them apart sends
        the wrong person to fix the wrong thing."""
        # (a) the building is wrong -- SEAT-W has no ramp at all.
        clean = self.harness.run(self.base_scenario())
        kinds = {f.invariant for f in clean.findings}
        self.assertIn("step-free-egress-exists", kinds)
        self.assertNotIn("step-free-egress-after-closure", kinds)

        # (b) the closure plan is wrong -- SEAT-N has a ramp, and this scenario shuts it.
        closed = self.harness.run(self.base_scenario(closed_edges=["RAMP-N"]))
        after = [f for f in closed.findings if f.invariant == "step-free-egress-after-closure"]
        self.assertTrue(after)
        self.assertIn("SEAT-N", after[0].detail)
        self.assertIn("RAMP-N", after[0].detail)

        # (c) the stewarding is wrong -- the ramp is open, just too crowded to use.
        crowded = self.harness.run(
            self.base_scenario(edge_density={"RAMP-N": 2.0, "CORR-N1": 2.0, "CORR-NW": 2.0,
                                            "CORR-N2": 2.0, "CORR-NE": 2.0})
        )
        loaded = [f for f in crowded.findings if f.invariant == "step-free-egress-under-load"]
        self.assertTrue(loaded)
        self.assertIn("held clear", loaded[0].detail)

    def test_a_dead_end_spur_is_not_reported_as_a_severed_concourse(self) -> None:
        """The only path to the lost-property office is the only path to the
        lost-property office. That is a tautology, not a finding, and a harness
        that reports tautologies stops being read."""
        result = self.harness.run(self.base_scenario(edge_density={"SP-LOST": 3.0}))
        self.assertFalse(
            [f for f in result.findings if f.invariant == "critical-corridor-has-alternative"]
        )

    def test_no_spectator_is_ever_routed_through_a_closed_corridor(self) -> None:
        for kind in SCENARIO_KINDS:
            for scenario in SeededSampler(self.plane, seed=21).sample(kind, n=4):
                result = self.harness.run(scenario)
                self.assertFalse(
                    [f for f in result.findings if f.invariant == "no-route-through-closure"],
                    f"{scenario['scenario_id']} routed a spectator through a closure",
                )

    def test_a_jammed_corridor_on_an_intact_ring_is_not_a_single_point_of_failure(self) -> None:
        """The concourse is a ring, so a LOS-F corridor normally has a long way
        round. Cordoning it is inconvenient, not severing -- and the harness must
        not cry wolf about it, or nobody will read its output."""
        result = self.harness.run(self.base_scenario(edge_density={"CORR-NE": 3.6}))
        self.assertFalse(
            [f for f in result.findings if f.invariant == "critical-corridor-has-alternative"]
        )

    def test_a_jammed_corridor_on_a_broken_ring_IS_a_single_point_of_failure(self) -> None:
        """Now take out CORR-N2 as well. C-N3's only remaining spectator link to
        the rest of the concourse is CORR-NE -- which is at LOS F, and which the
        medical SOP is about to demand we cordon.

        That is the interaction a single-fault analysis never finds: each fault is
        survivable and the pair is not. It is precisely what the model-driven
        scenario generator exists to propose.
        """
        result = self.harness.run(
            self.base_scenario(edge_density={"CORR-NE": 3.6}, closed_edges=["CORR-N2"])
        )
        single_points = [
            f for f in result.findings if f.invariant == "critical-corridor-has-alternative"
        ]
        self.assertTrue(single_points)
        self.assertIn("CORR-NE", single_points[0].detail)

    def test_a_gate_beyond_mitigation_is_reported(self) -> None:
        result = self.harness.run(
            self.base_scenario(
                gate_overrides=[
                    # 12 installed lanes x 0.55/s = 6.6/s of capacity. 20/s cannot
                    # be absorbed by opening lanes; it has to be diverted.
                    {"gate_id": "G3", "arrival_rate_per_s": 20.0, "open_lanes": 6}
                ]
            )
        )
        gate_findings = [f for f in result.findings if f.invariant == "gate-mitigable"]
        self.assertTrue(gate_findings)
        self.assertIn("diversion", gate_findings[0].detail)

    def test_isolating_a_stand_completely_is_a_critical_finding(self) -> None:
        """Close both routes off the north stand and the harness must say so
        rather than quietly reporting a long walk."""
        result = self.harness.run(
            self.base_scenario(closed_edges=["STAIR-N", "RAMP-N"])
        )
        stranded = [f for f in result.findings if f.invariant == "egress-exists"]
        self.assertTrue(stranded)
        self.assertIn("SEAT-N", stranded[0].detail)
        self.assertFalse(result.passed)

    def test_a_scenario_the_venue_survives_is_reported_as_such(self) -> None:
        """A pass is a pass on the *ambulatory* invariants. The west-stand refuge
        finding is structural and shows up in every run -- which is the harness
        telling the truth, not crying wolf."""
        result = self.harness.run(self.base_scenario(edge_density={"CORR-N2": 1.4}))
        non_structural = [
            f
            for f in result.findings
            if f.invariant != "step-free-egress-exists"
        ]
        self.assertFalse(non_structural, [str(f) for f in non_structural])

    def test_a_scenario_that_does_not_validate_never_reaches_the_venue(self) -> None:
        with self.assertRaises(schemas.SchemaError):
            self.harness.run({"schema": schemas.SCENARIO, "scenario_id": "X"})


if __name__ == "__main__":
    unittest.main()
