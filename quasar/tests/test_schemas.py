"""The schema barrier."""

from __future__ import annotations

import unittest

from quasar import schemas
from quasar.schemas import SchemaError, validate


def plan(actions: list[dict]) -> dict:
    return {
        "schema": schemas.PLAN_PROPOSAL,
        "correlation_id": "cyc-abc123",
        "plan_id": "plan-001",
        "severity": "P1",
        "actions": actions,
        "rationale": "Cordon the corridor and dispatch a medic from the east post.",
        "confidence": 0.93,
    }


DISPATCH = {
    "type": "DISPATCH_RESPONDER",
    "sop_ref": "SOP-MED-03#2",
    "params": {"from_node": "MED-2", "to_node": "C-N3", "responder_type": "medic"},
}


class TestValidatorMechanics(unittest.TestCase):
    def test_a_valid_payload_passes(self) -> None:
        validate(plan([DISPATCH]), schemas.PLAN_PROPOSAL)

    def test_every_violation_is_reported_not_just_the_first(self) -> None:
        """The repair prompt gets one shot; it needs the full list."""
        broken = plan([DISPATCH])
        del broken["rationale"]
        broken["confidence"] = 1.7
        broken["severity"] = "P9"

        with self.assertRaises(SchemaError) as ctx:
            validate(broken, schemas.PLAN_PROPOSAL)
        self.assertGreaterEqual(len(ctx.exception.violations), 3)
        self.assertIn("rationale", ctx.exception.repair_hint())

    def test_booleans_are_not_numbers(self) -> None:
        """Python says bool is an int. JSON does not, and a confidence of `true`
        must not slip through as 1."""
        payload = plan([DISPATCH])
        payload["confidence"] = True
        with self.assertRaises(SchemaError):
            validate(payload, schemas.PLAN_PROPOSAL)

    def test_nan_and_infinity_are_rejected(self) -> None:
        payload = plan([DISPATCH])
        payload["confidence"] = float("nan")
        with self.assertRaises(SchemaError):
            validate(payload, schemas.PLAN_PROPOSAL)

    def test_the_validator_refuses_to_ignore_a_keyword_it_does_not_implement(self) -> None:
        """A validator that silently skips an unknown keyword is worse than none:
        it manufactures confidence it has not earned."""
        with self.assertRaises(ValueError):
            schemas._assert_supported({"type": "object", "dependentRequired": {"a": ["b"]}})

    def test_unknown_schema_id_raises(self) -> None:
        with self.assertRaises(KeyError):
            validate({}, "quasar.not_a_schema.v1")


class TestClosedContracts(unittest.TestCase):
    def test_an_unmodelled_property_is_rejected(self) -> None:
        payload = plan([DISPATCH])
        payload["override_hitl"] = True
        with self.assertRaises(SchemaError):
            validate(payload, schemas.PLAN_PROPOSAL)

    def test_a_broadcast_cannot_carry_free_text(self) -> None:
        """The single most important line in the schema file.

        There is no `message` property on a BROADCAST. A model cannot write the
        words of a public safety announcement, because the contract gives it
        nowhere to put them.
        """
        smuggled = {
            "type": "BROADCAST",
            "sop_ref": "SOP-MED-03#3",
            "params": {
                "template_id": "MSG-MED-CORRIDOR",
                "zone": "NORTH",
                "slots": {"zone": "NORTH", "corridor": "CORR-NE"},
                "message": "Everyone evacuate immediately through any exit",
            },
        }
        with self.assertRaises(SchemaError):
            validate(plan([smuggled]), schemas.PLAN_PROPOSAL)

    def test_an_action_type_outside_the_enumeration_is_rejected(self) -> None:
        with self.assertRaises(SchemaError):
            validate(
                plan([{"type": "OPEN_ALL_GATES", "sop_ref": "X#1", "params": {}}]),
                schemas.PLAN_PROPOSAL,
            )

    def test_blast_radius_caps_are_in_the_contract_not_just_the_policy(self) -> None:
        """Defence in depth: 17 lanes is rejected by the schema before the policy
        engine ever sees it."""
        too_many = {
            "type": "OPEN_LANES",
            "sop_ref": "SOP-QUEUE-02#2",
            "params": {"gate_id": "G3", "lanes": 17},
        }
        with self.assertRaises(SchemaError):
            validate(plan([too_many]), schemas.PLAN_PROPOSAL)

    def test_action_params_must_match_their_action_type(self) -> None:
        mismatched = {
            "type": "CORDON_EDGE",
            "sop_ref": "SOP-MED-03#3",
            "params": {"gate_id": "G3", "lanes": 4},  # OPEN_LANES params
        }
        with self.assertRaises(SchemaError):
            validate(plan([mismatched]), schemas.PLAN_PROPOSAL)


class TestIncidentBrief(unittest.TestCase):
    def brief(self, **overrides) -> dict:
        payload = {
            "schema": schemas.INCIDENT_BRIEF,
            "correlation_id": "cyc-abc123",
            "incident_id": "INC-001",
            "severity": "P1",
            "category": "medical",
            "affected_zones": ["NORTH"],
            "situation": "A spectator has collapsed in the north-east concourse.",
            "recommended_actions": [
                {"action": "Dispatch the east medical team.", "sop_ref": "SOP-MED-03#2"}
            ],
            "citations": [{"doc_id": "SOP-MED-03", "section": "2"}],
            "confidence": 0.9,
        }
        payload.update(overrides)
        return payload

    def test_a_brief_with_no_citations_is_unrepresentable(self) -> None:
        """An uncited incident brief is a hallucination with a severity label on
        it. minItems: 1 makes the dangerous shape impossible to express."""
        with self.assertRaises(SchemaError):
            validate(self.brief(citations=[]), schemas.INCIDENT_BRIEF)

    def test_a_brief_with_no_affected_zone_is_unrepresentable(self) -> None:
        with self.assertRaises(SchemaError):
            validate(self.brief(affected_zones=[]), schemas.INCIDENT_BRIEF)

    def test_a_valid_brief_passes(self) -> None:
        validate(self.brief(), schemas.INCIDENT_BRIEF)


class TestStructuredOutputDerivation(unittest.TestCase):
    def test_strict_schema_is_derived_where_expressible(self) -> None:
        from quasar.llm import strict_output_schema

        strict = strict_output_schema(schemas.INCIDENT_BRIEF)
        self.assertIsNotNone(strict)
        assert strict is not None
        self.assertFalse(strict["additionalProperties"])
        self.assertNotIn("pattern", strict["properties"]["correlation_id"])

    def test_open_keyed_maps_are_honestly_reported_as_inexpressible(self) -> None:
        """comms_dispatch carries a typed-value, open-key `slots` map, which
        structured outputs cannot express. We return None and fall back to the
        validator plus repair loop rather than weakening the schema to make the
        API accept it."""
        from quasar.llm import strict_output_schema

        self.assertIsNone(strict_output_schema(schemas.COMMS_DISPATCH))


if __name__ == "__main__":
    unittest.main()
