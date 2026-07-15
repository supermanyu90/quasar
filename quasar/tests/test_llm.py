"""The model plane: schema delivery, JSON extraction, failover, disabled mode."""

from __future__ import annotations

import json
import unittest

from quasar import schemas
from quasar.llm import (
    DisabledModel,
    FailoverModel,
    ModelRequest,
    ModelResponse,
    ModelUnavailable,
    extract_json,
    schema_prompt,
    strict_output_schema,
)

from tests.fixtures import SequenceModel


def request(schema_id: str) -> ModelRequest:
    return ModelRequest(system="s", user="u", schema_id=schema_id)


class TestSchemaDelivery(unittest.TestCase):
    """The bug this guards against: an agent whose schema cannot be pushed into the
    decoder was being sent no schema AT ALL -- not in output_config, not in the
    prompt. It would have guessed field names, failed validation, and silently
    fallen back to the deterministic planner on every live call."""

    def test_every_schema_reaches_the_model_one_way_or_the_other(self) -> None:
        for schema_id in schemas.SCHEMAS:
            strict = strict_output_schema(schema_id)
            if strict is None:
                # Not expressible as a structured output -> it MUST be in the prompt.
                prompt = schema_prompt(schema_id)
                self.assertIn(schema_id, prompt)
                self.assertIn('"properties"', prompt)
            else:
                self.assertEqual(strict.get("additionalProperties"), False)

    def test_the_open_keyed_payloads_are_the_ones_that_need_the_prompt(self) -> None:
        self.assertIsNone(strict_output_schema(schemas.PLAN_PROPOSAL))
        self.assertIsNone(strict_output_schema(schemas.COMMS_DISPATCH))
        self.assertIsNotNone(strict_output_schema(schemas.CROWD_ASSESSMENT))

    def test_the_rendered_schema_is_the_published_one_verbatim(self) -> None:
        """Not a paraphrase of the contract. The contract."""
        prompt = schema_prompt(schemas.PLAN_PROPOSAL)
        body = prompt.split("```json")[1].split("```")[0]
        self.assertEqual(json.loads(body), json.loads(json.dumps(schemas.SCHEMAS[schemas.PLAN_PROPOSAL])))

    def test_it_names_the_action_variants_the_planner_has_to_get_right(self) -> None:
        prompt = schema_prompt(schemas.PLAN_PROPOSAL)
        for variant in ("DISPATCH_RESPONDER", "CORDON_EDGE", "OPEN_LANES", "BROADCAST"):
            self.assertIn(variant, prompt)

    def test_an_unknown_schema_raises(self) -> None:
        with self.assertRaises(KeyError):
            schema_prompt("quasar.nope.v1")


class TestDisabledModel(unittest.TestCase):
    def test_it_always_refuses(self) -> None:
        with self.assertRaises(ModelUnavailable):
            DisabledModel().complete(request(schemas.CROWD_ASSESSMENT))

    def test_the_reason_is_carried_through(self) -> None:
        with self.assertRaises(ModelUnavailable) as ctx:
            DisabledModel("certification run").complete(request(schemas.CROWD_ASSESSMENT))
        self.assertIn("certification run", str(ctx.exception))


class TestFailover(unittest.TestCase):
    def test_the_edge_model_catches_a_cloud_outage(self) -> None:
        edge = SequenceModel('{"ok": true}')
        model = FailoverModel(primary=DisabledModel("uplink down"), secondary=edge)

        response = model.complete(request(schemas.CROWD_ASSESSMENT))
        self.assertEqual(response.text, '{"ok": true}')
        self.assertEqual(len(model.partition_events), 1)
        self.assertIn("uplink down", model.partition_events[0])

    def test_with_no_secondary_the_outage_propagates(self) -> None:
        """It does not invent an answer. That is the whole point."""
        model = FailoverModel(primary=DisabledModel())
        with self.assertRaises(ModelUnavailable):
            model.complete(request(schemas.CROWD_ASSESSMENT))

    def test_both_planes_down_propagates_to_the_deterministic_twin(self) -> None:
        model = FailoverModel(primary=DisabledModel(), secondary=DisabledModel())
        with self.assertRaises(ModelUnavailable):
            model.complete(request(schemas.CROWD_ASSESSMENT))


class TestExtractJson(unittest.TestCase):
    def test_plain_json(self) -> None:
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})

    def test_markdown_fence(self) -> None:
        """A weaker edge model wraps its output. A formatting quirk is not a safety
        event -- do not burn the one repair round trip on it."""
        self.assertEqual(extract_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_leading_prose(self) -> None:
        self.assertEqual(extract_json('Sure! Here you go:\n{"a": 1}'), {"a": 1})

    def test_braces_inside_strings_do_not_confuse_the_scanner(self) -> None:
        self.assertEqual(
            extract_json('prose {"a": "a } brace", "b": 2}'),
            {"a": "a } brace", "b": 2},
        )

    def test_escaped_quotes_inside_strings(self) -> None:
        self.assertEqual(extract_json(r'x {"a": "he said \"hi\""}'), {"a": 'he said "hi"'})

    def test_no_json_at_all_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_json("I'm sorry, I can't help with that.")

    def test_unterminated_object_raises(self) -> None:
        with self.assertRaises(ValueError):
            extract_json('{"a": 1')


if __name__ == "__main__":
    unittest.main()
