"""CommunicationAgent -- selects approved, human-validated announcement templates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from quasar import schemas
from quasar.language import CATALOGUE, SUPPORTED_LANGUAGES
from quasar.llm import ModelRequest
from quasar.types import LangCode
from quasar.agents.base import Agent, Corroboration, _HOUSE_RULES


@dataclass(frozen=True, slots=True)
class CommsTask:
    correlation_id: str
    template_id: str
    slots: Mapping[str, Any]
    zones: Sequence[str]
    languages: Sequence[LangCode]


class CommunicationAgent(Agent):
    id = "CommunicationAgent"
    schema_id = schemas.COMMS_DISPATCH
    VALUE_OVER_FALLBACK = (
        "For Tier-1 traffic the agent adds nothing to the words -- by design -- and "
        "the fallback is byte-identical. What it adds is audience selection: which "
        "zones need to hear this, in which languages, given who scanned in at which "
        "gates. The fallback broadcasts venue-wide in the fixture's default languages, "
        "which is safe, louder than necessary, and desensitising."
    )
    system = (
        _HOUSE_RULES
        + """
Your role: decide who hears an approved announcement, in which languages.

You do NOT write the announcement. You are given an approved template id and its
typed slots; you choose the zones and the languages. If the template is
safety-critical, the catalogue will refuse any language it has not had validated
by a human translator, and a steward will be dispatched to cover the gap. That is
correct behaviour, not an error: do not attempt to route around it by choosing a
different template.
"""
    )

    def request(self, task: CommsTask) -> ModelRequest:
        template = CATALOGUE[task.template_id]
        return ModelRequest(
            system=self.system,
            user="\n".join([
                f"correlation_id: {task.correlation_id}",
                f"template_id: {task.template_id} (tier: {template.tier.value})",
                f"slots: {json.dumps(dict(task.slots))}",
                f"zones affected: {', '.join(task.zones)}",
                f"languages present in those zones (from ticketing): "
                f"{', '.join(task.languages)}",
                f"languages with a validated catalogue entry for this template: "
                f"{', '.join(template.validated_languages())}",
            ]),
            schema_id=self.schema_id,
            effort="low",
        )

    def corroborate(self, payload: Mapping[str, Any], task: CommsTask) -> Corroboration:
        template = CATALOGUE.get(payload["template_id"])
        if template is None:
            return Corroboration.fail(f"unknown template {payload['template_id']!r}")
        if payload["template_id"] != task.template_id:
            return Corroboration.fail(
                f"substitutes template {payload['template_id']!r} for the approved "
                f"{task.template_id!r}"
            )
        if payload["tier"] != template.tier.value:
            return Corroboration.fail(
                f"declares tier {payload['tier']!r} for a {template.tier.value} template"
            )
        if dict(payload["slots"]) != dict(task.slots):
            return Corroboration.fail("alters the approved slot values")
        if unknown := set(payload["languages"]) - set(SUPPORTED_LANGUAGES):
            return Corroboration.fail(f"names unsupported languages {sorted(unknown)}")
        return Corroboration.ok()

    def fallback(self, task: CommsTask) -> dict[str, Any]:
        template = CATALOGUE[task.template_id]
        return {
            "schema": self.schema_id,
            "correlation_id": task.correlation_id,
            "tier": template.tier.value,
            "template_id": task.template_id,
            "slots": dict(task.slots),
            "languages": list(task.languages),
            "zones": list(task.zones),
            "confidence": 1.0,
        }
