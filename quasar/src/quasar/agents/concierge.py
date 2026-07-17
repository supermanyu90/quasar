"""ConciergeAgent -- the fan-facing assistant that decides *where*, never *how*."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from quasar import schemas
from quasar.language import SUPPORTED_LANGUAGES
from quasar.llm import ModelRequest
from quasar.plane import Assessment
from quasar.types import LangCode, NodeId
from quasar.agents.base import Agent, Corroboration, _HOUSE_RULES


@dataclass(frozen=True, slots=True)
class ConciergeTask:
    correlation_id: str
    utterance: str
    language: LangCode
    at_node: NodeId
    accessible: bool
    assessment: Assessment


_INTENT_TAGS: Mapping[str, str | None] = {
    "wayfinding": "seating",
    "seat": "seating",
    "food": "fnb",
    "washroom": "washroom",
    "medical": "medical",
    "lost_and_found": "lost_and_found",
    "match_info": None,
    "emergency": "medical",
    "other": None,
}


_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "emergency": ("help", "emergency", "collapsed", "hurt", "मदद", "आपातकाल", "मदत"),
    "washroom": ("toilet", "washroom", "restroom", "shauchalay", "शौचालय", "स्वच्छतागृह"),
    "medical": ("doctor", "medic", "first aid", "डॉक्टर", "प्राथमिक"),
    "food": ("food", "eat", "drink", "water", "खाना", "पाणी", "अन्न"),
    "lost_and_found": ("lost", "found", "missing", "खोया", "हरवले"),
    "seat": ("seat", "block", "row", "सीट", "आसन"),
    "wayfinding": ("gate", "where", "how do i get", "कहाँ", "कुठे", "गेट"),
}


class ConciergeAgent(Agent):
    id = "ConciergeAgent"
    schema_id = schemas.CONCIERGE_REPLY
    VALUE_OVER_FALLBACK = (
        "The fallback is a keyword matcher and a menu. The agent is the entire "
        "multilingual product: it understands 'I'm at the wrong end and my mother "
        "can't manage stairs' in Marathi, resolves it to an accessible-route request "
        "from the fan's actual BLE-fixed position, and answers in Marathi. This is the "
        "one component where the generative model IS the feature -- but note that even "
        "here it does not choose the route, and if it classifies the turn as an "
        "emergency it loses the pen entirely and the controlled catalogue answers."
    )
    system = (
        _HOUSE_RULES
        + """
Your role: the fan-facing concierge.

Classify the fan's intent, resolve it to a destination category if it needs one,
and reply in the fan's own language. You are given the fan's map-matched position
and whether they need a step-free route. You do NOT produce the route -- set
requires_route and the routing engine will compute it and the reply will be
assembled around it.

Safety tier: if the fan is reporting an emergency, a collapse, a fire, a crush, or
anything where a wrong answer could hurt someone, set safety_tier to
"safety_critical". You will then NOT be the one who speaks: the controlled message
catalogue will answer and a steward will be dispatched. Setting this correctly is
more important than answering well.
"""
    )

    def request(self, task: ConciergeTask) -> ModelRequest:
        lang = SUPPORTED_LANGUAGES.get(task.language)
        return ModelRequest(
            system=self.system,
            user="\n".join([
                f"correlation_id: {task.correlation_id}",
                f"fan_language: {task.language} ({lang.name if lang else 'unknown'})",
                f"fan_position (BLE map-matched): node {task.at_node}",
                f"step_free_required: {task.accessible}",
                "",
                f'fan said: "{task.utterance}"',
            ]),
            schema_id=self.schema_id,
            effort="low",  # latency matters more than depth on a fan's phone
        )

    def corroborate(self, payload: Mapping[str, Any], task: ConciergeTask) -> Corroboration:
        if payload["language"] != task.language:
            return Corroboration.fail(
                f"replies in {payload['language']!r} to a fan writing {task.language!r}"
            )

        intent = payload["intent"]
        expected_tag = _INTENT_TAGS[intent]
        tag = payload["destination_tag"]

        if intent == "emergency" and payload["safety_tier"] != "safety_critical":
            return Corroboration.fail(
                "classifies the turn as an emergency but not as safety critical"
            )

        # A deterministic second opinion on the safety tier. If the keyword
        # classifier smells an emergency and the model did not, we do not average
        # the two -- we take the alarming one. Asymmetric costs, asymmetric rule.
        if self._keyword_intent(task.utterance) == "emergency" and payload["safety_tier"] != "safety_critical":
            return Corroboration.fail(
                "utterance matches emergency vocabulary but the model marked it informational"
            )

        notes: list[str] = []
        if expected_tag is not None and tag != expected_tag and payload["requires_route"]:
            notes.append(f"intent {intent!r} with destination_tag {tag!r}")
        if payload["requires_route"] and tag is None:
            return Corroboration.fail("requests a route with no destination tag")

        return Corroboration(score=1.0 if not notes else 0.8, notes=tuple(notes))

    @staticmethod
    def _keyword_intent(utterance: str) -> str:
        low = utterance.lower()
        for intent, words in _KEYWORDS.items():
            if any(w in low for w in words):
                return intent
        return "other"

    def fallback(self, task: ConciergeTask) -> dict[str, Any]:
        intent = self._keyword_intent(task.utterance)
        tag = _INTENT_TAGS[intent]
        critical = intent == "emergency"
        return {
            "schema": self.schema_id,
            "correlation_id": task.correlation_id,
            "language": task.language,
            "intent": intent,
            "destination_tag": tag,
            # English, because the fallback may not translate safely. The app
            # renders a language-picker menu alongside it.
            "reply_text": (
                "A steward is on the way to you."
                if critical
                else "Choose what you need and I will show you the way."
            ),
            "requires_route": tag is not None and not critical,
            "safety_tier": "safety_critical" if critical else "informational",
            "confidence": 1.0,
        }
