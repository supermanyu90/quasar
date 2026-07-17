"""VolunteerBriefingAgent -- briefs on-ground volunteers in their own language."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from quasar import schemas
from quasar.language import SUPPORTED_LANGUAGES
from quasar.llm import ModelRequest
from quasar.rag import Retrieved, render_context
from quasar.types import LangCode
from quasar.agents.base import Agent, Corroboration, _HOUSE_RULES


@dataclass(frozen=True, slots=True)
class VolunteerTask:
    correlation_id: str
    volunteer_id: str
    language: LangCode
    role: str
    zone: str
    fixture: str
    known_risks: Sequence[str]
    retrieved: Sequence[Retrieved]


class VolunteerBriefingAgent(Agent):
    id = "VolunteerBriefingAgent"
    schema_id = schemas.VOLUNTEER_BRIEF
    VALUE_OVER_FALLBACK = (
        "The fallback hands every volunteer the same PDF. The agent writes 400 "
        "different briefings -- one per volunteer, in their language, for their zone, "
        "naming the three things that will actually happen to them today. A briefing "
        "nobody reads is not a control; personalisation at 400x is not a nicety, it "
        "is the difference between a control that exists on paper and one that exists "
        "in the volunteer's head at 19:40."
    )
    system = (
        _HOUSE_RULES
        + """
Your role: write one volunteer's shift briefing, in their language, for their zone
and role.

Be specific and short. Name the corridors and gates they will stand near. Tell
them what today's known risks are and what to do about each. Tell them how to
escalate. Ground every instruction in the procedure sections you are given and
cite them.

This is a Tier-2 (informational) output, so you may write freely -- but if you
find yourself writing an evacuation instruction, stop: that is a Tier-1 message
and it comes from the catalogue, not from you. Tell the volunteer where to find
it instead.
"""
    )

    def request(self, task: VolunteerTask) -> ModelRequest:
        lang = SUPPORTED_LANGUAGES.get(task.language)
        return ModelRequest(
            system=self.system,
            context=render_context(task.retrieved),
            user="\n".join([
                f"correlation_id: {task.correlation_id}",
                f"volunteer_id: {task.volunteer_id}",
                f"language: {task.language} ({lang.name if lang else 'unknown'})",
                f"role: {task.role}",
                f"zone: {task.zone}",
                f"fixture: {task.fixture}",
                "known risks for this fixture:",
                *(f"  - {r}" for r in task.known_risks),
            ]),
            schema_id=self.schema_id,
            effort="medium",
        )

    def corroborate(self, payload: Mapping[str, Any], task: VolunteerTask) -> Corroboration:
        if payload["language"] != task.language:
            return Corroboration.fail("briefing is not in the volunteer's language")
        if payload["zone"] != task.zone:
            return Corroboration.fail("briefing is for the wrong zone")
        return Corroboration.ok()

    def fallback(self, task: VolunteerTask) -> dict[str, Any]:
        refs = [r.ref for r in task.retrieved] or ["SOP-MED-03#1"]
        return {
            "schema": self.schema_id,
            "correlation_id": task.correlation_id,
            "volunteer_id": task.volunteer_id,
            "language": "en",  # the generic briefing exists in English only
            "role": task.role,
            "zone": task.zone,
            "sections": [
                {
                    "heading": "Standing procedure",
                    "body": (
                        "Deterministic fallback: the generic English shift briefing has been "
                        "issued because no model was available to personalise it. Read the "
                        "cited procedure sections before your shift and ask your supervisor "
                        "if anything is unclear."
                    ),
                },
                {
                    "heading": "Escalation",
                    "body": (
                        "Report any medical incident, crush, or unattended item to the control "
                        "room immediately by radio, then stay with the incident until relieved."
                    ),
                },
            ],
            "risks": list(task.known_risks)[:8],
            "citations": [
                {"doc_id": r.split("#")[0], "section": r.split("#")[1]} for r in refs[:4]
            ],
            "confidence": 1.0,
        }
