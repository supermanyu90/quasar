"""IncidentResponseAgent -- turns a free-text report into a graded, SOP-cited brief."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from quasar import schemas
from quasar.llm import ModelRequest
from quasar.plane import Assessment
from quasar.rag import Retrieved, render_context
from quasar.types import IncidentReport, Severity
from quasar.agents.base import Agent, Corroboration, _HOUSE_RULES


@dataclass(frozen=True, slots=True)
class IncidentTask:
    correlation_id: str
    report: IncidentReport
    assessment: Assessment
    category: str  # classified upstream (voice intent) or defaulted
    retrieved: Sequence[Retrieved]
    severity_floor: Severity
    # The zone the incident is in, resolved from THIS venue's graph by the caller.
    # It used to come from a module-level global, which quietly asserted that only
    # one venue exists in the world. It does not, and a venue operating system that
    # cannot hold two venues in memory at once is not one.
    zone: str


class IncidentResponseAgent(Agent):
    id = "IncidentResponseAgent"
    schema_id = schemas.INCIDENT_BRIEF
    VALUE_OVER_FALLBACK = (
        "The fallback emits the SOP verbatim. The agent turns a garbled voice note "
        "from a volunteer in the middle of a crush -- 'someone's down near the food "
        "place, people are pushing' -- into a structured brief that names the "
        "corridor, the zone, the severity and the clauses that apply, in eight "
        "seconds. Free-text incident intake is irreducibly a language problem: there "
        "is no form field for panic."
    )
    system = (
        _HOUSE_RULES
        + """
Your role: convert a raw incident report and live telemetry into a structured
situation report for the command centre.

You are given the reporter's words, the measured crowd density around the scene,
and the relevant sections of the venue's standing procedure. Ground every
recommended action in a section you were given, and cite it by its exact
[DOC-ID#section] reference. Do not cite a section you were not shown. Do not
invent a section number.

You are given a severity floor computed from procedure. You may grade the
incident MORE severely than the floor if the report warrants it. You may never
grade it less severely.
"""
    )

    def request(self, task: IncidentTask) -> ModelRequest:
        r = task.report
        scene = [
            f"correlation_id: {task.correlation_id}",
            f"incident_id: {r.id}",
            f"reported_by: {r.reporter_role.value} at node {r.at_node}",
            f"category (classified): {task.category}",
            f"severity_floor (from procedure, may not be undercut): {task.severity_floor.value}",
            "",
            f'reporter said: "{r.text}"',
            "",
            "Measured density on corridors adjacent to the scene:",
        ]
        return ModelRequest(
            system=self.system,
            context=render_context(task.retrieved),
            user="\n".join(scene + self._scene_density(task)),
            schema_id=self.schema_id,
            effort="high",
        )

    @staticmethod
    def _scene_density(task: IncidentTask) -> list[str]:
        rows: list[str] = []
        for h in task.assessment.hotspots:
            rows.append(
                f"  {h.edge_id} (zone {h.zone}): {h.density:.2f} ped/m2, LOS {h.los.value}"
            )
        return rows or ["  (all adjacent corridors below advisory density)"]

    def corroborate(self, payload: Mapping[str, Any], task: IncidentTask) -> Corroboration:
        order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
        if order[payload["severity"]] > order[task.severity_floor.value]:
            return Corroboration.fail(
                f"grades the incident {payload['severity']} but procedure sets a floor of "
                f"{task.severity_floor.value}"
            )

        if task.zone not in payload["affected_zones"]:
            return Corroboration.fail(
                f"omits the zone the incident was reported in ({task.zone})"
            )

        notes: list[str] = []
        if payload["category"] != task.category:
            notes.append(
                f"reclassifies category {task.category!r} -> {payload['category']!r}"
            )
        # Citation validity is checked separately by rag.check_grounding, which
        # governance runs; here we only require that actions are cited at all.
        uncited = [a for a in payload["recommended_actions"] if not a.get("sop_ref")]
        if uncited:
            return Corroboration.fail("recommends actions with no SOP reference")

        return Corroboration(score=1.0 if not notes else 0.9, notes=tuple(notes))

    def fallback(self, task: IncidentTask) -> dict[str, Any]:
        refs = [r.ref for r in task.retrieved] or ["SOP-MED-03#1"]
        return {
            "schema": self.schema_id,
            "correlation_id": task.correlation_id,
            "incident_id": task.report.id,
            "severity": task.severity_floor.value,
            "category": task.category,
            "affected_zones": [task.zone],
            "situation": (
                f"{task.category} incident reported by a {task.report.reporter_role.value} "
                f"at {task.report.at_node}. Graded at the procedural floor. Verbatim report: "
                f'"{task.report.text[:400]}". '
                "[Deterministic fallback: no model synthesis available; procedure applied as written.]"
            ),
            "recommended_actions": [
                {
                    "action": task.retrieved[i].section.title if i < len(task.retrieved)
                    else "Apply standing procedure for this incident category.",
                    "sop_ref": refs[min(i, len(refs) - 1)],
                }
                for i in range(min(3, max(1, len(refs))))
            ],
            "citations": [
                {"doc_id": ref.split("#")[0], "section": ref.split("#")[1]} for ref in refs[:4]
            ],
            "confidence": 1.0,
        }
