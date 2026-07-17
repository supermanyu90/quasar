"""CrowdIntelligenceAgent -- reads the corridor state and names the story in it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from quasar import schemas
from quasar.crowd import level_of_service
from quasar.llm import ModelRequest
from quasar.plane import Assessment
from quasar.agents.base import Agent, Corroboration, _HOUSE_RULES


@dataclass(frozen=True, slots=True)
class CrowdTask:
    correlation_id: str
    assessment: Assessment
    previous: Assessment | None = None


class CrowdIntelligenceAgent(Agent):
    id = "CrowdIntelligenceAgent"
    schema_id = schemas.CROWD_ASSESSMENT
    VALUE_OVER_FALLBACK = (
        "The fallback lists hotspots. The agent explains them -- it correlates a "
        "rising corridor density with the gate that is feeding it and the fixture "
        "event that caused it (half-time, a goal, a weather hold), which is the "
        "difference between an operator seeing twelve red numbers and an operator "
        "seeing one story. No deterministic system can write that sentence, because "
        "the causal link is not in the telemetry; it is in the operator's world model."
    )
    system = (
        _HOUSE_RULES
        + """
Your role: read the deterministic crowd assessment and turn it into a situational
summary for the command centre.

You are given, for every corridor above the advisory density, its measured density,
its Fruin level of service, and its trend. You are also given every gate's queue
utilisation. These are measurements. Report them; do not re-derive them.

Your value is the synthesis: which of these numbers are the same problem, what is
driving it, and what an operator should look at first. Set action_required on a
gate if and only if its utilisation is at or above 0.90.
"""
    )

    def request(self, task: CrowdTask) -> ModelRequest:
        a = task.assessment
        lines = [f"correlation_id: {task.correlation_id}", "", "Corridors above advisory density:"]
        if a.hotspots:
            for h in a.hotspots:
                lines.append(
                    f"  {h.edge_id} (zone {h.zone}): {h.density:.2f} ped/m2, "
                    f"LOS {h.los.value}, {h.trend}"
                )
        else:
            lines.append("  (none)")
        lines += ["", "Gates:"]
        for gid, m in sorted(a.gates.items()):
            wait = "unstable" if not m.stable else f"{m.wait_s:.0f}s wait"
            lines.append(
                f"  {gid}: utilisation {m.utilisation:.2f}, {wait}, "
                f"{'BREACH' if m.breaches_trigger else 'ok'}"
            )
        return ModelRequest(
            system=self.system,
            user="\n".join(lines),
            schema_id=self.schema_id,
            effort="medium",
        )

    def corroborate(self, payload: Mapping[str, Any], task: CrowdTask) -> Corroboration:
        a = task.assessment
        notes: list[str] = []

        truth_edges = {h.edge_id for h in a.hotspots}
        claimed = {h["edge_id"] for h in payload["hotspots"]}
        if unknown := claimed - set(a.edge_density):
            return Corroboration.fail(f"reports hotspots on unknown edges: {sorted(unknown)}")
        if missed := truth_edges - claimed:
            notes.append(f"omits {len(missed)} hotspot(s): {sorted(missed)}")
        if invented := claimed - truth_edges:
            notes.append(f"reports {len(invented)} non-hotspot edge(s): {sorted(invented)}")

        union = truth_edges | claimed
        coverage = len(truth_edges & claimed) / len(union) if union else 1.0

        # Any restatement of a measured density that drifts from the measurement
        # is fatal: the operator will act on the number, and the number is wrong.
        for h in payload["hotspots"]:
            actual = a.density(h["edge_id"])
            if abs(h["density_ped_m2"] - actual) > 0.05:
                return Corroboration.fail(
                    f"restates {h['edge_id']} density as {h['density_ped_m2']} "
                    f"(measured {actual:.2f})"
                )
            if h["los"] != level_of_service(actual).value:
                return Corroboration.fail(
                    f"restates {h['edge_id']} LOS as {h['los']} "
                    f"(measured {level_of_service(actual).value})"
                )

        gate_score = 1.0
        for g in payload["gate_pressure"]:
            metrics = a.gates.get(g["gate_id"])
            if metrics is None:
                return Corroboration.fail(f"reports an unknown gate {g['gate_id']!r}")
            if g["action_required"] != metrics.breaches_trigger:
                gate_score = 0.0
                notes.append(
                    f"{g['gate_id']}: action_required={g['action_required']} but "
                    f"utilisation is {metrics.utilisation:.2f}"
                )

        return Corroboration(
            score=0.6 * coverage + 0.4 * gate_score, notes=tuple(notes)
        )

    def fallback(self, task: CrowdTask) -> dict[str, Any]:
        a = task.assessment
        breaching = a.breaching_gates
        summary = (
            f"{len(a.hotspots)} corridor(s) at or above advisory density; "
            f"{len(a.critical_edges)} at LOS F. "
            + (f"Gate(s) {', '.join(breaching)} at or above the 0.90 utilisation trigger."
               if breaching else "All gates below the utilisation trigger.")
            + " [Deterministic fallback: generated from telemetry without model synthesis.]"
        )
        return {
            "schema": self.schema_id,
            "correlation_id": task.correlation_id,
            "summary": summary,
            "hotspots": [
                {
                    "edge_id": h.edge_id,
                    "density_ped_m2": round(h.density, 2),
                    "los": h.los.value,
                    "trend": h.trend,
                }
                for h in a.hotspots
            ],
            "gate_pressure": [
                {
                    "gate_id": gid,
                    "utilisation": round(min(m.utilisation, 10.0), 2),
                    "action_required": m.breaches_trigger,
                }
                for gid, m in sorted(a.gates.items())
            ],
            "confidence": 1.0,
        }
