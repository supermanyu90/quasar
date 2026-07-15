"""The six generative components, each with a deterministic twin.

Every agent here has three parts, and the third is the one that matters:

1. a **prompt** -- what we ask the model to do;
2. a **corroborator** -- a deterministic function that scores the model's answer
   against ground truth the model did not produce and cannot influence;
3. a **fallback** -- a deterministic implementation of the same output that runs
   when the model is unreachable, unparseable, ungrounded, or uncorroborated.

The fallback is not a stub. It is a real, SOP-derived implementation that the
venue could run on for the whole fixture with a degraded but safe experience.
That is the test of whether GenAI is load-bearing or decorative here: what is
*lost* when the fallback fires. For each agent the answer is stated in
``VALUE_OVER_FALLBACK`` -- and it is never "safety". Safety is what the fallback
guarantees. What GenAI buys is synthesis, language coverage, and the ability to
handle the inputs nobody enumerated in advance.

The corroborator is what makes the confidence gate meaningful. A model's
self-reported confidence is a generated token, not a calibrated probability, and
gating on it alone is theatre. Governance gates on
``min(self_reported, corroboration_score)``, and a *fatal* corroboration failure
-- an action naming a gate that does not exist, an incident graded below its SOP
floor -- forces the fallback no matter how confident the model claims to be.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from quasar import schemas
from quasar.crowd import CRITICAL_DENSITY, level_of_service
from quasar.language import CATALOGUE, SUPPORTED_LANGUAGES, SlotKind, Tier
from quasar.llm import ModelRequest
from quasar.plane import Assessment, DeterministicPlane
from quasar.rag import Retrieved, render_context
from quasar.routing import NoRouteError
from quasar.types import (
    IncidentReport,
    LangCode,
    NodeId,
    Severity,
    TelemetrySnapshot,
)

# Shared preamble. Stable across the fixture, so it sits in the cached prefix.
_HOUSE_RULES = """\
You are a component of Quasar, the operating system of a 60,000-seat stadium on
a match day. Real people will move because of what you output.

Rules that are not negotiable:
- You do not compute distances, routes, capacities, walking times or evacuation
  times. Those are computed for you and given to you. Use the numbers you are
  given; never invent one and never adjust one.
- You never write the text of a safety-critical announcement. You choose an
  approved template and fill typed slots.
- If the evidence you are given does not support a conclusion, say so in your
  output and lower your confidence. An honest "I don't know" is cheap. A fluent
  wrong answer costs a life.
- Output exactly one JSON object conforming to the named schema. No prose.
"""


@dataclass(frozen=True, slots=True)
class Corroboration:
    """A deterministic score of a model payload against ground truth."""

    score: float
    notes: tuple[str, ...] = ()
    fatal: bool = False

    @staticmethod
    def ok() -> "Corroboration":
        return Corroboration(1.0)

    @staticmethod
    def fail(note: str) -> "Corroboration":
        return Corroboration(0.0, (note,), fatal=True)


def _bad_slot_values(
    template: Any, slots: Mapping[str, Any], venue: Any, zones: frozenset[str]
) -> list[str]:
    """Check every slot VALUE names a real thing in this venue.

    The message catalogue does this again at render time -- deliberately, since it
    is the last line before the public address system. Doing it here as well means
    a model that invents a gate is told so at corroboration, where it can be
    replaced by the deterministic plan, rather than at actuation, where the only
    option left is to fail loudly.
    """
    gates = {n.id for n in venue.nodes_tagged("gate")}
    problems: list[str] = []
    for slot in template.slots:
        value = slots.get(slot.name)
        match slot.kind:
            case SlotKind.GATE:
                if value not in gates:
                    problems.append(f"slot {slot.name}={value!r} is not a gate")
            case SlotKind.EDGE:
                if value not in venue.edges:
                    problems.append(f"slot {slot.name}={value!r} is not a corridor")
            case SlotKind.ZONE:
                if value not in zones:
                    problems.append(f"slot {slot.name}={value!r} is not a zone")
            case SlotKind.INTEGER:
                if not isinstance(value, int) or isinstance(value, bool):
                    problems.append(f"slot {slot.name}={value!r} is not an integer")
    return problems


class Agent(ABC):
    id: str
    schema_id: str
    system: str
    # What is lost when the deterministic fallback runs instead of the model.
    VALUE_OVER_FALLBACK: str = ""

    @abstractmethod
    def request(self, task: Any) -> ModelRequest: ...

    @abstractmethod
    def corroborate(self, payload: Mapping[str, Any], task: Any) -> Corroboration: ...

    @abstractmethod
    def fallback(self, task: Any) -> dict[str, Any]: ...


# ==========================================================================
# 1. CrowdIntelligenceAgent
# ==========================================================================


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


# ==========================================================================
# 2. IncidentResponseAgent
# ==========================================================================


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
        primary = refs[0]
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


# ==========================================================================
# 3. PlannerAgent
# ==========================================================================


@dataclass(frozen=True, slots=True)
class PlanTask:
    correlation_id: str
    plan_id: str
    brief: Mapping[str, Any]
    assessment: Assessment
    snapshot: TelemetrySnapshot
    casualty_node: NodeId | None
    retrieved: Sequence[Retrieved]
    plane: DeterministicPlane


class PlannerAgent(Agent):
    id = "PlannerAgent"
    schema_id = schemas.PLAN_PROPOSAL
    VALUE_OVER_FALLBACK = (
        "The fallback runs the playbook for the incident it was given. The agent "
        "handles the compound situation the playbook does not have a page for -- a "
        "medical emergency inside a corridor that is already the diversion route for "
        "a saturated gate, during a VIP movement -- by reasoning about which SOPs "
        "conflict and in what order to apply them. It never computes the routes or "
        "the capacities; it sequences actions the deterministic plane prices."
    )
    system = (
        _HOUSE_RULES
        + """
Your role: propose an ordered set of actions for the command centre to approve.

Available actions and their required parameters are fixed by schema. You choose
which actions, in which order, with which parameters. You do not compute their
consequences -- the deterministic plane does that, and it will reject any action
whose parameters are infeasible (a gate that does not exist, more lanes than are
installed, a route that cannot be walked).

Order matters and procedure constrains it. Read the cited SOP sections carefully:
one of them tells you whether a diversion is announced before or after a corridor
is cordoned, and getting that order backwards walks arriving spectators into a
closed corridor.

Every action must carry the SOP reference that authorises it.
"""
    )

    def request(self, task: PlanTask) -> ModelRequest:
        a = task.assessment
        lines = [
            f"correlation_id: {task.correlation_id}",
            f"plan_id: {task.plan_id}",
            f"severity: {task.brief['severity']}",
            "",
            "Incident brief:",
            f"  {task.brief['situation']}",
            "",
            "Corridors at LOS F (must be cordoned to spectators):",
        ]
        lines += [f"  {e}" for e in a.critical_edges] or ["  (none)"]
        lines += ["", "Gates at or above the 0.90 utilisation trigger:"]
        for gid in a.breaching_gates:
            t = task.snapshot.gates[gid]
            m = a.gates[gid]
            need = task.plane.lanes_needed(t)
            lines.append(
                f"  {gid}: utilisation {m.utilisation:.2f} with {t.open_lanes} of "
                f"{t.installed_lanes} lanes open; {need} lanes would be needed to hold "
                f"the trigger"
            )
        if not a.breaching_gates:
            lines.append("  (none)")
        if task.casualty_node:
            lines += ["", f"Casualty is at node {task.casualty_node}."]
            lines += ["Medical posts: " + ", ".join(
                n.id for n in task.plane.venue.nodes_tagged("medical")
            )]
        lines += ["", "Approved broadcast templates: " + ", ".join(sorted(CATALOGUE))]
        return ModelRequest(
            system=self.system,
            context=render_context(task.retrieved),
            user="\n".join(lines),
            schema_id=self.schema_id,
            effort="high",
        )

    def corroborate(self, payload: Mapping[str, Any], task: PlanTask) -> Corroboration:
        """Every action is checked for feasibility against the venue and the
        deterministic plane. An infeasible action is fatal -- there is no partial
        credit for a plan that dispatches a medic from a medical post that does
        not exist."""
        venue = task.plane.venue
        zones = frozenset(n.zone for n in venue.nodes.values())
        notes: list[str] = []

        for action in payload["actions"]:
            kind, params = action["type"], action["params"]
            match kind:
                case "DISPATCH_RESPONDER":
                    for key in ("from_node", "to_node"):
                        if params[key] not in venue.nodes:
                            return Corroboration.fail(
                                f"DISPATCH_RESPONDER names unknown node {params[key]!r}"
                            )
                    try:
                        task.plane.dispatch_route(
                            task.assessment,
                            from_node=params["from_node"],
                            to_node=params["to_node"],
                            cordoned=frozenset(task.assessment.critical_edges),
                        )
                    except NoRouteError:
                        return Corroboration.fail(
                            f"DISPATCH_RESPONDER {params['from_node']}->{params['to_node']} "
                            "has no feasible route"
                        )
                case "CORDON_EDGE":
                    if params["edge_id"] not in venue.edges:
                        return Corroboration.fail(
                            f"CORDON_EDGE names unknown edge {params['edge_id']!r}"
                        )
                    if task.assessment.density(params["edge_id"]) < CRITICAL_DENSITY:
                        notes.append(
                            f"cordons {params['edge_id']}, which is below LOS F"
                        )
                case "OPEN_LANES":
                    gate = task.snapshot.gates.get(params["gate_id"])
                    if gate is None:
                        return Corroboration.fail(
                            f"OPEN_LANES names unknown gate {params['gate_id']!r}"
                        )
                    if params["lanes"] > gate.installed_lanes:
                        return Corroboration.fail(
                            f"OPEN_LANES asks for {params['lanes']} lanes at "
                            f"{gate.gate_id}; only {gate.installed_lanes} are installed"
                        )
                case "DIVERT_ARRIVALS":
                    for key in ("from_gate", "to_gate"):
                        if params[key] not in task.snapshot.gates:
                            return Corroboration.fail(
                                f"DIVERT_ARRIVALS names unknown gate {params[key]!r}"
                            )
                    if params["from_gate"] == params["to_gate"]:
                        return Corroboration.fail("DIVERT_ARRIVALS diverts a gate to itself")
                case "REROUTE_FLOW":
                    if params["avoid_edge"] not in venue.edges:
                        return Corroboration.fail(
                            f"REROUTE_FLOW names unknown edge {params['avoid_edge']!r}"
                        )
                case "BROADCAST":
                    template = CATALOGUE.get(params["template_id"])
                    if template is None:
                        return Corroboration.fail(
                            f"BROADCAST names unknown template {params['template_id']!r}"
                        )
                    expected = {s.name for s in template.slots}
                    if set(params["slots"]) != expected:
                        return Corroboration.fail(
                            f"BROADCAST {params['template_id']} slots "
                            f"{sorted(params['slots'])} != required {sorted(expected)}"
                        )
                    # Slot NAMES matching is not enough. A live model announced a
                    # diversion for zone "NORTH-EAST" -- a zone that does not exist --
                    # and the old check waved it through, because it only compared key
                    # names. Every entity a broadcast names must be a real place in
                    # this venue.
                    if params["zone"] not in zones:
                        return Corroboration.fail(
                            f"BROADCAST addresses zone {params['zone']!r}, which does not "
                            f"exist in this venue (zones: {sorted(zones)})"
                        )
                    if bad := _bad_slot_values(template, params["slots"], venue, zones):
                        return Corroboration.fail(
                            f"BROADCAST {params['template_id']}: {'; '.join(bad)}"
                        )
                case "ESCALATE":
                    pass

        # Procedural ordering (SOP-MED-03#3): the diversion is announced before
        # the cordon goes in, never after. A model that gets this backwards is
        # fluent, plausible and wrong, and no schema can catch it.
        kinds = [a["type"] for a in payload["actions"]]
        if "BROADCAST" in kinds and "CORDON_EDGE" in kinds:
            if kinds.index("CORDON_EDGE") < kinds.index("BROADCAST"):
                return Corroboration.fail(
                    "cordons the corridor before announcing the diversion, contrary to "
                    "SOP-MED-03#3; arriving spectators would be walked into a closed corridor"
                )

        # COMPLETENESS. Feasibility asks "could each action be carried out?".
        # It never asked "does this plan actually address the emergency?" -- and a
        # live model exploited exactly that gap: every action it proposed was
        # perfectly executable, and it left a casualty lying in a LOS-F corridor
        # with no cordon and no medic. A plan of valid actions that ignores the
        # incident is the most dangerous thing this system can produce, because
        # everything about it looks right.
        #
        # So the deterministic plane states what MUST be answered, and the plan must
        # answer all of it -- or escalate to a human and say it cannot.
        unaddressed = self._unaddressed(payload, task)
        if unaddressed and "ESCALATE" not in kinds:
            return Corroboration.fail(
                "leaves the emergency unaddressed: "
                + "; ".join(unaddressed)
                + ". A plan that does not solve the problem must escalate, not proceed"
            )
        if unaddressed:
            notes.append("escalates rather than resolving: " + "; ".join(unaddressed))

        if task.brief["severity"] != payload["severity"]:
            notes.append(
                f"plan severity {payload['severity']} differs from brief "
                f"{task.brief['severity']}"
            )

        return Corroboration(score=1.0 if not notes else 0.85, notes=tuple(notes))

    @staticmethod
    def _unaddressed(payload: Mapping[str, Any], task: PlanTask) -> list[str]:
        """What the deterministic plane says must be answered, and the plan did not."""
        actions = payload["actions"]
        gaps: list[str] = []

        closed = {
            a["params"]["edge_id"] for a in actions if a["type"] == "CORDON_EDGE"
        } | {
            a["params"]["avoid_edge"] for a in actions if a["type"] == "REROUTE_FLOW"
        }
        for edge_id in task.assessment.critical_edges:
            if edge_id not in closed:
                gaps.append(
                    f"{edge_id} is at LOS F and is neither cordoned nor rerouted around"
                )

        if task.casualty_node is not None:
            reached = {
                a["params"]["to_node"]
                for a in actions
                if a["type"] == "DISPATCH_RESPONDER"
            }
            if task.casualty_node not in reached:
                gaps.append(f"no responder is dispatched to the casualty at {task.casualty_node}")

        relieved = {
            a["params"]["gate_id"] for a in actions if a["type"] == "OPEN_LANES"
        } | {
            a["params"]["from_gate"] for a in actions if a["type"] == "DIVERT_ARRIVALS"
        }
        for gate_id in task.assessment.breaching_gates:
            if gate_id not in relieved:
                gaps.append(
                    f"{gate_id} is above the 0.90 utilisation trigger and is not relieved"
                )

        return gaps

    def fallback(self, task: PlanTask) -> dict[str, Any]:
        """The SOP playbook, executed literally."""
        a = task.assessment
        actions: list[dict[str, Any]] = []

        critical = a.critical_edges
        if critical and task.casualty_node:
            edge = task.plane.venue.edge(critical[0])
            zone = task.plane.venue.node(edge.u).zone
            # Announce first, then cordon (SOP-MED-03#3).
            actions.append({
                "type": "BROADCAST",
                "sop_ref": "SOP-MED-03#3",
                "params": {
                    "template_id": "MSG-MED-CORRIDOR",
                    "zone": zone,
                    "slots": {"zone": zone, "corridor": critical[0]},
                },
            })
            actions.append({
                "type": "CORDON_EDGE",
                "sop_ref": "SOP-MED-03#3",
                "params": {
                    "edge_id": critical[0],
                    "reason": "level of service F with a medical incident in the corridor",
                },
            })

        if task.casualty_node:
            post = self._nearest_post(task)
            if post is not None:
                actions.append({
                    "type": "DISPATCH_RESPONDER",
                    "sop_ref": "SOP-MED-03#2",
                    "params": {
                        "from_node": post,
                        "to_node": task.casualty_node,
                        "responder_type": "medic",
                    },
                })

        for gid in a.breaching_gates:
            gate = task.snapshot.gates[gid]
            need = min(task.plane.lanes_needed(gate), gate.installed_lanes)
            if need > gate.open_lanes:
                actions.append({
                    "type": "OPEN_LANES",
                    "sop_ref": "SOP-QUEUE-02#2",
                    "params": {"gate_id": gid, "lanes": need},
                })

        if not actions:
            actions.append({
                "type": "ESCALATE",
                "sop_ref": "SOP-EVAC-01#1",
                "params": {
                    "to": "commander",
                    "reason": "no deterministic playbook matched the current state",
                },
            })

        return {
            "schema": self.schema_id,
            "correlation_id": task.correlation_id,
            "plan_id": task.plan_id,
            "severity": task.brief["severity"],
            "actions": actions[:8],
            "rationale": (
                "Deterministic fallback plan: standing procedure applied literally to the "
                "measured state. No model synthesis was available, so no cross-SOP "
                "reasoning has been applied and the plan does not account for interactions "
                "between concurrent incidents."
            ),
            "confidence": 1.0,
        }

    @staticmethod
    def _nearest_post(task: PlanTask) -> NodeId | None:
        assert task.casualty_node is not None
        best: tuple[float, NodeId] | None = None
        cordon = frozenset(task.assessment.critical_edges)
        for post in task.plane.venue.nodes_tagged("medical"):
            try:
                route = task.plane.dispatch_route(
                    task.assessment,
                    from_node=post.id,
                    to_node=task.casualty_node,
                    cordoned=cordon,
                )
            except NoRouteError:
                continue
            if best is None or route.eta_s < best[0]:
                best = (route.eta_s, post.id)
        return best[1] if best else None


# ==========================================================================
# 4. ConciergeAgent
# ==========================================================================


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

# The deterministic fallback's intent classifier. Keyword matching in the eleven
# languages we serve is exactly as brittle as it sounds -- which is the honest
# argument for the model, and the reason the fallback also offers a menu.
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


# ==========================================================================
# 5. CommunicationAgent
# ==========================================================================


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


# ==========================================================================
# 6. VolunteerBriefingAgent
# ==========================================================================


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
