"""PlannerAgent -- sequences approved, feasible actions the deterministic plane prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from quasar import schemas
from quasar.crowd import CRITICAL_DENSITY
from quasar.language import CATALOGUE, SlotKind
from quasar.llm import ModelRequest
from quasar.plane import Assessment, DeterministicPlane
from quasar.rag import Retrieved, render_context
from quasar.routing import NoRouteError
from quasar.types import NodeId, TelemetrySnapshot
from quasar.agents.base import Agent, Corroboration, _HOUSE_RULES


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
