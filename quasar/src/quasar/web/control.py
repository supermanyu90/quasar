"""The control room: run an agent through the barrier, then actuate an approved plan."""

from __future__ import annotations


from typing import Any, Mapping, Sequence

from quasar import demo_data
from quasar.agents import CrowdIntelligenceAgent, CrowdTask, IncidentResponseAgent, IncidentTask, PlanTask, PlannerAgent
from quasar.governance import AuditLog

from quasar.web.core import ModelPlane, _plane, assessment, orchestrator, zone_of
from quasar.web.serializers import result_json, route_json


def run_agent(
    name: str,
    venue_id: str,
    plane: ModelPlane,
    *,
    brief: Mapping[str, Any] | None = None,
    audit_chain: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], AuditLog]:
    orch = orchestrator(venue_id, plane, audit_chain)
    p, snap, a = assessment(venue_id)
    det = _plane(venue_id)
    hits = orch.retriever.for_incident(p.fixture.category, p.fixture.report.text)
    correlation = f"cyc-{p.id[:8]}-0001"

    match name:
        case "crowd":
            result = orch.runner.run(
                CrowdIntelligenceAgent(),
                CrowdTask(correlation_id=correlation, assessment=a),
            )
        case "incident":
            result = orch.runner.run(
                IncidentResponseAgent(),
                IncidentTask(
                    correlation_id=correlation,
                    report=p.fixture.report,
                    assessment=a,
                    category=p.fixture.category,
                    retrieved=hits,
                    severity_floor=det.severity_floor(
                        a, p.fixture.casualty_node, p.fixture.category
                    ),
                    zone=zone_of(venue_id, p.fixture.casualty_node),
                ),
                retrieved=hits,
            )
        case "plan":
            if brief is None:
                raise ValueError("the planner needs the incident brief")
            from quasar import schemas

            # The brief came from a browser. Re-validate before it shapes a plan.
            schemas.validate(brief, schemas.INCIDENT_BRIEF)
            result = orch.runner.run(
                PlannerAgent(),
                PlanTask(
                    correlation_id=correlation,
                    plan_id=f"plan-{p.fixture.report.id}",
                    brief=brief,
                    assessment=a,
                    snapshot=snap,
                    casualty_node=p.fixture.casualty_node,
                    retrieved=hits,
                    plane=det,
                ),
            )
        case _:
            raise ValueError(f"unknown agent {name!r}")

    return result_json(result), orch.audit


def actuate(
    plan: Mapping[str, Any],
    approver: str | None,
    venue_id: str,
    model_plane: ModelPlane,
    *,
    audit_chain: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], AuditLog]:
    """Policy, then the human barrier, then execution — on a plan that came from a
    browser and is therefore assumed hostile until it has passed all three."""
    from quasar import schemas

    orch = orchestrator(venue_id, model_plane, audit_chain)
    p, snap, a = assessment(venue_id)
    det = _plane(venue_id)

    schemas.validate(plan, schemas.PLAN_PROPOSAL)

    corroboration = PlannerAgent().corroborate(
        plan,
        PlanTask(
            correlation_id=f"cyc-{p.id[:8]}-0001",
            plan_id=plan["plan_id"],
            brief={"severity": plan["severity"]},
            assessment=a,
            snapshot=snap,
            casualty_node=p.fixture.casualty_node,
            retrieved=(),
            plane=det,
        ),
    )
    if corroboration.fatal:
        orch.audit.append(
            "plan.rejected",
            {"plan_id": plan["plan_id"], "reason": list(corroboration.notes)},
        )
        raise ValueError("; ".join(corroboration.notes))

    orch.submit_for_approval(plan)
    approval = None
    if approver:
        operator = demo_data.OPERATORS.get(approver)
        if operator is None:
            raise ValueError(f"unknown operator role {approver!r}")
        approval = orch.hitl.decide(operator, plan["plan_id"], approved=True, note="via console")

    execution = orch.actuate(
        plan, a, snap, languages=list(p.languages), approval=approval
    )

    return {
        "plan_id": execution.plan_id,
        "applied": list(execution.applied),
        "cordoned": sorted(execution.cordoned),
        "routes": {k: route_json(r) for k, r in execution.routes.items()},
        "gate_after": {
            gid: {
                "utilisation": round(m.utilisation, 3),
                "breaches": m.breaches_trigger,
                "open_lanes": execution.gate_state[gid].open_lanes,
            }
            for gid, m in execution.gate_after.items()
        },
        "gate_before": {gid: round(m.utilisation, 3) for gid, m in sorted(a.gates.items())},
        "dispatches": [
            {
                "template_id": d.template_id,
                "announcements": [
                    {"language": x.language, "text": x.text, "status": x.status.value}
                    for x in d.announcements
                ],
                "refused_languages": list(d.refused_languages),
                "pictogram": d.pictogram,
                "steward_required": d.steward_required,
            }
            for d in execution.dispatches
        ],
        "escalations": list(execution.escalations),
        "warnings": list(execution.warnings),
        "approved_by": approval.operator.name if approval else None,
    }, orch.audit
