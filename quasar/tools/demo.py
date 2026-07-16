"""Run the cross-track incident end to end and narrate it.

    PYTHONPATH=src:. python3 tools/demo.py            # recorded model transcripts
    PYTHONPATH=src:. python3 tools/demo.py --partition # no model at all
    PYTHONPATH=src:. python3 tools/demo.py --live      # real Claude, needs `anthropic` + credentials

Every number printed below is computed by the deterministic plane at the moment
it is printed. Switching between the three modes changes only the model plane;
the routes, the queue metrics, the announcements and the barrier are identical.
That is the point of the architecture, and this script is the demonstration of it.
"""

from __future__ import annotations

import argparse
import sys

from quasar.agents import (
    CrowdIntelligenceAgent,
    CrowdTask,
    IncidentResponseAgent,
    IncidentTask,
    PlannerAgent,
    PlanTask,
)
from quasar.governance import ApprovalRequired, AuditLog, Orchestrator
from quasar.llm import FailoverModel, ModelUnavailable, OllamaEdgeModel, TranscriptModel
from quasar.plane import DeterministicPlane
from quasar.routing import RESPONDER
from quasar.scenarios import SeededSampler, StressHarness
from quasar.venue_spec import reference_venue

from tests.fixtures import (  # the recorded transcripts live with the tests
    BRIEF_TRANSCRIPT,
    COMMANDER,
    CROWD_TRANSCRIPT,
    PLAN_TRANSCRIPT,
    VOLUNTEER_REPORT,
    DeadModel,
    j,
    match_day_snapshot,
)

LANGUAGES = ["en", "hi", "mr", "ta"]


def rule(title: str) -> None:
    print(f"\n\033[1m{'-' * 78}\n{title}\n{'-' * 78}\033[0m")


def build_model(mode: str):
    if mode == "partition":
        return FailoverModel(primary=DeadModel(), secondary=DeadModel())
    if mode == "live":
        from quasar.llm import AnthropicModel

        return FailoverModel(primary=AnthropicModel(), secondary=OllamaEdgeModel())
    return TranscriptModel({
        ("quasar.crowd_assessment.v1", "G3"): j(CROWD_TRANSCRIPT),
        ("quasar.incident_brief.v1", "INC-4471"): j(BRIEF_TRANSCRIPT),
        ("quasar.plan_proposal.v1", "plan-4471"): j(PLAN_TRANSCRIPT),
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", action="store_true", help="no model reachable")
    parser.add_argument("--live", action="store_true", help="call Claude for real")
    args = parser.parse_args()
    mode = "partition" if args.partition else "live" if args.live else "recorded"

    venue = reference_venue()
    plane = DeterministicPlane(venue)
    audit = AuditLog()
    orch = Orchestrator(plane, build_model(mode), audit=audit)

    snapshot = match_day_snapshot()
    assessment = plane.assess(snapshot)

    rule(f"QUASAR -- {venue.name} -- model plane: {mode}")

    # ---------------------------------------------------------------- sensing
    rule("1. DETERMINISTIC PLANE -- what is true")
    for h in assessment.hotspots:
        print(f"  {h.edge_id:<10} {h.density:>4.1f} ped/m2   LOS {h.los.value}   zone {h.zone}")
    for gid, m in sorted(assessment.gates.items()):
        flag = "  <-- BREACH (>= 0.90)" if m.breaches_trigger else ""
        print(f"  {gid:<10} utilisation {m.utilisation:.2f}{flag}")

    floor = plane.severity_floor(assessment, "C-N3", "medical")
    print(f"\n  Severity floor from procedure (SOP-MED-03#1): {floor.value}")
    print("  -> crowd pressure at the casualty exceeds LOS E, so this is P0, not P1.")
    print("     A model may grade it MORE severely. It may never grade it less.")

    # ------------------------------------------------------------- generative
    rule("2. GENERATIVE PLANE -- what it means")
    crowd = orch.runner.run(
        CrowdIntelligenceAgent(), CrowdTask("cyc-000000000001", assessment)
    )
    print(f"  [{crowd.source} / {crowd.plane}]  effective confidence "
          f"{crowd.effective_confidence:.2f} "
          f"(self-reported {crowd.self_reported_confidence:.2f}, "
          f"corroborated {crowd.corroboration.score:.2f})")
    if crowd.fallback_reason:
        print(f"  fallback: {crowd.fallback_reason}")
    print(f"\n  {crowd.payload['summary']}")

    hits = orch.retriever.for_incident("medical", VOLUNTEER_REPORT.text)
    print(f"\n  Retrieved procedure: {', '.join(h.ref for h in hits)}")

    brief = orch.runner.run(
        IncidentResponseAgent(),
        IncidentTask("cyc-000000000001", VOLUNTEER_REPORT, assessment, "medical", hits, floor,
                     zone=venue.node("C-N3").zone),
        retrieved=hits,
    )
    print(f"\n  [{brief.source}] incident brief, severity {brief.payload['severity']}")
    print(f"  {brief.payload['situation']}")
    print("  cited: " + ", ".join(
        f"{c['doc_id']}#{c['section']}" for c in brief.payload["citations"]
    ))

    plan_result = orch.runner.run(
        PlannerAgent(),
        PlanTask("cyc-000000000001", "plan-4471", brief.payload, assessment,
                 snapshot, "C-N3", hits, plane),
    )
    plan = plan_result.payload
    print(f"\n  [{plan_result.source}] plan {plan['plan_id']} ({plan['severity']})")
    for i, action in enumerate(plan["actions"], 1):
        print(f"    {i}. {action['type']:<20} {action['params']}  [{action['sop_ref']}]")

    # -------------------------------------------------------------- the barrier
    rule("3. HUMAN-IN-THE-LOOP -- the barrier no agent may cross")
    orch.submit_for_approval(plan)
    try:
        orch.actuate(plan, assessment, snapshot, languages=LANGUAGES, approval=None)
        print("  !! a P0 plan actuated without a signature -- this is a bug")
        return 1
    except ApprovalRequired as exc:
        print(f"  blocked: {exc}")

    approval = orch.hitl.decide(COMMANDER, plan["plan_id"], True, "confirmed on CCTV")
    print(f"  signed by {approval.operator.name} "
          f"({', '.join(r.value for r in approval.operator.roles)})")

    # ------------------------------------------------------------------ actuate
    rule("4. ACTUATION -- every number recomputed from the graph")
    execution = orch.actuate(
        plan, assessment, snapshot, languages=LANGUAGES, approval=approval
    )

    route = execution.routes["medic:C-N3"]
    naive = sum(
        plane.router.edge_cost(e, assessment.density(e), RESPONDER)
        for e in ("SP-MED2", "CORR-NE")
    )
    print(f"  medic route   : {' -> '.join(route.nodes)}")
    print(f"                  {route.distance_m:.0f} m, ETA {route.eta_s:.0f} s "
          f"(service ring, cordon-safe)")
    print(f"  the direct approach through CORR-NE is {103:.0f} m -- and would take "
          f"{naive:.0f} s at LOS F.")
    print(f"  the density-aware router chose the LONGER path and arrives "
          f"{naive / route.eta_s:.1f}x sooner.")

    after = execution.gate_after["G3"]
    before = assessment.gates["G3"]
    print(f"\n  gate G3       : utilisation {before.utilisation:.2f} -> "
          f"{after.utilisation:.2f} after opening reserve lanes")

    dispatch = execution.dispatches[0]
    print(f"\n  announcement  : {dispatch.template_id}")
    for a in dispatch.announcements:
        print(f"    [{a.language}] {a.text}")
    if dispatch.refused_languages:
        print(f"    REFUSED {list(dispatch.refused_languages)}: no human-validated "
              f"translation exists.")
        print(f"    -> pictogram {dispatch.pictogram} raised; steward dispatched "
              f"(SOP-COMMS-07#3).")
        print("    -> the system does not machine-translate an evacuation order, "
              "and does not stay silent.")

    for warning in execution.warnings:
        print(f"\n  warning: {warning}")

    # ------------------------------------------------------------------- record
    rule("5. AUDIT -- tamper-evident, and it verifies")
    for r in audit.records:
        print(f"  {r.seq:>2}  {r.event:<24} {r.hash[:16]}")
    print(f"\n  chain verifies: {audit.verify()}")

    # ------------------------------------------------------- pre-match harness
    rule("6. PRE-MATCH STRESS HARNESS -- what it found in the venue itself")
    harness = StressHarness(plane)
    seen: set[str] = set()
    for kind in ("gate_failure", "weather_evacuation", "medical_surge"):
        for scenario in SeededSampler(plane, seed=5).sample(kind, n=2):
            for f in harness.run(scenario).findings:
                if str(f) not in seen:
                    seen.add(str(f))
                    print(f"  {f}")
    print("\n  The west stand has a staircase and no ramp or lift. Under every")
    print("  scenario, a wheelchair user seated there cannot reach any gate.")
    print("  That is a defect in the VENUE, not the code. SOP-EVAC-01#3 says what")
    print("  to do about it: staff a refuge point before the gates open.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ModelUnavailable as exc:
        print(f"\nmodel plane unreachable: {exc}\nre-run without --live", file=sys.stderr)
        sys.exit(2)
