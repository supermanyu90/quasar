"""The cross-track integration test.

One incident, all four tracks, end to end, with nothing stubbed on the safety path.

    A spectator collapses at C-N3 during the second half. The corridor beside them
    is at LOS F. Gate 3 is simultaneously at 0.98 utilisation, still admitting late
    arrivals -- which is what is loading that corridor. A Marathi-speaking fan whose
    mother cannot manage stairs is at the wrong gate, on the other side of the
    stadium, trying to reach the north stand.

Every number in this test is computed by the deterministic plane at the moment it
is used. The model's contribution is recorded (see ``tests/fixtures.py``) so CI
runs offline -- but the router, the M/M/c model, the message catalogue, the schema
validator, the grounding check, the corroborators, the policy engine and the
human-in-the-loop barrier all execute for real, and it is their behaviour that is
asserted here.

The final test pulls the plug on the model entirely and requires the venue to keep
working.
"""

from __future__ import annotations

import unittest

from quasar.agents import (
    ConciergeAgent,
    ConciergeTask,
    CrowdIntelligenceAgent,
    CrowdTask,
    IncidentResponseAgent,
    IncidentTask,
    PlanTask,
    PlannerAgent,
    VolunteerBriefingAgent,
    VolunteerTask,
)
from quasar.governance import ApprovalRequired, AuditLog, NotAuthorised, Orchestrator
from quasar.llm import FailoverModel, TranscriptModel
from quasar.routing import ACCESSIBLE, RESPONDER
from quasar.types import Severity

from tests.fixtures import (
    zone_of,
    BRIEF_TRANSCRIPT,
    CASUALTY_NODE,
    COMMANDER,
    CONCIERGE_TRANSCRIPT,
    CROWD_TRANSCRIPT,
    PINCH,
    PLAN_TRANSCRIPT,
    STEWARD,
    VENUE,
    VOLUNTEER_BRIEF_TRANSCRIPT,
    VOLUNTEER_REPORT,
    DeadModel,
    j,
    match_day_snapshot,
    plane,
)


CORRELATION = "cyc-000000000001"

# The languages ticketing says are present in the north zone tonight.
ZONE_LANGUAGES = ["en", "hi", "mr", "ta"]

TRANSCRIPTS = {
    ("quasar.crowd_assessment.v1", "G3"): j(CROWD_TRANSCRIPT),
    ("quasar.incident_brief.v1", "INC-4471"): j(BRIEF_TRANSCRIPT),
    ("quasar.plan_proposal.v1", "plan-4471"): j(PLAN_TRANSCRIPT),
    ("quasar.concierge_reply.v1", "fan_language: mr"): j(CONCIERGE_TRANSCRIPT),
    ("quasar.volunteer_brief.v1", "VOL-218"): j(VOLUNTEER_BRIEF_TRANSCRIPT),
}


class CrossTrackCase(unittest.TestCase):
    def setUp(self) -> None:
        self.plane = plane()
        self.snapshot = match_day_snapshot()
        self.assessment = self.plane.assess(self.snapshot)
        self.audit = AuditLog()
        self.model = TranscriptModel(TRANSCRIPTS)
        self.orch = Orchestrator(self.plane, self.model, audit=self.audit)


class TestFullIncident(CrossTrackCase):
    """The whole pipeline, in the order it actually runs."""

    def test_medical_incident_in_a_dense_corridor(self) -> None:
        orch = self.orch

        # ---------------------------------------------------------------
        # 1. Sensing. The deterministic plane, and only it, decides what is true.
        # ---------------------------------------------------------------
        self.assertEqual(self.assessment.critical_edges, (PINCH,))
        self.assertEqual(self.assessment.breaching_gates, ("G3",))
        self.assertGreater(self.assessment.gates["G3"].utilisation, 0.90)

        # ---------------------------------------------------------------
        # TRACK 1 -- dynamic crowd management.
        # ---------------------------------------------------------------
        crowd = orch.runner.run(
            CrowdIntelligenceAgent(),
            CrowdTask(correlation_id=CORRELATION, assessment=self.assessment),
        )
        self.assertEqual(crowd.source, "model")
        # The model is allowed to explain the numbers. It is not allowed to change
        # them: the corroborator checked every restated density against the sensor.
        self.assertGreaterEqual(crowd.corroboration.score, 0.85)

        # ---------------------------------------------------------------
        # TRACK 3 -- real-time decision support. Grounded brief, then a plan.
        # ---------------------------------------------------------------
        hits = orch.retriever.for_incident("medical", VOLUNTEER_REPORT.text)
        floor = self.plane.severity_floor(self.assessment, CASUALTY_NODE, "medical")

        # Procedure, not the model, decides this is P0: crowd pressure at the scene
        # is above LOS E (SOP-MED-03#1).
        self.assertIs(floor, Severity.P0)

        brief = orch.runner.run(
            IncidentResponseAgent(),
            IncidentTask(
                correlation_id=CORRELATION,
                report=VOLUNTEER_REPORT,
                assessment=self.assessment,
                category="medical",
                retrieved=hits,
                severity_floor=floor,
                zone=zone_of(CASUALTY_NODE),
            ),
            retrieved=hits,
        )
        self.assertEqual(brief.source, "model")
        self.assertEqual(brief.payload["severity"], "P0")
        # Every citation names a section that exists AND was in the context.
        self.assertTrue(brief.payload["citations"])
        for citation in brief.payload["citations"]:
            ref = f"{citation['doc_id']}#{citation['section']}"
            self.assertIn(ref, {h.ref for h in hits})

        plan_result = orch.runner.run(
            PlannerAgent(),
            PlanTask(
                correlation_id=CORRELATION,
                plan_id="plan-4471",
                brief=brief.payload,
                assessment=self.assessment,
                snapshot=self.snapshot,
                casualty_node=CASUALTY_NODE,
                retrieved=hits,
                plane=self.plane,
            ),
        )
        self.assertEqual(plan_result.source, "model")
        plan = plan_result.payload

        # ---------------------------------------------------------------
        # 4. The human barrier. This is not advisory.
        # ---------------------------------------------------------------
        orch.submit_for_approval(plan)

        with self.assertRaises(ApprovalRequired):
            orch.actuate(
                plan, self.assessment, self.snapshot,
                languages=ZONE_LANGUAGES, approval=None,
            )
        with self.assertRaises(NotAuthorised):
            orch.hitl.decide(STEWARD, plan["plan_id"], approved=True)

        approval = orch.hitl.decide(
            COMMANDER, plan["plan_id"], approved=True, note="confirmed on CCTV"
        )

        # ---------------------------------------------------------------
        # 5. Actuation. Every number recomputed from the graph, now.
        # ---------------------------------------------------------------
        execution = orch.actuate(
            plan, self.assessment, self.snapshot,
            languages=ZONE_LANGUAGES, approval=approval,
        )

        # -- TRACK 2: smart indoor navigation (responder side).
        route = execution.routes["medic:C-N3"]
        self.assertIn(PINCH, execution.cordoned)
        self.assertNotIn(PINCH, route.edges)
        # The medic goes the LONGER way -- via the inner service ring -- because
        # under a LOS-F corridor the long way is more than twice as fast.
        self.assertTrue(all(VENUE.edge(e).staff_only for e in route.edges))

        through_the_crush = sum(
            self.plane.router.edge_cost(e, self.assessment.density(e), RESPONDER)
            for e in ("SP-MED2", PINCH)
        )
        self.assertLess(route.eta_s, 0.5 * through_the_crush)
        self.assertLess(route.eta_s, 200.0)

        # -- TRACK 1: the gate that was causing the corridor loading is relieved.
        after = execution.gate_after["G3"]
        self.assertLess(after.utilisation, 0.90)
        self.assertFalse(after.breaches_trigger)

        # -- TRACK 4: multilingual assistance, safety-critical tier.
        dispatch = execution.dispatches[0]
        self.assertEqual(set(dispatch.languages), {"en", "hi", "mr"})
        # Tamil is in the zone tonight and the catalogue HAS a Tamil string --
        # but it is a machine draft, so it is refused. The gap is covered by a
        # pictogram and a steward, and it is reported, not swallowed.
        self.assertEqual(dispatch.refused_languages, ("ta",))
        self.assertEqual(dispatch.pictogram, "PICTO-MEDICAL-CORRIDOR-CLOSED")
        self.assertTrue(dispatch.steward_required)
        self.assertTrue(any("steward dispatched" in w for w in execution.warnings))

        for announcement in dispatch.announcements:
            # The announcement names the corridor the way the SIGNAGE does.
            # An internal identifier on the public address system is useless:
            # no spectator has ever seen the string "CORR-NE".
            self.assertIn("North-East", announcement.text)
            self.assertNotIn("CORR-NE", announcement.text)
            self.assertNotIn("{", announcement.text)  # every slot was filled

        # -- Ordering: the diversion was announced BEFORE the corridor was closed.
        self.assertLess(
            execution.applied.index("BROADCAST"),
            execution.applied.index("CORDON_EDGE"),
        )

        # ---------------------------------------------------------------
        # 6. The record.
        # ---------------------------------------------------------------
        self.assertTrue(self.audit.verify())
        self.assertTrue(self.audit.events("hitl.decided"))
        self.assertEqual(len(self.audit.events("action.executed")), 4)
        signed = self.audit.events("hitl.decided")[0]
        self.assertEqual(signed.data["operator"], COMMANDER.id)
        self.assertTrue(signed.data["approved"])


class TestFanJourney(CrossTrackCase):
    """TRACK 2 + TRACK 4, from the fan's side."""

    def test_marathi_fan_with_a_step_free_requirement_at_the_wrong_gate(self) -> None:
        reply = self.orch.runner.run(
            ConciergeAgent(),
            ConciergeTask(
                correlation_id="cyc-000000000002",
                utterance=(
                    "माझी आई पायऱ्या चढू शकत नाही आणि आम्ही चुकीच्या गेटवर आलो आहोत. "
                    "आमच्या जागेपर्यंत कसे जायचे?"
                ),
                language="mr",
                at_node="G5",  # south gate; their seats are in the north stand
                accessible=True,
                assessment=self.assessment,
            ),
        )
        self.assertEqual(reply.source, "model")
        self.assertEqual(reply.payload["language"], "mr")
        self.assertEqual(reply.payload["safety_tier"], "informational")
        self.assertTrue(reply.payload["requires_route"])

        # The concierge decided WHERE. The router decides HOW, and it is bound by
        # constraints the concierge cannot override.
        route = self.plane.fan_route(
            self.assessment,
            from_node="G5",
            to_node="SEAT-N",
            profile=ACCESSIBLE,
            cordoned=frozenset({PINCH}),
        )
        for edge_id in route.edges:
            edge = VENUE.edge(edge_id)
            self.assertTrue(edge.step_free, f"step-free route crossed {edge_id}")
            self.assertFalse(edge.staff_only)
        self.assertNotIn(PINCH, route.edges)
        self.assertIn("RAMP-N", route.edges)
        # ...and it never sends them into the crush that an unimpeded adult would
        # be allowed to walk through.
        self.assertLess(route.worst_density, 1.075)

    def test_a_fan_reporting_an_emergency_takes_the_pen_away_from_the_model(self) -> None:
        """The concierge is the one place the model IS the product. It is also the
        place where, the moment the turn becomes safety-critical, the model stops
        being allowed to speak.

        Here the model classifies a collapse as an ordinary informational turn.
        The deterministic second opinion disagrees, and asymmetric costs mean we
        take the alarming reading, not the average.
        """
        wrong = dict(CONCIERGE_TRANSCRIPT)
        wrong["intent"] = "other"
        wrong["safety_tier"] = "informational"
        wrong["confidence"] = 0.97

        model = TranscriptModel({("quasar.concierge_reply.v1", "fan_language: mr"): j(wrong)})
        orch = Orchestrator(self.plane, model, audit=self.audit)

        reply = orch.runner.run(
            ConciergeAgent(),
            ConciergeTask(
                correlation_id="cyc-000000000002",
                utterance="मदत! एक माणूस पडला आहे, he has collapsed",
                language="mr",
                at_node="C-N3",
                accessible=False,
                assessment=self.assessment,
            ),
        )

        self.assertEqual(reply.source, "deterministic_fallback")
        self.assertIn("emergency vocabulary", reply.fallback_reason or "")
        self.assertEqual(reply.payload["safety_tier"], "safety_critical")
        self.assertEqual(reply.payload["intent"], "emergency")


class TestVolunteerBriefing(CrossTrackCase):
    def test_a_personalised_marathi_briefing_is_grounded(self) -> None:
        hits = self.orch.retriever.search(
            "concourse steward north zone congestion gate", k=6, pinned_docs=("SOP-QUEUE-02",)
        )
        result = self.orch.runner.run(
            VolunteerBriefingAgent(),
            VolunteerTask(
                correlation_id="cyc-000000000003",
                volunteer_id="VOL-218",
                language="mr",
                role="concourse steward",
                zone="NORTH",
                fixture="Semi-final, 19:30",
                known_risks=["CORR-NE congestion at half-time", "Gate 3 late arrivals"],
                retrieved=hits,
            ),
            retrieved=hits,
            require_retrieved_citations=False,
        )
        self.assertEqual(result.source, "model")
        self.assertEqual(result.payload["language"], "mr")
        self.assertEqual(result.payload["zone"], "NORTH")
        # Named entities survive into the Marathi briefing.
        body = " ".join(s["body"] for s in result.payload["sections"])
        self.assertIn("CORR-NE", body)
        self.assertIn("3", body)


class TestNetworkPartition(CrossTrackCase):
    """Pull the plug. The venue does not stop being a venue."""

    def setUp(self) -> None:
        super().setUp()
        # Cloud is gone. Edge box is gone too -- the worst case, not the easy one.
        self.orch = Orchestrator(
            self.plane, FailoverModel(primary=DeadModel(), secondary=DeadModel()),
            audit=self.audit,
        )

    def test_the_same_incident_with_no_model_at_all(self) -> None:
        orch = self.orch
        hits = orch.retriever.for_incident("medical", VOLUNTEER_REPORT.text)
        floor = self.plane.severity_floor(self.assessment, CASUALTY_NODE, "medical")

        brief = orch.runner.run(
            IncidentResponseAgent(),
            IncidentTask(
                correlation_id=CORRELATION,
                report=VOLUNTEER_REPORT,
                assessment=self.assessment,
                category="medical",
                retrieved=hits,
                severity_floor=floor,
                zone=zone_of(CASUALTY_NODE),
            ),
            retrieved=hits,
        )
        plan_result = orch.runner.run(
            PlannerAgent(),
            PlanTask(
                correlation_id=CORRELATION,
                plan_id="plan-4471",
                brief=brief.payload,
                assessment=self.assessment,
                snapshot=self.snapshot,
                casualty_node=CASUALTY_NODE,
                retrieved=hits,
                plane=self.plane,
            ),
        )

        self.assertEqual(brief.source, "deterministic_fallback")
        self.assertEqual(plan_result.source, "deterministic_fallback")

        plan = plan_result.payload

        # The severity is still right, because procedure -- not the model -- set it.
        self.assertEqual(plan["severity"], "P0")

        # The human barrier is still there. Degrading the model does not degrade
        # the governance.
        orch.submit_for_approval(plan)
        with self.assertRaises(ApprovalRequired):
            orch.actuate(
                plan, self.assessment, self.snapshot,
                languages=ZONE_LANGUAGES, approval=None,
            )
        approval = orch.hitl.decide(COMMANDER, plan["plan_id"], approved=True)

        execution = orch.actuate(
            plan, self.assessment, self.snapshot,
            languages=ZONE_LANGUAGES, approval=approval,
        )

        # And the venue still does the four things that matter:
        self.assertIn(PINCH, execution.cordoned)  # the corridor is closed
        route = execution.routes["medic:C-N3"]  # the medic is dispatched
        self.assertNotIn(PINCH, route.edges)  # by a safe route
        self.assertLess(execution.gate_after["G3"].utilisation, 0.90)  # gate relieved

        dispatch = execution.dispatches[0]  # and the crowd is told, in three languages
        self.assertEqual(set(dispatch.languages), {"en", "hi", "mr"})
        self.assertLess(
            execution.applied.index("BROADCAST"),
            execution.applied.index("CORDON_EDGE"),
        )
        self.assertTrue(self.audit.verify())

    def test_what_is_actually_lost_is_synthesis_not_safety(self) -> None:
        """The honest accounting of what GenAI buys, made testable.

        Under partition the venue is safe and the fan is served. What it loses is
        the sentence that tells the operator these are one problem -- and the
        ability to answer a Marathi speaker at all.
        """
        crowd = self.orch.runner.run(
            CrowdIntelligenceAgent(),
            CrowdTask(correlation_id=CORRELATION, assessment=self.assessment),
        )
        self.assertEqual(crowd.source, "deterministic_fallback")

        # Every hotspot is still reported: no safety-relevant information is lost.
        reported = {h["edge_id"] for h in crowd.payload["hotspots"]}
        self.assertEqual(reported, {h.edge_id for h in self.assessment.hotspots})
        self.assertTrue(
            [g for g in crowd.payload["gate_pressure"] if g["action_required"]]
        )

        # But the causal story -- Gate 3 is what is loading CORR-NE -- is gone.
        self.assertIn("Deterministic fallback", crowd.payload["summary"])
        self.assertNotIn("upstream", crowd.payload["summary"])

        # And the concierge degrades to a keyword matcher answering in English.
        reply = self.orch.runner.run(
            ConciergeAgent(),
            ConciergeTask(
                correlation_id="cyc-000000000002",
                utterance="माझी आई पायऱ्या चढू शकत नाही",
                language="mr",
                at_node="G5",
                accessible=True,
                assessment=self.assessment,
            ),
        )
        self.assertEqual(reply.source, "deterministic_fallback")
        self.assertEqual(reply.payload["intent"], "other")  # the keywords miss it


if __name__ == "__main__":
    unittest.main()
