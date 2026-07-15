"""The governance barrier: the gauntlet every model payload runs."""

from __future__ import annotations

import copy
import unittest

from quasar.agents import (
    CrowdIntelligenceAgent,
    CrowdTask,
    IncidentResponseAgent,
    IncidentTask,
    PlanTask,
    PlannerAgent,
)
from quasar.governance import (
    AgentRunner,
    ApprovalRequired,
    AuditLog,
    AuditRecord,
    Executor,
    HumanInTheLoop,
    NotAuthorised,
    Orchestrator,
    PolicyEngine,
    PolicyError,
)
from quasar.language import MessageCatalogue
from quasar.llm import TranscriptModel
from quasar.rag import Retriever
from quasar.types import Severity

from tests.fixtures import (
    zone_of,
    BRIEF_TRANSCRIPT,
    COMMANDER,
    CROWD_TRANSCRIPT,
    PLAN_TRANSCRIPT,
    STEWARD,
    VENUE,
    VOLUNTEER_REPORT,
    DeadModel,
    SequenceModel,
    j,
    match_day_snapshot,
    plane,
)



class GovernanceCase(unittest.TestCase):
    def setUp(self) -> None:
        self.plane = plane()
        self.snapshot = match_day_snapshot()
        self.assessment = self.plane.assess(self.snapshot)
        self.audit = AuditLog()
        self.retriever = Retriever()
        # Retrieved the way the orchestrator does it: the SOPs that *govern* a
        # medical incident are pinned, and BM25 adds what the report's words suggest.
        self.hits = self.retriever.for_incident("medical", VOLUNTEER_REPORT.text)

    def crowd_task(self) -> CrowdTask:
        return CrowdTask(correlation_id="cyc-000000000001", assessment=self.assessment)

    def incident_task(self) -> IncidentTask:
        return IncidentTask(
            correlation_id="cyc-000000000001",
            report=VOLUNTEER_REPORT,
            assessment=self.assessment,
            category="medical",
            retrieved=self.hits,
            severity_floor=self.plane.severity_floor(self.assessment, "C-N3", "medical"),
            zone=zone_of("C-N3"),
        )

    def plan_task(self) -> PlanTask:
        return PlanTask(
            correlation_id="cyc-000000000001",
            plan_id="plan-4471",
            brief=BRIEF_TRANSCRIPT,
            assessment=self.assessment,
            snapshot=self.snapshot,
            casualty_node="C-N3",
            retrieved=self.hits,
            plane=self.plane,
        )


class TestAuditChain(GovernanceCase):
    def test_the_chain_verifies(self) -> None:
        for i in range(5):
            self.audit.append("test.event", {"i": i})
        self.assertTrue(self.audit.verify())

    def test_altering_a_record_breaks_every_record_after_it(self) -> None:
        """The property that makes the log worth having: you cannot rewrite what
        the system proposed after you have seen how it turned out."""
        for i in range(5):
            self.audit.append("test.event", {"i": i})
        self.assertTrue(self.audit.verify())

        tampered = self.audit.records[2]
        self.audit._records[2] = AuditRecord(
            seq=tampered.seq,
            t=tampered.t,
            event=tampered.event,
            data={"i": 999},  # someone edits the history
            prev_hash=tampered.prev_hash,
            hash=tampered.hash,
        )
        self.assertFalse(self.audit.verify())

    def test_deleting_a_record_breaks_the_chain(self) -> None:
        for i in range(4):
            self.audit.append("test.event", {"i": i})
        del self.audit._records[1]
        self.assertFalse(self.audit.verify())

    def test_a_chain_survives_a_round_trip_through_a_stateless_client(self) -> None:
        """A serverless function has no memory between invocations, so the chain
        travels with the client and comes back. It must be resumable."""
        for i in range(3):
            self.audit.append("test.event", {"i": i})

        resumed = AuditLog.resume(self.audit.to_json())
        self.assertTrue(resumed.verify())

        resumed.append("test.event", {"i": 3})
        self.assertTrue(resumed.verify())
        self.assertEqual(len(resumed.records), 4)
        self.assertEqual(resumed.records[3].prev_hash, self.audit.records[2].hash)

    def test_a_chain_survives_a_round_trip_through_a_browser(self) -> None:
        """The bug this guards against: JavaScript has one number type, so a
        corroboration score of 1.0 comes back from the browser as 1. The chain would
        cry tampering when nobody had tampered -- a false positive on an integrity
        check, which is the fastest way to teach people to ignore it."""
        self.audit.append("agent.accepted", {"corroboration_score": 1.0, "conf": 0.94})
        self.audit.append("agent.accepted", {"corroboration_score": 0.85, "lanes": 12})

        def as_javascript_would(value):
            # JSON.stringify(1.0) -> "1"; JSON.parse gives a Number back as an int.
            if isinstance(value, bool):
                return value
            if isinstance(value, float) and value == int(value):
                return int(value)
            if isinstance(value, dict):
                return {k: as_javascript_would(v) for k, v in value.items()}
            if isinstance(value, list):
                return [as_javascript_would(v) for v in value]
            return value

        round_tripped = as_javascript_would(self.audit.to_json())
        self.assertEqual(round_tripped[0]["data"]["corroboration_score"], 1)  # not 1.0

        resumed = AuditLog.resume(round_tripped)  # must not raise
        self.assertTrue(resumed.verify())

    def test_a_tampered_chain_is_refused_rather_than_extended(self) -> None:
        """The client is now a potential attacker. A chain that does not verify is
        not appended to -- it is rejected at the door."""
        for i in range(3):
            self.audit.append("test.event", {"i": i})

        poisoned = self.audit.to_json()
        poisoned[1]["data"] = {"i": 999}

        with self.assertRaises(ValueError) as ctx:
            AuditLog.resume(poisoned)
        self.assertIn("tampered", str(ctx.exception))


class TestSeverityFloor(GovernanceCase):
    def test_a_medical_incident_in_a_los_f_corridor_is_p0_not_p1(self) -> None:
        """SOP-MED-03#1. The floor is computed from the measured crowd pressure at
        the scene, not from the model's reading of the report."""
        floor = self.plane.severity_floor(self.assessment, "C-N3", "medical")
        self.assertIs(floor, Severity.P0)

    def test_the_same_incident_in_a_calm_corridor_is_p1(self) -> None:
        from tests.fixtures import quiet_snapshot

        calm = self.plane.assess(quiet_snapshot())
        self.assertIs(self.plane.severity_floor(calm, "C-N3", "medical"), Severity.P1)


class TestAgentBarrier(GovernanceCase):
    def runner(self, model) -> AgentRunner:
        return AgentRunner(model, self.audit)

    def test_a_good_payload_is_accepted(self) -> None:
        model = TranscriptModel({("quasar.crowd_assessment.v1", "G3"): j(CROWD_TRANSCRIPT)})
        result = self.runner(model).run(CrowdIntelligenceAgent(), self.crowd_task())

        self.assertEqual(result.source, "model")
        self.assertGreaterEqual(result.effective_confidence, 0.85)
        self.assertTrue(self.audit.events("agent.accepted"))

    def test_a_dead_model_falls_back_to_a_schema_valid_deterministic_payload(self) -> None:
        """Network partition mid-fixture. The venue does not stop working."""
        result = self.runner(DeadModel()).run(CrowdIntelligenceAgent(), self.crowd_task())

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertEqual(result.plane, "deterministic")
        self.assertIn("model unavailable", result.fallback_reason or "")
        # The fallback still names every real hotspot -- the safety-relevant content
        # is all there; what is lost is the synthesis.
        edges = {h["edge_id"] for h in result.payload["hotspots"]}
        self.assertEqual(edges, {h.edge_id for h in self.assessment.hotspots})

    def test_a_hallucinated_density_is_fatal_even_at_high_confidence(self) -> None:
        """The model restates a measured number and gets it wrong. An operator
        would act on that number. Self-reported confidence of 0.99 buys nothing."""
        lying = copy.deepcopy(CROWD_TRANSCRIPT)
        lying["hotspots"][0]["density_ped_m2"] = 1.1  # actually 3.4 -- LOS F, not D
        lying["confidence"] = 0.99

        model = TranscriptModel({("quasar.crowd_assessment.v1", "G3"): j(lying)})
        result = self.runner(model).run(CrowdIntelligenceAgent(), self.crowd_task())

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("corroboration failed", result.fallback_reason or "")
        self.assertIn("CORR-NE", result.fallback_reason or "")

    def test_undercutting_the_severity_floor_is_fatal(self) -> None:
        """The failure mode a fluent model is most prone to: reading a panicked
        report calmly and grading it P1 when procedure says P0."""
        soft = copy.deepcopy(BRIEF_TRANSCRIPT)
        soft["severity"] = "P1"
        soft["confidence"] = 0.97

        model = TranscriptModel({("quasar.incident_brief.v1", "INC-4471"): j(soft)})
        result = self.runner(model).run(
            IncidentResponseAgent(), self.incident_task(), retrieved=self.hits
        )

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("floor", result.fallback_reason or "")
        # The fallback grades it at the floor, which is where it belongs.
        self.assertEqual(result.payload["severity"], "P0")

    def test_grading_more_severely_than_the_floor_is_allowed(self) -> None:
        """The floor is a floor, not a target."""
        from tests.fixtures import quiet_snapshot

        calm = self.plane.assess(quiet_snapshot())
        task = IncidentTask(
            correlation_id="cyc-000000000001",
            report=VOLUNTEER_REPORT,
            assessment=calm,
            category="medical",
            retrieved=self.hits,
            severity_floor=self.plane.severity_floor(calm, "C-N3", "medical"),  # P1
            zone=zone_of("C-N3"),
        )
        self.assertIs(task.severity_floor, Severity.P1)

        model = TranscriptModel({("quasar.incident_brief.v1", "INC-4471"): j(BRIEF_TRANSCRIPT)})
        result = self.runner(model).run(IncidentResponseAgent(), task, retrieved=self.hits)

        self.assertEqual(result.source, "model")
        self.assertEqual(result.payload["severity"], "P0")  # upgraded by the model

    def test_a_fabricated_citation_is_rejected(self) -> None:
        """SOP-EVAC-01#9 does not exist. A citation to it is a hallucination with
        a footnote."""
        fake = copy.deepcopy(BRIEF_TRANSCRIPT)
        fake["citations"] = [{"doc_id": "SOP-EVAC-01", "section": "9"}]

        model = TranscriptModel({("quasar.incident_brief.v1", "INC-4471"): j(fake)})
        result = self.runner(model).run(
            IncidentResponseAgent(), self.incident_task(), retrieved=self.hits
        )

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("non-existent", result.fallback_reason or "")

    def test_citing_a_real_section_that_was_never_retrieved_is_rejected(self) -> None:
        """SOP-VIP-04#1 exists, but was not in the context for a medical incident.
        The model cannot have read it, so it cannot honestly cite it."""
        unretrieved = copy.deepcopy(BRIEF_TRANSCRIPT)
        unretrieved["citations"] = [{"doc_id": "SOP-VIP-04", "section": "1"}]
        self.assertNotIn("SOP-VIP-04#1", {h.ref for h in self.hits})

        model = TranscriptModel({("quasar.incident_brief.v1", "INC-4471"): j(unretrieved)})
        result = self.runner(model).run(
            IncidentResponseAgent(), self.incident_task(), retrieved=self.hits
        )

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("not in the retrieved context", result.fallback_reason or "")

    def test_low_self_reported_confidence_falls_back(self) -> None:
        timid = copy.deepcopy(CROWD_TRANSCRIPT)
        timid["confidence"] = 0.4

        model = TranscriptModel({("quasar.crowd_assessment.v1", "G3"): j(timid)})
        result = self.runner(model).run(CrowdIntelligenceAgent(), self.crowd_task())

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("below floor", result.fallback_reason or "")

    def test_a_schema_violation_gets_exactly_one_repair_attempt(self) -> None:
        broken = copy.deepcopy(CROWD_TRANSCRIPT)
        del broken["summary"]

        model = SequenceModel(j(broken), j(CROWD_TRANSCRIPT))
        result = self.runner(model).run(CrowdIntelligenceAgent(), self.crowd_task())

        self.assertEqual(result.source, "model_repaired")
        self.assertEqual(len(model.requests), 2)
        # The repair prompt carries the actual violation, not a generic scolding.
        self.assertIn("summary", model.requests[1].user)
        self.assertTrue(self.audit.events("agent.repair_attempted"))

    def test_a_violation_that_survives_the_repair_falls_back(self) -> None:
        broken = copy.deepcopy(CROWD_TRANSCRIPT)
        del broken["summary"]

        model = SequenceModel(j(broken), j(broken))  # the model does not learn
        result = self.runner(model).run(CrowdIntelligenceAgent(), self.crowd_task())

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("survived one repair", result.fallback_reason or "")

    def test_unparseable_output_is_repaired_not_guessed_at(self) -> None:
        model = SequenceModel("I'm sorry, I can't help with that.", j(CROWD_TRANSCRIPT))
        result = self.runner(model).run(CrowdIntelligenceAgent(), self.crowd_task())
        self.assertEqual(result.source, "model_repaired")

    def test_a_fenced_json_block_is_tolerated(self) -> None:
        """The edge model wraps its output in markdown. That is a formatting quirk,
        not a safety event -- do not burn a repair round trip on it."""
        model = SequenceModel(f"```json\n{j(CROWD_TRANSCRIPT)}\n```")
        result = self.runner(model).run(CrowdIntelligenceAgent(), self.crowd_task())
        self.assertEqual(result.source, "model")

    def test_an_infeasible_plan_is_fatal(self) -> None:
        """The model asks for 14 lanes at a gate with 12. The queueing model would
        happily compute a number for that. The corroborator refuses to let it."""
        impossible = copy.deepcopy(PLAN_TRANSCRIPT)
        impossible["actions"][3]["params"]["lanes"] = 14

        model = TranscriptModel({("quasar.plan_proposal.v1", "plan-4471"): j(impossible)})
        result = self.runner(model).run(PlannerAgent(), self.plan_task())

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("installed", result.fallback_reason or "")

    def test_cordoning_before_announcing_is_fatal(self) -> None:
        """SOP-MED-03#3. Reversing these two actions is fluent, plausible, and
        walks arriving spectators into a closed corridor. No schema catches it;
        the corroborator does."""
        reversed_plan = copy.deepcopy(PLAN_TRANSCRIPT)
        actions = reversed_plan["actions"]
        actions[0], actions[1] = actions[1], actions[0]

        model = TranscriptModel({("quasar.plan_proposal.v1", "plan-4471"): j(reversed_plan)})
        result = self.runner(model).run(PlannerAgent(), self.plan_task())

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("SOP-MED-03#3", result.fallback_reason or "")

    # ------------------------------------------------------------------
    # Holes found by running a REAL model (gemma3:4b) against the barrier.
    # Every one of these plans is made of perfectly executable actions, and
    # every one of them would have got someone hurt. The recorded transcripts
    # never caught them, because I wrote the transcripts and I wrote them right.
    # ------------------------------------------------------------------

    def test_a_plan_that_ignores_the_casualty_is_fatal(self) -> None:
        """Feasibility asks 'could each action be carried out?'. It never asked
        'does this plan address the emergency?' -- and a live model walked straight
        through the gap, leaving a man on the floor of a LOS-F corridor."""
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        plan["actions"] = [
            a for a in plan["actions"] if a["type"] != "DISPATCH_RESPONDER"
        ]
        model = TranscriptModel({("quasar.plan_proposal.v1", "plan-4471"): j(plan)})
        result = self.runner(model).run(PlannerAgent(), self.plan_task())

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("no responder is dispatched", result.fallback_reason or "")

    def test_a_plan_that_leaves_a_los_f_corridor_open_is_fatal(self) -> None:
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        plan["actions"] = [a for a in plan["actions"] if a["type"] != "CORDON_EDGE"]
        model = TranscriptModel({("quasar.plan_proposal.v1", "plan-4471"): j(plan)})
        result = self.runner(model).run(PlannerAgent(), self.plan_task())

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("CORR-NE", result.fallback_reason or "")
        self.assertIn("neither cordoned nor rerouted", result.fallback_reason or "")

    def test_a_plan_that_leaves_a_saturated_gate_unrelieved_is_fatal(self) -> None:
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        plan["actions"] = [a for a in plan["actions"] if a["type"] != "OPEN_LANES"]
        model = TranscriptModel({("quasar.plan_proposal.v1", "plan-4471"): j(plan)})
        result = self.runner(model).run(PlannerAgent(), self.plan_task())

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("G3", result.fallback_reason or "")
        self.assertIn("not relieved", result.fallback_reason or "")

    def test_a_plan_may_escalate_instead_of_solving_but_must_say_so(self) -> None:
        """An honest 'I cannot handle this, wake a human' is a legitimate answer.
        Silently doing nothing is not."""
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        plan["actions"] = [
            {
                "type": "ESCALATE",
                "sop_ref": "SOP-EVAC-01#1",
                "params": {"to": "commander", "reason": "compound incident beyond the playbook"},
            }
        ]
        model = TranscriptModel({("quasar.plan_proposal.v1", "plan-4471"): j(plan)})
        result = self.runner(model).run(PlannerAgent(), self.plan_task())

        self.assertEqual(result.source, "model")
        self.assertTrue(result.corroboration.notes)
        self.assertIn("escalates rather than resolving", result.corroboration.notes[0])

    def test_a_broadcast_to_a_zone_that_does_not_exist_is_fatal(self) -> None:
        """The live model announced a diversion for zone 'NORTH-EAST'. There is no
        such zone. The old check compared slot *names* and waved it through."""
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        plan["actions"][0]["params"]["zone"] = "NORTH-EAST"
        model = TranscriptModel({("quasar.plan_proposal.v1", "plan-4471"): j(plan)})
        result = self.runner(model).run(PlannerAgent(), self.plan_task())

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("NORTH-EAST", result.fallback_reason or "")
        self.assertIn("does not exist", result.fallback_reason or "")

    def test_a_broadcast_naming_a_gate_that_does_not_exist_is_fatal(self) -> None:
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        plan["actions"][0]["params"] = {
            "template_id": "MSG-GATE-DIVERT",
            "zone": "NORTH",
            "slots": {"from_gate": "G3", "to_gate": "G99"},  # G99 is not a gate
        }
        model = TranscriptModel({("quasar.plan_proposal.v1", "plan-4471"): j(plan)})
        result = self.runner(model).run(PlannerAgent(), self.plan_task())

        self.assertEqual(result.source, "deterministic_fallback")
        self.assertIn("is not a gate", result.fallback_reason or "")

    def test_the_fallback_itself_must_satisfy_the_contract(self) -> None:
        """A fallback that does not validate is a bug in us, not a degraded mode.
        It must blow up here rather than downstream."""
        from quasar import schemas

        for agent, task in (
            (CrowdIntelligenceAgent(), self.crowd_task()),
            (IncidentResponseAgent(), self.incident_task()),
            (PlannerAgent(), self.plan_task()),
        ):
            schemas.validate(agent.fallback(task), agent.schema_id)


class TestPolicy(GovernanceCase):
    def setUp(self) -> None:
        super().setUp()
        self.policy = PolicyEngine()

    def test_a_good_plan_passes(self) -> None:
        self.policy.check(PLAN_TRANSCRIPT, self.assessment)

    def test_an_evacuation_announcement_requires_p0(self) -> None:
        """SOP-EVAC-01#1. An evacuation is not something a P1 plan may reach for,
        no matter how confidently it argues for one."""
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        plan["severity"] = "P1"
        plan["actions"][0]["params"] = {
            "template_id": "MSG-EVAC-GATE",
            "zone": "NORTH",
            "slots": {"gate": "G4"},
        }
        with self.assertRaises(PolicyError) as ctx:
            self.policy.check(plan, self.assessment)
        self.assertIn("evacuation", str(ctx.exception))

    def test_blast_radius_caps_the_number_of_cordons(self) -> None:
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        plan["actions"] = [
            {
                "type": "CORDON_EDGE",
                "sop_ref": "SOP-MED-03#3",
                "params": {"edge_id": e, "reason": "test cordon"},
            }
            for e in ("CORR-NE", "CORR-N2", "CORR-N1", "CORR-E")
        ]
        with self.assertRaises(PolicyError) as ctx:
            self.policy.check(plan, self.assessment)
        self.assertIn("blast-radius", str(ctx.exception))

    def test_a_single_action_cannot_divert_more_than_half_a_gate(self) -> None:
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        plan["actions"] = [{
            "type": "DIVERT_ARRIVALS",
            "sop_ref": "SOP-QUEUE-02#2",
            "params": {"from_gate": "G3", "to_gate": "G2", "share": 0.9},
        }]
        with self.assertRaises(PolicyError):
            self.policy.check(plan, self.assessment)

    def test_the_policy_binds_the_deterministic_fallback_too(self) -> None:
        """The caps are not a distrust of the model specifically. A buggy playbook
        can cordon the whole venue just as effectively as a confused model can."""
        fallback = PlannerAgent().fallback(self.plan_task())
        self.policy.check(fallback, self.assessment)  # must not raise


class TestHumanInTheLoop(GovernanceCase):
    def setUp(self) -> None:
        super().setUp()
        self.hitl = HumanInTheLoop(self.audit)
        self.hitl.submit("plan-4471", Severity.P0, "cordon + dispatch")

    def test_a_p0_plan_cannot_actuate_without_a_signature(self) -> None:
        with self.assertRaises(ApprovalRequired):
            self.hitl.check("plan-4471", Severity.P0, None)

    def test_a_steward_may_not_approve_a_p0_action(self) -> None:
        with self.assertRaises(NotAuthorised):
            self.hitl.decide(STEWARD, "plan-4471", approved=True)
        self.assertTrue(self.audit.events("hitl.refused"))

    def test_a_commander_may(self) -> None:
        approval = self.hitl.decide(COMMANDER, "plan-4471", approved=True, note="go")
        self.hitl.check("plan-4471", Severity.P0, approval)  # must not raise
        self.assertTrue(self.audit.events("hitl.decided"))

    def test_a_rejection_blocks_actuation(self) -> None:
        approval = self.hitl.decide(COMMANDER, "plan-4471", approved=False, note="hold")
        with self.assertRaises(ApprovalRequired):
            self.hitl.check("plan-4471", Severity.P0, approval)

    def test_an_approval_for_a_different_plan_does_not_transfer(self) -> None:
        """Approval is for an exact plan, not for a mood. Re-planning after a
        signature invalidates the signature."""
        approval = self.hitl.decide(COMMANDER, "plan-4471", approved=True)
        with self.assertRaises(ApprovalRequired):
            self.hitl.check("plan-9999", Severity.P0, approval)

    def test_p2_and_below_actuate_without_a_human(self) -> None:
        self.hitl.check("plan-4471", Severity.P2, None)  # must not raise
        self.hitl.check("plan-4471", Severity.P3, None)


class TestExecutor(GovernanceCase):
    def setUp(self) -> None:
        super().setUp()
        self.catalogue = MessageCatalogue(
            known_gates=frozenset(n.id for n in VENUE.nodes_tagged("gate")),
            known_edges=frozenset(VENUE.edges),
            known_zones=frozenset(n.zone for n in VENUE.nodes.values()),
        )
        self.executor = Executor(self.plane, self.catalogue, self.audit)

    def test_actions_compose_a_diversion_sees_lanes_opened_before_it(self) -> None:
        """The bug this test exists to prevent: evaluating a plan's second half
        against a venue state its first half already changed."""
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        plan["actions"] = [
            {
                "type": "OPEN_LANES",
                "sop_ref": "SOP-QUEUE-02#2",
                "params": {"gate_id": "G3", "lanes": 12},
            },
            {
                "type": "DIVERT_ARRIVALS",
                "sop_ref": "SOP-QUEUE-02#2",
                "params": {"from_gate": "G3", "to_gate": "G2", "share": 0.2},
            },
        ]
        execution = self.executor.execute(
            plan, self.assessment, self.snapshot, languages=["en"]
        )

        # G3 ends up with 12 lanes AND 20% fewer arrivals. If the divert had been
        # computed against the original 10-lane state, this would be wrong.
        self.assertEqual(execution.gate_state["G3"].open_lanes, 12)
        self.assertAlmostEqual(
            execution.gate_state["G3"].arrival_rate_per_s, 5.4 * 0.8, places=6
        )
        self.assertLess(execution.gate_after["G3"].utilisation, 0.90)
        self.assertLess(execution.gate_after["G2"].utilisation, 0.85)
        self.assertFalse(execution.warnings)

    def test_a_diversion_that_overloads_the_target_is_reported(self) -> None:
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        plan["actions"] = [{
            "type": "DIVERT_ARRIVALS",
            "sop_ref": "SOP-QUEUE-02#2",
            "params": {"from_gate": "G3", "to_gate": "G4", "share": 0.5},
        }]
        execution = self.executor.execute(
            plan, self.assessment, self.snapshot, languages=["en"]
        )
        self.assertTrue(any("0.85 ceiling" in w for w in execution.warnings))

    def test_a_responder_route_is_recomputed_against_the_live_cordon(self) -> None:
        """The plan named a medical post. Everything else -- the path, the ETA, the
        guarantee it does not cross the cordon -- is computed here, at actuation,
        from the graph."""
        execution = self.executor.execute(
            PLAN_TRANSCRIPT, self.assessment, self.snapshot, languages=["en", "hi", "mr"]
        )
        route = execution.routes["medic:C-N3"]
        self.assertIn("CORR-NE", execution.cordoned)
        self.assertNotIn("CORR-NE", route.edges)
        self.assertTrue(all(VENUE.edge(e).staff_only for e in route.edges))

    def test_an_unreachable_casualty_escalates_rather_than_failing_silently(self) -> None:
        plan = copy.deepcopy(PLAN_TRANSCRIPT)
        # Cordon the service ring as well as the corridor: now nothing reaches C-N3.
        plan["actions"] = [
            {"type": "CORDON_EDGE", "sop_ref": "SOP-MED-03#3",
             "params": {"edge_id": "CORR-NE", "reason": "LOS F"}},
            {"type": "CORDON_EDGE", "sop_ref": "SOP-MED-03#3",
             "params": {"edge_id": "SVC-2", "reason": "structural"}},
            {"type": "CORDON_EDGE", "sop_ref": "SOP-MED-03#3",
             "params": {"edge_id": "CORR-N2", "reason": "structural"}},
            {"type": "DISPATCH_RESPONDER", "sop_ref": "SOP-MED-03#2",
             "params": {"from_node": "MED-2", "to_node": "C-N3", "responder_type": "medic"}},
        ]
        execution = self.executor.execute(
            plan, self.assessment, self.snapshot, languages=["en"]
        )
        self.assertNotIn("medic:C-N3", execution.routes)
        self.assertTrue(any("manual dispatch" in e for e in execution.escalations))
        self.assertTrue(execution.warnings)


class TestOrchestratorWiring(GovernanceCase):
    def test_actuation_is_blocked_end_to_end_without_approval(self) -> None:
        model = TranscriptModel({("quasar.plan_proposal.v1", "plan-4471"): j(PLAN_TRANSCRIPT)})
        orch = Orchestrator(self.plane, model, audit=self.audit)
        orch.submit_for_approval(PLAN_TRANSCRIPT)

        with self.assertRaises(ApprovalRequired):
            orch.actuate(
                PLAN_TRANSCRIPT, self.assessment, self.snapshot,
                languages=["en"], approval=None,
            )
        # Nothing was executed.
        self.assertFalse(self.audit.events("action.executed"))

    def test_policy_is_checked_before_the_human_is_asked(self) -> None:
        """Do not put an illegal plan in front of an operator at 2 a.m. and rely on
        them to catch it."""
        illegal = copy.deepcopy(PLAN_TRANSCRIPT)
        illegal["severity"] = "P1"
        illegal["actions"][0]["params"] = {
            "template_id": "MSG-EVAC-GATE", "zone": "NORTH", "slots": {"gate": "G4"},
        }
        model = TranscriptModel({("quasar.plan_proposal.v1", "x"): "{}"})
        orch = Orchestrator(self.plane, model, audit=self.audit)
        orch.submit_for_approval(illegal)
        approval = orch.hitl.decide(COMMANDER, illegal["plan_id"], approved=True)

        with self.assertRaises(PolicyError):
            orch.actuate(
                illegal, self.assessment, self.snapshot,
                languages=["en"], approval=approval,
            )


if __name__ == "__main__":
    unittest.main()
