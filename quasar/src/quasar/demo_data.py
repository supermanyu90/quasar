"""The reference match-day scenario, and the recorded model transcripts.

This lives in the package rather than in `tests/` because the web console and the
CLI demo both need it, and neither should import from a test package.

**The transcripts are recordings, not simulations.** They stand in for the model
so the deployed site works with no API key and costs nothing to run. They fake
none of the safety logic, because none of the safety logic lives in the model:
under a recorded transcript the router, the queueing model, the message
catalogue, the schema validator, the grounding check, the corroborators, the
policy engine and the human-in-the-loop barrier all execute for real, and it is
their output the console displays.

The scenario, throughout: a spectator has collapsed at C-N3, the north-east
concourse junction, during the second half. The corridor beside them (CORR-NE,
4 m wide, the only fan-side link between the north and east concourses) is at
3.4 ped/m^2 -- level of service F. Gate 3 is simultaneously at 0.98 utilisation
because the away end is still arriving late. Three of the four tracks collide in
one place, which is the point.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from quasar.types import GateTelemetry, IncidentReport, Operator, Role, TelemetrySnapshot

CASUALTY_NODE = "C-N3"
PINCH = "CORR-NE"

# Languages ticketing reports present in the north zone tonight. Tamil is in the
# list deliberately: the catalogue has a Tamil string for the medical template,
# but it is a machine draft, so the Tier-1 gate refuses it and the console shows
# the pictogram-and-steward fallback instead.
ZONE_LANGUAGES = ["en", "hi", "mr", "ta"]


def match_day_snapshot(*, t: float = 1000.0) -> TelemetrySnapshot:
    density = {
        "CORR-NE": 3.4,  # LOS F: the crush around the casualty
        "CORR-N2": 1.6,  # LOS E: the north concourse feeding it
        "E-G3": 1.2,  # LOS E: the gate hall backing up
    }
    return TelemetrySnapshot(
        t=t,
        edge_density=density,
        gates={
            "G3": GateTelemetry("G3", 5.4, 0.55, 10, 12),  # utilisation 0.98 -- breach
            "G2": GateTelemetry("G2", 2.2, 0.55, 8, 12),
            "G4": GateTelemetry("G4", 1.5, 0.55, 6, 10),
        },
        source_ids=("cv-north", "turnstile-log", "ble-gw-3"),
    )


def quiet_snapshot() -> TelemetrySnapshot:
    return TelemetrySnapshot(
        t=100.0,
        edge_density={},
        gates={"G3": GateTelemetry("G3", 2.0, 0.55, 10, 12)},
    )


def full_density(venue: Any, snapshot: TelemetrySnapshot) -> dict[str, float]:
    """Every edge, defaulting to a quiet 0.3 ped/m^2 where no sensor reports."""
    return {e: snapshot.density(e) or 0.3 for e in venue.edges}


VOLUNTEER_REPORT = IncidentReport(
    id="INC-4471",
    reporter_role=Role.VOLUNTEER,
    at_node=CASUALTY_NODE,
    text=(
        "someone's gone down near the north-east food stand, an older man, he's not "
        "getting up and people are pushing past, it's really tight in here"
    ),
    language="en",
    t=1001.0,
)

COMMANDER = Operator("op-cmd-1", "R. Deshmukh", frozenset({Role.COMMANDER}))
STEWARD = Operator("op-stw-7", "A. Khan", frozenset({Role.STEWARD}))

OPERATORS: Mapping[str, Operator] = {"commander": COMMANDER, "steward": STEWARD}

CORRELATION = "cyc-000000000001"


# ==========================================================================
# Recorded transcripts
# ==========================================================================

CROWD_TRANSCRIPT: Mapping[str, Any] = {
    "schema": "quasar.crowd_assessment.v1",
    "correlation_id": CORRELATION,
    "summary": (
        "Gate 3 is at 0.98 utilisation and is feeding the north concourse faster than "
        "it can clear. That inflow is what is loading CORR-NE, which is now at LOS F "
        "with a medical incident inside it. These are one problem, not two: relieving "
        "Gate 3 is the upstream fix for the corridor."
    ),
    "hotspots": [
        {"edge_id": "CORR-NE", "density_ped_m2": 3.4, "los": "F", "trend": "steady"},
        {"edge_id": "CORR-N2", "density_ped_m2": 1.6, "los": "E", "trend": "steady"},
        {"edge_id": "E-G3", "density_ped_m2": 1.2, "los": "E", "trend": "steady"},
    ],
    "gate_pressure": [
        {"gate_id": "G3", "utilisation": 0.98, "action_required": True},
        {"gate_id": "G2", "utilisation": 0.5, "action_required": False},
        {"gate_id": "G4", "utilisation": 0.45, "action_required": False},
    ],
    "confidence": 0.94,
}

BRIEF_TRANSCRIPT: Mapping[str, Any] = {
    "schema": "quasar.incident_brief.v1",
    "correlation_id": CORRELATION,
    "incident_id": "INC-4471",
    "severity": "P0",
    "category": "medical",
    "affected_zones": ["NORTH"],
    "situation": (
        "An elderly male spectator has collapsed and is unresponsive at C-N3, inside "
        "the north-east concourse. The adjacent corridor CORR-NE is at 3.4 ped/m2 "
        "(LOS F) and spectators are being pushed past the casualty. Crowd pressure at "
        "the scene exceeds LOS E, which upgrades this from P1 to P0 under SOP-MED-03#1. "
        "Gate 3 at 0.98 utilisation is the upstream cause of the corridor loading."
    ),
    "recommended_actions": [
        {
            "action": "Dispatch the east medical team via the inner service ring; the "
                      "direct corridor approach is at LOS F and may not be used.",
            "sop_ref": "SOP-MED-03#2",
        },
        {
            "action": "Announce the diversion in the north zone, then cordon CORR-NE. "
                      "In that order.",
            "sop_ref": "SOP-MED-03#3",
        },
        {
            "action": "Open Gate 3's reserve lanes to relieve the inflow driving the "
                      "corridor density.",
            "sop_ref": "SOP-QUEUE-02#2",
        },
    ],
    "citations": [
        {"doc_id": "SOP-MED-03", "section": "1"},
        {"doc_id": "SOP-MED-03", "section": "2"},
        {"doc_id": "SOP-MED-03", "section": "3"},
    ],
    "confidence": 0.91,
}

PLAN_TRANSCRIPT: Mapping[str, Any] = {
    "schema": "quasar.plan_proposal.v1",
    "correlation_id": CORRELATION,
    "plan_id": "plan-4471",
    "severity": "P0",
    "actions": [
        {
            "type": "BROADCAST",
            "sop_ref": "SOP-MED-03#3",
            "params": {
                "template_id": "MSG-MED-CORRIDOR",
                "zone": "NORTH",
                "slots": {"zone": "NORTH", "corridor": "CORR-NE"},
            },
        },
        {
            "type": "CORDON_EDGE",
            "sop_ref": "SOP-MED-03#3",
            "params": {
                "edge_id": "CORR-NE",
                "reason": "LOS F with a medical incident inside the corridor",
            },
        },
        {
            "type": "DISPATCH_RESPONDER",
            "sop_ref": "SOP-MED-03#2",
            "params": {
                "from_node": "MED-2",
                "to_node": "C-N3",
                "responder_type": "medic",
            },
        },
        {
            "type": "OPEN_LANES",
            "sop_ref": "SOP-QUEUE-02#2",
            "params": {"gate_id": "G3", "lanes": 12},
        },
    ],
    "rationale": (
        "Announce before cordoning (SOP-MED-03#3) so arriving spectators are not walked "
        "into a closed corridor. Dispatch from MED-2 rather than MED-1: the service ring "
        "reaches C-N3 without entering the crush. Open Gate 3's reserve lanes to remove "
        "the inflow that is loading the corridor -- treating the cause, not the symptom."
    ),
    "confidence": 0.92,
}

CONCIERGE_TRANSCRIPT: Mapping[str, Any] = {
    "schema": "quasar.concierge_reply.v1",
    "correlation_id": "cyc-000000000002",
    "language": "mr",
    "intent": "wayfinding",
    "destination_tag": "seating",
    "reply_text": (
        "काळजी करू नका. तुम्ही चुकीच्या गेटवर आला आहात, पण मी तुम्हाला पायऱ्यांशिवाय "
        "मार्ग दाखवतो. तुमच्या जागेपर्यंतचा उतार-मार्ग खाली दिला आहे."
    ),
    "requires_route": True,
    "safety_tier": "informational",
    "confidence": 0.93,
}

VOLUNTEER_BRIEF_TRANSCRIPT: Mapping[str, Any] = {
    "schema": "quasar.volunteer_brief.v1",
    "correlation_id": "cyc-000000000003",
    "volunteer_id": "VOL-218",
    "language": "mr",
    "role": "concourse steward",
    "zone": "NORTH",
    "sections": [
        {
            "heading": "तुमचे क्षेत्र",
            "body": (
                "तुम्ही उत्तर कॉनकोर्सवर, गेट 3 आणि मार्गिका CORR-NE यांच्या दरम्यान "
                "असाल. ही मार्गिका फक्त 4 मीटर रुंद आहे आणि हाफ-टाइमला येथे सर्वाधिक "
                "गर्दी होते."
            ),
        },
        {
            "heading": "आजचे धोके",
            "body": (
                "गेट 3 उशिरा येणाऱ्या प्रेक्षकांमुळे भरून जाईल. गर्दी वाढल्यास "
                "नियंत्रण कक्षाला त्वरित कळवा आणि प्रेक्षकांना गेट 2 कडे वळवा."
            ),
        },
    ],
    "risks": ["CORR-NE congestion at half-time", "Gate 3 late arrivals"],
    "citations": [
        {"doc_id": "SOP-QUEUE-02", "section": "1"},
        {"doc_id": "SOP-MED-03", "section": "3"},
    ],
    "confidence": 0.9,
}


# ==========================================================================
# Coastal Arena (Chennai) -- a different venue, a different incident.
#
# The recordings are per-venue because the incidents are: a plan that names
# CORR-NE is nonsense at an arena whose corridors are called R-E1, and the
# corroborator will say so. This is the venue-customisation story stated in the
# one place it cannot be faked -- the model's own output has to change.
# ==========================================================================

ARENA_CROWD: Mapping[str, Any] = {
    "schema": "quasar.crowd_assessment.v1",
    "correlation_id": "cyc-coastal--0001",
    "summary": (
        "The east link (R-E1) is the whole problem. It is 3.5 m wide and it is the only "
        "spectator path between the north and east concourses, and it is now at 3.9 ped/m2 "
        "with a casualty in it. Gate 2 at 0.94 is feeding it. This is one problem with two "
        "symptoms, and the gate is the end you can actually pull on."
    ),
    "hotspots": [
        {"edge_id": "R-E1", "density_ped_m2": 3.9, "los": "F", "trend": "rising"},
        {"edge_id": "R-N2", "density_ped_m2": 1.9, "los": "E", "trend": "rising"},
        {"edge_id": "E-G2", "density_ped_m2": 1.3, "los": "E", "trend": "steady"},
    ],
    "gate_pressure": [
        {"gate_id": "G1", "utilisation": 0.44, "action_required": False},
        {"gate_id": "G2", "utilisation": 0.94, "action_required": True},
        {"gate_id": "G3", "utilisation": 0.39, "action_required": False},
    ],
    "confidence": 0.93,
}

ARENA_BRIEF: Mapping[str, Any] = {
    "schema": "quasar.incident_brief.v1",
    "correlation_id": "cyc-coastal--0001",
    "incident_id": "INC-0912",
    "severity": "P0",
    "category": "medical",
    "affected_zones": ["EAST"],
    "situation": (
        "A woman has collapsed at C-NE, at the head of the east link. The link is 3.5 m "
        "wide, is at 3.9 ped/m2 (LOS F), and is the only spectator route between the north "
        "and east concourses, so the crowd coming off the north side has nowhere else to go "
        "and is pushing through the casualty. Crowd pressure at the scene exceeds LOS E, "
        "which makes this P0 under SOP-MED-03#1. Gate 2 at 0.94 utilisation is the inflow "
        "driving it."
    ),
    "recommended_actions": [
        {
            "action": "Dispatch the east medical team by the service corridor; the link "
                      "itself is at LOS F and may not be used to approach.",
            "sop_ref": "SOP-MED-03#2",
        },
        {
            "action": "Announce the diversion in the east zone, then close the link. In "
                      "that order -- there is no second route, so anyone already walking "
                      "toward it must be turned before it shuts.",
            "sop_ref": "SOP-MED-03#3",
        },
        {"action": "Open Gate 2's reserve lanes to cut the inflow.", "sop_ref": "SOP-QUEUE-02#2"},
    ],
    "citations": [
        {"doc_id": "SOP-MED-03", "section": "1"},
        {"doc_id": "SOP-MED-03", "section": "2"},
        {"doc_id": "SOP-MED-03", "section": "3"},
    ],
    "confidence": 0.9,
}

ARENA_PLAN: Mapping[str, Any] = {
    "schema": "quasar.plan_proposal.v1",
    "correlation_id": "cyc-coastal--0001",
    "plan_id": "plan-INC-0912",
    "severity": "P0",
    "actions": [
        {
            "type": "BROADCAST",
            "sop_ref": "SOP-MED-03#3",
            "params": {
                "template_id": "MSG-MED-CORRIDOR",
                "zone": "EAST",
                "slots": {"zone": "EAST", "corridor": "R-E1"},
            },
        },
        {
            "type": "CORDON_EDGE",
            "sop_ref": "SOP-MED-03#3",
            "params": {"edge_id": "R-E1", "reason": "LOS F with a casualty in the link"},
        },
        {
            "type": "DISPATCH_RESPONDER",
            "sop_ref": "SOP-MED-03#2",
            "params": {"from_node": "MED-1", "to_node": "C-NE", "responder_type": "medic"},
        },
        {
            "type": "OPEN_LANES",
            "sop_ref": "SOP-QUEUE-02#2",
            "params": {"gate_id": "G2", "lanes": 8},
        },
    ],
    "rationale": (
        "This venue has no ring redundancy at the east link, so cordoning it severs the "
        "concourse -- the announcement must go first and it must name where to go instead. "
        "MED-1 reaches C-NE by the service corridor without entering the crush. Gate 2's "
        "reserve lanes remove the inflow rather than managing its consequences."
    ),
    "confidence": 0.91,
}


def transcripts(venue_id: str = "national-stadium") -> dict[tuple[str, str], str]:
    """Recorded model output for a venue's fixture.

    Keyed by (schema_id, a cue that must appear in the request), so a recording can
    only satisfy the request it was made for. A venue with no recording returns
    nothing, every agent falls back, and the console says so -- which is honest, and
    is why Edge mode exists.
    """
    if venue_id == "coastal-arena":
        return {
            ("quasar.crowd_assessment.v1", "G2"): json.dumps(ARENA_CROWD),
            ("quasar.incident_brief.v1", "INC-0912"): json.dumps(ARENA_BRIEF),
            ("quasar.plan_proposal.v1", "plan-INC-0912"): json.dumps(ARENA_PLAN),
        }
    if venue_id == "national-stadium":
        return {
            ("quasar.crowd_assessment.v1", "G3"): json.dumps(CROWD_TRANSCRIPT),
            ("quasar.incident_brief.v1", "INC-4471"): json.dumps(BRIEF_TRANSCRIPT),
            ("quasar.plan_proposal.v1", "plan-INC-4471"): json.dumps(PLAN_TRANSCRIPT),
            ("quasar.concierge_reply.v1", "fan_language: mr"): json.dumps(CONCIERGE_TRANSCRIPT),
            ("quasar.volunteer_brief.v1", "VOL-218"): json.dumps(VOLUNTEER_BRIEF_TRANSCRIPT),
        }
    return {}
