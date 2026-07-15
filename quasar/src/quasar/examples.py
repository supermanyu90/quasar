"""One valid example instance per published schema.

A JSON Schema tells a model what is *permitted*. It does not reliably tell a small
model what to *emit* -- a 4B parameter model given a schema will cheerfully hand
back the schema, `$id` and `properties` and all. That is not a hypothetical: it is
what the on-venue edge model actually did, and the barrier caught it.

So every schema ships with a worked example, and the prompt carries both.

Two properties keep this honest.

**The examples cannot drift.** Every one is validated against its own schema at
import time. A contract change that invalidates an example fails the process on
startup, not on match night.

**The examples cannot leak the answer.** Each describes a *different* situation
from the one the agent is being asked about -- a different corridor, a different
gate, different numbers. A model that copies the example instead of reading the
telemetry produces a payload whose restated densities do not match the sensors,
and the corroborator kills it. The example teaches the *shape*, and the shape is
the only thing it is allowed to teach.
"""

from __future__ import annotations

from typing import Any, Mapping

from quasar import schemas

_CORRELATION = "cyc-EXAMPLE00001"

EXAMPLES: Mapping[str, Mapping[str, Any]] = {
    schemas.CROWD_ASSESSMENT: {
        "schema": schemas.CROWD_ASSESSMENT,
        "correlation_id": _CORRELATION,
        "summary": (
            "The south-west corridor is filling because Gate 6 is admitting faster than "
            "the concourse clears. Watch CORR-SW: it is the constriction, and Gate 6 is "
            "the cause."
        ),
        "hotspots": [
            {"edge_id": "CORR-SW", "density_ped_m2": 2.9, "los": "F", "trend": "rising"},
        ],
        "gate_pressure": [
            {"gate_id": "G6", "utilisation": 0.93, "action_required": True},
            {"gate_id": "G1", "utilisation": 0.41, "action_required": False},
        ],
        "confidence": 0.88,
    },
    schemas.INCIDENT_BRIEF: {
        "schema": schemas.INCIDENT_BRIEF,
        "correlation_id": _CORRELATION,
        "incident_id": "INC-0000",
        "severity": "P1",
        "category": "crush",
        "affected_zones": ["SOUTH"],
        "situation": (
            "Spectators are being pressed against the barrier at the south concourse "
            "while the south-west corridor is at 2.9 ped/m2. No casualties are reported "
            "yet. The pressure is building rather than dispersing."
        ),
        "recommended_actions": [
            {"action": "Hold arrivals at Gate 6 and divert to Gate 1.", "sop_ref": "SOP-QUEUE-02#2"},
        ],
        "citations": [{"doc_id": "SOP-QUEUE-02", "section": "2"}],
        "confidence": 0.86,
    },
    schemas.PLAN_PROPOSAL: {
        "schema": schemas.PLAN_PROPOSAL,
        "correlation_id": _CORRELATION,
        "plan_id": "plan-0000",
        "severity": "P1",
        "actions": [
            {
                "type": "BROADCAST",
                "sop_ref": "SOP-QUEUE-02#3",
                "params": {
                    "template_id": "MSG-GATE-DIVERT",
                    "zone": "SOUTH",
                    "slots": {"from_gate": "G6", "to_gate": "G1"},
                },
            },
            {
                "type": "OPEN_LANES",
                "sop_ref": "SOP-QUEUE-02#2",
                "params": {"gate_id": "G1", "lanes": 10},
            },
            {
                "type": "DISPATCH_RESPONDER",
                "sop_ref": "SOP-MED-03#2",
                "params": {"from_node": "MED-3", "to_node": "C-S1", "responder_type": "security"},
            },
        ],
        "rationale": (
            "Announce the diversion before anything is closed, so arriving spectators are "
            "redirected rather than walked into a wall of people. Then add lanes at the "
            "gate receiving them."
        ),
        "confidence": 0.87,
    },
    schemas.CONCIERGE_REPLY: {
        "schema": schemas.CONCIERGE_REPLY,
        "correlation_id": _CORRELATION,
        "language": "hi",
        "intent": "washroom",
        "destination_tag": "washroom",
        "reply_text": "निकटतम शौचालय यहाँ से थोड़ी दूर है। मैं आपको रास्ता दिखाता हूँ।",
        "requires_route": True,
        "safety_tier": "informational",
        "confidence": 0.9,
    },
    schemas.COMMS_DISPATCH: {
        "schema": schemas.COMMS_DISPATCH,
        "correlation_id": _CORRELATION,
        "tier": "safety_critical",
        "template_id": "MSG-GATE-DIVERT",
        "slots": {"from_gate": "G6", "to_gate": "G1"},
        "languages": ["en", "hi", "mr"],
        "zones": ["SOUTH"],
        "confidence": 0.92,
    },
    schemas.VOLUNTEER_BRIEF: {
        "schema": schemas.VOLUNTEER_BRIEF,
        "correlation_id": _CORRELATION,
        "volunteer_id": "VOL-0000",
        "language": "en",
        "role": "gate steward",
        "zone": "SOUTH",
        "sections": [
            {
                "heading": "Your post tonight",
                "body": (
                    "You are on Gate 6. It is the busiest gate in the second half of the "
                    "arrival window and it feeds the narrow south-west corridor."
                ),
            },
        ],
        "risks": ["Gate 6 saturation in the last 20 minutes before kick-off"],
        "citations": [{"doc_id": "SOP-QUEUE-02", "section": "1"}],
        "confidence": 0.89,
    },
    schemas.SCENARIO: {
        "schema": schemas.SCENARIO,
        "scenario_id": "GEN-EXAMPLE",
        "name": "turnstile failure at the west gate during a weather hold",
        "kind": "gate_failure",
        "edge_density": {"CORR-W": 3.1, "CORR-SW": 2.6},
        "closed_edges": ["E-G6"],
        "gate_overrides": [
            {"gate_id": "G1", "arrival_rate_per_s": 9.5, "open_lanes": 8},
        ],
    },
}

# Fail at import, not on match night, if a contract change orphans an example.
for _schema_id, _example in EXAMPLES.items():
    schemas.validate(_example, _schema_id)
del _schema_id, _example
