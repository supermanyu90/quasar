"""Published JSON Schemas for every agent payload, and the validator that enforces them.

This module is the contract between the probabilistic plane and everything else.
A model output that does not validate here does not exist as far as the rest of
the system is concerned: it never reaches the router, the queueing model, the PA
system, or an operator's screen. There is exactly one door, and it is this one.

The validator implements the Draft-07 subset the schemas actually use (types,
``enum``, ``const``, ``properties``, ``required``, ``additionalProperties`` as
boolean *or* schema, ``items``, ``anyOf``, and the numeric/string/array bounds).
It is deliberately hand-rolled and dependency-free: a venue's safety-critical
validation path should not be able to break because a transitive dependency of a
schema library changed its coercion rules. Anything the schemas do not use is
rejected loudly by :func:`_assert_supported` rather than silently ignored --
a validator that quietly skips a keyword it does not understand is worse than no
validator at all, because it produces false confidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

_SUPPORTED_KEYWORDS = frozenset(
    {
        "$id",
        "title",
        "description",
        "type",
        "enum",
        "const",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "anyOf",
    }
)


@dataclass(frozen=True, slots=True)
class Violation:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '<root>'}: {self.message}"


class SchemaError(Exception):
    """A payload failed validation. Carries every violation, not just the first --
    the repair prompt needs the full list to have any chance of fixing it in one
    round trip."""

    def __init__(self, schema_id: str, violations: Sequence[Violation]) -> None:
        super().__init__(
            f"payload failed schema {schema_id}: "
            + "; ".join(str(v) for v in violations)
        )
        self.schema_id = schema_id
        self.violations = tuple(violations)

    def repair_hint(self) -> str:
        """Rendered back into the model on the single repair attempt."""
        return "\n".join(f"- {v}" for v in self.violations)


def _assert_supported(schema: Mapping[str, Any], path: str = "") -> None:
    for key in schema:
        if key not in _SUPPORTED_KEYWORDS:
            raise ValueError(
                f"schema at {path or '<root>'} uses unsupported keyword {key!r}; "
                "the validator refuses to silently ignore it"
            )
    for name, sub in (schema.get("properties") or {}).items():
        _assert_supported(sub, f"{path}.{name}")
    items = schema.get("items")
    if isinstance(items, dict):
        _assert_supported(items, f"{path}[]")
    extra = schema.get("additionalProperties")
    if isinstance(extra, dict):
        _assert_supported(extra, f"{path}.*")
    for i, sub in enumerate(schema.get("anyOf") or ()):
        _assert_supported(sub, f"{path}|anyOf[{i}]")


_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    # bool is a subclass of int in Python; a JSON boolean is not a JSON number.
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float))
    and not isinstance(v, bool)
    and math.isfinite(v),
    "null": lambda v: v is None,
}


def _validate(value: Any, schema: Mapping[str, Any], path: str) -> Iterator[Violation]:
    expected = schema.get("type")
    if expected is not None:
        types = (expected,) if isinstance(expected, str) else tuple(expected)
        if not any(_TYPE_CHECKS[t](value) for t in types):
            yield Violation(path, f"expected type {'|'.join(types)}, got {type(value).__name__}")
            return  # every other keyword assumes the type held

    if "const" in schema and value != schema["const"]:
        yield Violation(path, f"must equal {schema['const']!r}")

    if "enum" in schema and value not in schema["enum"]:
        yield Violation(path, f"must be one of {schema['enum']!r}, got {value!r}")

    if "anyOf" in schema:
        branches = schema["anyOf"]
        errors: list[list[Violation]] = []
        for sub in branches:
            found = list(_validate(value, sub, path))
            if not found:
                break
            errors.append(found)
        else:
            best = min(errors, key=len)
            yield Violation(
                path,
                "matched no permitted variant; closest failure was "
                + "; ".join(v.message for v in best),
            )

    if isinstance(value, str):
        if (lo := schema.get("minLength")) is not None and len(value) < lo:
            yield Violation(path, f"shorter than minLength {lo}")
        if (hi := schema.get("maxLength")) is not None and len(value) > hi:
            yield Violation(path, f"longer than maxLength {hi}")
        if (pattern := schema.get("pattern")) is not None and not re.search(pattern, value):
            yield Violation(path, f"does not match pattern {pattern!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if (lo := schema.get("minimum")) is not None and value < lo:
            yield Violation(path, f"below minimum {lo}")
        if (hi := schema.get("maximum")) is not None and value > hi:
            yield Violation(path, f"above maximum {hi}")
        if (lo := schema.get("exclusiveMinimum")) is not None and value <= lo:
            yield Violation(path, f"not above exclusiveMinimum {lo}")
        if (hi := schema.get("exclusiveMaximum")) is not None and value >= hi:
            yield Violation(path, f"not below exclusiveMaximum {hi}")

    if isinstance(value, list):
        if (lo := schema.get("minItems")) is not None and len(value) < lo:
            yield Violation(path, f"fewer than minItems {lo}")
        if (hi := schema.get("maxItems")) is not None and len(value) > hi:
            yield Violation(path, f"more than maxItems {hi}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for i, item in enumerate(value):
                yield from _validate(item, item_schema, f"{path}[{i}]")

    if isinstance(value, dict):
        properties: Mapping[str, Any] = schema.get("properties") or {}
        for name in schema.get("required") or ():
            if name not in value:
                yield Violation(path, f"missing required property {name!r}")
        for name, item in value.items():
            child = f"{path}.{name}" if path else name
            if name in properties:
                yield from _validate(item, properties[name], child)
                continue
            extra = schema.get("additionalProperties", True)
            if extra is False:
                yield Violation(child, "property is not permitted by the schema")
            elif isinstance(extra, dict):
                yield from _validate(item, extra, child)


def validate(payload: Any, schema_id: str) -> Mapping[str, Any]:
    """Validate ``payload`` against the registered schema. Raises SchemaError.

    Returns the payload so call sites can write ``msg = validate(raw, ID)``.
    """
    schema = SCHEMAS.get(schema_id)
    if schema is None:
        raise KeyError(f"no schema registered under {schema_id!r}")
    violations = list(_validate(payload, schema, ""))
    if violations:
        raise SchemaError(schema_id, violations)
    return payload


# ==========================================================================
# The published schemas.
# ==========================================================================

_CONFIDENCE = {
    "type": "number",
    "minimum": 0.0,
    "maximum": 1.0,
    "description": (
        "The agent's self-reported confidence. Treated as a weak signal only: "
        "it is a generated number, not a calibrated probability. The governance "
        "layer gates on min(self_reported, corroboration_score), where the "
        "corroboration score is computed deterministically. See quasar.governance."
    ),
}
_CORRELATION_ID = {"type": "string", "pattern": r"^[A-Za-z0-9._:-]{6,64}$"}
_ID = {"type": "string", "pattern": r"^[A-Za-z0-9._:-]{1,64}$"}
_SEVERITY = {"type": "string", "enum": ["P0", "P1", "P2", "P3"]}
_LOS = {"type": "string", "enum": ["A", "B", "C", "D", "E", "F"]}

# A SOP reference is DOC#SECTION and nothing else. Constraining the *shape* here
# means a fabricated citation has to at least look like a citation; the grounding
# check in quasar.rag then verifies that it names a section that exists and was
# actually retrieved. Shape is cheap; existence is the real gate.
_SOP_REF = {"type": "string", "pattern": r"^[A-Z]{2,6}-[A-Z]{2,8}-\d{2}#\d{1,3}$"}

_CITATION = {
    "type": "object",
    "additionalProperties": False,
    "required": ["doc_id", "section"],
    "properties": {
        "doc_id": _ID,
        "section": {"type": "string", "minLength": 1, "maxLength": 32},
    },
}

VENUE_SPEC = "quasar.venue_spec.v1"
CROWD_ASSESSMENT = "quasar.crowd_assessment.v1"
INCIDENT_BRIEF = "quasar.incident_brief.v1"
PLAN_PROPOSAL = "quasar.plan_proposal.v1"
CONCIERGE_REPLY = "quasar.concierge_reply.v1"
COMMS_DISPATCH = "quasar.comms_dispatch.v1"
VOLUNTEER_BRIEF = "quasar.volunteer_brief.v1"
SCENARIO = "quasar.scenario.v1"

# Typed action variants. Each is closed (additionalProperties: false), so an
# agent cannot smuggle an unmodelled parameter past the barrier -- for example a
# free-text `message` on a BROADCAST, which is exactly the hole through which an
# unvalidated safety announcement would reach the PA system.
_ACTION_VARIANTS = [
    {
        "title": "DISPATCH_RESPONDER",
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "params", "sop_ref"],
        "properties": {
            "type": {"const": "DISPATCH_RESPONDER"},
            "sop_ref": _SOP_REF,
            "params": {
                "type": "object",
                "additionalProperties": False,
                "required": ["from_node", "to_node", "responder_type"],
                "properties": {
                    "from_node": _ID,
                    "to_node": _ID,
                    "responder_type": {"type": "string", "enum": ["medic", "security"]},
                },
            },
        },
    },
    {
        "title": "CORDON_EDGE",
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "params", "sop_ref"],
        "properties": {
            "type": {"const": "CORDON_EDGE"},
            "sop_ref": _SOP_REF,
            "params": {
                "type": "object",
                "additionalProperties": False,
                "required": ["edge_id", "reason"],
                "properties": {
                    "edge_id": _ID,
                    "reason": {"type": "string", "minLength": 3, "maxLength": 200},
                },
            },
        },
    },
    {
        "title": "OPEN_LANES",
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "params", "sop_ref"],
        "properties": {
            "type": {"const": "OPEN_LANES"},
            "sop_ref": _SOP_REF,
            "params": {
                "type": "object",
                "additionalProperties": False,
                "required": ["gate_id", "lanes"],
                "properties": {
                    "gate_id": _ID,
                    # Upper bound is a blast-radius control, not a data-entry
                    # convenience: no single agent action may open more than 16
                    # lanes anywhere in the venue.
                    "lanes": {"type": "integer", "minimum": 1, "maximum": 16},
                },
            },
        },
    },
    {
        "title": "DIVERT_ARRIVALS",
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "params", "sop_ref"],
        "properties": {
            "type": {"const": "DIVERT_ARRIVALS"},
            "sop_ref": _SOP_REF,
            "params": {
                "type": "object",
                "additionalProperties": False,
                "required": ["from_gate", "to_gate", "share"],
                "properties": {
                    "from_gate": _ID,
                    "to_gate": _ID,
                    "share": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 1.0},
                },
            },
        },
    },
    {
        "title": "REROUTE_FLOW",
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "params", "sop_ref"],
        "properties": {
            "type": {"const": "REROUTE_FLOW"},
            "sop_ref": _SOP_REF,
            "params": {
                "type": "object",
                "additionalProperties": False,
                "required": ["avoid_edge", "zone"],
                "properties": {"avoid_edge": _ID, "zone": _ID},
            },
        },
    },
    {
        "title": "BROADCAST",
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "params", "sop_ref"],
        "properties": {
            "type": {"const": "BROADCAST"},
            "sop_ref": _SOP_REF,
            "params": {
                "type": "object",
                "additionalProperties": False,
                # Note what is absent: there is no free-text field. A broadcast
                # names a catalogue template and supplies typed slots. The model
                # chooses *which* approved sentence to say, never what it says.
                "required": ["template_id", "zone", "slots"],
                "properties": {
                    "template_id": _ID,
                    "zone": _ID,
                    "slots": {
                        "type": "object",
                        "additionalProperties": {
                            "anyOf": [
                                {"type": "string", "maxLength": 40},
                                {"type": "integer"},
                            ]
                        },
                    },
                },
            },
        },
    },
    {
        "title": "ESCALATE",
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "params", "sop_ref"],
        "properties": {
            "type": {"const": "ESCALATE"},
            "sop_ref": _SOP_REF,
            "params": {
                "type": "object",
                "additionalProperties": False,
                "required": ["to", "reason"],
                "properties": {
                    "to": {
                        "type": "string",
                        "enum": ["commander", "emergency_services", "venue_operator"],
                    },
                    "reason": {"type": "string", "minLength": 3, "maxLength": 200},
                },
            },
        },
    },
]

_LANG = {"type": "string", "pattern": r"^[a-z]{2}$"}

SCHEMAS: Mapping[str, Mapping[str, Any]] = {
    # A venue config is untrusted input, like any model payload. A typo in a
    # corridor width is a wrong evacuation time; a typo in a node id is a route to
    # a place that does not exist. So it goes through the same door.
    VENUE_SPEC: {
        "$id": VENUE_SPEC,
        "title": "Venue specification",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema", "id", "name", "city", "capacity", "languages",
            "service_rate_per_s", "zones", "nodes", "edges", "beacons", "scenario",
        ],
        "properties": {
            "schema": {"const": VENUE_SPEC},
            "id": {"type": "string", "pattern": r"^[a-z0-9-]{3,40}$"},
            "name": {"type": "string", "minLength": 3, "maxLength": 80},
            # The name used at the tournament (FIFA does not allow sponsor names),
            # kept alongside the real one so a fan sees "New York New Jersey Stadium"
            # while operations still recognises MetLife.
            "fifa_name": {"type": "string", "minLength": 3, "maxLength": 80},
            "city": {"type": "string", "minLength": 2, "maxLength": 60},
            "country": {"type": "string", "minLength": 2, "maxLength": 40},
            "capacity": {"type": "integer", "minimum": 100, "maximum": 200000},
            # Where the graph came from. "surveyed" is a real floor-plan survey with
            # measured corridor widths and verified step-free routes. "representative"
            # is a parametric model fitted to the venue's public capacity and gate
            # count -- correct for scale and planning, NOT a substitute for a survey,
            # and the readiness audit and UI say so. Defaulting to representative when
            # absent is the conservative, honest assumption.
            "topology": {"type": "string", "enum": ["surveyed", "representative"]},
            # The languages this venue's crowd actually speaks -- not the languages
            # the software happens to support. The gap between those two is the
            # readiness finding.
            "languages": {"type": "array", "minItems": 1, "maxItems": 16, "items": _LANG},
            "service_rate_per_s": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 5.0},
            "zones": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": {"type": "string", "minLength": 1, "maxLength": 60},
                },
            },
            "nodes": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2000,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "name", "x", "y", "level", "zone", "tags"],
                    "properties": {
                        "id": _ID,
                        "name": {"type": "string", "minLength": 1, "maxLength": 80},
                        "x": {"type": "number", "minimum": -5000, "maximum": 5000},
                        "y": {"type": "number", "minimum": -5000, "maximum": 5000},
                        "level": {"type": "integer", "minimum": -5, "maximum": 20},
                        "zone": _ID,
                        "tags": {"type": "array", "maxItems": 12, "items": {"type": "string"}},
                    },
                },
            },
            "edges": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4000,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "u", "v", "length_m", "width_m", "kind", "step_free"],
                    "properties": {
                        "id": _ID,
                        "u": _ID,
                        "v": _ID,
                        "length_m": {"type": "number", "exclusiveMinimum": 0.0, "maximum": 2000.0},
                        # A corridor narrower than 1 m cannot carry a crowd, and one
                        # wider than 60 m is a mis-entered figure, not a corridor.
                        "width_m": {"type": "number", "minimum": 0.8, "maximum": 60.0},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "concourse", "corridor", "stair", "ramp",
                                "tunnel", "vomitory", "service",
                            ],
                        },
                        "step_free": {"type": "boolean"},
                        "staff_only": {"type": "boolean"},
                        "label": {"type": "string", "minLength": 1, "maxLength": 40},
                    },
                },
            },
            "beacons": {"type": "array", "maxItems": 500, "items": _ID},
            "scenario": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "name", "casualty_node", "category", "background_density",
                    "edge_density", "gates", "report", "fan",
                ],
                "properties": {
                    "name": {"type": "string", "minLength": 3, "maxLength": 120},
                    "casualty_node": _ID,
                    "category": {
                        "type": "string",
                        "enum": [
                            "medical", "crush", "fire", "security",
                            "weather", "infrastructure", "other",
                        ],
                    },
                    "background_density": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                    "edge_density": {
                        "type": "object",
                        "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 6.0},
                    },
                    "gates": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 24,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "gate_id", "arrival_rate_per_s", "open_lanes", "installed_lanes",
                            ],
                            "properties": {
                                "gate_id": _ID,
                                "arrival_rate_per_s": {"type": "number", "minimum": 0.0, "maximum": 60.0},
                                "open_lanes": {"type": "integer", "minimum": 1, "maximum": 40},
                                "installed_lanes": {"type": "integer", "minimum": 1, "maximum": 40},
                            },
                        },
                    },
                    "report": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "reporter_role", "language", "text"],
                        "properties": {
                            "id": _ID,
                            "reporter_role": {
                                "type": "string",
                                "enum": ["fan", "volunteer", "steward", "medic", "security", "commander"],
                            },
                            "language": _LANG,
                            "text": {"type": "string", "minLength": 5, "maxLength": 600},
                        },
                    },
                    "fan": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["at_node", "seat", "language", "accessible", "utterance"],
                        "properties": {
                            "at_node": _ID,
                            "seat": _ID,
                            "language": _LANG,
                            "accessible": {"type": "boolean"},
                            "utterance": {"type": "string", "minLength": 3, "maxLength": 400},
                        },
                    },
                },
            },
        },
    },
    CROWD_ASSESSMENT: {
        "$id": CROWD_ASSESSMENT,
        "title": "Crowd assessment",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "correlation_id", "summary", "hotspots", "gate_pressure", "confidence"],
        "properties": {
            "schema": {"const": CROWD_ASSESSMENT},
            "correlation_id": _CORRELATION_ID,
            "summary": {"type": "string", "minLength": 10, "maxLength": 600},
            "hotspots": {
                "type": "array",
                "maxItems": 24,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["edge_id", "density_ped_m2", "los", "trend"],
                    "properties": {
                        "edge_id": _ID,
                        "density_ped_m2": {"type": "number", "minimum": 0.0, "maximum": 10.0},
                        "los": _LOS,
                        "trend": {"type": "string", "enum": ["rising", "steady", "falling"]},
                    },
                },
            },
            "gate_pressure": {
                "type": "array",
                "maxItems": 24,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["gate_id", "utilisation", "action_required"],
                    "properties": {
                        "gate_id": _ID,
                        "utilisation": {"type": "number", "minimum": 0.0, "maximum": 10.0},
                        "action_required": {"type": "boolean"},
                    },
                },
            },
            "confidence": _CONFIDENCE,
        },
    },
    INCIDENT_BRIEF: {
        "$id": INCIDENT_BRIEF,
        "title": "Generative incident brief",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema", "correlation_id", "incident_id", "severity", "category",
            "affected_zones", "situation", "recommended_actions", "citations", "confidence",
        ],
        "properties": {
            "schema": {"const": INCIDENT_BRIEF},
            "correlation_id": _CORRELATION_ID,
            "incident_id": _ID,
            "severity": _SEVERITY,
            "category": {
                "type": "string",
                "enum": ["medical", "crush", "fire", "security", "weather", "infrastructure", "other"],
            },
            "affected_zones": {"type": "array", "minItems": 1, "maxItems": 12, "items": _ID},
            "situation": {"type": "string", "minLength": 20, "maxLength": 1200},
            "recommended_actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action", "sop_ref"],
                    "properties": {
                        "action": {"type": "string", "minLength": 5, "maxLength": 240},
                        "sop_ref": _SOP_REF,
                    },
                },
            },
            # An incident brief with no citation is a hallucination with a
            # severity label attached. minItems: 1 makes that unrepresentable.
            "citations": {"type": "array", "minItems": 1, "maxItems": 12, "items": _CITATION},
            "confidence": _CONFIDENCE,
        },
    },
    PLAN_PROPOSAL: {
        "$id": PLAN_PROPOSAL,
        "title": "Plan proposal",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "correlation_id", "plan_id", "severity", "actions", "rationale", "confidence"],
        "properties": {
            "schema": {"const": PLAN_PROPOSAL},
            "correlation_id": _CORRELATION_ID,
            "plan_id": _ID,
            "severity": _SEVERITY,
            "actions": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"anyOf": _ACTION_VARIANTS}},
            "rationale": {"type": "string", "minLength": 20, "maxLength": 1200},
            "confidence": _CONFIDENCE,
        },
    },
    CONCIERGE_REPLY: {
        "$id": CONCIERGE_REPLY,
        "title": "Multilingual concierge reply",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema", "correlation_id", "language", "intent", "destination_tag",
            "reply_text", "requires_route", "safety_tier", "confidence",
        ],
        "properties": {
            "schema": {"const": CONCIERGE_REPLY},
            "correlation_id": _CORRELATION_ID,
            "language": {"type": "string", "pattern": r"^[a-z]{2}$"},
            "intent": {
                "type": "string",
                "enum": [
                    "wayfinding", "seat", "food", "washroom", "medical",
                    "lost_and_found", "match_info", "emergency", "other",
                ],
            },
            "destination_tag": {
                "anyOf": [
                    {"type": "string", "enum": [
                        "washroom", "accessible", "fnb", "medical",
                        "lost_and_found", "seating", "gate", "control",
                    ]},
                    {"type": "null"},
                ]
            },
            "reply_text": {"type": "string", "minLength": 1, "maxLength": 800},
            "requires_route": {"type": "boolean"},
            # The concierge must classify its own turn. If it says
            # "safety_critical", governance takes the pen away and renders from
            # the controlled catalogue instead of shipping reply_text.
            "safety_tier": {"type": "string", "enum": ["informational", "safety_critical"]},
            "confidence": _CONFIDENCE,
        },
    },
    COMMS_DISPATCH: {
        "$id": COMMS_DISPATCH,
        "title": "Communication dispatch",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "correlation_id", "tier", "template_id", "slots", "languages", "zones", "confidence"],
        "properties": {
            "schema": {"const": COMMS_DISPATCH},
            "correlation_id": _CORRELATION_ID,
            "tier": {"type": "string", "enum": ["safety_critical", "informational"]},
            "template_id": _ID,
            "slots": {
                "type": "object",
                "additionalProperties": {
                    "anyOf": [{"type": "string", "maxLength": 40}, {"type": "integer"}]
                },
            },
            "languages": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": {"type": "string", "pattern": r"^[a-z]{2}$"},
            },
            "zones": {"type": "array", "minItems": 1, "maxItems": 12, "items": _ID},
            "confidence": _CONFIDENCE,
        },
    },
    VOLUNTEER_BRIEF: {
        "$id": VOLUNTEER_BRIEF,
        "title": "Volunteer shift briefing",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema", "correlation_id", "volunteer_id", "language", "role",
            "zone", "sections", "risks", "citations", "confidence",
        ],
        "properties": {
            "schema": {"const": VOLUNTEER_BRIEF},
            "correlation_id": _CORRELATION_ID,
            "volunteer_id": _ID,
            "language": {"type": "string", "pattern": r"^[a-z]{2}$"},
            "role": {"type": "string", "minLength": 3, "maxLength": 40},
            "zone": _ID,
            "sections": {
                "type": "array",
                "minItems": 1,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["heading", "body"],
                    "properties": {
                        "heading": {"type": "string", "minLength": 3, "maxLength": 80},
                        "body": {"type": "string", "minLength": 10, "maxLength": 900},
                    },
                },
            },
            "risks": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 200}},
            "citations": {"type": "array", "minItems": 1, "maxItems": 12, "items": _CITATION},
            "confidence": _CONFIDENCE,
        },
    },
    SCENARIO: {
        "$id": SCENARIO,
        "title": "Synthetic stress scenario",
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "scenario_id", "name", "kind", "edge_density", "closed_edges", "gate_overrides"],
        "properties": {
            "schema": {"const": SCENARIO},
            "scenario_id": _ID,
            "name": {"type": "string", "minLength": 3, "maxLength": 120},
            "kind": {
                "type": "string",
                "enum": [
                    "gate_failure", "weather_evacuation", "vip_movement",
                    "medical_surge", "power_loss", "pitch_invasion",
                ],
            },
            "edge_density": {
                "type": "object",
                "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 6.0},
            },
            "closed_edges": {"type": "array", "maxItems": 12, "items": _ID},
            "gate_overrides": {
                "type": "array",
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["gate_id", "arrival_rate_per_s", "open_lanes"],
                    "properties": {
                        "gate_id": _ID,
                        "arrival_rate_per_s": {"type": "number", "minimum": 0.0, "maximum": 60.0},
                        "open_lanes": {"type": "integer", "minimum": 0, "maximum": 24},
                    },
                },
            },
        },
    },
}

# Fail at import time, not in production, if a schema uses a keyword the
# validator does not implement.
for _sid, _schema in SCHEMAS.items():
    _assert_supported(_schema, _sid)
del _sid, _schema
