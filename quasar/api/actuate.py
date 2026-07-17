"""POST /api/actuate -- policy, then the human barrier, then execution.

The plan arrives from a browser, where anyone can edit it. It is re-validated
against the published schema, re-corroborated against the deterministic plane, and
re-gated on the operator's signature before a single action runs. A hostile client
is just an agent with a worse prompt, and it meets the same barrier.
"""

from __future__ import annotations

from typing import Any

from _shared import Endpoint
from quasar.web import actuate, resolve_model


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    plan = payload.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("a plan is required")

    venue = payload.get("venue") or self.venue_param or ""
    plane = resolve_model(payload.get("mode", "recorded"), self.live_secret, venue)
    execution, audit = actuate(
        plan, payload.get("approver"), venue, plane, audit_chain=payload.get("audit")
    )
    return {"execution": execution, "audit": audit.to_json()}
