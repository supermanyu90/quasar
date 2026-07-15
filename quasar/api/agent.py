"""POST /api/agent -- run one agent through the full governance barrier.

One agent per invocation, so each gets its own timeout budget and the console can
reveal them progressively. Three sequential model calls in a single request would
exceed the function limit on a live run.
"""

from __future__ import annotations

import os
import sys

# Vercel imports this file directly; api/ is not guaranteed to be on sys.path, so
# put it there before reaching for the shared plumbing (which in turn puts src/ on
# the path so the quasar package is importable).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Any

from _shared import Endpoint, endpoint, live_budget_ok
from quasar.web import LiveDenied, resolve_model, run_agent


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    name = payload.get("agent", "crowd")
    mode = payload.get("mode", "recorded")
    venue = payload.get("venue") or self.venue_param or ""

    plane = resolve_model(mode, self.live_secret, venue)
    if plane.mode == "live" and not live_budget_ok():
        raise LiveDenied("live rate limit reached on this instance; try again shortly")

    result, audit = run_agent(
        name, venue, plane, brief=payload.get("brief"), audit_chain=payload.get("audit")
    )
    return {"result": result, "audit": audit.to_json(), "mode": plane.mode, "note": plane.note}


handler = endpoint(run)
