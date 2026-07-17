"""POST /api/concierge -- the fan-facing assistant.

The agent decides *where* the fan wants to go. The router decides *how*, and is
bound by constraints the agent cannot override: a step-free fan is never routed
down a staircase, and never into a crush an unimpeded adult would be allowed to
walk through.
"""

from __future__ import annotations

from typing import Any

from _shared import Endpoint, live_budget_ok
from quasar.web import LiveDenied, concierge, resolve_model


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    venue = payload.get("venue") or self.venue_param or ""
    plane = resolve_model(payload.get("mode", "recorded"), self.live_secret, venue)
    if plane.mode == "live" and not live_budget_ok():
        raise LiveDenied("live rate limit reached on this instance; try again shortly")

    return concierge(
        utterance=str(payload.get("utterance", "")),
        language=str(payload.get("language", "en")),
        at_node=str(payload.get("at_node", "")),
        accessible=bool(payload.get("accessible", False)),
        venue_id=venue,
        model_plane=plane,
        seat=payload.get("seat"),
        cordoned=payload.get("cordoned") or (),
    )
