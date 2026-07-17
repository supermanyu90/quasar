"""GET /api/state -- what the deterministic plane believes, before any model runs.

Served first and rendered first, so the console visibly shows the severity floor,
the hotspots and the gate metrics as *measurements* -- not as the model's opinion."""

from __future__ import annotations

from typing import Any

from _shared import Endpoint
from quasar.web import state_json


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    return state_json(payload.get("venue") or self.venue_param)
