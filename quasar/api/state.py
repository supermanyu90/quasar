"""GET /api/state -- what the deterministic plane believes, before any model runs.

Served first and rendered first, so the console visibly shows the severity floor,
the hotspots and the gate metrics as *measurements* -- not as the model's opinion."""

from __future__ import annotations

import os
import sys

# Vercel imports this file directly; api/ is not guaranteed to be on sys.path, so
# put it there before reaching for the shared plumbing (which in turn puts src/ on
# the path so the quasar package is importable).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Any

from _shared import Endpoint, endpoint
from quasar.web import state_json


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    return state_json(payload.get("venue") or self.venue_param)


handler = endpoint(run)
