"""POST/GET /api/readiness -- can this venue safely open its gates?

The question only a venue-aware system can ask. A Mumbai stadium and a Chennai
arena run the same code and fail for opposite reasons: one has a stand with no
step-free exit, the other cannot lawfully tell its own majority-language crowd to
evacuate. Neither finding is visible from a floor plan."""

from __future__ import annotations

import os
import sys

# Vercel imports this file directly; api/ is not guaranteed to be on sys.path, so
# put it there before reaching for the shared plumbing (which in turn puts src/ on
# the path so the quasar package is importable).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Any

from _shared import Endpoint, endpoint
from quasar.web import readiness_json


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    return readiness_json(payload.get("venue") or self.venue_param)


handler = endpoint(run)
