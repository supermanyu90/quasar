"""POST /api/stress -- fire scenarios at the venue and report what breaks.

Findings are not bugs in the code. They are defects in the *venue*: the harness
exists to find them before the gates open, and against the reference stadium it
finds a real one.
"""

from __future__ import annotations

import os
import sys

# Vercel imports this file directly; api/ is not guaranteed to be on sys.path, so
# put it there before reaching for the shared plumbing (which in turn puts src/ on
# the path so the quasar package is importable).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Any

from _shared import Endpoint, endpoint
from quasar.web import stress


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    venue = payload.get("venue") or self.venue_param or ""
    return stress(venue, str(payload.get("kind", "gate_failure")), int(payload.get("n", 3)))


handler = endpoint(run)
