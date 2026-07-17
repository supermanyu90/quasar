"""POST /api/stress -- fire scenarios at the venue and report what breaks.

Findings are not bugs in the code. They are defects in the *venue*: the harness
exists to find them before the gates open, and against the reference stadium it
finds a real one.
"""

from __future__ import annotations

from typing import Any

from _shared import Endpoint
from quasar.web import stress


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    venue = payload.get("venue") or self.venue_param or ""
    return stress(venue, str(payload.get("kind", "gate_failure")), int(payload.get("n", 3)))
