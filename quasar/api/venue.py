"""GET /api/venue?venue=<id> -- one venue's graph. The console draws the map from it."""

from __future__ import annotations

from typing import Any

from _shared import Endpoint
from quasar.web import venue_json


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    return venue_json(payload.get("venue") or self.venue_param)
