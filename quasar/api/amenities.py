"""GET /api/amenities?venue=<id> — the attendee amenity catalogue for a venue,
flagged with what this venue actually has mapped so the UI can grey out the rest."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Any

from _shared import Endpoint, endpoint
from quasar.web import amenities_json


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    return amenities_json(payload.get("venue") or self.venue_param or "")


handler = endpoint(run)
