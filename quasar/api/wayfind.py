"""POST /api/wayfind — route an attendee to an amenity.

The friendly face of the same density-aware router the control room uses: the
route is crowd-aware, and step-free / calm when asked. Pure deterministic plane —
no model, so it always works and costs nothing.
"""

from __future__ import annotations

from typing import Any

from _shared import Endpoint
from quasar.web import wayfind


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    return wayfind(
        payload.get("venue") or self.venue_param or "",
        from_node=str(payload.get("from_node", "")),
        amenity_key=str(payload.get("amenity", "")),
        accessible=bool(payload.get("accessible", False)),
        calm=bool(payload.get("calm", False)),
        language=str(payload.get("language", "en")),
        seat=payload.get("seat"),
        cordoned=payload.get("cordoned") or (),
    )
