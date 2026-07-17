"""GET /api/venues -- every venue this deployment serves.

The venue is configuration, not code. Adding a stadium is a data change."""

from __future__ import annotations

from typing import Any

from _shared import Endpoint
from quasar.web import venues_json


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    return venues_json()
