"""GET /api/guide — the problem-statement alignment guide (challenge, users,
objectives), each item carrying a deep-link into the feature that proves it."""

from __future__ import annotations

from typing import Any

from _shared import Endpoint
from quasar.web import guide_json


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    return guide_json()
