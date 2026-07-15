"""GET /api/venues -- every venue this deployment serves.

The venue is configuration, not code. Adding a stadium is a data change."""

from __future__ import annotations

import os
import sys

# Vercel imports this file directly; api/ is not guaranteed to be on sys.path, so
# put it there before reaching for the shared plumbing (which in turn puts src/ on
# the path so the quasar package is importable).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Any

from _shared import Endpoint, endpoint
from quasar.web import venues_json


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    return venues_json()


handler = endpoint(run)
