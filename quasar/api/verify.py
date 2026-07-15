"""POST /api/verify -- re-verify an audit chain.

Backs the console's "tamper" button: edit any record and every hash after it stops
matching. This is what makes the log worth keeping -- you cannot rewrite what the
system proposed after you have seen how it turned out.
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
from quasar.governance import AuditLog


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("audit") or []
    try:
        AuditLog.resume(records)
    except ValueError as exc:
        return {"valid": False, "detail": str(exc)}
    return {"valid": True, "records": len(records)}


handler = endpoint(run)
