"""POST /api/verify -- re-verify an audit chain.

Backs the console's "tamper" button: edit any record and every hash after it stops
matching. This is what makes the log worth keeping -- you cannot rewrite what the
system proposed after you have seen how it turned out.
"""

from __future__ import annotations

from typing import Any

from _shared import Endpoint
from quasar.governance import AuditLog


def run(self: Endpoint, payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("audit") or []
    try:
        AuditLog.resume(records)
    except ValueError as exc:
        return {"valid": False, "detail": str(exc)}
    return {"valid": True, "records": len(records)}
