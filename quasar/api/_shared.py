"""Request plumbing shared by the Vercel functions.

Underscore-prefixed, so Vercel treats it as a library rather than routing it as an
endpoint. It contains no venue logic — everything of substance lives in
``quasar.web`` so it is testable without a web server.
"""

from __future__ import annotations

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler
from typing import Any, Callable

# The package lives in src/ and the functions in api/. Vercel bundles src/** via
# the `includeFiles` rule in vercel.json; this makes it importable.
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from quasar.governance import (  # noqa: E402
    ApprovalRequired,
    NotAuthorised,
    PolicyError,
)
from quasar.language import CatalogueError  # noqa: E402
from quasar.routing import NoRouteError  # noqa: E402
from quasar.schemas import SchemaError  # noqa: E402
from quasar.web import LiveDenied, UnknownVenue  # noqa: E402

# Best-effort, per-warm-instance rate limit on live model calls. It is not a real
# limiter — serverless instances are ephemeral and parallel, so this leaks. It is
# a speed bump in front of somebody else's budget, and the honest defence is that
# live mode needs a secret at all. Anything stronger needs shared state (KV).
_LIVE_CALLS: list[float] = []
LIVE_CALLS_PER_MINUTE = 12


def live_budget_ok() -> bool:
    now = time.time()
    _LIVE_CALLS[:] = [t for t in _LIVE_CALLS if now - t < 60.0]
    if len(_LIVE_CALLS) >= LIVE_CALLS_PER_MINUTE:
        return False
    _LIVE_CALLS.append(now)
    return True


# Errors the barrier raises on purpose. Each maps to a status the console renders
# as a *result*, not as a crash: being refused is the system working.
_REFUSALS: dict[type[Exception], tuple[int, str]] = {
    ApprovalRequired: (403, "approval_required"),
    NotAuthorised: (403, "not_authorised"),
    PolicyError: (422, "policy_violation"),
    SchemaError: (422, "schema_violation"),
    CatalogueError: (422, "catalogue_refusal"),
    NoRouteError: (409, "no_route"),
    UnknownVenue: (404, "unknown_venue"),
    LiveDenied: (402, "live_denied"),
    ValueError: (400, "rejected"),
}


class Endpoint(BaseHTTPRequestHandler):
    """Base handler. Subclasses implement ``run(payload) -> dict``."""

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    # -- HTTP ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler's contract)
        self._dispatch({})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "malformed_json"})
            return
        if not isinstance(payload, dict):
            self._send(400, {"error": "expected a JSON object"})
            return
        self._dispatch(payload)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, None)

    def _dispatch(self, payload: dict[str, Any]) -> None:
        try:
            self._send(200, self.run(payload))
        except Exception as exc:  # noqa: BLE001 -- the boundary; classify below
            for kind, (status, code) in _REFUSALS.items():
                if isinstance(exc, kind):
                    self._send(status, {"error": code, "detail": str(exc)})
                    return
            self._send(500, {"error": "internal", "detail": str(exc)})

    def _send(self, status: int, body: dict[str, Any] | None) -> None:
        self.send_response(status)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-headers", "content-type, x-quasar-key")
        self.send_header("cache-control", "no-store")
        self.end_headers()
        if body is not None:
            self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args: Any) -> None:  # keep the function logs quiet
        return

    # -- helpers ---------------------------------------------------------

    @property
    def live_secret(self) -> str | None:
        return self.headers.get("x-quasar-key")

    @property
    def venue_param(self) -> str | None:
        """?venue=<id> on a GET. POSTs put it in the body instead."""
        from urllib.parse import parse_qs, urlparse

        q = parse_qs(urlparse(self.path).query)
        return (q.get("venue") or [None])[0]


def endpoint(run: Callable[[Endpoint, dict[str, Any]], dict[str, Any]]) -> type[Endpoint]:
    """Build a Vercel `handler` class from a single function."""
    return type("handler", (Endpoint,), {"run": run})
