"""Single Vercel entrypoint for every /api/* route.

Modern Vercel builds one *named* Python entrypoint per project (``index.py`` /
``app.py`` / ...), not every file in ``api/`` the way the older runtime did. This
module is that entrypoint. It does not reimplement anything: it reuses the exact
per-endpoint ``run()`` functions that the local dev server (``tools/serve.py``)
routes to file-by-file, so there is a single implementation of each endpoint,
exercised identically in both places.

Routing: ``vercel.json`` rewrites ``/api/<name>`` to ``/api/index?endpoint=<name>``
and merges the original query string, so the endpoint name and ``?venue=...`` arrive
together as query parameters. Dispatch reuses :class:`_shared.Endpoint`'s HTTP
plumbing (body parsing, CORS, the refusal-to-status map) unchanged -- this module
only chooses which ``run`` to call.
"""

from __future__ import annotations

import os
import sys
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _shared import Endpoint

from actuate import run as _actuate
from agent import run as _agent
from amenities import run as _amenities
from concierge import run as _concierge
from guide import run as _guide
from readiness import run as _readiness
from state import run as _state
from stress import run as _stress
from venue import run as _venue
from venues import run as _venues
from verify import run as _verify
from wayfind import run as _wayfind

_ROUTES = {
    "actuate": _actuate,
    "agent": _agent,
    "amenities": _amenities,
    "concierge": _concierge,
    "guide": _guide,
    "readiness": _readiness,
    "state": _state,
    "stress": _stress,
    "venue": _venue,
    "venues": _venues,
    "verify": _verify,
    "wayfind": _wayfind,
}


# The name Vercel loads: a BaseHTTPRequestHandler subclass called `handler`.
# Declared as an explicit class (not endpoint()'s dynamic type()), because Vercel's
# Python entrypoint detection scans for a top-level `handler`/`app`/`application`.
class handler(Endpoint):
    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        name = (parse_qs(urlparse(self.path).query).get("endpoint") or [None])[0]
        fn = _ROUTES.get(name)
        if fn is None:
            raise ValueError(f"unknown endpoint {name!r}")
        return fn(self, payload)
