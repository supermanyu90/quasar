"""Local dev server that mirrors the Vercel deployment, so the console can be
verified before it ships.

    PYTHONPATH=src python3 tools/serve.py     # http://127.0.0.1:8000

Serves public/ as static files and routes every /api/<name> request through
api/index.py's `handler` -- the exact single entrypoint and dispatch that runs in
production, behind the same rewrite. What you see here is what deploys.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

_HANDLER: type | None = None


def index_handler() -> type:
    """api/index.py's handler, loaded once. It dispatches every endpoint by the
    ?endpoint= parameter, exactly as it does behind Vercel's rewrite."""
    global _HANDLER
    if _HANDLER is None:
        spec = importlib.util.spec_from_file_location("api_index", ROOT / "api" / "index.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _HANDLER = module.handler
        # BaseHTTPRequestHandler.__init__ reads and serves the request on construction;
        # bypass it so we can hand the class a connection we have already accepted.
        _HANDLER.__init__ = lambda *a, **k: None  # type: ignore[method-assign]
    return _HANDLER


class Router(SimpleHTTPRequestHandler):
    def _route(self) -> bool:
        if not self.path.startswith("/api/"):
            return False
        parsed = urlparse(self.path)
        # Mimic the vercel.json rewrite: /api/<name> -> /api?endpoint=<name>, merging
        # the original query so ?venue=... still reaches the handler.
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        query["endpoint"] = parsed.path[len("/api/"):].strip("/")

        h = index_handler()()
        h.rfile, h.wfile, h.headers = self.rfile, self.wfile, self.headers
        h.request_version, h.requestline, h.client_address = (
            self.request_version, self.requestline, self.client_address,
        )
        h.path, h.command = "/api?" + urlencode(query), self.command
        h.send_response = self.send_response  # type: ignore[method-assign]
        h.send_header = self.send_header  # type: ignore[method-assign]
        h.end_headers = self.end_headers  # type: ignore[method-assign]
        getattr(h, f"do_{self.command}")()
        return True

    def do_GET(self) -> None:  # noqa: N802
        if not self._route():
            super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if not self._route():
            self.send_response(405)
            self.end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not self._route():
            self.send_response(204)
            self.end_headers()

    def end_headers(self) -> None:
        # Never let the browser cache the console during development. Editing a file
        # while the page is open otherwise leaves a stale index.html running against
        # a fresh app.js, and the mismatch throws on an element that "isn't there".
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    os.chdir(ROOT / "public")
    server = HTTPServer(("127.0.0.1", port), partial(Router, directory=str(ROOT / "public")))
    print(f"Quasar control room -> http://127.0.0.1:{port}")
    print("(mirrors Vercel: public/ static + api/index.py dispatch)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
