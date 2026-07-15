"""Local dev server that mimics Vercel's routing, so the console can be verified
before it is deployed.

    PYTHONPATH=src python3 tools/serve.py     # http://127.0.0.1:8000

Serves public/ as static files and dispatches /api/<name> to api/<name>.py's
`handler`, exactly as Vercel does. It is a development tool; Vercel itself runs
the same handler classes in production, so what you see here is what deploys.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

_HANDLERS: dict[str, type] = {}


def load(name: str) -> type | None:
    if name in _HANDLERS:
        return _HANDLERS[name]
    path = ROOT / "api" / f"{name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"api_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _HANDLERS[name] = module.handler
    return module.handler


class Router(SimpleHTTPRequestHandler):
    def _route(self) -> bool:
        if not self.path.startswith("/api/"):
            return False
        name = self.path[len("/api/"):].split("?")[0].strip("/")
        handler_cls = load(name)
        if handler_cls is None:
            self.send_response(404)
            self.end_headers()
            return True
        # Re-dispatch this live connection through the Vercel handler class.
        handler_cls.__init__ = lambda *a, **k: None  # type: ignore[method-assign]
        h = handler_cls()
        h.rfile, h.wfile, h.headers = self.rfile, self.wfile, self.headers
        h.request_version, h.requestline, h.client_address = (
            self.request_version, self.requestline, self.client_address,
        )
        # The endpoint reads self.path (for the ?venue= query param) and self.command;
        # copy the whole request line's worth of state so the handler sees this request.
        h.path, h.command = self.path, self.command
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

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}\n")


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    os.chdir(ROOT / "public")
    server = HTTPServer(("127.0.0.1", port), partial(Router, directory=str(ROOT / "public")))
    print(f"Quasar control room -> http://127.0.0.1:{port}")
    print("(mimics Vercel: public/ static + api/*.py functions)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
