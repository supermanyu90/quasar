"""The web adapter: the control room, for whichever venue you point it at.

The serverless functions in ``api/`` are deliberately thin — they parse a request,
call in here, and serialise the answer. All of the wiring lives in the package so
it is testable without a web server, and so the console cannot become a second,
drifting implementation of the venue's logic.

Three properties matter more than anything else in this module.

**The venue is configuration, not code.** Every function here takes a venue id and
resolves it through :mod:`quasar.venue_spec`. Nothing is hardcoded to one stadium.
Two venues are held in memory at once and answer independently — which is the whole
test of whether this is a venue operating system or a demo of one particular venue.

**The server never trusts the client.** A serverless function has no memory, so the
plan the console renders travels back to `/api/actuate` as JSON — through the
browser, where anyone can edit it. That is not a weakness to apologise for; it is
the barrier's job. Every payload arriving from the client is re-validated against
the published schema, re-corroborated against the deterministic plane, re-checked
against policy, and re-gated on the human signature, exactly as if a model had just
produced it. A hostile client is an agent with a worse prompt.

**Live mode is gated.** A public URL with an API key behind it is a stranger's
budget."""

from __future__ import annotations

from quasar.web.core import Mode, MODES, VENUES, DEFAULT_VENUE, LiveDenied, UnknownVenue, ModelPlane, profile, resolve_model, orchestrator, assessment, zone_of
from quasar.web.serializers import result_json, route_json
from quasar.web.views import guide_json, venues_json, venue_json, state_json, readiness_json, stress
from quasar.web.control import run_agent, actuate
from quasar.web.attendee import concierge, amenities_json, wayfind

# Re-exported so tests can assert on the controlled catalogues directly.
from quasar.web.attendee import (  # noqa: F401
    _CONCIERGE_ACK,
    _WORDING,
    _localise_concierge_reply,
)

__all__ = [
    "DEFAULT_VENUE",
    "LiveDenied",
    "MODES",
    "Mode",
    "ModelPlane",
    "UnknownVenue",
    "VENUES",
    "actuate",
    "amenities_json",
    "assessment",
    "concierge",
    "guide_json",
    "orchestrator",
    "profile",
    "readiness_json",
    "resolve_model",
    "result_json",
    "route_json",
    "run_agent",
    "state_json",
    "stress",
    "venue_json",
    "venues_json",
    "wayfind",
    "zone_of",
]
