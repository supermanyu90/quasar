"""Test fixtures.

The match-day scenario and the recorded transcripts now live in
:mod:`quasar.demo_data`, because the web console and the CLI demo need them too
and neither should import from a test package. This module re-exports them and
adds the test-only doubles.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from quasar.demo_data import (  # noqa: F401  (re-exported for the test modules)
    BRIEF_TRANSCRIPT,
    CASUALTY_NODE,
    COMMANDER,
    CONCIERGE_TRANSCRIPT,
    CORRELATION,
    CROWD_TRANSCRIPT,
    PINCH,
    PLAN_TRANSCRIPT,
    STEWARD,
    VOLUNTEER_BRIEF_TRANSCRIPT,
    VOLUNTEER_REPORT,
    ZONE_LANGUAGES,
    full_density,
    match_day_snapshot as _base_snapshot,
    quiet_snapshot as _base_quiet,
)
from quasar.llm import DisabledModel, ModelRequest, ModelResponse, ModelUnavailable
from quasar.plane import DeterministicPlane
from quasar.types import TelemetrySnapshot
from quasar.venue_spec import reference_venue

VENUE = reference_venue()


def zone_of(node_id: str) -> str:
    """The venue resolves its own geography now -- there is no global to bind."""
    return VENUE.node(node_id).zone


def plane() -> DeterministicPlane:
    return DeterministicPlane(VENUE)


def match_day_snapshot(*, t: float = 1000.0) -> TelemetrySnapshot:
    """The reference scenario, with every un-sensed edge filled in at a quiet 0.3."""
    base = _base_snapshot(t=t)
    return TelemetrySnapshot(
        t=base.t,
        edge_density=full_density(VENUE, base),
        gates=base.gates,
        weather=base.weather,
        source_ids=base.source_ids,
    )


def quiet_snapshot() -> TelemetrySnapshot:
    base = _base_quiet()
    return TelemetrySnapshot(
        t=base.t,
        edge_density=full_density(VENUE, base),
        gates=base.gates,
    )


class DeadModel(DisabledModel):
    """Total partition: no cloud, no edge box."""

    name = "dead"

    def __init__(self) -> None:
        super().__init__("network partition: no cloud, no edge")


class SequenceModel:
    """Returns queued responses in order. Lets a test drive the repair loop."""

    name = "sequence"

    def __init__(self, *responses: str) -> None:
        self._queue = list(responses)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if not self._queue:
            raise ModelUnavailable("sequence exhausted")
        return ModelResponse(
            text=self._queue.pop(0), model="sequence", latency_ms=0.0, plane="transcript"
        )


def j(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload)
