"""Foundation: the venue registry, model-plane resolution, and the assessment
every view and action reads from."""

from __future__ import annotations

import os

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Mapping, Sequence

from quasar import demo_data
from quasar.governance import AuditLog, Orchestrator
from quasar.language import MessageCatalogue
from quasar.llm import AnthropicModel, DisabledModel, FailoverModel, LanguageModel, OllamaEdgeModel, TranscriptModel
from quasar.plane import Assessment, DeterministicPlane
from quasar.types import TelemetrySnapshot
from quasar.venue_spec import VenueProfile, discover


Mode = Literal["recorded", "edge", "live", "partition"]


MODES: tuple[Mode, ...] = ("recorded", "edge", "live", "partition")


# Venue specs are read once per warm instance. A malformed spec fails here, loudly,
# rather than dropping a venue out of the list where an operator would conclude the
# software was broken rather than their config.
VENUES: Mapping[str, VenueProfile] = discover()


DEFAULT_VENUE = "national-stadium" if "national-stadium" in VENUES else next(iter(VENUES))


class LiveDenied(Exception):
    """Live mode was requested without the operator's secret, or without a key."""


class UnknownVenue(KeyError):
    """The request named a venue this deployment does not serve."""


def profile(venue_id: str | None) -> VenueProfile:
    vid = venue_id or DEFAULT_VENUE
    if vid not in VENUES:
        raise UnknownVenue(f"unknown venue {vid!r}; this deployment serves {sorted(VENUES)}")
    return VENUES[vid]


@lru_cache(maxsize=16)
def _plane(venue_id: str) -> DeterministicPlane:
    """One deterministic plane per venue. Stateless, so it is safe to cache."""
    return DeterministicPlane(profile(venue_id).venue)


@lru_cache(maxsize=16)
def _catalogue(venue_id: str) -> MessageCatalogue:
    p = profile(venue_id)
    return MessageCatalogue(
        known_gates=p.gate_ids,
        known_edges=frozenset(p.venue.edges),
        known_zones=p.zones,
        labels=p.labels,
    )


@dataclass(frozen=True, slots=True)
class ModelPlane:
    model: LanguageModel
    mode: Mode
    note: str


def resolve_model(mode: str, secret: str | None, venue_id: str) -> ModelPlane:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    if mode == "partition":
        return ModelPlane(
            model=FailoverModel(
                primary=DisabledModel("network partition: no cloud"),
                secondary=DisabledModel("network partition: no edge box"),
            ),
            mode="partition",
            note="Model plane disabled. Every agent takes its deterministic twin.",
        )

    if mode == "edge":
        model = os.environ.get("QUASAR_EDGE_MODEL", "gemma3:4b")
        endpoint = os.environ.get("QUASAR_EDGE_ENDPOINT", "http://127.0.0.1:11434/api/chat")
        return ModelPlane(
            model=OllamaEdgeModel(endpoint=endpoint, model=model, timeout_s=55.0),
            mode="edge",
            note=(
                f"Real inference on the on-venue edge model ({model}). No API key, no "
                "internet. It is a small model and it will sometimes fail the barrier — "
                "watch which agents fall back, and why."
            ),
        )

    if mode == "live":
        expected = os.environ.get("QUASAR_LIVE_TOKEN")
        if not expected:
            raise LiveDenied("live mode is not enabled on this deployment")
        if not secret or secret != expected:
            raise LiveDenied("live mode requires the operator's key")
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise LiveDenied("no Anthropic credentials are configured on this deployment")
        name = os.environ.get("QUASAR_LIVE_MODEL", "claude-opus-4-8")
        return ModelPlane(
            model=AnthropicModel(model=name, timeout_s=50.0),
            mode="live",
            note=f"Live: {name}. Output goes through exactly the same barrier.",
        )

    transcripts = demo_data.transcripts(venue_id)
    return ModelPlane(
        model=TranscriptModel(transcripts),
        mode="recorded",
        note=(
            "Recorded model output. Nothing safety-critical is faked: the router, the "
            "queueing model, the catalogue, the schema validator, the grounding check, "
            "the corroborators and the human barrier all run for real."
        )
        if transcripts
        else (
            "No recording exists for this venue's fixture, so every agent will take its "
            "deterministic twin. Switch to Edge to run a real model against it."
        ),
    )


def orchestrator(
    venue_id: str, plane: ModelPlane, audit_chain: Sequence[Mapping[str, Any]] | None = None
) -> Orchestrator:
    log = AuditLog.resume(audit_chain) if audit_chain else AuditLog()
    return Orchestrator(
        _plane(venue_id), plane.model, catalogue=_catalogue(venue_id), audit=log
    )


def assessment(venue_id: str) -> tuple[VenueProfile, TelemetrySnapshot, Assessment]:
    p = profile(venue_id)
    snap = p.fixture.snapshot(p.venue)
    return p, snap, _plane(venue_id).assess(snap)


def zone_of(venue_id: str, node_id: str) -> str:
    return profile(venue_id).venue.node(node_id).zone
