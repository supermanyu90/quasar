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
budget.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Mapping, Sequence

from quasar import alignment, demo_data
from quasar.amenities import AMENITIES, BY_KEY, CALM_MAX_DENSITY, GROUPS
from quasar.agents import (
    ConciergeAgent,
    ConciergeTask,
    CrowdIntelligenceAgent,
    CrowdTask,
    IncidentResponseAgent,
    IncidentTask,
    PlannerAgent,
    PlanTask,
)
from quasar.governance import AgentResult, AuditLog, Orchestrator
from quasar.language import MessageCatalogue
from quasar.llm import (
    AnthropicModel,
    DisabledModel,
    FailoverModel,
    LanguageModel,
    OllamaEdgeModel,
    TranscriptModel,
)
from quasar.plane import Assessment, DeterministicPlane
from quasar.readiness import audit
from quasar.routing import ACCESSIBLE, FAN, NoRouteError, Profile, Route
from quasar.scenarios import SeededSampler, StressHarness
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


# ==========================================================================
# Model plane
# ==========================================================================


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


# ==========================================================================
# Guide — problem-statement alignment
# ==========================================================================


def guide_json() -> dict[str, Any]:
    """The 'How to use' guide: the challenge, the users, the objectives, each with
    a live deep-link into the feature that proves it."""
    def item(i: alignment.Item) -> dict[str, Any]:
        return {
            "key": i.key, "icon": i.icon, "title": i.title,
            "summary": i.summary, "how": i.how, "cta": i.cta,
            "where": dict(i.where), "modules": list(i.modules),
        }

    return {
        "challenge": alignment.CHALLENGE,
        "thesis": alignment.THESIS,
        "sections": [
            {"key": k, "heading": h, "items": [item(i) for i in items]}
            for k, h, items in alignment.sections()
        ],
    }


# ==========================================================================
# State
# ==========================================================================


def assessment(venue_id: str) -> tuple[VenueProfile, TelemetrySnapshot, Assessment]:
    p = profile(venue_id)
    snap = p.fixture.snapshot(p.venue)
    return p, snap, _plane(venue_id).assess(snap)


def zone_of(venue_id: str, node_id: str) -> str:
    return profile(venue_id).venue.node(node_id).zone


# ==========================================================================
# Serialisation
# ==========================================================================


def venues_json() -> dict[str, Any]:
    return {
        "default": DEFAULT_VENUE,
        "venues": [
            {
                "id": p.id,
                "name": p.name,
                "fifa_name": p.fifa_name,
                "city": p.city,
                "country": p.country,
                "capacity": p.capacity,
                "topology": p.topology,
                "languages": list(p.languages),
                "nodes": len(p.venue.nodes),
                "edges": len(p.venue.edges),
                "fixture": p.fixture.name,
            }
            for p in VENUES.values()
        ],
    }


def venue_json(venue_id: str) -> dict[str, Any]:
    p = profile(venue_id)
    return {
        "id": p.id,
        "name": p.name,
        "fifa_name": p.fifa_name,
        "city": p.city,
        "country": p.country,
        "capacity": p.capacity,
        "topology": p.topology,
        "languages": list(p.languages),
        "nodes": [
            {
                "id": n.id, "name": n.name, "x": n.x, "y": n.y,
                "level": n.level, "zone": n.zone, "tags": sorted(n.tags),
            }
            for n in p.venue.nodes.values()
        ],
        "edges": [
            {
                "id": e.id, "u": e.u, "v": e.v,
                "length_m": e.length_m, "width_m": e.width_m, "kind": e.kind.value,
                "step_free": e.step_free, "staff_only": e.staff_only,
            }
            for e in p.venue.edges.values()
        ],
        "casualty_node": p.fixture.casualty_node,
        "fan": {
            "at_node": p.fixture.fan_at_node,
            "seat": p.fixture.fan_seat,
            "language": p.fixture.fan_language,
            "accessible": p.fixture.fan_accessible,
            "utterance": p.fixture.fan_utterance,
        },
    }


def state_json(venue_id: str) -> dict[str, Any]:
    p, snap, a = assessment(venue_id)
    plane = _plane(venue_id)
    floor = plane.severity_floor(a, p.fixture.casualty_node, p.fixture.category)

    disabled = ModelPlane(DisabledModel(), "partition", "")
    retrieved = orchestrator(venue_id, disabled).retriever.for_incident(
        p.fixture.category, p.fixture.report.text
    )

    return {
        "venue": p.id,
        "fixture": p.fixture.name,
        "t": snap.t,
        "density": dict(snap.edge_density),
        "hotspots": [
            {
                "edge_id": h.edge_id, "zone": h.zone, "density": round(h.density, 2),
                "los": h.los.value, "trend": h.trend, "critical": h.critical,
            }
            for h in a.hotspots
        ],
        "gates": {
            gid: {
                "utilisation": round(m.utilisation, 3),
                "wait_s": None if not m.stable else round(m.wait_s, 1),
                "stable": m.stable,
                "breaches": m.breaches_trigger,
                "open_lanes": snap.gates[gid].open_lanes,
                "installed_lanes": snap.gates[gid].installed_lanes,
                "lanes_needed": plane.lanes_needed(snap.gates[gid]),
            }
            for gid, m in sorted(a.gates.items())
        },
        "critical_edges": list(a.critical_edges),
        "breaching_gates": list(a.breaching_gates),
        "incident": {
            "id": p.fixture.report.id,
            "at_node": p.fixture.casualty_node,
            "reporter": p.fixture.report.reporter_role.value,
            "text": p.fixture.report.text,
        },
        "severity_floor": floor.value,
        "severity_floor_reason": (
            "SOP-MED-03#1: crowd pressure at the casualty exceeds LOS E, so a medical "
            "incident here is P0, not P1. An agent may grade it higher. It may never "
            "grade it lower."
        ),
        "retrieved": [h.ref for h in retrieved],
        "languages": list(p.languages),
    }


def readiness_json(venue_id: str) -> dict[str, Any]:
    r = audit(profile(venue_id))
    return {
        "venue": r.venue_id,
        "name": r.venue_name,
        "ready": r.ready,
        "checks": [
            {
                "id": c.id, "severity": c.severity, "title": c.title,
                "detail": c.detail, "remedy": c.remedy,
            }
            for c in r.checks
        ],
    }


def result_json(result: AgentResult) -> dict[str, Any]:
    return {
        "agent": result.agent,
        "schema": result.schema_id,
        "source": result.source,
        "plane": result.plane,
        "payload": dict(result.payload),
        "self_reported_confidence": result.self_reported_confidence,
        "corroboration_score": result.corroboration.score,
        "corroboration_notes": list(result.corroboration.notes),
        "effective_confidence": result.effective_confidence,
        "fallback_reason": result.fallback_reason,
        "latency_ms": round(result.latency_ms, 1),
    }


def route_json(route: Route) -> dict[str, Any]:
    return {
        "origin": route.origin,
        "destination": route.destination,
        "nodes": list(route.nodes),
        "edges": list(route.edges),
        "distance_m": round(route.distance_m, 1),
        "eta_s": round(route.eta_s, 1),
        "worst_density": round(route.worst_density, 2),
        "worst_los": route.worst_los.value,
        "profile": route.profile,
    }


# ==========================================================================
# Pipeline stages -- one per invocation, so each gets its own timeout budget
# ==========================================================================


def run_agent(
    name: str,
    venue_id: str,
    plane: ModelPlane,
    *,
    brief: Mapping[str, Any] | None = None,
    audit_chain: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], AuditLog]:
    orch = orchestrator(venue_id, plane, audit_chain)
    p, snap, a = assessment(venue_id)
    det = _plane(venue_id)
    hits = orch.retriever.for_incident(p.fixture.category, p.fixture.report.text)
    correlation = f"cyc-{p.id[:8]}-0001"

    match name:
        case "crowd":
            result = orch.runner.run(
                CrowdIntelligenceAgent(),
                CrowdTask(correlation_id=correlation, assessment=a),
            )
        case "incident":
            result = orch.runner.run(
                IncidentResponseAgent(),
                IncidentTask(
                    correlation_id=correlation,
                    report=p.fixture.report,
                    assessment=a,
                    category=p.fixture.category,
                    retrieved=hits,
                    severity_floor=det.severity_floor(
                        a, p.fixture.casualty_node, p.fixture.category
                    ),
                    zone=zone_of(venue_id, p.fixture.casualty_node),
                ),
                retrieved=hits,
            )
        case "plan":
            if brief is None:
                raise ValueError("the planner needs the incident brief")
            from quasar import schemas

            # The brief came from a browser. Re-validate before it shapes a plan.
            schemas.validate(brief, schemas.INCIDENT_BRIEF)
            result = orch.runner.run(
                PlannerAgent(),
                PlanTask(
                    correlation_id=correlation,
                    plan_id=f"plan-{p.fixture.report.id}",
                    brief=brief,
                    assessment=a,
                    snapshot=snap,
                    casualty_node=p.fixture.casualty_node,
                    retrieved=hits,
                    plane=det,
                ),
            )
        case _:
            raise ValueError(f"unknown agent {name!r}")

    return result_json(result), orch.audit


def actuate(
    plan: Mapping[str, Any],
    approver: str | None,
    venue_id: str,
    model_plane: ModelPlane,
    *,
    audit_chain: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], AuditLog]:
    """Policy, then the human barrier, then execution — on a plan that came from a
    browser and is therefore assumed hostile until it has passed all three."""
    from quasar import schemas

    orch = orchestrator(venue_id, model_plane, audit_chain)
    p, snap, a = assessment(venue_id)
    det = _plane(venue_id)

    schemas.validate(plan, schemas.PLAN_PROPOSAL)

    corroboration = PlannerAgent().corroborate(
        plan,
        PlanTask(
            correlation_id=f"cyc-{p.id[:8]}-0001",
            plan_id=plan["plan_id"],
            brief={"severity": plan["severity"]},
            assessment=a,
            snapshot=snap,
            casualty_node=p.fixture.casualty_node,
            retrieved=(),
            plane=det,
        ),
    )
    if corroboration.fatal:
        orch.audit.append(
            "plan.rejected",
            {"plan_id": plan["plan_id"], "reason": list(corroboration.notes)},
        )
        raise ValueError("; ".join(corroboration.notes))

    orch.submit_for_approval(plan)
    approval = None
    if approver:
        operator = demo_data.OPERATORS.get(approver)
        if operator is None:
            raise ValueError(f"unknown operator role {approver!r}")
        approval = orch.hitl.decide(operator, plan["plan_id"], approved=True, note="via console")

    execution = orch.actuate(
        plan, a, snap, languages=list(p.languages), approval=approval
    )

    return {
        "plan_id": execution.plan_id,
        "applied": list(execution.applied),
        "cordoned": sorted(execution.cordoned),
        "routes": {k: route_json(r) for k, r in execution.routes.items()},
        "gate_after": {
            gid: {
                "utilisation": round(m.utilisation, 3),
                "breaches": m.breaches_trigger,
                "open_lanes": execution.gate_state[gid].open_lanes,
            }
            for gid, m in execution.gate_after.items()
        },
        "gate_before": {gid: round(m.utilisation, 3) for gid, m in sorted(a.gates.items())},
        "dispatches": [
            {
                "template_id": d.template_id,
                "announcements": [
                    {"language": x.language, "text": x.text, "status": x.status.value}
                    for x in d.announcements
                ],
                "refused_languages": list(d.refused_languages),
                "pictogram": d.pictogram,
                "steward_required": d.steward_required,
            }
            for d in execution.dispatches
        ],
        "escalations": list(execution.escalations),
        "warnings": list(execution.warnings),
        "approved_by": approval.operator.name if approval else None,
    }, orch.audit


def concierge(
    utterance: str,
    language: str,
    at_node: str,
    accessible: bool,
    venue_id: str,
    model_plane: ModelPlane,
    *,
    seat: str | None = None,
    cordoned: Sequence[str] = (),
) -> dict[str, Any]:
    p, _snap, a = assessment(venue_id)
    orch = orchestrator(venue_id, model_plane)
    det = _plane(venue_id)
    seat = seat or p.fixture.fan_seat

    result = orch.runner.run(
        ConciergeAgent(),
        ConciergeTask(
            correlation_id=f"cyc-{p.id[:8]}-0002",
            utterance=utterance,
            language=language,
            at_node=at_node,
            accessible=accessible,
            assessment=a,
        ),
    )

    route: dict[str, Any] | None = None
    error: str | None = None
    payload = result.payload

    if payload["requires_route"] and payload["destination_tag"]:
        prof = ACCESSIBLE if accessible else FAN
        tag = payload["destination_tag"]
        try:
            if tag == "seating":
                # A seat is not an amenity. "Take me to my seat" resolves to the
                # stand on the fan's ticket; routing them to the *nearest* seating
                # block walks them to the wrong end of the venue, which is the exact
                # problem they came to the concierge with.
                r = det.fan_route(
                    a, from_node=at_node, to_node=seat, profile=prof,
                    cordoned=frozenset(cordoned),
                )
            else:
                r = det.nearest_amenity(
                    a, from_node=at_node, tag=tag, profile=prof,
                    cordoned=frozenset(cordoned),
                )
            route = route_json(r)
        except NoRouteError as exc:
            error = str(exc)

    return {
        "result": result_json(result),
        "route": route,
        "route_error": error,
        "audit": orch.audit.to_json(),
    }


# ==========================================================================
# Attendee companion — deterministic wayfinding to amenities
# ==========================================================================

@dataclass(frozen=True, slots=True)
class _Wording:
    """Controlled, human-authored wayfinding phrasing for one language.

    This is the Tier-2 *informational* path — not the safety-critical announcement
    catalogue — but it obeys the same two rules the catalogue does: the strings are
    authored, never machine-translated at request time, and the entities that must
    survive a language switch are interpolated, never translated — amenity and
    destination NAMES (as the signage reads) and NUMBERS. A whole result card can
    therefore switch language without a single number or place name being at risk.

    Placeholders, by field: ``route`` uses {dest} {m} {mins} {tail}; ``worst`` uses
    {los}; ``more`` uses {n}; ``none_mapped`` and ``no_route`` use {amenity};
    ``no_route`` also uses {sfp}; ``assist`` uses {loc}.
    """

    route: str
    step_free: str          # qualifier word, e.g. "step-free"
    calm: str               # qualifier word, e.g. "calm"
    worst: str              # "Worst crowding on the way: level of service {los}."
    more: str               # "+{n} more of these nearby."
    none_mapped: str
    no_route: str
    step_free_required: str  # the {sfp} clause, or "" when not step-free
    assist: str


# The six languages this deployment ships human-authored phrasing for. Any language
# outside this table falls back to English rather than being machine-translated.
_WORDING: Mapping[str, _Wording] = {
    "en": _Wording(
        route="Here’s your route to {dest} — {m} m, about {mins} min{tail}.",
        step_free="step-free", calm="calm",
        worst="Worst crowding on the way: level of service {los}.",
        more="+{n} more of these nearby.",
        none_mapped="No {amenity} is mapped at this venue yet.",
        no_route="No route to a {amenity} is open from here right now{sfp}. "
                 "Ask the nearest steward for help.",
        step_free_required=" (a step-free route was required)",
        assist="Assistance is on its way to you at {loc}. A steward has been notified.",
    ),
    "es": _Wording(
        route="Aquí tienes tu ruta a {dest}: {m} m, unos {mins} min{tail}.",
        step_free="sin escalones", calm="tranquila",
        worst="Mayor aglomeración en el camino: nivel de servicio {los}.",
        more="+{n} más cerca de aquí.",
        none_mapped="Todavía no hay ningún {amenity} señalizado en este recinto.",
        no_route="Ahora mismo no hay ninguna ruta abierta a un {amenity} desde "
                 "aquí{sfp}. Pide ayuda al acomodador más cercano.",
        step_free_required=" (se requería una ruta sin escalones)",
        assist="La asistencia va en camino hacia ti en {loc}. Se ha avisado a un acomodador.",
    ),
    "fr": _Wording(
        route="Voici votre itinéraire vers {dest} : {m} m, environ {mins} min{tail}.",
        step_free="sans marches", calm="calme",
        worst="Plus forte affluence en chemin : niveau de service {los}.",
        more="+{n} autres à proximité.",
        none_mapped="Aucun {amenity} n’est encore répertorié dans cette enceinte.",
        no_route="Aucun itinéraire vers un {amenity} n’est ouvert d’ici pour le "
                 "moment{sfp}. Demandez de l’aide au steward le plus proche.",
        step_free_required=" (un itinéraire sans marches était requis)",
        assist="De l’aide arrive vers vous à {loc}. Un steward a été prévenu.",
    ),
    "hi": _Wording(
        route="{dest} तक आपका मार्ग — {m} मीटर, लगभग {mins} मिनट{tail}।",
        step_free="बिना सीढ़ी", calm="शांत",
        worst="रास्ते में सबसे अधिक भीड़: सेवा स्तर {los}।",
        more="+{n} और पास में हैं।",
        none_mapped="इस स्थल पर अभी कोई {amenity} मानचित्र पर नहीं है।",
        no_route="अभी यहाँ से किसी {amenity} तक कोई मार्ग खुला नहीं है{sfp}। "
                 "कृपया निकटतम कर्मचारी से सहायता लें।",
        step_free_required=" (बिना सीढ़ी वाला मार्ग आवश्यक था)",
        assist="{loc} पर आपके पास सहायता आ रही है। एक कर्मचारी को सूचित कर दिया गया है।",
    ),
    "mr": _Wording(
        route="{dest} पर्यंत तुमचा मार्ग — {m} मीटर, अंदाजे {mins} मिनिटे{tail}.",
        step_free="पायऱ्यांशिवाय", calm="शांत",
        worst="वाटेत सर्वाधिक गर्दी: सेवा स्तर {los}.",
        more="+{n} आणखी जवळपास आहेत.",
        none_mapped="या ठिकाणी अजून कोणतेही {amenity} नकाशावर नाही.",
        no_route="सध्या इथून कोणत्याही {amenity} पर्यंत मार्ग उपलब्ध नाही{sfp}. "
                 "कृपया जवळच्या कर्मचाऱ्याला विचारा.",
        step_free_required=" (पायऱ्यांशिवाय मार्ग आवश्यक होता)",
        assist="{loc} येथे तुमच्यापर्यंत मदत येत आहे. कर्मचाऱ्याला कळवले आहे.",
    ),
    "ta": _Wording(
        route="{dest} செல்லும் வழி — {m} மீ, சுமார் {mins} நிமிடம்{tail}.",
        step_free="படிக்கட்டு இல்லாத", calm="அமைதியான",
        worst="வழியில் அதிக கூட்டம்: சேவை நிலை {los}.",
        more="+{n} அருகில் உள்ளன.",
        none_mapped="இந்த அரங்கில் இன்னும் {amenity} வரைபடத்தில் இல்லை.",
        no_route="இப்போது இங்கிருந்து {amenity} செல்ல வழி எதுவும் திறந்து இல்லை{sfp}. "
                 "அருகிலுள்ள ஊழியரிடம் உதவி கேளுங்கள்.",
        step_free_required=" (படிக்கட்டு இல்லாத வழி தேவைப்பட்டது)",
        assist="{loc} இல் உங்களை நோக்கி உதவி வந்துகொண்டிருக்கிறது. ஊழியருக்கு அறிவிக்கப்பட்டது.",
    ),
}


def _words(language: str) -> _Wording:
    return _WORDING.get(language, _WORDING["en"])


def _fan_profile(accessible: bool, calm: bool) -> Profile:
    base = ACCESSIBLE if accessible else FAN
    if calm:
        # A sensory-calmer walk: hold below Fruin LOS D. Keeps step-free if asked.
        return dataclasses.replace(
            base, name=f"{base.name}+calm", max_density=CALM_MAX_DENSITY
        )
    return base


def _matching_nodes(venue_id: str, tags: Sequence[str]) -> list[str]:
    ts = set(tags)
    return [n.id for n in profile(venue_id).venue.nodes.values() if ts <= n.tags]


def amenities_json(venue_id: str) -> dict[str, Any]:
    """The amenity catalogue, flagged with what this venue actually has mapped."""
    p = profile(venue_id)
    items = []
    for a in AMENITIES:
        available = True
        if a.kind == "tag":
            available = bool(_matching_nodes(venue_id, a.tags))
        elif a.kind == "seat":
            available = bool(p.venue.nodes_tagged("seating"))
        items.append({
            "key": a.key, "icon": a.icon, "label": a.label,
            "group": a.group, "kind": a.kind, "available": available,
        })
    return {
        "venue": venue_id,
        "groups": [{"key": k, "label": l} for k, l in GROUPS],
        "amenities": items,
    }


def wayfind(
    venue_id: str,
    from_node: str,
    amenity_key: str,
    *,
    accessible: bool = False,
    calm: bool = False,
    language: str = "en",
    seat: str | None = None,
    cordoned: Sequence[str] = (),
) -> dict[str, Any]:
    """Route an attendee to an amenity. Pure deterministic plane — no model.

    This is the friendly face of the same router the control room uses: the route
    it returns is crowd-aware (it avoids corridors above the profile's density
    limit) and, when asked, step-free and calm. A fan and a commander are served by
    the identical geometry; only the framing differs.
    """
    p, snap, a = assessment(venue_id)
    plane = _plane(venue_id)
    am = BY_KEY.get(amenity_key)
    if am is None:
        raise ValueError(f"unknown amenity {amenity_key!r}")
    if from_node not in p.venue.nodes:
        raise ValueError(f"unknown location {from_node!r}")

    base = {"amenity": am.key, "icon": am.icon, "label": am.label, "language": language}
    w = _words(language)

    if am.kind == "assist":
        # A request, not a destination. Acknowledge honestly; in a real deployment
        # this notifies stewarding rather than routing to a fixed point.
        return {
            **base, "request": True, "route": None, "destination": None, "notes": [],
            "message": w.assist.format(loc=p.venue.node(from_node).name),
        }

    if am.kind == "seat":
        candidates = [seat or p.fixture.fan_seat]
    else:
        candidates = _matching_nodes(venue_id, am.tags)

    if not candidates:
        return {
            **base, "found": False, "route": None, "destination": None, "notes": [],
            "message": w.none_mapped.format(amenity=am.label.lower()),
        }

    prof = _fan_profile(accessible, calm)
    cordon = frozenset(cordoned)
    best: Route | None = None
    reachable = 0
    for cid in candidates:
        try:
            r = plane.fan_route(a, from_node=from_node, to_node=cid, profile=prof, cordoned=cordon)
        except NoRouteError:
            continue
        reachable += 1
        if best is None or r.eta_s < best.eta_s:
            best = r

    if best is None:
        # Every candidate exists but none is reachable under the fan's constraints
        # (e.g. an accessible route is demanded and only stepped ones exist). The
        # router refuses rather than sending them somewhere unsafe.
        sfp = w.step_free_required if accessible else ""
        return {
            **base, "found": True, "route": None, "destination": None, "notes": [],
            "message": w.no_route.format(amenity=am.label.lower(), sfp=sfp),
        }

    dest = p.venue.node(best.destination)
    rj = route_json(best)
    extras = [w.step_free] if accessible else []
    if calm:
        extras.append(w.calm)
    tail = f" ({', '.join(extras)})" if extras else ""

    # Secondary lines, pre-localised so the whole card switches language, not just
    # the greeting. The LOS letter and the count are universal; the words are not.
    notes = [w.worst.format(los=rj["worst_los"])]
    if reachable > 1:
        notes.append(w.more.format(n=reachable - 1))

    return {
        **base,
        "found": True,
        "route": rj,
        "destination": {"id": dest.id, "name": dest.name, "info": dest.info, "zone": dest.zone},
        "alternatives": max(0, reachable - 1),
        "notes": notes,
        "message": w.route.format(
            dest=dest.name,
            m=round(best.distance_m),
            mins=max(1, round(best.eta_s / 60)),
            tail=tail,
        ),
    }


def stress(venue_id: str, kind: str, n: int = 3) -> dict[str, Any]:
    p = profile(venue_id)
    det = _plane(venue_id)
    harness = StressHarness(
        det,
        service_rate_per_s=p.service_rate_per_s,
        installed_lanes={g: t.installed_lanes for g, t in p.fixture.gates.items()},
    )
    sampler = SeededSampler(det, seed=5)

    results = []
    for scenario in sampler.sample(kind, n=n):
        r = harness.run(scenario)
        results.append({
            "scenario_id": r.scenario_id,
            "kind": r.kind,
            "name": scenario["name"],
            "closed_edges": list(scenario["closed_edges"]),
            "passed": r.passed,
            "findings": [
                {"invariant": f.invariant, "severity": f.severity, "detail": f.detail}
                for f in r.findings
            ],
        })
    return {"venue": venue_id, "results": results}
