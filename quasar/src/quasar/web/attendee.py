"""The attendee companion: the concierge, amenity wayfinding, and the controlled
per-language phrasing that lets a whole reply switch language without a model."""

from __future__ import annotations

import dataclasses

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from quasar.agents import ConciergeAgent, ConciergeTask
from quasar.amenities import AMENITIES, BY_KEY, CALM_MAX_DENSITY, GROUPS
from quasar.routing import ACCESSIBLE, FAN, NoRouteError, Profile, Route

from quasar.web.core import ModelPlane, _plane, assessment, orchestrator, profile
from quasar.web.serializers import result_json, route_json


# Controlled, human-authored concierge acknowledgements. Same discipline as the
# safety announcement catalogue: authored per language, never machine-translated,
# and these phrases carry no entities to preserve. Safety-critical acknowledgements
# come ONLY from here -- the model does not speak for safety. Informational ones
# localise the deterministic fallback/transcript so a fan is greeted in their own
# language even with no model reachable; a real model's free-text reply is left as
# the model wrote it (it is never one of these fixed strings).
_CONCIERGE_ACK: Mapping[str, Mapping[str, str]] = {
    "safety_critical": {
        "en": "A steward is on the way to you.",
        "es": "Un acomodador va de camino hacia ti.",
        "fr": "Un steward est en route vers vous.",
        "hi": "एक कर्मचारी आपके पास आ रहा है।",
        "mr": "एक कर्मचारी तुमच्याकडे येत आहे.",
        "ta": "ஒரு ஊழியர் உங்களை நோக்கி வந்துகொண்டிருக்கிறார்.",
    },
    "informational": {
        "en": "Choose what you need and I will show you the way.",
        "es": "Elige lo que necesitas y te mostraré el camino.",
        "fr": "Choisissez ce dont vous avez besoin et je vous montrerai le chemin.",
        "hi": "जो चाहिए वह चुनें, मैं आपको रास्ता दिखाऊँगा।",
        "mr": "तुम्हाला काय हवे ते निवडा, मी तुम्हाला मार्ग दाखवतो.",
        "ta": "உங்களுக்கு என்ன வேண்டும் என்பதைத் தேர்ந்தெடுங்கள், நான் வழி காட்டுகிறேன்.",
    },
}


def _localise_concierge_reply(payload: Mapping[str, Any], language: str) -> str:
    """Localise a deterministic concierge acknowledgement from the controlled
    catalogue, leaving a real model's free-text reply untouched.

    Safety-critical replies are always taken from the catalogue (policy: the model
    does not speak for safety). Informational replies are localised only when they
    are the canned fallback/transcript string; anything else came from the model,
    which produced it in the fan's language already.
    """
    tier = payload.get("safety_tier")
    table = _CONCIERGE_ACK.get(tier or "")
    if table is None:
        return payload.get("reply_text", "")
    if tier == "safety_critical" or payload.get("reply_text") == table["en"]:
        return table.get(language, table["en"])
    return payload.get("reply_text", "")


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

    # Localise the acknowledgement for presentation only. result_json copies the
    # payload, so the audited record keeps the canonical decision untouched.
    res = result_json(result)
    res["payload"]["reply_text"] = _localise_concierge_reply(payload, language)

    return {
        "result": res,
        "route": route,
        "route_error": error,
        "audit": orch.audit.to_json(),
    }


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
        "groups": [{"key": k, "label": label} for k, label in GROUPS],
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
