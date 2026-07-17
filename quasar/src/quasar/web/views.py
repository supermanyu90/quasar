"""Read-only views: the JSON the console renders (venues, state, readiness, guide)
and the pre-match stress report."""

from __future__ import annotations


from typing import Any

from quasar import alignment
from quasar.llm import DisabledModel
from quasar.readiness import audit
from quasar.scenarios import SeededSampler, StressHarness

from quasar.web.core import DEFAULT_VENUE, ModelPlane, VENUES, _plane, assessment, orchestrator, profile


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
