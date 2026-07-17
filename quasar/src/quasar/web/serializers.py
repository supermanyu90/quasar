"""The two wire serialisers shared by every endpoint: an agent result and a route."""

from __future__ import annotations


from typing import Any

from quasar.governance import AgentResult
from quasar.routing import Route


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
