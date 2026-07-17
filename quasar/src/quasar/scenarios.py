"""Synthetic scenario generation, and the harness that fires scenarios at the venue.

The point of generating scenarios with a model is not to make up numbers -- a
random sampler does that better and cheaper. It is to propose *situations a
planner did not think of*: the gate that fails while the VIP movement is running
and the weather hold has just pushed the whole upper tier into the concourse. The
combinatorics of a stadium's failure modes are far too large to enumerate by hand,
and the interesting ones are interactions, not single faults.

What the model proposes, the deterministic layer then has to survive.
:class:`StressHarness` takes a scenario, applies it to the venue, and checks
invariants that must hold no matter what: no spectator is routed through a closed
corridor, every seat can still reach a gate, every seat with a step-free
requirement can still reach a gate step-free, and no zone's egress time exceeds
the eight-minute ceiling in SOP-EVAC-01#2.

Findings are not test failures. They are *design* failures -- the harness is a
tool for discovering that your venue, not your code, has a problem. Running it
against the reference stadium finds a real one, and the test suite asserts that
it does.

For CI, :class:`SeededSampler` emits scenarios against the same schema without a
model, so the harness itself is testable offline. In production the generator is
the model; the harness does not know or care which produced the scenario, because
both go through the schema.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

from quasar import schemas
from quasar.crowd import ADVISORY_DENSITY
from quasar.llm import LanguageModel, ModelRequest, extract_json
from quasar.plane import DeterministicPlane
from quasar.queueing import lanes_required
from quasar.routing import ACCESSIBLE, FAN, NoRouteError, RouteRequest
from quasar.types import EdgeId, GateTelemetry, TelemetrySnapshot

# SOP-EVAC-01#2.
MAX_EGRESS_S: float = 480.0

SCENARIO_KINDS = (
    "gate_failure",
    "weather_evacuation",
    "vip_movement",
    "medical_surge",
    "power_loss",
    "pitch_invasion",
)


@dataclass(frozen=True, slots=True)
class Finding:
    invariant: str
    severity: str  # "critical" | "major" | "minor"
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.invariant}: {self.detail}"


@dataclass(frozen=True, slots=True)
class StressResult:
    scenario_id: str
    kind: str
    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        return not any(f.severity == "critical" for f in self.findings)


# ==========================================================================
# Generation
# ==========================================================================

_GENERATOR_SYSTEM = """\
You generate adversarial match-day scenarios for a 60,000-seat stadium, to be
fired at its crowd-safety software before the gates open.

You are not trying to produce plausible averages. You are trying to find the
combination of conditions the venue's planners did not consider: two faults that
interact, a mitigation that becomes the problem, a corridor that is safe alone and
lethal when it is the only one open.

Emit one scenario as JSON conforming to the schema. Densities are ped/m2 and must
be physically possible (a corridor cannot hold more than about 5.4 people per
square metre). Closed edges are corridors taken out of service by the scenario.
Gate overrides set arrival rates and open lane counts.

Explain nothing. Emit the object.
"""


class ScenarioGenerator:
    """Model-driven scenario proposal. Output goes through the schema like anything else."""

    schema_id = schemas.SCENARIO

    def __init__(self, model: LanguageModel, plane: DeterministicPlane) -> None:
        self._model = model
        self._plane = plane

    def propose(self, kind: str, *, seed_note: str = "") -> Mapping[str, Any]:
        if kind not in SCENARIO_KINDS:
            raise ValueError(f"unknown scenario kind {kind!r}")
        venue = self._plane.venue
        request = ModelRequest(
            system=_GENERATOR_SYSTEM,
            user="\n".join([
                f"kind: {kind}",
                f"corridors: {', '.join(sorted(venue.edges))}",
                f"gates: {', '.join(sorted(n.id for n in venue.nodes_tagged('gate')))}",
                seed_note,
            ]),
            schema_id=self.schema_id,
            effort="high",
        )
        response = self._model.complete(request)  # ModelUnavailable propagates
        payload = extract_json(response.text)
        schemas.validate(payload, self.schema_id)
        self._check_references(payload)
        return payload

    def _check_references(self, payload: Mapping[str, Any]) -> None:
        venue = self._plane.venue
        for edge_id in list(payload["edge_density"]) + list(payload["closed_edges"]):
            if edge_id not in venue.edges:
                raise ValueError(f"scenario references unknown corridor {edge_id!r}")
        gates = {n.id for n in venue.nodes_tagged("gate")}
        for override in payload["gate_overrides"]:
            if override["gate_id"] not in gates:
                raise ValueError(f"scenario references unknown gate {override['gate_id']!r}")


class SeededSampler:
    """Deterministic scenario source for CI.

    Emits the same schema as the model so the harness is exercised identically.
    It is a *sampler*, not a simulation of the model: it will never find the
    interaction cases that are the whole reason to use a model here, and it is not
    pretending to.
    """

    def __init__(self, plane: DeterministicPlane, *, seed: int = 7) -> None:
        self._plane = plane
        self._rng = random.Random(seed)

    def sample(self, kind: str, n: int = 1) -> Iterator[Mapping[str, Any]]:
        venue = self._plane.venue
        fan_edges = sorted(e.id for e in venue.edges.values() if not e.staff_only)
        gates = sorted(n.id for n in venue.nodes_tagged("gate"))

        for i in range(n):
            hot = self._rng.sample(fan_edges, k=min(4, len(fan_edges)))
            closed = (
                [self._rng.choice(fan_edges)]
                if kind in ("gate_failure", "power_loss", "pitch_invasion")
                else []
            )
            payload = {
                "schema": schemas.SCENARIO,
                "scenario_id": f"SEED-{kind}-{i}",
                "name": f"seeded {kind.replace('_', ' ')} #{i}",
                "kind": kind,
                "edge_density": {
                    e: round(self._rng.uniform(ADVISORY_DENSITY, 4.6), 2) for e in hot
                },
                "closed_edges": closed,
                "gate_overrides": [
                    {
                        "gate_id": g,
                        "arrival_rate_per_s": round(self._rng.uniform(1.0, 9.0), 2),
                        "open_lanes": self._rng.randint(2, 8),
                    }
                    for g in self._rng.sample(gates, k=2)
                ],
            }
            schemas.validate(payload, schemas.SCENARIO)
            yield payload


# ==========================================================================
# The harness
# ==========================================================================


class StressHarness:
    """Fires a scenario at the deterministic layer and checks the invariants."""

    def __init__(
        self,
        plane: DeterministicPlane,
        *,
        service_rate_per_s: float = 0.55,
        installed_lanes: Mapping[str, int] | None = None,
        default_installed_lanes: int = 12,
    ) -> None:
        self._plane = plane
        self._mu = service_rate_per_s
        # How many lanes a gate ACTUALLY has is a fact about the building, and it
        # differs per venue. Assuming 12 everywhere told a 8-lane arena it needed a
        # diversion it did not need, and would have told a 20-lane stadium it was
        # fine when it was not.
        self._installed = dict(installed_lanes or {})
        self._default_installed = default_installed_lanes

    def _installed_at(self, gate_id: str) -> int:
        return self._installed.get(gate_id, self._default_installed)

    def run(self, scenario: Mapping[str, Any]) -> StressResult:
        """Apply a scenario and check every invariant. Reads as its checklist."""
        schemas.validate(scenario, schemas.SCENARIO)
        snapshot = self._snapshot(scenario)
        assessment = self._plane.assess(snapshot)
        closed = frozenset(scenario["closed_edges"])
        density = dict(snapshot.edge_density)

        findings: list[Finding] = []
        for seat in self._plane.venue.nodes_tagged("seating"):
            findings += self._check_seat_egress(seat, closed, density)
        findings += self._check_gate_mitigation(assessment, snapshot)
        findings += self._check_single_point_corridors(assessment, closed, density)

        return StressResult(
            scenario_id=scenario["scenario_id"],
            kind=scenario["kind"],
            findings=tuple(findings),
        )

    def _snapshot(self, scenario: Mapping[str, Any]) -> TelemetrySnapshot:
        venue = self._plane.venue
        return TelemetrySnapshot(
            t=0.0,
            edge_density={e: float(scenario["edge_density"].get(e, 0.0)) for e in venue.edges},
            gates={
                o["gate_id"]: GateTelemetry(
                    gate_id=o["gate_id"],
                    arrival_rate_per_s=float(o["arrival_rate_per_s"]),
                    service_rate_per_s=self._mu,
                    open_lanes=max(1, min(int(o["open_lanes"]), self._installed_at(o["gate_id"]))),
                    installed_lanes=self._installed_at(o["gate_id"]),
                )
                for o in scenario["gate_overrides"]
            },
        )

    def _check_seat_egress(
        self, seat: Any, closed: frozenset[EdgeId], density: Mapping[EdgeId, float]
    ) -> list[Finding]:
        """Can this stand get out -- on foot, and step-free?"""
        plane = self._plane
        gates = plane.venue.nodes_tagged("gate")
        findings: list[Finding] = []

        # I1/I2 -- an ambulatory spectator reaches some gate, by a route that never
        # crosses a closed corridor, within the eight-minute ceiling.
        best = None
        for gate in gates:
            try:
                route = plane.router.route(
                    RouteRequest(seat.id, gate.id, FAN, cordoned_edges=closed), density
                )
            except NoRouteError:
                continue
            if any(e in closed for e in route.edges):
                findings.append(Finding(
                    "no-route-through-closure", "critical",
                    f"{seat.id} -> {gate.id} was routed through a closed corridor",
                ))
            if best is None or route.eta_s < best.eta_s:
                best = route

        if best is None:
            findings.append(Finding(
                "egress-exists", "critical",
                f"{seat.id} has no route to any gate under this scenario",
            ))
        elif best.eta_s > MAX_EGRESS_S:
            findings.append(Finding(
                "egress-time", "major",
                f"{seat.id} egress is {best.eta_s / 60:.1f} min to {best.destination}, "
                f"above the {MAX_EGRESS_S / 60:.0f} min ceiling (SOP-EVAC-01#2)",
            ))

        # I3 -- and so can a spectator who cannot use stairs. Three very different
        # failures hide behind "no step-free route", and each has a different remedy
        # and a different owner, so we distinguish them rather than send the wrong
        # person to fix the wrong thing:
        #   * none in an empty, fully-open venue -> the BUILDING is wrong (architect);
        #   * exists normally but this scenario's CLOSURES remove it -> the closure
        #     plan is wrong (operations);
        #   * exists and open but too CROWDED for assisted mobility -> hold the route
        #     clear (stewarding, solvable tonight).
        empty = self._empty(plane.venue)
        baseline = self._step_free_route(seat.id, gates, empty, frozenset())
        after_closure = self._step_free_route(seat.id, gates, empty, closed)
        under_load = self._step_free_route(seat.id, gates, density, closed)

        if baseline is None:
            findings.append(Finding(
                "step-free-egress-exists", "critical",
                f"{seat.id} has no step-free route to any gate in an empty, fully open "
                "venue. This is a defect in the building, not the crowd: a staffed "
                "refuge point is mandatory here before the fixture (SOP-EVAC-01#3)",
            ))
        elif after_closure is None:
            findings.append(Finding(
                "step-free-egress-after-closure", "critical",
                f"{seat.id} normally has a step-free route to {baseline.destination}, "
                f"but this scenario's closures ({', '.join(sorted(closed))}) remove "
                "every one. That closure may not be applied without standing up a "
                "refuge point first (SOP-EVAC-01#3)",
            ))
        elif under_load is None:
            findings.append(Finding(
                "step-free-egress-under-load", "major",
                f"{seat.id} has a step-free route ({after_closure.destination}) but every "
                "such route is above the assisted-mobility density limit under this "
                "scenario; the accessible route must be held clear",
            ))
        return findings

    def _check_gate_mitigation(self, assessment: Any, snapshot: TelemetrySnapshot) -> list[Finding]:
        """I4 -- every saturated gate has a mitigation within its installed lanes."""
        findings: list[Finding] = []
        for gid, metrics in assessment.gates.items():
            if not metrics.breaches_trigger:
                continue
            telemetry = snapshot.gates[gid]
            try:
                needed = lanes_required(telemetry)
            except RuntimeError:
                findings.append(Finding(
                    "gate-mitigable", "critical",
                    f"{gid} is saturated and no lane count can hold the trigger; arrivals "
                    "must be diverted",
                ))
                continue
            if needed > telemetry.installed_lanes:
                findings.append(Finding(
                    "gate-mitigable", "major",
                    f"{gid} needs {needed} lanes to hold the 0.90 trigger but only "
                    f"{telemetry.installed_lanes} are installed; a diversion is mandatory",
                ))
        return findings

    def _check_single_point_corridors(
        self, assessment: Any, closed: frozenset[EdgeId], density: Mapping[EdgeId, float]
    ) -> list[Finding]:
        """I5 -- a LOS-F corridor with no parallel path is a single point of failure.

        Spurs to dead ends (the VIP box, lost property, a gate) are excluded: "the
        only path to a cul-de-sac is the only path to a cul-de-sac" is a tautology,
        not a finding, and a harness that reports tautologies stops being read. What
        matters is severing the *ring* -- disconnecting one part of the concourse.
        """
        venue = self._plane.venue
        findings: list[Finding] = []
        for edge_id in assessment.critical_edges:
            if edge_id in closed:
                continue
            edge = venue.edge(edge_id)
            if self._is_spur(edge.u) or self._is_spur(edge.v):
                continue
            try:
                self._plane.router.route(
                    RouteRequest(edge.u, edge.v, FAN, cordoned_edges=closed | {edge_id}),
                    density,
                )
            except NoRouteError:
                findings.append(Finding(
                    "critical-corridor-has-alternative", "major",
                    f"{edge_id} is at LOS F and is the only spectator path between "
                    f"{edge.u} and {edge.v}; cordoning it severs the concourse",
                ))
        return findings

    # -- helpers ----------------------------------------------------------

    def _is_spur(self, node_id: str) -> bool:
        """A node reachable by exactly one spectator-usable edge is a dead end."""
        venue = self._plane.venue
        fan_edges = [
            eid
            for _, eid in venue.neighbours(node_id)
            if not venue.edge(eid).staff_only
        ]
        return len(fan_edges) <= 1

    def _empty(self, venue: Any) -> dict[EdgeId, float]:
        return {e: 0.0 for e in venue.edges}

    def _step_free_route(
        self,
        origin: str,
        gates: Sequence[Any],
        density: Mapping[EdgeId, float],
        closed: frozenset[EdgeId],
    ):
        best = None
        for gate in gates:
            try:
                route = self._plane.router.route(
                    RouteRequest(
                        origin=origin,
                        destination=gate.id,
                        profile=ACCESSIBLE,
                        cordoned_edges=closed,
                    ),
                    density,
                )
            except NoRouteError:
                continue
            if best is None or route.eta_s < best.eta_s:
                best = route
        return best

    def run_all(self, scenarios: Sequence[Mapping[str, Any]]) -> tuple[StressResult, ...]:
        return tuple(self.run(s) for s in scenarios)
