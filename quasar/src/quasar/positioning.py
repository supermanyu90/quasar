"""Indoor positioning: a graph-constrained particle filter over BLE / Wi-Fi RTT.

GPS does not work under a stadium roof, and raw RSSI trilateration in a hall
full of bodies is worth roughly +/- 8 m -- enough to send a fan to the wrong
vomitory, and far too coarse to say which side of a cordon a casualty is on.

The fix is to stop estimating a free-space position at all. A pedestrian inside
a venue is always *on a walkable edge*, so we estimate a position **on the
graph**: each particle is a point on an edge, motion is dead reckoning along
edges with transitions only at real junctions, and the RSSI likelihood only ever
scores positions that are physically walkable. Map-matching is therefore not a
post-processing step that can be wrong; it is the state space.

This is the deterministic substrate the conversational wayfinding in
:mod:`quasar.agents` sits on top of. The concierge decides *where the fan wants
to go*; this decides *where the fan is*. Neither is a language problem.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from quasar.types import EdgeId, NodeId
from quasar.venue import Venue

# Log-distance path-loss model, calibrated for a concrete-and-steel concourse
# with a dense human absorber. n = 2.0 is free space; crowds push it up.
PATH_LOSS_EXPONENT: float = 2.2
RSSI_SIGMA_DB: float = 4.0
REF_DISTANCE_M: float = 1.0
LEVEL_SEPARATION_M: float = 4.0


@dataclass(frozen=True, slots=True)
class RssiReading:
    beacon_id: str
    rssi_dbm: float


@dataclass(frozen=True, slots=True)
class Fix:
    """A map-matched position estimate."""

    x: float
    y: float
    level: int
    edge_id: EdgeId
    offset: float  # 0..1 along edge.u -> edge.v
    nearest_node: NodeId
    confidence: float  # share of particle weight on the MAP edge, 0..1

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass(slots=True)
class _Particle:
    edge_id: EdgeId
    offset: float  # 0..1 along u -> v
    heading: int  # +1 travelling u -> v, -1 travelling v -> u
    weight: float


def expected_rssi(distance_m: float, tx_power_dbm: float) -> float:
    d = max(distance_m, REF_DISTANCE_M)
    return tx_power_dbm - 10.0 * PATH_LOSS_EXPONENT * math.log10(d / REF_DISTANCE_M)


class GraphParticleFilter:
    """Particle filter whose state space is the venue's walkable graph."""

    def __init__(
        self,
        venue: Venue,
        *,
        n_particles: int = 600,
        seed: int = 0,
        step_noise: float = 0.15,
    ) -> None:
        if n_particles < 32:
            raise ValueError("particle count too small for a stable estimate")
        self._venue = venue
        self._n = n_particles
        self._rng = random.Random(seed)
        self._step_noise = step_noise
        self._particles: list[_Particle] = []

    # -- initialisation ---------------------------------------------------

    def seed_uniform(self, *, walkable_only: bool = True) -> None:
        """No prior: spread particles over every walkable edge."""
        edges = [
            e for e in self._venue.edges.values() if not (walkable_only and e.staff_only)
        ]
        if not edges:
            raise ValueError("venue has no walkable edges")
        w = 1.0 / self._n
        self._particles = [
            _Particle(
                edge_id=self._rng.choice(edges).id,
                offset=self._rng.random(),
                heading=self._rng.choice((-1, 1)),
                weight=w,
            )
            for _ in range(self._n)
        ]

    def seed_at_node(self, node_id: NodeId) -> None:
        """Strong prior: the fan just tapped in at a known gate."""
        incident = [eid for _, eid in self._venue.neighbours(node_id)]
        if not incident:
            raise ValueError(f"node {node_id} has no incident edges")
        w = 1.0 / self._n
        particles: list[_Particle] = []
        for _ in range(self._n):
            eid = self._rng.choice(incident)
            edge = self._venue.edge(eid)
            at_u = edge.u == node_id
            particles.append(
                _Particle(
                    edge_id=eid,
                    offset=0.0 if at_u else 1.0,
                    heading=1 if at_u else -1,
                    weight=w,
                )
            )
        self._particles = particles

    # -- prediction -------------------------------------------------------

    def predict(self, step_m: float) -> None:
        """Advance every particle by a dead-reckoned step with process noise.

        ``step_m`` comes from the handset's pedometer (step count x calibrated
        stride), not from any model output.
        """
        if step_m < 0.0:
            raise ValueError("step must be non-negative")
        for p in self._particles:
            remaining = max(0.0, step_m * (1.0 + self._rng.gauss(0.0, self._step_noise)))
            self._advance(p, remaining)

    def _advance(self, p: _Particle, remaining: float) -> None:
        # Walk the particle forward, hopping to a new edge at each junction it
        # reaches. Bounded to keep a pathological step from looping forever.
        for _ in range(8):
            edge = self._venue.edge(p.edge_id)
            if edge.length_m <= 0.0:
                return
            travelled = p.offset * edge.length_m
            room = (edge.length_m - travelled) if p.heading > 0 else travelled
            if remaining <= room:
                delta = remaining / edge.length_m
                p.offset = min(1.0, max(0.0, p.offset + p.heading * delta))
                return
            remaining -= room
            junction = edge.v if p.heading > 0 else edge.u
            arrival_bearing = self._bearing(edge.u, edge.v) if p.heading > 0 else self._bearing(edge.v, edge.u)
            self._transition(p, junction, arrival_bearing)

    def _transition(self, p: _Particle, junction: NodeId, arrival_bearing: float) -> None:
        """Choose an outgoing edge at a junction, favouring going straight on.

        Pedestrians rarely reverse mid-corridor; weighting by heading alignment
        is what keeps the filter from smearing backwards along the path already
        walked.
        """
        options = self._venue.neighbours(junction)
        if not options:
            p.heading *= -1  # dead end: turn around
            return

        weights: list[float] = []
        for neighbour, _eid in options:
            bearing = self._bearing(junction, neighbour)
            turn = abs(_wrap_pi(bearing - arrival_bearing))
            # 1.0 straight ahead, ~0.02 doubling back.
            weights.append(math.exp(-2.0 * turn) + 0.02)

        chosen_neighbour, chosen_edge = self._rng.choices(options, weights=weights, k=1)[0]
        edge = self._venue.edge(chosen_edge)
        p.edge_id = chosen_edge
        if edge.u == junction:
            p.offset, p.heading = 0.0, 1
        else:
            p.offset, p.heading = 1.0, -1
        del chosen_neighbour

    def _bearing(self, a: NodeId, b: NodeId) -> float:
        na, nb = self._venue.node(a), self._venue.node(b)
        return math.atan2(nb.y - na.y, nb.x - na.x)

    # -- measurement update -----------------------------------------------

    def update(self, readings: Sequence[RssiReading]) -> None:
        """Reweight particles by BLE likelihood and resample if degenerate."""
        if not readings:
            return
        beacons = self._venue.beacons

        total = 0.0
        for p in self._particles:
            x, y, level = self._position(p)
            log_lik = 0.0
            for r in readings:
                b = beacons.get(r.beacon_id)
                if b is None:
                    continue  # unknown beacon: ignore rather than guess
                dz = abs(b.level - level) * LEVEL_SEPARATION_M
                d = math.sqrt((b.x - x) ** 2 + (b.y - y) ** 2 + dz * dz)
                residual = r.rssi_dbm - expected_rssi(d, b.tx_power_dbm)
                log_lik += -0.5 * (residual / RSSI_SIGMA_DB) ** 2
            p.weight *= math.exp(max(log_lik, -700.0))
            total += p.weight

        if total <= 0.0:
            # Every particle is impossible under the measurement: the prior was
            # wrong (fan teleported by lift, or beacons remapped). Reset rather
            # than propagate a confidently wrong fix.
            self.seed_uniform()
            return

        for p in self._particles:
            p.weight /= total

        ess = 1.0 / sum(p.weight**2 for p in self._particles)
        if ess < self._n / 2.0:
            self._resample()

    def _resample(self) -> None:
        """Systematic resampling: lower variance than multinomial, O(n)."""
        step = 1.0 / self._n
        u = self._rng.random() * step
        cumulative = 0.0
        source = iter(self._particles)
        current = next(source)
        cumulative += current.weight
        resampled: list[_Particle] = []
        for i in range(self._n):
            threshold = u + i * step
            while threshold > cumulative:
                try:
                    current = next(source)
                except StopIteration:
                    break
                cumulative += current.weight
            resampled.append(
                _Particle(current.edge_id, current.offset, current.heading, step)
            )
        self._particles = resampled

    # -- estimation -------------------------------------------------------

    def _position(self, p: _Particle) -> tuple[float, float, int]:
        edge = self._venue.edge(p.edge_id)
        u, v = self._venue.node(edge.u), self._venue.node(edge.v)
        x = u.x + (v.x - u.x) * p.offset
        y = u.y + (v.y - u.y) * p.offset
        level = u.level if p.offset < 0.5 else v.level
        return x, y, level

    def estimate(self) -> Fix:
        """Weighted-mean position, map-matched to the modal edge."""
        if not self._particles:
            raise RuntimeError("filter has not been seeded")

        by_edge: dict[EdgeId, float] = {}
        for p in self._particles:
            by_edge[p.edge_id] = by_edge.get(p.edge_id, 0.0) + p.weight
        map_edge = max(by_edge, key=lambda e: by_edge[e])
        confidence = by_edge[map_edge]

        # Average only the particles on the MAP edge: averaging across edges
        # would place the fix in a wall.
        on_edge = [p for p in self._particles if p.edge_id == map_edge]
        w = sum(p.weight for p in on_edge) or 1.0
        offset = sum(p.offset * p.weight for p in on_edge) / w

        edge = self._venue.edge(map_edge)
        u, v = self._venue.node(edge.u), self._venue.node(edge.v)
        x = u.x + (v.x - u.x) * offset
        y = u.y + (v.y - u.y) * offset
        nearest = edge.u if offset < 0.5 else edge.v

        return Fix(
            x=x,
            y=y,
            level=u.level if offset < 0.5 else v.level,
            edge_id=map_edge,
            offset=offset,
            nearest_node=nearest,
            confidence=confidence,
        )


def simulate_readings(
    venue: Venue,
    x: float,
    y: float,
    level: int,
    *,
    rng: random.Random,
    max_beacons: int = 4,
    noise_db: float = RSSI_SIGMA_DB,
) -> list[RssiReading]:
    """Synthesise the RSSI vector a handset at (x, y, level) would report.

    Used by the test suite and the scenario harness to exercise the filter
    without a physical venue. It is *not* used at runtime -- production readings
    come from the handset SDK.
    """
    scored: list[tuple[float, RssiReading]] = []
    for b in venue.beacons.values():
        dz = abs(b.level - level) * LEVEL_SEPARATION_M
        d = math.sqrt((b.x - x) ** 2 + (b.y - y) ** 2 + dz * dz)
        rssi = expected_rssi(d, b.tx_power_dbm) + rng.gauss(0.0, noise_db)
        scored.append((d, RssiReading(beacon_id=b.id, rssi_dbm=rssi)))
    scored.sort(key=lambda item: item[0])
    return [r for _, r in scored[:max_beacons]]


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
