"""M/M/c queueing for turnstile banks.

Utilisation is rho_q = lambda / (c * mu). The Erlang-C formula gives the
probability an arriving fan has to wait; from it we get expected queue length
and expected wait. The 0.90 utilisation trigger in ``REROUTE_TRIGGER`` is a
*policy* threshold, not a stability threshold: an M/M/c queue is stable for any
rho_q < 1, but wait time grows without bound as rho_q -> 1, so operations shape
flow well before that.

The Erlang-C evaluation uses the numerically stable recursion for the Erlang-B
loss probability rather than the factorial form, which overflows for the lane
counts and offered loads a stadium gate actually sees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from quasar.types import GateTelemetry

# Policy trigger: at or above this utilisation the deterministic plane demands
# a reallocation proposal.
REROUTE_TRIGGER: float = 0.90

# Service-level objective for gate ingress.
TARGET_WAIT_S: float = 180.0


@dataclass(frozen=True, slots=True)
class QueueMetrics:
    gate_id: str
    offered_load: float  # a = lambda / mu (erlangs)
    utilisation: float  # rho_q = a / c
    prob_wait: float  # Erlang C
    queue_length: float  # Lq
    wait_s: float  # Wq
    stable: bool

    @property
    def breaches_trigger(self) -> bool:
        return (not self.stable) or self.utilisation >= REROUTE_TRIGGER

    @property
    def breaches_slo(self) -> bool:
        return (not self.stable) or self.wait_s > TARGET_WAIT_S


def erlang_b(c: int, a: float) -> float:
    """Erlang-B blocking probability via the stable recursion B(0,a) = 1."""
    if c < 0:
        raise ValueError("server count must be non-negative")
    if a < 0.0:
        raise ValueError("offered load must be non-negative")
    b = 1.0
    for n in range(1, c + 1):
        b = (a * b) / (n + a * b)
    return b


def erlang_c(c: int, a: float) -> float:
    """Probability that an arriving customer must queue (Erlang C).

    Derived from Erlang B: C = B / (1 - rho*(1 - B)) with rho = a/c.
    Returns 1.0 for an unstable queue (a >= c), where every arrival eventually waits.
    """
    if c <= 0:
        raise ValueError("open lanes must be positive")
    rho = a / c
    if rho >= 1.0:
        return 1.0
    b = erlang_b(c, a)
    return b / (1.0 - rho * (1.0 - b))


def analyse_gate(t: GateTelemetry) -> QueueMetrics:
    """Full M/M/c metric set for one turnstile bank."""
    if t.service_rate_per_s <= 0.0:
        raise ValueError(f"gate {t.gate_id}: service rate must be positive")
    if t.open_lanes <= 0:
        raise ValueError(f"gate {t.gate_id}: at least one lane must be open")
    if t.open_lanes > t.installed_lanes:
        raise ValueError(f"gate {t.gate_id}: cannot open more lanes than installed")

    a = t.arrival_rate_per_s / t.service_rate_per_s
    rho = a / t.open_lanes
    stable = rho < 1.0
    p_wait = erlang_c(t.open_lanes, a)

    if stable:
        lq = p_wait * rho / (1.0 - rho)
        wq = lq / t.arrival_rate_per_s if t.arrival_rate_per_s > 0.0 else 0.0
    else:
        lq = math.inf
        wq = math.inf

    return QueueMetrics(
        gate_id=t.gate_id,
        offered_load=a,
        utilisation=rho,
        prob_wait=p_wait,
        queue_length=lq,
        wait_s=wq,
        stable=stable,
    )


def lanes_required(
    t: GateTelemetry,
    *,
    target_utilisation: float = REROUTE_TRIGGER,
    target_wait_s: float = TARGET_WAIT_S,
) -> int:
    """Smallest lane count meeting both the utilisation and wait targets.

    Returns a value that may exceed ``installed_lanes``; the caller decides
    whether that means "open every lane" or "divert arrivals to another gate".
    """
    a = t.arrival_rate_per_s / t.service_rate_per_s
    c = max(1, math.ceil(a / target_utilisation))
    while c < 512:
        probe = GateTelemetry(
            gate_id=t.gate_id,
            arrival_rate_per_s=t.arrival_rate_per_s,
            service_rate_per_s=t.service_rate_per_s,
            open_lanes=c,
            installed_lanes=max(c, t.installed_lanes),
        )
        m = analyse_gate(probe)
        if m.stable and m.utilisation < target_utilisation and m.wait_s <= target_wait_s:
            return c
        c += 1
    raise RuntimeError(f"gate {t.gate_id}: no lane count within bounds meets the target")


def divertible_arrivals(t: GateTelemetry, *, target_utilisation: float = REROUTE_TRIGGER) -> float:
    """Arrival rate (ped/s) that must be sent elsewhere to hold the target
    utilisation with the lanes currently installed."""
    capacity = target_utilisation * t.installed_lanes * t.service_rate_per_s
    return max(0.0, t.arrival_rate_per_s - capacity)
