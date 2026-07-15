"""Core domain types shared by Quasar's deterministic and generative planes.

Every value that crosses a plane boundary is one of these types or a JSON payload
validated against a published schema (see :mod:`quasar.schemas`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

NodeId = str
EdgeId = str
ZoneId = str
GateId = str
LangCode = str

# JSON payloads exchanged with the model plane. Deliberately loose at the type
# level and tightened at runtime by the schema validator, which is the only
# trust boundary that matters for these values.
Json = object
JsonObject = Mapping[str, object]


class Severity(str, Enum):
    """Incident severity. P0/P1 actions may not actuate without operator sign-off."""

    P0 = "P0"  # life-safety, venue-wide (evacuation, structural, active threat)
    P1 = "P1"  # life-safety, localised (medical emergency, crush risk, fire alarm)
    P2 = "P2"  # service-affecting (gate saturation, queue breach, lost child)
    P3 = "P3"  # informational (wayfinding, amenity, lost property)

    @property
    def requires_human_approval(self) -> bool:
        return self in (Severity.P0, Severity.P1)


class LOS(str, Enum):
    """Fruin level of service for walkways, derived from pedestrian density."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class Role(str, Enum):
    FAN = "fan"
    VOLUNTEER = "volunteer"
    STEWARD = "steward"
    MEDIC = "medic"
    SECURITY = "security"
    COMMANDER = "commander"  # only role that may approve P0/P1 actuation


class EdgeKind(str, Enum):
    CONCOURSE = "concourse"
    CORRIDOR = "corridor"
    STAIR = "stair"
    RAMP = "ramp"
    TUNNEL = "tunnel"
    VOMITORY = "vomitory"
    SERVICE = "service"  # staff-only; fans are never routed through these


@dataclass(frozen=True, slots=True)
class Node:
    id: NodeId
    name: str
    x: float  # metres, venue-local grid
    y: float
    level: int
    zone: ZoneId
    tags: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class Edge:
    id: EdgeId
    u: NodeId
    v: NodeId
    length_m: float
    width_m: float
    kind: EdgeKind
    step_free: bool
    staff_only: bool = False

    @property
    def area_m2(self) -> float:
        return self.length_m * self.width_m


@dataclass(frozen=True, slots=True)
class Beacon:
    id: str
    node: NodeId
    x: float
    y: float
    level: int
    tx_power_dbm: float = -45.0  # calibrated RSSI at 1 m


@dataclass(frozen=True, slots=True)
class GateTelemetry:
    """Turnstile bank modelled as an M/M/c queue."""

    gate_id: GateId
    arrival_rate_per_s: float  # lambda
    service_rate_per_s: float  # mu, per lane
    open_lanes: int  # c
    installed_lanes: int


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    """One fused sensing frame: CV crowd counters, turnstile logs, BLE gateways."""

    t: float
    edge_density: Mapping[EdgeId, float]  # ped / m^2
    gates: Mapping[GateId, GateTelemetry]
    weather: str = "clear"
    source_ids: Sequence[str] = ()

    def density(self, edge_id: EdgeId) -> float:
        return self.edge_density.get(edge_id, 0.0)


@dataclass(frozen=True, slots=True)
class IncidentReport:
    """Raw report from a human (volunteer voice note, steward radio, fan app)."""

    id: str
    reporter_role: Role
    at_node: NodeId
    text: str
    language: LangCode = "en"
    t: float = field(default_factory=time.time)


@dataclass(frozen=True, slots=True)
class Operator:
    id: str
    name: str
    roles: frozenset[Role]

    def may_approve(self, severity: Severity) -> bool:
        if severity.requires_human_approval:
            return Role.COMMANDER in self.roles
        return bool(self.roles)


@dataclass(frozen=True, slots=True)
class Approval:
    """A signed operator decision. Produced only by a human, never by an agent."""

    operator: Operator
    plan_id: str
    approved: bool
    t: float
    note: str = ""
