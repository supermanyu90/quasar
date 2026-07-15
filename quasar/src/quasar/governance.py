"""The governance barrier. Nothing generative reaches the venue except through here.

Every model payload runs the same gauntlet, in this order:

1. **transport** -- if the model is unreachable, fail over to the edge model; if
   that is gone too, take the deterministic path. Never invent.
2. **parse** -- extract JSON. One repair attempt, with the parser's complaint fed
   back. Then the deterministic path.
3. **schema** -- validate against the published contract (:mod:`quasar.schemas`).
   One repair attempt, with the full violation list fed back. Then the
   deterministic path.
4. **grounding** -- every cited SOP section must exist *and* must have been in the
   retrieved context. A brief that cites a section it was never shown is a
   hallucination wearing a citation, and it is rejected.
5. **corroboration** -- a deterministic check of the payload against ground truth
   the model did not produce: densities it restated, gates it named, routes it
   assumed, the severity floor procedure sets. A fatal failure here forces the
   deterministic path regardless of how confident the model claims to be.
6. **confidence** -- gate on ``min(self_reported, corroboration_score)`` against
   ``CONFIDENCE_FLOOR``. Self-reported confidence alone is a generated token and
   gating on it would be theatre; the corroboration score is the load-bearing half.
7. **policy** -- action allowlist, blast-radius caps, and role authority.
8. **human** -- P0 and P1 actions do not actuate without a commander's signature.

Then, and only then, the *deterministic plane* executes the action, recomputing
every number from the graph. The model's plan is a selection among priced
options; it is never itself the price.

The audit log is a hash chain. Each record commits to its predecessor, so a
record cannot be altered or removed after the fact without breaking every record
that follows -- which matters, because the first thing anyone will want after an
incident is to know exactly what the system proposed, what a human approved, and
in what order.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

from quasar import schemas
from quasar.agents import Agent, Corroboration
from quasar.language import CATALOGUE, Dispatch, MessageCatalogue
from quasar.llm import (
    LanguageModel,
    ModelRequest,
    ModelResponse,
    ModelUnavailable,
    extract_json,
)
from quasar.plane import Assessment, DeterministicPlane
from quasar.rag import Retriever, check_grounding
from quasar.routing import NoRouteError, Route
from quasar.schemas import SchemaError
from quasar.venue import public_labels
from quasar.types import (
    Approval,
    EdgeId,
    GateId,
    GateTelemetry,
    LangCode,
    Operator,
    Severity,
    TelemetrySnapshot,
)

# A payload whose effective confidence falls below this floor is discarded in
# favour of the deterministic implementation. Mandated by the specification;
# what the specification does not say -- and what makes it mean anything -- is
# what "confidence" is measured against. See CONFIDENCE, below.
CONFIDENCE_FLOOR: float = 0.85

Source = Literal["model", "model_repaired", "deterministic_fallback"]


# ==========================================================================
# Audit
# ==========================================================================


@dataclass(frozen=True, slots=True)
class AuditRecord:
    seq: int
    t: float
    event: str
    data: Mapping[str, Any]
    prev_hash: str
    hash: str


def _normalise(value: Any) -> Any:
    """Collapse integral floats to ints before hashing.

    JavaScript has exactly one number type. `JSON.stringify(1.0)` emits `1`, so a
    corroboration score of exactly 1.0 leaves Python as `1.0`, comes back from the
    browser as `1`, and the canonical bytes — and therefore the SHA-256 — differ.
    The chain would report tampering when nobody had tampered: a false positive on
    an integrity check, which is the fastest way to teach people to ignore it.

    Normalising here makes the digest invariant under a round trip through any
    JSON implementation that cannot distinguish 1 from 1.0, which is most of them.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value == int(value) else value
    if isinstance(value, dict):
        return {k: _normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    return value


def _canonical(obj: Any) -> str:
    return json.dumps(_normalise(obj), sort_keys=True, separators=(",", ":"), default=str)


class AuditLog:
    """Append-only, tamper-evident record of every decision the system took."""

    GENESIS = "0" * 64

    def __init__(self) -> None:
        self._records: list[AuditRecord] = []

    @property
    def records(self) -> tuple[AuditRecord, ...]:
        return tuple(self._records)

    def append(self, event: str, data: Mapping[str, Any]) -> AuditRecord:
        prev = self._records[-1].hash if self._records else self.GENESIS
        seq = len(self._records)
        t = time.time()
        digest = hashlib.sha256(
            f"{prev}|{seq}|{event}|{_canonical(data)}".encode()
        ).hexdigest()
        record = AuditRecord(seq=seq, t=t, event=event, data=dict(data), prev_hash=prev, hash=digest)
        self._records.append(record)
        return record

    def verify(self) -> bool:
        """Recompute the chain. False if any record was altered, inserted or removed."""
        prev = self.GENESIS
        for i, r in enumerate(self._records):
            if r.seq != i or r.prev_hash != prev:
                return False
            expected = hashlib.sha256(
                f"{r.prev_hash}|{r.seq}|{r.event}|{_canonical(r.data)}".encode()
            ).hexdigest()
            if expected != r.hash:
                return False
            prev = r.hash
        return True

    def events(self, event: str) -> tuple[AuditRecord, ...]:
        return tuple(r for r in self._records if r.event == event)

    def to_json(self) -> list[dict[str, Any]]:
        return [
            {
                "seq": r.seq,
                "t": r.t,
                "event": r.event,
                "data": dict(r.data),
                "prev_hash": r.prev_hash,
                "hash": r.hash,
            }
            for r in self._records
        ]

    @classmethod
    def resume(cls, records: Sequence[Mapping[str, Any]]) -> "AuditLog":
        """Rebuild a log from a serialised chain so it can be appended to.

        A stateless serverless function has no memory between invocations, so the
        chain travels with the client and comes back. That makes the client a
        potential attacker, and this is where that is dealt with: the chain is
        **re-verified on load** and a tampered one is rejected outright rather than
        extended.

        Honest about what this does and does not buy: a SHA-256 chain with no
        secret is *tamper-evident against post-hoc edits*, not *unforgeable*. A
        determined client can recompute a whole consistent chain. In a real
        deployment the log is written to append-only storage server-side and this
        method is only used to carry a session across function invocations; the
        integrity claim rests on the store, not on the hashes alone. The hashes are
        what let you prove, afterwards, that the stored log was not edited.
        """
        log = cls()
        for raw in records:
            log._records.append(
                AuditRecord(
                    seq=int(raw["seq"]),
                    t=float(raw["t"]),
                    event=str(raw["event"]),
                    data=dict(raw["data"]),
                    prev_hash=str(raw["prev_hash"]),
                    hash=str(raw["hash"]),
                )
            )
        if not log.verify():
            raise ValueError("refusing to extend a tampered audit chain")
        return log


# ==========================================================================
# The barrier
# ==========================================================================


@dataclass(frozen=True, slots=True)
class AgentResult:
    agent: str
    schema_id: str
    payload: Mapping[str, Any]
    source: Source
    self_reported_confidence: float
    corroboration: Corroboration
    effective_confidence: float
    plane: str  # cloud | edge | transcript | deterministic
    fallback_reason: str | None = None
    latency_ms: float = 0.0

    @property
    def from_model(self) -> bool:
        return self.source != "deterministic_fallback"


class AgentRunner:
    """Runs one agent turn through the full barrier."""

    def __init__(
        self,
        model: LanguageModel,
        audit: AuditLog,
        *,
        confidence_floor: float = CONFIDENCE_FLOOR,
    ) -> None:
        self._model = model
        self._audit = audit
        self._floor = confidence_floor

    def run(
        self,
        agent: Agent,
        task: Any,
        *,
        retrieved: Sequence[Any] = (),
        require_retrieved_citations: bool = True,
    ) -> AgentResult:
        request = agent.request(task)

        try:
            response = self._model.complete(request)
        except ModelUnavailable as exc:
            return self._fallback(agent, task, f"model unavailable: {exc}")

        payload, source, error = self._parse_and_validate(agent, request, response)
        if payload is None:
            return self._fallback(agent, task, error or "unparseable model output")

        # -- grounding
        if "citations" in payload:
            grounding = check_grounding(
                payload["citations"], retrieved, require_retrieved=require_retrieved_citations
            )
            if not grounding.ok:
                return self._fallback(agent, task, f"ungrounded: {grounding.reason}")

        # -- corroboration
        corroboration = agent.corroborate(payload, task)
        if corroboration.fatal:
            return self._fallback(
                agent, task, f"corroboration failed: {'; '.join(corroboration.notes)}"
            )

        self_reported = float(payload["confidence"])
        effective = min(self_reported, corroboration.score)
        if effective < self._floor:
            return self._fallback(
                agent,
                task,
                f"effective confidence {effective:.2f} below floor {self._floor:.2f} "
                f"(self-reported {self_reported:.2f}, corroboration {corroboration.score:.2f})",
            )

        result = AgentResult(
            agent=agent.id,
            schema_id=agent.schema_id,
            payload=payload,
            source=source,
            self_reported_confidence=self_reported,
            corroboration=corroboration,
            effective_confidence=effective,
            plane=response.plane,
            latency_ms=response.latency_ms,
        )
        self._audit.append("agent.accepted", _audit_view(result))
        return result

    def _parse_and_validate(
        self, agent: Agent, request: ModelRequest, response: ModelResponse
    ) -> tuple[Mapping[str, Any] | None, Source, str | None]:
        """Parse, validate, and make exactly one repair attempt on failure."""
        try:
            payload = extract_json(response.text)
            schemas.validate(payload, agent.schema_id)
            return payload, "model", None
        except (ValueError, SchemaError) as first:
            complaint = (
                first.repair_hint() if isinstance(first, SchemaError) else str(first)
            )
            self._audit.append(
                "agent.repair_attempted",
                {"agent": agent.id, "schema": agent.schema_id, "error": complaint[:800]},
            )

        repair = ModelRequest(
            system=request.system,
            context=request.context,
            user=(
                f"{request.user}\n\n"
                "Your previous response was rejected at the schema barrier:\n"
                f"{complaint}\n\n"
                "Emit a corrected JSON object. Change only what the errors require."
            ),
            schema_id=request.schema_id,
            effort=request.effort,
            max_tokens=request.max_tokens,
        )
        try:
            retry = self._model.complete(repair)
            payload = extract_json(retry.text)
            schemas.validate(payload, agent.schema_id)
            return payload, "model_repaired", None
        except ModelUnavailable as exc:
            return None, "deterministic_fallback", f"model unavailable during repair: {exc}"
        except (ValueError, SchemaError) as second:
            return (
                None,
                "deterministic_fallback",
                f"schema violation survived one repair attempt: {second}",
            )

    def _fallback(self, agent: Agent, task: Any, reason: str) -> AgentResult:
        payload = agent.fallback(task)
        # The fallback is held to the same contract as the model. A fallback that
        # does not validate is a bug in us, not a degraded mode, and it must fail
        # loudly here rather than quietly downstream.
        schemas.validate(payload, agent.schema_id)

        result = AgentResult(
            agent=agent.id,
            schema_id=agent.schema_id,
            payload=payload,
            source="deterministic_fallback",
            self_reported_confidence=1.0,
            corroboration=Corroboration.ok(),
            effective_confidence=1.0,
            plane="deterministic",
            fallback_reason=reason,
        )
        self._audit.append("agent.fallback", _audit_view(result))
        return result


def _audit_view(result: AgentResult) -> dict[str, Any]:
    return {
        "agent": result.agent,
        "schema": result.schema_id,
        "source": result.source,
        "plane": result.plane,
        "self_reported_confidence": result.self_reported_confidence,
        "corroboration_score": result.corroboration.score,
        "corroboration_notes": list(result.corroboration.notes),
        "effective_confidence": result.effective_confidence,
        "fallback_reason": result.fallback_reason,
        "payload": dict(result.payload),
    }


# ==========================================================================
# Policy
# ==========================================================================


@dataclass(frozen=True, slots=True)
class PolicyViolation:
    action_index: int
    action_type: str
    message: str

    def __str__(self) -> str:
        return f"action[{self.action_index}] {self.action_type}: {self.message}"


class PolicyError(Exception):
    def __init__(self, violations: Sequence[PolicyViolation]) -> None:
        super().__init__("; ".join(str(v) for v in violations))
        self.violations = tuple(violations)


class PolicyEngine:
    """Blast-radius and authority limits. Applies to model plans *and* fallbacks.

    The caps are not there because we distrust the model specifically. They are
    there because a single automated actor should never be able to move the whole
    venue at once, whoever wrote the plan -- a buggy deterministic playbook can
    cordon every corridor just as effectively as a confused model can.
    """

    MAX_CORDONS = 3
    MAX_DIVERT_SHARE = 0.5
    # An evacuation announcement is P0-only and commander-only, whatever the
    # plan claims its severity is.
    EVACUATION_TEMPLATES = frozenset({"MSG-EVAC-GATE"})

    def check(self, plan: Mapping[str, Any], assessment: Assessment) -> None:
        violations: list[PolicyViolation] = []
        severity = plan["severity"]
        cordons = 0

        for i, action in enumerate(plan["actions"]):
            kind, params = action["type"], action["params"]

            if kind == "CORDON_EDGE":
                cordons += 1
                if cordons > self.MAX_CORDONS:
                    violations.append(PolicyViolation(
                        i, kind,
                        f"exceeds the blast-radius cap of {self.MAX_CORDONS} cordons "
                        "in a single plan",
                    ))

            elif kind == "DIVERT_ARRIVALS":
                if params["share"] > self.MAX_DIVERT_SHARE:
                    violations.append(PolicyViolation(
                        i, kind,
                        f"diverts {params['share']:.0%} of a gate's arrivals; the cap is "
                        f"{self.MAX_DIVERT_SHARE:.0%} in one action",
                    ))

            elif kind == "BROADCAST":
                template_id = params["template_id"]
                template = CATALOGUE.get(template_id)
                if template is None:
                    violations.append(PolicyViolation(i, kind, f"unknown template {template_id!r}"))
                    continue
                if template_id in self.EVACUATION_TEMPLATES and severity != "P0":
                    violations.append(PolicyViolation(
                        i, kind,
                        f"an evacuation announcement may only be issued under P0; this "
                        f"plan is graded {severity} (SOP-EVAC-01#1)",
                    ))

        if violations:
            raise PolicyError(violations)


# ==========================================================================
# Human in the loop
# ==========================================================================


class ApprovalRequired(Exception):
    """A P0/P1 plan was submitted for execution without a valid approval."""


class NotAuthorised(Exception):
    """The operator who signed does not hold the authority to sign this."""


class HumanInTheLoop:
    """The barrier that no automated component may cross.

    Approval is not a boolean the orchestrator sets. It is a signed object
    produced by an :class:`Operator` whose role is checked against the severity,
    referencing the exact plan id. Approving a P1 plan requires the commander
    role; the system cannot manufacture one.
    """

    def __init__(self, audit: AuditLog) -> None:
        self._audit = audit
        self._pending: dict[str, tuple[str, Severity]] = {}

    def submit(self, plan_id: str, severity: Severity, summary: str) -> None:
        self._pending[plan_id] = (summary, severity)
        self._audit.append(
            "hitl.submitted",
            {"plan_id": plan_id, "severity": severity.value, "summary": summary},
        )

    def decide(
        self, operator: Operator, plan_id: str, approved: bool, note: str = ""
    ) -> Approval:
        if plan_id not in self._pending:
            raise KeyError(f"no plan pending approval under id {plan_id!r}")
        _summary, severity = self._pending[plan_id]
        if not operator.may_approve(severity):
            self._audit.append(
                "hitl.refused",
                {
                    "plan_id": plan_id,
                    "operator": operator.id,
                    "reason": f"role(s) {sorted(r.value for r in operator.roles)} may not "
                              f"approve a {severity.value} action",
                },
            )
            raise NotAuthorised(
                f"operator {operator.id} may not approve a {severity.value} action"
            )

        approval = Approval(
            operator=operator, plan_id=plan_id, approved=approved, t=time.time(), note=note
        )
        self._audit.append(
            "hitl.decided",
            {
                "plan_id": plan_id,
                "operator": operator.id,
                "roles": sorted(r.value for r in operator.roles),
                "approved": approved,
                "note": note,
            },
        )
        return approval

    def check(self, plan_id: str, severity: Severity, approval: Approval | None) -> None:
        """Raises unless this exact plan is cleared to actuate."""
        if not severity.requires_human_approval:
            return
        if approval is None:
            raise ApprovalRequired(
                f"plan {plan_id} is graded {severity.value} and may not actuate without "
                "operator sign-off"
            )
        if approval.plan_id != plan_id:
            raise ApprovalRequired(
                f"approval is for plan {approval.plan_id!r}, not {plan_id!r}"
            )
        if not approval.approved:
            raise ApprovalRequired(f"plan {plan_id} was rejected by {approval.operator.id}")
        if not approval.operator.may_approve(severity):
            raise NotAuthorised(
                f"operator {approval.operator.id} may not approve a {severity.value} action"
            )


# ==========================================================================
# Execution
# ==========================================================================


@dataclass(slots=True)
class Execution:
    """What actually happened, with every number recomputed by the deterministic plane."""

    plan_id: str
    cordoned: set[EdgeId] = field(default_factory=set)
    routes: dict[str, Route] = field(default_factory=dict)
    gate_after: dict[GateId, Any] = field(default_factory=dict)
    dispatches: list[Dispatch] = field(default_factory=list)
    escalations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    applied: list[str] = field(default_factory=list)
    # Gate state as the plan mutates it. Actions in a plan compose: a diversion
    # that follows a lane opening must see the lanes that were opened, or the
    # plan's second half is evaluated against a venue that no longer exists.
    gate_state: dict[GateId, GateTelemetry] = field(default_factory=dict)


class Executor:
    """Applies an approved plan by calling the deterministic plane.

    Note what this class does *not* do: it never reads a number out of the plan
    and uses it. The plan says "dispatch a medic from MED-2"; the route, the ETA
    and the cordon-avoidance guarantee are all recomputed here, from the graph, at
    the moment of actuation.
    """

    def __init__(
        self,
        plane: DeterministicPlane,
        catalogue: MessageCatalogue,
        audit: AuditLog,
    ) -> None:
        self._plane = plane
        self._catalogue = catalogue
        self._audit = audit

    def execute(
        self,
        plan: Mapping[str, Any],
        assessment: Assessment,
        snapshot: TelemetrySnapshot,
        *,
        languages: Sequence[LangCode],
    ) -> Execution:
        result = Execution(plan_id=plan["plan_id"], gate_state=dict(snapshot.gates))

        for action in plan["actions"]:
            kind, params = action["type"], action["params"]
            match kind:
                case "BROADCAST":
                    result.dispatches.append(
                        self._broadcast(params, languages, result)
                    )
                case "CORDON_EDGE":
                    result.cordoned.add(params["edge_id"])
                case "DISPATCH_RESPONDER":
                    self._dispatch(params, assessment, result)
                case "OPEN_LANES":
                    self._open_lanes(params, result)
                case "DIVERT_ARRIVALS":
                    self._divert(params, result)
                case "REROUTE_FLOW":
                    result.cordoned.add(params["avoid_edge"])
                case "ESCALATE":
                    result.escalations.append(f"{params['to']}: {params['reason']}")
            result.applied.append(kind)
            self._audit.append(
                "action.executed",
                {"plan_id": plan["plan_id"], "type": kind, "params": dict(params)},
            )

        return result

    def _broadcast(
        self, params: Mapping[str, Any], languages: Sequence[LangCode], result: Execution
    ) -> Dispatch:
        dispatch = self._catalogue.dispatch(
            params["template_id"], languages, params["slots"]
        )
        if dispatch.refused_languages:
            # Not a warning we can swallow: a spectator population we cannot
            # address in their language is an operational fact the control room
            # must see, and a steward must be sent (SOP-COMMS-07#3).
            result.warnings.append(
                f"{params['template_id']}: no validated translation for "
                f"{', '.join(dispatch.refused_languages)}; pictogram "
                f"{dispatch.pictogram} raised and a steward dispatched"
            )
        return dispatch

    def _dispatch(
        self, params: Mapping[str, Any], assessment: Assessment, result: Execution
    ) -> None:
        try:
            route = self._plane.dispatch_route(
                assessment,
                from_node=params["from_node"],
                to_node=params["to_node"],
                cordoned=frozenset(result.cordoned),
            )
        except NoRouteError as exc:
            result.warnings.append(
                f"no responder route {params['from_node']} -> {params['to_node']} "
                f"with the cordon in place: {exc}"
            )
            result.escalations.append(
                f"commander: responder cannot reach {params['to_node']}; manual dispatch required"
            )
            return
        # Re-verify at the moment of actuation: the cordon set has changed since
        # the plan was written.
        self._plane.verify_route(route, cordoned=frozenset(result.cordoned))
        result.routes[f"{params['responder_type']}:{params['to_node']}"] = route

    def _open_lanes(self, params: Mapping[str, Any], result: Execution) -> None:
        gate = result.gate_state[params["gate_id"]]
        metrics = self._plane.reallocate_lanes(gate, params["lanes"])
        result.gate_state[params["gate_id"]] = GateTelemetry(
            gate_id=gate.gate_id,
            arrival_rate_per_s=gate.arrival_rate_per_s,
            service_rate_per_s=gate.service_rate_per_s,
            open_lanes=params["lanes"],
            installed_lanes=gate.installed_lanes,
        )
        result.gate_after[params["gate_id"]] = metrics
        if metrics.breaches_trigger:
            result.warnings.append(
                f"{params['gate_id']}: still at utilisation {metrics.utilisation:.2f} "
                f"after opening {params['lanes']} lanes; diversion required "
                "(SOP-QUEUE-02#2)"
            )

    def _divert(self, params: Mapping[str, Any], result: Execution) -> None:
        source = result.gate_state[params["from_gate"]]
        target = result.gate_state[params["to_gate"]]
        after_source, after_target = self._plane.divert(source, target, params["share"])

        moved = source.arrival_rate_per_s * params["share"]
        result.gate_state[source.gate_id] = GateTelemetry(
            source.gate_id, source.arrival_rate_per_s - moved,
            source.service_rate_per_s, source.open_lanes, source.installed_lanes,
        )
        result.gate_state[target.gate_id] = GateTelemetry(
            target.gate_id, target.arrival_rate_per_s + moved,
            target.service_rate_per_s, target.open_lanes, target.installed_lanes,
        )
        result.gate_after[params["from_gate"]] = after_source
        result.gate_after[params["to_gate"]] = after_target

        if after_target.utilisation >= 0.85:
            result.warnings.append(
                f"{params['to_gate']}: post-diversion utilisation "
                f"{after_target.utilisation:.2f} exceeds the 0.85 ceiling for a "
                "diversion target (SOP-QUEUE-02#2)"
            )


# ==========================================================================
# Orchestrator
# ==========================================================================


@dataclass(slots=True)
class Cycle:
    """One full pass of the cross-track pipeline."""

    correlation_id: str
    assessment: Assessment
    crowd: AgentResult
    brief: AgentResult | None = None
    plan: AgentResult | None = None
    execution: Execution | None = None
    approval: Approval | None = None
    policy_violations: tuple[PolicyViolation, ...] = ()
    blocked_reason: str | None = None


class Orchestrator:
    """Wires the planes together. Holds no state a decision depends on."""

    def __init__(
        self,
        plane: DeterministicPlane,
        model: LanguageModel,
        *,
        retriever: Retriever | None = None,
        catalogue: MessageCatalogue | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        venue = plane.venue
        self.plane = plane
        self.audit = audit or AuditLog()
        self.runner = AgentRunner(model, self.audit)
        self.retriever = retriever or Retriever()
        self.policy = PolicyEngine()
        self.hitl = HumanInTheLoop(self.audit)
        self.catalogue = catalogue or MessageCatalogue(
            known_gates=frozenset(n.id for n in venue.nodes_tagged("gate")),
            known_edges=frozenset(venue.edges),
            known_zones=frozenset(n.zone for n in venue.nodes.values()),
            labels=public_labels(venue),
        )
        self.executor = Executor(plane, self.catalogue, self.audit)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def new_correlation_id(prefix: str = "cyc") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"

    def submit_for_approval(self, plan: Mapping[str, Any]) -> None:
        self.hitl.submit(
            plan["plan_id"], Severity(plan["severity"]), plan["rationale"][:200]
        )

    def actuate(
        self,
        plan: Mapping[str, Any],
        assessment: Assessment,
        snapshot: TelemetrySnapshot,
        *,
        languages: Sequence[LangCode],
        approval: Approval | None = None,
    ) -> Execution:
        """Policy, then the human barrier, then execution. In that order."""
        self.policy.check(plan, assessment)
        self.hitl.check(plan["plan_id"], Severity(plan["severity"]), approval)
        self.audit.append(
            "plan.actuating",
            {
                "plan_id": plan["plan_id"],
                "severity": plan["severity"],
                "approved_by": approval.operator.id if approval else None,
            },
        )
        return self.executor.execute(plan, assessment, snapshot, languages=languages)
