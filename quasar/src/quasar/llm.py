"""The model plane: cloud, edge, and the failover between them.

Three implementations of one protocol.

:class:`AnthropicModel` is the primary: Claude Opus 4.8, adaptive thinking, and
-- where the payload schema permits it -- server-side structured outputs, so the
JSON is constrained at generation time as well as validated at the barrier. The
agent's system prompt and the SOP context are placed in a stable cache prefix
and the volatile telemetry after it, because a match-day control room issues
hundreds of queries against an identical SOP corpus and paying full price for
that prefix every time is simply a bug.

:class:`OllamaEdgeModel` is the survival path: a small open-weights instruct
model on a box in the comms room, reachable over the venue LAN with no internet.
When the uplink drops mid-fixture -- which is when a stadium is at its most
dangerous, not its least -- the copilot degrades to a weaker model rather than
disappearing. It speaks the same protocol and its output goes through exactly
the same schema, grounding and corroboration gates.

:class:`TranscriptModel` replays recorded model responses. It exists so CI can
exercise the full agent path without a network call, and it is honest about what
it is: a *recording*, not a simulation. It cannot be used to fake the safety
logic, because none of the safety logic lives in the model -- the router, the
queueing model, the catalogue and the governance barrier all run for real under
the transcript, and the tests assert on their outputs.

None of these classes decides anything. They return text. Everything that
happens to that text happens in :mod:`quasar.governance`.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from quasar.schemas import SCHEMAS

# Per Anthropic's current model line. Opus 4.8 removed budget_tokens and the
# sampling parameters; thinking is adaptive and depth is controlled by effort.
DEFAULT_MODEL = "claude-opus-4-8"
DEFAULT_EDGE_MODEL = "llama3.1:8b-instruct-q4_K_M"


class ModelUnavailable(Exception):
    """The model could not be reached or returned no usable content.

    Distinct from a *bad* response, which is a governance concern. This is a
    transport concern, and it is what triggers failover to the edge model.
    """


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One agent turn.

    ``system`` is stable across the fixture and is what gets cached.
    ``context`` is the retrieved SOP text -- stable for a given query shape.
    ``user`` carries the volatile telemetry and must come last.
    """

    system: str
    user: str
    schema_id: str
    context: str = ""
    effort: str = "high"
    max_tokens: int = 8000


@dataclass(frozen=True, slots=True)
class ModelResponse:
    text: str
    model: str
    latency_ms: float
    plane: str = "cloud"  # "cloud" | "edge" | "transcript"
    cache_read_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class LanguageModel(Protocol):
    name: str

    def complete(self, request: ModelRequest) -> ModelResponse: ...


# --------------------------------------------------------------------------
# Structured-output schema derivation
# --------------------------------------------------------------------------

# Keywords the Messages API's structured-output mode accepts. Everything else
# (pattern, minimum, minLength, ...) is a constraint we still enforce locally in
# quasar.schemas but cannot push into the decoder.
_STRICT_KEYWORDS = frozenset(
    {"type", "enum", "const", "properties", "required", "additionalProperties", "items", "anyOf", "description"}
)


def schema_prompt(schema_id: str) -> str:
    """Render the published schema for inclusion in the prompt.

    Structured outputs constrain the decoder, which is strictly better than asking
    nicely -- but two of our payloads carry an open-keyed ``slots`` map that
    ``additionalProperties: false`` cannot express, so the decoder cannot be
    constrained for them. Without this, those agents were sent *no schema at all*:
    not in ``output_config``, and not in the prompt either. The model was expected
    to guess the exact field names of seven action variants, would fail validation,
    burn its one repair attempt, and fall back to the deterministic planner.

    The system would have stayed safe -- that is what the barrier is for -- but it
    would have silently stopped using the model, which is the failure you least
    want to be silent. The recorded transcripts hid it, because they were already
    schema-perfect.

    The schema is stable for the life of the fixture, so it belongs in the cached
    prefix and costs nothing after the first call.
    """
    schema = SCHEMAS.get(schema_id)
    if schema is None:
        raise KeyError(f"no schema registered under {schema_id!r}")

    from quasar.examples import EXAMPLES  # local: examples import schemas

    parts = [
        f"# Output contract: {schema_id}",
        "",
        "Emit exactly one JSON object validating against this schema. It is enforced at",
        "a hard barrier: a payload that does not validate is discarded and a deterministic",
        "fallback runs instead. `additionalProperties: false` means exactly that -- an",
        "unlisted field causes rejection.",
        "",
        "```json",
        json.dumps(schema, indent=2, sort_keys=True),
        "```",
    ]

    example = EXAMPLES.get(schema_id)
    if example is not None:
        parts += [
            "",
            "## What a valid answer looks like",
            "",
            "Below is a VALID payload -- for a completely different situation. Emit an",
            "object of THIS shape. Do NOT echo the schema above: no `$id`, no `properties`,",
            "no `type` keys. The schema describes the answer; it is not the answer.",
            "",
            "Do NOT copy the values below. They describe a different incident, at different",
            "corridors, with different numbers. Every figure you emit must come from the",
            "telemetry you are given -- restating a measurement incorrectly is checked, and",
            "your answer will be discarded if it does not match the sensors.",
            "",
            "```json",
            json.dumps(example, indent=2, ensure_ascii=False),
            "```",
        ]

    return "\n".join(parts)


def strict_output_schema(schema_id: str) -> Mapping[str, Any] | None:
    """Derive a structured-outputs schema, or None if the schema cannot be expressed.

    Structured outputs require ``additionalProperties: false`` on every object.
    Two of our payloads carry an open-ended ``slots`` map (typed values, open
    keys), which is legitimately inexpressible under that rule -- so for those we
    fall back to instructed JSON plus the local validator and repair loop, rather
    than quietly weakening the schema to make the API happy.
    """
    def convert(node: Any) -> Any | None:
        if not isinstance(node, dict):
            return node
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            return None  # open-keyed map: not expressible
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key not in _STRICT_KEYWORDS:
                continue
            if key == "properties":
                props: dict[str, Any] = {}
                for name, sub in value.items():
                    converted = convert(sub)
                    if converted is None:
                        return None
                    props[name] = converted
                out[key] = props
            elif key == "items":
                converted = convert(value)
                if converted is None:
                    return None
                out[key] = converted
            elif key == "anyOf":
                branches = []
                for sub in value:
                    converted = convert(sub)
                    if converted is None:
                        return None
                    branches.append(converted)
                out[key] = branches
            else:
                out[key] = value
        if out.get("type") == "object":
            out["additionalProperties"] = False
            # Structured outputs require every property to be listed as required.
            if "properties" in out and set(out.get("required") or ()) != set(out["properties"]):
                return None
        return out

    schema = SCHEMAS.get(schema_id)
    if schema is None:
        raise KeyError(f"no schema registered under {schema_id!r}")
    return convert(schema)


# --------------------------------------------------------------------------
# Cloud
# --------------------------------------------------------------------------


class AnthropicModel:
    """Claude via the official SDK. Import is lazy so the venue's deterministic
    plane runs on a box with no SDK installed and no network."""

    name = "anthropic"

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._model = model
        self._timeout = timeout_s
        if client is not None:
            self._client = client
            return
        try:
            import anthropic  # noqa: PLC0415  (deliberate: optional dependency)
        except ImportError as exc:  # pragma: no cover - exercised by deployment, not CI
            raise ModelUnavailable(
                "the anthropic SDK is not installed; install `anthropic` or run "
                "against the edge model"
            ) from exc
        # Zero-arg constructor: resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
        # or an `ant auth login` profile. Never hardcode a key into a venue box.
        self._client = anthropic.Anthropic(timeout=timeout_s)

    def complete(self, request: ModelRequest) -> ModelResponse:
        # Cache breakpoint sits on the last stable block. The SOP context and the
        # schema are part of the prefix; the telemetry in `user` is not, so it
        # never invalidates.
        system: list[dict[str, Any]] = [{"type": "text", "text": request.system}]
        if request.context:
            system.append({"type": "text", "text": request.context})

        strict = strict_output_schema(request.schema_id)
        if strict is None:
            # The decoder cannot be constrained for this payload, so the model must
            # at least be *shown* the contract it is being held to. See schema_prompt().
            system.append({"type": "text", "text": schema_prompt(request.schema_id)})

        system[-1]["cache_control"] = {"type": "ephemeral"}

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": request.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": request.user}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": request.effort},
        }
        if strict is not None:
            kwargs["output_config"]["format"] = {"type": "json_schema", "schema": strict}

        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:  # transport, rate limit, overload -> failover
            raise ModelUnavailable(f"anthropic call failed: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if getattr(response, "stop_reason", None) == "refusal":
            raise ModelUnavailable("model declined the request")

        text = next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"), ""
        )
        if not text.strip():
            raise ModelUnavailable("model returned no text content")

        usage = getattr(response, "usage", None)
        return ModelResponse(
            text=text,
            model=self._model,
            latency_ms=elapsed_ms,
            plane="cloud",
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )


# --------------------------------------------------------------------------
# Edge
# --------------------------------------------------------------------------


class OllamaEdgeModel:
    """A small open-weights instruct model on the venue LAN.

    Speaks HTTP to a local Ollama daemon using only the standard library, so the
    edge box carries no Python dependency tree that could fail to install on a
    machine that will never see the internet again after commissioning.
    """

    name = "edge"

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:11434/api/chat",
        model: str = DEFAULT_EDGE_MODEL,
        timeout_s: float = 20.0,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._timeout = timeout_s

    def complete(self, request: ModelRequest) -> ModelResponse:
        system = request.system
        if request.context:
            system = f"{system}\n\n# Retrieved procedure\n{request.context}"
        # The edge model has no structured-output mode at all, so it always gets
        # the full schema in-band -- naming the schema is not enough, the model
        # cannot be expected to know what fields a name implies. The decoder is
        # pinned to JSON, and the output still goes through the same validator,
        # the same grounding check and the same corroborator as the cloud model.
        system = f"{system}\n\n{schema_prompt(request.schema_id)}\n\nNo prose. No markdown fence."
        body = json.dumps(
            {
                "model": self._model,
                "stream": False,
                "format": "json",
                "options": {"num_predict": request.max_tokens},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": request.user},
                ],
            }
        ).encode()

        req = urllib.request.Request(
            self._endpoint, data=body, headers={"Content-Type": "application/json"}
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
                payload = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            raise ModelUnavailable(f"edge model unreachable: {exc}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        text = (payload.get("message") or {}).get("content", "")
        if not text.strip():
            raise ModelUnavailable("edge model returned no content")

        return ModelResponse(
            text=text, model=self._model, latency_ms=elapsed_ms, plane="edge"
        )


# --------------------------------------------------------------------------
# Disabled
# --------------------------------------------------------------------------


class DisabledModel:
    """No model plane. Every call fails; every agent takes its deterministic twin.

    This is a first-class operating mode, not a test double. An operator may
    deliberately disable the generative plane -- during a certification run, under
    a supplier incident, or because the venue's safety case has not yet cleared it
    -- and the venue must continue to work. It is also what a total network
    partition looks like from the inside.
    """

    name = "disabled"

    def __init__(self, reason: str = "model plane disabled by operator") -> None:
        self._reason = reason

    def complete(self, request: ModelRequest) -> ModelResponse:
        raise ModelUnavailable(self._reason)


# --------------------------------------------------------------------------
# Failover
# --------------------------------------------------------------------------


@dataclass(slots=True)
class FailoverModel:
    """Cloud first, edge on failure, deterministic fallback if both are gone.

    The third case is not handled here -- it is handled by returning
    ModelUnavailable and letting :mod:`quasar.governance` take the deterministic
    path. A model plane that invents an answer when it cannot reach a model is
    the single most dangerous thing this system could contain.
    """

    primary: LanguageModel
    secondary: LanguageModel | None = None
    name: str = "failover"
    partition_events: list[str] = field(default_factory=list)

    def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            return self.primary.complete(request)
        except ModelUnavailable as exc:
            if self.secondary is None:
                raise
            self.partition_events.append(str(exc))
            return self.secondary.complete(request)


# --------------------------------------------------------------------------
# Recorded transcripts (CI only)
# --------------------------------------------------------------------------


class TranscriptModel:
    """Replays recorded model output, keyed by (schema_id, cue).

    Used by the test suite so the agent path is exercised end to end without a
    network. It deliberately raises on an unknown key rather than improvising:
    a test that silently gets a generic answer is a test that proves nothing.
    """

    name = "transcript"

    def __init__(self, transcripts: Mapping[tuple[str, str], str]) -> None:
        self._transcripts = dict(transcripts)
        self.calls: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        for (schema_id, cue), text in self._transcripts.items():
            if schema_id == request.schema_id and cue in request.user:
                return ModelResponse(
                    text=text, model="transcript", latency_ms=0.0, plane="transcript"
                )
        raise ModelUnavailable(
            f"no recorded transcript for schema {request.schema_id!r} matching the request"
        )


def extract_json(text: str) -> Any:
    """Pull a JSON object out of a model response.

    Tolerates a markdown fence and leading prose, because a weaker edge model
    will produce both. Raises ValueError if there is no parseable object -- which
    the governance layer treats as a parse failure and repairs or falls back.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start == -1:
        raise ValueError("no JSON object found in model output")
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(stripped)):
        ch = stripped[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stripped[start : i + 1])
                except json.JSONDecodeError as exc:
                    raise ValueError(f"malformed JSON object in model output: {exc}") from exc
    raise ValueError("unterminated JSON object in model output")
