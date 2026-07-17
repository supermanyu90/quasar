"""Shared agent scaffolding: the house rules, the corroboration result, and the
abstract :class:`Agent` (a prompt, a corroborator, a deterministic fallback)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

from quasar.llm import ModelRequest


_HOUSE_RULES = """\
You are a component of Quasar, the operating system of a 60,000-seat stadium on
a match day. Real people will move because of what you output.

Rules that are not negotiable:
- You do not compute distances, routes, capacities, walking times or evacuation
  times. Those are computed for you and given to you. Use the numbers you are
  given; never invent one and never adjust one.
- You never write the text of a safety-critical announcement. You choose an
  approved template and fill typed slots.
- If the evidence you are given does not support a conclusion, say so in your
  output and lower your confidence. An honest "I don't know" is cheap. A fluent
  wrong answer costs a life.
- Output exactly one JSON object conforming to the named schema. No prose.
"""


@dataclass(frozen=True, slots=True)
class Corroboration:
    """A deterministic score of a model payload against ground truth."""

    score: float
    notes: tuple[str, ...] = ()
    fatal: bool = False

    @staticmethod
    def ok() -> "Corroboration":
        return Corroboration(1.0)

    @staticmethod
    def fail(note: str) -> "Corroboration":
        return Corroboration(0.0, (note,), fatal=True)


class Agent(ABC):
    id: str
    schema_id: str
    system: str
    # What is lost when the deterministic fallback runs instead of the model.
    VALUE_OVER_FALLBACK: str = ""

    @abstractmethod
    def request(self, task: Any) -> ModelRequest: ...

    @abstractmethod
    def corroborate(self, payload: Mapping[str, Any], task: Any) -> Corroboration: ...

    @abstractmethod
    def fallback(self, task: Any) -> dict[str, Any]: ...
