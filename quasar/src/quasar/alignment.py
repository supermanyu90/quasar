"""Problem-statement alignment: the challenge, the users, the objectives.

One source of truth for *why* the system is shaped the way it is, mapped to the
places you can go and see each claim for yourself. The console serves this over
the wire and renders it as the "Guide" tab; a test asserts it covers every track
and every persona, so the alignment can't silently fall out of date as features
move. Each item carries a ``where`` -- the console query string that jumps you to
the feature that proves it -- so the guide is a set of live links, not a brochure.

Keeping this as structured data (rather than prose baked into the page) is the
same discipline as :mod:`quasar.amenities`: the taxonomy lives once, is served,
and is tested for completeness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# The problem statement, verbatim in intent.
CHALLENGE = (
    "Build a GenAI-enabled architecture that directly optimizes venue operations "
    "and elevates the tournament experience for fans, organizers, volunteers, and "
    "on-ground staff — covering dynamic crowd management, smart indoor navigation, "
    "real-time decision support, and multi-language assistance."
)

# The one idea the whole design turns on.
THESIS = (
    "Two planes, one barrier. A deterministic plane owns every number a life-safety "
    "decision depends on. A generative plane owns interpretation, language, and "
    "proposal. A governance barrier validates, corroborates, and refuses to actuate "
    "a P0/P1 action without a human signature. GenAI is load-bearing — not "
    "decorative — and it never computes a route, a capacity, or an evacuation time."
)


@dataclass(frozen=True, slots=True)
class Item:
    key: str
    icon: str
    title: str
    summary: str          # what it is / who they are
    how: str              # how Quasar meets the need
    cta: str              # the "show me" button label
    where: Mapping[str, str]  # console query params that jump to the proof
    modules: tuple[str, ...] = field(default=())  # the code behind it


# -- the four tracks --------------------------------------------------------
TRACKS: tuple[Item, ...] = (
    Item(
        "crowd", "🌊", "Dynamic crowd management",
        "Keep a 60,000-person crowd flowing and catch a build-up before it becomes a crush.",
        "Weidmann speed–density and Fruin level-of-service turn corridor counts into a "
        "picture; M/M/c queueing flags any gate past the 0.90 utilisation trigger; the "
        "system proposes lane reallocation or diversion, and the pre-match harness fires "
        "generated failure scenarios at the venue before doors open.",
        "Run the incident", {"tab": "control", "run": "1", "approve": "commander"},
        ("crowd.py", "queueing.py", "plane.py", "scenarios.py"),
    ),
    Item(
        "navigation", "🧭", "Smart indoor navigation",
        "Get anyone — including a wheelchair user — to where they need to be, around the crowd.",
        "Density-aware Dijkstra prices corridors by walking time, not distance, and steers "
        "around busy ones; a step-free profile treats accessibility as a hard constraint; a "
        "graph-constrained particle filter answers 'where am I' from BLE; and the 3D view "
        "shows ramps against stairs at a glance.",
        "Find me a step-free route", {"tab": "fan", "find": "accessible_restroom", "view": "3d"},
        ("routing.py", "positioning.py", "amenities.py"),
    ),
    Item(
        "decision", "🧠", "Real-time decision support",
        "Turn a panicked radio call into a graded, sourced, actionable brief for the command centre.",
        "A RAG copilot grounds every recommendation in the venue's standing procedure and "
        "cites the clause; the incident brief and planner are checked against ground truth "
        "the model didn't produce; and no P0/P1 action actuates without a commander's "
        "signature — a barrier you can try to defeat.",
        "Watch the barrier work", {"tab": "control", "run": "1"},
        ("rag.py", "agents.py", "governance.py", "sops.py"),
    ),
    Item(
        "language", "🗣️", "Multi-language assistance",
        "Speak to every fan in their language — and never mistranslate a safety announcement.",
        "A two-tier policy: safety-critical messages come only from a human-validated "
        "catalogue (no machine translation, ever), with a check that gate numbers survive "
        "translation; informational help is generated and translated behind quality gates. "
        "The readiness audit tells a venue, before it opens, whether it can lawfully address "
        "its own majority-language crowd.",
        "See a language readiness block", {"tab": "ready", "venue": "fwc-mexico-city"},
        ("language.py", "readiness.py"),
    ),
)

# -- the four personas ------------------------------------------------------
PERSONAS: tuple[Item, ...] = (
    Item(
        "fan", "🎟️", "Fan / attendee",
        "Lost at the wrong gate, needs a restroom, a seat, a quiet room — maybe in Marathi, maybe step-free.",
        "The Attendee tab: tap what you need and get a crowd-aware, step-free, calm route "
        "drawn on the map, with the reply in your language; or just ask the concierge in "
        "your own words.",
        "Open the attendee companion", {"tab": "fan", "find": "food"},
        ("amenities.py", "agents.py", "routing.py"),
    ),
    Item(
        "organizer", "🎛️", "Organizer / commander",
        "Needs the ground truth, a proposed plan, and the final say — accountably.",
        "The control room shows the measured state first (nobody's opinion), then the "
        "agents' proposal with its corroboration score, then the human-in-the-loop barrier: "
        "approve, deny, or watch it refuse an under-authorised or tampered plan. Every step "
        "is in a tamper-evident audit chain.",
        "Take the commander's seat", {"tab": "control", "run": "1", "approve": "commander"},
        ("governance.py", "plane.py"),
    ),
    Item(
        "volunteer", "🙋", "Volunteer",
        "In the middle of it, describing what they see over a radio — no form, no dropdown.",
        "Their free-text report is turned into a structured, severity-graded, SOP-cited "
        "incident brief in seconds. The severity floor is set by procedure from the measured "
        "crowd pressure, so a calm-sounding report of a real crush can't be under-graded.",
        "See a report become a brief", {"tab": "control", "run": "1"},
        ("agents.py", "rag.py"),
    ),
    Item(
        "staff", "⛑️", "On-ground staff (medic / security)",
        "Has to reach a casualty behind a cordon, fast, without threading the crowd that caused it.",
        "Dispatch arrives with a route computed from the graph at the moment of actuation — "
        "cordon-safe, via the staff-only service ring, and density-aware, so the responder "
        "takes the longer path when it is genuinely the faster one.",
        "Watch a medic dispatched", {"tab": "control", "run": "1", "approve": "commander"},
        ("routing.py", "plane.py"),
    ),
)

# -- core objectives (how the design earns the rubric) ----------------------
OBJECTIVES: tuple[Item, ...] = (
    Item(
        "loadbearing", "⚡", "GenAI is load-bearing, not decorative",
        "The test of whether the model earns its place is what is lost when it's gone.",
        "Switch the model plane to Partition (no model at all) and re-run the incident: the "
        "venue still cordons, dispatches, relieves the gate, and announces safely — what it "
        "loses is the synthesis and the language. That diff is the argument.",
        "Pull the model out", {"tab": "control", "mode": "partition", "run": "1", "approve": "commander"},
        ("llm.py", "agents.py"),
    ),
    Item(
        "safety", "🛡️", "Hybrid safety architecture",
        "A fluent, confident, wrong model must never move the crowd on its own.",
        "Every model payload is validated against a published schema, corroborated against "
        "the sensors, and gated on min(self-reported, corroboration). Try to tamper with the "
        "plan or approve a P0 as a steward — the barrier refuses, and says why.",
        "Try to defeat the barrier", {"tab": "control", "run": "1"},
        ("schemas.py", "governance.py"),
    ),
    Item(
        "accessibility", "♿", "Accessibility as a hard constraint",
        "Step-free isn't a preference; a stand with no ramp is a defect found before the season.",
        "The accessible profile refuses a stepped route rather than degrading; calm mode "
        "holds a fan below a comfortable density; the 3D view colours ramps against stairs; "
        "and the readiness audit blocks a venue whose stand has no step-free exit.",
        "See the accessibility audit", {"tab": "ready", "venue": "national-stadium"},
        ("routing.py", "readiness.py"),
    ),
    Item(
        "honesty", "📐", "Honest about what's real",
        "Safety-critical data is never fabricated and passed off as surveyed.",
        "The 16 FIFA World Cup 2026 venues carry real identity but a representative graph, "
        "and every one is stamped and surfaced as such — correct for planning, explicitly "
        "not a substitute for a floor-plan survey.",
        "See a representative venue", {"tab": "ready", "venue": "fwc-new-york"},
        ("venue_factory.py", "readiness.py"),
    ),
)


def sections() -> tuple[tuple[str, str, tuple[Item, ...]], ...]:
    """(key, heading, items) for each part of the guide, in reading order."""
    return (
        ("tracks", "The four tracks", TRACKS),
        ("personas", "For each person in the venue", PERSONAS),
        ("objectives", "How the design earns it", OBJECTIVES),
    )
