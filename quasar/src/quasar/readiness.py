"""Venue readiness: can this venue safely open its gates?

Run once, when a venue is onboarded, and again before every fixture. It asks three
questions the venue's own floor plan cannot answer, and it answers them *for this
venue* rather than in general -- which is the entire point of making the venue a
configuration rather than a hardcoded assumption.

**Can we tell this crowd to leave?** A venue spec declares the languages its crowd
actually speaks. The Tier-1 message catalogue declares the languages in which a
safety-critical announcement has been *human-validated*. The gap between those two
sets is not a software defect and it cannot be fixed by better code: it means that
tonight, if this venue has to evacuate, there are people in it who will not be told
to in a language they read. Quasar refuses to machine-translate an evacuation order
(SOP-COMMS-07#1), so the only honest thing to do is surface the gap before the
gates open, while a translator can still be hired.

This finding is the reason venue selection matters. The same code, pointed at a
Mumbai stadium (Hindi and Marathi: validated) and a Chennai arena (Tamil: majority
language, machine draft only), gives opposite answers -- and the second one is a
finding no amount of engineering will close.

**Can everyone get out?** Step-free egress from every stand, in an empty venue.
A stand with a staircase and no ramp is a defect in the building.

**Does the venue survive its own failure modes?** The stress harness, run over
generated scenarios (:mod:`quasar.scenarios`).
"""

from __future__ import annotations

from dataclasses import dataclass

from quasar.language import CATALOGUE, SUPPORTED_LANGUAGES, Tier
from quasar.plane import DeterministicPlane
from quasar.routing import ACCESSIBLE, NoRouteError, RouteRequest
from quasar.scenarios import SCENARIO_KINDS, SeededSampler, StressHarness
from quasar.venue_spec import VenueProfile


@dataclass(frozen=True, slots=True)
class Check:
    id: str
    severity: str  # "blocker" | "critical" | "major" | "info"
    title: str
    detail: str
    remedy: str

    @property
    def blocking(self) -> bool:
        return self.severity == "blocker"


@dataclass(frozen=True, slots=True)
class Readiness:
    venue_id: str
    venue_name: str
    checks: tuple[Check, ...]

    @property
    def ready(self) -> bool:
        return not any(c.blocking for c in self.checks)

    def by_severity(self, severity: str) -> tuple[Check, ...]:
        return tuple(c for c in self.checks if c.severity == severity)


def language_readiness(profile: VenueProfile) -> list[Check]:
    """Which of this crowd's languages can we lawfully make a safety announcement in?"""
    checks: list[Check] = []

    tier1 = {tid: t for tid, t in CATALOGUE.items() if t.tier is Tier.SAFETY_CRITICAL}
    spoken = list(profile.languages)

    # The venue's majority language is the first one declared in its spec.
    majority = spoken[0] if spoken else "en"

    for language in spoken:
        name = SUPPORTED_LANGUAGES[language].name if language in SUPPORTED_LANGUAGES else language
        missing = sorted(
            tid for tid, t in tier1.items() if language not in t.validated_languages()
        )
        if not missing:
            continue

        is_majority = language == majority
        checks.append(
            Check(
                id=f"lang.tier1.{language}",
                # A venue that cannot tell its *majority* crowd to evacuate should not
                # open. For a minority language it is serious but survivable: the
                # pictogram and the steward carry it (SOP-COMMS-07#3).
                severity="blocker" if is_majority else "critical",
                title=f"No validated safety announcements in {name}",
                detail=(
                    f"{name} is {'the majority language of this crowd' if is_majority else 'spoken by this crowd'}, "
                    f"but {len(missing)} of {len(tier1)} safety-critical templates have no "
                    f"human-validated {name} entry: {', '.join(missing)}. Quasar will not "
                    "machine-translate an evacuation instruction (SOP-COMMS-07#1), so tonight "
                    "these announcements will fall back to a pictogram and a steward."
                ),
                remedy=(
                    f"Commission a certified translation of {len(missing)} template(s) into {name} "
                    "and mark the catalogue entries human_validated. This is a translation "
                    "procurement, not an engineering task: no change to this software will fix it."
                ),
            )
        )

    if unsupported := [lang for lang in spoken if lang not in SUPPORTED_LANGUAGES]:
        checks.append(
            Check(
                id="lang.unsupported",
                severity="critical",
                title="Crowd speaks languages the system does not know",
                detail=f"The venue declares {unsupported}, which the concierge cannot serve at all.",
                remedy="Add the language to SUPPORTED_LANGUAGES and to the message catalogue.",
            )
        )

    return checks


def accessibility_readiness(profile: VenueProfile) -> list[Check]:
    """Step-free egress from every stand, in an empty, fully open venue.

    Empty and fully open on purpose: this is the question of whether the *building*
    works, separated from whether tonight's crowd or tonight's closures work. A
    stand that fails here fails on every night of the year.
    """
    plane = DeterministicPlane(profile.venue)
    empty = {e: 0.0 for e in profile.venue.edges}
    gates = profile.venue.nodes_tagged("gate")
    checks: list[Check] = []

    for seat in profile.venue.nodes_tagged("seating"):
        reachable = False
        for gate in gates:
            try:
                plane.router.route(
                    RouteRequest(origin=seat.id, destination=gate.id, profile=ACCESSIBLE),
                    empty,
                )
                reachable = True
                break
            except NoRouteError:
                continue
        if not reachable:
            checks.append(
                Check(
                    id=f"access.egress.{seat.id}",
                    severity="blocker",
                    title=f"No step-free way out of {seat.name}",
                    detail=(
                        f"{seat.id} has no step-free route to any gate even in an empty, "
                        "fully open venue. A spectator who cannot use stairs cannot leave "
                        "this stand unaided."
                    ),
                    remedy=(
                        "Install a ramp or lift, or staff a refuge point at this stand for "
                        "every fixture and evacuate by assisted lift (SOP-EVAC-01#3)."
                    ),
                )
            )
    return checks


def topology_readiness(profile: VenueProfile, *, per_kind: int = 2) -> list[Check]:
    """Fire generated scenarios at the venue and report what the building cannot take."""
    plane = DeterministicPlane(profile.venue)
    harness = StressHarness(
        plane,
        service_rate_per_s=profile.service_rate_per_s,
        installed_lanes={g: t.installed_lanes for g, t in profile.fixture.gates.items()},
    )
    sampler = SeededSampler(plane, seed=11)

    seen: dict[tuple[str, str], Check] = {}
    for kind in SCENARIO_KINDS:
        for scenario in sampler.sample(kind, n=per_kind):
            for f in harness.run(scenario).findings:
                # The accessibility audit above already covers the structural
                # step-free case, and covers it better (empty venue, nothing closed).
                if f.invariant == "step-free-egress-exists":
                    continue
                # Deduplicate on the SUBJECT, not the sentence. "G3 needs 16 lanes"
                # and "G3 needs 13 lanes" are the same finding about the same gate
                # under two random scenarios, and printing both twice teaches the
                # reader to skim.
                subject = f.detail.split()[0]
                key = (f.invariant, subject)
                if key not in seen:
                    seen[key] = Check(
                        id=f"topology.{f.invariant}.{subject}",
                        severity="critical" if f.severity == "critical" else "major",
                        title=f.invariant.replace("-", " "),
                        detail=f.detail,
                        remedy="Review the venue's closure plan and stewarding for this scenario.",
                    )
    return list(seen.values())


def provenance_readiness(profile: VenueProfile) -> list[Check]:
    """Is the graph a real survey, or a representative model?

    This is the honesty check. A representative graph is right for scale, planning
    and training, and it is *not* a surveyed floor plan -- the corridor widths and
    step-free routes are fitted from public capacity, not measured. Every route and
    evacuation time computed on it inherits that caveat, and an operator must know
    it before trusting the system in a live control room. It is not a blocker (the
    hand-authored demo venues are representative too), but it is the first thing the
    audit says.
    """
    if profile.surveyed:
        return []
    return [
        Check(
            id="provenance.representative",
            severity="critical",
            title="Topology is representative, not surveyed",
            detail=(
                f"{profile.name}'s graph is a parametric model fitted to its public "
                f"capacity ({profile.capacity:,}) and gate count -- not a floor-plan "
                "survey. It is correct for scale, planning and training. It is not a "
                "substitute for measured corridor widths and verified step-free routes, "
                "and every route and evacuation time here inherits that caveat."
            ),
            remedy=(
                "Commission a floor-plan survey (corridor widths, gate lane counts, "
                "step-free routes) and load it as topology: \"surveyed\" before using "
                "this venue for live operations."
            ),
        )
    ]


def audit(profile: VenueProfile) -> Readiness:
    """The full pre-season audit for one venue."""
    checks: list[Check] = []
    checks += provenance_readiness(profile)
    checks += accessibility_readiness(profile)
    checks += language_readiness(profile)
    checks += topology_readiness(profile)

    order = {"blocker": 0, "critical": 1, "major": 2, "info": 3}
    checks.sort(key=lambda c: (order[c.severity], c.id))

    return Readiness(
        venue_id=profile.id, venue_name=profile.name, checks=tuple(checks)
    )
