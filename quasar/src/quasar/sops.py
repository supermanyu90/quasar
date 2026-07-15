"""The venue's standing operating procedures -- the corpus the copilot is grounded in.

Each SOP is split into numbered sections because a citation to a *document* is
not a citation: an operator being told to act at 02:00 on a Saturday needs the
clause, not the manual. Section ids are stable and are what the grounding check
in :mod:`quasar.rag` verifies against.

In production this corpus is the venue's controlled document set, ingested from
the safety certificate holder's DMS with a version hash per section, and
re-indexed on publication. The text here is representative and complete enough
to drive the system end to end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class SopSection:
    doc_id: str
    section: str
    title: str
    text: str

    @property
    def ref(self) -> str:
        return f"{self.doc_id}#{self.section}"


SOP_CORPUS: Sequence[SopSection] = (
    # ---------------- Medical ----------------
    SopSection(
        "SOP-MED-03", "1", "Medical emergency in a public circulation area",
        "On report of a collapsed or seriously injured person in a concourse, corridor or "
        "vomitory, the control room must dispatch the nearest available medical team within "
        "60 seconds of the report being logged. The incident is graded P1 by default and is "
        "upgraded to P0 if the casualty is unresponsive or if crowd pressure at the scene "
        "exceeds level of service E.",
    ),
    SopSection(
        "SOP-MED-03", "2", "Clearing an approach route",
        "A medical team must never be routed through a corridor at level of service F. Where "
        "the direct approach is congested, the team uses the inner service corridor ring, "
        "which is reserved for staff movement and is not to be opened to spectators under any "
        "circumstances. Stewards clear a working space of at least three metres around the "
        "casualty before the team arrives.",
    ),
    SopSection(
        "SOP-MED-03", "3", "Crowd management around a casualty",
        "Spectator flow past a medical incident is stopped in the affected corridor and "
        "diverted at the two nearest junctions. The corridor is cordoned. A cordon applies to "
        "spectators and staff alike; responders reach the far side by the service ring. The "
        "diversion is announced in the affected zone before the cordon is placed, never after, "
        "so that arriving spectators are not walked into a closed corridor.",
    ),
    # ---------------- Queueing / ingress ----------------
    SopSection(
        "SOP-QUEUE-02", "1", "Turnstile utilisation limits",
        "Each gate is modelled as a multi-server queue. Sustained utilisation at or above 0.90 "
        "of open-lane capacity is the intervention threshold: at this point expected wait grows "
        "sharply with any further arrivals and the queue outside the gate begins to build into "
        "the public highway. On breach, the duty manager must open additional lanes, divert "
        "arrivals to an adjacent gate, or both.",
    ),
    SopSection(
        "SOP-QUEUE-02", "2", "Lane reallocation",
        "Additional lanes are opened from the gate's installed reserve before any diversion is "
        "announced. Lanes may only be reallocated to a gate that has stewarding present. If "
        "opening every installed lane still leaves utilisation at or above 0.90, arrivals must "
        "be diverted to the nearest gate whose post-diversion utilisation would remain below "
        "0.85.",
    ),
    SopSection(
        "SOP-QUEUE-02", "3", "Diversion messaging",
        "A diversion is communicated to affected spectators in the languages configured for the "
        "fixture, using the controlled message catalogue. Diversion messages must name the "
        "replacement gate explicitly. A diversion announced without a named alternative gate "
        "increases pressure on the remaining gates and is not permitted.",
    ),
    # ---------------- Evacuation ----------------
    SopSection(
        "SOP-EVAC-01", "1", "Evacuation authority",
        "Only the safety officer or the duty commander may order an evacuation. No automated "
        "system may initiate an evacuation announcement. Systems may recommend, prepare and "
        "pre-render an evacuation instruction, but the instruction is not broadcast until a "
        "commander has authorised it against their own credentials.",
    ),
    SopSection(
        "SOP-EVAC-01", "2", "Egress routing under evacuation",
        "Under evacuation, spectators are routed to the nearest gate by the shortest step-free "
        "route consistent with the density of the intervening corridors. Egress time is "
        "estimated using the conservative speed-density envelope. Where the estimate for any "
        "zone exceeds eight minutes, additional gates are opened before the announcement is "
        "made.",
    ),
    SopSection(
        "SOP-EVAC-01", "3", "Assisted egress",
        "Spectators with reduced mobility are directed to refuge points and evacuated by "
        "step-free routes only. A step-free route requirement is never relaxed under time "
        "pressure. Where no step-free route to a gate exists from a zone, that zone's refuge "
        "point is staffed before the fixture begins and evacuation is by assisted lift.",
    ),
    # ---------------- Weather ----------------
    SopSection(
        "SOP-WX-05", "1", "Lightning and severe weather hold",
        "On a lightning detection within eight kilometres, play is suspended and spectators in "
        "uncovered seating are moved to the covered concourse. The concourse is not an "
        "evacuation route in this state and its density must be monitored: a weather hold "
        "concentrates the crowd rather than dispersing it, and the resulting concourse density "
        "routinely exceeds match-day peak.",
    ),
    # ---------------- VIP ----------------
    SopSection(
        "SOP-VIP-04", "1", "VIP movement",
        "A VIP movement corridor is held for no more than four minutes and is never held "
        "through a corridor whose density exceeds level of service D. Where the planned route "
        "would breach this, the movement is re-timed rather than the corridor forced. A VIP "
        "movement never takes precedence over a medical dispatch.",
    ),
    # ---------------- Lost and found / child ----------------
    SopSection(
        "SOP-LOST-06", "1", "Lost child",
        "A lost child report is graded P2 and is handled at the lost and found point in the "
        "south-west concourse. The child's description is not broadcast on the public address "
        "system. Stewards in the reporting zone are notified directly and the gates are advised "
        "to watch for an unaccompanied minor attempting to exit.",
    ),
    # ---------------- Communications ----------------
    SopSection(
        "SOP-COMMS-07", "1", "Safety-critical announcements",
        "Any announcement concerning evacuation, medical emergency, fire, or crowd movement "
        "under emergency conditions is safety critical. Safety-critical announcements are made "
        "only from the controlled message catalogue, in languages whose catalogue entries carry "
        "a recorded human validation. Machine translation is not permitted in a safety-critical "
        "announcement in any language, in any circumstances.",
    ),
    SopSection(
        "SOP-COMMS-07", "2", "Informational announcements",
        "Wayfinding, catering, lost property and match information are informational. These may "
        "be generated conversationally and machine translated, subject to an automatic check "
        "that named entities -- gate numbers, seat numbers, block letters and times -- survive "
        "the translation unaltered.",
    ),
    SopSection(
        "SOP-COMMS-07", "3", "Fallback on translation failure",
        "Where a safety-critical announcement cannot be rendered in a spectator's language "
        "because the catalogue entry for that language is not human validated, the system falls "
        "back to the validated languages, displays the corresponding pictogram, and dispatches a "
        "steward to the affected zone. It does not machine translate, and it does not stay "
        "silent.",
    ),
)

SOP_INDEX = {s.ref: s for s in SOP_CORPUS}

# Which document *governs* an incident category. This is not a retrieval hint --
# it is procedure. The venue's safety case says a medical incident is handled
# under SOP-MED-03, and that is true whether or not a lexical retriever happens
# to rank it highly for the particular words a panicking volunteer used.
#
# This matters because grounding is strict: an agent may only cite what it was
# shown. Strict grounding plus imperfect recall means a *correct* brief gets
# rejected because the retriever missed the clause the model correctly applied.
# Pinning the governing document removes retrieval quality from the safety path
# and leaves it doing what it is good at -- finding the *additional*, less
# obvious sections that also bear on the situation.
SOP_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "medical": ("SOP-MED-03", "SOP-COMMS-07"),
    "crush": ("SOP-MED-03", "SOP-EVAC-01", "SOP-COMMS-07"),
    "fire": ("SOP-EVAC-01", "SOP-COMMS-07"),
    "security": ("SOP-EVAC-01", "SOP-COMMS-07"),
    "weather": ("SOP-WX-05", "SOP-EVAC-01", "SOP-COMMS-07"),
    "infrastructure": ("SOP-EVAC-01",),
    "queue": ("SOP-QUEUE-02", "SOP-COMMS-07"),
    "lost_and_found": ("SOP-LOST-06",),
    "vip": ("SOP-VIP-04",),
    "other": ("SOP-COMMS-07",),
}


def sections_of(doc_id: str) -> tuple[SopSection, ...]:
    return tuple(s for s in SOP_CORPUS if s.doc_id == doc_id)


def valid_ref(ref: str) -> bool:
    """True if ``ref`` names a real SOP section. The grounding check hangs off this."""
    return ref in SOP_INDEX
