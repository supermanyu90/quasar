"""The attendee-facing amenity catalogue: one source of truth.

Every amenity a fan can ask for is defined here once -- its node tag, its icon, its
label, and how the router should treat a request for it. The venue factory reads
this to place amenities, the wayfinding endpoint reads it to resolve a request, and
the console reads it (over the wire) to build its buttons. Keeping the taxonomy in
one place is what stops the map, the router, and the UI from drifting apart.

Two "amenities" are not places and are handled specially by the router:

* ``seat`` -- routes to the fan's own ticketed seat, not the nearest seating block
  (walking someone to the closest stand is the exact mistake they came to us with);
* ``assist`` -- a wheelchair-assistance *request*, which acknowledges and (in a
  real deployment) notifies stewarding, rather than routing to a fixed point.

A "calm" route is a *mode*, not a destination: it holds the fan below a comfortable
density (Fruin LOS D) for a sensory-friendlier walk, and is offered on top of any
amenity.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Amenity:
    key: str          # stable id used across API + UI
    icon: str         # emoji for the button
    label: str        # human label
    group: str        # "essentials" | "comfort" | "practical" | "inclusive"
    # How to resolve it. "tag" -> nearest node carrying `tags`; "seat" and "assist"
    # are handled specially by the router.
    kind: str = "tag"
    tags: tuple[str, ...] = ()


AMENITIES: tuple[Amenity, ...] = (
    # -- essentials (mostly already in every venue graph) --
    Amenity("food", "🍔", "Food & drink", "essentials", tags=("fnb",)),
    Amenity("restroom", "🚻", "Restroom", "essentials", tags=("washroom",)),
    Amenity("accessible_restroom", "♿", "Accessible restroom", "essentials",
            tags=("washroom", "accessible")),
    Amenity("seat", "🪑", "My seat", "essentials", kind="seat"),
    Amenity("first_aid", "⛑️", "First aid", "essentials", tags=("medical",)),
    Amenity("exit", "🚪", "Nearest exit", "essentials", tags=("gate",)),
    Amenity("lost_found", "🎒", "Lost & found", "essentials", tags=("lost_and_found",)),
    # -- retail & comfort --
    Amenity("merch", "🛍️", "Merch shop", "comfort", tags=("merch",)),
    Amenity("lounge", "🛋️", "Lounge", "comfort", tags=("lounge",)),
    Amenity("atm", "🏧", "ATM", "comfort", tags=("atm",)),
    # -- practical --
    Amenity("water", "🚰", "Water refill", "practical", tags=("water",)),
    Amenity("charging", "🔌", "Charging point", "practical", tags=("charging",)),
    Amenity("family", "👶", "Family / baby care", "practical", tags=("family",)),
    # -- inclusive --
    Amenity("quiet", "🧘", "Quiet / prayer room", "inclusive", tags=("quiet",)),
    Amenity("assist", "🦽", "Request wheelchair assist", "inclusive", kind="assist"),
)

BY_KEY = {a.key: a for a in AMENITIES}

GROUPS: tuple[tuple[str, str], ...] = (
    ("essentials", "Essentials"),
    ("comfort", "Shop & relax"),
    ("practical", "Practical"),
    ("inclusive", "Inclusive"),
)

# Fruin LOS-D density ceiling for a sensory-calm route. Anything busier than this
# is avoided when the fan asks for a calm walk.
CALM_MAX_DENSITY: float = 1.075
