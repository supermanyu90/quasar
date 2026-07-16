"""Parametric venue models, from real metadata.

You cannot pull a real stadium's *internal* graph: no venue publishes its concourse
topology with corridor widths and verified step-free routes -- that lives in private
safety surveys. Quasar's whole thesis is that safety-critical geometry is never
fabricated and passed off as real, so this module does the honest thing instead.

Given a venue's *public* identity -- name, city, country, seating capacity -- it
generates a **representative** graph: an oval bowl sized to the capacity, with a
gate count and lane counts scaled from it, a concourse ring, four stands each
reachable step-free, medical posts on a staff-only service ring, and amenities.
Every spec it emits is stamped ``topology: "representative"``, and the readiness
audit and the console surface that stamp. It is correct for scale, for planning,
and for demonstrating that the system operates at 16 venues from data -- and it is
explicitly *not* a substitute for a surveyed floor plan.

The generation is deterministic in the venue id, so the same venue always produces
the same graph (a stable demo, and a reproducible artifact).
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping

# Language priors by country. The FIRST entry is treated as the majority language
# by the readiness audit, so ordering is load-bearing: a Mexican venue's crowd is
# majority Spanish, and Spanish is not human-validated in the Tier-1 catalogue, so
# the audit correctly reports those venues as not ready to open.
LANGUAGES_BY_COUNTRY: Mapping[str, list[str]] = {
    "USA": ["en", "es"],
    "Canada": ["en", "fr"],
    "Mexico": ["es", "en"],
}

_ZONE_NAMES = {
    "NORTH": {"en": "North Stand", "es": "Tribuna Norte", "fr": "Tribune Nord"},
    "EAST": {"en": "East Stand", "es": "Tribuna Este", "fr": "Tribune Est"},
    "SOUTH": {"en": "South Stand", "es": "Tribuna Sur", "fr": "Tribune Sud"},
    "WEST": {"en": "West Stand", "es": "Tribuna Oeste", "fr": "Tribune Ouest"},
    "GATE": {"en": "the entry gates", "es": "las puertas", "fr": "les portes"},
}


def _seed(venue_id: str) -> int:
    return int(hashlib.sha256(venue_id.encode()).hexdigest()[:8], 16)


def _zone_of(angle_deg: float) -> str:
    a = angle_deg % 360
    if a < 45 or a >= 315:
        return "NORTH"
    if a < 135:
        return "EAST"
    if a < 225:
        return "SOUTH"
    return "WEST"


def generate_spec(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Build a representative venue spec from public metadata.

    ``meta`` needs ``id``, ``name``, ``city``, ``country``, ``capacity`` and
    optionally ``fifa_name``.
    """
    vid = meta["id"]
    capacity = int(meta["capacity"])
    country = meta["country"]
    seed = _seed(vid)

    # Gate count and lane counts scale with capacity, bounded to plausible ranges.
    gates = max(6, min(10, round(capacity / 9000)))
    installed_lanes = max(6, min(18, round(capacity / (gates * 900))))

    # Bowl size scales with the square root of capacity (area ~ capacity).
    scale = math.sqrt(capacity / 68000.0)
    a_out, b_out = 150 * scale, 130 * scale  # outer ellipse (gates)
    a_in, b_in = 105 * scale, 90 * scale  # concourse ring
    a_seat, b_seat = 60 * scale, 52 * scale  # seating bowl

    languages = LANGUAGES_BY_COUNTRY.get(country, ["en"])

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    beacons: list[str] = []

    def node(nid, name, x, y, level, zone, tags):
        nodes.append({
            "id": nid, "name": name, "x": round(x, 1), "y": round(y, 1),
            "level": level, "zone": zone, "tags": tags,
        })

    def edge(eid, u, v, length, width, kind, step_free, staff=False, label=None):
        e = {
            "id": eid, "u": u, "v": v, "length_m": round(length, 1),
            "width_m": round(width, 1), "kind": kind, "step_free": step_free,
        }
        if staff:
            e["staff_only"] = True
        if label:
            e["label"] = label
        edges.append(e)

    def dist(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    # -- gates and concourse ring -------------------------------------------
    gate_pos: dict[str, tuple[float, float]] = {}
    conc_pos: dict[str, tuple[float, float]] = {}
    conc_ids: list[str] = []
    for i in range(gates):
        theta = 2 * math.pi * i / gates
        deg = math.degrees(theta)
        zone = _zone_of(deg)
        gx, gy = a_out * math.sin(theta), b_out * math.cos(theta)
        cx, cy = a_in * math.sin(theta), b_in * math.cos(theta)
        gid, cid = f"G{i + 1}", f"C{i + 1}"
        node(gid, f"Gate {i + 1}", gx, gy, 1, "GATE", ["gate"])
        node(cid, f"Concourse {i + 1}", cx, cy, 1, zone, ["concourse"])
        gate_pos[gid] = (gx, gy)
        conc_pos[cid] = (cx, cy)
        conc_ids.append(cid)
        beacons.append(cid)
        # gate hall
        edge(f"E-{gid}", gid, cid, dist(gx, gy, cx, cy), 8.0, "vomitory", True)

    # concourse ring (a cycle -> the graph is connected by construction). Every
    # third link is a narrower "corridor" pinch point, which is where crushes form.
    ring_edges: list[tuple[str, str, str]] = []
    for i in range(gates):
        u, v = conc_ids[i], conc_ids[(i + 1) % gates]
        ux, uy = conc_pos[u]
        vx, vy = conc_pos[v]
        pinch = i % 3 == 0
        eid = f"R-{i + 1}"
        edge(eid, u, v, dist(ux, uy, vx, vy),
             3.5 if pinch else 7.0, "corridor" if pinch else "concourse", True,
             label=f"Ring {i + 1}")
        ring_edges.append((eid, u, v))

    # -- seating bowl: four stands, each reachable step-free --------------------
    stands = [
        ("SEAT-N", "North Bowl", 0, b_seat, "NORTH"),
        ("SEAT-E", "East Bowl", a_seat, 0, "EAST"),
        ("SEAT-S", "South Bowl", 0, -b_seat, "SOUTH"),
        ("SEAT-W", "West Bowl", -a_seat, 0, "WEST"),
    ]

    def nearest_concourse(x, y, exclude=()):
        best, bd = None, 1e18
        for cid in conc_ids:
            if cid in exclude:
                continue
            cx, cy = conc_pos[cid]
            d = dist(x, y, cx, cy)
            if d < bd:
                best, bd = cid, d
        return best

    for sid, sname, sx, sy, zone in stands:
        node(sid, sname, sx, sy, 2, zone, ["seating"])
        c_stair = nearest_concourse(sx, sy)
        c_ramp = nearest_concourse(sx, sy, exclude={c_stair})
        sc = conc_pos[c_stair]
        rc = conc_pos[c_ramp]
        # A stair (short, steep) and a ramp (long, gentle) -- modern venues are
        # step-free, so every stand gets a ramp. The two hand-authored venues show
        # the failure case (a stand with no ramp); these real venues pass.
        edge(f"STAIR-{sid}", c_stair, sid, dist(sc[0], sc[1], sx, sy), 5.0, "stair", False)
        edge(f"RAMP-{sid}", c_ramp, sid, dist(rc[0], rc[1], sx, sy) * 1.9, 3.0, "ramp", True)

    # -- amenities, medical, control, service ----------------------------------
    c_east = nearest_concourse(a_in, 0)
    c_west = nearest_concourse(-a_in, 0)
    c_north = nearest_concourse(0, b_in)
    c_south = nearest_concourse(0, -b_in)

    def spur(nid, name, tags, anchor, dx, dy, kind="corridor", step_free=True, info=None):
        ax, ay = conc_pos[anchor]
        node(nid, name, ax + dx, ay + dy, 1, nodes_zone(anchor), tags)
        if info:
            nodes[-1]["info"] = info
        edge(f"SP-{nid}", anchor, nid, max(12.0, dist(ax, ay, ax + dx, ay + dy)),
             3.0, kind, step_free)

    def nodes_zone(cid):
        return next(n["zone"] for n in nodes if n["id"] == cid)

    # Amenities the attendee companion can route to. Spread around the ring so the
    # "nearest one" actually differs by where the fan is standing. Accessible spurs
    # are step-free; the plain washroom is up a stair, so the accessible-restroom
    # request genuinely resolves to a different, reachable node.
    spur("WC-N-ACC", "North Accessible Restroom", ["washroom", "accessible"], c_north, -14, 6)
    spur("WC-S-ACC", "South Accessible Restroom", ["washroom", "accessible"], c_south, 14, -6)
    spur("WC-E", "East Restroom", ["washroom"], c_east, 12, -8, kind="stair", step_free=False)
    spur("FNB-N", "North Food Court", ["fnb"], c_north, 16, 8, info="Cashless · veg & halal options")
    spur("FNB-S", "South Kiosks", ["fnb"], c_south, -16, 8, info="Cashless · snacks & drinks")
    spur("LOST", "Lost & Found", ["lost_and_found"], c_south, -16, -8, info="Open all match")
    spur("MED-1", "First Aid (East)", ["medical"], c_east, 18, 4, info="Staffed paramedics")
    spur("MED-2", "First Aid (West)", ["medical"], c_west, -18, -4, info="Staffed paramedics")
    spur("CONTROL", "Command Centre", ["control"], c_west, -16, 10)

    # Retail & comfort
    spur("MERCH", "Team Store", ["merch"], c_east, 16, -10, info="Official kit & souvenirs")
    spur("LOUNGE", "Members' Lounge", ["lounge"], c_north, 20, -6, info="Seating, bar, step-free")
    spur("ATM", "Cash Machine", ["atm"], c_west, 12, 10)
    # Practical
    spur("WATER", "Water Refill", ["water"], c_south, 8, 10, info="Free · bring a bottle")
    spur("CHARGE", "Charging Point", ["charging"], c_east, 10, 12, info="USB-A / USB-C")
    spur("FAMILY", "Family & Baby Care", ["family"], c_north, -20, -4, info="Changing, feeding, quiet")
    # Inclusive
    spur("QUIET", "Quiet / Prayer Room", ["quiet"], c_west, -12, -12, info="Low light, low noise")

    # Service ring: a staff-only bypass so responders reach the far side of a
    # cordon without threading the crowd. Level 0, beneath the bowl.
    node("SVC-E", "Service Corridor East", a_in * 0.7, 12, 0, "EAST", ["service"])
    edge("SVC-1", "MED-1", "SVC-E", 40.0, 3.0, "service", True, staff=True)
    edge("SVC-2", "SVC-E", c_north, 55.0, 3.0, "service", True, staff=True)

    # -- the fixture: a crush at a pinch point during the match ----------------
    # Pick the first pinch corridor and the gate feeding it.
    pinch_eid, pinch_u, pinch_v = ring_edges[0]
    casualty = pinch_u
    feed_gate = f"G{conc_ids.index(pinch_u) + 1}"

    # Gate arrivals scaled so the feeding gate breaches the 0.90 trigger.
    mu = 0.58
    breach_arrivals = round(0.97 * installed_lanes * mu, 2)
    calm_arrivals = round(0.45 * installed_lanes * mu, 2)
    other_gates = [f"G{i + 1}" for i in range(gates) if f"G{i + 1}" != feed_gate][:2]

    zones = {z: dict(names) for z, names in _ZONE_NAMES.items()}

    # A fan at a gate opposite their seat, needing step-free help, in the crowd's
    # majority language.
    fan_gate = f"G{(conc_ids.index(pinch_u) + gates // 2) % gates + 1}"

    return {
        "schema": "quasar.venue_spec.v1",
        "id": vid,
        "name": meta["name"],
        "fifa_name": meta.get("fifa_name", meta["name"]),
        "city": meta["city"],
        "country": country,
        "capacity": capacity,
        "topology": "representative",
        "languages": languages,
        "service_rate_per_s": mu,
        "zones": zones,
        "nodes": nodes,
        "edges": edges,
        "beacons": beacons[: min(len(beacons), 16)],
        "scenario": {
            "name": f"Crush at a concourse pinch point, {meta['city']}",
            "casualty_node": casualty,
            "category": "medical",
            "background_density": 0.35,
            "edge_density": {
                pinch_eid: 3.7,
                ring_edges[1][0]: 1.9,
                f"E-{feed_gate}": 1.2,
            },
            "gates": (
                [{"gate_id": feed_gate, "arrival_rate_per_s": breach_arrivals,
                  "open_lanes": max(1, installed_lanes - 2), "installed_lanes": installed_lanes}]
                + [{"gate_id": g, "arrival_rate_per_s": calm_arrivals,
                    "open_lanes": max(1, installed_lanes - 3), "installed_lanes": installed_lanes}
                   for g in other_gates]
            ),
            "report": {
                "id": f"INC-{seed % 9000 + 1000}",
                "reporter_role": "steward",
                "language": "en",
                "text": (
                    "someone's gone down at the pinch by the concourse, the crowd off "
                    "the stand is still pushing through and it's only a few metres wide "
                    "here, we can't hold them back"
                ),
            },
            "fan": {
                "at_node": fan_gate,
                "seat": "SEAT-S" if casualty in ("C1", "C2") else "SEAT-N",
                "language": languages[0],
                "accessible": True,
                "utterance": _fan_utterance(languages[0]),
            },
        },
    }


def _fan_utterance(language: str) -> str:
    return {
        "en": "My mother can't manage stairs and we've come in at the wrong gate. How do we get to our seats?",
        "es": "Mi madre no puede subir escaleras y hemos entrado por la puerta equivocada. ¿Cómo llegamos a nuestros asientos?",
        "fr": "Ma mère ne peut pas monter les escaliers et nous sommes entrés par la mauvaise porte. Comment rejoindre nos sièges ?",
    }.get(language, "How do we get to our seats without stairs?")


# The 16 host venues of the 2026 FIFA World Cup (Canada / Mexico / USA), with real
# public identity and tournament capacity. The FIFA name is the sponsor-free name
# used during the tournament. Source: FIFA / Wikipedia, verified 2026-07.
FWC_2026_VENUES: tuple[Mapping[str, Any], ...] = (
    {"id": "fwc-mexico-city", "name": "Estadio Azteca", "fifa_name": "Estadio Azteca",
     "city": "Mexico City", "country": "Mexico", "capacity": 80824},
    {"id": "fwc-new-york", "name": "MetLife Stadium", "fifa_name": "New York New Jersey Stadium",
     "city": "East Rutherford", "country": "USA", "capacity": 80663},
    {"id": "fwc-dallas", "name": "AT&T Stadium", "fifa_name": "Dallas Stadium",
     "city": "Arlington", "country": "USA", "capacity": 70649},
    {"id": "fwc-los-angeles", "name": "SoFi Stadium", "fifa_name": "Los Angeles Stadium",
     "city": "Inglewood", "country": "USA", "capacity": 70492},
    {"id": "fwc-kansas-city", "name": "Arrowhead Stadium", "fifa_name": "Kansas City Stadium",
     "city": "Kansas City", "country": "USA", "capacity": 69045},
    {"id": "fwc-bay-area", "name": "Levi's Stadium", "fifa_name": "San Francisco Bay Area Stadium",
     "city": "Santa Clara", "country": "USA", "capacity": 68827},
    {"id": "fwc-houston", "name": "NRG Stadium", "fifa_name": "Houston Stadium",
     "city": "Houston", "country": "USA", "capacity": 68777},
    {"id": "fwc-philadelphia", "name": "Lincoln Financial Field", "fifa_name": "Philadelphia Stadium",
     "city": "Philadelphia", "country": "USA", "capacity": 68324},
    {"id": "fwc-atlanta", "name": "Mercedes-Benz Stadium", "fifa_name": "Atlanta Stadium",
     "city": "Atlanta", "country": "USA", "capacity": 68239},
    {"id": "fwc-seattle", "name": "Lumen Field", "fifa_name": "Seattle Stadium",
     "city": "Seattle", "country": "USA", "capacity": 66925},
    {"id": "fwc-miami", "name": "Hard Rock Stadium", "fifa_name": "Miami Stadium",
     "city": "Miami Gardens", "country": "USA", "capacity": 64478},
    {"id": "fwc-boston", "name": "Gillette Stadium", "fifa_name": "Boston Stadium",
     "city": "Foxborough", "country": "USA", "capacity": 64146},
    {"id": "fwc-vancouver", "name": "BC Place", "fifa_name": "Vancouver Stadium",
     "city": "Vancouver", "country": "Canada", "capacity": 52497},
    {"id": "fwc-monterrey", "name": "Estadio BBVA", "fifa_name": "Estadio Monterrey",
     "city": "Guadalupe", "country": "Mexico", "capacity": 51243},
    {"id": "fwc-guadalajara", "name": "Estadio Akron", "fifa_name": "Estadio Guadalajara",
     "city": "Zapopan", "country": "Mexico", "capacity": 45664},
    {"id": "fwc-toronto", "name": "BMO Field", "fifa_name": "Toronto Stadium",
     "city": "Toronto", "country": "Canada", "capacity": 43036},
)
