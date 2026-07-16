# Quasar — a GenAI-native operating system for smart sporting venues

**Code:** [`quasar/`](quasar/)
**Tests:** `PYTHONPATH=src python3 -m unittest discover -s tests -t .` — 171 tests, ~0.1 s, no network, no third-party dependencies
**Console:** `PYTHONPATH=src python3 tools/serve.py` → http://127.0.0.1:8000 — a control room with a live venue map, the human-in-the-loop barrier you can try to defeat, and a **real local model** you can watch fail it
**CLI:** `PYTHONPATH=src:. python3 tools/demo.py [--partition]`

---

## 1. Executive summary

A stadium concourse fails the way a fluid fails: slowly, then all at once. Existing
tooling shows what the numbers are at 3.4 people per square metre, not what they
mean — while a volunteer three levels down describes a collapsed spectator over a
radio, in Marathi, to someone who does not speak it.

Quasar is a venue operating system with two planes and one barrier. A
**deterministic plane** owns every number a life-safety decision depends on: crowd
speed from the Weidmann speed–density relation, gate queues from an M/M/c model,
routes from a density-aware Dijkstra over the venue graph. A **generative plane**
owns what the numbers cannot express: what a panicked voice note means, which SOPs
conflict, how to tell a Marathi-reading fan that a corridor is closed. A
**governance barrier** validates every payload against a published schema, checks
every citation against the document it quotes, corroborates every restated number
against the sensor that produced it, and refuses to actuate a P0 or P1 action
without a commander's signature.

GenAI is essential because the hardest problems — intake, synthesis, language, the
situations nobody enumerated — are irreducibly linguistic. It is *safe* because it
never computes a route, a capacity, or an evacuation time.

*(200 words)*

---

## 1a. Problem-statement alignment & how to use

> **Build a GenAI-enabled architecture that directly optimizes venue operations and
> elevates the tournament experience for fans, organizers, volunteers, and on-ground
> staff — covering dynamic crowd management, smart indoor navigation, real-time
> decision support, and multi-language assistance.**

The console opens on a **Guide** tab (🎯) that *is* this alignment, made
interactive: every claim below ends in a button that deep-links to the exact place
in the running system where you can watch it hold. The mapping is not prose baked
into a page — it lives once, as structured data in `src/quasar/alignment.py`, is
served at `/api/guide`, and is covered by `tests/test_alignment.py`, which fails if
a track or persona loses its feature or a deep link points at a tab or venue that no
longer exists. What follows is that same table, for readers of the document.

**The four tracks → where the design meets each, and where to see it:**

| Track | How Quasar meets it | See it (Guide → “show me”) |
|---|---|---|
| Dynamic crowd management | Weidmann/Fruin turn counts into level-of-service; M/M/c flags any gate past the 0.90 trigger; the system proposes lane reallocation/diversion; the pre-match harness fires generated failure scenarios before doors open. | Control room · run the incident |
| Smart indoor navigation | Density-aware Dijkstra prices corridors by walking time; a step-free profile is a hard constraint; a graph-constrained particle filter answers “where am I”; the 3D view shows ramps against stairs. | Attendee · step-free route + 3D |
| Real-time decision support | A RAG copilot cites the SOP clause; the brief and plan are corroborated against ground truth the model did not produce; no P0/P1 actuates without a commander’s signature. | Control room · the barrier |
| Multi-language assistance | Two tiers: safety-critical messages come only from a human-validated catalogue (never machine-translated, with a gate-number-preservation check); informational help is generated behind quality gates. | Readiness · a Spanish/Tamil venue |

**The four users → what the system does for each:**

| Persona | Their need | Where |
|---|---|---|
| Fan / attendee | Find a restroom, seat or quiet room — step-free, calm-routed, in their language | Attendee tab |
| Organizer / commander | Ground truth, a proposed plan, and the accountable final say | Control room · approve as commander |
| Volunteer | A free-text radio report turned into a graded, SOP-cited brief | Control room · the report |
| On-ground staff (medic / security) | A cordon-safe, density-aware route to a casualty | Control room · actuation |

**Core objectives** — that GenAI is load-bearing not decorative (switch the model
plane to *Partition* and watch what is lost), that a fluent-but-wrong model can
never move the crowd alone (the hybrid safety architecture), that accessibility is a
hard constraint, and that safety-critical geometry is never fabricated and passed
off as surveyed — are each their own card in the Guide, with a deep link to the
proof. See also §5 (persona journeys) and §7 (rubric alignment).

---

## 2. Mathematical foundations

### 2.1 Crowd fluid dynamics

Pedestrian walking speed collapses with density. The specification mandates the
exponential-in-density form:

```
v(ρ) = v_max · (1 − e^(−γ(ρ_max − ρ)))
```

Weidmann's published relation is exponential in *reciprocal* density — in
pedestrian spacing, not in density itself:

```
v_c(ρ) = v_max · (1 − e^(−γ_c(1/ρ − 1/ρ_max)))
```

These are not the same curve, and the difference is safety-relevant. We
implemented both ([`crowd.py`](quasar/src/quasar/crowd.py)), fitted the mandated
form's single free parameter to the canonical one by least squares over the
operational band, and shipped the fitted constant rather than a hand-picked one
([`tools/calibrate_gamma.py`](quasar/tools/calibrate_gamma.py) reproduces it;
`test_crowd.py` asserts the shipped value is still the optimum).

| Parameter | Value | Source |
|---|---|---|
| `v_max` (free-flow speed) | 1.34 m/s | Weidmann (1993), mixed adult crowd |
| `ρ_max` (jam density) | 5.4 ped/m² | Weidmann (1993) |
| `γ_c` (canonical shape) | 1.913 ped/m² | Weidmann (1993) |
| `γ` (mandated form) | **0.2144 m²/ped** | least-squares fit, this work |
| fit residual | **RMSE 0.207 m/s** | over ρ ∈ [0.2, 5.4] |

**The best achievable fit is not good, and the error has a sign.** The mandated
form under-predicts speed at free flow and **over-predicts it under congestion** —
at ρ = 3.0 ped/m² it returns 0.54 m/s against Weidmann's 0.33 m/s. An
over-predicted speed in a jammed corridor is precisely the error a router must
never make: it prices a lethal edge as cheap.

So we retain the mandated model and strengthen it. Every speed that feeds a
*decision* — route cost, ETA, evacuation margin — goes through the **conservative
envelope**:

```
v_safe(ρ) = min( v(ρ), v_c(ρ) )
```

No edge can ever be priced faster than *both* models believe it to be. The
mandated form remains the model of record wherever it is the conservative one.
The cost of this choice is stated honestly in §8.

Derived quantities: specific flow `J(ρ) = ρ · v_safe(ρ)` [ped·m⁻¹·s⁻¹]; corridor
capacity `J · w`. Level of service uses Fruin's walkway thresholds, obtained by
inverting his pedestrian-area modules:

| LOS | A | B | C | D | **E** | **F** |
|---|---|---|---|---|---|---|
| density < (ped/m²) | 0.308 | 0.431 | 0.719 | **1.075** | **2.174** | ≥ 2.174 |

`ADVISORY_DENSITY = 1.075` (LOS D/E) starts flow shaping.
`CRITICAL_DENSITY = 2.174` (LOS E/F) forces a cordon and excludes spectators.

### 2.2 Queueing at the turnstiles

Each gate is an M/M/c queue over `c` open lanes with per-lane service rate `μ`:

```
offered load    a   = λ / μ                        [erlangs]
utilisation     ρ_q = a / c = λ / (c·μ)
P(wait)         C(c,a) = B / (1 − ρ_q(1 − B))      [Erlang C]
    where       B(c,a) = a·B(c−1,a) / (c + a·B(c−1,a)),  B(0,a) = 1   [Erlang B]
queue length    L_q = C(c,a) · ρ_q / (1 − ρ_q)
expected wait   W_q = L_q / λ
```

Erlang-C is evaluated through the **stable Erlang-B recursion**, not the factorial
form, which overflows at the lane counts and loads a real gate sees.
`test_queueing.py` checks it against the closed form (`C(1,a) = a` for M/M/1) and
against the factorial expression at small `c`.

| Parameter | Value |
|---|---|
| `μ` (per-lane service rate) | 0.55 ped/s (≈ 1.8 s per spectator) |
| `REROUTE_TRIGGER` | **ρ_q ≥ 0.90** |
| `TARGET_WAIT_S` | 180 s |

The 0.90 trigger is a *policy* threshold, not a stability one: the queue is stable
for any ρ_q < 1, but `W_q` grows without bound as ρ_q → 1. `test_queueing.py`
asserts the justification directly — wait at ρ_q = 0.98 is more than **4×** the
wait at 0.90.

### 2.3 Density-aware routing

Dijkstra over the venue graph with edge cost in *seconds*, not metres:

```
cost(e) = length(e) / (v_safe(ρ_e) · speed_factor(profile))
        + counterflow_penalty · length(e) · (ρ_e − 1)      [responders, ρ_e > 1]
        + stair_penalty                                     [stair edges]
```

Cost is strictly positive and finite (`v_safe ≥ V_MIN = 0.05 m/s`), so Dijkstra's
optimality argument holds. Four profiles, and the difference between them is the
safety design:

| Profile | step-free | staff corridors | max density | speed factor |
|---|---|---|---|---|
| `FAN` | no | no | 2.174 (LOS F) | 1.0 |
| `ACCESSIBLE` | **required** | no | **1.075 (LOS E)** | 0.6 |
| `STAFF` | no | yes | 2.174 | 1.0 |
| `RESPONDER` | no | yes | **none** | 1.0 + counterflow |

A medic must be able to reach a casualty *through* the crowd that created the
emergency, so a dense edge is *priced* for them, not removed. A fan is excluded
from it outright. An accessible fan is held to a stricter density limit than an
unimpeded adult — being sent into a LOS-E crush is not equally survivable for
everyone.

### 2.4 Indoor positioning

Raw RSSI trilateration in a hall full of bodies is worth roughly ±8 m — enough to
send a fan to the wrong vomitory. So we do not estimate a free-space position at
all. A **graph-constrained particle filter**
([`positioning.py`](quasar/src/quasar/positioning.py)) puts every particle *on a
walkable edge*: motion is dead reckoning along edges with transitions only at real
junctions (weighted by heading alignment), and the likelihood only ever scores
positions a human could physically occupy.

```
RSSI(d) = P_tx − 10·n·log₁₀(d/d₀)        n = 2.2, σ = 4 dB, P_tx = −45 dBm @ 1 m
w_i ∝ exp( −½ Σ_b ((rssi_b − RSSI(‖x_i − x_b‖)) / σ)² )
```

Systematic resampling when ESS < N/2. Map-matching is not a post-processing step
that can be wrong — it *is* the state space, so a fix can never land inside a wall.
`test_positioning.py` walks a synthetic fan from Gate 2 along the north concourse
with the filter seeded **uniformly over the entire venue** and asserts convergence
to **< 8 m** and to the correct edge.

---

## 3. System architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ L0  SENSING                                                                  │
│  CV crowd counters · turnstile logs · BLE gateways / Wi-Fi RTT · weather      │
│  volunteer radio + fan app (voice, text — free-form, 14 languages)            │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │  TelemetrySnapshot  (typed, validated)
                                ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ L1  DETERMINISTIC PLANE          ← owns every number a decision depends on    ║
║                                                                              ║
║   crowd.py      v_safe(ρ) = min(mandated, canonical) · Fruin LOS             ║
║   queueing.py   M/M/c · Erlang-B recursion → C · ρ_q ≥ 0.90 trigger          ║
║   routing.py    density-aware Dijkstra · 4 profiles · hard constraints       ║
║   positioning.py graph-constrained particle filter (map-matched by design)   ║
║   plane.py      Assessment  +  severity_floor()  +  the EXECUTORS            ║
╚═══════════════════════════════╤══════════════════════════════════════════════╝
                                │  Assessment: hotspots, LOS, queue metrics,
                                │  severity floor  ── ground truth, not opinion
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ L2  GENERATIVE PLANE   (Claude Opus 4.8 · on-venue edge model · failover)     │
│                                                                              │
│   CrowdIntelligence → IncidentResponse → Planner → Communication             │
│                    ↘  Concierge (fan-facing)   ↘  VolunteerBriefing          │
│                                                                              │
│   Every agent: prompt + deterministic CORROBORATOR + deterministic TWIN      │
│   Agents never compute a route, a capacity, or a time. They SELECT.          │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │  JSON payload (unvalidated, untrusted)
                                ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║ L3  GOVERNANCE BARRIER            ← the only door. governance.py             ║
║                                                                              ║
║   1 transport   model down? → edge model → deterministic twin. never invent. ║
║   2 parse       extract JSON  ─┐                                             ║
║   3 schema      validate       ├─ 1 repair attempt, violations fed back      ║
║   4 grounding   every citation exists AND was in the retrieved context       ║
║   5 corroborate deterministic check vs. ground truth the model didn't make   ║
║   6 confidence  gate on min(self_reported, corroboration) ≥ 0.85             ║
║   7 policy      action allowlist · blast-radius caps · role authority        ║
║   8 HUMAN       P0/P1 → commander's signature. no exceptions, no override.   ║
║                                                                              ║
║   audit: SHA-256 hash chain — alter one record, every later record breaks    ║
╚═══════════════════════════════╤══════════════════════════════════════════════╝
                                │  approved plan (a SELECTION, never a number)
                                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ L4  ACTUATION — the deterministic plane RECOMPUTES everything, now           │
│                                                                              │
│   PA + push (controlled catalogue only) · signage · cordon · lane            │
│   reallocation · responder dispatch (route computed here, at this instant)   │
└──────────────────────────────────────────────────────────────────────────────┘
```

**The RAG pipeline** (`sops.py` → `rag.py`):

```
SOP corpus, section-level (15 sections, 7 documents, stable ids)
   │
   ├─ PINNED: the document that PROCEDURE says governs this category
   │          (medical → SOP-MED-03 + SOP-COMMS-07).  Not a retrieval hint.
   │
   └─ BM25 (k₁=1.5, b=0.75, hand-rolled, exactly reproducible)
              │
              ▼
   context rendered WITH refs inline: "[SOP-MED-03#2] Clearing an approach route…"
              │
              ▼
   model emits citations ──► check_grounding(): every ref must (a) EXIST and
                             (b) have been in the context. Otherwise → fallback.
```

Pinning the governing document is load-bearing, and we found out why the hard way:
grounding is strict, so a retriever that *misses* the clause the model correctly
applied causes a **correct brief to be rejected**. Pinning takes retrieval recall
off the safety path and leaves BM25 doing what it is good at — surfacing the
*additional*, non-obvious sections.

**The multilingual pipeline** (`language.py`) is split by consequence, not by
language:

```
                      ┌─────────── is a wrong answer dangerous? ───────────┐
                      │                                                     │
              YES: TIER 1                                          NO: TIER 2
   (evacuation, medical, crowd movement, weather)     (wayfinding, food, lost property)
                      │                                                     │
   model may ONLY choose a template id + fill           generative translation ALLOWED,
   TYPED slots (gate enum, zone enum, integer).         behind two automatic gates:
   There is no free-text field on this path             ① round-trip back-translation
   ANYWHERE — see the BROADCAST schema.                    similarity ≥ 0.45 (chrF-style)
                      │                                  ② NAMED-ENTITY PRESERVATION:
   sentences are human-written, human-validated            every gate number, seat, block
   per language. renderer REFUSES any language             letter and time survives, or
   not marked HUMAN_VALIDATED — even if a                   the translation is discarded
   machine draft for it exists.                          │
                      │                                  └─ on failure: show the SOURCE
   no validated entry?  →  fall back to the                  language. An English answer
   validated languages + raise the PICTOGRAM +               a fan can half-read beats a
   dispatch a STEWARD. (SOP-COMMS-07#3)                      fluent Marathi lie about
   Not machine translation. Not silence.                     which gate to use.
```

---

## 4. GenAI component map

Every generative component, what it does that a deterministic system cannot, and
what happens when it fails. **The fallback column is not a stub — each one is a
real, SOP-derived implementation the venue could run on all night.** The test of
whether GenAI is load-bearing is what is *lost* when the fallback fires, and the
answer is never "safety".

| Component | Model class | Grounded in | Failure mode | Detection | Fallback — and what is lost |
|---|---|---|---|---|---|
| **CrowdIntelligenceAgent** | Opus 4.8, `effort: medium` | live `Assessment` (measured densities, LOS, queue metrics) | restates a measured density wrongly; misses a hotspot; wrong `action_required` | corroborator compares **every restated number against the sensor**; any drift > 0.05 ped/m² is fatal | Deterministic hotspot list from telemetry. *Lost:* the sentence that says Gate 3 is what is **loading** CORR-NE — the causal link is not in the telemetry, it is in the operator's world model. |
| **IncidentResponseAgent** (RAG copilot) | Opus 4.8, `effort: high` | pinned + BM25-retrieved SOP sections, cited by `DOC#section` | fabricates a section number; cites a section it was never shown; **under-grades severity** | `check_grounding()` + severity floor from `plane.severity_floor()`. Grading below the floor is fatal. | SOP emitted verbatim at the procedural floor. *Lost:* turning *"someone's gone down near the food stand, people are pushing"* into a structured brief in eight seconds. Free-text intake is irreducibly a language problem — there is no form field for panic. |
| **PlannerAgent** | Opus 4.8, `effort: high` | `Assessment` + retrieved SOPs + venue graph | infeasible action (14 lanes at a 12-lane gate); **cordons before announcing** (SOP-MED-03#3); **a plan of valid actions that ignores the emergency** (§6.5 — a live model did exactly this) | every action re-priced against the deterministic plane; ordering checked; **completeness checked** — every LOS-F corridor cordoned, every casualty reached, every breaching gate relieved, or the plan must escalate. All fatal. | The playbook, executed literally. *Lost:* the compound case the playbook has no page for — a medical emergency inside the corridor that is *also* the diversion route for a saturated gate. |
| **ConciergeAgent** (fan-facing) | Opus 4.8, `effort: low` (latency) | BLE map-matched position + venue graph | misclassifies an **emergency** as informational | deterministic keyword second opinion; disagreement resolves to the **alarming** reading, not the average | Keyword matcher + menu, in English. *Lost:* the entire multilingual product. **This is the one place the model IS the feature** — and even here it does not choose the route, and the moment it flags safety-critical it loses the pen entirely. |
| **CommunicationAgent** | Opus 4.8, `effort: low` | approved template id + typed slots | substitutes a template; alters approved slots | corroborator rejects any deviation from the approved broadcast | Broadcast venue-wide in the fixture's default languages. *Lost:* audience selection. The fallback is safe, louder than necessary, and desensitising. |
| **VolunteerBriefingAgent** | Opus 4.8, `effort: medium` | fixture, zone, role, SOPs | wrong zone/language; writes an evacuation instruction | corroborator checks zone + language; Tier-1 content is structurally unreachable from this path | The same generic English PDF for everyone. *Lost:* 400 personalised briefings. A briefing nobody reads is not a control. |
| **ScenarioGenerator** (pre-match) | Opus 4.8, `effort: high` | venue graph | invents geography; physically impossible density | schema (`density ≤ 6.0`) + reference check against the graph | Seeded sampler. *Lost:* **the interaction cases** — and those are the entire point. A random sampler will never propose "the gate fails *while* the VIP movement is running *and* the weather hold has pushed the upper tier into the concourse". |

**Prompt caching** is architectural, not an optimisation: each agent's system
prompt and the retrieved SOP context form a stable prefix with the cache
breakpoint on the last stable block; volatile telemetry goes *after* it, in the
user turn, so it never invalidates the prefix. A control room issues hundreds of
queries against an identical SOP corpus in a night.

---

## 4a. The venue is configuration, not code

A venue operating system that only works for the venue it was written for is a
demo of that venue, not a product. Quasar reads a **venue spec** — a JSON document
([`venues/*.json`](quasar/venues/)) describing the graph, the zones, the languages
the crowd actually speaks, and the fixture — and configures itself from it. Adding a
stadium is a data change ([`venue_spec.py`](quasar/src/quasar/venue_spec.py)); the
spec goes through the same published-schema validator as every model payload, then a
referential-integrity pass (every edge names real nodes, the graph is connected — a
stranded stand found here, not at 19:40).

Two venues ship, and they are structurally different, not one scaled down:

| | National Stadium | Coastal Arena |
|---|---|---|
| | Mumbai, 60,000, oval | Chennai, 12,000, square |
| Crowd languages | English, Hindi, Marathi, Tamil | **Tamil**, English, Hindi |
| Step-free egress | West stand has none | every stand has a ramp |

**The payoff is the readiness audit** ([`readiness.py`](quasar/src/quasar/readiness.py)),
which asks a question a venue-blind system cannot: *can this venue safely open its
gates?* The same code gives the two venues **opposite blocking verdicts**:

- **National Stadium is blocked by its architecture.** The west stand has a
  staircase and no ramp — a wheelchair user cannot leave it unaided, on any night.
  No software fixes that; the remedy is a ramp or a staffed refuge point.
- **Coastal Arena is blocked by its language.** Tamil is the majority language of
  its crowd, and the Tier-1 catalogue has Tamil only as a *machine draft*, which the
  safety gate refuses. So if that arena has to evacuate, most of the people in it
  will not be told to in a language they read. Actuating the arena incident shows it
  directly: the medical announcement goes out in English and Hindi, and **Tamil is
  refused** — pictogram and steward instead of words. The audit calls this a blocker
  and says so in plain terms: *"this is a translation procurement, not an engineering
  task. No change to this software will fix it."*

That second finding is the entire argument for making the venue configurable. A
system hardcoded to one stadium cannot even ask the question, let alone notice that
the answer is different — and lethal — somewhere else. (`test_venues.py` asserts both
verdicts.)

## 5. Persona journeys

### 5.1 Fan — Priya, 34, Marathi, arriving with her mother who cannot manage stairs

1. Priya taps in at **Gate 5 (south)**. Her seats are in the **north stand**. She has come to the wrong end of a 60,000-seat stadium.
2. The app's particle filter, seeded at the gate she scanned through, map-matches her to edge `E-G5` within seconds — no GPS under a roof.
3. She types, in Marathi: *"माझी आई पायऱ्या चढू शकत नाही आणि आम्ही चुकीच्या गेटवर आलो आहोत."* The **ConciergeAgent** classifies `intent: wayfinding`, `destination_tag: seating`, `safety_tier: informational`, and replies in Marathi.
4. The concierge decides **where**. The router decides **how** — and is bound by constraints the concierge cannot override. Under the `ACCESSIBLE` profile it returns a route via `RAMP-N` that touches **no staircase**, **no service corridor**, and **no corridor above LOS D**.
5. Mid-journey the north-east corridor is cordoned (§5.2). Her route is recomputed around it before she reaches it — she never learns it existed.
6. In-seat, she asks where to get water. Tier-2: generated, machine-translated to Marathi, and put through the **entity-preservation gate** — the stand number in the answer must survive the translation or the answer is shown in English instead.
7. Her mother needs a washroom. *"Nearest accessible washroom"* resolves to `WC-N-ACC` — **not** the washroom 17 m away, which is up a staircase. Proximity does not override a hard constraint.

*Verified by `test_integration_cross_track.py::TestFanJourney` and `test_routing.py::TestNearestTagged`.*

### 5.2 Organiser — the duty commander in the control room

1. Telemetry: `CORR-NE` at **3.4 ped/m² (LOS F)**; `G3` at **ρ_q = 0.98**, above the 0.90 trigger.
2. **CrowdIntelligenceAgent**: *"Gate 3 is at 0.98 utilisation and is feeding the north concourse faster than it can clear. That inflow is what is loading CORR-NE… These are one problem, not two: relieving Gate 3 is the upstream fix for the corridor."* Twelve red numbers become one story.
3. A volunteer reports a collapse at `C-N3`. The deterministic plane sets the **severity floor**: because crowd pressure at the scene exceeds LOS E, SOP-MED-03#1 makes this **P0, not P1**. *Procedure* decides this — the model may grade it higher, never lower.
4. **PlannerAgent** proposes four actions, each carrying the SOP clause that authorises it: broadcast → cordon → dispatch → open lanes.
5. The commander sees the plan, the citations, and the corroboration score. She tries nothing; the system has already tried: an unapproved P0 plan **raises `ApprovalRequired`** and a steward's credentials **raise `NotAuthorised`**.
6. She signs. Only then does actuation begin — and every number is recomputed from the graph at that instant, because the cordon set has changed since the plan was written.
7. Outcome: `G3` **0.98 → 0.82**. The corridor is cordoned. The medic is moving. The whole chain is in a tamper-evident audit log.

*Verified by `test_integration_cross_track.py::TestFullIncident`.*

### 5.3 Volunteer — Sunil, gate steward, north concourse, prefers Marathi

1. 17:40, pre-shift. **VolunteerBriefingAgent** generates *his* briefing — his zone, his role, in Marathi — not the 40-page generic PDF.
2. It names the three things that will actually happen to him: `CORR-NE` is 4 m wide and jams at half-time; Gate 3 will fill with late arrivals; here is how to escalate. Every instruction cites a SOP section, and the citations are checked against the corpus.
3. 20:14. He sees a man go down. He holds his radio and says what he sees — no form, no dropdown, no English: *"someone's gone down near the north-east food stand… people are pushing past."*
4. That sentence becomes a structured P0 incident brief with a zone, a severity, and three cited SOP clauses. This is the step no deterministic system performs.
5. His handset shows the approved instruction from the **controlled catalogue**, in Marathi. It is not a translation of anything the model wrote — it is a human-validated sentence with typed slots filled in.
6. He holds the line at the junction. The diversion was announced **before** the cordon went in, so the fans arriving toward him already know.

### 5.4 On-ground staff — the east medical team

1. Dispatch arrives with a **generative incident brief**: severity, casualty location, what the crowd is doing around them, the SOP clauses that apply.
2. It also arrives with a route, and the route is the whole point:

| | path | distance | actual traversal time |
|---|---|---|---|
| distance-shortest router | `MED-2 → C-E1 → C-N3` **through the LOS-F corridor** | **103 m** | **381 s** |
| Quasar (density-aware) | `MED-2 → SVC-NE → C-N3` **via the inner service ring** | 145 m | **163 s** |

   The correct route is **41 % longer and 2.3× faster**. A router that optimises
   metres sends the medic into the crush that caused the emergency.
3. The cordon binds them too — SOP-MED-03#2 forbids a medical team from a LOS-F corridor. They do not get an exemption; they get the service ring.
4. If the cordon leaves *no* route, the system does not improvise. It raises `NoRouteError`, logs it, and escalates to the commander for manual dispatch. *(`test_governance.py::test_an_unreachable_casualty_escalates_rather_than_failing_silently`)*

---

## 6. Core implementation

15 modules, 145 tests, **zero third-party dependencies on the safety path** — including the JSON Schema validator and the BM25 retriever, both hand-rolled deliberately: a venue's life-safety validation should not be able to break because a transitive dependency changed its coercion rules.

### 6.1 The conservative envelope — `crowd.py`

```python
def safe_speed(rho: float) -> float:
    """Conservative envelope of the two models -- the speed of record for every
    decision (route cost, ETA, evacuation margin).

    Taking the pointwise minimum guarantees no edge is ever priced as faster
    than *both* models believe it to be. This is what stops the mandated form's
    congested-regime over-prediction from making a jammed corridor look cheap.
    """
    return min(weidmann_speed(rho), weidmann_canonical_speed(rho))
```

### 6.2 The schema that makes a lie unrepresentable — `schemas.py`

The most important thing in the schema file is a field that **does not exist**:

```python
{
    "title": "BROADCAST",
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "params", "sop_ref"],
    "properties": {
        "type": {"const": "BROADCAST"},
        "sop_ref": _SOP_REF,                     # ^[A-Z]{2,6}-[A-Z]{2,8}-\d{2}#\d{1,3}$
        "params": {
            "type": "object",
            "additionalProperties": False,
            # Note what is absent: there is no free-text field. A broadcast
            # names a catalogue template and supplies typed slots. The model
            # chooses *which* approved sentence to say, never what it says.
            "required": ["template_id", "zone", "slots"],
            "properties": {
                "template_id": _ID,
                "zone": _ID,
                "slots": {
                    "type": "object",
                    "additionalProperties": {
                        "anyOf": [{"type": "string", "maxLength": 40},
                                  {"type": "integer"}]
                    },
                },
            },
        },
    },
}
```

A model cannot write the words of a public safety announcement, because the
contract gives it **nowhere to put them**.
*(`test_schemas.py::test_a_broadcast_cannot_carry_free_text`)*

The validator refuses to ignore a keyword it does not implement — a validator that
silently skips an unknown keyword is worse than none, because it manufactures
confidence it has not earned.

### 6.3 Corroboration — the reason the confidence gate means anything

A model's self-reported confidence is a *generated token*, not a calibrated
probability. Gating on it alone is theatre. So governance gates on
`min(self_reported, corroboration_score)`, where the corroboration score is
computed **deterministically, against ground truth the model did not produce and
cannot influence**:

```python
# agents.py — CrowdIntelligenceAgent.corroborate
for h in payload["hotspots"]:
    actual = a.density(h["edge_id"])
    if abs(h["density_ped_m2"] - actual) > 0.05:
        return Corroboration.fail(
            f"restates {h['edge_id']} density as {h['density_ped_m2']} "
            f"(measured {actual:.2f})"
        )
```

```python
# agents.py — PlannerAgent.corroborate
# Procedural ordering (SOP-MED-03#3): the diversion is announced before
# the cordon goes in, never after. A model that gets this backwards is
# fluent, plausible and wrong, and no schema can catch it.
kinds = [a["type"] for a in payload["actions"]]
if "BROADCAST" in kinds and "CORDON_EDGE" in kinds:
    if kinds.index("CORDON_EDGE") < kinds.index("BROADCAST"):
        return Corroboration.fail(
            "cordons the corridor before announcing the diversion, contrary to "
            "SOP-MED-03#3; arriving spectators would be walked into a closed corridor"
        )
```

A *fatal* corroboration failure forces the deterministic path **regardless of how
confident the model claims to be**. `test_governance.py` proves it: a payload that
restates a 3.4 ped/m² corridor as 1.1 ped/m² with `confidence: 0.99` is discarded.

### 6.4 The human barrier

```python
def check(self, plan_id: str, severity: Severity, approval: Approval | None) -> None:
    """Raises unless this exact plan is cleared to actuate."""
    if not severity.requires_human_approval:
        return
    if approval is None:
        raise ApprovalRequired(...)
    if approval.plan_id != plan_id:          # re-planning invalidates the signature
        raise ApprovalRequired(...)
    if not approval.approved:
        raise ApprovalRequired(...)
    if not approval.operator.may_approve(severity):   # RBAC: commander only, for P0/P1
        raise NotAuthorised(...)
```

Approval is not a boolean the orchestrator sets. It is a **signed object** naming
an operator whose role is checked against the severity and an exact plan id. The
system cannot manufacture one. Approval for one plan does not transfer to another.

### 6.5 What happened when we ran a real model at it

Everything above was built and tested against *recorded* model output. That is a
comfortable place to be, and it is a lie of omission: I wrote the transcripts, so I
wrote them correct. The barrier had never once been attacked by a model that was
genuinely trying to be helpful and getting it wrong.

So we pointed it at a real one — `gemma3:4b`, running locally through Ollama with
no API key and no internet, which is exactly the on-venue edge box the architecture
already specifies. **It found two holes in my safety layer within three runs.**

**Hole 1 — the corroborator validated slot *names*, not slot *values*.** The model
proposed a broadcast addressed to zone `"NORTH-EAST"`. There is no such zone. The
check compared the *keys* of the slot map against the template and waved it
through, because the keys were right. A public announcement was one step from
naming a place that does not exist.

**Hole 2 — and this is the serious one — I checked that every action was *feasible*,
and never that the plan was *complete*.** The model produced this:

```
BROADCAST          MSG-GATE-DIVERT   (a gate-diversion message, for a medical emergency)
OPEN_LANES         G3, 10 lanes
DISPATCH_RESPONDER MED-3 → C-N3
```

Every one of those actions is executable. The gate exists, the lane count is within
the installed reserve, the route is walkable. It scored **1.00 on corroboration**.
And it leaves a man lying unconscious in a corridor at level of service F **with no
cordon**, while announcing a queue diversion over the PA.

That is precisely the "fluent, plausible, and wrong" failure this whole document
claims to defend against, and it sailed straight through. A schema cannot catch it.
A confidence score cannot catch it. Only a deterministic check of *what the plan
failed to do* catches it.

The fix is a **completeness check**: the deterministic plane states what must be
answered, and the plan must answer all of it — or escalate to a human and say it
cannot.

```python
# agents.py -- PlannerAgent._unaddressed
for edge_id in task.assessment.critical_edges:
    if edge_id not in closed:
        gaps.append(f"{edge_id} is at LOS F and is neither cordoned nor rerouted around")

if task.casualty_node is not None and task.casualty_node not in reached:
    gaps.append(f"no responder is dispatched to the casualty at {task.casualty_node}")

for gate_id in task.assessment.breaching_gates:
    if gate_id not in relieved:
        gaps.append(f"{gate_id} is above the 0.90 utilisation trigger and is not relieved")
```

An honest *"I cannot handle this, wake a human"* remains a legitimate answer — a plan
carrying `ESCALATE` is accepted with a note. Silently doing nothing is not.

Six regression tests now hold both holes shut (`test_governance.py`), and replaying
the model's own plan against the hardened corroborator scores it **0.00, fatal**.

**One more result worth quoting.** On a later run the crowd agent reported
**self-reported confidence 0.95** on an assessment that **corroborated at 0.60** — it
had drifted from the sensor readings and had no idea. It fell back. If the
specification's 0.85 confidence floor had been gated on the model's own number, as a
literal reading invites, that payload would have been accepted and an operator would
have acted on it. This is the single clearest piece of evidence in the project that
`min(self_reported, corroboration)` is not pedantry.

### 6.6 The runnable integration test

`tests/test_integration_cross_track.py` drives one incident across all four tracks,
end to end, with **nothing stubbed on the safety path**. The model's contribution is
a *recording* so CI runs offline; the router, the M/M/c model, the catalogue, the
schema validator, the grounding check, the corroborators, the policy engine and the
HITL barrier all execute for real, and it is **their** behaviour that is asserted:

| Assertion | Result |
|---|---|
| severity floor from measured crowd pressure | **P0** (not P1) — set by procedure, not the model |
| every citation exists **and** was in the retrieved context | ✔ |
| unapproved P0 plan actuates | ✗ `ApprovalRequired` |
| steward approves a P0 plan | ✗ `NotAuthorised` |
| medic route crosses the cordon | ✗ — routed via the service ring |
| medic ETA vs. the "shortest" path through the crush | **163 s vs 381 s** |
| `G3` utilisation after lane reallocation | **0.98 → 0.82**, below trigger |
| announcement languages delivered | `en`, `hi`, `mr` |
| Tamil (machine draft exists) | **refused** → pictogram + steward dispatched |
| public announcement contains the internal id `CORR-NE` | ✗ — says *"the North-East corridor"* |
| diversion announced **before** the cordon | ✔ (SOP-MED-03#3) |
| audit hash chain verifies | ✔ |

And then `TestNetworkPartition` **pulls the plug on the model entirely** — no cloud,
no edge box — and reruns the same incident. The venue still cordons the corridor,
still dispatches the medic by a safe route, still relieves the gate, still
announces in three languages, still demands a commander's signature, and still
produces a verifiable audit chain. What it loses is stated and asserted: the
causal sentence, and the ability to answer a Marathi speaker at all.

### 6.7 Pre-match stress harness

`scenarios.py` fires generated scenarios at the deterministic layer and checks
invariants. **Run against the reference venue, it finds a real defect in the
building:**

```
[critical] step-free-egress-exists: SEAT-W has no step-free route to any gate in an
empty, fully open venue. This is a defect in the building, not the crowd: a staffed
refuge point is mandatory here before the fixture (SOP-EVAC-01#3)
```

The west stand has a staircase and no ramp. No amount of crowd management fixes
that. The harness distinguishes **three causes** of lost accessible egress, because
each has a different owner: the *building* is wrong (staff a refuge — an architect's
problem, found before the season), the *closure plan* is wrong (don't apply that
closure without a refuge — an operations problem), or the route is merely *crowded*
(hold it clear — a stewarding problem, solvable tonight in minutes). Reporting all
three as "no step-free egress" would send the wrong person to fix the wrong thing.

---

## 7. Rubric alignment

**Problem alignment.** All four tracks are implemented as executing code and are
exercised together in one integration test, not four demos in a trench coat:
dynamic crowd management (`crowd.py`, `queueing.py`, the 0.90 trigger and the
lane-reallocation executor), smart indoor navigation (`routing.py` +
`positioning.py` + conversational wayfinding), real-time decision support (the RAG
copilot, the incident brief, the planner, the HITL barrier), and multi-language
assistance (14 languages on the Tier-2 path, a human-validated controlled
catalogue on Tier-1). The scenario that drives the test — a medical incident inside
the corridor that a saturated gate is loading — was chosen because it *forces* the
tracks to interact.

**Innovation.** The contribution is not "an LLM in a stadium". It is the
**corroborator**: a deterministic function per agent that scores the model's output
against ground truth the model did not produce and cannot influence, making the
mandated confidence floor mean something instead of gating on a generated token.
Alongside it: the **two-tier language policy**, which refuses machine translation on
any safety-critical path and refuses a *machine-drafted* catalogue entry even when
one exists; the **conservative speed envelope**, which stops a mandated model's known
over-prediction from pricing a jammed corridor as cheap; and the **graph-constrained
particle filter**, where map-matching is the state space rather than a
post-processing step that can be wrong.

**Code quality.** Fully typed, frozen dataclasses, `slots`, exceptions rather than
sentinel returns on every safety path (`NoRouteError` is raised, never returned, so
"no safe route exists" cannot be mistaken for "route of length zero"). 145 tests
run in ~0.1 s with **no third-party dependencies**, including a hand-rolled JSON
Schema validator that *refuses to ignore* a keyword it does not implement. The
tests found and fixed four real bugs during development, all recorded in this
document rather than quietly patched: an entity check that would have rejected
every correct Indic translation; an executor that evaluated a plan's second half
against a venue state its first half had already changed; a strict-grounding /
retrieval-recall interaction that rejected *correct* briefs; and a stress harness
that reported tautologies as findings.

**Architecture.** The two-plane split is enforced by construction, not convention:
the generative plane has no import path to the router, the queueing model, or the
PA system. It emits JSON; the barrier validates it; the deterministic plane
recomputes every number *at the moment of actuation*, because the cordon set can
change between planning and dispatch and a stale route is exactly as dangerous as a
wrong one. The single sentence that describes the whole system: **the model may
choose which medical post to dispatch from; the ETA, the path, and the guarantee
that the path does not cross the cordon are computed from the graph, every time.**

**Security & privacy.** Every agent payload is validated against a published,
*closed* schema (`additionalProperties: false` everywhere), so an unmodelled field —
`"override_hitl": true` — is rejected at the door. Prompt injection through a
broadcast is not defended against, it is made **unrepresentable**: there is no
free-text field on the Tier-1 path, and slot values are checked against the venue's
real gate and corridor ids, so injected prose cannot reach the PA system even as an
entity. Blast-radius caps bind the deterministic fallback exactly as they bind the
model. Credentials are never hardcoded (zero-arg SDK constructor, environment or
`ant auth` profile). The audit log is a SHA-256 hash chain: altering one record
breaks every record after it, so the "what did it propose and who approved it"
question survives contact with an inquiry.

**Accessibility.** Step-free is a **hard constraint**, never a preference: the
`ACCESSIBLE` profile removes stepped edges from the graph, and when no path
remains, the router raises rather than degrading — it will not send a wheelchair
user down a staircase because the alternative was long. It also holds an
assisted-mobility fan to a stricter density limit (LOS E) than an unimpeded adult
(LOS F), because a crush is not equally survivable for everyone. The stress harness
audits accessible egress from every stand before the gates open and found a real
gap in the reference venue's west stand. Every safety announcement carries a
pictogram fallback, and public announcements name locations the way the *signage*
does, not the way the database does.

**Responsible AI.** Every generative component has a deterministic twin the venue
could run on all night, and the submission states — and the tests assert — exactly
what is lost when each fires. The system is designed around the assumption that the
model will be **fluently, confidently wrong**, and we did not have to imagine what
that looks like: we ran a real model at it and it produced a plan of individually
valid actions that left a man unconscious in a LOS-F corridor with no cordon, at a
corroboration score of 1.00 (§6.5). That plan is the reason a completeness check now
exists. On a later run the same model reported **0.95 self-reported confidence on an
assessment that corroborated at 0.60** — which is the empirical case for gating on
`min(self_reported, corroboration)` rather than on the model's own number, as a
literal reading of the specification's 0.85 floor would invite. And on the one
question where a model's judgement is genuinely load-bearing — "is this fan reporting
an emergency?" — a disagreement between the model and the deterministic second
opinion resolves to the **alarming** reading, not the average, because the costs are
asymmetric.

---

## 8. Limitations and roadmap

Written in the order we would fix them.

**1. The mandated speed–density form is a poor fit, and we are paying for it.**
The best least-squares fit still leaves RMSE 0.207 m/s. We contain the dangerous
half of that error with `safe_speed`, but the containment has a price: at free flow
the envelope returns **0.89 m/s where Weidmann gives 1.34** — roughly **33 %
pessimistic** on an uncongested corridor. Every ETA we show a fan is therefore
conservative, and evacuation estimates are conservative in the safe direction but
will over-provision gates. *Mitigation:* the canonical form is already implemented
and tested; switching the model of record is a one-line change once the mandate
permits, and `test_crowd.py` will fail loudly if a recalibration ever makes
`safe_speed` redundant.

**2. Sensor latency is unmodelled.** The assessment treats a `TelemetrySnapshot` as
instantaneous truth. CV crowd counting has a 2–5 s pipeline latency, and at LOS F a
crowd front moves ~1 m/s — so a cordon decision may be acting on a corridor state
that is 5 m stale. *Mitigation:* the `trend` field (rising/steady/falling) already
exists on every hotspot and is currently only *reported*. It should feed a
short-horizon extrapolation so the trigger fires on *projected* density. Sensor
health and staleness are not tracked at all: a frozen camera currently reads as a
calm corridor, which is the worst possible failure. **A dead sensor must be treated
as a full corridor, not an empty one** — this is the highest-priority gap in the
system and it is not yet implemented.

**3. Translation coverage is three languages, not fourteen.** Tier-1 is
human-validated in English, Hindi and Marathi only. Every other language falls back
to pictogram + steward, which is *safe* and *not good enough* for a national
fixture. This is honestly a translation-vendor procurement problem rather than an
engineering one — the code is structured so that landing a validated entry is a
data change with no code change — but it is the single biggest gap between this
system and one that could actually open the gates. The machine drafts shipped in
the catalogue are deliberately marked unvalidated and deliberately refused; a draft
that exists is more dangerous than one that does not, because it looks ready.

**4. Network partition degrades the fan experience to nothing.** The edge model
keeps the copilot alive, but under a *total* partition the concierge falls back to
a keyword matcher answering in English — and the tests assert that the Marathi
utterance is simply missed. *Mitigation:* the Tier-1 catalogue and the router work
entirely offline, so safety survives; a small on-device intent classifier (not a
generative model) would restore basic multilingual wayfinding without a network.

**5. Grounding is strict, which trades a hallucination risk for a recall risk.**
An agent may only cite what it was shown. We found during development that this
causes *correct* briefs to be rejected when BM25 misses a relevant clause, and
mitigated it by pinning the SOP that procedure says governs the category. That fix
is only as good as `SOP_BY_CATEGORY`, which is hand-maintained. *Mitigation:* a
hybrid retriever (the `Retriever` already accepts a reranker) plus an alerting
signal when a fallback fires *because* of grounding — a spike there means the
corpus, not the model, is the problem.

**6. The corroborators are the security boundary, they are hand-written, and we
have proof they leak.** Every one is a deterministic function we wrote, and a gap in
one is a hole in the barrier. This is not a theoretical worry: pointing a real model
at the system for the first time found **two holes in three runs** (§6.5), and one of
them would have let a plan through that left a casualty in a LOS-F corridor with no
cordon. Both are now closed and regression-tested, but the honest inference is that
**there are more**, and that every hole found so far was found by a model *trying to
help* — not by one trying to get past us.

*Mitigation, in priority order:* (a) point the scenario generator at the
**corroborators** rather than only the routing layer — generating plans designed to
slip past the checks is a far better use of a model than generating densities;
(b) run the full agent suite against a real model on every CI run, not against
recordings, because recordings are written by the same person who wrote the checks
and share its blind spots; (c) treat every corroborator gap found in production as a
Sev-1, because by construction it is one.

**7. The frontier model has still never been run against this barrier.** Everything
demonstrated end-to-end has been either a recording or a 4B edge model. The design
assumes the primary is a frontier model and the edge box is the survival path — that
assumption is *documented* and *untested*. Until a real Opus-class run is logged, the
GenAI component map in §4 is a specification, not a measurement.

**8. Not yet built.** Multi-incident contention (two P1s competing for one medical
team; the planner currently reasons about one incident at a time). VIP-movement
scheduling against live density (SOP-VIP-04 is in the corpus and cited, but no
executor implements it). Wi-Fi RTT fusion is modelled but only BLE is wired.
Per-fan push delivery, load-shedding under a full-stadium broadcast, and the
Console/ops UI are all out of scope here.
