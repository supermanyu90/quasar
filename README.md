# Quasar — a GenAI-native operating system for smart sporting venues

> Build a GenAI-enabled architecture that directly optimizes venue operations and
> elevates the tournament experience for fans, organizers, volunteers, and
> on-ground staff — covering dynamic crowd management, smart indoor navigation,
> real-time decision support, and multi-language assistance.

A stadium concourse fails the way a fluid fails: slowly, then all at once. Quasar
is a venue operating system built on one idea — **two planes and one barrier** —
that lets a generative model do the linguistic work it is good at without ever
letting it move a crowd on its own.

- A **deterministic plane** owns every number a life-safety decision depends on:
  crowd speed from the Weidmann speed–density relation, gate queues from an M/M/c
  model, routes from a density-aware Dijkstra over the venue graph.
- A **generative plane** owns what the numbers cannot express: what a panicked
  radio call means, which procedures conflict, how to tell a Marathi-reading fan
  that a corridor is closed.
- A **governance barrier** validates every model payload against a published
  schema, checks every citation against the document it quotes, corroborates every
  restated number against the sensor that produced it, and refuses to actuate a P0
  or P1 action without a human signature.

GenAI is **load-bearing** — the intake, synthesis, language and long-tail
situations are irreducibly linguistic — and it is **safe** because it never
computes a route, a capacity, or an evacuation time.

## Where to start

| If you want… | Read |
|---|---|
| The full engineering write-up (maths, architecture, safety proofs, rubric alignment) | **[`SUBMISSION.md`](SUBMISSION.md)** |
| How to run it, test it, and reproduce the crowd-model calibration | **[`quasar/README.md`](quasar/README.md)** |
| The problem-statement alignment, interactively | the **🎯 Guide** tab in the console (below) |

## The console

The web console opens on a **Guide** tab that maps the challenge to the running
system — every claim ends in a button that deep-links to the exact place you can
watch it hold. The other tabs:

- **🎛️ Control room** — the measured state first (nobody's opinion), then the
  agents' proposal with its corroboration score, then the human-in-the-loop
  barrier you can try to defeat, then actuation with routes recomputed from the
  graph.
- **🎟️ Attendee** — tap what you need (food, restroom, seat, quiet room) and get a
  crowd-aware, step-free, calm route drawn on the map, with the reply in your
  language; or ask the concierge in your own words.
- **✅ Readiness** — *can this venue open?* Two venues fail for opposite reasons on
  the same code: one has a stand with no step-free exit, the other cannot lawfully
  address its own majority-language crowd.
- **🧪 Pre-match** — fire generated failure scenarios at the venue before doors
  open; findings are defects in the *venue*, not bugs in the code.
- **🔗 Audit** — a tamper-evident hash chain; alter one record and every record
  after it stops matching.

Deep links drive the whole page from the URL — `?venue=`, `?tab=`, `?view=3d`,
`?find=`, `?mode=`, `?run=1`, `?approve=commander`, `?theme=` — which is how the
Guide's "show me" buttons work.

## Quick start

Python 3.11+. **The deterministic plane, the barrier, the schema validator, the
retriever and the entire test suite have no third-party dependencies** — a venue's
life-safety path should not break because a transitive dependency changed its
coercion rules. The `anthropic` SDK is optional (real Claude); Ollama is optional
(a real on-venue edge model).

```sh
cd quasar

# 202 tests, ~0.2 s, no network
PYTHONPATH=src python3 -m unittest discover -s tests -t .

# the CLI demo — the three modes change ONLY the model plane
PYTHONPATH=src:. python3 tools/demo.py              # recorded transcripts
PYTHONPATH=src:. python3 tools/demo.py --partition  # no model reachable at all
PYTHONPATH=src:. python3 tools/demo.py --live       # real Claude (needs credentials)

# the web console — http://127.0.0.1:8000
PYTHONPATH=src python3 tools/serve.py
```

Diffing `--partition` against the default is the fastest way to see exactly what
the generative layer is, and is not, responsible for.

## The venue is configuration, not code

The 18 venue specs in [`quasar/venues/`](quasar/venues) are data. Two are
hand-authored (Mumbai, Chennai); the 16 FIFA World Cup 2026 host venues are
generated from real public capacity and gate data and honestly flagged
`topology: "representative"` — a planning model, **not** a surveyed floor plan, and
the console says so. Adding a stadium is a JSON change validated by schema,
referential integrity, and connectivity — never a code change.

## Layout

```
SUBMISSION.md        the full engineering submission
quasar/
  README.md          how to run it (detailed)
  src/quasar/        the two planes, the barrier, the agents, the venue model
  api/               thin serverless endpoints (state, agent, actuate, guide, …)
  public/            the vanilla-JS console (index.html, app.js, styles.css)
  venues/            18 venue specs — adding a stadium is a data change
  tests/             the dependency-free test suite
  tools/             demo.py, serve.py, calibrate_gamma.py, gen_fifa_venues.py
  vercel.json        one-command deploy (functions are glob-matched)
```

Deployable to Vercel as-is (`outputDirectory: public`, `api/*.py` as serverless
functions). Live mode stays gated behind an operator key and server-side
credentials — a public URL with an open API key is a stranger's budget.
