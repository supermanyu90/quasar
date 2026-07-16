# Quasar

A GenAI-native operating system for smart sporting venues.

The full engineering submission is in **[`../SUBMISSION.md`](../SUBMISSION.md)**. This
file is how to run it.

## Requirements

Python 3.11+. **The deterministic plane, the governance barrier, the schema
validator, the retriever and the entire test suite have no third-party
dependencies at all** — that is deliberate, not minimalism for its own sake: a
venue's life-safety path should not be able to break because a transitive
dependency changed its coercion rules.

The `anthropic` SDK is optional and is needed only to call a real model
(`pip install anthropic`).

## Run the tests

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -t .
```

202 tests, ~0.2 s, no network.

## Run the demo

```sh
PYTHONPATH=src:. python3 tools/demo.py              # recorded model transcripts
PYTHONPATH=src:. python3 tools/demo.py --partition  # no model reachable at all
PYTHONPATH=src:. python3 tools/demo.py --live       # real Claude (needs credentials)
```

The three modes change **only the model plane**. The routes, the queue metrics,
the announcements, the human-in-the-loop barrier and the audit chain are
identical in all three. Diffing `--partition` against the default is the fastest
way to see exactly what the generative layer is, and is not, responsible for.

## Reproduce the crowd-model calibration

```sh
PYTHONPATH=src python3 tools/calibrate_gamma.py
```

Prints the least-squares fit of the mandated speed–density curve against
Weidmann's published one, and the residual table that motivates `safe_speed`.

## Layout

```
venues/          venue specs — adding a stadium is a data change, not a code change
  national-stadium.json   Mumbai, 60,000, oval (hand-authored)
  coastal-arena.json      Chennai, 12,000, square (hand-authored)
  fwc-*.json              the 16 FIFA World Cup 2026 host venues, generated from
                          real public capacity/gate data — flagged topology:
                          "representative" (a model, not a surveyed floor plan)
src/quasar/
  types.py        domain types crossing every boundary
  crowd.py        Weidmann speed-density, Fruin LOS, the conservative envelope
  queueing.py     M/M/c turnstiles, Erlang-B/C, the 0.90 trigger
  venue.py        the venue graph primitives (the graphs themselves are data, in venues/)
  routing.py      density-aware Dijkstra; fan / accessible / staff / responder profiles
  positioning.py  graph-constrained particle filter over BLE / Wi-Fi RTT
  sops.py         the standing procedure corpus (the ground truth for RAG)
  rag.py          BM25 retrieval + the citation-grounding check
  schemas.py      published JSON Schemas + a dependency-free validator
  language.py     the two-tier translation safety policy + controlled catalogue
  llm.py          Claude / on-venue edge model / recorded transcripts + failover
  plane.py        the deterministic plane: assessment and executors
  agents.py       six agents, each with a corroborator and a deterministic twin
  governance.py   the barrier: schema, grounding, corroboration, policy, HITL, audit
  scenarios.py    synthetic scenario generation + the pre-match stress harness
  venue_spec.py   loads + validates a venue from JSON (schema + referential integrity)
  readiness.py    can this venue open? language + accessibility + topology audit
  amenities.py    the attendee amenity catalogue (icons, tags, calm-route rule)
  alignment.py    the problem-statement guide as data (served at /api/guide)
  web.py          the venue-aware control-room adapter
  venue_factory.py  parametric representative graphs from real venue metadata
tools/
  gen_fifa_venues.py   regenerate the 16 FIFA venue specs (deterministic)
  (venue specs now carry food/restroom/merch/lounge/water/charging/family/quiet
   amenity nodes; the Attendee tab routes to them, crowd-aware and step-free)
```
