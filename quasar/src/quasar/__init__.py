"""Quasar -- a GenAI-native operating system for smart sporting venues.

Two planes, one barrier:

* the *deterministic plane* (:mod:`quasar.crowd`, :mod:`quasar.queueing`,
  :mod:`quasar.routing`, :mod:`quasar.positioning`) owns every number that a
  life-safety decision depends on;
* the *generative plane* (:mod:`quasar.agents`, :mod:`quasar.rag`) owns
  interpretation, language, and proposal;
* the *governance barrier* (:mod:`quasar.governance`) validates every payload
  against a published schema, corroborates it against the deterministic plane,
  and refuses to actuate a P0/P1 action without a human signature.

Nothing in the generative plane computes a route, a capacity, or an evacuation
time. It selects among options the deterministic plane has already priced.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
