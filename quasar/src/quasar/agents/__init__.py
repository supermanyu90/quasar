"""The six generative components, each with a deterministic twin.

Every agent here has three parts, and the third is the one that matters:

1. a **prompt** -- what we ask the model to do;
2. a **corroborator** -- a deterministic function that scores the model's answer
   against ground truth the model did not produce and cannot influence;
3. a **fallback** -- a deterministic implementation of the same output that runs
   when the model is unreachable, unparseable, ungrounded, or uncorroborated.

The fallback is not a stub. It is a real, SOP-derived implementation that the
venue could run on for the whole fixture with a degraded but safe experience.
That is the test of whether GenAI is load-bearing or decorative here: what is
*lost* when the fallback fires. For each agent the answer is stated in
``VALUE_OVER_FALLBACK`` -- and it is never "safety". Safety is what the fallback
guarantees. What GenAI buys is synthesis, language coverage, and the ability to
handle the inputs nobody enumerated in advance.

The corroborator is what makes the confidence gate meaningful. A model's
self-reported confidence is a generated token, not a calibrated probability, and
gating on it alone is theatre. Governance gates on
``min(self_reported, corroboration_score)``, and a *fatal* corroboration failure
-- an action naming a gate that does not exist, an incident graded below its SOP
floor -- forces the fallback no matter how confident the model claims to be."""

from __future__ import annotations

from quasar.agents.base import Agent, Corroboration
from quasar.agents.crowd import CrowdIntelligenceAgent, CrowdTask
from quasar.agents.incident import IncidentResponseAgent, IncidentTask
from quasar.agents.planner import PlannerAgent, PlanTask
from quasar.agents.concierge import ConciergeAgent, ConciergeTask
from quasar.agents.communication import CommsTask, CommunicationAgent
from quasar.agents.volunteer import VolunteerBriefingAgent, VolunteerTask

__all__ = [
    "Agent",
    "CommsTask",
    "CommunicationAgent",
    "ConciergeAgent",
    "ConciergeTask",
    "Corroboration",
    "CrowdIntelligenceAgent",
    "CrowdTask",
    "IncidentResponseAgent",
    "IncidentTask",
    "PlanTask",
    "PlannerAgent",
    "VolunteerBriefingAgent",
    "VolunteerTask",
]
