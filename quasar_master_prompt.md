# Master Prompt — Quasar: GenAI-Native Operating System for Smart Sporting Venues

## ROLE
You are a senior AI systems architect and technical writer preparing a national-level AI hackathon submission. Your output must read as a production-ready engineering specification, not a concept note.

## OBJECTIVE
Design **Quasar**, a GenAI-enabled architecture that directly optimizes venue operations and elevates the tournament experience for **fans, organizers, volunteers, and on-ground staff**. The solution must demonstrably cover all four core tracks from the problem statement:

1. **Dynamic crowd management**
2. **Smart indoor navigation**
3. **Real-time decision support**
4. **Multi-language assistance**

## NON-NEGOTIABLE DESIGN MANDATES

### 1. GenAI must be load-bearing, not decorative
For every generative component, state explicitly what it does that a deterministic system cannot. Include at minimum:
- **RAG-grounded operations copilot**: LLM answers command-center queries grounded in venue SOPs, evacuation plans, and live telemetry (retrieval over a vector store of operational documents), with citations to source SOP sections.
- **Multilingual fan concierge**: conversational assistant supporting English, Hindi, Marathi, and at least 7 other Indian/international languages — handling wayfinding, seat location, food/washroom/medical queries, lost-and-found, and match information. Specify how translation quality is validated for safety-critical announcements.
- **Generative incident briefs**: the system converts raw multi-sensor telemetry into structured natural-language situation reports for organizers, with severity, affected zones, and recommended SOP actions.
- **Synthetic scenario generation**: GenAI generates edge-case crowd scenarios (gate failure, weather evacuation, VIP movement) used to stress-test the deterministic routing layer before match day.
- **Volunteer briefing generator**: auto-generated, role-specific and language-specific shift briefings for volunteers based on the day's fixtures, zones, and known risks.

### 2. Persona coverage — one concrete user journey each
Write a short end-to-end scenario (5–8 steps) for each persona showing where the system touches them:
- **Fan**: e.g., arrives at wrong gate → concierge reroutes in their language → density-aware navigation to seat → in-seat food ordering guidance.
- **Organizer**: e.g., queue utilization at Gate 3 projected to breach ρ ≥ 0.90 → decision-support agent proposes reallocation → HITL approval → push notifications dispatched.
- **Volunteer**: receives AI-generated shift brief in preferred language; escalates an incident via voice; gets SOP-grounded instructions.
- **On-ground staff (medical/security)**: receives generative incident brief with optimal density-aware route to the casualty.

### 3. Hybrid safety architecture (retain and strengthen)
- **Deterministic plane**: density-aware Dijkstra routing; Weidmann speed-density model v(ρ) = v_max(1 − e^(−γ(ρ_max − ρ))); M/M/c queueing for turnstiles with utilization ρ_q = λ/(cμ) and a 0.90 rerouting trigger.
- **Probabilistic GenAI plane**: multi-agent coordination (CrowdIntelligenceAgent, IncidentResponseAgent, ConciergeAgent, PlannerAgent, CommunicationAgent) communicating via strict JSON-schema contracts.
- **Human-in-the-loop barrier**: all P0/P1 actions require validated operator sign-off; deterministic SOP fallbacks trigger on low model confidence (< 0.85) or parsing failure.
- **Offline resilience**: edge-deployed local models (e.g., a small open-weights instruct model) on venue LAN so core functions survive network partition.

### 4. Smart indoor navigation as a first-class module
Not just backend routing — specify the user-facing experience: BLE beacon / Wi-Fi RTT positioning, conversational wayfinding through the multilingual concierge ("take me to the nearest accessible washroom"), accessibility-aware routes (step-free, low-density options for elderly/disabled fans), and dynamic re-routing when zone density changes mid-journey.

### 5. Implementation quality
- Fully typed, production-grade Python; no mock objects or stub safety logic.
- Include the core modules: density-aware routing engine, AI governance orchestrator with fallbacks, and a runnable integration test that demonstrates one full cross-track scenario (e.g., medical incident in a dense corridor triggering rerouting + multilingual crowd advisory + incident brief).
- Every agent payload validated against a published JSON Schema; show the schema.

## REQUIRED OUTPUT STRUCTURE
1. **Executive summary** (≤ 200 words) — problem, solution, why GenAI is essential to it.
2. **Mathematical foundations** — crowd fluid dynamics and queueing models with parameter values.
3. **System architecture** — layered ASCII diagram from IoT sensing → deterministic engine → GenAI agent layer → HITL barrier → actuation (PA, push, signage), plus the RAG and multilingual pipelines.
4. **GenAI component map** — a table listing each generative component, its model class, grounding source, failure mode, and fallback.
5. **Persona journeys** — the four scenarios from Mandate 2.
6. **Core implementation** — the Python modules and integration test.
7. **Rubric alignment** — for each judging dimension (problem alignment, innovation, code quality, architecture, security & privacy, accessibility, responsible AI), write 2–3 sentences justifying how the design satisfies it. **Do not assign yourself numeric scores.**
8. **Limitations and roadmap** — honest weaknesses (sensor latency, network partition, translation risk in emergencies) with concrete mitigations.

## ANTI-PATTERNS — DO NOT
- Do not bolt GenAI onto a deterministic system as an afterthought; every track must have a generative component earning its place.
- Do not frame the solution as a generic chatbot.
- Do not include self-assigned scores, marketing superlatives, or unverifiable performance claims.
- Do not use placeholder/mock code in safety-critical paths.
- Do not ignore the fan experience — operations excellence and fan delight must both be visible.

## TONE
Precise, systems-engineering register. Every claim traceable to a mechanism, equation, schema, or code path.
