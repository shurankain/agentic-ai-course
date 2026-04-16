## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Multi-Agent Systems
**Previous:** [[01_MAS_Basics|Multi-Agent Systems Basics]]
**Next:** [[03_Agent_Orchestration|Agent Orchestration]]

---

# Multi-Agent System Patterns

## Architectural Patterns as a Design Language

Patterns are proven architectural solutions that can be combined for specific tasks. They prevent reinventing the wheel and create a shared vocabulary: the phrase "supervisor pattern" conveys an entire layer of information.

## Choosing a Pattern

### Criteria

**Decomposability** — can the task be broken down into independent subtasks?
**Predictability** — are the types of subtasks known in advance?
**Coordination** — how tightly coupled are the agents?
**Fault tolerance** — what is the cost of a single agent failing?
**Latency** — what delay is acceptable?
**Scalability** — what load growth is expected?

### Algorithm

Fixed steps? → Handoff
Known domains? → Mixture of Experts
Central coordination acceptable? → Supervisor/Hierarchical
No → Peer-to-Peer

### Metrics

**Coupling Score** — ratio of interaction operations to total operations. <0.2 (peer-to-peer), 0.2–0.5 (supervisor), >0.5 (hierarchy).

**SPOF Risk** — ratio of critical path length to redundancy. >3 — avoid centralization.

**Expertise Overlap** — fraction of tasks any agent can handle. >0.7 (simple supervisor), <0.3 (MoE/hierarchy).

## Supervisor Pattern

One manager agent, the rest are workers. The supervisor receives a task, identifies the required workers, distributes subtasks, monitors execution, processes results, and produces the final response.

Advantages: simplicity, predictability, clear responsibility, full visibility, single point of control. Suitable for tasks with clear decomposition.

Disadvantages: single point of failure (supervisor failure halts the system), bottleneck as workers scale, poor dynamics for tasks with unpredictable subtasks.

## Hierarchical Agents

Multi-level tree structure: top managers → middle managers → workers. Each level abstracts the details of the level below, exposing a high-level interface to the level above.

Use cases: complex multi-level tasks (creating a marketing campaign: strategist → channel agents → copywriters/designers). Complexity isolation — the top level has no knowledge of lower-level details.

Challenges: communication overhead (information passes up/down, losing detail), cross-branch coordination (communication through a distant common ancestor).

## Peer-to-Peer Collaboration

Equal interaction with no central authority. Agents discover each other, negotiate, and share information on their own. Coordination emerges bottom-up as an emergent property of local interactions.

Advantages: fault resilience (no single point of failure), dynamic adaptation (agents join/leave freely), efficiency for tasks with unpredictable decomposition.

Difficulties: global consistency (each agent sees only its neighbors), preventing duplication/conflicts. Solutions: consensus protocols, reputation systems, market mechanisms.

## Debate Pattern

Adversarial system: agents take opposing positions, challenge each other, and a **judge agent synthesizes** the best elements from each position. The classic structure: proposer, critic, judge — but variations exist with multiple proposers, multiple critics, or rotating roles.

Use cases: tasks requiring high reliability (financial analysis: "bull" vs. "bear"), generation and verification (code author vs. reviewer), factual accuracy on contested or ambiguous topics where a single model might commit to a wrong answer confidently.

**Real-world example:** Perplexity Computer's **Model Council** sends the same query to multiple LLMs simultaneously, compares their answers, and selects the best response (or synthesizes from multiple). This is the Debate Pattern applied at the model level — using diversity of models as the source of independent perspectives.

Key requirement: true independence of opponents (different models, prompts, data sources). Identical models produce identical blind spots. The judge must be capable of evaluating the quality of arguments, not just averaging positions.

## Swarm Intelligence

Inspired by social insects: a colony exhibits complex behavior from simple individual rules. No central control — global behavior emerges from local interactions.

**Stigmergy** — communication through environmental modification. Agents leave "marks" (traces in shared storage), others detect them and adjust behavior. Shorter paths are reinforced faster. Types: quantitative (mark intensity), qualitative (mark types → specialization), temporal (decay → recency priority).

**Emergence** — system properties absent in individual elements. The system finds optimal solutions without explicit optimization and adapts without reprogramming.

**Self-organization** requires: positive feedback (amplifying successes), negative feedback (limiting growth), randomness (new patterns), sufficient interactions.

Use cases: optimization and search in vast, poorly structured spaces. A thousand agents sharing discoveries outperform a single one.

Drawback: unpredictability, difficulty debugging emergent bugs.

**OpenAI Swarm** — originally released as an experimental framework (October 2024) demonstrating lightweight agents with handoffs instead of orchestration and stateless execution. Swarm's core patterns — agent handoffs, routine-based delegation — were incorporated into the production **OpenAI Agents SDK** (March 2025), which adds guardrails, tracing, and MCP tool integration. Swarm itself remains educational; the Agents SDK is the production successor.

## Mixture of Experts

A set of narrowly specialized experts on a single level. A router analyzes the task and directs it to the appropriate expert (the router is not a manager — it only determines relevance).

Advantages: deep specialization in a niche (SQL agent vs. generalist), system versatility (new task types → new experts without rework).

Critical point: routing quality. Incorrect routing → poor results. Alternative: multiple experts + response aggregation (more expensive but more reliable).

## Handoff Pattern

Sequential task transfer between specialist agents. Each performs its stage and passes the result to the next (assembly line).

Example: parser (text extraction) → analyzer (entities, facts) → enricher (external information) → synthesizer (final report).

Advantages: clear separation of responsibilities, independent development/testing, easy to add/replace stages.

Limitations: sequential nature (failure/delay in one stage → blocks the entire chain), poor support for feedback loops (late-stage results requiring revision of earlier stages).

## Fan-Out/Fan-In

Dispatch subtasks to multiple agents in parallel, then aggregate results. The pattern: an orchestrator decomposes the task into independent subtasks, dispatches each to a specialist agent, waits for all to complete (barrier synchronization), then combines results.

**Common applications:**
- Research agents — parallel web searches across different sources, results synthesized into a report
- Code review — parallel analysis of different files or concerns (security, performance, style)
- Data analysis — parallel queries to different databases or data sources
- Multi-document summarization — parallel summarization of individual documents, then a synthesis pass

**Implementation:** A directed acyclic graph (DAG) where the fan-out node creates N parallel tasks and the fan-in node waits for all N to complete. LangGraph, CrewAI, and most orchestration frameworks support this natively. The critical design decision is the **aggregation strategy** at the fan-in node: simple concatenation, LLM-based synthesis, voting, or weighted combination.

**Trade-off:** Fan-Out/Fan-In trades cost (N parallel agent calls instead of 1) for latency (all run simultaneously) and quality (specialized agents per subtask). It is the most cost-efficient multi-agent pattern when subtasks are genuinely independent — the total wall-clock time equals the slowest subtask, not the sum.

---

## When Single-Agent Suffices

Not every problem needs multiple agents. Reasoning models (o3, Claude with extended thinking, Gemini with thinking) increasingly internalize multi-step reasoning that previously required multi-agent orchestration. A single agent with tools and a reasoning model can often solve tasks that, in 2024, required 3-4 specialized agents in a pipeline.

**The cost of multi-agent coordination:**
- ~10x cost per query compared to single-agent (each agent is an LLM call)
- Added latency from sequential agent handoffs
- Debugging complexity (tracing decisions across multiple agents)
- Failure modes multiply (each agent can fail, miscommunicate, or drift)

**The 2026 production trend:** Deterministic orchestration for flow control (the developer designs the sequence of steps) + LLM for bounded decisions within each step (the model decides what to do within each step, not where the flow goes). This is Anthropic's "Level 2 — Workflows" in practice. See [[../../03_AI_Agents_Core/02_Agent_Architectures|Agent Architectures]] for the Augmented-LLM Progression.

**Decision rule:** Prototype with a single agent + reasoning model first. Add multi-agent orchestration only when you hit clear capability boundaries: the task requires genuinely different expertise (security review ≠ performance review), parallelism provides measurable latency benefit, or the context required exceeds what a single agent can hold.

---

## Key Takeaways

**Supervisor** — simplicity, single point of failure. For tasks with clear decomposition.
**Hierarchical** — scalability, complexity isolation. For multi-level tasks.
**Peer-to-peer** — resilience, complex coordination. For dynamic tasks.
**Debate** — adversarial approach → quality. For decision reliability.
**Swarm** — collective intelligence. For large search spaces.
**MoE** — specialization + routing. For heterogeneous queries.
**Handoff** — pipeline. For pipeline workflows.
**Fan-Out/Fan-In** — parallel dispatch + aggregation. For independent subtasks needing speed.

The best systems combine patterns. But first, verify that a single agent with a reasoning model cannot solve the problem — multi-agent adds 10x cost and significant complexity.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Multi-Agent Systems
**Previous:** [[01_MAS_Basics|Multi-Agent Systems Basics]]
**Next:** [[03_Agent_Orchestration|Agent Orchestration]]
