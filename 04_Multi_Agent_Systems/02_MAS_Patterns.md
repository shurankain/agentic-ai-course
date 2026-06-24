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

## Hierarchical (Manager-Worker)

The most common multi-agent pattern in production. A manager agent decomposes the task, distributes subtasks to specialized workers, and synthesizes results. The key distinction from the supervisor pattern: multiple levels of hierarchy are possible, with middle managers coordinating groups of workers.

**Why this works in practice.** The manager-worker pattern maps naturally to how human organizations solve complex problems: a project lead breaks down the work, assigns it to specialists, reviews results, and assembles the final deliverable. LLMs are remarkably good at task decomposition — breaking a complex request into subtasks is a natural language task that plays to the model's strengths.

**Production example — Kiro's spec-driven approach.** Amazon's Kiro IDE uses a hierarchical pattern: a planning agent generates requirements → a design agent produces architecture → a task agent decomposes into implementation steps → coding agents write code. Each level operates independently with a clear handoff contract. The result: spec-driven development where the human reviews artifacts at each level, not just the final code.

**Another production pattern — code review.** A review manager receives a PR, dispatches it to parallel specialist agents (security reviewer, performance reviewer, style reviewer, architecture reviewer), collects their findings, resolves conflicting recommendations, and produces a unified review comment. This is hierarchical + fan-out/fan-in combined.

Use cases: complex multi-level tasks (creating a marketing campaign: strategist → channel agents → copywriters/designers), software projects with natural decomposition, research tasks with specialized sub-questions. Complexity isolation — the top level has no knowledge of lower-level details.

Challenges: communication overhead (information passes up/down, losing detail at each level), cross-branch coordination (communication through a distant common ancestor), manager as bottleneck (the manager must be capable enough to decompose correctly — a weak manager undermines the entire system).

## Peer-to-Peer Collaboration

Equal interaction with no central authority. Agents discover each other, negotiate, and share information on their own. Coordination emerges bottom-up as an emergent property of local interactions.

Advantages: fault resilience (no single point of failure), dynamic adaptation (agents join/leave freely), efficiency for tasks with unpredictable decomposition.

Difficulties: global consistency (each agent sees only its neighbors), preventing duplication/conflicts. Solutions: consensus protocols, reputation systems, market mechanisms.

## Debate Pattern

Adversarial system: agents take opposing positions, challenge each other, and a **judge agent synthesizes** the best elements from each position. The classic structure: proposer, critic, judge — but variations exist with multiple proposers, multiple critics, or rotating roles.

Use cases: tasks requiring high reliability (financial analysis: "bull" vs. "bear"), generation and verification (code author vs. reviewer), factual accuracy on contested or ambiguous topics where a single model might commit to a wrong answer confidently.

**Real-world example:** Perplexity Computer's **Model Council** sends the same query to multiple LLMs simultaneously, compares their answers, and selects the best response (or synthesizes from multiple). This is the Debate Pattern applied at the model level — using diversity of models as the source of independent perspectives.

**Quantified benefit:** Research consistently shows 15-30% improvement in factual accuracy through adversarial review. The mechanism: a single agent commits to an answer and defends it even when wrong (confidence bias). Two opposing agents surface weaknesses that a single agent would overlook. The judge sees both sides and makes a more informed decision.

**Voting/ensemble variant:** Instead of adversarial debate, multiple agents solve the same problem independently, then results are aggregated through majority vote or weighted average. This is more expensive (N full solutions instead of 1 solution + critique) but more reliable for critical decisions. The trade-off: debate is 2-3x cost (proposer + critic + judge); voting is N×cost (N independent solutions). Use debate for tasks with clear right/wrong answers; use voting for subjective or ambiguous tasks.

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

## Handoff / Pipeline Pattern

Sequential task transfer between specialist agents. Each performs its stage and passes the result to the next — an assembly line for information processing. This is the most deterministic multi-agent pattern and the easiest to debug, because the flow is linear and each agent's input/output is clearly defined.

Example: parser (text extraction) → analyzer (entities, facts) → enricher (external information) → synthesizer (final report). Document processing pipelines, ETL workflows, and multi-stage data analysis all follow this pattern.

**Why pipelines are underrated.** In a field excited about emergent agent coordination, the simple pipeline is often the right answer. When you know the processing steps in advance and each step has a well-defined input/output contract, a pipeline gives you: deterministic behavior (same input → same output), easy debugging (inspect the artifact at each stage), independent testing (unit test each agent in isolation), and straightforward error recovery (retry the failed stage).

**The OpenAI Agents SDK handoff pattern** is a lightweight implementation: Agent A transfers control to Agent B along with the conversation context. The triage agent determines intent and hands off to the appropriate specialist. This is a pipeline with dynamic routing at the first stage.

Advantages: clear separation of responsibilities, independent development/testing, easy to add/replace stages, lowest debugging complexity of all multi-agent patterns.

Limitations: sequential nature (failure/delay in one stage → blocks the entire chain), poor support for feedback loops (late-stage results requiring revision of earlier stages), total latency is the sum of all stages. To add feedback loops, combine with the debate pattern or add a "quality gate" agent that can route back to previous stages.

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

Before selecting a pattern, verify that multi-agent is justified — see the decision framework in [[01_MAS_Basics|MAS Basics]]. A single agent with a reasoning model handles most production use cases (~57% of deployed agent systems are single-agent). Multi-agent adds ~3-15x cost and is justified only when you hit clear capability boundaries: genuinely distinct expertise domains, measurable parallelism benefit, or context that exceeds a single agent's window.

---

## Pattern Selection Framework

Choosing a pattern is an architectural decision with direct cost, latency, and reliability implications. This table maps task characteristics to recommended patterns:

| Dimension | Supervisor | Hierarchical | Pipeline | Fan-Out/Fan-In | Debate | Peer-to-Peer |
|-----------|-----------|-------------|----------|----------------|--------|-------------|
| **Complexity** | Simple | Complex, multi-level | Linear, sequential | Parallelizable | Needs validation | Dynamic, unpredictable |
| **Cost (vs single-agent)** | 2-3x | 3-10x | 2-5x | 2-Nx (N = parallelism) | 2-3x | 3-10x |
| **Latency** | Medium | High (deep hierarchy) | High (sum of stages) | Low (max of parallel) | Medium (2-3 rounds) | Variable |
| **Debuggability** | Easy | Medium | Easy | Medium | Easy | Hard |
| **Fault tolerance** | Low (SPOF) | Medium | Low (one stage fails all) | Medium (partial results) | Medium | High (no SPOF) |
| **Best use case** | Clear decomposition | Large projects | ETL, document processing | Research, code review | High-stakes decisions | Autonomous coordination |

**Decision shortcuts:**
- "I know the steps" → Pipeline or Supervisor
- "Steps are independent" → Fan-Out/Fan-In
- "Accuracy matters more than speed" → Debate
- "I need to scale to 100 agents" → Hierarchical or Peer-to-Peer
- "Cost is the primary concern" → Single agent (reconsider multi-agent)

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
