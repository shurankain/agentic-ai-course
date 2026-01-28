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

Adversarial system: agents take opposing positions, challenge each other, and an arbiter renders a decision. Idea generator vs. critic vs. arbiter.

Use cases: tasks requiring high reliability (financial analysis: "bull" vs. "bear"), generation and verification (code author vs. reviewer).

Key requirement: true independence of opponents (different models, prompts, data sources). Identical models produce identical blind spots.

## Swarm Intelligence

Inspired by social insects: a colony exhibits complex behavior from simple individual rules. No central control — global behavior emerges from local interactions.

**Stigmergy** — communication through environmental modification. Agents leave "marks" (traces in shared storage), others detect them and adjust behavior. Shorter paths are reinforced faster. Types: quantitative (mark intensity), qualitative (mark types → specialization), temporal (decay → recency priority).

**Emergence** — system properties absent in individual elements. The system finds optimal solutions without explicit optimization and adapts without reprogramming.

**Self-organization** requires: positive feedback (amplifying successes), negative feedback (limiting growth), randomness (new patterns), sufficient interactions.

Use cases: optimization and search in vast, poorly structured spaces. A thousand agents sharing discoveries outperform a single one.

Drawback: unpredictability, difficulty debugging emergent bugs.

**OpenAI Swarm** — an experimental framework: lightweight agents, handoffs instead of orchestration, stateless execution. Suited for prototyping with clear domains, not for complex state management.

## Mixture of Experts

A set of narrowly specialized experts on a single level. A router analyzes the task and directs it to the appropriate expert (the router is not a manager — it only determines relevance).

Advantages: deep specialization in a niche (SQL agent vs. generalist), system versatility (new task types → new experts without rework).

Critical point: routing quality. Incorrect routing → poor results. Alternative: multiple experts + response aggregation (more expensive but more reliable).

## Handoff Pattern

Sequential task transfer between specialist agents. Each performs its stage and passes the result to the next (assembly line).

Example: parser (text extraction) → analyzer (entities, facts) → enricher (external information) → synthesizer (final report).

Advantages: clear separation of responsibilities, independent development/testing, easy to add/replace stages.

Limitations: sequential nature (failure/delay in one stage → blocks the entire chain), poor support for feedback loops (late-stage results requiring revision of earlier stages).

## Key Takeaways

**Supervisor** — simplicity, single point of failure. For tasks with clear decomposition.
**Hierarchical** — scalability, complexity isolation. For multi-level tasks.
**Peer-to-peer** — resilience, complex coordination. For dynamic tasks.
**Debate** — adversarial approach → quality. For decision reliability.
**Swarm** — collective intelligence. For large search spaces.
**MoE** — specialization + routing. For heterogeneous queries.
**Handoff** — pipeline. For pipeline workflows.

The best systems combine patterns.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Multi-Agent Systems
**Previous:** [[01_MAS_Basics|Multi-Agent Systems Basics]]
**Next:** [[03_Agent_Orchestration|Agent Orchestration]]
