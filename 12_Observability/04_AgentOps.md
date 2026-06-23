# AgentOps: Operations for AI Agents

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Observability
**Previous:** [[03_LLM_Debugging|LLM Debugging]]
**Next:** [[../13_Deployment/01_Deployment_Strategies|Deployment Strategies]]

---

## What is AgentOps

AgentOps is an emerging discipline that combines DevOps and MLOps practices for AI agent systems. It addresses the unique challenges of operationalizing systems where behavior is determined not only by code but also by the model, prompts, tools, and dynamic context.

**Evolution:** 2000s DevOps (Code → Deployment), 2010s MLOps (Models → Production), 2020s LLMOps (LLM APIs → Applications), 2024+ AgentOps (Agents → Autonomous systems).

**Why AgentOps ≠ MLOps:**

Artifact: MLOps — trained model, AgentOps — agent (model + prompt + tools).

Behavior: MLOps — deterministic inference, AgentOps — stochastic reasoning loops.

Debugging: MLOps — Input → Output, AgentOps — Multi-step traces.

Metrics: MLOps — Accuracy/latency, AgentOps — Session success/task completion.

Rollback: MLOps — model version, AgentOps — prompt + tools + guardrails.

## Agent Lifecycle

**Plan — Design:** Defining scope (tasks, tools, guardrails), architectural decisions (Single vs Multi-Agent, model selection, tool design, memory strategy).

**Build — Development:** Prompt engineering and iterations, tool implementation, integration testing, safety testing. Version control: prompts as code (git-versioned), tool configurations, guardrail rules.

**Evaluate — Testing:** Pre-production evaluation (benchmark datasets, red teaming, edge case testing, regression testing). Evaluation dimensions: task success rate, safety compliance, response quality, tool usage accuracy.

**Deploy — Deployment:** Deployment strategies (canary releases, A/B testing, shadow mode, gradual rollout). Infrastructure: scaling configuration, rate limiting, failover setup.

**Monitor — Monitoring:** Real-time observability (session tracking, error detection, anomaly detection, cost monitoring). Alerting: guardrail violations, error rate spikes, latency degradation, cost overruns.

## Key Metrics for AgentOps

**Session-Level:** Session Success Rate (target >90%), Session Duration (target <30s for simple tasks), Turns per Session (optimal 3-5), Abandonment Rate (target <10%).

**Token & Cost:** Tokens per Task (optimize through prompt optimization), Cost per Task (model selection, caching), Token Efficiency (useful/total), Cache Hit Rate (semantic caching).

**Tool Usage:** Tool Call Accuracy (red flag <85%), Tool Failure Rate (red flag >5%), Tool Relevance (red flag <70%), Unused Tools (any).

**Safety & Guardrail:** Guardrail Trigger Rate (target <5%), False Positive Rate (target <1%), Safety Violation Rate (target 0%), PII Leak Rate (target 0%).

**Agent-Specific Quality:** **Loop Detection Rate** — how often agents get stuck in repetitive cycles (same tool, same args, same error; target: <2% of sessions). **Decision Quality** — correctness of intermediate steps, not just the final output (did the agent choose the right tool, formulate the right query, interpret the result correctly?). **Planning Efficiency** — ratio of useful actions to total actions (an agent that reaches the goal in 5 steps is more efficient than one that takes 15 with 10 wasted steps; target varies by task complexity). These three metrics complement Session Success Rate — a session can "succeed" through an inefficient or fragile path that will not generalize.

## Agent Tracing

Agent trace includes: Session metadata (ID, user, timestamp, context), LLM calls (each request with input/output), Tool calls (tools with parameters), Retrieval events (what was retrieved for RAG), Guardrail checks (which checks triggered), Timing data (latency of each step).

**Structured Logging:**

Required fields: session_id, trace_id, span_id, agent_id, timestamp, event_type.

Contextual fields: user_id, task_type, model, tokens_in, tokens_out, latency_ms, success.

## Tools Ecosystem

### Observability Platforms

**LangSmith (LangChain):** Trace visualization, dataset management, prompt hub, LangChain native. Use if you already use LangChain, need prompt versioning, dataset management. Per-trace pricing.

**Langfuse:** Open-source alternative. Acquired by ClickHouse (January 2026), now fully MIT-licensed. Self-hostable, provider agnostic, cost-effective. Use for data residency requirements, budget constraints, customization. Free/Self pricing.

**Phoenix (Arize):** Enterprise observability. Embedding analysis, LLM evaluation, RAG analysis, production scale. Use for enterprise deployment, RAG-heavy systems, embedding monitoring. Freemium pricing.

**Braintrust:** End-to-end evaluation and observability platform. Logging, evals, prompt playground, dataset management. Growing adoption in 2025 for its unified evaluation-observability approach. Supports **merge-blocking quality gates** — PRs are blocked if evaluation quality drops below a threshold, integrating eval directly into CI/CD.

**DeepEval:** Open-source evaluation framework specifically designed for agent evaluation. Key differentiator: **PlanQualityMetric** and **PlanAdherenceMetric** — metrics that evaluate the quality of an agent's plan (is the plan reasonable?) and whether the agent adhered to its plan during execution (did it follow through?). These evaluate the reasoning process, not just the final output. Essential for debugging agents that produce correct results through unreliable paths.

### Framework-Native Observability (2024-2025)

**LangGraph Studio:** LangChain's visual debugging environment for LangGraph agents. Provides: real-time visualization of graph execution (nodes, edges, state transitions), step-through debugging of agent reasoning, state inspection at each node, time-travel debugging (replay from any checkpoint). Particularly valuable for complex multi-step agents where understanding the execution flow is critical.

**CrewAI Observability:** CrewAI added built-in observability features: crew execution traces, agent-level metrics (tasks completed, tools used, tokens consumed), delegation tracking between agents, and integration with external platforms (Langfuse, LangSmith) via callbacks. The multi-agent visibility — seeing how agents delegate, collaborate, and hand off tasks — fills a gap that general-purpose observability tools miss.

**AG2 (formerly AutoGen):** Microsoft's AutoGen was forked and rebranded as AG2 in late 2024 by the original AutoGen team, who continued development independently. AG2 includes: conversation logging across agent groups, structured message tracing, cost tracking per agent, and integration with observability backends. Note the distinction: **AutoGen** (the Microsoft repo) is now in maintenance mode with Microsoft's investment shifting to Microsoft Agent Framework, while **AG2** (the community fork at ag2ai/ag2) continues active development as an independent project. For observability purposes, both share similar tracing patterns inherited from AutoGen 0.4.

### AgentOps.ai Evolution

AgentOps.ai (the dedicated agent observability tool) has evolved alongside the broader ecosystem. It now provides: session replay with full tool call visualization, multi-agent session tracking, LLM cost analytics with provider breakdown, benchmark tracking over time, and webhook integrations for alerting. Its 2-line integration remains its key differentiator for rapid adoption.

### Choosing an Observability Tool

| Need | Recommended Tool |
|------|-----------------|
| LangChain/LangGraph ecosystem | LangSmith + LangGraph Studio |
| Open-source / self-hosted | Langfuse |
| Multi-agent (CrewAI) | CrewAI built-in + Langfuse |
| Multi-agent (AG2/AutoGen) | AG2 built-in logging |
| Enterprise RAG | Phoenix (Arize) |
| Quick agent debugging | AgentOps.ai |
| Unified eval + observability | Braintrust |

## Automation Pipeline

**CI/CD for Agents:**

Prompt Testing: prompt syntax, injection vulnerability scan, regression tests.

Tool Testing: unit tests for each tool, integration tests, error handling tests.

Agent Testing: end-to-end scenarios, benchmark datasets, red team tests.

Deployment: canary deployment, shadow mode verification, gradual rollout.

**Continuous Evaluation:**

Scheduled evaluations: daily regression tests, weekly benchmark runs, monthly red team exercises.

Triggered evaluations: after each prompt change, after tool updates, after model updates.

## Incident Response for Agents

**Typical Incidents:**

Prompt leak (system prompt in responses) → Immediate rollback.

Tool abuse (anomalous tool usage) → Rate limit + investigate.

Safety violation (harmful content) → Block + review.

Cost spike (10x+ cost increase) → Throttle + audit.

Hallucination surge (grounding score decline) → Shadow mode + fix.

**Runbook structure:** Detection (how to detect), Triage (severity assessment), Mitigation (immediate actions), Resolution (long-term fix), Post-mortem (root cause analysis).

## Multi-Agent Observability

Single-agent tracing is straightforward — one trace, one span tree. Multi-agent systems produce trace forests: multiple interleaved traces where causality crosses agent boundaries. Debugging "why did the system give a wrong answer?" requires understanding which agent made the wrong decision and what information it had when it decided.

**Cross-agent communication traces.** Each inter-agent message (handoff, delegation, result return) should be a span that links the sender's trace to the receiver's trace. The parent-child span relationship crosses agent boundaries: Agent A's "delegate_to_B" span is the parent of Agent B's root span. This creates a unified trace tree across agents. LangSmith and Langfuse support this through trace correlation IDs.

**Delegation chain visualization.** In a hierarchical multi-agent system, a task may be delegated through 3-4 levels: Orchestrator → Research Manager → Web Search Agent → URL Fetcher. Each level adds latency and token cost. Visualizing the delegation chain reveals: which agent consumed the most time (bottleneck identification), which agent consumed the most tokens (cost optimization target), where errors originated and how they propagated (root cause analysis). CrewAI's built-in delegation tracking provides this natively; for LangGraph and custom systems, implement explicit delegation logging.

**Agent-level metrics comparison dashboard.** A side-by-side view of all agents in the system: success rate per agent, average latency per agent, average token cost per agent, error rate per agent. This immediately reveals imbalances — if the security review agent takes 30 seconds while the style review agent takes 2 seconds, you know where to optimize. If one agent's error rate spikes, you know which agent to debug first.

## Knowledge Base Freshness Monitoring

RAG systems degrade silently when the knowledge base becomes stale. If the product pricing changed last week but the indexed documents still show old prices, the agent confidently gives wrong answers — and no error is logged.

**Document staleness detection.** Track the last-updated timestamp for every document in the knowledge base. Alert when: any document exceeds its expected refresh interval (e.g., pricing documents should be refreshed weekly), a significant fraction of the corpus (>20%) has not been updated in more than the expected cycle, new documents are ingested at a rate significantly below the expected rate (ingestion pipeline may be broken).

**Retrieval coverage metrics.** Track the percentage of user queries where the retrieval system returns no relevant results (all similarity scores below threshold). A coverage rate below 80% indicates knowledge gaps — topics users ask about that the knowledge base does not address. Log these "no-good-result" queries and review them weekly to identify content gaps.

**Knowledge gap identification.** Cluster the "no-good-result" queries by topic. If 50 users asked about "return policy for electronics" and the knowledge base has no document covering this, that is a content gap that directly impacts agent quality. Automated gap detection: embed the "no-result" queries, cluster them, and generate a weekly report of topic clusters ranked by query volume. This transforms observability data into actionable content improvement recommendations.

## Agent Health Scoring

As agent fleets grow, individual agent monitoring becomes impractical. An aggregate health score — a single number that captures whether an agent is performing well — enables fleet-level management.

**Composite health metric.** Combine four dimensions into a 0-100 score: success rate (weight 40% — the agent completes tasks successfully), efficiency (weight 20% — tasks completed within expected token/time budgets), cost (weight 20% — cost per task relative to baseline), safety (weight 20% — guardrail violations, PII leaks, policy breaches). Each dimension is normalized to 0-100, then weighted. An agent scoring below 60 is flagged for investigation; below 40 triggers automatic failover to a backup.

**Traffic-light dashboard for agent fleets.** Green (score 80-100): healthy, no action needed. Yellow (score 60-79): degrading, investigate within 24 hours. Red (score 0-59): failing, immediate investigation required. For a fleet of 20 agents, a single dashboard with 20 traffic lights provides instant operational awareness. Drill-down into any red/yellow agent shows the contributing factors (which dimension is pulling the score down).

**Health score trends.** A declining health score over time (e.g., 85 → 75 → 65 over 3 weeks) signals gradual degradation — perhaps the model provider's behavior changed, or the knowledge base became stale, or the user query distribution shifted. Trend alerts catch slow degradation that point-in-time metrics miss.

## Practical Examples

**Basic Observability Integration:**

AgentOps.ai requires just 2 lines for initialization. All LLM model calls are automatically intercepted regardless of the library, tool calls are tracked with parameters and results, session metrics are collected (duration, tokens, cost, iterations), errors are captured, and a full trace is created for reproduction.

LangSmith integration uses a callback mechanism. You create a LangChainTracer with a project name and pass it into callbacks when creating the agent executor.

Langfuse uses a decorator approach. You annotate agent functions with the observe decorator, and all calls are automatically traced. Alternatively, you can explicitly create trace objects with metadata.

**Custom Metrics System Architecture:**

The AgentSessionMetrics class includes: session_id, start_time, end_time, success, llm_calls, tool_calls, tokens_in, tokens_out, and a cost_estimate property for cost calculation.

The metrics collector creates a metrics object at session start, increments counters on events, finalizes on completion, and stores history.

The statistics aggregator computes summary metrics: success rate, average duration, average call count, total cost, and guardrail frequency.

The alerting system checks metrics against thresholds. On anomalies (success rate <80%, budget overrun, guardrail trigger rate >10%) it sends notifications through handlers (Slack webhooks, incident management tickets, critical logs).

**Continuous Evaluation Pipeline:**

Test cases define input data, expected results, and success criteria. The evaluator runs the agent on each case and collects results: actual output, invoked tools, iteration count, success flag, errors, and execution time. The summary aggregates results: total count, passed/failed, pass rate (target >90%), average time, and error list. CI/CD integration runs evaluation automatically on changes. If the pass rate drops below the threshold, deployment is blocked and manual review is required.

## Key Takeaways

- **AgentOps is a distinct discipline from MLOps.** The artifact is an agent (model + prompt + tools), behavior is stochastic, debugging requires multi-step traces, and rollback involves prompt + tools + guardrails -- not just a model version.

- **Session-level metrics matter more than request-level metrics.** Session success rate, turns per session, abandonment rate, and loop detection rate capture the user experience that per-call metrics miss.

- **Multi-agent observability requires cross-agent trace linking.** Each inter-agent message should be a span connecting sender and receiver traces, creating a unified trace tree that reveals bottlenecks, cost distribution, and error propagation across agent boundaries.

- **Knowledge base freshness degrades RAG agents silently.** Document staleness detection, retrieval coverage monitoring, and automated knowledge gap identification prevent agents from confidently serving outdated information with no error logged.

- **Agent health scoring enables fleet-level management.** A composite 0-100 score combining success rate (40%), efficiency (20%), cost (20%), and safety (20%) provides a traffic-light dashboard for rapid operational awareness across agent fleets.

- **Incident response for agents requires specialized runbooks.** Prompt leaks, tool abuse, safety violations, cost spikes, and hallucination surges each demand specific detection, triage, and mitigation steps beyond traditional DevOps playbooks.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Observability
**Previous:** [[03_LLM_Debugging|LLM Debugging]]
**Next:** [[../13_Deployment/01_Deployment_Strategies|Deployment Strategies]]
