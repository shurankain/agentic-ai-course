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

## Agent Tracing

Agent trace includes: Session metadata (ID, user, timestamp, context), LLM calls (each request with input/output), Tool calls (tools with parameters), Retrieval events (what was retrieved for RAG), Guardrail checks (which checks triggered), Timing data (latency of each step).

**Structured Logging:**

Required fields: session_id, trace_id, span_id, agent_id, timestamp, event_type.

Contextual fields: user_id, task_type, model, tokens_in, tokens_out, latency_ms, success.

## Tools Ecosystem

**AgentOps.ai:** Specialized tool for agents. 2-line integration, session replay, LLM cost tracking, multi-agent support. Natively built for agents, session-centric view, built-in benchmarking. Freemium pricing.

**LangSmith (LangChain):** Trace visualization, dataset management, prompt hub, LangChain native. Use if you already use LangChain, need prompt versioning, dataset management. Per-trace pricing.

**Langfuse:** Open-source alternative. Self-hostable, open-source, provider agnostic, cost-effective. Use for data residency requirements, budget constraints, customization. Free/Self pricing.

**Phoenix (Arize):** Enterprise observability. Embedding analysis, LLM evaluation, RAG analysis, production scale. Use for enterprise deployment, RAG-heavy systems, embedding monitoring. Freemium pricing.

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

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Observability
**Previous:** [[03_LLM_Debugging|LLM Debugging]]
**Next:** [[../13_Deployment/01_Deployment_Strategies|Deployment Strategies]]
