# Project: Production-Ready AI Agent

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Practical Projects
**Previous:** [[04_Fine_Tuning_Pipeline|Fine-Tuning Pipeline]]
**Next:** [[../20_Architecture_Research/01_Mixture_of_Experts|Mixture of Experts]]

---

## Project Overview

A production-ready agent is a complex engineering challenge: infrastructure, scaling, fault tolerance, monitoring, security, continuous improvement. We will build a support service assistant agent.

Knowing the answers is not enough — the agent must operate reliably, handle odd requests correctly, avoid disclosing confidential information, and remain controllable and improvable.

## Framework Choice

Building a production agent from scratch is possible but unnecessary — modern frameworks handle orchestration, state management, and tool execution:

**LangGraph** — recommended for production agents with well-defined workflows. Graph-based state machines provide explicit control, checkpointing for long-running conversations, and LangSmith for observability. Best for: support agents with escalation flows, compliance-critical workflows.

**CrewAI** — role-based multi-agent coordination. Best for: systems where multiple specialized agents collaborate (research + writing + QA). Quick to prototype but less fine-grained control than LangGraph.

**AWS Strands** — model-driven approach with minimal code. Best for: flexible agents where the LLM should decide the workflow. Native MCP support and Bedrock AgentCore for managed deployment.

**Spring AI** — for Java/Spring ecosystems. Advisor pattern provides AOP-like interception for logging, safety, RAG. Best for: enterprise teams already on Spring, when the agent is part of a larger Spring application.

**Framework-agnostic patterns:** Regardless of framework, production agents need the same infrastructure: stateless compute, external state storage, health checks, observability. The sections below apply to any framework choice.

## Architecture and Infrastructure

### Principles

Separation of concerns — distinct components: API layer, orchestrator, tool executors, storage, monitoring. Each scales, updates, and debugs independently.

Stateless computation — the agent does not store state in process memory. All information (sessions, history, intermediate results) resides in external storage. Horizontal scaling, painless restarts.

Graceful degradation — when something goes wrong, the agent degrades gracefully. Knowledge base unavailable — it honestly states it cannot find the information but continues operating. LLM is slow — a timeout fires with a clear message.

### Containerization

Docker ensures reproducibility — identical behavior on a laptop and in the cloud. Kubernetes adds orchestration: autoscaling, self-healing, rolling updates.

Multi-stage builds: dependency assembly → only what is needed for runtime. Do not include secrets — pass them through environment variables or secret management.

Kubernetes: Deployment (how to run pods, how many replicas, how to update), Service (stable entry point), Ingress (external access, TLS, routing), HorizontalPodAutoscaler (load-based autoscaling).

### Configuration

12-factor apps principle: configuration through environment variables, no hardcoded values. Separation by environment (development, staging, production) with identical structure — only values differ.

Secrets (API keys, tokens, passwords) in specialized solutions: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, Kubernetes Secrets with encryption. Regular rotation, access logging.

## API and Request Handling

### API Design

REST for synchronous operations, WebSockets/SSE for streaming. Versioning (/api/v1/chat) — old clients are not broken as the API evolves. Support the previous version for migration.

Every request contains a request_id/correlation_id that permeates logs and metrics. Tracing the path through the system is invaluable when issues arise.

### Rate Limiting

Multi-level: global API limit, limits per organization/API key, limits per IP, limits per tokens (a single request can generate a huge response).

Token bucket — allows short bursts while limiting sustained load. Sliding window counter — more predictable behavior.

When limits are exceeded, return a Retry-After header. Clients implement exponential backoff instead of hammering.

### Queues

Complex tasks (document analysis, multi-step research) take minutes. Pattern "request-queue-callback": the client sends a request and receives a job_id with HTTP 202 Accepted. The request goes into a queue (Redis, RabbitMQ, SQS). A worker executes it and saves the result. The client retrieves it via webhook or polling.

Queues provide resilience to spikes — requests accumulate and are processed as resources become available.

## Reliability

### Error Handling

Classification by recoverability:

Transient errors (network failures, rate limiting, timeouts) — retries with exponential backoff.

Permanent errors (invalid data, missing resources) — a proper message without retries.

Unknown errors (unexpected exceptions) — logging and alerting.

Circuit Breaker protects against cascading failures. If a service responds with errors frequently, the circuit breaker "opens" and subsequent requests are rejected immediately. It periodically attempts to "close" — if the service has recovered, operation resumes.

Timeouts for LLM: aggressive at the request level (30-60 seconds), lenient at the operation level. If the LLM is "thinking too long" — better to abort and retry.

### Idempotency

Idempotency key — a unique identifier from the client. Upon receipt, the system checks whether it was already processed. Yes — the stored result is returned. No — it is processed and saved.

Critical for actions with side effects (creating a Jira task, sending an email).

### Health Checks

Liveness probe — "is the process alive?". The simplest check; do not include external dependencies — if the database is unavailable, a restart will not help.

Readiness probe — "is it ready to accept traffic?". Checks critical dependencies (database, LLM API). Fails — traffic goes to other replicas.

Startup probe — for slow-starting applications (loading a model into memory). Until it passes, other probes are not executed.

## Observability

### Three Pillars

Metrics — aggregated numerical indicators (latency P50/P95/P99, request rate, error rate, token usage). Time-series databases (Prometheus, InfluxDB), dashboards (Grafana).

Logs — records of discrete events. Structured JSON is easy to parse. Context: request_id, user_id, timestamp, severity. Centralization (ELK Stack, Loki).

Traces — the path of a request through the system. Each span is a single operation (LLM call, database search, API request). They reveal where time is lost and which component is failing. OpenTelemetry is the standard.

### AI-Specific Metrics

Response quality — proxy metrics: escalation rate, repeat contacts on the same topic, explicit feedback (thumbs up/down).

Token economics — tokens per request, cost per conversation, token efficiency (output to input ratio). Anomalies indicate prompt injection or inefficient prompts.

Tool usage — which tools are used and how successfully. A high error rate demands attention. Unused tools can be removed.

Latency breakdown — where time is spent (LLM inference, tool execution, retrieval, preprocessing). Optimizing the right components.

### Alerting

Define SLOs (Service Level Objectives) before configuring alerts: 99% of requests <5 seconds, error rate <1%. Alerts trigger when an SLO is at risk.

Multi-level: Warning (requires attention, not immediate), Critical (requires immediate intervention), Emergency (critical business impact).

Every alert contains: what happened, why it is bad, how to diagnose, how to fix. A runbook is linked to the alert.

## Security

### Authentication and Authorization

Authentication: API keys (machine-to-machine), OAuth 2.0/OIDC (user sessions), JWT tokens (authorization claims).

Authorization: RBAC (roles and permissions), ABAC (decisions based on attributes of the user, resource, environment).

For agents, authorization at the tool and data level is important. Different users get different tools and documents.

### Input Validation and Prompt Security

Validation of size, format, characters. But for agents this is not enough.

Prompt injection is the main threat. Defense: system prompt with clear instructions and constraints, input/output filtering of known patterns, monitoring for anomalous behavior.

Sensitive data in the system prompt, inaccessible to the user. Do not include internal instructions in the response. Exercise caution with RAG — documents can also contain injection.

### Audit Logging

GDPR, HIPAA, SOC 2 require data access logging. An audit log is an immutable journal of significant actions: authentication, access to sensitive data, administrative actions, tool usage. Retention per requirements (often 7 years).

Data retention policy — how long data is stored, when it is deleted. Right to erasure (GDPR Article 17) — the ability to delete all user data.

## MCP Integration

### Agent Tools via MCP

Instead of hardcoding tool implementations, expose them as MCP servers:

**Internal tools as MCP servers:** Knowledge base search, ticket creation, user lookup — each becomes an MCP server. The agent connects as an MCP client, discovering available tools at startup. Benefits: tools are reusable across different agents and AI clients, versioned independently, testable in isolation.

**External MCP servers:** Connect to third-party MCP servers for capabilities like file system access, database queries, web search, or Slack integration. The growing MCP ecosystem provides pre-built servers for common integrations.

**Security:** Use OAuth 2.1 for remote MCP servers. Map user permissions to MCP scopes — a support agent gets `tickets:read,write` but not `admin:*`. Audit all MCP tool invocations.

### MCP vs Native Function Calling

For production agents, prefer MCP when: tools are shared across multiple agents or AI clients, tools need independent versioning and deployment, you want to leverage the existing MCP ecosystem. Use native function calling when: tools are tightly coupled to the agent logic, minimal latency is required (MCP adds transport overhead), the tool is a simple computation.

## Cost Management

### Token Economics

LLM costs dominate production agent budgets. Track and optimize:

**Cost per conversation:** Total tokens (input + output) × price per token. Typical support conversation: 2K-10K tokens. At GPT-4o pricing ($2.50/M input, $10/M output): $0.005-0.05 per conversation.

**Cost drivers:** System prompts (repeated every turn — large system prompts multiply cost), conversation history (grows linearly with turns — summarize or truncate), RAG context (retrieved chunks add input tokens — retrieve only what's needed), tool call overhead (each tool use adds a round trip).

### Optimization Strategies

**Prompt caching:** Anthropic and OpenAI offer prompt caching for repeated prefixes (system prompts, few-shot examples). Reduces cost by 50-90% for the cached portion.

**Model routing:** Use a smaller/cheaper model for simple queries (classification, FAQ lookup), route complex queries to a capable model. A router model or keyword heuristics decides. Typical savings: 40-60%.

**Context management:** Summarize conversation history after N turns instead of sending full history. Limit RAG context to top-K most relevant chunks. Remove tool call details from history after processing.

**Batching:** For non-interactive workloads (ticket analysis, bulk classification), use batch APIs (50% cheaper on Anthropic/OpenAI).

### Budget Controls

Set per-user, per-organization, and global token budgets. Alert on anomalies (sudden spike = prompt injection or loop). Track cost per feature to identify optimization targets.

## Continuous Improvement

### A/B Testing

Controlled experiments. Define the success metric before starting: resolution rate, CSAT score, handle time, escalation rate.

Randomization — users are assigned to groups randomly. Sticky sessions — a single user sees one version.

Statistical significance requires a sufficient sample size. Do not declare a winner after 100 requests.

### Feedback Loop

Explicit feedback — "helpful"/"not helpful". Implicit feedback — the user rephrased the question (the answer did not help), escalated, or closed as resolved.

Human-in-the-loop — the agent is uncertain (low confidence) → request to a human. The human's response becomes a fine-tuning example. Difficult cases augment the test suite.

Quality assurance reviews a sample of responses. Experts assess correctness, completeness, tone. Issues become improvement tasks. Good examples become few-shot prompts.

### Canary Deployments

1-5% of traffic goes to the new version. Monitor metrics. All good — increase (10%, 25%, 50%, 100%). Problems — instant rollback.

Feature flags enable functionality for specific users/groups. Beta testing, gradual rollout, quick disabling of a problematic feature.

Blue-green deployment — two identical environments. Blue is active, green is standby. The new version goes to green, is tested, and traffic switches over. Problems — instant switch back to blue.

## Key Takeaways

A production-ready agent is a complex system:

Use established frameworks (LangGraph, CrewAI, Strands, Spring AI) instead of building orchestration from scratch. They handle state management, tool execution, and provide observability integrations.

MCP standardizes tool integration — expose tools as MCP servers for reuse across agents and AI clients. OAuth 2.1 secures remote MCP connections.

Architecture: stateless components, separation of concerns, horizontal scaling. Kubernetes provides resilience and auto-scaling.

API Design: versioning, rate limiting, idempotency. Asynchronous processing for long-running tasks. Streaming for responsive UX.

Reliability: graceful degradation, circuit breakers, retries with backoff. Health checks for auto-recovery.

Cost management is essential — track token economics, implement model routing (cheap model for simple queries), use prompt caching, set budget controls.

Observability: metrics, logs, traces. AI-specific metrics (quality, token economics, tool usage). Alerting based on SLOs.

Security: multi-level authentication/authorization, prompt injection protection, audit logging for compliance.

Continuous Improvement: A/B testing for data-driven decisions, feedback loop for learning, canary deployments for safe rollout.

Production is the beginning of a continuous improvement process. The agent generates data that makes it better. Monitoring detects issues before users do. Experiments prove that changes work.

---

## Navigation
**Previous:** [[04_Fine_Tuning_Pipeline|Fine-Tuning Pipeline]]
**Next:** [[../20_Architecture_Research/01_Mixture_of_Experts|Mixture of Experts]]
