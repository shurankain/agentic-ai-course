# Tracing and Logging for LLM Applications

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Observability
**Previous:** [[../11_Evaluation_Testing/05_RAG_Evaluation|RAG Evaluation]]
**Next:** [[02_Metrics_and_Dashboards|Metrics and Dashboards]]

---

## Unique Observability Challenges of LLM Systems

Observability of LLM applications differs from traditional systems. Non-determinism means that identical inputs produce different outputs. Understanding a problem requires knowing exactly what happened in a specific request: prompt, context, parameters, response.

Multi-step nature of agents creates complex operation chains. An agent may make ten LLM calls, invoke five tools, and query RAG. Without tracing, localizing the problem is virtually impossible. External dependencies add unreliability: latency, errors, rate limiting. High cost requires tracking every token.

## Three Pillars of Observability

Logs — discrete events with timestamps and context. They provide detailed information about specific events but are difficult to aggregate.

Metrics — aggregated numerical data. They allow seeing trends and anomalies at scale but lack details of specific incidents.

Traces — linked sequences of operations. They reveal the structure of complex operations and time distribution across components.

## Anatomy of an LLM Application Trace

Unlike traditional distributed systems, LLM traces have specific nesting patterns. The root span covers the entire user request, containing a preprocessing span, then the main agent loop span with nested operations: LLM calls for reasoning, tool invocations, final response synthesis. It concludes with a postprocessing span.

**Span Design Principles:**

**Semantic Boundaries** — each span represents a semantically cohesive operation.

**Measurability** — each span has measurable attributes: duration, token count, result size.

**Debuggability** — spans provide sufficient information for debugging, balancing detail and volume.

**Cardinality Control** — avoid high-cardinality attributes in labels. Use them as metadata instead.

**Trace Components:**

**Trace** contains a unique trace_id, timestamps, context metadata, and overall status.

**Span** — an individual operation with a span_id and a reference to parent_span_id. Spans form a tree structure.

**LLM Span** — a specialized span for model invocation. Contains: model name and provider, full prompt, response, token count, generation parameters, latency.

**Tool Span** captures tool name, parameters, result, execution time, status.

**Retrieval Span** stores the search query, retrieved documents with IDs, similarity scores, execution time.

**Agent Step Span** — a single agent step in the ReAct loop with think and act phases.

## What to Include in a Trace

Required information: identifiers, timestamps, operation type, status, basic metrics.

Model information: provider, name, version, parameters.

Request context: user_id, session_id, environment, service version.

Inputs and outputs — with caution due to data volume, privacy, and security. Trade-offs: logging only for errors, sampling, PII masking, separate secure storage.

## Structured Logging

Structured logs in JSON format are easy to parse and aggregate. Each log entry contains a trace_id for linking to traces, timestamp, event type, and relevant metadata.

Event types: llm_call_started, llm_call_completed, llm_call_failed, tool_executed, agent_step, retrieval_completed.

Levels: ERROR for failures, WARN for abnormal situations, INFO for normal events, DEBUG for detailed information.

MDC allows automatically adding trace_id and user_id to all logs within a request scope.

## OpenTelemetry for LLM

OpenTelemetry is the de facto standard for distributed tracing. It requires adaptation for LLM use cases.

Semantic conventions for LLM: gen_ai.system (provider), gen_ai.request.model, gen_ai.usage.input_tokens, gen_ai.usage.output_tokens.

Propagation passes trace context between services.

Exporters send data to backends: Jaeger, Zipkin, Datadog, New Relic.

## Specialized LLM Observability Platforms

Platforms specialized in LLM observability have emerged: LangSmith, LangFuse, Helicone, Arize Phoenix.

Advantages: understanding of LLM specifics, built-in visualizations, integration with evaluation, playground for experiments.

**LangSmith** — a platform by LangChain with deep integration. Strengths: chain visualization, dataset management, evaluation. Prompt Hub versions prompts as artifacts. Prompt Playground enables A/B testing and cross-model comparison. Datasets — structured collections for evaluation with automatic addition from production traces. User feedback is linked to specific runs for building golden datasets.

**Langfuse** — an open-source alternative with self-hosting. Acquired by ClickHouse in January 2026 and relicensed to fully MIT — ensuring long-term open-source viability with ClickHouse's analytics backend. Deployed via Docker Compose or Kubernetes Helm chart. Requires PostgreSQL. Production: minimum 3 replicas, persistent storage 100Gi+, resource limits 2 CPU / 4GB RAM. Python SDK uses decorators for automatic tracing. Supports scores for trace evaluation and batch evaluation.

**Arize Phoenix** — open-source with a focus on embeddings and drift detection. Embedding Visualization via UMAP shows query clusters. Drift Detection identifies changes in embeddings: PSI > 0.2 indicates significant drift. LLM Evaluation via evaluators: HallucinationEvaluator, RelevanceEvaluator, ToxicityEvaluator. RAG Retrieval Analysis via metrics: MRR@k, NDCG@k, Precision@k.

## Platform Comparison

**LangSmith**: Cloud + self-hosted (Enterprise plan supports deployment on Kubernetes in AWS/GCP/Azure since 2025), LangChain integration, prompt hub, advanced datasets, built-in evaluation. Choose if you use LangChain, need full lifecycle management.

**Langfuse**: Cloud + Self-host (fully MIT-licensed after ClickHouse acquisition, Jan 2026), SDK integration, scores, multi-framework. Choose for self-hosting, limited budget, customization.

**Phoenix**: Self-host OSS, drift detection, embeddings visualization, RAG analysis. Choose for ML observability, research, full control.

**Helicone**: Cloud only, proxy-based, minimal integration. Choose for a quick start without code changes.

## Production Checklist

**Minimum set:**
- Tracing of all LLM calls with complete information
- Structured logging with trace_id correlation
- Error tracking with alerting
- Cost monitoring with budget alerts
- Retention policy defined

**Advanced set:**
- User feedback collection (feedback collection pipeline)
- Evaluation on golden datasets (weekly)
- Drift detection on embeddings
- A/B testing prompts with metrics
- PII masking in logs
- Compliance audit trail

## PII Masking in Traces

LLM traces contain sensitive data by nature — user inputs often include names, emails, phone numbers, addresses, medical information, and financial data. Storing this unmasked creates legal liability under GDPR (EU), CCPA (California), HIPAA (healthcare), and other privacy frameworks. PII masking is not optional for production systems.

**What to mask:** Personally identifiable information varies by jurisdiction but generally includes: email addresses, phone numbers, government IDs (SSN, passport), credit card numbers, physical addresses, full names (when combined with other identifiers), medical records, financial account numbers. The challenge: aggressive masking can destroy debugging value, while insufficient masking creates compliance risk.

**Masking approaches, from simple to sophisticated:**

**Regex-based masking** — pattern matching for structured PII (emails, phone numbers, credit cards, SSNs). Fast and deterministic. Handles well-formatted data but misses unstructured mentions ("my social is nine four three..."). Typical implementation: a pre-processing pipeline that runs regex patterns before trace storage. Effectiveness: catches 60-70% of PII in typical LLM traces.

**NER-based masking** — Named Entity Recognition models identify PERSON, LOCATION, ORGANIZATION entities in free text. Microsoft Presidio is the standard open-source library for this. Higher recall than regex (catches names, addresses in narrative text) but slower and can produce false positives (masking product names that resemble person names). Effectiveness: catches 85-90% of PII.

**Hybrid approach** — regex for structured patterns (fast, cheap, high precision) + NER for unstructured text (higher recall). Run regex first, NER second, union the results. This is the production standard for regulated industries. For LLM traces specifically, apply masking to: user inputs, model outputs, tool call arguments, and tool results. Leave trace metadata (trace_id, timestamps, model name, token counts) unmasked — these are needed for debugging and contain no PII.

**GDPR/CCPA implications for trace storage:** Under GDPR, LLM traces containing personal data are "processing" that requires a legal basis. Article 17 (right to erasure) means you must be able to delete all traces associated with a specific user. This requires indexing traces by user_id — which many observability platforms do not support natively. Retention policies must account for both debugging value (short) and compliance requirements (varies by regulation). Standard approach: 7 days full traces with masked PII, 90 days aggregated metrics without PII, indefinite for compliance audit trails (anonymized).

## Multi-Tenant Tracing

When a single observability platform serves multiple customers or business units, tenant isolation in traces becomes critical. Leaking one tenant's trace data to another is a security incident.

**Namespace isolation:** Each tenant's traces are stored in a separate namespace (project in LangSmith, organization in Langfuse). Queries are scoped to the authenticated tenant's namespace — cross-tenant queries are architecturally impossible, not just access-controlled. This is the simplest and strongest isolation model.

**Per-tenant retention policies:** Different tenants may have different compliance requirements. A healthcare tenant needs HIPAA-compliant retention (6 years), while a marketing tenant may prefer 30-day retention to minimize data liability. The observability platform must support per-namespace retention configuration.

**Tenant-aware cost attribution:** In multi-tenant deployments, each tenant should see only their own token consumption, cost, and performance metrics. This enables chargeback (billing tenants for their AI usage) and prevents cross-tenant information leakage through side channels (e.g., inferring a competitor's usage volume from aggregate metrics).

## Compliance Audit Trails

For regulated industries, observability is not just debugging — it is a legal requirement. Auditors will ask: what data did the AI system process? What decisions did it make? Who approved those decisions? Can you prove the system behaved correctly at a specific point in time?

**What auditors actually ask for:** In healthcare (HIPAA): who accessed patient data, when, for what purpose, and what the AI recommended. In finance (SOX): complete trace of every AI-influenced financial decision, with timestamps and approval chains. In any EU deployment (AI Act): evidence of human oversight for high-risk AI decisions, conformity assessment documentation.

**Immutable audit logs:** Audit trails must be tamper-proof — append-only storage where entries cannot be modified or deleted (except under a court-ordered data deletion). Options: append-only databases (Amazon QLDB), cryptographic chaining (each log entry includes a hash of the previous entry), write-once storage (S3 Object Lock, Azure Immutable Blob Storage). The cost of immutable storage is higher, but for regulated deployments, it is non-negotiable.

**Practical architecture:** LLM observability traces → PII masking pipeline → two output streams: (1) debugging traces with masked PII to the observability platform (Langfuse, LangSmith) with short retention, and (2) compliance audit records to immutable storage with long retention and full metadata (but PII replaced with pseudonymous identifiers).

## Storage and Retention

LLM observability data is voluminous. Tiered storage: hot data in fast storage, cold data in archival storage. Sampling: 1% random + 100% errors + 100% slow requests. Retention policies: 7 days with full prompts (masked PII), 90 days of aggregated metrics, indefinite for compliance audit trails (anonymized). Consider compliance requirements — HIPAA requires 6 years, SOX requires 7 years, GDPR requires deletion upon request but retention for legitimate purposes.

## Practical Tracing Implementation

The data model includes a Trace class with traceId, userId, timestamps, and a list of spans. The Span class contains spanId, parentSpanId, operation type, timestamps, status. The LLMSpan class adds model, inputTokens, outputTokens.

The instrumentation service provides methods: startTrace creates a trace and stores context in ThreadLocal, adds trace_id to MDC. startLLMSpan creates a span with spanId and model. endLLMSpan records the end time and token count. endTrace finalizes the trace, saves it to the repository, and clears ThreadLocal.

Structured logging records successful calls and errors with trace_id for correlation.

Key patterns: ThreadLocal context for automatic propagation, MDC integration for automatic trace_id addition, structured logging for JSON logs.

## Key Takeaways

- **LLM observability rests on three pillars — logs, metrics, and traces — but LLM-specific span types are essential.** Standard distributed tracing must be extended with LLM spans (tokens, prompts, parameters), tool spans, retrieval spans, and agent step spans.

- **PII masking is not optional for production traces.** A hybrid approach (regex for structured patterns + NER for free text) catches 85-90% of PII; the standard pipeline applies masking before storage to both user inputs and model outputs.

- **Multi-tenant tracing requires namespace isolation, not just access control.** Cross-tenant queries must be architecturally impossible, with per-tenant retention policies and cost attribution to prevent information leakage.

- **Compliance audit trails serve legal requirements, not just debugging.** Immutable, append-only logs with tamper-proof storage (QLDB, S3 Object Lock) are non-negotiable for regulated industries under HIPAA, SOX, or the EU AI Act.

- **Platform selection depends on ecosystem and constraints.** LangSmith for LangChain-native workflows, Langfuse (fully MIT after ClickHouse acquisition) for self-hosting and budget constraints, Arize Phoenix for embedding drift and RAG analysis.

- **Tiered retention balances debugging value and compliance cost.** 7 days of full traces with masked PII, 90 days of aggregated metrics, and indefinite anonymized audit trails is the standard production pattern.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Observability
**Previous:** [[../11_Evaluation_Testing/05_RAG_Evaluation|RAG Evaluation]]
**Next:** [[02_Metrics_and_Dashboards|Metrics and Dashboards]]
