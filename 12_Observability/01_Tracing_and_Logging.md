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

**Langfuse** — an open-source alternative with self-hosting. Deployed via Docker Compose or Kubernetes Helm chart. Requires PostgreSQL. Production: minimum 3 replicas, persistent storage 100Gi+, resource limits 2 CPU / 4GB RAM. Python SDK uses decorators for automatic tracing. Supports scores for trace evaluation and batch evaluation.

**Arize Phoenix** — open-source with a focus on embeddings and drift detection. Embedding Visualization via UMAP shows query clusters. Drift Detection identifies changes in embeddings: PSI > 0.2 indicates significant drift. LLM Evaluation via evaluators: HallucinationEvaluator, RelevanceEvaluator, ToxicityEvaluator. RAG Retrieval Analysis via metrics: MRR@k, NDCG@k, Precision@k.

## Platform Comparison

**LangSmith**: Cloud only, LangChain integration, prompt hub, advanced datasets, built-in evaluation. Choose if you use LangChain, need full lifecycle management, and are willing to pay for a managed service.

**Langfuse**: Cloud + Self-host, SDK integration, scores, multi-framework. Choose for self-hosting, limited budget, customization.

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

## Storage and Retention

LLM observability data is voluminous. Tiered storage: hot data in fast storage, cold data in archival storage. Sampling: 1% random + 100% errors + 100% slow requests. Retention policies: 7 days with full prompts, 90 days of aggregated metrics. Consider compliance requirements.

## Practical Tracing Implementation

The data model includes a Trace class with traceId, userId, timestamps, and a list of spans. The Span class contains spanId, parentSpanId, operation type, timestamps, status. The LLMSpan class adds model, inputTokens, outputTokens.

The instrumentation service provides methods: startTrace creates a trace and stores context in ThreadLocal, adds trace_id to MDC. startLLMSpan creates a span with spanId and model. endLLMSpan records the end time and token count. endTrace finalizes the trace, saves it to the repository, and clears ThreadLocal.

Structured logging records successful calls and errors with trace_id for correlation.

Key patterns: ThreadLocal context for automatic propagation, MDC integration for automatic trace_id addition, structured logging for JSON logs.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Observability
**Previous:** [[../11_Evaluation_Testing/05_RAG_Evaluation|RAG Evaluation]]
**Next:** [[02_Metrics_and_Dashboards|Metrics and Dashboards]]
