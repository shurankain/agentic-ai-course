# Metrics and Dashboards for LLM Systems

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Observability
**Previous:** [[01_Tracing_and_Logging|Tracing and Logging]]
**Next:** [[03_LLM_Debugging|LLM Debugging]]

---

## Philosophy of Metrics in the World of Language Models

Traditional metrics — response time, request count, error rate — remain important but insufficient. LLMs introduce new dimensions: token cost, generation quality, hallucinations, context relevance. A system that responds quickly and does not crash but gives incorrect answers causes more harm than a system with a high error rate. Quality metrics become primary.

## Metrics Hierarchy

At the lowest level — infrastructure metrics: GPU, network, memory. Important for capacity planning but say little about quality.

The next level — LLM operational metrics: token count, time to first token (TTFT), total latency, call cost. High TTFT creates the perception of a "thinking" system. Excessive token consumption increases costs without improving quality.

The third level — quality metrics: answer accuracy, hallucination rate, document relevance, instruction adherence. Measurement requires human evaluation, LLM-as-Judge, or automated checks.

At the top — business metrics: conversion, customer satisfaction, problem resolution time, support cost savings.

## Tokens as the Currency of the LLM Economy

Tokenization algorithms create uneven splitting. English text: ~4 characters or 0.75 words per token. Non-Latin scripts are less efficient: for example, a word in Cyrillic or CJK may take 2-3 tokens, while the English "Hello" takes one. Code is denser: ~3 characters per token.

A support system operating in a non-Latin-script language consumes 30-50% more tokens than an English-language one. RAG with multilingual documents requires different chunking strategies.

Cost is calculated as: input tokens × input price + output tokens × output price (prices are quoted per 1M tokens by most providers).

For GPT-4o, input tokens cost $2.50/1M, output — $10/1M (GPT-5.4: $2.50/$10; budget GPT-5-mini: $0.25/$2.00). The 1:4 ratio makes limiting generation length for brief responses especially important.

**Optimization Strategies:**

**Prompt Caching** — providers cache repeated parts of the prompt, charging a lower fee. With proper architecture, this reduces costs by 2-5x.

**Output Limiting** via max_tokens prevents verbose responses. Protects against cost spikes.

**Model Tiering** — simple requests go to cheaper models, complex ones to powerful models. Intelligent routing reduces average costs by 40-60%.

Token metrics: average count per request, length distribution, input-to-output ratio, cost breakdown by model. Anomalies indicate problems: growing input tokens — context duplication, long responses — looping.

For RAG, the "tokens per relevant document" metric is important for identifying retrieval inefficiencies.

## Latency

Total latency consists of: request processing, retrieval, prompt construction, time to first token, generation time, post-processing.

Time to First Token is a critical metric for interactive applications. It is determined by the prefill phase duration. Long prompts increase TTFT, creating a perception of slowness.

Generation time depends on the number of output tokens and the decode phase speed. For streaming, token flow smoothness is important.

Percentiles are critically important: P50 shows the typical experience, P95 — the "unlucky" users, P99 — the worst case. Optimizing only averages masks serious problems.

## Quality Metrics

For RAG: Retrieval Precision (proportion of relevant documents), Retrieval Recall (proportion of relevant documents found), Answer Faithfulness (alignment with context), Answer Relevance (alignment with the question).

Hallucination Rate — the proportion of responses containing factually incorrect information. Measurement requires comparison with ground truth, consistency checks, or validation through external sources.

Compliance Rate — how well responses adhere to instructions: format, constraints, style.

## Business Metrics and ROI

Cost per Conversation shows the real cost of serving a single user, including API calls, infrastructure, retrieval, and overhead.

Resolution Rate — the proportion of requests handled without escalation. For chatbots, this is tied to staff cost savings. It is important to track the qualitative Resolution Rate — when the user is genuinely satisfied.

User Satisfaction (CSAT, NPS) — direct feedback. Correlation with technical metrics helps understand which improvements matter.

Retention and Engagement indicate the appeal of AI functionality.

## Metrics Collection Architecture

Collection at multiple levels: application (request types, results), LLM integration (tokens, latencies, statuses), infrastructure (resources, queues, errors).

The pull model works well for long-lived services. The push model is better for serverless and dynamic environments.

Aggregation at multiple levels: real-time (last few minutes) for rapid problem detection, hourly/daily for trend analysis, long-term for capacity planning.

Avoid high-cardinality labels (user_id, request_id) — they quickly exhaust monitoring system resources.

## Dashboards

An effective dashboard is a decision-making tool. Each element answers a specific question.

Structure follows the hierarchy: Executive Dashboard (business metrics: cost, quality, satisfaction), Operations Dashboard (operational health: latencies, errors, throughput), Technical Dashboard (details: token distributions, component performance).

Specific visualizations: Token Distribution Histogram (distribution of request and response lengths), Latency Heatmap (latency distribution over time), Cost Attribution Chart (costs by model, feature, segment).

Comparative Views juxtapose metrics across different models, prompts, and A/B variants.

## Alerting

Traditional threshold-based alerts require careful calibration due to the high variability of LLM metrics.

Anomaly detection is a better fit — the system learns "normal" behavior and alerts on deviations. Effective for metrics with complex seasonality.

Alert composition combines signals: "high latency AND low quality" is more informative than two separate alerts.

Runbooks are an integral part. Each alert is accompanied by instructions: which metrics to check, typical causes, actions to take. For LLMs, specific steps include: checking prompt changes, analyzing token distributions, verifying provider status.

## Cost Monitoring

Real-time Cost Tracking shows the current spending level. Comparison with budget enables early detection of overspending. Forecasting provides time to react.

Cost Attribution breaks down expenses by dimensions: model, feature, segment, request type. Helps identify where overspending occurs.

Budget Alerts warn about approaching limits. Multi-level alerts (50%, 75%, 90%) provide time to react. Automatic actions (switching to a cheaper model, rate limiting) protect against catastrophic overspending.

Cost Optimization Insights suggest optimizations: "Prompt X consumes 40% more tokens," "Caching will save $Y per month."

## SLA Monitoring for AI Systems

Defining SLAs for non-deterministic systems is fundamentally harder than for traditional software. A database either returns the correct result or it does not. An LLM returns answers on a quality spectrum — and "good enough" depends on the use case.

**Two SLA dimensions for AI:** Response quality SLA — "95% of responses must score above 4.0 on our 1-5 quality rubric, as measured by weekly LLM-as-Judge evaluation." Latency SLA — "P95 response time must be under 10 seconds for standard queries, under 60 seconds for reasoning queries." These are independent: a fast, wrong answer violates the quality SLA; a slow, correct answer violates the latency SLA. Both must be monitored separately.

**Measuring "good enough" at scale.** You cannot evaluate every response with a human or even with LLM-as-Judge (too expensive). Practical approach: sample-based SLA monitoring. Evaluate 1-5% of responses on a rolling basis. If the sampled quality drops below the SLA threshold for 3 consecutive evaluation windows, trigger an alert. The sampling strategy matters: stratified sampling (proportional representation of different query types) is more informative than random sampling.

**SLA dashboards.** The executive view: a single traffic-light indicator — green (SLA met), yellow (within 10% of threshold), red (SLA violated). Drill-down: quality trend over time, latency percentiles over time, SLA compliance percentage by query type, by model, by feature. Alerts: PagerDuty integration when SLA is breached for more than N consecutive minutes.

## Cost Attribution in Multi-Agent Systems

When a multi-agent system processes a request, the total cost is the sum of all agent invocations, tool calls, and orchestration overhead. Without per-agent cost attribution, you cannot identify which agent is consuming disproportionate resources or optimize the most expensive components.

**Per-agent cost tracking.** Each agent invocation should record: model used, input tokens, output tokens, thinking tokens (for reasoning models), cost computed from the model's pricing table. Aggregate by agent role to answer: "The research agent costs $0.08 per session on average, while the summary agent costs $0.005." When the research agent consumes 16x the budget of the summary agent, you know where to optimize.

**Per-user and per-feature attribution.** Beyond agent-level, track costs by: user (for chargeback or fair-use policies), feature (which product feature drives the most AI spend), query type (simple queries vs complex reasoning). This enables data-driven decisions: "Feature X costs $50K/month in AI spend but generates $10K in revenue — is it worth it?"

**Chargeback models.** In multi-tenant or multi-department deployments, each team should pay for their AI consumption. Options: direct pass-through (each team pays their actual token costs), tiered allocation (teams get a budget, overages are charged), shared infrastructure cost (AI spend is allocated proportionally to usage). The direct pass-through model aligns incentives best — teams that waste tokens pay for it.

## Token Economics by Language

Tokenization algorithms create uneven cost distributions across languages. This is a material concern for multilingual deployments — the same semantic content costs dramatically different amounts depending on the language.

| Language | Token Multiplier vs English | Reason | Cost Implication |
|----------|---------------------------|--------|-----------------|
| English | 1.0x | BPE training corpus majority | Baseline |
| German | 1.2-1.3x | Compound words ("Handschuhschneeballwerfer") | +20-30% |
| Ukrainian | 1.7-1.8x | Cyrillic, less training data | +70-80% |
| Arabic | 1.5-2.0x | RTL, connected script | +50-100% |
| Japanese | 2.0-3.0x | CJK characters, mixed scripts | +100-200% |
| Chinese | 1.8-2.5x | Each character may be a separate token | +80-150% |

**Practical impact.** A customer support system operating in Japanese consumes 2-3x more tokens than the same system in English. At 100K queries/day with an average of 500 tokens per query, the annual cost difference between English and Japanese deployments can exceed $200K for a frontier model. System prompts (which repeat on every request) should be in English where possible — they benefit from prompt caching (up to 90% savings) and lower tokenization overhead. User-facing responses are generated in the user's language.

**Monitoring recommendation.** Add "tokens per semantic unit" as a metric — not just raw token count, but token count normalized by the semantic content length. Track this by language to identify tokenization inefficiency and inform model selection (some models handle CJK/Cyrillic better than others).

## Production Quality Monitoring

Drift detection identifies gradual quality degradation.

Sampling-based Evaluation periodically assesses a random sample of responses. For RAG, Faithfulness and Relevance are checked. For generative systems — instruction adherence and absence of hallucinations.

User feedback integration links technical metrics with user ratings.

Automated Quality Gates prevent deploying changes that degrade quality. Continuous Evaluation on live traffic (shadow testing) enables assessing changes before full rollout.

## Practical Example: Simplified Metrics Collector

The SimpleLLMMetrics class tracks the total number of calls, input and output tokens, and total cost. The record_call method accepts token counts and a model, computes cost based on a pricing table (input and output tokens are priced differently), and accumulates metrics. The get_summary method returns aggregated data: total call count, token count, average input and output tokens, total cost, and average cost per call.

Production systems extend this approach: thread safety, histograms for distributions, integration with Prometheus/Grafana, specialized metrics for RAG and quality scoring.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Observability
**Previous:** [[01_Tracing_and_Logging|Tracing and Logging]]
**Next:** [[03_LLM_Debugging|LLM Debugging]]
