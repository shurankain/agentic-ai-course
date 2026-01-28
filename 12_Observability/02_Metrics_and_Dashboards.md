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

Cost is calculated as: input tokens × input price + output tokens × output price, all divided by 1000.

For GPT-4o, input tokens cost $0.0025/1K, output — $0.01/1K. The 1:4 ratio makes limiting generation length for brief responses especially important.

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
