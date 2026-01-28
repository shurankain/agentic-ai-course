# Debugging LLM Applications

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Observability
**Previous:** [[02_Metrics_and_Dashboards|Metrics and Dashboards]]
**Next:** [[04_AgentOps|AgentOps]]

---

## The Unique Nature of Bugs in LLM Systems

Debugging LLMs differs radically from traditional development. In classical programming, a bug is deterministic: identical inputs always produce the same incorrect result. LLMs break this paradigm: identical prompts yield different results, "correctness" is subjective, and the problem may lie in data, prompts, or the model itself rather than in the code.

A chatbot that sometimes gives excellent answers and sometimes nonsense. No stack trace. No specific line of code to fix. The problem could be in prompt wording, retrieval quality, temperature, or model version specifics. Debugging LLMs requires detective-like thinking.

## Problem Taxonomy

Infrastructure: timeouts, rate limiting, network errors, authentication. Deterministic, diagnosed with standard methods.

Integration: incorrect response parsing, context loss between calls, incorrect serialization. Manifests as "sometimes works" due to variability in LLM response formats.

Prompt: insufficiently clear instructions, contradictory requirements, overly long context, suboptimal structure. The model technically works but does not do what is needed.

Data quality in RAG: irrelevant documents, outdated information, contradictory sources, poor chunking. The model answers correctly based on incorrect context.

Model behavior: hallucinations, ignoring instructions, undesirable tone, excessive or insufficient detail. Most complex — may require changing the model or accepting limitations.

## Debugging Toolkit

Prompt Playground — an interactive environment for experimentation. Rapid iteration of wording, result comparison, edge case testing. Fixing the seed for reproducibility is critical.

Trace Viewer visualizes the full request path: from input through retrieval, prompt formation, model call, post-processing, to response. At each step, input data, output, latency, and metadata are visible.

Comparison Tools compare behavior before and after changes: different prompt versions, models, parameters. A/B comparison is critical for understanding the effect of changes.

Dataset Management provides organized storage of test cases with expected results. Rapid execution of changes against regression tests.

## Reproducibility

Non-reproducibility is the main enemy of LLM debugging. Fixing the seed (random seed or temperature=0) makes generation deterministic. If you cannot reproduce the problem, you cannot confirm the fix. However, some providers do not guarantee full determinism even with seed=0.

State snapshots capture the environment at the time of the problem: model version, prompts, retrieval configuration, vector store state. Allows returning to the problematic state.

Logging complete requests and responses is critical. The full prompt is needed, including the system message, few-shot examples, and retrieved context.

Version Control for prompts should be as strict as for code. Every prompt change is a potential behavior change.

## Debugging Response Quality

Precisely articulate what is wrong. "Bad response" is not a diagnosis. Is the response inaccurate? Does it not match the format? Too long? Does it not account for context?

For factual errors, check the source of incorrect information. In RAG, the problem is often retrieval — irrelevant or outdated documents. If the model hallucinates, instructions on verification need to be strengthened.

For format issues, analyze the clarity of format specification in the prompt. Few-shot examples work better than textual descriptions.

For context issues, check the availability of necessary information. Critical information may be truncated due to token limits or lost in the middle of long context (lost in the middle).

## Debugging RAG Systems

Retrieval Debugging: are we finding the right documents? Logging queries to the vector store, visualizing similarity scores, checking top-k for relevance. If needed documents do not appear in top-k, the problem is in embedding, chunking, or data.

Chunking Issues: relevant information is split between chunks or context is lost. Reviewing source documents and chunks, checking overlap, analyzing chunk sizes.

Context Assembly: how retrieved documents are assembled into the prompt. Is there enough context? Too much? Is it properly formatted? Is it lost during truncation?

Answer Grounding verifies that the response is based on context rather than on internal knowledge or hallucinations. Automated verification through NLI models or LLM-as-Judge.

## Debugging Call Chains

Step-by-Step Analysis — reviewing the output of each step. Often the problem becomes obvious when you see what is passed between steps. Incorrect parsing at step 2 renders subsequent steps meaningless.

Decision Point Debugging for agents — why the agent chose a particular tool. Logging the reasoning process helps understand the logic. Non-obvious decisions indicate problems in tool descriptions or insufficient context.

Error Propagation Analysis — how errors propagate through the system. Soft errors accumulate and can lead to failure at the final step.

## Edge Cases and Adversarial Inputs

Edge cases are particularly insidious — models confidently respond even when they should not.

Categories: unusual input formats (emojis, special characters, code injection), extreme lengths (very short or long queries), ambiguous queries (requiring clarification), adversarial prompts (manipulation attempts).

Systematic testing requires a dataset of problematic cases. Every production incident is added for regression testing.

Graceful Degradation — the system recognizes situations where it cannot provide a quality response and explicitly communicates this rather than generating a poor response with a confident tone.

## Debugging Performance

Latency Breakdown decomposes total latency into components: preprocessing, retrieval, prompt assembly, API call (prefill + generation), postprocessing. Reveals bottlenecks.

Token Analysis identifies inefficient usage. Prompts with excessive context increase latency and cost without improving quality.

Caching Strategy checks the effectiveness of caching usage. Repeated queries should be cached; semantic cache can reduce the number of API calls.

Concurrency Issues arise during parallel processing: rate limiting, resource contention, head-of-line blocking.

## Anomaly Detection for LLMs

Anomaly detection requires statistical methods adapted to the specifics of generative models.

**Z-Score Monitoring** — a basic method for sharp deviations. Computed as: z = (x - μ_baseline) / σ_baseline. Alerts at thresholds |z| > 2.5 (99%) or |z| > 3.0 (99.7%). Applied for monitoring latency, tokens, error rates. Effective for sudden changes but misses gradual drift.

**CUSUM (Cumulative Sum)** — for detecting slow trends. Accumulates deviations from the target value: S_n = max(0, S_{n-1} + (x_n - μ - k)), where k is the allowable deviation (typically 0.5σ). Alert when threshold h is exceeded (typically 4-5σ). Valuable for detecting gradual quality degradation.

**EWMA (Exponential Weighted Moving Average)** — smoothed tracking with priority on recent data. Formula: EWMA_t = λ × x_t + (1-λ) × EWMA_{t-1}, where λ (typically 0.2-0.3 for LLMs). Balances between fast response and noise resilience.

**PSI (Population Stability Index)** — detection of changes in input data distribution. Computed as: PSI = Σᵢ (Actual_i - Expected_i) × ln(Actual_i / Expected_i). Interpretation: PSI < 0.1 — no changes, 0.1 ≤ PSI < 0.2 — moderate changes require investigation, PSI ≥ 0.2 — significant changes require action. Critical for RAG systems, where query drift indicates the need to update the knowledge base.

## Debugging in Production

Sampling-based debugging analyzes a random sample with extended logging. Provides a statistical picture without affecting all users.

Feature Flags enable debug mode for individual users. Critical for reproducing specific user problems.

Replay Capability — "replaying" a problematic request in a test environment with full context preservation.

Canary Analysis compares behavior between canary and stable versions. Degradation in canary is detected before full rollout.

## ML Debugging: Key Concepts

ML Debugging is a separate interview round at OpenAI and DeepMind.

Main bug categories: Shape Mismatches (broadcasting hides dimension mismatches), Gradient Issues (detach breaks gradient flow, missing residual connections lead to explosion/vanishing), Data Processing (incorrect shuffle, tokenization mismatch, data leakage in data pipeline), Optimizer (creation before setting requires_grad, weight decay on bias/LayerNorm).

Debugging strategy: Reproduce → Hypothesize → Test → Fix.

Interview checklist: check tensor shapes first, verify gradient flow, Cross-entropy expects logits not probs, create Optimizer after requires_grad, effects of eval vs train mode.

## Practical Examples

The snapshot mechanism captures system state: timestamp, user input, model, parameters, prompts, response, latency, tokens, errors. The replay mechanism re-executes the request with exactly the same parameters to verify fixes.

The RAG problem analyzer checks relevance scores of retrieved documents (if average < 0.7, embedding problems). Grounding check via overlap between words in the response and context (ratio < 0.3 signals hallucinations). Problem categorization system: NO_RELEVANT_DOCS, LOW_RELEVANCE_SCORES, ANSWER_NOT_GROUNDED with descriptions and recommendations.

The hypothesis testing system formulates specific testable hypotheses. Instead of "the model responds poorly" — "The model ignores the system prompt when the user message exceeds 500 tokens." Hypotheses have statuses: PENDING, CONFIRMED (>90% successful tests), REJECTED (<10%). Each hypothesis is linked to test cases for verification.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Observability
**Previous:** [[02_Metrics_and_Dashboards|Metrics and Dashboards]]
**Next:** [[04_AgentOps|AgentOps]]
