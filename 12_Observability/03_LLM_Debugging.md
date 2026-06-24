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

## Hallucination Detection: From Metric to System

"Hallucination rate" as a single metric is insufficient — it tells you that hallucinations exist but not how to detect or prevent them. A systematic approach requires multiple detection layers, each catching a different category of hallucination.

**Consistency checks across runs.** The same query run 3 times should produce consistent factual claims. If the model says "revenue was $50M" in run 1 and "$75M" in run 2, at least one is hallucinated. Implementation: for critical queries, run the model N times (N=3 is the sweet spot between cost and coverage), extract factual claims from each response, and flag inconsistencies. Cost: N× the original query cost. Use selectively for high-stakes outputs only.

**Grounding ratio monitoring.** For RAG systems, measure the overlap between the model's response and the retrieved context. A grounding ratio below 0.3 (less than 30% of response content can be traced to retrieved documents) signals that the model is generating from internal knowledge rather than the provided context — a primary hallucination vector in RAG systems. The ratio is computed as: (tokens in response that match or paraphrase context tokens) / (total tokens in response). NLI (Natural Language Inference) models provide a more sophisticated grounding check than word overlap.

**Semantic contradiction detection.** Over a conversation or across sessions, the model may contradict its own previous statements. "The deadline is Friday" followed later by "The deadline is next Wednesday" is a contradiction that indicates at least one hallucination. Implementation: maintain a running fact store of claims made in the session, and for each new response, check new claims against stored claims using an NLI model. Flag contradictions for review.

**Domain-specific fact validation.** For applications with access to ground truth (databases, APIs, knowledge bases), validate factual claims in real time. "Your order #12345 was shipped on June 10" can be verified against the order database. This is the most reliable hallucination detection method but requires domain-specific integration.

**Production hallucination monitoring pipeline.** Combine the layers: grounding ratio monitoring on every response (cheap, continuous), consistency checks on a 5% random sample (moderate cost, statistical coverage), semantic contradiction detection within sessions (moderate cost, session-scoped), and domain fact validation where ground truth is available (most reliable, domain-specific). Alert when: grounding ratio drops below threshold for more than N consecutive responses, consistency checks show >20% divergence, or any domain fact validation fails.

## RAG Debugging Workflow

When a RAG system gives a wrong answer, the bug can be in any of four stages: retrieval, ranking, context assembly, or generation. A systematic debugging workflow isolates the problem stage.

**Stage 1 — Retrieval: are we finding the right documents?** Inspect the query sent to the vector store. Is it a good query? (Sometimes the LLM reformulates the user's question poorly.) Check the top-K retrieved documents — are the relevant documents present at all? If not, the problem is embedding quality, chunking strategy, or the document is missing from the index entirely. Red flags: top-K similarity scores all below 0.5 (the query does not match anything well), the relevant document is not in the index, the query was reformulated incorrectly.

**Stage 2 — Ranking: are the right documents at the top?** The relevant document may be retrieved but ranked below irrelevant ones. Check the reranker output — did it improve or degrade the ranking? A cross-encoder reranker occasionally demotes relevant results when they contain unusual formatting. Red flags: relevant document present in top-20 but absent from top-5 after reranking, reranker scores are flat (no clear separation between relevant and irrelevant).

**Stage 3 — Context assembly: is the context well-formed?** The right documents may be retrieved and ranked correctly, but the context window assembly may be wrong. Check: is the context truncated (exceeds token budget)? Are chunks from different documents interleaved confusingly? Is critical information split across chunks (the answer spans a chunk boundary)? Red flags: context exceeds 80% of the model's effective window, chunks from unrelated documents are mixed, the answer-containing sentence is split between two chunks.

**Stage 4 — Generation: is the model using the context correctly?** The right context is provided, but the model ignores it or draws from internal knowledge instead. Check the grounding ratio (see hallucination detection above). Try adding explicit instructions: "Answer ONLY based on the provided context. If the context does not contain the answer, say so." Red flags: grounding ratio below 0.3, the answer contains information not present in the context, the model says "based on my knowledge" rather than citing the context.

**The debugging shortcut:** Start from Stage 4 and work backward. If the grounding ratio is high but the answer is wrong, the problem is in retrieval (wrong context). If the grounding ratio is low but good documents exist in the index, the problem is in the generation stage (model ignoring context). This top-down approach finds the root cause faster than bottom-up inspection.

## Practical Examples

The snapshot mechanism captures system state: timestamp, user input, model, parameters, prompts, response, latency, tokens, errors. The replay mechanism re-executes the request with exactly the same parameters to verify fixes.

The RAG problem analyzer checks relevance scores of retrieved documents (if average < 0.7, embedding problems). Grounding check via overlap between words in the response and context (ratio < 0.3 signals hallucinations). Problem categorization system: NO_RELEVANT_DOCS, LOW_RELEVANCE_SCORES, ANSWER_NOT_GROUNDED with descriptions and recommendations.

The hypothesis testing system formulates specific testable hypotheses. Instead of "the model responds poorly" — "The model ignores the system prompt when the user message exceeds 500 tokens." Hypotheses have statuses: PENDING, CONFIRMED (>90% successful tests), REJECTED (<10%). Each hypothesis is linked to test cases for verification.

## Key Takeaways

- **Non-determinism is the core debugging challenge.** Identical inputs produce different outputs, "correctness" is subjective, and the bug may be in data, prompts, or the model itself rather than in code.

- **A structured problem taxonomy accelerates root-cause analysis.** Categorizing issues into infrastructure, integration, prompt, data quality, and model behavior directs investigation to the right layer immediately.

- **Reproducibility requires fixing the seed, capturing full state snapshots, and logging complete prompts.** Without the ability to reproduce a problem, you cannot confirm a fix — version control prompts as strictly as code.

- **Hallucination detection requires multiple layers, not a single metric.** Consistency checks across runs, grounding ratio monitoring, semantic contradiction detection, and domain-specific fact validation each catch a different category.

- **RAG debugging follows a four-stage workflow: retrieval, ranking, context assembly, generation.** Start from Stage 4 (generation) and work backward — checking the grounding ratio first identifies whether the problem is in context quality or model behavior.

- **Statistical anomaly detection methods must be adapted for LLM specifics.** Z-score catches sudden deviations, CUSUM detects slow trends, EWMA tracks recent behavior, and PSI identifies input distribution drift — each serves a different failure mode.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Observability
**Previous:** [[02_Metrics_and_Dashboards|Metrics and Dashboards]]
**Next:** [[04_AgentOps|AgentOps]]
