# Long Context Inference

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[05_Inference_Cost_Optimization|Cost Optimization Strategies]]
**Next:** [[07_SGLang_and_Alternatives|SGLang and vLLM Alternatives]]

---

## The Problem

Modern LLMs strive for enormous contexts: Claude 4.6/4.7 200K (1M on Opus 4.6/4.7), GPT-5.4 128K–1M, Gemini 3 Flash/3.1 Pro up to 1M, Llama 4 Scout up to 10M. The problem: classical attention is quadratic in complexity.

For a sequence of length N:
- Time: O(N²) operations
- Memory for attention matrix: O(N²), for KV-cache: O(N)
- Bandwidth: O(N²) data

At N = 128K the attention matrix contains 16 billion elements. For a 32-head, 32-layer model — terabytes of memory.

## RoPE and Extrapolation

**Rotary Position Embeddings:**
The dominant method in modern LLMs (LLaMA, Mistral). Rotates query and key vectors by an angle dependent on position. Relative positional encoding with natural integration into attention.

**The Problem:**
A model trained on 4K tokens, during inference on 32K, encounters positions it has never seen. Result: quality degradation, "forgetting" the beginning of the context, hallucinations.

## RoPE Scaling Techniques

**Position Interpolation (PI):**
Instead of extrapolation — interpolation within the trained range. To extend 4K→32K, all positions are "compressed" into the [0, 4K] range. Simple to implement, works without fine-tuning, but "compression" reduces position resolution.

**NTK-aware Scaling:**
Not all RoPE frequencies are equally important. High frequencies capture local patterns, low frequencies capture global ones. The NTK approach modifies base frequency, effectively "stretching" low frequencies more. Variants: NTK-by-parts (different scaling per frequency), Dynamic NTK (depends on length).

**YaRN:**
Combines PI and NTK with additional improvements: NTK-aware interpolation, attention scaling (softmax temperature correction), fine-tuning friendly. Enables extending 4K→128K with minimal fine-tuning (~400 steps).

**Practice:**
HuggingFace parameter rope_scaling with type ("linear", "yarn", "dynamic") and coefficient. vLLM supports Dynamic NTK automatically without explicit configuration.

## Architectural Solutions

**Sliding Window Attention:**
Each token "sees" only a fixed window W of the most recent tokens. O(N × W) instead of O(N²). Fixed KV-cache. Limitation: tokens beyond the window are "forgotten". Mitigation: information propagates through layers — with L layers and window W, effective reach = L × W. Mistral: W=4096, 32 layers, reach ~128K.

**Landmark Attention:**
Select "landmark" tokens that summarize segments. Each token attends to landmarks + local window. Complexity O(N × (W + N/S)) — linear.

**StreamingLLM:**
Observation: attention weights on the first tokens ("attention sinks") regardless of semantic content. Always retain the first k tokens + sliding window of the last W. Discard the middle. KV-cache = [sink_tokens][recent_window]. Infinite streaming without memory growth. Limitation: no access to middle tokens. Suitable for streaming/chat, not document analysis.

## Memory Requirements

**KV-cache sizing:**
Formula: 2 × H × D × L × N × sizeof(dtype)

LLaMA-70B (64 KV heads / MHA, dim=128, 80 layers), per request:
- 4K tokens: 13.4 GB
- 128K tokens: 419 GB (larger than the model!)

Note: these are per-request figures with full MHA (64 KV heads). With GQA (8 KV heads, as in LLaMA-2-70B), sizes drop ~10x. For batched serving, multiply by concurrent requests — see [[../15_GPU_Architecture/05_Quantization_Deep_Dive|KV-Cache Quantization]] for production batched calculations.

**Optimizations:**

KV-cache compression: quantization INT8/INT4 (2-4× compression), eviction policies (LRU, attention-score based, H2O), KV-cache merging (clustering).

Multi-Query Attention (MQA): all heads share a single K,V — dramatic reduction of 1/H. Some loss of expressiveness.

Grouped-Query Attention (GQA): G groups, each with shared K,V. Trade-off between MHA and MQA. LLaMA-2 and Mistral use GQA.

LLaMA-2-70B with 8 KV heads (instead of 80): 128K cache 419 GB → 42 GB (10× reduction).

## The Cost of Long Context

Long context is not free. Understanding the cost structure helps decide when it is justified and when RAG is cheaper.

**Attention cost scales quadratically.** Doubling the context from 32K to 64K tokens quadruples the attention computation. This affects prefill latency directly — 32K: ~800ms, 64K: ~2s, 128K: ~4s on A100. For interactive applications, 4-second prefill latency is noticeable.

**KV-cache cost scales linearly but is large.** A 70B model with GQA at 128K context uses ~42 GB of KV-cache per request. At 10 concurrent requests: 420 GB just for KV-cache — more than the model weights. This limits how many concurrent long-context requests a server can handle, directly impacting throughput and cost per request.

**Token cost at different context lengths.** Most providers charge per input token regardless of context length, but some apply multipliers for very long contexts. Even without multipliers, the raw token cost is significant: a 100K-token document at $2.50/MTok input costs $0.25 just to read — before generating any output. Processing this document 100 times (during development and testing) costs $25 for input alone. RAG retrieves only 2-5K relevant tokens per query at a fraction of the cost.

**When 1M tokens is justified.** Genuinely useful for: analyzing an entire codebase (understanding architecture across hundreds of files), legal document review (cross-referencing clauses across a contract suite), book-length content analysis (summarizing, extracting themes), and multi-document synthesis where the relationships between documents matter. Not justified for: simple Q&A where the answer is in one paragraph (RAG is 100x cheaper), keyword-based search tasks, or scenarios where only a small fraction of the context is relevant to any given query.

## The Lost-in-the-Middle Problem

A well-documented phenomenon: information placed in the middle of a long context is retrieved less accurately than information at the beginning or end. The model "forgets" the middle, even though it is technically within the context window.

**What happens.** In experiments with 20+ documents, models successfully find information in the first 5 and last 5 positions but show significant accuracy degradation for documents in positions 6-15. This is not a hard cutoff but a gradient — accuracy drops by 10-25% for middle positions compared to beginning/end positions. The effect is stronger with longer contexts and weaker with shorter ones. Flash Attention and architectural improvements have reduced but not eliminated this effect.

**Mitigation strategies.** Strategic information placement: put the most important information at the beginning and end of the context, not the middle. For RAG systems, this means placing the highest-relevance retrieved documents first and last in the context, with lower-relevance documents in the middle. Recursive summarization: for very long documents, summarize sections independently, then summarize the summaries, creating a hierarchical compression that preserves key information at every level. Context compression: use a smaller model to extract the essential information from a long document, then feed the compressed version to the main model. This trades one cheap LLM call for a significantly shorter context on the expensive model. Chunked processing: process the long document in segments, extract relevant information from each segment independently, then combine the extracted information for final reasoning. This eliminates the lost-in-the-middle effect entirely but loses cross-segment relationships.

**Practical recommendation.** For production systems, do not assume that putting more context is always better. Test with your specific model and workload: insert a known fact at different positions in the context and measure retrieval accuracy. If accuracy drops below 80% for middle positions, apply mitigation strategies or switch to RAG for that use case.

## Long Context vs RAG

**Long Context — when to use:**
No pipeline complexity, full context for reasoning, no information loss from chunking, simple deployment. Works when documents fit, the task requires cross-document reasoning, low latency is critical, data is static.

**RAG — when to use:**
Scales to terabytes, cheaper (only relevant chunks), updatable, explainability. Works for massive corpora, frequent updates, cost-sensitivity, source attribution (see [[../06_RAG/01_RAG_Basics|RAG Basics]] for the full RAG pipeline).

**Cost comparison for a concrete scenario.** Processing a 200-page technical manual (approximately 80K tokens) to answer user questions. Long context approach: $0.20 per query (80K input tokens at $2.50/MTok). RAG approach: $0.005 per query (2K retrieved tokens). At 1,000 queries/day: long context costs $200/day, RAG costs $5/day — a 40x difference. The trade-off: RAG requires an indexing pipeline (one-time setup cost) and may miss cross-document relationships. Long context provides better reasoning quality but at dramatically higher per-query cost.

**Hybrid:**
Retrieve top-k documents, concatenate into a long context, model reasons over the combined input. RAG scalability + long context quality. This is the emerging best practice: use RAG to narrow 1M tokens to 10-30K relevant tokens, then use long context for reasoning over the retrieved set.

Hierarchical: summarize documents, retrieve summaries, expand as needed.

## Practical Recommendations

**Choosing context length:**
- Chat/Assistant: 4-8K
- Code completion: 8-16K
- Document QA: 16-32K
- Multi-doc analysis: 64-128K
- Book/Report analysis: 128K+

**Hardware:**
- 4K: 16 GB (RTX 4090)
- 16K: 24-40 GB (A100-40GB)
- 64K: 80+ GB (A100-80GB, H100)
- 128K+: 160+ GB (Multi-GPU tensor parallelism)

**Latency:**
Prefill scales linearly: 4K ~100ms on A100, 32K ~800ms, 128K ~3-4s. Decode is relatively constant (memory-bound).

## Key Takeaways

Quadratic attention complexity is a fundamental limitation. Long context requires architectural solutions: sliding window, landmarks, sparse attention.

RoPE extrapolation is non-trivial. PI, NTK-aware, and YaRN enable extension without full retraining.

KV-cache is the dominant memory consumer. GQA, quantization, and eviction are critical.

StreamingLLM enables "infinite" context for streaming, but with loss of middle tokens.

Long context vs RAG is not a binary choice. Hybrid approaches combine scalability and reasoning quality.

Hardware requirements grow rapidly. 128K requires enterprise GPUs or multi-GPU setups.

Prefill latency scales linearly — this determines practical limits for real-time applications.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[05_Inference_Cost_Optimization|Cost Optimization Strategies]]
**Next:** [[07_SGLang_and_Alternatives|SGLang and vLLM Alternatives]]
