# Long Context Inference

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[05_Inference_Cost_Optimization|Cost Optimization Strategies]]
**Next:** [[07_SGLang_and_Alternatives|SGLang and vLLM Alternatives]]

---

## The Problem

Modern LLMs strive for enormous contexts: Claude 4 200K (1M beta), GPT-4o/GPT-5 128K–1M, Gemini 2.5 Pro up to 1M, Llama 4 Scout up to 10M. The problem: classical attention is quadratic in complexity.

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

LLaMA-70B (80 heads, dim=128, 80 layers):
- 4K tokens: 13.4 GB
- 128K tokens: 419 GB (larger than the model!)

**Optimizations:**

KV-cache compression: quantization INT8/INT4 (2-4× compression), eviction policies (LRU, attention-score based, H2O), KV-cache merging (clustering).

Multi-Query Attention (MQA): all heads share a single K,V — dramatic reduction of 1/H. Some loss of expressiveness.

Grouped-Query Attention (GQA): G groups, each with shared K,V. Trade-off between MHA and MQA. LLaMA-2 and Mistral use GQA.

LLaMA-2-70B with 8 KV heads (instead of 80): 128K cache 419 GB → 42 GB (10× reduction).

## Long Context vs RAG

**Long Context — when to use:**
No pipeline complexity, full context for reasoning, no information loss from chunking, simple deployment. Works when documents fit, the task requires cross-document reasoning, low latency is critical, data is static.

**RAG — when to use:**
Scales to terabytes, cheaper (only relevant chunks), updatable, explainability. Works for massive corpora, frequent updates, cost-sensitivity, source attribution.

**Hybrid:**
Retrieve top-k documents, concatenate into a long context, model reasons over the combined input. RAG scalability + long context quality.

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
