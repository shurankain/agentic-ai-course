# Production Inference Architecture

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[../16_Distributed_Training/05_Distributed_Training_Practice|Practice: Distributed Training]]
**Next:** [[02_vLLM_internals|vLLM internals: PagedAttention and KV-cache]]

---

## LLM Inference Characteristics

LLMs generate text token by token — each new token requires a full forward pass through the model. A 500-token response requires 500 sequential forward passes.

Key challenges:
- **Sequential dependency:** tokens cannot be generated in parallel
- **Variable output length:** response length is unknown in advance
- **KV-cache explosion:** intermediate states grow with each token
- **Bottleneck — memory bandwidth:** memory reads dominate over computations

## Prefill and Decode Phases

**Prefill (prompt processing):**
All input tokens are processed in parallel. A compute-intensive phase with a large number of matrix multiplications. Generates the initial KV-cache.

**Decode (token generation):**
Each token is processed sequentially. A memory-intensive phase — attention requires reading the entire KV-cache. Low arithmetic intensity makes the GPU memory-bound.

Decode is critically slow: for a single token at context length N, the entire KV-cache of size N × hidden_size × 2 elements must be read, performing only 2 × N × hidden_size operations. At such low arithmetic intensity, a GPU with 3 TB/s memory bandwidth and 1000 TFLOPS utilizes only ~1% of its potential.

## KV-Cache

**Why it is needed:**
Without caching, each new token would require recomputing attention for all previous tokens — quadratic complexity. KV-cache stores the Key and Value vectors of already processed tokens for reuse.

**Size:**
Formula: 2 × num_layers × num_heads × head_dim × seq_len × batch_size × bytes_per_element

For LLaMA-70B with MHA (80 layers, 64 KV heads, head_dim=128, FP16) — worst case without GQA:
- 1 token = 2.62 MB
- 4K context = 10.7 GB
- 128K context = 343 GB (larger than the model itself ~140 GB!)

*Note:* In practice, LLaMA-2-70B and later models use GQA (8 KV heads instead of 64), reducing these numbers by ~8x — see below.

**Grouped-Query Attention (GQA):**
Heads are grouped, with each group having its own K,V. LLaMA-2 70B uses 8 KV heads with 64 query heads — an eightfold cache reduction. Modern models use GQA as a standard.

## Throughput vs Latency

**Metrics:**
- **TTFT:** time to first token (prefill)
- **ITL:** inter-token latency (decode)
- **Throughput:** tokens/second or requests/second

**Trade-off:**
Larger batch size increases throughput but degrades latency and requires more memory. The choice depends on the scenario:
- Real-time chat: small batch, low latency
- Batch processing: large batch, high throughput
- API service: dynamic batching

**Continuous batching:**
Requests are added to and removed from the batch dynamically. A completed request frees up space immediately. Maximum GPU utilization. All modern inference engines use continuous batching.

## Request Scheduling

**FCFS (First-Come-First-Served):**
Simple, but a long request blocks shorter ones (head-of-line blocking).

**Shortest Job First:**
Optimizes average latency, but response length is unknown in advance — heuristics are needed.

**Priority-based:**
Different priority levels for different clients or request types.

**Preemption:**
Interrupting a current request to process a higher-priority one. vLLM supports this via swapping KV-cache to CPU or recomputation.

## Inference Architecture Taxonomy

Not all inference workloads are the same. The right architecture depends on volume, latency requirements, data sensitivity, and budget. Five deployment patterns exist, from simplest to most complex.

**Cloud API (OpenAI, Anthropic, Google, DeepSeek):** No infrastructure to manage. Pay per token. Fastest time-to-production. Best for: prototyping, low-to-medium volume, when frontier model quality is needed, when operational simplicity matters. Limitations: data leaves your infrastructure, cost unpredictable at high volume, provider outages affect you directly, rate limits constrain burst capacity.

**Self-hosted single-GPU:** One GPU, one model, one process. vLLM or TGI serving Llama-8B or Qwen-14B in INT4. Suitable for models <10B parameters, low traffic, development, privacy-sensitive workloads. The entry point for self-hosting: a single A100-80GB runs a quantized 70B model. Cost: $3-5K/month for cloud GPU, predictable.

**Self-hosted multi-GPU (tensor parallelism):** The model is distributed across GPUs via tensor parallelism. Each layer is split across devices. Required for models that do not fit in a single GPU's memory (70B+ in FP16). Minimal per-request latency, but throughput does not scale linearly — NVLink communication adds overhead. Cost: $10-20K/month for a 2-4 GPU setup.

**Serverless inference (SageMaker Endpoints, Vertex AI, Replicate):** Managed infrastructure with auto-scaling, including scale-to-zero. You deploy a model, the platform handles scaling, load balancing, and GPU management. Best for: variable-load workloads where maintaining always-on GPUs is wasteful, when operational overhead must be minimized. Limitations: cold start latency (30-120 seconds for GPU initialization), limited model customization, vendor lock-in.

**Edge inference (on-device):** Models running on user devices — laptops (Apple Silicon with MLX), phones, IoT devices. Ollama on MacBook, NVIDIA RTX Spark (128GB, 1 PFLOP). Best for: privacy-critical applications where data cannot leave the device, offline scenarios, ultra-low latency (no network round-trip). Limitations: model size constrained by device memory, no access to frontier-quality models, inference speed limited by consumer hardware.

**Multiple replicas (data parallelism):** Independent model copies behind a load balancer. High throughput, fault tolerance, linear scaling. The production standard for self-hosted inference at scale.

**Disaggregated serving:** Prefill and decode on separate clusters. Prefill is compute-intensive (powerful GPUs), decode is memory-intensive (high memory bandwidth). Amazon-Cerebras alliance uses AWS Trainium for prefill + Cerebras WSE-3 for decode, achieving 5x throughput improvement. NVIDIA Dynamo provides a framework for disaggregated serving. This is now a production-standard approach for high-volume deployments — see [[07_SGLang_and_Alternatives|SGLang and Alternatives]].

| Pattern | Latency | Cost | Ops Complexity | Data Privacy | Best For |
|---------|---------|------|---------------|-------------|----------|
| Cloud API | Low | Per-token | Minimal | Low (data leaves) | Prototyping, medium volume |
| Single GPU | Low | Fixed | Low | High | Dev, small models, privacy |
| Multi-GPU TP | Lowest | Fixed (high) | Medium | High | Large models, low-latency |
| Serverless | Variable (cold start) | Per-use | Low | Medium | Variable load |
| Edge | Lowest (no network) | Device cost | Low | Highest | Offline, on-device |
| Multi-replica DP | Low | Fixed (scales) | Medium | High | High throughput |
| Disaggregated | Low | Optimized | High | High | High volume, cost-critical |

## The Request Lifecycle

Understanding what happens between "user sends a prompt" and "user receives a response" is essential for identifying optimization opportunities. Each stage contributes latency, and the slowest stage determines the user experience.

**1. Input processing.** The user's text is received via HTTP/WebSocket. The prompt is assembled: system prompt + conversation history + user message + tool descriptions. Token count is estimated for routing decisions (does this fit the context window? which model tier?).

**2. Tokenization.** The assembled prompt is converted from text to token IDs using the model's tokenizer (BPE). This is CPU-bound and fast (<10ms for most prompts). Token count determines cost and whether the prompt fits the model's context window.

**3. Prefill (prompt processing).** All input tokens are processed in a single forward pass through the model. Compute-intensive: matrix multiplications across all layers. Generates the initial KV-cache — the key and value vectors for every input token at every layer. Latency scales linearly with input length: ~100ms for 4K tokens, ~800ms for 32K, ~3-4 seconds for 128K on an A100. This is the phase where prompt caching saves time — if the prefix matches a cached KV-cache, only the new tokens need prefill.

**4. Decode (token generation).** Tokens are generated one at a time, autoregressively. Each token requires reading the entire KV-cache (memory-bound) and performing a small amount of computation. Inter-token latency (ITL) is typically 20-50ms per token on modern hardware. For a 500-token response: 10-25 seconds of decode time. This is where speculative decoding helps — a smaller draft model generates candidate tokens in bulk, and the main model verifies them in parallel.

**5. Detokenization and delivery.** Token IDs are converted back to text. For streaming responses, each token is sent to the client as it is generated (via SSE or WebSocket). For non-streaming, the complete response is assembled and returned. Post-processing may include: content filtering, structured output validation, tool call extraction.

**Optimization opportunities by stage:** Tokenization (negligible). Prefill (prompt caching: 50-90% savings on repeated prefixes). Decode (speculative decoding: 2-3x speedup; KV-cache quantization: more concurrent requests). Delivery (streaming: perceived latency reduction).

## Key Takeaways

- Autoregressive generation creates unique bottlenecks distinct from training
- Memory bandwidth is the main limitation of the decode phase
- KV-cache grows linearly with context and can exceed the model size
- The trade-off between throughput and latency is inevitable
- Continuous batching is critical for production

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[../16_Distributed_Training/05_Distributed_Training_Practice|Practice: Distributed Training]]
**Next:** [[02_vLLM_internals|vLLM internals: PagedAttention and KV-cache]]
