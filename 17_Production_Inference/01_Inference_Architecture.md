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

For LLaMA-70B (80 layers, 64 heads, head_dim=128, FP16):
- 1 token = 2.62 MB
- 4K context = 10.7 GB
- 128K context = 343 GB (larger than the model itself ~140 GB!)

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

## Serving Architecture Patterns

**Single GPU:**
The simplest option. The model runs on a single GPU. Suitable for models < 10B parameters, low traffic, development.

**Multi-GPU (tensor parallelism):**
The model is distributed across GPUs via tensor parallelism. Each layer is split across devices. Minimal latency, but throughput does not scale.

**Multiple replicas (data parallelism):**
Independent model copies behind a load balancer. High throughput, fault tolerance, linear scaling.

**Disaggregated serving:**
Prefill and decode on separate clusters. Prefill is compute-intensive (powerful GPUs), decode is memory-intensive (high memory bandwidth). Significant improvements in cost efficiency.

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
