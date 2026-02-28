# vLLM: Modern Inference Engine

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[01_Inference_Architecture|Production Inference Architecture]]
**Next:** [[03_Speculative_Decoding|Speculative Decoding: Accelerating Generation]]

---

## Why vLLM Became the Standard

vLLM (UC Berkeley, 2023) solved the problem of inefficient memory usage through PagedAttention — an algorithm inspired by OS virtual memory. Result: 2-4x throughput increase.

Advantages:
- Memory efficiency through fragmentation elimination
- High throughput thanks to continuous batching
- Simple API, HuggingFace compatibility
- OpenAI-compatible server
- Active development

## PagedAttention

**The fragmentation problem:**

The traditional approach allocates contiguous memory for the maximum sequence length. Internal fragmentation: 2 GB allocated, 0.5 GB used — 1.5 GB wasted. External fragmentation: after requests complete, memory is fragmented — a new large request cannot find contiguous space.

In production, 60-80% of memory is lost to fragmentation. A system with 32 GB effectively uses only 6-12 GB.

**Solution through paging:**

An idea from 1960s OS design: divide memory into small fixed pages. The KV-cache is divided into blocks (typically 16 tokens). Each request receives a block table, where blocks can be scattered across memory — contiguity is not required.

During prefill, the system allocates the required number of blocks from the pool. During decode, a new token is appended to the last block. When a block fills up, a new one is allocated. When a request completes, all blocks are returned to the pool.

The attention kernel is modified: it uses the block table to locate K and V values, reads data from different blocks, and performs computations. This is transparent to the model.

**Results:**

Internal fragmentation is virtually eliminated — losses only in the last block (~15 tokens). External fragmentation is eliminated — all blocks are the same size. Systems use 60-80% more memory for actual work. A GPU that served 20 requests now serves 50-80.

## Continuous Batching

**The static batching problem:**

The system collects N requests, processes them as a batch, and waits for the longest one to finish. Short requests receive padding. The GPU computes useless padding. New requests wait for the batch to complete.

**Solution:**

Requests are added to and leave the batch on every iteration. When a request finishes, it is immediately released and a new one is added from the queue. The batch is always filled with active requests. No padding from dead requests.

Algorithm at each iteration: identify completed requests (EOS or max length), return their blocks to the pool, estimate available memory, add new requests from the queue, execute a decode step for all active requests.

**Throughput advantage:**

Static batching is limited by the maximum length in the batch. Continuous batching constantly processes new requests. With heterogeneous response lengths, it yields a 2-3x improvement.

**Scheduling:**

The scheduler balances memory management (whether there are enough blocks, whether to apply preemption), fairness (preventing starvation), and batching efficiency (grouping similar requests).

Chunked prefill splits long prompts into parts to reduce processing time variance.

**Disaggregated prefill/decode:** An established best practice in modern inference engines. Prefill (processing the prompt) and decode (generating tokens) have fundamentally different compute characteristics: prefill is compute-bound (parallel processing of many tokens), while decode is memory-bandwidth-bound (sequential token generation). Disaggregated serving separates them onto different GPU pools, each optimized for its workload. Prefill GPUs maximize throughput with large batch sizes; decode GPUs maximize memory efficiency with continuous batching. Communication between pools transfers KV-cache state. This improves overall throughput by 30-60% compared to co-located prefill/decode.

## Tensor Parallelism

**When it is needed:**

LLaMA-70B requires ~140 GB for weights alone in FP16. The H100 has 80 GB — not enough. Solution: distribute the model across multiple GPUs within a server via tensor parallelism.

**Implementation:**

The --tensor-parallel-size parameter specifies the number of GPUs. Attention layers are split by heads: 64 heads across 4 GPUs = 16 heads per GPU. FFN layers are split by intermediate dimension. Communication via AllReduce after each layer. NVLink provides 600-900 GB/s bandwidth.

Pipeline parallelism splits the model by layers for even larger models or multi-node setups. These can be combined: tensor-parallel-size 4 + pipeline-parallel-size 2 for 8 GPUs.

## Prefix Caching

**The problem:**

A chatbot uses the same system prompt for all users. RAG includes the same context for similar questions. Few-shot prompts repeat examples. Without caching, every request triggers a full prefill.

**Automatic prefix caching:**

The --enable-prefix-caching parameter. vLLM computes a hash of the prefix and searches the cache. Cache hit — reuses KV blocks, computes only the unique part. Cache miss — performs prefill, saves blocks with the hash.

A chatbot with a 1000-token system prompt: first request takes 100ms for prefill, subsequent requests take 2ms instead of 102ms — a 50x speedup. Cached blocks are physically shared through PagedAttention — all requests reference the same blocks.

## Practical Deployment

**Basic launch:**
python -m vllm.entrypoints.openai.api_server with model and port parameters.

**Production configuration:**
- --tensor-parallel-size for multi-GPU
- --dtype bfloat16 for H100
- --max-model-len sets the maximum context
- --gpu-memory-utilization 0.90 (typically 0.85-0.90 in production)
- --enable-prefix-caching
- --disable-log-requests

The gpu-memory-utilization parameter controls the reservation for KV-cache. max-model-len determines allocation planning — a smaller value allows more requests.

## vLLM V1 Engine Architecture (2025)

vLLM has undergone a major internal rewrite with the V1 engine, designed to support the expanding requirements of modern LLM serving.

### Multi-Modal Serving

vLLM V1 natively supports multi-modal models — serving models that accept images, audio, and video alongside text:
- **Image inputs** are preprocessed, tokenized (via the model's vision encoder), and managed alongside text tokens in the KV-cache
- **Variable-length visual tokens** — different images produce different numbers of visual tokens, requiring dynamic memory management (PagedAttention handles this naturally)
- Supported models include LLaVA, Llama 4 (native multimodal), Qwen-VL, InternVL, and others
- The OpenAI-compatible API accepts image URLs or base64-encoded images in the messages format

### LoRA Serving

Production systems often need to serve multiple fine-tuned variants of the same base model. vLLM supports **multi-LoRA serving** — loading and serving multiple LoRA adapters simultaneously on a single base model:
- The base model weights are loaded once and shared across all requests
- LoRA adapter weights (typically 0.1-1% of base model size) are loaded per-adapter
- Request-level routing: each request specifies which LoRA adapter to use
- **S-LoRA optimization:** adapters are dynamically loaded and unloaded from GPU memory based on demand, with a unified memory pool for both base model KV-cache and adapter weights
- This enables serving hundreds of fine-tuned models on a single GPU cluster without deploying separate instances for each

### Structured Output (Guided Decoding)

vLLM V1 supports guided decoding to constrain model output to valid formats:
- **JSON schema enforcement** — output conforms to a specified JSON schema
- **Regex constraints** — output matches a specified regular expression
- **Grammar-based** — output follows a formal grammar (e.g., for SQL, code)
- Implementation uses token-level masking: at each step, only tokens consistent with the constraint are allowed

### Performance Improvements

The V1 engine brings significant architectural improvements:
- **Torch.compile integration** — automatic kernel fusion and optimization for supported models
- **Improved scheduling** — better handling of mixed prefill/decode batches
- **Reduced overhead** — lower Python-level overhead for high-throughput serving
- **Speculative decoding integration** — native support for draft-model-based speculative decoding to reduce latency

## Alternatives

**SGLang (LMSYS):**
RadixAttention — prefix caching on a radix tree instead of a hash. Finds the longest matching prefix, allows partial reuse. Up to 3x improvement in chatbots. Python DSL for constrained generation.

**TensorRT-LLM (NVIDIA):**
Maximum performance through deep kernel optimization. Difficult to configure, but best latency for single requests. Enterprise deployments.

**llama.cpp:**
Edge and local deployments. GGML quantization for extreme compression. Simple setup, low requirements. Popular for local AI.

**Choosing:**
- High-volume API serving — vLLM
- Chatbots with prefix reuse — SGLang
- Maximum performance on NVIDIA — TensorRT-LLM
- Local/edge — llama.cpp

## Key Takeaways

PagedAttention solves memory fragmentation through the OS virtual memory concept. 2-4x increase in concurrent requests.

Continuous batching eliminates GPU idle time through dynamic batch management at the iteration level. 2-3x throughput improvement.

Prefix caching eliminates redundant computations. Up to 50x prefill speedup for shared prefixes.

Tensor parallelism makes large models accessible through transparent distribution across GPUs.

Production-ready out of the box with an OpenAI-compatible API.

vLLM V1 extends capabilities: multi-modal serving (images, audio, video), multi-LoRA serving (hundreds of fine-tuned variants on shared base), disaggregated prefill/decode (30-60% throughput gain), structured output enforcement, and speculative decoding integration.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[01_Inference_Architecture|Production Inference Architecture]]
**Next:** [[03_Speculative_Decoding|Speculative Decoding: Accelerating Generation]]
