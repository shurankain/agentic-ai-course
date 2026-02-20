# SGLang and vLLM Alternatives

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[06_Long_Context_Inference|Long Context Inference]]
**Next:** [[../18_AI_Governance/01_Regulatory_Landscape|AI Regulatory Landscape]]

---

## Introduction

vLLM is the de facto standard, but not the only option. Alternatives offer significant advantages depending on the use case:

- **SGLang:** Up to 3x throughput for chatbots via RadixAttention
- **TensorRT-LLM:** Maximum performance on NVIDIA GPUs
- **Text Generation Inference (TGI):** Production-ready from Hugging Face

## SGLang

**Philosophy:**
Focus on RadixAttention (KV-cache caching on a prefix tree) and Constrained Decoding (efficient structured output generation).

**RadixAttention:**

Problem: chatbots and multi-turn applications share common prefixes (system prompt, conversation history, few-shot examples). vLLM uses hash-based prefix caching — it searches for an exact match.

SGLang organizes cached prefixes in a radix tree (prefix tree). The root is the system prompt, branches are conversations, and from each — turns. On a new request, the system finds the longest matching prefix in O(L), reusing its entire KV-cache. Stale branches are evicted via LRU.

Advantage: efficient handling of many users with a shared system prompt or parallel conversation branches. Up to 3x throughput for multi-turn conversations.

**Constrained Decoding:**

Structured output generation (JSON, XML, code) from a schema. You describe the structure in code: "generate a name until newline", "age as digits only via regex", "category from a list A/B/C". SGLang translates this into a finite state machine with logit masking on GPU. Batch processing without significant overhead.

Practice: critical for production systems that require guaranteed correct format — data extraction, valid JSON for APIs, code with specific syntax. Correct result on the first attempt without retry.

**Comparison with vLLM:**
- Prefix Caching: vLLM hash-based, SGLang radix tree (better for chatbots)
- Constrained Decoding: vLLM supports JSON mode and guided decoding (via outlines), SGLang full regex/CFG with GPU-accelerated FSM
- Throughput chat: SGLang up to 3x
- Throughput batch: comparable
- Maturity: both production-ready; vLLM has broader adoption, SGLang has matured significantly

**When to choose:**
Ideal: chatbots with long history, applications with frequent similar prefixes, structured output (JSON, code), multi-turn agents. Less suitable: one-off batch, very short prompts, maximum stability required.

## TensorRT-LLM

**Philosophy:**
Maximum performance through custom CUDA kernels, graph compilation, hardware-specific optimizations.

**Architecture:**
The model undergoes conversion into a TensorRT Engine with quantization (FP8/INT8), kernel fusion, and memory optimizations applied. The result is a highly optimized engine file for specific NVIDIA hardware.

**Optimizations:**

Inflight Batching: analogous to continuous batching with low-level GPU resource control.

FP8 on Hopper/Blackwell: native FP8 tensor core support. Doubles throughput vs FP16 with minimal quality loss.

Custom Attention Kernels: specialized CUDA kernels for fused multi-head attention — Flash Attention + paged KV-cache. Optimized at the GPU register level.

**Triton Inference Server:**
Production-grade platform from NVIDIA. Accepts requests, routes them to the TensorRT-LLM backend, and loads the engine file. Adds dynamic batching, model versioning, ensemble pipelines, and monitoring.

**Comparison with vLLM:**
- Max Performance: TensorRT-LLM better by 5-20%
- Setup Complexity: vLLM simple, TensorRT complex
- Model Conversion: vLLM not required, TensorRT required
- Flexibility: vLLM high, TensorRT limited
- FP8: vLLM via parameter, TensorRT native

**When to choose:**
Ideal: maximum performance is critical, NVIDIA GPUs (especially Hopper+), stable production model, cost optimization at scale. Less suitable: rapid prototyping, frequent model changes, limited DevOps resources.

## Text Generation Inference (TGI)

**Philosophy:**
Production-ready serving focused on deployment simplicity, integration with the HF ecosystem, and enterprise features.

**Features:**

Tensor Parallelism: automatic distribution via --num-shard. Llama-2-70B on 4 GPUs: --num-shard 4.

Quantization: bitsandbytes (4/8-bit), GPTQ, AWQ built in.

Speculation: full speculative decoding support, Medusa multi-head, N-gram speculation.

API: simple Python client for connecting, sending a prompt, and receiving results. Synchronous, asynchronous generation, and streaming.

**Comparison with vLLM:**
- HF Integration: vLLM good, TGI native
- Throughput: vLLM higher, TGI good
- Enterprise Features: vLLM basic, TGI more
- Documentation: vLLM good, TGI excellent
- Medusa: vLLM limited, TGI full

**When to choose:**
Ideal: already in the HF ecosystem, enterprise features needed, team familiar with HF, Medusa speculation.

## Comparison Table

| Feature | vLLM | SGLang | TensorRT-LLM | TGI |
|---------|------|--------|--------------|-----|
| Throughput batch | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Throughput chat | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Latency | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Ease of Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Constrained Gen | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Model Support | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| FP8 Support | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Maturity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Decision Framework

**Step 1:** Multi-turn chatbot or agent? → **SGLang** (RadixAttention)

**Step 2:** Maximum performance is critical? → **TensorRT-LLM** (especially Hopper FP8)

**Step 3:** Need structured output? → **SGLang** (constrained decoding)

**Step 4:** General use case? → **vLLM** (universal, reliable)

**Quick Guide:**
- General batch inference: vLLM
- Chatbot / Multi-turn: SGLang
- Maximum throughput, stable model: TensorRT-LLM
- HF ecosystem, enterprise: TGI
- Structured JSON output: SGLang
- Quick prototype: vLLM or TGI
- Hopper GPU, FP8: TensorRT-LLM

## Migration Between Frameworks

**vLLM → SGLang:**
vLLM is imperative (LLM object, generate method). SGLang is declarative (backend endpoint, functions with @sgl.function, operators += and sgl.gen()).

**Pitfalls:**
1. Model compatibility — check the compatibility table
2. Quantization — different methods, requantization may be needed
3. Memory footprint — different overheads, test on target hardware
4. API differences — streaming, batching, timeout, edge cases differ

## Benchmarking

**What to measure:**
1. Throughput (tokens/sec at various batch sizes)
2. Latency (TTFT, ITL)
3. Memory (peak GPU usage)
4. Quality (output identical or equivalent)

**Approach:**
Production-like mode, standard benchmark_serving scripts, identical load parameters. Vary batch size, prompt length, measure sustained throughput.

**Typical results Llama-2-7B A100:**
vLLM: 2,500 tok/s, TTFT 45ms, 14.2GB. SGLang single-turn: 2,400 tok/s, multi-turn: 4,200 tok/s. TensorRT-LLM: 2,900 tok/s, TTFT 38ms, less memory. TGI: slightly lower throughput, easier to use.

## Production Considerations

**Monitoring:**
Prometheus endpoints with request/token counters, latency histograms, queue depths. Monitor queue depth — growth signals the need to scale.

**Scaling:**
Kubernetes HorizontalPodAutoscaler with custom metrics (llm_queue_depth). Queue depth >10 → add pods. Account for cold start (minutes) — aggressive scale-up, conservative scale-down.

**Fallback:**
Primary service (SGLang) with a short timeout (5s) → fallback to backup (vLLM) with increased timeout → last resort external API (OpenAI, Anthropic). High availability at reasonable cost.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[06_Long_Context_Inference|Long Context Inference]]
**Next:** [[../18_AI_Governance/01_Regulatory_Landscape|AI Regulatory Landscape]]
