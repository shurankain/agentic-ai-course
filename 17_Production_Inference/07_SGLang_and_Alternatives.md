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
- Maturity: both production-ready; vLLM has broader adoption, SGLang (v0.5.9+) has matured significantly and matches vLLM in stability

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

**Simplification in 2025:** TensorRT-LLM has progressively improved its developer experience. The high-level Python `LLM` API (introduced in late 2024, stabilized in 2025) reduces the model conversion and deployment process from dozens of steps to a few lines of code, with automatic quantization and engine building. The API now explicitly labels stable vs. unstable components, making it clearer what is production-ready. While still more complex than vLLM, the gap has narrowed significantly.

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
| Maturity | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

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

**Reference results Llama-2-7B A100** (historical baseline; current benchmarks typically use Llama-3-8B+ on H100):
vLLM: 2,500 tok/s, TTFT 45ms, 14.2GB. SGLang single-turn: 2,400 tok/s, multi-turn: 4,200 tok/s. TensorRT-LLM: 2,900 tok/s, TTFT 38ms, less memory. TGI: slightly lower throughput, easier to use.

On current hardware (H100, Llama-3-8B class models), SGLang achieves ~16,200 tok/s vs vLLM's ~12,500 tok/s — a **29% throughput advantage** in high-throughput scenarios (as of early 2026). On DeepSeek V3, SGLang delivers **3.1x faster inference** than vLLM. Prefill-decode disaggregation with large-scale expert parallelism reaches 52,300 input tok/sec and 22,300 output tok/sec per node across 96 H100 GPUs. SGLang now runs natively on TPU via the SGLang-Jax backend, and achieved **25x inference performance on NVIDIA GB300 NVL72** (February 2026). Disaggregated serving implementations across both vLLM and SGLang demonstrate up to 6.4x throughput improvements and 20x reduction in latency variance — this is now a production feature, not experimental.

**vLLM rapid release cycle (as of late June 2026):** vLLM reached **v0.23** (June 13 — 408 commits, 200 contributors) with DiffusionGemma support (first diffusion LLM natively served), DeepSeek-V4 hardening, Model Runner V2 as default for Llama/Mistral, multi-tier KV cache offloading, unified parser for reasoning + tool-call, and Transformers v5 compatibility. The earlier v0.18-v0.19 rapid cycle (late May) had added async scheduler on-by-default, gRPC serving, GPU NGram speculative decoding, FlexKV cache offload, and MLA optimizations for DeepSeek/Kimi models on Blackwell GPUs.

**SGLang hardware expansion (as of late May 2026):** SGLang added support for GB300/B300 Blackwell Ultra systems, RTX PRO 6000 Blackwell workstation GPUs, and DGX Spark — extending its reach from cloud-scale inference to desktop development and edge deployment scenarios.

## NVIDIA Dynamo: Disaggregated Serving Framework

**NVIDIA Dynamo** (2026) is NVIDIA's answer to the disaggregated inference trend — a production framework that separates prefill and decode stages onto different hardware optimized for each workload. Prefill (compute-bound, processes the input prompt) runs on compute-optimized GPUs, while decode (memory-bandwidth-bound, generates tokens one at a time) runs on memory-optimized GPUs or specialized hardware.

Dynamo complements the Amazon+Cerebras disaggregated approach (AWS Trainium for prefill + Cerebras WSE-3 for decode) but keeps the entire pipeline on NVIDIA hardware. Combined with Blackwell GPUs, Dynamo enables 4-10x cost-per-token reduction compared to H100-era monolithic serving.

**Dynamo Snapshot** (June 5, 2026): CRIU + cuda-checkpoint based checkpoint/restore for single-GPU inference on Kubernetes. Serializes full GPU + CPU state, restores on the same or different node. KV cache unmapping reduces checkpoint size from ~190 GiB to ~6 GiB (Qwen3-0.6B on B200). Result: **21x startup reduction** for gpt-oss-120b (under 5 seconds with 8 striped NVMe SSDs). Eliminates cold-start latency for autoscaled inference pods. Currently limited to vLLM workers in preview; no multi-GPU TP validation yet.

**SGLang DFlash + Spec V2** (June 15, 2026): Spec V2 is now the default speculative decoding engine in SGLang. Combined with DFlash (Z Lab), it achieves **>4.3x baseline throughput** and 1.5x throughput vs MTP at concurrency 1 on HumanEval. New DFlash model for Qwen 3.5 397B-A17B outperforms both baseline and native MTP. Tree drafting across triton, FA3, MLA, and aiter backends.

**Practical impact:** Disaggregated serving is now production-standard across both vLLM and SGLang. The benchmarks above (6.4x throughput, 20x latency variance reduction) reflect disaggregated deployments. For new production deployments on Blackwell hardware, disaggregated prefill-decode should be the default architecture.

## Cloudflare Infire: Rust-Based Inference

**Cloudflare Infire** (as of late May 2026) is the first major production-grade Rust-based LLM inference engine. Cloudflare built Infire from scratch, motivated by "lack of control in the Python-based vLLM stack" — Python's GIL, garbage collection pauses, and difficulty debugging production issues at scale drove the decision to rewrite in Rust.

**Performance:** Up to 7% faster than vLLM 0.10 on H100 NVL benchmarks, with significantly lower CPU overhead. The Rust implementation eliminates Python's GIL contention and provides deterministic memory management, resulting in more consistent latency under load.

**Significance:** Infire represents a potential inflection point for inference infrastructure. While vLLM and SGLang dominate the Python ecosystem, a production-validated Rust alternative opens the door for performance-critical deployments where Python's overhead is unacceptable. Cloudflare runs Infire in production for their Workers AI platform, serving millions of inference requests.

**Trade-offs:** The Rust ecosystem for ML is less mature — fewer model implementations, smaller contributor community, and steeper development curve. For most teams, vLLM or SGLang remain the pragmatic choice. Infire is most relevant for hyperscale operators who need maximum control over the inference stack.

## Test-Time Compute: Quality vs Cost

A counterintuitive finding from 2026 production deployments: longer chain-of-thought does **not** guarantee better answers. Correct solutions are often **shorter** than incorrect ones — the model spends more tokens when it is uncertain or going down a wrong path.

This has direct implications for agent loops: use `reasoning_effort` (OpenAI) or `budget_tokens` (Anthropic) not just as cost controls but as quality controls. Low reasoning effort on routine steps (tool call formatting, simple decisions), high reasoning only on complex decisions (architectural choices, ambiguous requirements).

**Reasoning for verification, not generation:** A cost-effective pattern — use a cheap, fast model to generate candidate responses, then use a reasoning model to **verify** the best one. Verification is cheaper than generation because the reasoning model works with a concrete answer rather than an open-ended problem. This mirrors the Architect/Editor pattern from coding agents (see [[../03_AI_Agents_Core/07_Code_Generation_Agents|Code Generation Agents]]).

## Production Considerations

**Monitoring:**
Prometheus endpoints with request/token counters, latency histograms, queue depths. Monitor queue depth — growth signals the need to scale.

**Scaling:**
Kubernetes HorizontalPodAutoscaler with custom metrics (llm_queue_depth). Queue depth >10 → add pods. Account for cold start (minutes) — aggressive scale-up, conservative scale-down.

**Fallback:**
Primary service (SGLang) with a short timeout (5s) → fallback to backup (vLLM) with increased timeout → last resort external API (OpenAI, Anthropic). High availability at reasonable cost.

With production infrastructure understood — from GPU architecture to inference serving — Module 18 addresses the governance, regulatory, and organizational constraints that shape how these systems are deployed in enterprises.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[06_Long_Context_Inference|Long Context Inference]]
**Next:** [[../18_AI_Governance/01_Regulatory_Landscape|AI Regulatory Landscape]]
