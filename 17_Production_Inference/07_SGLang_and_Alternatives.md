# SGLang and vLLM Alternatives

vLLM is the de facto standard, but not the only option. Alternatives offer significant advantages depending on the use case:

- **SGLang:** Up to 3x throughput for chatbots via RadixAttention
- **TensorRT-LLM:** Maximum performance on NVIDIA GPUs
- **Text Generation Inference (TGI):** Production-ready from Hugging Face

## SGLang

**Philosophy:**
Focus on RadixAttention (KV-cache caching on a prefix tree) and Constrained Decoding (efficient structured output generation).

**RadixAttention:**

Problem: chatbots and multi-turn applications share common prefixes (system prompt, conversation history, few-shot examples). vLLM uses hash-based prefix caching, it searches for an exact match.

SGLang organizes cached prefixes in a radix tree (prefix tree). The root is the system prompt, branches are conversations, and from each, turns. On a new request, the system finds the longest matching prefix in O(L), reusing its entire KV-cache. Stale branches are evicted via LRU.

Advantage: efficient handling of many users with a shared system prompt or parallel conversation branches. Up to 3x throughput for multi-turn conversations.

**Constrained Decoding:**

Structured output generation (JSON, XML, code) from a schema. You describe the structure in code: "generate a name until newline", "age as digits only via regex", "category from a list A/B/C". SGLang translates this into a finite state machine with logit masking on GPU. Batch processing without significant overhead.

Practice: critical for production systems that require guaranteed correct format: data extraction, valid JSON for APIs, code with specific syntax. Correct result on the first attempt without retry.

**Comparison with vLLM:**
- Prefix Caching: vLLM hash-based, SGLang radix tree (better for chatbots)
- Constrained Decoding: vLLM supports JSON mode and guided decoding (via outlines), SGLang full regex/CFG with GPU-accelerated FSM
- Throughput chat: SGLang up to 3x
- Throughput batch: comparable
- Maturity: both production-ready; vLLM has broader adoption, SGLang (v0.5.9+) has matured and matches vLLM in stability

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

Custom Attention Kernels: specialized CUDA kernels for fused multi-head attention: Flash Attention + paged KV-cache. Optimized at the GPU register level.

**Triton Inference Server:**
Production-grade platform from NVIDIA. Accepts requests, routes them to the TensorRT-LLM backend, and loads the engine file. Adds dynamic batching, model versioning, ensemble pipelines, and monitoring.

**Comparison with vLLM:**
- Max Performance: TensorRT-LLM better by 5-20%
- Setup Complexity: vLLM simple, TensorRT complex
- Model Conversion: vLLM not required, TensorRT required
- Flexibility: vLLM high, TensorRT limited
- FP8: vLLM via parameter, TensorRT native

**Simplification in 2025:** TensorRT-LLM has progressively improved its developer experience. The high-level Python `LLM` API (introduced in late 2024, stabilized in 2025) reduces the model conversion and deployment process from dozens of steps to a few lines of code, with automatic quantization and engine building. The API now explicitly labels stable vs. Unstable components. This makes it clearer what is production-ready. While still more complex than vLLM, the gap has narrowed.

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

Rated 1 to 5, where 5 is the best of these four for that dimension rather than an absolute score.

| Feature | vLLM | SGLang | TensorRT-LLM | TGI |
|---------|------|--------|--------------|-----|
| Throughput, batch | 5 | 4 | 5 | 4 |
| Throughput, chat | 4 | 5 | 4 | 3 |
| Latency | 4 | 4 | 5 | 4 |
| Ease of use | 5 | 4 | 2 | 5 |
| Constrained generation | 3 | 5 | 2 | 3 |
| Model support | 5 | 4 | 3 | 5 |
| FP8 support | 3 | 3 | 5 | 3 |
| Maturity | 5 | 4 | 4 | 5 |

## Decision Framework

**Step 1:** Multi-turn chatbot or agent? Use **SGLang** (RadixAttention).

**Step 2:** Maximum performance is critical? Use **TensorRT-LLM** (especially Hopper FP8).

**Step 3:** Need structured output? Use **SGLang** (constrained decoding).

**Step 4:** General use case? Use **vLLM** (universal, reliable).

**Quick Guide:**
- General batch inference: vLLM
- Chatbot / Multi-turn: SGLang
- Maximum throughput, stable model: TensorRT-LLM
- HF ecosystem, enterprise: TGI
- Structured JSON output: SGLang
- Quick prototype: vLLM or TGI
- Hopper GPU, FP8: TensorRT-LLM

## Migration Between Frameworks

**vLLM to SGLang:**
vLLM is imperative (LLM object, generate method). SGLang is declarative (backend endpoint, functions with @sgl.function, operators += and sgl.gen()).

**Pitfalls:**
1. Model compatibility: check the compatibility table
2. Quantization: different methods, requantization may be needed
3. Memory footprint: different overheads, test on target hardware
4. API differences, streaming, batching, timeout, edge cases differ

## Benchmarking

**What to measure:**
1. Throughput (tokens/sec at batch sizes)
2. Latency (TTFT, ITL)
3. Memory (peak GPU usage)
4. Quality (output identical or equivalent)

**Approach:**
Production-like mode, standard benchmark_serving scripts, identical load parameters. Vary batch size, prompt length, measure sustained throughput.

**Reference results Llama-2-7B A100** (historical baseline; current benchmarks typically use Llama-3-8B+ on H100):
vLLM: 2,500 tok/s, TTFT 45ms, 14.2GB. SGLang single-turn: 2,400 tok/s, multi-turn: 4,200 tok/s. TensorRT-LLM: 2,900 tok/s, TTFT 38ms, less memory. TGI: slightly lower throughput, easier to use.

On current hardware (H100, Llama-3-8B class models), SGLang achieves ~16,200 tok/s vs vLLM's ~12,500 tok/s, a **29% throughput advantage** in high-throughput scenarios (as of early 2026). On DeepSeek V3, SGLang delivers **3.1x faster inference** than vLLM. Prefill-decode disaggregation with large-scale expert parallelism reaches 52,300 input tok/sec and 22,300 output tok/sec per node across 96 H100 GPUs. SGLang now runs natively on TPU via the SGLang-Jax backend, and achieved **25x inference performance on NVIDIA GB300 NVL72** (February 2026). Disaggregated serving implementations across both vLLM and SGLang demonstrate up to 6.4x throughput improvements and 20x reduction in latency variance. This is now a production feature, not experimental.

**vLLM rapid release cycle (as of October 2026):** the current line is **v0.32.0**, released October 18 with 814 commits from 363 contributors, two weeks after v0.31.0. Three changes in that pair matter for production. v0.31 added admission-control flags, `--max-num-queued-reqs` and `--max-num-queued-tokens`, which let a server refuse work at the door instead of letting its queue grow without bound, and removed the need to pin `PYTHONHASHSEED` for KV-cache users. v0.32 added a generic tool-call and reasoning parser that reads a checkpoint's own response template, so a new model no longer needs a model-specific parser, and a served-by-default endpoint that answers choice and score questions, the open-source counterpart of the decision endpoints in Module 08, Structured Output Techniques. Check the release notes before relying on it; it is days old.

**v0.24.0** (August 2026) carried the breaking change most likely to bite a deployment upgrading from the summer: **vLLM no longer sets `CUDA_VISIBLE_DEVICES` internally**, and a new `device_ids` argument replaces it. Any deployment that relied on the old implicit behaviour to pin GPUs will bind differently after the upgrade, and the failure is silent rather than loud. Also new: online FP8 per-token-per-channel quantization, cgroup-aware KV cache sizing (so a container's memory limit is respected instead of the host's), and support for MiniMax-M3 and DiffusionGemma.

Read the version number correctly, because the numbering misleads people into waiting for a milestone that is not coming. Since v0.12.0, ordinary releases increment the **minor** version, so v0.32 is a routine release rather than a late beta, and there is no v1.0 pending. Pin exact versions and read the changelog per bump; the project's cadence means a minor number can carry a breaking change.

**v0.23** (June 13, 2026: 408 commits, 200 contributors) added DiffusionGemma support (first diffusion LLM natively served), DeepSeek-V4 hardening, Model Runner V2 as default for Llama/Mistral, multi-tier KV cache offloading, a unified parser for reasoning and tool calls, and Transformers v5 compatibility. The earlier v0.18-v0.19 rapid cycle (late May) had added async scheduler on-by-default, gRPC serving, GPU NGram speculative decoding, FlexKV cache offload, and MLA optimizations for DeepSeek/Kimi models on Blackwell GPUs.

**SGLang hardware expansion (as of late May 2026):** SGLang added support for GB300/B300 Blackwell Ultra systems, RTX PRO 6000 Blackwell workstation GPUs, and DGX Spark, extending its reach from cloud-scale inference to desktop development and edge deployment scenarios.

## NVIDIA Dynamo: Disaggregated Serving Framework

**NVIDIA Dynamo** (2026) is NVIDIA's answer to the disaggregated inference trend, a production framework that separates prefill and decode stages onto different hardware optimized for each workload. Prefill (compute-bound, processes the input prompt) runs on compute-optimized GPUs, while decode (memory-bandwidth-bound, generates tokens one at a time) runs on memory-optimized GPUs or specialized hardware.

Dynamo complements the Amazon+Cerebras disaggregated approach (AWS Trainium for prefill + Cerebras WSE-3 for decode) but keeps the entire pipeline on NVIDIA hardware. Combined with Blackwell GPUs, Dynamo enables 4-10x cost-per-token reduction compared to H100-era monolithic serving.

**Dynamo Snapshot** (June 5, 2026): CRIU + cuda-checkpoint based checkpoint/restore for single-GPU inference on Kubernetes. Serializes full GPU + CPU state, restores on the same or different node. KV cache unmapping reduces checkpoint size from ~190 GiB to ~6 GiB (Qwen3-0.6B on B200). Result: **21x startup reduction** for gpt-oss-120b (under 5 seconds with 8 striped NVMe SSDs). Eliminates cold-start latency for autoscaled inference pods. Currently limited to vLLM workers in preview; no multi-GPU TP validation yet.

**SGLang DFlash + Spec V2** (June 15, 2026): Spec V2 is now the default speculative decoding engine in SGLang. Combined with DFlash (Z Lab), it achieves **>4.3x baseline throughput** and 1.5x throughput vs MTP at concurrency 1 on HumanEval. New DFlash model for Qwen 3.5 397B-A17B outperforms both baseline and native MTP. Tree drafting across triton, FA3, MLA, and aiter backends.

**Practical impact:** Disaggregated serving is now production-standard across both vLLM and SGLang. The benchmarks above (6.4x throughput, 20x latency variance reduction) reflect disaggregated deployments. For new production deployments on Blackwell hardware, disaggregated prefill-decode should be the default architecture.

## Cloudflare Infire: Rust-Based Inference

**Cloudflare Infire** (as of late May 2026) is the first major production-grade Rust-based LLM inference engine. Cloudflare built Infire from scratch, motivated by "lack of control in the Python-based vLLM stack": Python's GIL, garbage collection pauses, and difficulty debugging production issues at scale drove the decision to rewrite in Rust.

**Performance:** Up to 7% faster than vLLM 0.10 on H100 NVL benchmarks, with lower CPU overhead. The Rust implementation eliminates Python's GIL contention and provides deterministic memory management, resulting in more consistent latency under load.

**Significance:** Infire represents a potential inflection point for inference infrastructure. While vLLM and SGLang dominate the Python ecosystem, a production-validated Rust alternative opens the door for performance-critical deployments where Python's overhead is unacceptable. Cloudflare runs Infire in production for their Workers AI platform, serving millions of inference requests.

**Trade-offs:** The Rust ecosystem for ML is less mature: fewer model implementations, smaller contributor community, and steeper development curve. For most teams, vLLM or SGLang remain the pragmatic choice. Infire is most relevant for hyperscale operators who need maximum control over the inference stack.

## Test-Time Compute: Quality vs Cost

A counterintuitive finding from 2026 production deployments: longer chain-of-thought does **not** guarantee better answers. Correct solutions are often **shorter** than incorrect ones, the model spends more tokens when it is uncertain or going down a wrong path.

This has direct implications for agent loops: use `reasoning_effort` (OpenAI) or `budget_tokens` (Anthropic) as quality controls as well as cost controls. Low reasoning effort on routine steps (tool call formatting, simple decisions), high reasoning only on complex decisions (architectural choices, ambiguous requirements).

**Reasoning for verification, not generation:** A cost-effective pattern: use a cheap, fast model to generate candidate responses, then use a reasoning model to **verify** the best one. Verification is cheaper than generation because the reasoning model works with a concrete answer rather than an open-ended problem. This mirrors the Architect/Editor pattern from coding agents (see Module 03, Code Generation Agents).

## Production Considerations

**Monitoring:**
Prometheus endpoints with request/token counters, latency histograms, queue depths. Monitor queue depth: growth signals the need to scale.

**Scaling:**
Kubernetes HorizontalPodAutoscaler with custom metrics (llm_queue_depth). Queue depth >10: add pods. Account for cold start (minutes): aggressive scale-up, conservative scale-down.

**Fallback:**
Primary service (SGLang) with a short timeout (5s), then fallback to backup (vLLM) with increased timeout, then last resort external API (OpenAI, Anthropic). High availability at reasonable cost.

**Security of the serving stack:**
The engine is rarely the only component, and the add-ons carry their own risk. In October 2026 JFrog disclosed CVE-2026-105192, an unauthenticated remote-code-execution flaw rated CVSS 9.8 in the multiprocess mode of LMCache, a KV-cache layer commonly paired with vLLM, with no fixed version at disclosure. Three habits limit the damage of the next one: inventory every component in the serving path, including the add-ons around the engine; keep cache and control endpoints off any network a client can reach; and pin and track each component's version the same way the engine is pinned, since a two-week release cadence applies to the ecosystem as well. Check the LMCache advisory for a fix before deploying it.

With production infrastructure understood: from GPU architecture to inference serving: Module 18 addresses the governance, regulatory, and organizational constraints that shape how these systems are deployed in enterprises.

## Key Takeaways

1. **SGLang's RadixAttention** organizes cached prefixes in a radix tree. This enables efficient KV-cache reuse across multi-turn conversations: up to 3x throughput for chatbots and agent loops with shared system prompts.

2. **TensorRT-LLM delivers maximum single-GPU throughput** (5-20% over vLLM) through custom CUDA kernels and graph compilation, at the cost of higher operational complexity and a required model conversion step.

3. **Decision framework: vLLM for general purpose, SGLang for multi-turn agents, TRT-LLM for maximum throughput.** Start with vLLM as the default; switch to SGLang when multi-turn prefix reuse or constrained decoding matters; switch to TRT-LLM only when peak performance justifies the operational overhead.

4. **NVIDIA Dynamo separates prefill and decode** onto hardware optimized for each workload. Combined with Blackwell GPUs, disaggregated serving enables 4-10x cost-per-token reduction and is now the default architecture for new production deployments.

5. **Production infrastructure requires Prometheus monitoring, HPA scaling on queue depth, and multi-service fallback.** Queue depth growth signals the need to scale; a primary-backup-API fallback chain (SGLang with 5s timeout, vLLM backup, external API last resort) provides high availability at reasonable cost.
