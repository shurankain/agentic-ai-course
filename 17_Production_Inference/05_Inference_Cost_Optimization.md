# Cost Optimization Strategies

## LLM Inference Economics

Inference requires significant expenditure. A single GPT-5.6 request with long context costs several cents. At millions of requests per day, costs reach hundreds of thousands of dollars per month.

**Cost Structure:**

GPU compute is the primary expense. Cloud service prices (approximate, vary by provider and region, as of August 2026):
- T4: $0.50-1.00/hour
- L4: $0.70-1.20/hour
- A100 40GB: $2.50-4.00/hour
- A100 80GB: $3.50-5.00/hour
- H100: $4.50-8.00/hour
- H200: $6.00-10.00/hour
- B200 (Blackwell): $8.00-12.00/hour (in production, available from major cloud providers)

Llama 70B on A100-80GB: ~$0.001-0.002 per 1000 tokens. At 100M tokens per day: $3000-6000 per month (compute only).

Memory limits model size and batch size. Networking in multi-GPU adds 10-20%. Storage negligible.

**Treat these hourly rates as the least stable line in your model.** Memory, not silicon, became the binding cost during 2026: DRAM went from about $2.80 per GB to roughly $12 as AI data centers absorbed the bulk of global output, and Microsoft attributed $25B of its CapEx increase to memory and component prices alone. Accelerator rental prices lag that move rather than escaping it. Two practical consequences. A twelve-month cost projection built on today's rates needs a sensitivity band rather than a point estimate, and any optimization that cuts memory footprint, meaning quantization, KV-cache compression, and paged attention, now buys more than it did when memory was cheap. Module 15, GPU Architecture covers the supply picture.

**Metrics:**

Cost per token (separate input/output). Cost per request (look at p50, p90, p99, not just average). GPU utilization (target >70%). Throughput per dollar (tokens/second/$).

## Spot Instances: 60-90% Savings

**What they are:**
Unused compute capacity at a discount. AWS Spot 60-90% cheaper (average ~70%), GCP Preemptible 60-80%, Azure Spot 60-90%.

A100 $4/hour on-demand, spot $0.80-1.20/hour. Running 24/7 saves $60-75K per year per instance.

Catch: instances can be reclaimed with 30 seconds notice.

**When suitable:**
Batch processing (progress checkpointing), development/testing, async workloads, mixed architectures (spot for burst on top of on-demand baseline).

**NOT suitable:**
Real-time API with strict SLA, stateful long-running connections, mission-critical inference.

**Strategies:**

Checkpointing for batch, saving every N batches, resuming from checkpoint. Overhead ~1-2%, savings 70%.

Mixed fleet: 2-3 on-demand baseline + 5-10 spot burst. On spot preemption, on-demand absorbs load.

Multi-region/multi-zone deployment: diversification reduces risk.

Fallback on spot loss, switching to smaller model or queue.

## Cascading Models

**Concept:**
Not all requests require a large model. Simple questions: small model (fractions of a cent). Complex ones: large model.

Router directs by complexity:
- Tier 1 (7B): simple factual questions
- Tier 2 (13-30B): general-purpose, moderate complexity
- Tier 3 (70B+): complex reasoning, code generation

**Routing strategies:**

Rule-based: heuristics by length, keywords, patterns. Zero latency, deterministic, but not always accurate.

Model-based: lightweight classifier predicts complexity. More accurate (85-90% vs 70-75%), but adds latency ~10-20ms.

Confidence-based: attempt small model, fallback to large if confidence is low (via logprobs). Adaptive, but two inference calls for some.

**Cost savings, real numbers:**

Traffic: 60% simple (7B), 30% medium (13B), 10% complex (70B).

Without cascading: 100K requests/day x $0.01 = $1000/day = $30K/month.

With cascading: 60K x $0.001 + 30K x $0.003 + 10K x $0.01 = $250/day = $7.5K/month.

Savings: 75%.

Trade-offs: complexity, potential quality degradation, latency overhead ~10-50ms.

**Optimization:**
Monitor routing accuracy (if >20% negative feedback, router is too aggressive). A/B testing quality. Dynamic thresholds based on load.

**Two costs a naive router misses.** The first is cache invalidation: switching models mid-session can drop the prompt cache, and on a long agentic session that loss can exceed the per-token saving that motivated the switch. Count it in the cost function. The second is retries. A cheaper model that needs two extra attempts is not cheaper, so the unit to optimize is **cost per accepted result**, not cost per token or per call.

That second point generalizes past routing. Every economy in this chapter should be measured against accepted results, because quantization, a smaller model, and a shorter context all buy their savings with some probability of a rerun. A comparison denominated in tokens hides exactly the failure mode that makes the cheap option expensive.

Cursor's production router, published in August 2026, is the most detailed public account of one and reports 41% to 68% cost reductions at equal or better satisfaction. Module 03, Resource Optimization covers its design, its training label, and the reasons to treat its percentages as specific to a coding product.

## Right-Sizing Infrastructure

**GPU selection:**

Do not overpay. Match GPU to workload.

Llama 7B: T4 (20-30 tok/s, $0.50/hr) or L4 (40-60 tok/s, $1.00/hr) perfectly adequate. A100 ($3.50/hr) overkill.

Llama 70B: A100-40GB will not fit. A100-80GB 1 GPU (8-12 tok/s, $5/hr) with quantization. H100 1 GPU (15-20 tok/s, $7/hr) better price/performance.

Anti-patterns: 7B on H100 (10x overpaying), A100-80GB instead of A100-40GB for models <40GB, multi-GPU without tensor parallelism.

**Instance right-sizing:**

CPU bottlenecks, preprocessing can be limiting. If GPU utilization <60% at full load, the problem is CPU. More vCPUs or optimize preprocessing.

Memory requirements: insufficient RAM leads to swapping. Rule: RAM >= 2x model size. For Llama 70B FP16: 140GB model + 50GB overhead = 190GB+ RAM.

Network bandwidth: multi-GPU requires high bandwidth. Monitor utilization, if >80% network-bound.

Monitoring: GPU utilization target >70%. CPU 50-70%. Memory <80%.

**Batch size optimization:**

Larger batch to higher throughput, but higher latency. The trade-off depends on the use case.

Llama 13B on A100: batch 1 (50 tok/s, 20ms), batch 4 (180 tok/s, 45ms), batch 16 (500 tok/s, 150ms), batch 64 (800 tok/s, 600ms).

Real-time chat: batch 1-4. Document processing: batch 16-64. Async API: batch 8-16.

Dynamic batching: collect up to max_batch_size or timeout (50ms). Low traffic: smaller batches (low latency), high traffic: larger (high throughput).

Cost impact: batch 32 can be 3-4x cheaper per token than batch 1.

## Autoscaling

**Metrics-based (reactive):**

GPU utilization: scale up >80% for 3-5 min, scale down <40% for 10-15 min. Asymmetric timings: fast up, slow down.

Queue depth: if estimated wait time > target latency, scale up. More direct link to user experience.

Custom metrics: requests/sec, latency, error rate via Kubernetes HPA + Prometheus.

**Predictive (proactive):**

Traffic patterns are predictable (daily/weekly cycles). Train time-series model (ARIMA, Prophet, ML) on historical data. Predict load 15-30 min ahead, scale in advance. Zero latency spike.

Challenges: prediction errors. Combine predictive + reactive: predictive for baseline, reactive for spikes.

**Scale-to-zero:**

For irregular usage, complete shutdown when no traffic. Dramatic savings for low-traffic (dev/staging 67%, internal tools 95%+).

Trade-off: cold start latency 30-90 sec. Mitigation: keep-warm requests, gradual scale-down, pre-warming.

Best for dev/staging/internal tools. Not for production APIs with SLA.

## Monitoring and Budgeting

**Key metrics:**

Cost: per 1K tokens (input vs output), per request (p50/p90/p99), daily/weekly/monthly spend, by customer/feature.

Efficiency: GPU utilization (>70%), throughput per GPU, cost vs baseline, cache hit rate (>50% good, <20% not worthwhile).

**Alerting:**

Cost anomalies: daily spend >120% of 7-day average, cost per request >150% baseline, customer spend >$X threshold.

Performance: GPU utilization <50% for 30+ min, latency >2x normal, error rate >5%.

Budget: 70% daily budget warning, 90% critical, 100% rate limiting or fallback.

**Budget controls:**

Hard limits: max requests per period, max tokens per request, spend caps per customer, total budget cap.

Soft limits: warning at 70%, degraded service at 85%, queue prioritization.

Graceful degradation: Normal, then 70% warnings, then 85% aggressive cascading, then 95% queue/rate limiting, then 100% free tier disabled.

## Serving MoE Models

Mixture-of-Experts (MoE) models like DeepSeek V3 (671B total, 37B active), Mixtral 8x22B, and Llama 4 Maverick require specialized serving strategies because their total parameter count far exceeds what's needed per inference.

**Memory vs. compute tradeoff:** An MoE model's full weights must fit in memory (DeepSeek V3: ~670GB in INT8), but only a fraction activates per token (37B for DeepSeek V3). This means high memory requirements but lower per-token compute than a dense model of the same total size.

**Serving strategies:**

**Tensor parallelism across GPUs:** Distribute the full model across multiple GPUs. For DeepSeek V3 in INT8: 8-9x A100 80GB or 4x H100. Expert layers are distributed so each GPU holds a subset of experts, with all-to-all communication during expert routing.

**Expert offloading:** Keep inactive experts on CPU/NVMe and load them on demand. Dramatically reduces GPU memory (serve DeepSeek V3 on 2x A100 instead of 8), but adds latency per token. Viable for batch or latency-tolerant workloads. Implementations: llama.cpp, ExLlamaV2.

**Quantization is highly effective for MoE:** Since most parameters are expert weights that activate sparsely, aggressive quantization (INT4, even INT2 for inactive experts) has less impact on quality than for dense models. DeepSeek V3 in INT4 fits on 4x A100 80GB with minimal quality loss.

**KV-cache considerations:** MoE models with MLA (Multi-head Latent Attention, used by DeepSeek V3) compress the KV-cache, 3-5x smaller than standard MHA. This partially offsets the high parameter memory. For MoE models without MLA (Mixtral), KV-cache follows standard sizing.

**vLLM and SGLang support:** Both support MoE model serving with tensor parallelism. vLLM's V1 engine added optimized MoE routing kernels. SGLang supports expert parallelism (EP) for distributing experts across GPUs independently of tensor parallelism.

**Practical guidance:** For models <100B total params (Mixtral 8x7B): standard serving, 1-2 GPUs. For 100B-700B total (Mixtral 8x22B, DeepSeek V3): multi-GPU tensor parallelism + quantization. For cost optimization: expert offloading trades latency for 2-4x memory reduction.

## Prompt Caching Economics

For API-based inference (as opposed to self-hosted), **prompt caching** is one of the highest-impact cost optimizations, and it requires no code changes beyond structuring prompts correctly.

**How it works:** LLM providers cache the KV-cache for the prefix (identical initial portion) of your prompt. If the same prefix appears in the next request, the cached KV-cache is reused, and the provider charges a discounted rate for the cached tokens.

**Provider economics (as of early 2026):**

| Provider | Cache Hit Discount | Min Prefix Length | TTL | Effective Savings |
|----------|-------------------|-------------------|-----|-------------------|
| **Anthropic** | 90% off input | 1,024 tokens | 5 min | 50-80% per-turn for agents |
| **OpenAI** | 50% off input | 1,024 tokens | 5-10 min | 25-40% per-turn for agents |
| **Google** | Varies | 32K tokens | Varies | Significant for long-context |

**The pattern for agents:** Structure every LLM call as: `[system prompt] + [tool descriptions] + [static context]` -> `[dynamic context] + [conversation] + [current message]`. The first block is identical across all calls in a session: fully cacheable. The second block varies per call: never cached.

**Why this matters for multi-turn agents:** An agent making 15 calls per task with a 3K-token system prompt and 5K tokens of tool descriptions saves 8K cached tokens per call x 14 subsequent calls = 112K tokens cached. At Anthropic's 90% discount, this saves approximately $0.50 per task, material at thousands of tasks per day.

**Batching API** offers a complementary optimization: 50% discount on all tokens for deferred processing (24-hour SLA). Suitable for evaluation runs, document processing, and any workload that does not require real-time response. Combine prompt caching (for real-time) with batching (for async) for maximum savings.

See Module 02, Context Engineering for the broader context management framework, including KV-cache management patterns for agent identity stability.

## Queueing Theory for Capacity Planning

**Little's Law: the fundamental law:**

The formula `L = lambda * W` connects three key metrics:
- `L`: average number of requests in the system (queue + processing)
- `λ`: request arrival rate (requests/sec)
- `W`: average time in the system (queue time + processing time)

**Application example:**

If you have 10 requests per second and each takes 3 seconds to process, then L = 10 x 3 = 30 requests are simultaneously in the system. This means you need at least 30 concurrent connections. But this is the average case. To handle peaks (P99 latency = 15s), you need peak = 10 x 15 = 150 connections. Therefore, capacity planning should target percentile latency, not the average.

**M/M/c Model for Capacity Planning:**

The classic queueing model helps determine system utilization using the formula `Utilization (ρ) = λ / (c x μ)`, where μ is the service rate (requests per second per worker), and c is the number of workers.

**Utilization rules:**
- ρ > 1 to the queue grows infinitely (system is overloaded)
- ρ > 0.8 to sharp latency increase (danger zone)
- ρ = 0.6-0.7 to optimal load

**Practical capacity planning:**

Suppose you need to handle 100 requests per second, and the LLM latency is 3 seconds on average. Service rate per worker = 1/3 = 0.33 requests per second. Minimum workers: 100 / 0.33 = 300. But accounting for a target utilization of ρ = 0.7 for headroom: 300 / 0.7 ≈ 430 concurrent connections. In practice, plan for ~450 connections to account for burst load.

**Why this matters for LLMs:**
- LLM latency is high (seconds) and variable, unlike microsecond REST APIs
- Cost of waiting is high, users leave after a few seconds of waiting
- API rate limits create an artificial ceiling that cannot be overcome simply by adding workers
- Load spikes require buffer capacity, otherwise the queue overflows

---

## Key Takeaways

Spot instances: low-hanging fruit. 60-90% savings for eligible workloads. Combine with on-demand for resilience.

Cascading architecture: biggest reducer. With 70% simple requests, savings reach 75%.

Right-size aggressively. Don't run 7B on H100. Monitor utilization, target >70% GPU, 50-70% CPU.

Autoscaling essential for variable traffic. Reactive + predictive for best results.

Monitor everything relentlessly. Track cost per token, utilization, throughput per dollar. Set alerts.

Implement budget controls. Hard limits prevent surprises. Soft limits balance cost and quality.

Think about trade-offs: spot (cost vs availability), cascading (cost vs quality), large batches (cost vs latency), quantization (cost vs accuracy).

Prompt caching is free optimization for API users: structure prompts with static prefix for automatic 50-90% input cost reduction on multi-turn agent sessions.

**Hardware cost trajectory (as of mid-June 2026):** Inference costs have fallen approximately 280x over two years (early 2024 to early 2026), and the decline continues: **$0.06/1K tokens (early 2025) to $0.006/1K tokens (mid-2026) = 10x in 18 months**. The combination of Blackwell hardware (~1,000 NVL72 racks/week per major hyperscaler), optimized serving stacks, and open-source MoE models drives an additional **4-10x reduction**. The economic center of gravity is shifting from training to inference: hyperscalers now evaluate accelerators on cost/million tokens and revenue/watt, not peak FLOPS. The Jevons Paradox applies: as per-unit costs fall, total usage rises proportionally: overall spending often remains flat or grows.

**Disaggregated inference alliances:** Amazon and Cerebras formed a "disaggregated inference" partnership (March 2026): AWS Trainium handles prefill, Cerebras WSE-3 handles decode, achieving 5x token throughput versus H100. This is a direct challenge to NVIDIA's memory monopoly. NVIDIA responded with **Dynamo**, a disaggregated serving framework that separates prefill and decode onto optimal hardware. Disaggregated serving is now a production-standard approach, not experimental (see SGLang and Alternatives for benchmarks).
