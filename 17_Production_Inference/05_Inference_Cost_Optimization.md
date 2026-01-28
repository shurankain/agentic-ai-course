# Cost Optimization Strategies

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[04_Model_Quantization|Quantization: Memory vs Quality]]
**Next:** [[06_Long_Context_Inference|Long Context Inference]]

---

## LLM Inference Economics

Inference requires significant expenditure. A single GPT-4 request with long context costs several cents. At millions of requests per day, costs reach hundreds of thousands of dollars per month.

**Cost Structure:**

GPU compute is the primary expense. Cloud service prices:
- A100 40GB: $2.50-4.00/hour
- A100 80GB: $3.50-5.00/hour
- H100: $4.50-8.00/hour
- T4: $0.50-1.00/hour
- L4: $0.70-1.20/hour

Llama 70B on A100-80GB: ~$0.001-0.002 per 1000 tokens. At 100M tokens per day: $3000-6000 per month (compute only).

Memory limits model size and batch size. Networking in multi-GPU adds 10-20%. Storage negligible.

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

Checkpointing for batch — saving every N batches, resuming from checkpoint. Overhead ~1-2%, savings 70%.

Mixed fleet: 2-3 on-demand baseline + 5-10 spot burst. On spot preemption, on-demand absorbs load.

Multi-region/multi-zone deployment — diversification reduces risk.

Fallback on spot loss — switching to smaller model or queue.

## Cascading Models

**Concept:**
Not all requests require a large model. Simple questions — small model (fractions of a cent). Complex ones — large model.

Router directs by complexity:
- Tier 1 (7B): simple factual questions
- Tier 2 (13-30B): general-purpose, moderate complexity
- Tier 3 (70B+): complex reasoning, code generation

**Routing strategies:**

Rule-based: heuristics by length, keywords, patterns. Zero latency, deterministic, but not always accurate.

Model-based: lightweight classifier predicts complexity. More accurate (85-90% vs 70-75%), but adds latency ~10-20ms.

Confidence-based: attempt small model, fallback to large if confidence is low (via logprobs). Adaptive, but two inference calls for some.

**Cost savings — real numbers:**

Traffic: 60% simple (7B), 30% medium (13B), 10% complex (70B).

Without cascading: 100K requests/day × $0.01 = $1000/day = $30K/month.

With cascading: 60K × $0.001 + 30K × $0.003 + 10K × $0.01 = $250/day = $7.5K/month.

Savings: 75%.

Trade-offs: complexity, potential quality degradation, latency overhead ~10-50ms.

**Optimization:**
Monitor routing accuracy (if >20% negative feedback, router is too aggressive). A/B testing quality. Dynamic thresholds based on load.

## Right-Sizing Infrastructure

**GPU selection:**

Do not overpay. Match GPU to workload.

Llama 7B: T4 (20-30 tok/s, $0.50/hr) or L4 (40-60 tok/s, $1.00/hr) perfectly adequate. A100 ($3.50/hr) overkill.

Llama 70B: A100-40GB will not fit. A100-80GB 1 GPU (8-12 tok/s, $5/hr) with quantization. H100 1 GPU (15-20 tok/s, $7/hr) better price/performance.

Anti-patterns: 7B on H100 (10× overpaying), A100-80GB instead of A100-40GB for models <40GB, multi-GPU without tensor parallelism.

**Instance right-sizing:**

CPU bottlenecks — preprocessing can be limiting. If GPU utilization <60% at full load, the problem is CPU. More vCPUs or optimize preprocessing.

Memory requirements — insufficient RAM leads to swapping. Rule: RAM >= 2x model size. For Llama 70B FP16: 140GB model + 50GB overhead = 190GB+ RAM.

Network bandwidth — multi-GPU requires high bandwidth. Monitor utilization, if >80% network-bound.

Monitoring: GPU utilization target >70%. CPU 50-70%. Memory <80%.

**Batch size optimization:**

Larger batch → higher throughput, but higher latency. The trade-off depends on the use case.

Llama 13B on A100: batch 1 (50 tok/s, 20ms), batch 4 (180 tok/s, 45ms), batch 16 (500 tok/s, 150ms), batch 64 (800 tok/s, 600ms).

Real-time chat: batch 1-4. Document processing: batch 16-64. Async API: batch 8-16.

Dynamic batching — collect up to max_batch_size or timeout (50ms). Low traffic — smaller batches (low latency), high traffic — larger (high throughput).

Cost impact: batch 32 can be 3-4× cheaper per token than batch 1.

## Autoscaling

**Metrics-based (reactive):**

GPU utilization: scale up >80% for 3-5 min, scale down <40% for 10-15 min. Asymmetric timings — fast up, slow down.

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

Performance: GPU utilization <50% for 30+ min, latency >2× normal, error rate >5%.

Budget: 70% daily budget warning, 90% critical, 100% rate limiting or fallback.

**Budget controls:**

Hard limits: max requests per period, max tokens per request, spend caps per customer, total budget cap.

Soft limits: warning at 70%, degraded service at 85%, queue prioritization.

Graceful degradation: Normal → 70% warnings → 85% aggressive cascading → 95% queue/rate limiting → 100% free tier disabled.

## Key Takeaways

Spot instances — low-hanging fruit. 60-90% savings for eligible workloads. Combine with on-demand for resilience.

Cascading architecture — biggest reducer. With 70% simple requests, savings reach 75%.

Right-size aggressively. Don't run 7B on H100. Monitor utilization — target >70% GPU, 50-70% CPU.

Autoscaling essential for variable traffic. Reactive + predictive for best results.

Monitor everything relentlessly. Track cost per token, utilization, throughput per dollar. Set alerts.

Implement budget controls. Hard limits prevent surprises. Soft limits balance cost and quality.

Think about trade-offs: spot (cost vs availability), cascading (cost vs quality), large batches (cost vs latency), quantization (cost vs accuracy).

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[04_Model_Quantization|Quantization: Memory vs Quality]]
**Next:** [[06_Long_Context_Inference|Long Context Inference]]
