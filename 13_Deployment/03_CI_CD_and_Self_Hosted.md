# CI/CD and Self-Hosted Models

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Deployment
**Previous:** [[02_Containerization_and_Kubernetes|Containerization and Kubernetes]]
**Next:** [[../14_Security_Safety/01_Prompt_Injection|Prompt Injection]]

---

## Continuous Integration for LLM Applications

Continuous Integration for LLM projects introduces additional challenges. Beyond standard tests, it is necessary to test interactions with LLM APIs, prompt quality, and behavior under various model responses. These tests are more expensive in the literal sense — each call costs money.

The strategy balances coverage and cost. Unit tests with mocked LLM responses are fast and free but do not catch changes in real model behavior. Integration tests with real APIs are more accurate but expensive and slower. A sensible approach: extensive unit tests with mocks for processing logic, a limited set of integration tests for critical paths.

Prompt versioning deserves special attention. A prompt is an artifact like code and should be versioned accordingly. Changing a prompt radically alters system behavior; this change must be trackable, reviewable, and reversible.

## CI Pipeline Structure

**First stage** — standard build and static analysis: compilation, linting, formatting checks, dependency security scanning.

**Second stage** — unit tests. Executed with mocked LLM clients, no API keys required. Tests cover request and response processing logic, prompt construction, result parsing, and error handling. High coverage is critically important.

**Third stage** — integration tests with real APIs. A small set of smoke tests verifies basic integration functionality. API keys are provided through CI system secrets. To save costs, use cheaper models (GPT-4o-mini instead of GPT-4o).

**Fourth stage** — building and publishing the Docker image. Tagged with commit SHA for unique identification. The latest tag is updated only for the main branch. Parallel execution of independent stages accelerates the pipeline. Integration tests wait for unit test success.

**Typical workflow structure:** The build job compiles code and performs static analysis via dependencyCheck, then uploads artifacts. Unit tests run in parallel, executing tests with mocks, collecting coverage, and uploading to Codecov. Integration tests depend on unit tests (needs: unit-tests), spin up dependencies via docker-compose (PostgreSQL, Redis), run tests with real API keys from Secrets, and clean up resources through a cleanup step with if: always(). The Docker job builds the image using multi-stage build; metadata-action generates tags: commit SHA, branch name, latest for main. BuildKit cache accelerates builds through layer caching. Deploy jobs are separated by environment: staging from develop, production from main. Environments are used for secrets and protection rules. Kubectl commands update the Deployment via set image, rollout status waits for successful deployment, and smoke tests verify functionality.

## Continuous Deployment and Release Strategies

Continuous Deployment automates rollout after checks pass. For LLM applications, it requires safeguards due to the potential impact of prompt changes.

**Canary deployment** is the gold standard. The new version receives 1-5% of traffic. Metrics are compared against the stable version: latency, error rate, cost, and quality. On anomalies, the canary is rolled back. Practical implementation requires two Deployments: llm-agent-canary and llm-agent (stable). A Service or Ingress distributes traffic through weight-based routing. Prometheus collects metrics from both, labeling them deployment=canary or deployment=stable. A validation script periodically queries the Prometheus API, comparing metrics over a 5-minute window.

Critical metrics: error rate (percentage of 5xx), P95/P99 latency, LLM call cost, and throughput. If the canary shows an error rate above 5%, latency exceeds the threshold, or cost is anomalously high — rollback. Observation period is 10-15 minutes.

**Blue-green deployment** provides instant switchover. Both versions run in parallel, and traffic switches atomically. Rollback is an instant switch back. The downside is doubled resource usage.

**Feature flags** separate code deployment from feature activation. A new prompt or model is deployed in a disabled state, then gradually enabled for users or traffic percentages. Useful for testing new models — metrics on quality can be collected before rollout.

Implementation requires a flag management system (LaunchDarkly, Unleash, or custom-built). A flag defines the prompt, model, and parameters. Flags are stored in a distributed cache (Redis) for fast access. Changing a flag takes effect immediately without redeployment.

**Rolling update** is the simplest approach; pods are updated gradually. This is the Kubernetes default. Suitable for non-critical changes but does not provide fast rollback when quality issues arise.

## Smoke Tests and Quality Gates

Smoke tests verify basic functionality: the endpoint responds, the LLM provider is accessible, basic generation works, and formatting is correct. They execute predefined requests and check expected results. For example, "Translate 'hello' to French" should contain "bonjour". Checks include HTTP status code (200), required fields in the response, reasonable response time (< 30 seconds), and absence of error messages.

Quality gates define criteria for a successful deployment. Quantitative gates: error rate < 1%, P95 latency < 10s, cost per request within budget. Qualitative gates: automatic quality evaluation requires deterministic checks or LLM-as-Judge.

Deterministic checks use test cases with precisely known answers. Math problems, JSON formatting, information extraction — verified programmatically. The regression suite accumulates over time; each bug becomes a test case.

LLM-as-Judge uses another (more powerful) model to evaluate quality. GPT-4 evaluates responses by criteria: relevance, coherence, correctness, helpfulness. Scoring is 1-5, aggregated across dozens of cases. If the average score < 4.0, the deployment fails.

Rollback criteria are defined in advance. Automatic rollback on critical errors. Rollback on quality degradation requires fast evaluation — pre-prepared test cases with reference answers.

Post-deployment monitoring tracks behavior for 30-60 minutes after deployment. If no issues arise, the deployment is successful. Grafana dashboards show metrics in real time with deployment event annotations for correlation. Alerts are configured for critical deviations: spikes in error rate, increased latency, throughput drops. PagerDuty notifies the on-call engineer.

## Self-Hosted Models: Why and When

A self-hosted LLM means running models on your own infrastructure instead of cloud APIs. It introduces significant complexity but is justified in certain scenarios.

**Privacy and compliance** are the main drivers. Some data cannot be sent to external providers: medical records, financial information, military secrets. A self-hosted model processes data within the organization's perimeter. Regulations in healthcare (HIPAA), finance (PCI DSS, SOX), and the public sector require strict control. A self-hosted model on on-premises or private cloud satisfies the most stringent requirements.

**Latency and throughput** for high-load systems. At very high volumes, self-hosted is more economical than cloud APIs. Network latency to the provider is eliminated.

**Economics:** OpenAI API costs $0.15/1M input tokens for GPT-4o-mini, $2.50/1M for GPT-4o. At a million requests per day with an average of 500 tokens — hundreds to thousands of dollars daily. A self-hosted Llama-70B on a dedicated GPU server ($5000/month for an A100) pays for itself at a certain volume. The crossover point is typically around 100M tokens/day.

**Customization and fine-tuning.** Self-hosted infrastructure allows running custom fine-tuned models that are significantly better than base models for specific tasks. A fine-tuned model for medical diagnostics, legal analysis, or technical support can outperform frontier models on domain-specific tasks at a smaller size. OpenAI now offers fine-tuning for GPT-4o, but with limited control over the process. Self-hosted Llama, Mistral, and Qwen can be fully fine-tuned with complete control over hyperparameters, data, and training process.

**Control over availability.** Dependence on an external provider is a risk. An OpenAI outage means an outage of your service. Historical examples: an OpenAI outage in November 2023 lasting several hours, Anthropic rate limiting during peak loads. For critical systems (emergency response, financial trading, production control), such dependence is unacceptable. A self-hosted model with proper redundancy guarantees availability under your control.

## vLLM: High-Performance Inference

vLLM is one of the most efficient solutions for inference. It implements PagedAttention — a memory management technique that significantly increases throughput. Traditional implementations allocate memory for the maximum context length per request, even if the actual length is shorter. PagedAttention allocates memory dynamically, similar to how an OS manages memory pages. This allows serving more concurrent requests on the same GPU.

The mechanism: the KV-cache is split into fixed-size blocks (typically 16 tokens). Blocks are allocated as needed. When a request completes, blocks are freed and reused. This eliminates fragmentation and enables near-optimal memory utilization.

vLLM provides an OpenAI-compatible API, simplifying migration. If your application uses the openai-python client, it is sufficient to change the base_url to the vLLM server address. The rest of the code remains unchanged. Streaming, function calling, and response format are supported. This allows A/B testing self-hosted vs. cloud without code changes.

Continuous batching combines requests for efficient GPU utilization. Unlike static batching, where a batch is formed in advance and waits to fill, continuous batching adds new requests as previous ones complete. Static batching waits for 32 requests, then processes them together. If only 10 arrive — it either waits for a timeout (increasing latency) or processes an incomplete batch (losing throughput). Continuous batching processes 10 immediately, adding the 11th as soon as it arrives. It dynamically optimizes the latency-throughput trade-off.

Tensor Parallelism distributes a single model across multiple GPUs. For models that do not fit in a single GPU's memory (70B+ parameters), this is necessary. Configured via the --tensor-parallel-size parameter. It slices each layer along the dimension, placing parts on different GPUs. The forward pass requires all-reduce communication over NVLink or PCIe. This introduces overhead but allows running otherwise impossible models. Llama-70B requires ~140GB in FP16 and does not fit in an A100-80GB. With tensor-parallel-size=2, each GPU stores half, ~70GB.

## Infrastructure for GPU Workloads

GPU inference requires specialized infrastructure. NVIDIA GPUs with sufficient VRAM are the primary resource. For 7B parameters, 16GB is sufficient (RTX 4080, A4000). For 70B, 80+ GB is needed (A100-80GB or multiple GPUs).

Estimating requirements: a model in FP16 requires 2 bytes per parameter. 7B parameters = 14GB, plus KV-cache and activations — round up to 20GB. Quantized INT8 — 1 byte per parameter, 7B = 7GB. INT4 (GPTQ/AWQ) — 0.5 bytes, 7B = 3.5GB. This allows selecting GPUs to match the budget: an RTX 4090 24GB can run Llama-13B in INT4, an A100 80GB — Llama-70B in INT8.

In Kubernetes, GPU nodes require the NVIDIA device plugin and node labels. Tolerations and node selectors direct pods to GPU nodes. Resource requests for nvidia.com/gpu reserve a GPU for the pod.

The NVIDIA Device Plugin is installed as a DaemonSet and exports GPUs as a Kubernetes resource. The pod spec includes resources.requests and resources.limits with nvidia.com/gpu: 1, reserving one GPU exclusively. The node selector nvidia.com/gpu.product can require a specific model (A100, H100). Tolerations allow placement on GPU nodes with a NoSchedule taint.

Shared memory (shm) is critical for inference. PyTorch and vLLM use shared memory for inter-process communication. The standard /dev/shm size in Docker (64MB) is insufficient. A volume of type emptyDir with medium: Memory is required.

PyTorch DataLoader with num_workers > 0 uses shared memory. vLLM uses it for communication between Python and CUDA kernels. Exhausting shm leads to "OSError: No space left on device." The solution: an emptyDir volume with sizeLimit 16Gi, mounted to /dev/shm.

Model caching saves startup time. Models are downloaded from HuggingFace Hub on first launch, which takes minutes. A PersistentVolume for cache allows reusing weights between restarts. Llama-70B weighs ~140GB; downloading over the network takes 10-20 minutes. On every restart or rolling update, this is downtime. A PersistentVolumeClaim on fast SSD (100-200GB) mounted to ~/.cache/huggingface means the model is downloaded once, and subsequent starts use cache — pod ready in 2-3 minutes instead of 20.

Memory utilization in vLLM is controlled by --gpu-memory-utilization. A value of 0.9 means 90% of available GPU memory is allocated to KV-cache. The remaining 10% serves as a buffer. Too high (0.95-0.99) leads to OOM during peak loads. Too low (0.7-0.8) underutilizes the GPU, reducing throughput. The optimum depends on the workload: short requests tolerate high utilization, while long contexts require headroom.

## Inference Optimization

Quantization reduces model size and memory requirements at the cost of a small quality decrease. INT8 quantization halves the size. INT4 (GPTQ, AWQ) reduces it by a factor of four. For many tasks, the quality loss is negligible while the resource savings are substantial.

INT8 converts FP16/FP32 weights to 8-bit integers, applying per-channel scaling factors. Quality drops by 1-2% on benchmark metrics; subjectively, it is often indistinguishable. INT4 is more aggressive, grouping weights and applying shared scales; the loss is 3-5% but acceptable for most tasks.

GPTQ and AWQ are two popular INT4 methods. GPTQ minimizes quantization error through optimal brain quantization — slower but more accurate. AWQ (Activation-aware Weight Quantization) focuses on important weights through activations — faster but slightly less accurate. In practice, both yield similar results.

KV-cache management is the key to high throughput. vLLM automatically optimizes through PagedAttention. The KV-cache grows linearly with context length and the number of concurrent requests. A model with an 8K context window processing 20 concurrent requests requires cache for 160K tokens. In FP16, this amounts to gigabytes. PagedAttention optimizes through sharing and reuse, but fundamental limits remain.

Speculative decoding accelerates generation using a draft model. A small model quickly generates several tokens, and the large model verifies them in a single forward pass. With a high acceptance rate, this yields significant speedup. The small model (draft) is typically from the same family at a smaller size: Llama-7B as draft, Llama-70B as target. The draft generates 4-8 tokens speculatively, and the target verifies in parallel. If all are correct — accept and continue. If there is an error — reject from that point and continue with the target. With an acceptance rate > 70%, this gives 2-3x speedup.

Dynamic batching parameters affect the latency-throughput trade-off. A small batch means low latency but low throughput. A large batch means high throughput but potentially high latency. vLLM manages this automatically through continuous batching, but the max-num-seqs and max-num-batched-tokens parameters influence behavior. For latency-sensitive workloads, reduce both; for throughput-oriented workloads, increase them.

## Hybrid Approach: Self-Hosted + Cloud

The optimal architecture often combines self-hosted and cloud models. Self-hosted for high-frequency operations and privacy-critical data. Cloud APIs for complex tasks requiring frontier models.

A router determines where to direct a request. Criteria include: task complexity, data sensitivity, required model, and current load. Simple classifications go to the local Llama; complex reasoning tasks go to GPT-4o or o3.

**Router logic:** A privacy flag in the metadata immediately routes to self-hosted. Complexity estimation is based on heuristics: prompt length, required capabilities (code generation, complex reasoning), and expected output length. Scoring system: prompt > 4000 chars (+2), code generation (+2), reasoning (+3), output > 2000 tokens (+1). Score 0-1 = simple (self-hosted), 2-3 = medium (self-hosted if available), 4+ = complex (cloud).

A circuit breaker monitors self-hosted health. When the error rate or latency is high, the circuit opens and all requests go to the cloud. Resilience4j provides a circuit breaker: failure rate > 50% over a 1-minute sliding window opens the circuit for 30 seconds, then a half-open state tries a few requests.

Fallback between local and cloud models improves availability. API unification is achieved through OpenAI-compatible endpoints: vLLM exports the OpenAI API, and the client works identically. Differences in capabilities are handled through a capability registry: before sending, it checks whether the target model supports the required features; otherwise, fallback or error.

Cost optimization through the hybrid approach: 80% of requests are simple tasks (classification, extraction, simple QA), served by self-hosted Llama-8B at $0.0001 per request (amortized GPU cost). 15% are medium tasks, partially self-hosted, partially GPT-4o-mini ($0.0005). 5% are complex tasks, GPT-4o ($0.01). The blended average cost drops from $0.01 (pure cloud) to $0.003 (hybrid) — 70% savings.

## Monitoring Inference Servers

GPU-specific metrics: utilization, memory usage, temperature, and power consumption. NVIDIA DCGM (Data Center GPU Manager) exports metrics to Prometheus. Grafana visualizes the GPU cluster state.

DCGM is installed on GPU nodes as a DaemonSet and exports via the DCGM exporter. Key metrics: DCGM_FI_DEV_GPU_UTIL (0-100%), DCGM_FI_DEV_FB_USED and FB_FREE (memory in MB), DCGM_FI_DEV_GPU_TEMP (°C), DCGM_FI_DEV_POWER_USAGE (Watts). The Grafana dashboard shows a utilization heatmap, memory usage timeline, and alerts on high temperature (> 85°C) or memory exhaustion.

Inference-specific metrics: tokens per second, time to first token, batch size distribution, and queue depth. vLLM exports via the /metrics endpoint. Key metrics: vllm:num_requests_running, vllm:num_requests_waiting, vllm:time_to_first_token_seconds (histogram), vllm:time_per_output_token_seconds, vllm:e2e_request_latency_seconds. Token throughput is calculated as rate(vllm:request_success_total) * average tokens per request. The dashboard shows trends, percentiles (P50/P95/P99), and correlations between queue depth and latency.

Alerting: GPU temperature > 85°C for more than 5 minutes (warning), > 90°C (critical). GPU memory utilization > 95% for more than 2 minutes. Throughput drop > 50% over 5 minutes. Queue depth > 100 requests. P95 latency increase > 2x baseline. Each alert is sent to PagerDuty with severity and a runbook link.

Capacity planning is based on historical data. Analysis: average 100 requests/minute, peak 500 requests/minute. Average processing time is 5 seconds, so peak load requires handling 500 * 5/60 = ~42 concurrent requests. Each request uses 2GB of KV-cache. 42 * 2GB = 84GB, requiring a minimum of 2x A100-80GB. Adding 50% headroom for bursts and failover = 3x A100-80GB as the minimum configuration.

## Key Takeaways

The CI pipeline for LLM applications balances coverage and cost. Extensive unit tests with mocks and limited integration tests with real APIs for critical paths. Canary deployment is the preferred strategy. Comparing the new version's metrics against the stable version allows detecting degradation before full rollout. Feature flags separate code deployment from feature activation. New prompts or models are tested on limited traffic.

Self-hosted models are justified for: privacy/compliance requirements, high-load systems, specialized fine-tuned models, and eliminating provider dependencies. vLLM with PagedAttention provides high-performance inference. The OpenAI-compatible API simplifies migration. GPU infrastructure requires: device plugins, adequate shared memory, model caching, and correct resource limits.

The hybrid approach combines the advantages of self-hosted (cost, privacy) and cloud (capability, simplicity) models. A router directs requests based on their characteristics. Quantization (INT8, INT4) reduces GPU memory requirements at the cost of minimal quality loss. It enables running larger models on smaller GPUs.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Deployment
**Previous:** [[02_Containerization_and_Kubernetes|Containerization and Kubernetes]]
**Next:** [[../14_Security_Safety/01_Prompt_Injection|Prompt Injection]]
