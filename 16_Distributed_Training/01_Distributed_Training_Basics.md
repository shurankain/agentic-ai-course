# Distributed Training Basics

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Distributed Training
**Previous:** [[../15_GPU_Architecture/05_Quantization_Deep_Dive|Quantization: Deep Dive]]
**Next:** [[02_PyTorch_FSDP2|PyTorch FSDP2: Fully Sharded Data Parallel]]

---

## Why a Single GPU Is Not Enough

Models grow exponentially. GPT-3 with 175 billion parameters requires 500+ GB of memory for training (weights, gradients, optimizer states). H100 has 80 GB.

Training time on a single GPU is unacceptable. GPT-3 would require 355 years on a V100. Distribution across thousands of GPUs reduces this to weeks.

Fast training on many GPUs is more cost-effective than slow training on one.

## Types of Parallelism

### Data Parallelism

Each GPU receives a full copy of the model and different data. After backward, gradient synchronization via AllReduce, then simultaneous weight update.

**Advantages:** simplicity, linear throughput scaling, minimal code changes.

**Limitations:** the model must fit on a single GPU, overhead from gradient synchronization.

**Use case:** models up to 7-10B parameters in FP16, fast interconnect available.

**Ring AllReduce** organizes GPUs in a ring. In two phases (Scatter-Reduce, AllGather) each GPU receives aggregated gradients. Approximately 2x the data size is transferred regardless of the number of GPUs — bandwidth optimal. Modern libraries use a hybrid: tree within a node (NVLink), ring between nodes (InfiniBand).

### Model Parallelism (Tensor Parallelism)

Splitting individual operations across GPUs. A Linear layer with a 4096x16384 matrix is split into 4 parts, each GPU stores 4096x4096. Forward: broadcast input, multiply locally, AllGather to assemble.

For Transformers, this splits naturally: attention by heads, FFN by hidden dimension.

**Requirements:** tight NVLink coupling at 600+ GB/s. Communication on every layer.

**Use case:** model does not fit even with checkpointing, NVLink intra-node available, TP=2-8 within a node.

### Pipeline Parallelism

Splitting the model by layers. GPU 0 layers 1-10, GPU 1 layers 11-20. The naive approach is inefficient — GPUs idle while a batch passes through the pipeline.

**Micro-batching** solves the problem: the batch is split into small micro-batches that are pushed through the pipeline.

**Pipeline bubble** — idle time at the beginning and end. Bubble fraction: ~(N-1)/micro-batches. With 4 stages and 16 micro-batches: 19% overhead.

**1F1B schedule** alternates forward and backward, reducing memory footprint.

**Use case:** model is too large for tensor parallelism, slow interconnect, large batch with many micro-batches.

### Expert Parallelism

For Mixture of Experts architectures. The router determines which K experts to activate for each token. Experts are placed on different GPUs, tokens are routed via AllToAll.

**Load balancing** is critical. Auxiliary loss, capacity constraints, and expert choice routing prevent token concentration on a single expert.

**Sparse communication:** unlike dense parallelism, only activated tokens are transferred.

**Use case:** trillions of parameters with limited compute, constant compute per token for inference.

### Sequence Parallelism

For sequences of 100K+ tokens. Activations of a single layer exceed GPU memory. Splitting along the sequence dimension.

**Ring Attention** computes attention in blocks via ring communication. Query attends to local Key/Value, then Key/Value are passed along the ring.

**Combination with Tensor Parallelism:** two-dimensional splitting — sequence by length, heads by count.

### Context Parallelism

Context Parallelism (CP) is now the standard approach for long-context LLM training (100K-10M tokens), having evolved from Sequence Parallelism.

**DeepSpeed Ulysses (2024):** Partitions the sequence across GPUs along the sequence dimension. Each GPU holds a portion of Q, K, V. Before attention computation, AllToAll communication redistributes data so each GPU has full Q, K, V for a subset of attention heads. After attention, another AllToAll restores the sequence partition.

**Key advantage over Ring Attention:** Ulysses uses AllToAll (which is bandwidth-optimal) instead of point-to-point ring communication. On modern interconnects (NVLink 4.0+, NVLink 5.0), AllToAll is significantly faster. Result: 2-4x speedup over Ring Attention for equivalent sequence lengths.

**Llama 4 training context parallelism:** Meta used context parallelism (CP=8-16) for training Llama 4 Scout's 10M token context. Combined with tensor parallelism and data parallelism: TP=8 x CP=16 x DP=N. This configuration was critical — without CP, the activation memory for 10M-token sequences would be prohibitive even with Flash Attention.

**When to use CP:**
- Sequences > 64K tokens during training
- When activation memory exceeds GPU capacity even with gradient checkpointing
- NVLink or high-bandwidth interconnect available (CP is communication-intensive)
- Combine with TP: TP handles model dimensions, CP handles sequence dimensions

## Communication Primitives

**AllReduce** — each has a tensor, after the operation each has the reduction of all. Gradient synchronization in DDP.

**AllGather** — each has a part, after the operation each has everything. Output assembly in tensor parallelism.

**ReduceScatter** — reduces and distributes in parts. FSDP uses this instead of AllReduce.

**Broadcast** — one has it, all receive it. Distribution of initial weights.

**AllToAll** — all-to-all exchange. Expert parallelism for token routing.

### Ring AllReduce Efficiency

Naive AllReduce: each GPU transfers (N-1)xD data. For 8 GPUs and 10GB: 70GB per GPU.

Ring AllReduce: GPUs in a ring, two phases over 2(N-1) steps. Each GPU transfers ~2D data regardless of N. For 8 GPUs and 10GB: 20GB per GPU — 3.5x improvement.

### Gradient Compression

FP32→FP16: 2x compression, minimal quality loss.
FP32→INT8: 4x compression, requires scaling.
Top-K sparsification: transfer K% of gradients with maximum magnitude, error feedback accumulates the rest.
PowerSGD: low-rank approximation, 10-50x compression for large matrices.

### Communication-Computation Overlap

**Bucketed Gradient AllReduce:** DDP groups gradients into buckets (~25MB). As soon as a bucket is ready, asynchronous AllReduce runs while backward continues. Overlap hides 50-90% of communication.

**Prefetching in Pipeline:** loading weights for the next micro-batch in parallel with current computations.

Without overlap: 1200ms (1000ms compute + 200ms comm). With overlap: 1000ms. Speedup 20%.

## Bandwidth and Latency

### Connectivity Hierarchy

| Interconnect | Bandwidth | Generation | Notes |
|-------------|-----------|------------|-------|
| HBM3e | 8 TB/s | Blackwell B200 | Baseline for on-chip |
| HBM3 | 3.35 TB/s | H100/MI300X | Previous gen |
| NVLink 5.0 | 1.8 TB/s | Blackwell B200 | 2x over NVLink 4.0 |
| NVLink 4.0 | 900 GB/s | H100/H200 | Per GPU, bidirectional |
| NVLink-C2C | 900 GB/s | Grace Hopper GH200 | CPU-GPU coherent link |
| InfiniBand NDR | 50 GB/s | Current datacenter | 400 Gbps per port |
| RoCE/Ethernet | 12-50 GB/s | Cloud standard | Cheaper, higher latency |

**NVLink 5.0 (Blackwell, 2025):** Doubles bandwidth to 1.8 TB/s per GPU. The GB200 NVL72 rack connects 72 Blackwell GPUs via NVLink 5.0 into a single domain with 130 TB/s aggregate bisection bandwidth. This enables tensor parallelism across the entire rack — not just within a single node — fundamentally changing the TP/PP/DP decomposition strategy.

**Strategy:** Tensor parallelism on NVLink intra-node (or intra-rack on NVL72). Pipeline and Data parallelism on InfiniBand/Ethernet inter-node.

### Communication-to-Computation Ratio

70B model, gradients 140GB FP16. InfiniBand 50 GB/s: communication ~5.6 seconds. If forward+backward is 10 seconds: overhead 36%, efficiency 64%. If 3 seconds: overhead 65%, efficiency 35%.

**Improvement:** increase batch, overlap, gradient compression, accumulation, faster interconnect, FSDP instead of DDP.

**Rule:** for >80% efficiency, communication must take <20% of time.

### Gradient Accumulation

Accumulating gradients across N micro-batches, then a single synchronization. Communication reduced Nx.

Effective batch = micro_batch x accumulation x num_gpus. Equivalent to a large batch. Scale LR accordingly.

## Strategy Selection

### Decision Framework

**Model size vs memory:**
- 7B parameters: ~84GB. A100/H100 80GB is sufficient → DDP.
- 70B parameters: ~840GB. Model Parallelism required (FSDP/Tensor/Pipeline).

**Topology:**
- Single node 8 GPU NVLink: TP=8 or TP=2-4 + DP.
- Multiple nodes InfiniBand: TP intra-node, DP/PP inter-node. TP=8 x DP=16 = 128 GPUs.
- Cloud Ethernet: FSDP or Pipeline.

**Batch size:**
- Training: large batches, aggressive accumulation, pipeline bubble 10-20% acceptable.
- Inference: latency is critical, Tensor Parallelism preferred over Pipeline.

**Architecture:**
- Transformer: suitable for Tensor and Pipeline.
- MoE: Expert Parallelism + TP/PP for dense layers.
- Long sequences 100K+: Sequence Parallelism.

### Typical Configurations

**1-10B parameters:** Pure DDP on 8-64 GPUs. Alternative: FSDP if memory is tight.

**10-70B parameters:** FSDP (ZeRO-3) or TP=4-8 intra-node + DP inter-node. Example: TP=8 x DP=64 = 512 GPUs.

**100B-1T parameters:** 3D Parallelism. TP=8 x PP=8-16 x DP=16+. Example: TP=8 x PP=16 x DP=16 = 2048 GPUs for a 500B model.

**MoE trillions of parameters:** Expert Parallelism + TP for dense. 64 experts on 64 GPUs + TP=4 = 256 GPUs.

**Long context 64K-1M tokens:** SP=4 x TP=8 x DP=8 = 256 GPUs.

### Anti-patterns

**Tensor Parallelism over slow interconnect:** TP=8 across different nodes via Ethernet. AllGather/AllReduce on every layer, latency dominates, GPUs wait 80% of the time.

**Small batch with large DP:** Global batch 64 on 128 GPUs = 0.5 samples/GPU. Communication overhead dominates, efficiency <20%.

**Incorrect topology groups:** TP group contains GPUs from different nodes. Communication over slow inter-node instead of NVLink.

**Pipeline with few micro-batches:** 16 stages, 4 micro-batches. Bubble = 15/4 = 375% overhead.

**Gradient accumulation when batch fits:** Adds overhead without benefit. Use maximum batch without accumulation.

## Key Takeaways

Distributed training is a necessity for modern AI.

**Data Parallelism — always start here.** DDP is simple, efficient up to ~10B parameters. 80-95% efficiency with good interconnect.

**Model Parallelism only when necessary.** Tensor intra-node on NVLink, Pipeline inter-node, FSDP — middle ground.

**Communication is the bottleneck.** Minimize volume and hide latency behind computation.

**Topology determines strategy.** Match parallelism to physical topology. Tensor on Ethernet — anti-pattern.

**Context Parallelism for long sequences.** DeepSpeed Ulysses and CP enable training with 100K-10M token contexts. AllToAll-based CP is faster than Ring Attention on modern interconnects.

**NVLink 5.0 changes the topology calculus.** With 1.8 TB/s per GPU and NVL72 rack-level NVLink domains, tensor parallelism extends beyond single nodes.

**Start simple, scale complexity.** DDP → FSDP → Tensor → Pipeline → 3D → +CP. Each step adds complexity.

**Measure, don't guess.** Profile communication overhead, scaling efficiency, GPU utilization.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Distributed Training
**Previous:** [[../15_GPU_Architecture/05_Quantization_Deep_Dive|Quantization: Deep Dive]]
**Next:** [[02_PyTorch_FSDP2|PyTorch FSDP2: Fully Sharded Data Parallel]]
