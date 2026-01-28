# 3D Parallelism: Scaling to Trillions

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Distributed Training
**Previous:** [[03_DeepSpeed_ZeRO|DeepSpeed ZeRO: Zero Redundancy Optimizer]]
**Next:** [[05_Distributed_Training_Practice|Practice: Distributed Training]]

---

## When One Type Is Not Enough

Data parallelism scales the batch, but the model must fit on a single GPU. Tensor parallelism creates intensive communication. Pipeline has bubble overhead.

For hundreds of billions and trillions of parameters, no single approach works in isolation. The solution: combining different types — "3D parallelism" or "hybrid parallelism".

Megatron-Turing NLG (530B) from Microsoft and NVIDIA was the first model with full 3D. GPT-4, Claude, LLaMA 70B use variations.

## Components of 3D Parallelism

### Tensor Parallelism (TP)

Splitting operations across GPUs.

**Transformer:**
- Attention: Q, K, V by dimension heads, each GPU a subset of heads
- FFN: first linear column parallelism (by output dim), second row parallelism (by input dim)
- AllReduce after each

Megatron-LM: 2 AllReduce per transformer layer.

**Requirements:** NVLink high bandwidth, TP degree 2-8 within a node.

### Pipeline Parallelism (PP)

Splitting by layers. GPU 0 layers 1-10, GPU 1 layers 11-20.

**Schedules:**
- GPipe: forward all, then backward all. Large bubble, high memory.
- 1F1B: interleave forward and backward. Less memory, same bubbles.
- Interleaved 1F1B: multiple non-contiguous chunks per GPU. Smaller bubble, more communication.
- Zero-bubble: advanced scheduling, minimal idle. Requires careful orchestration.

**Requirements:** moderate bandwidth, PP degree determined by model depth and target bubble %.

### Data Parallelism (DP)

Model replication, data splitting. With ZeRO — sharded data parallelism (FSDP).

In combination: DP group handles different parts of the batch, AllReduce for gradients, with ZeRO-3 ReduceScatter + AllGather.

**Requirements:** works even with low bandwidth inter-node, DP degree scales throughput linearly.

## Combining: Megatron-DeepSpeed

### Architecture

Megatron-LM (tensor parallelism) + DeepSpeed ZeRO (data parallelism + memory optimization).

Typical configuration for a 100B+ model on 512 GPUs:
- TP = 8 (intra-node, NVLink)
- PP = 8 (8 pipeline stages)
- DP = 8 (8 data parallel replicas)
- Total: 8 × 8 × 8 = 512 GPU

**Process groups:** 512 processes organized into three types of groups (DP, TP, PP). Each GPU belongs to one DP, one TP, one PP group.

### Communication patterns

**Tensor parallelism (intra-node):** AllReduce after attention and FFN. ~2× data_per_layer. Requires NVLink.

**Pipeline parallelism:** Point-to-point send/recv activations between stages. ~batch × hidden_size per micro-batch. Moderate bandwidth.

**Data parallelism (inter-node):** AllReduce/ReduceScatter gradients once per iteration. Tolerant to lower bandwidth.

### Configuration Example

GPT-3 scale (175B parameters):
- Architecture: 96 layers, hidden 12288, 96 heads
- Cluster: 512 GPU A100-80GB (64 nodes × 8 GPU)

**Parallelism:**
- TP = 8, PP = 8, DP = 8

**Memory per GPU:**
- Parameters: 175B / 64 = 2.7B → ~5.4 GB FP16
- Optimizer states (ZeRO-1): ~10 GB
- Activations: 20-30 GB
- Total: 40-60 GB — fits within 80 GB

## Practical Example: 100B+ Model

### Hardware planning

A 100B model requires a minimum of 64-128 GPU A100-80GB/H100. 256+ recommended for reasonable throughput.

**Topology:** 64 nodes × 8 GPU = 512 GPU. Intra-node NVLink 600 GB/s for TP. Inter-node InfiniBand 200-400 Gb/s for PP and DP.

### Choosing Parallelism

**Step 1: TP degree** = GPUs per node = 8 (use all NVLink).

**Step 2: PP degree** = 8. 96 layers / 8 stages = 12 layers per stage. 100B / 8 stages ≈ 12.5B per stage × 2 bytes = 25 GB — fits.

**Step 3: DP degree** = Total GPU / (TP × PP) = 512 / 64 = 8.

**Step 4: Verify memory:**
- Parameters: 100B / 64 = 1.56B × 2 = 3.1 GB
- Gradients: 3.1 GB
- Optimizer (ZeRO-1): 12 × 1.56B / 8 = 2.3 GB
- Activations: ~20 GB
- Total: ~30 GB — fits within 80 GB

### Training configuration

**Megatron-DeepSpeed Parallelism:**
- TENSOR_PARALLEL = 8
- PIPELINE_PARALLEL = 8
- DATA_PARALLEL = 8 (auto)

**Model:**
- NUM_LAYERS = 96
- HIDDEN_SIZE = 12288
- NUM_HEADS = 96
- SEQ_LENGTH = 2048

**Training:**
- GLOBAL_BATCH_SIZE = 2048
- MICRO_BATCH_SIZE = 1
- GRADIENT_ACCUMULATION = GLOBAL / (MICRO × DP)

**DeepSpeed:**
- ZERO_STAGE = 1

## Pipeline Schedules

**GPipe:** all forward, then all backward. Bubble = (p-1)/m. Stores activations of all m micro-batches.

**1F1B:** interleaving. Same bubble, stores only p micro-batches — 4-8× memory savings.

**Interleaved 1F1B:** multiple chunks per GPU. Bubble = (p-1)/(m × v). More communication.

**Zero-Bubble:** ZB-H1 splits backward into B (activations) and W (weights), filling bubbles with W computation. ~1% bubble.

## Memory Optimization

**GPU Memory:**
- Parameters: Total/(TP × PP) × bytes
- Gradients: equal to parameters
- Optimizer: 12 bytes/param (Adam), divided across DP with ZeRO
- Activations: dominate with large batch/sequence

**Activation Checkpointing:** recomputation during backward instead of storing. 4-8× savings, overhead +33% compute.

**Selective checkpointing:** store what is cheap to compute but expensive for memory (QKV, FFN inputs). Do not store attention scores.

**CPU Offloading:** offloading activations to CPU RAM. Effective when PCIe bandwidth > compute time.

## Communication-Computation Overlap

**Three levels:**
1. DP gradient AllReduce: launch for layer N while computing gradients for layer N-1
2. PP activation transfer: micro-batches mask transfer waiting time
3. TP AllReduce: overlap with the start of the next layer

**Bucket AllReduce:** grouping gradients into buckets (~25-50 MB). Bucket full — async AllReduce, backward continues.

**Quantized Communication:** INT8 quantization 4× reduction, slight precision loss. Mixed-precision: compute FP32, communicate BF16.

## Multi-node Orchestration

**Launch:** torchrun with --nnodes, --nproc_per_node. SLURM integration via srun.

**Fault tolerance:** checkpointing every N iterations, elastic training (continue with fewer GPUs), checkpoint replication.

## Key Takeaways

3D Parallelism is a necessity for frontier models.

**Combine strengths.** TP for high-bandwidth, PP for layer distribution, DP for scaling throughput.

**Hardware topology determines strategy.** TP groups on NVLink, PP/DP between nodes.

**Communication is the limiting factor.** Profile and optimize collective operations.

**Pipeline schedule is a trade-off.** GPipe is simple but memory-hungry, 1F1B balances, Interleaved minimizes bubble at the cost of communication, Zero-Bubble reaches the limit.

**Bubble overhead decreases with micro-batches.** Formula (p-1)/m: more micro-batches = less idle time.

**Activation checkpointing is mandatory.** Without it, activations will consume GPU memory. Trade-off: +33% compute for 4-8× savings.

**Selective checkpointing is smarter.** Store what is expensive for memory, cheap to compute.

**Communication-computation overlap is critical.** Bucket AllReduce, prefetching, async operations hide network latency.

**Complexity is significant.** 3D parallelism requires careful configuration, debugging, monitoring.

**Tools help.** Megatron-LM, DeepSpeed, FSDP2 make 3D parallelism more accessible.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Distributed Training
**Previous:** [[03_DeepSpeed_ZeRO|DeepSpeed ZeRO: Zero Redundancy Optimizer]]
**Next:** [[05_Distributed_Training_Practice|Practice: Distributed Training]]
