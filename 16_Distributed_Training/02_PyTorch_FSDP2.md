# PyTorch FSDP2: The Modern Standard

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Distributed Training
**Previous:** [[01_Distributed_Training_Basics|Distributed Training Basics]]
**Next:** [[03_DeepSpeed_ZeRO|DeepSpeed ZeRO: Zero Redundancy Optimizer]]

---

## Evolution from DDP to FSDP

DDP replicates the full model on each GPU. For a 10 GB model on 8 GPUs, 80 GB is required, even though 10 GB total would suffice.

FSDP eliminates redundancy through sharding — each GPU stores only a portion of the parameters. For a 70B model on 8 GPUs: instead of 560B parameters total, we store 70B, approximately 9B per GPU.

FSDP2 (2024) — a redesigned version with 28.54% memory reduction and 68.67% throughput improvement.

## How FSDP Works

### Sharding: Divide and Conquer

For a model with N parameters on W workers: each stores N/W parameters.

**Forward pass:**
1. AllGather collects all shards of a layer's parameters
2. Each worker performs forward on its data
3. Collected parameters are freed immediately after computation

**Backward pass:**
1. AllGather of parameters for gradient computation
2. Gradients computed locally
3. ReduceScatter of gradients — each worker receives aggregated gradients only for its portion
4. Temporary parameters are freed

**Pattern:** gather → compute → free for each layer. At peak memory: one layer's parameters in full + sharded remainder. For a 70B model on 8 GPUs with a ~3B layer: ~12B instead of 70B.

### Three Levels of Sharding

**FULL_SHARD (ZeRO-3):** Sharding of parameters, gradients, optimizer states. For a 70B model on 8 GPUs: ~22GB per GPU instead of 1.4TB total. Requires AllGather on each forward/backward.

**SHARD_GRAD_OP (ZeRO-2):** Full parameters on each GPU, gradients and optimizer states sharded. ~350GB per GPU, but forward without AllGather. Balance of memory and speed.

**NO_SHARD:** Equivalent to DDP. For debugging and comparison.

**Recommendation:** FULL_SHARD for LLMs on NVLink/InfiniBand — memory savings are critical for increasing batch size.

### Unit of sharding

FSDP splits the model into "units" — groups of parameters that are sharded and gathered together.

Too large units: minimal AllGather, but peak memory approaches full model.
Too small units: many AllGather operations with small volumes, overhead from launching operations.

**Best practice for Transformers:** one unit = one transformer block. Sufficient size for efficient AllGather (~2-4B parameters), logical grouping, predictable memory footprint.

## FSDP2: Improvements

**Simplified API:** Composable approach via the fully_shard function applied to submodules. Explicit control, composability with torch.compile, simplified debugging, less boilerplate code.

**DTensor integration:** Built on DTensor for consistent semantics, integration with tensor parallelism, inspection and debugging.

**Performance:** Optimized AllGather, aggressive prefetching of next layer's parameters, reduced CPU overhead, aggressive buffer deallocation. Result: 40-70% faster, larger batch size.

## Mixed Precision Training

Mixed precision scheme:
- FP32 weights (master weights)
- Forward/backward FP16/BF16
- Gradient accumulation and optimizer updates FP32

**Advantages:** 2x memory savings for activations, 2-8x speedup on Tensor Cores, precision preserved through FP32 updates.

**BFloat16 vs FP16:**
- FP16: 5-bit exponent, overflow/underflow, requires gradient scaling.
- BFloat16: 8-bit exponent (same as FP32), resistant to overflow, no scaling required.

BFloat16 is the de facto standard for LLMs.

## Activation Checkpointing

Activations for deep models consume more memory than parameters. An LLM with 32 layers, batch 32, sequence 2048, hidden 4096: ~50 GB activations vs 17.5 GB parameters (with FSDP on 8 GPUs).

**Checkpointing strategy:** Save only the inputs of checkpointed layers, recompute activations during backward from the last checkpoint.

Typical strategy: checkpoint every transformer layer. Instead of ~50 GB for all activations, we store ~1.5 GB of layer inputs.

**Trade-off:** +33% compute (forward twice), but allows 3-4x larger batch. Often the difference between "fits" and "OOM".

## Memory Estimation

Formula for FSDP FULL_SHARD:

Memory per GPU ≈ (Params/N_GPUs) × bytes × multiplier + activations + buffers

- Params/N_GPUs: even distribution
- bytes: 2 for BF16
- multiplier: 4-6 (params, gradients, optimizer states, overhead)
- activations: depend on batch and checkpointing
- buffers: ~5-10% of parameters

**Example 70B on 8×A100-80GB:**
- Parameters: 70B/8 × 2 × 5 = 87.5 GB
- Activations (batch 16, seq 2048, checkpoint): ~20 GB
- Buffers: ~5 GB
- Total: ~112.5 GB — does not fit

Solution: batch 8 (~10 GB activations) or gradient accumulation.

## Debugging FSDP

**Verify sharding:** ensure modules are sharded correctly.

**Monitor memory:** torch.cuda.max_memory_allocated() after warm-up, compare with theoretical calculation.

**Common issues:**
- OOM: reduce batch, enable checkpointing, verify sharding
- Slow training: NCCL backend, interconnect speed, ensure sufficient batch size
- Numerical instability: mixed precision config, gradient reduction in FP32, gradient clipping

## Key Takeaways

FSDP2 is the modern standard for large models in PyTorch.

**Sharding eliminates redundancy.** Distributed storage with temporary gathering.

**Trade-off: memory vs communication.** FULL_SHARD saves memory, requires more AllGather.

**Mixed precision essential.** BF16 for compute, FP32 for critical operations.

**Checkpointing enables larger batches.** Recomputation is cheaper than OOM.

**Proper wrapping matters.** One unit per transformer layer is a good starting point.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Distributed Training
**Previous:** [[01_Distributed_Training_Basics|Distributed Training Basics]]
**Next:** [[03_DeepSpeed_ZeRO|DeepSpeed ZeRO: Zero Redundancy Optimizer]]
