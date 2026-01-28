# DeepSpeed ZeRO: An Alternative Approach

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Distributed Training
**Previous:** [[02_PyTorch_FSDP2|PyTorch FSDP2: Fully Sharded Data Parallel]]
**Next:** [[04_3D_Parallelism|3D Parallelism: Tensor, Pipeline, Data]]

---

## Origins and Philosophy of DeepSpeed

Microsoft Research (2020) introduced DeepSpeed with the central innovation ZeRO (Zero Redundancy Optimizer). Turing-NLG 17B, then Megatron-Turing 530B parameters.

The philosophy differs from PyTorch: comprehensive solution with its own training loop, configuration via JSON, integration with Hugging Face. An "all-in-one" approach.

ZeRO is mathematically equivalent to DDP — models are identical after an iteration. But the implementation radically optimizes memory down to the theoretical minimum.

## ZeRO: Zero Redundancy Optimizer

### The Redundancy Problem

Data parallel training with Adam stores on each GPU:
- Parameters FP16: 2N bytes
- Gradients FP16: 2N bytes
- Optimizer states FP32: 12N bytes (momentum, variance, master weights)
- Total: 16N bytes per parameter

For a 10B model: 160 GB on each GPU. With 8 GPUs: 1.28 TB total, even though unique data is only 160 GB.

ZeRO eliminates redundancy through partitioning.

### ZeRO Stage 1: Optimizer State Partitioning

Partitioning of optimizer states only.

Each GPU stores:
- Full parameters: 2N
- Full gradients: 2N
- Its portion of optimizer states: 12N/W

Memory per GPU: 4N + 12N/W

For 8 GPUs: 5.5N (vs 16N baseline). Savings ~3x.

Communication: AllGather of updated parameters after the optimizer step.

### ZeRO Stage 2: Gradient Partitioning

Adds gradient partitioning.

Each GPU:
- Full parameters: 2N
- Its portion of gradients: 2N/W
- Its portion of optimizer states: 12N/W

Memory: 2N + 14N/W

For 8 GPUs: 3.75N. Savings ~4x.

Communication: ReduceScatter of gradients, AllGather of parameters.

### ZeRO Stage 3: Parameter Partitioning

Full sharding, analogous to FSDP FULL_SHARD.

Each GPU:
- Its portion of parameters: 2N/W
- Its portion of gradients: 2N/W
- Its portion of optimizer states: 12N/W

Memory: 16N/W

For 8 GPUs: 2N. Savings 8x — linear with the number of GPUs!

Communication: AllGather of parameters before forward and backward of each layer.

### Stage Comparison

| Stage | Memory per GPU | Communication | Use case |
|-------|---------------|---------------|----------|
| 0 (DDP) | 16N | AllReduce gradients | Baseline |
| 1 | 4N + 12N/W | + AllGather params | Optimizer bottleneck |
| 2 | 2N + 14N/W | + ReduceScatter gradients | Gradient bottleneck |
| 3 | 16N/W | + AllGather params per layer | Maximum efficiency |

## ZeRO-Offload: Leveraging CPU Memory

When GPU memory is insufficient even with ZeRO-3, offload to CPU RAM.

**Offload Optimizer States:** Adam states in CPU RAM. Gradients are copied to CPU, optimizer step runs on CPU, updated weights are copied back.

**Offload Parameters:** Parameters also reside in CPU. Before forward/backward, copy to GPU; afterward, copy back.

**Trade-offs:** CPU memory is larger and cheaper, but PCIe bandwidth ~32 GB/s << GPU memory ~2 TB/s. CPU compute is slower than GPU.

**Efficiency:** When a model does not fit in GPU even with ZeRO-3, PCIe transfer can be overlapped with compute if batch is sufficiently large.

In practice: models ~10x larger than GPU memory, but slowdown 2-5x.

## ZeRO-Infinity: NVMe Offloading

The next step: offload to NVMe SSD. Terabytes of storage for training.

Architecture: parameters and optimizer states on NVMe, hierarchical GPU ← CPU ← NVMe, async prefetch hides latency.

Requirements: fast NVMe PCIe 4.0+, efficient async I/O, careful tuning.

In practice: rarely used — slowdown is significant for production. For research and fine-tuning on limited hardware — invaluable.

## Comparison with PyTorch FSDP

### When to Choose DeepSpeed

**DeepSpeed Advantages:**
- ZeRO-Offload and ZeRO-Infinity for extreme memory
- Mature ecosystem for very large models
- Integration with Hugging Face Trainer
- Advanced features: sparse attention, mixed precision policies

**FSDP Advantages:**
- Native PyTorch integration
- Compatibility with torch.compile
- Simpler API (FSDP2)
- Active development within PyTorch

### Recommendations

**FSDP when:**
- Model fits with ZeRO-3/FULL_SHARD
- PyTorch ecosystem integration is important
- torch.compile is planned
- Native PyTorch is preferred

**DeepSpeed when:**
- CPU/NVMe offload is needed
- Hugging Face Trainer is used
- Specific DeepSpeed features are needed
- Established DeepSpeed codebase

Both solutions deliver comparable performance for overlapping cases. The choice is determined by infrastructure and team experience.

## DeepSpeed Configuration

DeepSpeed is configured via a JSON file.

**train_batch_size and gradient_accumulation_steps** define the effective batch. DeepSpeed automatically computes the micro-batch size.

**fp16/bf16** controls mixed precision. loss_scale prevents underflow (0 = dynamic).

**zero_optimization** is the key section. stage defines the sharding level (0-3). offload_optimizer and offload_param activate ZeRO-Offload. pin_memory enables fast GPU-CPU transfers.

**Communication parameters:** allgather_bucket_size and reduce_bucket_size define buffer sizes. Typically 200-500 MB. overlap_comm enables overlapping communication with computation. contiguous_gradients accelerates AllReduce.

**optimizer and scheduler** define the optimizer and LR schedule. DeepSpeed provides optimized implementations. The value "auto" extracts parameters from TrainingArguments.

### Integration with Hugging Face

Hugging Face Transformers provides seamless integration via TrainingArguments. Specify the path to the JSON in the deepspeed parameter.

Trainer automatically initializes the DeepSpeed engine instead of the PyTorch distributed setup. Support for "auto" values in configuration.

When saving a checkpoint, DeepSpeed gathers sharded parameters on rank 0. stage3_gather_16bit_weights_on_model_save controls this for Stage 3.

Launch via deepspeed or accelerate launch.

## Key Takeaways

DeepSpeed ZeRO is a powerful alternative to FSDP with unique capabilities.

**Stages provide flexibility.** Stage 1/2/3 select the trade-off between memory and communication.

**Offload extends memory.** ZeRO-Offload and Infinity enable models that do not fit in GPU.

**Ecosystem matters.** Integration with Hugging Face simplifies adoption for NLP.

**Both solutions are valid.** FSDP and DeepSpeed solve the same problem through different approaches. The choice depends on requirements.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Distributed Training
**Previous:** [[02_PyTorch_FSDP2|PyTorch FSDP2: Fully Sharded Data Parallel]]
**Next:** [[04_3D_Parallelism|3D Parallelism: Tensor, Pipeline, Data]]
