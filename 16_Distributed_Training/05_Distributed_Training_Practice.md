# Practice: Distributed Training

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Distributed Training
**Previous:** [[04_3D_Parallelism|3D Parallelism: Tensor, Pipeline, Data]]
**Next:** [[../17_Production_Inference/01_Inference_Architecture|Production Inference Architecture]]

---

## From Theory to Practice

Distributed training gains meaning when you see it in action. This practice guide will take you from a first multi-GPU experiment to a production-ready pipeline.

Key principle: start simple, add complexity as needed. The path: single GPU → DDP → FSDP → tensor/pipeline parallelism.

## Practical Approach Philosophy

**Iterative complexity.** Do not start with 3D parallelism. Each step is added when the previous one hits its limits.

**Debugging is a critical skill.** 80% of time goes to debugging: why processes fail to synchronize, NCCL timeout, memory leak on rank 3. The ability to diagnose quickly matters more than knowing parameters.

**Checkpointing is not optional.** In production, any job can be interrupted: hardware failure, preemption, OOM after 12 hours. Without robust checkpointing, you lose days and money.

**Monitoring provides insight.** Low GPU utilization? Bottleneck in the dataloader. High memory on one rank? Unbalanced sharding. Gradients exploding? Problem with LR or stability.

## Setting Up a Multi-GPU Environment

### Hardware Verification

A simple nvidia-smi command shows available GPUs. For distributed training, inter-GPU connectivity is critical: NVLink provides 600 GB/s, PCIe only 32 GB/s. The command nvidia-smi topo -m shows the topology.

Verify PyTorch sees all GPUs: torch.cuda.device_count().

### Environment Variables

Distributed training involves multiple processes that must discover each other.

**MASTER_ADDR and MASTER_PORT** define the "coordinator" address. Single-node: localhost. Multi-node: IP/hostname of the first node.

**NCCL_DEBUG** is your best friend when debugging. INFO or WARN shows which GPUs communicate, where timeouts occur, and which backend is used.

**Performance tuning** (NCCL_IB_DISABLE, NCCL_NET_GDR_LEVEL) only if you know the topology. For starters, use defaults.

### Launch Methods

**torchrun** is the standard for PyTorch. It automatically sets RANK, LOCAL_RANK, WORLD_SIZE. Single-node: torchrun --nproc_per_node=NUM_GPUS train.py.

**DeepSpeed launcher** is convenient for DeepSpeed but not required — it works with torchrun.

**SLURM** is the standard in research clusters. Proper --ntasks-per-node configuration is critical.

## Production Training Script

A complete distributed script consists of mandatory components.

**Initialization:** dist.init_process_group(backend='nccl'). NCCL is optimized for NVIDIA. Each process receives rank and local_rank. torch.cuda.set_device(local_rank) assigns a dedicated GPU.

**FSDP wrapping:** a two-level operation. First, fully_shard() individual layers (transformer blocks), then root wrap the entire model. Mixed precision policy is separate: bfloat16 for parameters.

**DistributedSampler:** splits the dataset across processes for unique batches. sampler.set_epoch(epoch) before each epoch for proper shuffling.

**Training loop:** nearly identical to single-GPU, but logging and saving only on rank 0. Forward, backward, and optimizer step are coordinated automatically through FSDP.

**Gradient clipping:** torch.nn.utils.clip_grad_norm_() after backward. More important in distributed than on single GPU — asynchronous synchronization can cause instability.

**Launch:** torchrun --nproc_per_node=NUM_GPUS train.py. For multi-node: --nnodes and --master_addr.

## Checkpointing

### Why It Is Not Optional

A model trains for 3 days on 64 GPUs, costing $2,000/day. Hour 67 — hardware failure. Without checkpointing: $5,500 and 3 days lost. With checkpointing every 4 hours: at most 4 hours and $300 lost.

Checkpointing saves the complete state: model weights, optimizer state (critical for Adam), LR scheduler, random seeds, and the current step.

### Practical Frequency

A trade-off between overhead and risk. A synchronous checkpoint can take 30-60 seconds — GPUs sit idle.

**Guidelines:**
- Research (time is not critical): every 2-4 hours
- Production: every 30-60 minutes
- Spot instances: every 15-30 minutes

Do not checkpoint every 100 steps if it takes 30 seconds — the loss from checkpointing exceeds the savings on failure.

### Async Checkpointing

Synchronous checkpointing blocks GPUs while writing to disk. For a large model, that means 30-60 seconds of idle time.

Asynchronous checkpointing solves this in two stages: fast state dict copy GPU→CPU (1-3 seconds), slow write CPU→disk in a background thread.

**Structure:** first, assemble the full state, transfer to CPU via .cpu() or .state_dict(). A background thread writes without blocking training.

**Critical issue:** a crash between the start and completion of an async save results in a corrupted checkpoint. Production solution: atomic operations — write to a temporary file, then perform an atomic rename. Rename is atomic — the checkpoint either exists in full or does not exist at all.

**Trade-off:** adds code complexity, requires more CPU memory, but saves GPU time. For expensive training, this is practically mandatory.

## Fault Tolerance

### Failure Scenarios

**Hardware failures** — a GPU dies, network timeout, node reboots.

**OOM** — sometimes occurs unexpectedly with variable sequence lengths. One long batch after 10 hours of successful training.

**NCCL timeouts** — communication stalls. Network issues or mismatched synchronization.

**Spot preemption** — cloud provider reclaims the instance with 30-120 seconds notice.

### Automatic Restart

The simplest fault tolerance is a bash wrapper with a retry loop. The script attempts to launch; on failure (exit code != 0), it waits and restarts. --resume_from_checkpoint auto finds the latest checkpoint.

**Retry logic:** typically 3-5 attempts to avoid infinite restarts on syntax errors. A 30-60 second pause for transient issues.

Production: Kubernetes with restart policies, SLURM requeue, AWS Batch, GCP AI Platform.

### Resume Logic

Resume finds the latest checkpoint by timestamp/step number, loads the state, and restores everything as it was.

**Search:** scan the directory, match checkpoint-*, sort, take the latest. If none found — step 0.

**Loading:** torch.load() reads the dictionary. model.load_state_dict() and optimizer.load_state_dict() restore state. Scheduler too. Random seeds via torch.manual_seed().

**Critical order with FSDP:** the model must be wrapped BEFORE loading state_dict. FSDP changes the parameter structure. Correct order: initialize → FSDP wrap → load state_dict.

**Returning to step:** if crash at step 15000, resume returns 15000, next forward is 15001.

## Monitoring

### What to Track

**Loss and gradient norm** — fundamental ML metrics. Loss not decreasing? LR too low or bad data. Gradient norm exploding? Instability or high LR.

**GPU utilization** — efficiency. 50-60% often indicates a bottleneck in data loading. 95%+ is good.

**Memory usage** — OOM prevention. Linear growth over time signals a memory leak (often from accumulation without zero_grad()).

**Throughput (tokens/sec)** — the primary performance metric. Estimates cost and time to completion.

**Communication overhead** (multi-node) — AllReduce time vs compute. >20% of time in communication — the model may be too small for the number of nodes, or the network is slow.

### Practical Monitoring

Weights & Biases or TensorBoard is sufficient.

**Initialization:** only rank 0 to avoid duplication. Config with hyperparameters for tracking.

**Periodic logging:** every 10-50 steps with no overhead. Metrics: loss, LR (changes with schedulers), GPU memory. torch.cuda.max_memory_allocated() shows peak memory — useful for detecting memory leaks.

**Rank 0 synchronization:** if all 64 ranks write to wandb simultaneously, API rate limits kill performance and cause duplication.

System-level metrics (temperature, utilization, network bandwidth): nvidia-smi in a loop, nvitop interactive, Prometheus exporters + Grafana.

## Cost Optimization

### Spot Instances: 70% Savings with Risk

Spot instances (AWS Spot, GCP Preemptible, Azure Low Priority) cost 70-90% less than on-demand but can be revoked.

**When to use:**
- Training can be interrupted (robust checkpointing is mandatory)
- Exact completion time is not critical
- Long-running jobs where savings are significant

**When to avoid:**
- Deadlines within hours
- Experiments without checkpointing
- Critical production training

In practice: combine on-demand (1-2 coordinator nodes) + spot (workers). If a spot worker is revoked, training continues.

### Right-Sizing

A100-80GB costs $3-4/hour. T4 costs $0.50/hour. For many tasks, T4 is sufficient.

**Heuristics:**
- <1B parameters: T4/L4
- 1-10B: A100-40GB
- 10-70B: A100-80GB or multiple with FSDP
- 70B+: H100 or A100 cluster with 3D parallelism

More GPUs does not always mean faster. Communication overhead grows. For a 1B model, 8xT4 can be slower and more expensive than 2xA100.

### Practical Optimizations

**Gradient accumulation** instead of larger batches — emulate batch 128 on a single GPU instead of 8 GPUs with batch 16. Slower, but cheaper for research.

**Mixed precision (bf16/fp16)** — a nearly free 2x speedup.

**Efficient data loading** — a slow dataloader means GPU idle time means wasted money. num_workers, pin_memory, prefetching.

## Debugging

### Common Problems

**"NCCL timeout" / processes hang**

Causes: one process crashed before a collective, network issues, mismatched code paths.

Solution: NCCL_DEBUG=INFO, check where it stalls. dist.barrier() before the suspect location — if it hangs, the process is not reaching that point.

**"CUDA out of memory" on some ranks only**

Cause: unbalanced sharding or different sequence lengths.

Solution: verify DistributedSampler splits correctly. Variable-length sequences: padding or bucketing.

**Loss NaN or exploding gradients**

In distributed settings, this can stem from synchronization issues or instability amplified by parallelization.

**Gradient synchronization check:** compute gradient norms on each rank, collect via all_gather. If norms differ by >10% — desync. Different ranks see different gradients.

**Diagnostic mechanism:** iterate over parameters with gradients, compute L2 norm via grad.norm(), create a tensor for each rank, dist.all_gather() collects them. Rank 0 analyzes and outputs a warning on mismatch.

Causes of desync: a process crashed without killing others, mismatched code paths, corrupted NCCL. Once desync is detected — stop and restart.

**Slow training / low GPU utilization**

The bottleneck is not in compute but in data loading or I/O.

Check:
- Dataloader num_workers > 0
- pin_memory=True for CPU→GPU transfers
- Checkpoint is non-blocking (async)
- Network bandwidth is sufficient (iperf)

### Debug Mode

When debugging, use minimal configuration:
- Single node instead of multi-node
- 2 GPUs instead of 8
- Small model and dataset
- Synchronous checkpoint instead of async

Once it works in the simple case, gradually increase complexity.

## Key Takeaways

Practical distributed training is a balance of performance, cost, and reliability.

**Start simple.** Single GPU → DDP → FSDP → advanced. Do not start with 3D parallelism unless you are certain.

**Checkpointing is critical.** In production, every job must survive interruption. Robust checkpointing and automatic restart.

**Monitor actively.** Metrics reveal progress and system efficiency. Low utilization = wasted money.

**Debug methodically.** When problems arise, simplify to the minimum, isolate the issue, use NCCL_DEBUG and verification utilities.

**Optimize costs.** Spot instances, right-sizing, and efficient code reduce expenses by 50%+ without sacrificing quality.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Distributed Training
**Previous:** [[04_3D_Parallelism|3D Parallelism: Tensor, Pipeline, Data]]
**Next:** [[../17_Production_Inference/01_Inference_Architecture|Production Inference Architecture]]
