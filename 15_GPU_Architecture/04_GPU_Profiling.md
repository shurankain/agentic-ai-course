# GPU Profiling and Optimization

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → GPU Architecture
**Previous:** [[03_Triton_Programming|Triton: GPU Programming in Python]]
**Next:** [[05_Quantization_Deep_Dive|Quantization: A Deep Dive]]

---

## The Art of Performance Measurement

Optimization without measurement is guesswork. Intuition fails even experienced developers: an operation that seems slow may account for a fraction of a percent of total time, while "fast" code may be the bottleneck of the entire pipeline.

GPU profiling is harder than CPU profiling. Asynchronous execution masks the true duration of operations. Thousands of parallel threads create non-trivial resource utilization patterns. Memory bandwidth, occupancy, instruction mix — numerous factors affect performance.

The modern toolkit spans several levels: from the high-level PyTorch profiler to the low-level NVIDIA Nsight. Each tool provides its own view of performance, and the ability to combine them is a key skill for an optimization engineer.

## torch.profiler: The First Line of Analysis

### Quick Start

PyTorch profiler is the most accessible tool for analyzing GPU code. It is integrated into PyTorch, requires no additional installation, and provides information at the PyTorch operation level.

Basic usage involves wrapping the code under investigation in the torch.profiler.profile() context manager. During initialization, you specify activity types: CPU and CUDA. Additional parameters allow recording tensor shapes (record_shapes=True), tracking memory usage (profile_memory=True), and capturing the call stack (with_stack=True). After execution, the key_averages() method aggregates results by operation type, and .table() outputs a formatted table.

Sorting by cuda_time_total shows operations that consumed the most total GPU time.

### What the Profiler Shows

**Self time vs Total time.** Self time is the time spent directly on the operation, excluding sub-operations. Total time includes the entire call tree. For composite operations (e.g., nn.Linear includes matmul and bias add), total time > self time.

**CPU time vs CUDA time.** CPU time is the execution time on the CPU (including waiting for the GPU). CUDA time is the execution time of kernels on the GPU. A large discrepancy indicates synchronization issues or a CPU bottleneck.

**Memory usage.** self_cpu_memory_usage and self_cuda_memory_usage show how much memory an operation allocates. Helps find memory leaks and inefficient allocations.

**Call count.** The number of times an operation is called. An unexpectedly large number may indicate a logic issue (e.g., redundant computations).

### Schedule for Training Loops

Profiling training requires a special approach: the first iterations are "cold" (compilation, caching), and their results are not representative.

The schedule mechanism controls exactly when to collect metrics. A typical strategy: skip the first iteration (wait=1), warm up on the next one (warmup=1), then actively profile several iterations (active=3), and repeat the cycle as needed (repeat). Within the training loop, prof.step() is called after each step to notify the profiler.

The on_trace_ready parameter accepts a callback function. tensorboard_trace_handler('./log/') saves the trace in TensorBoard format. Chrome trace viewer (chrome://tracing in Chrome) also reads these files, providing an interactive timeline of all operations.

### Interpreting Results

**Large percentage of time in a single operation** — an obvious optimization target. If 50% of time is spent on attention — this is the place for Flash Attention.

**Many small operations** — a sign of fragmentation. Hundreds of operations at 0.1ms each create overhead from kernel launch. Solution: fusion, torch.compile.

**CPU time >> CUDA time** — a CPU bottleneck. Possible causes: data loading, preprocessing on CPU, Python overhead. Solutions: DataLoader with num_workers, moving preprocessing to GPU, torch.compile.

**High memory usage in unexpected places** — a memory leak or inefficient intermediate tensors. Check that you are not retaining unnecessary references.

## NVIDIA Nsight Systems: A System-Level View

### When You Need Nsight

torch.profiler shows the PyTorch abstraction level. But sometimes you need to go deeper: understand exactly how kernels use the GPU, where threads are idle, how operations overlap.

NVIDIA Nsight Systems shows:
- Timeline of all CUDA operations
- Kernel launches and their duration
- Memory transfers (H2D, D2H, D2D)
- CUDA streams and synchronizations
- CPU activity in the context of GPU operations

### Launching Profiling

Profiling with Nsight Systems is launched from the command line: nsys profile -o my_profile python train.py. This creates a report file my_profile.nsys-rep, which opens in the graphical interface.

Useful options:
- -t cuda,nvtx — specifies event types for tracing (CUDA API, NVTX markers)
- --stats=true — outputs statistics directly to the console after completion
- --capture-range=cudaProfilerApi — profiles only the code section between explicit calls to torch.cuda.cudart().cudaProfilerStart() and Stop()

### Reading the Timeline

The timeline in Nsight Systems is a powerful tool. The horizontal axis is time, vertical lanes represent different resources (CPU threads, CUDA streams, GPU kernels).

**CUDA API row** shows CUDA function calls from the CPU side (cudaLaunchKernel, cudaMemcpy, etc.).

**CUDA HW row** shows actual execution on the GPU. Kernels are displayed as blocks; block length = execution time.

**Gaps between kernels** indicate:
- Kernel launch overhead (if gaps are small but frequent)
- Synchronizations (if gaps are large)
- CPU bottleneck (if the CPU lane is busy during the gap)

**Overlapping** — when compute and memory transfer occur in parallel. Well-optimized code maximizes overlap.

### Analyzing Specific Problems

**Problem: Many short kernels**
The timeline shows thousands of small blocks. Each kernel launch has an overhead of ~5-10μs. For a kernel executing in 20μs, this is 25-50% overhead.

Solution: operation fusion (torch.compile, custom kernels), increasing batch size.

**Problem: Large gaps between kernels**
The GPU is idle, waiting for the next workload. Causes:
- Explicit synchronization (torch.cuda.synchronize())
- Implicit synchronization (.item(), print tensor values, to('cpu'))
- CPU-bound preprocessing

Solution: Eliminate synchronization where possible, use asynchronous operations, pin_memory for data loading.

**Problem: Memory transfers dominate**
On the timeline, H2D/D2H transfers take more time than compute. Typical for inference with small batch sizes or for code that moves data between CPU and GPU in a loop.

Solution: Keep data on GPU, use pinned memory, increase batch size to amortize transfer overhead.

## NVIDIA Nsight Compute: Deep Kernel Analysis

### When You Need Nsight Compute

Nsight Systems shows "what is happening" at the system level. Nsight Compute answers the question "why a kernel behaves the way it does" at the individual kernel level.

Nsight Compute provides:
- Achieved throughput (memory, compute)
- Occupancy and its limiting factors
- Warp execution efficiency
- Memory access patterns
- Roofline analysis

This is a heavyweight tool: profiling a single kernel can take seconds. It is used for deep dives into critical kernels.

### Roofline Model

Roofline is a visualization of theoretical performance limits. The horizontal axis is arithmetic intensity (FLOP/byte), the vertical axis is achieved performance (GFLOPS).

The "roof" is formed by two lines:
- Sloped: memory bandwidth limit (while the operation is memory-bound)
- Horizontal: compute limit (when the operation is compute-bound)

The intersection point is the "ridge point." Operations to its left are memory-bound, to its right — compute-bound.

The position of your kernel on the roofline shows:
- If far from the roof — there is optimization potential
- If close to the sloped part — you need to reduce memory traffic
- If close to the horizontal part — you need to reduce compute or it is already at the limit

### Metrics for Analysis

**Achieved Occupancy** — the percentage of maximum possible warps that are active. Low occupancy (< 50%) may indicate:
- Too many registers per thread
- Too much shared memory per block
- A small grid (insufficient blocks)

**Memory Throughput** — how much data per second is read/written. Compare with the GPU's peak bandwidth. If significantly lower — non-coalesced access or other issues.

**Compute Throughput** — how many operations per second. For FP16 on H100, the peak is ~1000 TFLOPS. If your kernel shows 100 TFLOPS — it is far from optimal.

**Warp Execution Efficiency** — the percentage of active lanes in a warp. < 100% means branch divergence: different threads in the warp took different if/else paths.

## Identifying Bottlenecks

### Compute-bound vs Memory-bound

The first question during optimization: what limits performance?

A **memory-bound kernel** is characterized by:
- High memory throughput (close to peak)
- Low compute throughput
- Low arithmetic intensity

Optimization: reduce memory access (fusion, recomputation), improve cache utilization, coalescing.

A **compute-bound kernel** is characterized by:
- High compute throughput (close to peak)
- Memory throughput below capacity
- High arithmetic intensity

Optimization: use Tensor Cores, reduce precision, algorithmic improvements.

### Latency-bound Operations

Some operations are limited not by bandwidth or compute, but by latency. Examples:
- Atomic operations (serialization)
- Memory access with high latency (non-cached, random access)
- Synchronization barriers

Signs: both memory and compute throughput are low. SMs are idle, waiting for data.

Solution: Increase parallelism to hide latency, avoid atomics where possible, improve memory access patterns.

### Kernel Launch Overhead

Each kernel launch takes ~5-10μs on the CPU side. For a short kernel, this can be a significant fraction of the total time.

Signs: In the Nsight Systems timeline — small gaps between many short kernels.

Solution: Operation fusion, CUDA Graphs to eliminate launch overhead, increasing work per kernel.

## Practical Optimization Patterns

### Fusion: Combining Operations

Fusion is the primary source of speedup for memory-bound operations. By combining n operations into one, we eliminate n-1 intermediate writes and reads to HBM.

**torch.compile** provides automatic fusion. When decorating a function with @torch.compile, PyTorch analyzes the computation graph and generates optimized fused kernels. For example, a sequence of layer_norm → matmul → bias_add → gelu can be combined into one or several kernels, eliminating intermediate writes to global memory. This is especially effective for element-wise operations and small matrix transformations.

**Manual fusion** is used when torch.compile is insufficient. For specific patterns, you can use Triton or write custom CUDA kernels, providing full control over shared memory and register file usage.

### Memory Optimization

**Gradient checkpointing** — a compute vs memory trade-off. Instead of storing all intermediate activations for the backward pass, we recompute them on the fly. This reduces the memory footprint by 2-4x at the cost of approximately 30% additional computation. The torch.utils.checkpoint.checkpoint function wraps a layer or block of layers. The use_reentrant=False parameter uses the newer and more stable checkpointing implementation.

**Mixed precision** — using FP16 or BF16 instead of FP32. Benefits: half the memory for tensor storage, double the effective memory bandwidth, plus the ability to use Tensor Core acceleration. torch.autocast automatically selects the appropriate precision for each operation, preserving numerical stability where it is critical. A context manager with parameters device_type='cuda' and dtype=torch.float16 activates mixed precision for all operations within the block.

**Efficient data layouts** — using contiguous tensors and specialized memory formats. Channels-last format for convolutions provides better access locality. Non-contiguous tensors require scatter/gather operations during reads, significantly reducing effective bandwidth. Calling .contiguous() before intensive operations can significantly speed up execution.

### Reducing Synchronization

**Avoid .item() and .cpu()** — these operations synchronize GPU and CPU. Calling .item() on a GPU tensor blocks the CPU thread until all pending GPU operations complete, copies the value to the CPU, and only then returns control. In a training loop, calling loss.item() every iteration for logging can add hundreds of milliseconds of overhead. A better strategy is to log less frequently (e.g., every 100 steps) or accumulate losses on the GPU and transfer a batch of values.

**Non-blocking transfers** — for data loading, use the non_blocking=True parameter when transferring data to the GPU: data.to(device, non_blocking=True). This allows the CPU to continue execution while the transfer happens in the background. Works optimally with pinned memory (pin_memory=True in DataLoader).

**CUDA Graphs** — a mechanism for capturing and replaying sequences of operations without kernel launch overhead. The graph is "recorded" once through the torch.cuda.graph() context manager, capturing the entire sequence of CUDA operations. Then the graph can be "replayed" multiple times via .replay(), avoiding kernel launch overhead. Critical: the graph requires fixed tensor shapes and cannot contain CPU-GPU synchronizations.

### Batch Size Tuning

Large batch size:
- Better GPU utilization
- Overhead amortization
- May require learning rate scaling

Small batch size:
- Lower memory usage
- Faster convergence (sometimes)
- Worse GPU utilization

The optimal batch size depends on the model, GPU, and task. Profiling is the only way to find it.

## Production Case Studies

### Case 1: Data Loading Bottleneck During Training

Symptoms: GPU utilization fluctuates. In Nsight Systems: large gaps between training iterations.

Diagnosis: torch.profiler showed that DataLoader.__next__ accounted for 40% of the time.

Solution: increase num_workers (from 0 to 8), enable pin_memory=True, use prefetch_factor=2.

Result: 2x speedup in training throughput.

### Case 2: Fragmented Operations in Inference

Symptoms: Low GPU utilization during small batch size inference. Nsight Systems timeline: thousands of micro-kernels.

Diagnosis: The model consisted of hundreds of small operations (residual connections, normalizations, activations).

Solution: apply torch.compile with mode='reduce-overhead', use CUDA Graphs for inference.

Result: 3x speedup in latency.

### Case 3: Memory-bound Attention

Symptoms: Attention layer accounts for 60% of time during training. Nsight Compute shows high memory throughput, low compute throughput.

Diagnosis: Standard attention materializes an N×N matrix in HBM.

Solution: enable Flash Attention via attn_implementation="flash_attention_2".

Result: 2.5x speedup in attention, 40% reduction in memory usage.

### Case 4: CPU Bottleneck Due to Tokenization

Symptoms: Low GPU utilization during inference. Profiler shows most of the time is spent on CPU-side preprocessing.

Diagnosis: CPU tokenization took longer than model inference.

Solution: batch tokenization instead of per-request, use fast tokenizers (Rust-based), cache tokenized inputs where possible.

Result: 4x improvement in throughput for batch inference.

## Key Takeaways

**Measure, do not guess.** Performance intuition is often wrong. Only measurements reveal the real bottlenecks.

**Use the right tool for the job.** torch.profiler for quick analysis, Nsight Systems for a system-level view, Nsight Compute for deep dives into specific kernels.

**Understand compute-bound vs memory-bound.** Identifying the type of bottleneck determines the optimization strategy.

**Fusion is usually the answer.** For memory-bound operations (the majority of ML operations), kernel fusion is the primary source of speedup.

**Eliminate synchronization.** Hidden syncs (.item(), .cpu(), print) can destroy performance.

**Profile in production-like conditions.** Batch size, sequence length, hardware — everything affects the profile. Optimizing for the wrong workload is useless.

## Practical Application

### Comprehensive Profiling

Full training loop profiling involves initializing the profiler with specified activities (CPU and CUDA), a schedule (skipping cold iterations, warmup, active profiling), and callbacks for saving results. The training loop calls prof.step() after each iteration. Results are analyzed via prof.key_averages().table() sorted by various metrics: CUDA time, CPU time, memory usage. Data is exported to Chrome trace format (export_chrome_trace) and stack traces (export_stacks) for detailed analysis in visual tools.

The with_flops=True parameter enables FLOPS counting for operations, helping assess arithmetic intensity and identify the type of bottleneck.

### Identifying Bottleneck Type

**Memory-bound operations** are characterized by low arithmetic intensity (0.1-0.5 FLOP/byte for element-wise operations) and high memory throughput close to the GPU's peak bandwidth. Examples: add, ReLU, softmax. Optimization strategies: fusion to reduce memory access, improved cache utilization, coalesced memory patterns.

**Compute-bound operations** have high arithmetic intensity (100+ FLOP/byte for large matmul) and high compute throughput approaching the GPU's peak performance. The memory subsystem is idle. Optimization strategies: use Tensor Cores, lower precision (FP16/BF16), algorithmic improvements.

torch.compile automatically applies fusion by analyzing the operation graph and combining sequences of memory-bound operations. Typical speedup for fragmented code: 2-4x due to eliminating intermediate writes to HBM.

### Memory Profiling and Optimization

Memory monitoring involves tracking three metrics via torch.cuda:
- memory_allocated() — memory actually occupied by tensors
- memory_reserved() — reserved CUDA memory (may exceed allocated due to allocator caching)
- max_memory_allocated() — peak usage since the last reset

Comparing with and without gradient checkpointing demonstrates the trade-off: checkpointing reduces peak memory by 2-4x but increases compute time by ~30% due to activation recomputation. Use checkpoint_sequential for automatic model segmentation with checkpointing.

### Inference Latency Profiling

For accurate inference latency measurement:
- Use CUDA Events (torch.cuda.Event()) instead of time.perf_counter() — this excludes CPU overhead and measures pure GPU time
- Warm up the GPU before measurements (50+ iterations) to stabilize frequencies and fill caches
- Test various batch sizes to assess GPU utilization and determine the optimal operating range
- With good utilization, latency should grow sublinearly with batch size (throughput grows faster than latency)

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → GPU Architecture
**Previous:** [[03_Triton_Programming|Triton: GPU Programming in Python]]
**Next:** [[05_Quantization_Deep_Dive|Quantization: A Deep Dive]]
