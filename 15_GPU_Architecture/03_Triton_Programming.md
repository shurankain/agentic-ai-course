# Triton: Custom Kernels in Python

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → GPU Architecture
**Previous:** [[02_Flash_Attention|Flash Attention and Mixed-Precision]]
**Next:** [[04_GPU_Profiling|GPU Profiling and Optimization]]

---

## Why Write Custom GPU Kernels

In an ideal world, libraries like PyTorch would cover all needs. In reality, even the most advanced frameworks cannot anticipate every scenario.

A typical scenario: a model applies a sequence of operations (LayerNorm → linear → GELU → dropout). PyTorch executes each operation as a separate kernel. Between operations, intermediate results are written to HBM and read back. For four operations, that is eight extra passes through slow memory.

A fused kernel combines all operations into one: data is loaded from HBM once, processed in fast SRAM, and the result is written back. The IO savings can speed up the operation by 2-5x.

Another scenario is non-standard operations. New architectures constantly introduce attention modifications, new activation functions, and specialized normalizations. Waiting for PyTorch to add an optimized implementation can take years. A custom kernel delivers performance immediately.

Historically, writing GPU kernels required deep knowledge of CUDA C++, understanding of thread hierarchy, memory coalescing, and register pressure. The learning curve was steep, and debugging was painful.

Triton changed this equation. Developed at OpenAI, Triton provides a Python-like DSL for writing GPU kernels. It abstracts thread management, automates many optimizations, and allows achieving 80-90% of hand-written CUDA performance in a fraction of the development time.

## Triton vs CUDA: Choosing the Right Tool

### CUDA: Full Control, Full Complexity

CUDA is the native language of GPU programming. It gives full control over every aspect of execution: thread scheduling, memory layout, synchronization, hardware intrinsics. An experienced CUDA programmer can squeeze out every percent of performance.

But this power comes at a cost. A CUDA kernel for a simple matrix multiply takes hundreds of lines of code. You need to manually manage data partitioning into tiles, loading into shared memory with coalescing, synchronization between threads, avoiding bank conflicts, optimizing occupancy, and utilizing Tensor Cores.

Debugging CUDA is a separate nightmare. Race conditions, uninitialized memory, implicit type conversions — bugs manifest as incorrect results or spontaneous crashes, often non-reproducible.

### Triton: Block-Level Abstraction

Triton operates at the block level rather than individual threads. Instead of "thread 0 processes element 0, thread 1 processes element 1", the programmer thinks "a block processes a 64×64 tensor".

This abstraction radically simplifies code:
- No explicit thread management
- No explicit shared memory (Triton manages it automatically)
- No explicit synchronization between threads within a block
- Automatic memory coalescing
- Automatic Tensor Core utilization (where applicable)

The Triton compiler analyzes the code and generates optimized PTX (assembly for NVIDIA GPUs). It automatically applies many optimizations that require manual effort in CUDA.

### Trade-offs

Triton is not a silver bullet. There are scenarios where CUDA wins:

**Extreme optimization.** For performance-critical kernels (Flash Attention, cuBLAS-level matrix multiply), hand-written CUDA can provide a 10-20% advantage.

**Non-standard patterns.** Some algorithms do not map well to Triton's block model. Irregular memory access patterns, complex inter-block communication — CUDA is preferable here.

**Specific hardware features.** The newest GPU capabilities (asynchronous copies, special instructions) appear in CUDA first.

For most practical tasks, Triton is the right choice. Development time shrinks from weeks to hours, and performance reaches 80-90% of optimal CUDA.

## Block-Level Programming Model

### Thinking in Blocks

In Triton, the basic unit is a program instance that operates on a block of data. All threads within a block are abstracted away — you operate on entire tensors.

When you write tl.load(pointer) with a block size of 64, Triton generates code where 64 threads load 64 elements in parallel. Coalescing, bank conflicts, memory alignment — everything is handled by the compiler.

Key concepts:

**Program ID** (tl.program_id) — the identifier of the current program instance. Analogous to blockIdx in CUDA. Used to determine which portion of data the given instance processes.

**Block pointer** — a pointer to a block of data in global memory. Computed based on the program ID and block dimensions.

**Block operations** — operations on entire blocks: tl.load, tl.store, arithmetic, reductions. All operations are vectorized by block size.

### Example: Vector Addition

The simplest kernel for vector addition demonstrates the basic Triton pattern:
- Get the program ID (data block number)
- Compute offsets for block elements via tl.arange
- Create a mask for boundary conditions (when data size is not a multiple of the block size)
- Load data via tl.load with the mask specified
- Perform the operation (addition)
- Store the result via tl.store

An equivalent CUDA version would require explicit management of threadIdx, blockIdx, grid stride loops for large arrays, and manual shared memory management for optimization.

### Memory Access Patterns

Triton automatically optimizes memory coalescing for standard patterns. Sequential access (base + tl.arange(0, N)) generates coalesced loads/stores.

For accessing multidimensional data (e.g., matrices), indices for rows and columns are created via tl.arange, then combined through broadcasting (using [:, None] and [None, :]) to create 2D grid indices from 1D ranges. Final offsets are computed as row_idx * stride_row + col_idx * stride_col, matching the memory layout.

## Practical Examples

### Fused Softmax

Softmax is a classic example of an operation that benefits from fusion. The standard implementation requires four separate passes: computing the max per row, subtracting the max and applying exp, summing exp per row, and dividing by the sum.

Four passes mean four rounds of reads/writes to slow HBM memory. A fused kernel combines everything into a single pass.

The Triton implementation uses an online algorithm: each program instance processes one row of the matrix. The row is loaded into SRAM once, all operations (max, exp, sum, division) are applied in registers, and the result is written back. During loading, a mask is used for boundary conditions, and elements outside the bounds are set to negative infinity, which is handled correctly by the max operation.

This kernel is efficient for rows up to BLOCK_SIZE in length (typically 1024-2048). For longer rows, a modification with multi-block reduction is required.

### Fused LayerNorm + Linear

A more complex example: combining normalization with a linear transformation. In Transformers, this combination appears constantly.

Separate implementation: LayerNorm loads x, computes mean and variance, normalizes, and writes to HBM. Then Linear loads the normalized x back from HBM, multiplies by weights, adds bias, and writes the result. Total: 3 HBM reads + 2 HBM writes.

Fused version: loads x once, computes normalization in registers, immediately performs matrix multiplication (potentially in chunks for large weight matrices using tiling), adds bias, and writes the final result. Savings: 2 HBM reads and 1 HBM write, yielding substantial speedup for memory-bound operations.

### Matrix Multiplication

Matrix multiply is the king of deep learning operations. An optimized GEMM is an art form. But Triton allows achieving 80-90% of cuBLAS performance with relatively compact code.

The basic idea: classic tiled matrix multiplication. Matrices are partitioned into blocks, blocks are loaded into fast SRAM, partial products are computed, and results are accumulated.

Algorithm:
- Each program instance processes a result block C of size BLOCK_M × BLOCK_N
- The program ID determines the block position in the output matrix
- Offsets are computed for rows of matrix A and columns of matrix B
- The main loop iterates over the K dimension with step BLOCK_K
- On each iteration, blocks of A and B are loaded, and tl.dot is executed (compiled into Tensor Core instructions)
- Results are accumulated in float32 for precision, then converted to float16 on write
- Masks are used for correct handling of boundary blocks

Critical details: proper choice of block sizes affects the balance between data reuse and occupancy. Blocks that are too large exhaust registers; blocks that are too small lose performance to overhead.

## Autotuning: Automatic Search for Optimal Parameters

### The Parameter Selection Problem

GPU kernel performance depends on numerous parameters: block sizes, number of warps, pipelining stages. Optimal values depend on the specific GPU (A100 vs H100 vs RTX 4090), input data sizes, memory access patterns, and occupancy trade-offs.

Manual parameter tuning is tedious and not portable across hardware. Triton provides a built-in autotuning mechanism.

### The autotune Decorator

The @triton.autotune decorator automates the search for optimal parameters. You define a list of configurations with different values for BLOCK_SIZE, num_warps, num_stages, and other parameters. You also specify a key — the set of input parameters that the optimal choice depends on (typically data dimensions).

How it works: on the first kernel call with specific dimensions, Triton benchmarks all specified configurations, measures execution time, and selects the best one. The result is cached in the file system — subsequent calls with the same dimensions use the optimal config with no overhead.

This is especially important for kernels in training loops where tensor sizes are fixed: autotuning runs once at the start, and the entire subsequent training uses optimal parameters.

### Autotuning Strategy

Defining the configuration space requires understanding trade-offs:

**Large blocks** increase data reuse (more compute per load) but require more registers and shared memory, reducing occupancy.

**Small blocks** ensure high occupancy but provide less data reuse and greater overhead from grid launch.

**BLOCK_K** affects the main loop depth and the amount of intermediate data in registers.

Practical approach: start with a small set of "standard" configurations, then expand based on profiling.

## Profiling and Debugging

### Triton Profiling

Triton kernels integrate with torch.profiler for basic profiling. You can wrap the kernel call in a profiler context and obtain timing information showing your kernel's time compared to other operations.

For detailed analysis, use NVIDIA Nsight Compute — a professional tool that shows low-level metrics: achieved occupancy, memory throughput, instruction mix, L1/L2 cache hit rates, and register usage. Nsight Compute can pinpoint bottlenecks: whether the kernel is memory-bound or compute-bound, whether there are bank conflicts, and whether Tensor Cores are being utilized.

### Debugging Tips

**Verify correctness first.** Before optimizing, ensure the kernel produces correct results. Compare with a PyTorch reference implementation.

**Print intermediate values.** Triton supports tl.device_print for debugging (use with caution — it significantly slows execution).

**Test edge cases.** Check boundary conditions: sizes not divisible by BLOCK_SIZE, very small and very large inputs.

**Check masking.** Incorrect masks are a common source of bugs. Out-of-bounds reads can return garbage.

### Performance Debugging

If the kernel is slower than expected:

1. **Check occupancy.** Too many registers per thread reduces parallelism.

2. **Check memory throughput.** Low throughput indicates non-coalesced access or bank conflicts.

3. **Check compute utilization.** Low utilization indicates a memory-bound operation.

4. **Try different configurations.** Autotuning may find an unexpectedly better option.

## Integration with PyTorch

### Custom Autograd Function

To use a Triton kernel in a PyTorch model with automatic differentiation support, create a class inheriting from torch.autograd.Function.

In the forward method, the forward pass is executed: memory is allocated for the result, the grid size is determined (how many program instances to launch), and the kernel is called with the required parameters. Important tensors are saved via ctx.save_for_backward for use in the backward pass.

In the backward method, gradients with respect to inputs are computed based on the output gradient. For softmax, the analytical gradient formula is used: output * (grad_output - sum(output * grad_output)). The backward pass can also be implemented as a separate Triton kernel for maximum performance.

Usage: instead of calling the kernel directly, create a function via TritonSoftmax.apply(x), which automatically works with the PyTorch autograd system.

### torch.compile Integration

PyTorch 2.0+ with torch.compile can automatically generate Triton kernels for your code. The compiler analyzes the operation graph, identifies chains of operations for fusion, and generates optimized Triton kernels automatically.

Simply wrap a function or model with the @torch.compile decorator. The compiler will detect patterns like softmax + matmul + bias add and generate a single fused kernel, eliminating intermediate memory transfers.

This is a "free" optimization for many cases — you get fusion without writing custom kernels. However, for specific operations or maximum performance, hand-written Triton kernels are still preferable.

## Key Takeaways

Triton opens GPU programming to Python developers while retaining most of CUDA's performance.

**Block-level abstraction is powerful.** By thinking in blocks instead of threads, code becomes simpler and the compiler gains freedom to optimize.

**Fusion is the key optimization.** Combining operations into a single kernel is the main source of speedup. Every saved HBM transaction is a win.

**Autotuning eliminates guesswork.** Instead of manually tuning parameters, let Triton find the optimal ones for your hardware and workload.

**Start simple, optimize incrementally.** The first version should be correct. Optimization is an iterative process with profiling at every step.

**Know when to use CUDA.** For extreme performance or non-standard patterns, CUDA may be necessary. Triton is a tool, not a replacement for everything.

## Brief Example

### Full Example: Softmax with Autograd

To demonstrate the full Triton kernel development cycle, consider a softmax implementation with automatic differentiation support.

**Forward kernel:**
- Gets the program ID to determine the row
- Loads the row with masking (out-of-bounds values = negative infinity)
- Computes max for numerical stability
- Applies exp to (x - max)
- Sums the results and normalizes
- Stores the result

**Backward kernel:**
- Loads the saved output and incoming gradient
- Computes the gradient using the formula: output * (grad_output - sum(output * grad_output))
- This is the analytical expression of the softmax derivative
- Stores the gradient for the input

**Autograd class:**
- forward: creates the output tensor, determines the grid size (one program per row), calls the forward kernel, saves the output for backward
- backward: creates the grad_input tensor, calls the backward kernel, returns the gradient
- Used via a wrapper function: triton_softmax = TritonSoftmaxFunction.apply

**Testing:**
- Comparison with PyTorch reference: torch.allclose(triton_output, torch_output)
- Gradient verification: backward pass should produce identical gradients
- Performance benchmark: Triton version is often faster thanks to fusion

This pattern (forward kernel + backward kernel + autograd wrapper) is applicable to any differentiable operation in Triton.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → GPU Architecture
**Previous:** [[02_Flash_Attention|Flash Attention and Mixed-Precision]]
**Next:** [[04_GPU_Profiling|GPU Profiling and Optimization]]
