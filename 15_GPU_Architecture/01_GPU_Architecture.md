# GPU Architecture and Memory Hierarchy

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → GPU Architecture
**Previous:** [[../14_Security_Safety/05_NeMo_Guardrails|NeMo Guardrails]]
**Next:** [[02_Flash_Attention|Flash Attention and Mixed-Precision]]

---

## Why GPUs Dominate AI

Graphics processors turned out to be ideal for neural network computations thanks to massive parallelism.

CPU is a general-purpose processor with powerful cores (8-64), optimized for complex sequential tasks. GPU follows the "strength in numbers" philosophy — NVIDIA H100 contains thousands of smaller cores operating in parallel.

Neural networks consist of matrix operations: matrix multiplications, convolutions, normalization. Each operation decomposes into thousands of independent computations. When multiplying 4096×4096 matrices, over 68 billion operations are performed. On CPU this happens sequentially; GPU executes thousands of operations in parallel.

## Architecture of Modern NVIDIA GPUs

### Evolution from Gaming to AI

The history begins with Kepler (2012), followed by Maxwell, Pascal, Volta. Volta (2017) became a turning point with the introduction of Tensor Cores — specialized modules for machine learning.

**NVIDIA A100** — the industry workhorse. 80 GB HBM2e, 2 TB/s bandwidth. Third-generation Tensor Cores support TF32 and BFloat16, accelerating training without sacrificing accuracy.

**NVIDIA H100** (Hopper, 2023) raised the bar further. Fourth-generation Tensor Cores added FP8 — a format that doubles performance. Transformer Engine dynamically selects the optimal precision for each layer. Fourth-generation NVLink provides 900 GB/s between GPUs.

**NVIDIA Grace Hopper (GH200)** — a CPU-GPU superchip combining the Grace ARM CPU with the H100 GPU via NVLink-C2C (900 GB/s CPU-GPU bandwidth). 624 GB unified memory (96 GB HBM3 + 528 GB LPDDR5X) with coherent memory access — the GPU can directly access CPU memory and vice versa without explicit copies. Designed for workloads that exceed GPU memory: very large models, large KV caches, data preprocessing pipelines. The unified memory model simplifies programming for memory-intensive AI workloads.

**NVIDIA Blackwell B200/GB200 (now in production, 2025-2026)** — a generational leap:
- **Fifth-generation Tensor Cores** with FP4 support — doubles throughput over FP8, enabling 4-bit inference at near-FP8 quality with proper calibration
- **HBM3e memory** — 192 GB per GPU (vs H100's 80 GB), 8 TB/s bandwidth (vs 3.35 TB/s)
- **Second-generation Transformer Engine** — dynamic per-layer precision selection now includes FP4, FP8, BF16
- **Dual-GPU NVLink module** — two B200 GPUs connected as a single logical GPU with 384 GB shared memory and 1.8 TB/s bisection bandwidth. This is the key innovation: models that previously required multi-GPU tensor parallelism now fit in a single module
- **Fifth-generation NVLink** — 1.8 TB/s per GPU, NVLink Switch enabling up to 576 GPUs in a single NVLink domain
- **GB200 NVL72** — a rack-scale system with 72 Blackwell GPUs interconnected as a single compute fabric for training trillion-parameter models
- DeepSeek V3 was trained using FP8 on Hopper GPUs; Blackwell's FP4 enables even more aggressive quantization during training

**AMD MI300X (2024)** — AMD's competitive entry into AI accelerators:
- 192 GB HBM3 memory (more than H100's 80 GB, competitive with B200's 192 GB)
- 5.3 TB/s memory bandwidth
- CDNA 3 architecture with dedicated matrix cores
- ROCm software stack (open-source alternative to CUDA)
- Adopted by several cloud providers and AI labs for inference workloads where the large memory capacity enables serving larger models without multi-GPU setups
- Key advantage: memory capacity per dollar is competitive with NVIDIA, making it attractive for inference-heavy deployments
- Key limitation: the ROCm ecosystem is less mature than CUDA — fewer optimized libraries, less community support, and some framework features (Flash Attention, custom kernels) may lag behind

### Anatomy of a Streaming Multiprocessor

The Streaming Multiprocessor (SM) is the heart of an NVIDIA GPU. Each SM contains several types of execution units.

**CUDA Cores** — basic computational units for floating-point and integer operations. H100 contains 132 SM × 128 CUDA cores = over 16,000 cores.

**Tensor Cores** — specialized modules for matrix operations. They perform a matrix multiply-accumulate operation on 4×4 or 8×8 matrices in a single cycle. A single Tensor Core performs hundreds of operations per cycle.

**Special Function Units (SFU)** handle transcendental functions: sines, cosines, exponentials, logarithms for activation functions.

**Shared Memory** — a fast cache at the SM level, accessible to all threads. Data in shared memory is accessible hundreds of times faster than in global memory.

## Memory Hierarchy: The Key to Performance

### Why Memory Matters More Than Compute

The paradox of modern AI: most operations are limited not by computational power, but by memory bandwidth.

H100 performs ~1000 teraflops in FP16. HBM3 memory provides ~3.35 TB/s. To feed operands for 1000 teraflops, several terabytes of data per second must be transferred. Multiplying two numbers requires 3 memory accesses per 1 compute operation. At this ratio, memory becomes the bottleneck.

### Memory Hierarchy Levels

**High Bandwidth Memory (HBM)** — the main GPU memory. HBM3 in H100 provides ~3.35 TB/s with 80 GB capacity.

**L2 Cache** — an intermediate level between HBM and compute units. H100 has 50 MB of L2. Data in L2 is accessible significantly faster than in HBM.

**Shared Memory / L1 Cache** — the fastest memory at the SM level. H100: 256 KB of combined shared/L1 memory per SM with ~19 TB/s bandwidth — nearly an order of magnitude higher than HBM.

**Registers** — the fastest level with zero latency. Approximately 255 registers per thread.

**H100 Level Comparison:**
- Registers: 256KB/SM, ~19 TB/s, 0 cycles latency
- Shared/L1: 256KB/SM, ~19 TB/s, ~30 cycles
- L2 Cache: 50MB total, ~10 TB/s, ~200 cycles
- HBM3: 80GB, 3.35 TB/s, ~400 cycles
- PCIe/NVLink: Host RAM, 64-900 GB/s, ~10000 cycles

**Critical Rule:** The Bandwidth Ratio is Registers : Shared : L2 : HBM ≈ 6x : 6x : 3x : 1x. Moving data from HBM to Shared provides approximately a 6x access speedup.

### Arithmetic Intensity and Roofline Model

Arithmetic intensity is the ratio of the number of compute operations to the number of bytes read from memory.

Element-wise vector addition has low intensity: 1 operation / 8 bytes = 0.125 FLOP/byte. Such operations are always memory-bound.

Matrix multiplication has high intensity. For N×N×N matrices, the number of operations scales as N³, while data scales as N². Intensity grows with size.

**Attention in Transformers** is an example of a memory-bound operation. The standard implementation requires materializing a (sequence_length × sequence_length) matrix, creating a massive volume of reads and writes.

## CUDA Programming Model

### Thread Hierarchy

CUDA organizes computations hierarchically: threads → blocks → grid.

**Thread** — the minimal unit of execution. Threads within a warp (a group of 32) execute synchronously.

**Block** — a group of threads on a single SM. Threads within a block can synchronize and exchange data through shared memory.

**Grid** — the collection of all blocks in a kernel. Blocks execute on different SMs in parallel and independently.

### Memory Coalescing

When threads of a single warp access sequential memory addresses, the hardware merges these accesses into a minimum number of transactions. 32 threads reading 32 sequential 4-byte values create a single 128-byte transaction. Coalesced access can be an order of magnitude more efficient than scattered access.

### Occupancy and Latency Hiding

GPUs hide memory latency through massive parallelism. While some threads await data, others perform computations.

Occupancy is the ratio of active warps to the maximum possible. High occupancy (75%+) ensures good latency hiding. However, a kernel with low occupancy but efficient use of shared memory can outperform a kernel with high occupancy.

Resources limiting occupancy: registers per thread, shared memory per block, block size.

## Tensor Cores: Specialized Accelerators for AI

### How Tensor Cores Work

Tensor Cores perform the operation D = A × B + C, where A, B, C, D are small matrices (4×4 to 16×16).

The key advantage is that the entire matrix operation completes in a single cycle, whereas CUDA cores would require many cycles. This yields a theoretical 8-16x speedup.

Constraints: matrix dimensions must be multiples of 8 or 16, data must be in specific formats (FP16, BF16, TF32, FP8, INT8), and proper memory alignment is required.

### Data Formats for Tensor Cores

**FP32** — standard 32-bit precision. Tensor Cores use TF32 — a format with FP32 range but precision close to FP16. Enabled by default in PyTorch.

**FP16 and BFloat16** — 16-bit formats. FP16 has more mantissa bits (10 vs 7), BFloat16 has a larger range. BFloat16 is better for training due to its range matching FP32.

**FP8** — introduced in H100, proven at scale by DeepSeek V3's full FP8 training pipeline. Doubles performance compared to FP16. Transformer Engine automates calibration.

**FP4** — new in Blackwell. Doubles performance over FP8 for inference. Requires careful calibration but enables 4-bit inference at near-FP8 quality. DeepSeek V3 demonstrated that aggressive low-precision training is practical; FP4 extends this further.

**INT8** — for inference with quantization. Maximum performance with acceptable quality loss.

## Relationship Between GPU Architecture and Model Design

### Why Transformers Map Well onto GPUs

The Transformer architecture aligns perfectly with GPU strengths. Core operations — matrix multiplication (attention and FFN), softmax, normalization — all parallelize well.

The attention mechanism consists of matrix multiplications efficiently accelerated by Tensor Cores. Feed-forward layers are a sequence of matrix multiplications with nonlinearities. LayerNorm parallelizes well.

Researchers design architectures with hardware constraints in mind. Mixture of Experts is designed so that active parameters fit in GPU memory. State Space Models are optimized for sequential processing while retaining the ability for parallel training.

### Model Sizes and GPU Memory

Rule of thumb: each billion parameters requires ~2 GB of memory in FP16 (or 4 GB in FP32). Training requires gradients and optimizer states — this increases requirements by 3-4x for AdamW.

A 70B parameter model requires ~140 GB for weights alone in FP16. Model parallelism is necessary — distributing the model across multiple GPUs.

KV-cache for inference grows linearly with sequence length and batch size. For long contexts (100K+ tokens), KV-cache can consume more memory than the model itself.

### Batching and GPU Utilization

GPUs achieve peak performance with sufficient parallelism. A small batch size underutilizes resources. A large batch size may not fit in memory.

For training, the optimal batch size balances memory utilization, training dynamics, and throughput. For inference, continuous batching dynamically forms batches from requests, maximizing utilization.

Gradient accumulation allows emulating a large batch size: gradients are accumulated over several forward passes before updating weights.

## Key Takeaways

**Memory hierarchy dominates performance.** Most ML operations are limited by memory bandwidth. Optimization means minimizing accesses to slow levels and maximizing the use of shared memory.

**Coalesced access patterns are essential.** Warp threads must access sequential addresses. Scattered access degrades performance by an order of magnitude.

**Tensor Cores require proper data formats.** FP16, BF16, TF32, FP8 provide 8-16x speedup compared to FP32, but require proper alignment.

**Model architecture affects hardware efficiency.** Transformers map well onto GPUs. Architectural decisions directly impact hardware utilization. MoE models benefit from large memory (B200's 192 GB, MI300X's 192 GB) for hosting all experts.

**The GPU landscape is no longer NVIDIA-only.** AMD MI300X offers competitive memory capacity and bandwidth. Grace Hopper's unified memory simplifies large-model workloads. Blackwell's dual-GPU modules change the multi-GPU boundary. Choose hardware based on workload: training favors NVIDIA (CUDA ecosystem maturity), inference can benefit from AMD's memory advantage.

**Profiling reveals real bottlenecks.** Only profiling shows the true picture. torch.profiler and Nsight are essential tools.

## Practical Examples

### Memory Hierarchy Demonstration

A CUDA kernel divides work into threads, where each thread processes a data element. Sequential (coalesced) access is 2-5x faster than strided access thanks to memory transaction merging.

### Using Tensor Cores via PyTorch

To compare performance across different formats, matrices are created in FP32, FP16, and BF16, then matrix multiplication is performed with timing measured via CUDA Events. TFLOPS (teraflops) is computed for each format. Typical results on A100: FP32 around 19 TFLOPS, FP16 around 150 TFLOPS (8x speedup), BF16 similar to FP16.

Enabling TF32 via torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32 parameters automatically accelerates FP32 operations through Tensor Cores without code changes.

### Profiling and Bottleneck Analysis

torch.profiler is used for performance analysis with CPU and CUDA time tracking, tensor shapes, and memory usage. Results are exported to TensorBoard or Chrome trace viewer.

Arithmetic intensity determines the type of bottleneck: operations with intensity below the ridge point are memory-bound, above — compute-bound. The ridge point for modern GPUs typically falls in the 20-50 FLOP/byte range.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → GPU Architecture
**Previous:** [[../14_Security_Safety/05_NeMo_Guardrails|NeMo Guardrails]]
**Next:** [[02_Flash_Attention|Flash Attention and Mixed-Precision]]
