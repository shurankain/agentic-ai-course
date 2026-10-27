# GPU Architecture and Memory Hierarchy

## Why GPUs Dominate AI

Graphics processors turned out to be ideal for neural network computations thanks to massive parallelism.

CPU is a general-purpose processor with powerful cores (8-64), optimized for complex sequential tasks. GPU follows the "strength in numbers" philosophy: NVIDIA H100 contains thousands of smaller cores operating in parallel.

Neural networks consist of matrix operations: matrix multiplications, convolutions, normalization. Each operation decomposes into thousands of independent computations. When multiplying 4096x4096 matrices, over 68 billion operations are performed. On CPU this happens sequentially; GPU executes thousands of operations in parallel.

## Architecture of Modern NVIDIA GPUs

### Evolution from Gaming to AI

The history begins with Kepler (2012), followed by Maxwell, Pascal, Volta. Volta (2017) became a turning point with the introduction of Tensor Cores: specialized modules for machine learning.

**NVIDIA A100**: the industry workhorse. 80 GB HBM2e, 2 TB/s bandwidth. Third-generation Tensor Cores support TF32 and BFloat16, accelerating training without sacrificing accuracy.

**NVIDIA H100** (Hopper architecture, announced 2022, GA 2023) raised the bar further. Fourth-generation Tensor Cores added FP8, a format that doubles performance. Transformer Engine dynamically selects the optimal precision for each layer. Fourth-generation NVLink provides 900 GB/s between GPUs.

**NVIDIA Grace Hopper (GH200)**: a CPU-GPU superchip combining the Grace ARM CPU with the H100 GPU via NVLink-C2C (900 GB/s CPU-GPU bandwidth). 624 GB unified memory (96 GB HBM3 + 528 GB LPDDR5X) with coherent memory access: the GPU can directly access CPU memory and vice versa without explicit copies. Designed for workloads that exceed GPU memory: very large models, large KV caches, data preprocessing pipelines. The unified memory model simplifies programming for memory-intensive AI workloads.

**NVIDIA Blackwell B200/GB200 (in production, 2025+)**: a generational leap:
- **Fifth-generation Tensor Cores** with FP4 support, doubles throughput over FP8. This enables 4-bit inference at near-FP8 quality with proper calibration
- **HBM3e memory**: 192 GB per GPU (vs H100's 80 GB), 8 TB/s bandwidth (vs 3.35 TB/s)
- **Second-generation Transformer Engine**: dynamic per-layer precision selection now includes FP4, FP8, BF16
- **Dual-GPU NVLink module**: two B200 GPUs connected as a single logical GPU with 384 GB aggregate HBM (192 GB per GPU) and 1.8 TB/s bisection bandwidth. This is the key innovation: models that previously required multi-GPU tensor parallelism now fit in a single module
- **Fifth-generation NVLink**: 1.8 TB/s per GPU, NVLink Switch enabling up to 576 GPUs in a single NVLink domain
- **GB200 NVL72**: a rack-scale system with 72 Blackwell GPUs interconnected as a single compute fabric for training trillion-parameter models
- DeepSeek V3 was trained using FP8 on Hopper GPUs; Blackwell's FP4 enables even more aggressive quantization during training

**AMD MI300X (2024)**: AMD's competitive entry into AI accelerators:
- 192 GB HBM3 memory (more than H100's 80 GB, competitive with B200's 192 GB)
- 5.3 TB/s memory bandwidth
- CDNA 3 architecture with dedicated matrix cores
- ROCm software stack (open-source alternative to CUDA)
- Adopted by several cloud providers and AI labs for inference workloads where the large memory capacity enables serving larger models without multi-GPU setups
- Key advantage: memory capacity per dollar is competitive with NVIDIA. This makes it attractive for inference-heavy deployments
- Key limitation: the ROCm ecosystem is less mature than CUDA: fewer optimized libraries, less community support, and some framework features (Flash Attention, custom kernels) may lag behind

**AMD MI325X (Q4 2024)**: 256 GB HBM3e (vs MI300X's 192 GB), improved memory bandwidth. Drop-in replacement for MI300X in existing deployments.

**AMD MI350 (Q3 2025)**: CDNA 4 architecture, significant compute uplift over MI300X. Positioned to compete directly with Blackwell B200.

**NVIDIA Blackwell Ultra GB300 NVL72 (as of late May 2026):** The production-grade Blackwell Ultra configuration. Per-rack: **1.1 EXAFLOPS** FP4, 288 GB HBM3e per GPU, liquid cooling mandatory (1,400W TDP per GPU). NVIDIA projects 60,000 GB300 NVL72 racks shipping in 2026 (+129% YoY). Inference cost on Blackwell Ultra is projected at **$0.15/M tokens for Llama-3-70B class models**: a 3-5x reduction from H100-era pricing. NVIDIA Q1 FY27 revenue reached $81.6B (+85% YoY), with data center revenue at $75.2B.

**AMD MI400 / Helios (as of late May 2026)**: 432 GB HBM4, 40 PFLOPS FP4, 2.9 ExaFLOPS per rack. H2 2026 production. The first serious Blackwell competitor: MI400 offers more memory (432 GB vs 288 GB HBM3e) and competitive raw throughput in a rack-scale configuration. AMD's Helios platform (MI400 + custom interconnect) targets the same hyperscale training and inference market as NVL72.

**NVIDIA Vera Rubin** (announced Computex, June 1, 2026; **in full production, with first customer deliveries due from August 2026** and hyperscaler integration through Q4). The next-generation platform after Blackwell. 50 PFLOPS FP4, 288 GB HBM4, 22 TB/s memory bandwidth, 10x performance-per-watt vs Grace Blackwell. NVLink 6 at 3.6 TB/s (2x Blackwell). TSMC 3nm process. Supply chain nearly 2x the size of the Grace Blackwell ramp based on what Jensen Huang called "outsized" AI demand. Assembly: 5 minutes per rack. 336B transistors on TSMC N3, paired with a Vera CPU carrying 88 Olympus ARM cores. Expected to generate $1T+ in orders through 2027.

**Supply is the binding constraint, not demand.** TSMC N3 capacity likely caps 2026 Rubin output at 200,000 to 300,000 units. For capacity planning that means allocation, not purchase: the question is what share you can get rather than what you can afford, and it favours committing early over waiting for better pricing. NVIDIA's output is running at about 1,000 NVL72 racks per week across all customers (72,000 Blackwell GPUs a week), consistent with the 60,000-rack projection for 2026.

**NVIDIA RTX Spark** (Computex 2026): the first ARM-based AI PC superchip. Grace CPU (20-core ARM, co-designed with MediaTek) + Blackwell RTX GPU (6,144 CUDA cores, fifth-gen Tensor Cores with FP4) connected via NVLink-C2C. Up to 128 GB unified LPDDR5X memory (300 GB/s bandwidth), 1 PFLOP of AI performance. Runs 120B+ parameter models locally. Ships fall 2026 in laptops and desktops from ASUS, Dell, HP, Lenovo, Microsoft Surface, and MSI. A three-generation roadmap: Spark (Blackwell, 2026), Vera Rubin Spark (2028), Rosa Feynman Spark (2030). This positions NVIDIA to compete directly in the consumer PC market, not just data centers.

**Google TPU v7 Ironwood (as of late May 2026):** Google's latest custom AI accelerator. Anthropic is deploying over 1 million Ironwood chips (exceeding 1 GW of compute capacity) exclusively for Claude inference and training in 2026: the largest single-customer TPU deployment announced to date. TPU v7 competes with Blackwell Ultra on training throughput while offering Google's proprietary interconnect advantages for large-scale distributed training.

**Huawei Ascend ramp-up (as of late June 2026):** China's primary GPU alternative to NVIDIA. Ascend 910C production target: ~600,000 units (2x 2025 volume). The 950PR was scheduled for Q1 2026 and the 950DT for Q4, with **Atlas 950 SuperPoD** (8,192 chips) also targeted at Q4. Huawei publishes no shipment figures, so read the whole roadmap as announced intent rather than confirmed availability. NVIDIA/SK hynix signed a multiyear memory co-development partnership (June 17) spanning Vera Rubin, RTX Spark, and Jetson Thor, deepening the NVIDIA supply chain advantage that Huawei cannot match. The Ascend ecosystem remains constrained by the ROCm-like software gap: fewer optimized libraries, limited framework support, and smaller developer community compared to CUDA. However, for Chinese AI labs operating under US export controls, Ascend is the only viable option at scale.

**Apple Silicon and local inference (as of late May 2026):** Ollama 0.19 introduced an MLX preview, marking a full transition to Apple's MLX framework on Apple Silicon (minimum 32 GB unified memory required). On M5 Max: 1,851 tok/s prefill and 134 tok/s decode on Qwen3.5-35B-A3B int4: a +57% prefill and +93% decode improvement over the previous Metal backend. This makes Apple Silicon competitive for local inference of 30B+ parameter models, relevant for on-device agent deployments and development workflows.

### Anatomy of a Streaming Multiprocessor

The Streaming Multiprocessor (SM) is the heart of an NVIDIA GPU. Each SM contains several types of execution units.

**CUDA Cores**: basic computational units for floating-point and integer operations. H100 contains 132 SM x 128 CUDA cores = over 16,000 cores.

**Tensor Cores**: specialized modules for matrix operations. They perform a matrix multiply-accumulate operation on 4x4 or 8x8 matrices in a single cycle. A single Tensor Core performs hundreds of operations per cycle.

**Special Function Units (SFU)** handle transcendental functions: sines, cosines, exponentials, logarithms for activation functions.

**Shared Memory**: a fast cache at the SM level, accessible to all threads. Data in shared memory is accessible hundreds of times faster than in global memory.

## Memory Hierarchy: The Key to Performance

### Why Memory Matters More Than Compute

The paradox of modern AI: most operations are limited not by computational power, but by memory bandwidth.

H100 performs ~1000 teraflops in FP16. HBM3 memory provides ~3.35 TB/s. To feed operands for 1000 teraflops, several terabytes of data per second must be transferred. Multiplying two numbers requires 3 memory accesses per 1 compute operation. At this ratio, memory becomes the bottleneck.

### Memory Hierarchy Levels

**High Bandwidth Memory (HBM)**: the main GPU memory. HBM3 in H100 provides ~3.35 TB/s with 80 GB capacity.

**L2 Cache**: an intermediate level between HBM and compute units. H100 has 50 MB of L2. Data in L2 is accessible faster than in HBM.

**Shared Memory / L1 Cache**: the fastest memory at the SM level. H100: 256 KB of combined shared/L1 memory per SM with ~19 TB/s bandwidth, nearly an order of magnitude higher than HBM.

**Registers**: the fastest level with zero latency. Approximately 255 registers per thread.

**H100 Level Comparison:**
- Registers: 256KB/SM, ~19 TB/s, 0 cycles latency
- Shared/L1: 256KB/SM, ~19 TB/s, ~30 cycles
- L2 Cache: 50MB total, ~10 TB/s, ~200 cycles
- HBM3: 80GB, 3.35 TB/s, ~400 cycles
- PCIe/NVLink: Host RAM, 64-900 GB/s, ~10000 cycles

**Critical Rule:** The Bandwidth Ratio is Registers: Shared: L2: HBM ≈ 6x: 6x: 3x: 1x. Moving data from HBM to Shared provides approximately a 6x access speedup.

### Arithmetic Intensity and Roofline Model

Arithmetic intensity is the ratio of the number of compute operations to the number of bytes read from memory.

Element-wise vector addition has low intensity: 1 operation / 8 bytes = 0.125 FLOP/byte. Such operations are always memory-bound.

Matrix multiplication has high intensity. For NxNxN matrices, the number of operations scales as N³, while data scales as N². Intensity grows with size.

**Attention in Transformers** is an example of a memory-bound operation. The standard implementation requires materializing a (sequence_length x sequence_length) matrix, creating a massive volume of reads and writes.

The number that turns this into a decision is the ridge point, where the roofline bends from bandwidth-limited to compute-limited. On an H100 it sits near 300 FLOP per byte in dense FP16 (989 TFLOPS over 3.35 TB/s), and roughly twice that in FP8. Compute the intensity of your kernel, compare it to that band, and you know which side you are on before profiling anything: below it, faster arithmetic buys nothing and only fewer memory trips help, and above it the reverse. Most attention kernels land far below, which is why Flash Attention attacks memory traffic rather than the multiply count.

## CUDA Programming Model

### Thread Hierarchy

CUDA organizes computations hierarchically: threads, then blocks, then grid.

**Thread**: the minimal unit of execution. Threads within a warp (a group of 32) execute synchronously.

**Block**: a group of threads on a single SM. Threads within a block can synchronize and exchange data through shared memory.

**Grid**: the collection of all blocks in a kernel. Blocks execute on different SMs in parallel and independently.

### Memory Coalescing

When threads of a single warp access sequential memory addresses, the hardware merges these accesses into a minimum number of transactions. 32 threads reading 32 sequential 4-byte values create a single 128-byte transaction. Coalesced access can be an order of magnitude more efficient than scattered access.

### Occupancy and Latency Hiding

GPUs hide memory latency through massive parallelism. While some threads await data, others perform computations.

Occupancy is the ratio of active warps to the maximum possible. High occupancy (75%+) ensures good latency hiding. However, a kernel with low occupancy but efficient use of shared memory can outperform a kernel with high occupancy.

Resources limiting occupancy: registers per thread, shared memory per block, block size.

## Tensor Cores: Specialized Accelerators for AI

### How Tensor Cores Work

Tensor Cores perform the operation D = A x B + C, where A, B, C, D are small matrices (4x4 to 16x16).

The key advantage is that the entire matrix operation completes in a single cycle, whereas CUDA cores would require many cycles. This yields a theoretical 8-16x speedup.

Constraints: matrix dimensions must be multiples of 8 or 16, data must be in specific formats (FP16, BF16, TF32, FP8, INT8), and proper memory alignment is required.

### Data Formats for Tensor Cores

**FP32**: standard 32-bit precision. Tensor Cores use TF32: a format with FP32 range but precision close to FP16. Enabled by default in PyTorch.

**FP16 and BFloat16**: 16-bit formats. FP16 has more mantissa bits (10 vs 7), BFloat16 has a larger range. BFloat16 is better for training due to its range matching FP32.

**FP8**: introduced in H100, proven at scale by DeepSeek V3's full FP8 training pipeline. Doubles performance compared to FP16. Transformer Engine automates calibration.

**FP4**: new in Blackwell. Doubles performance over FP8 for inference. Requires careful calibration but enables 4-bit inference at near-FP8 quality. DeepSeek V3 demonstrated that aggressive low-precision training is practical; FP4 extends this further.

**INT8**: for inference with quantization. Maximum performance with acceptable quality loss.

The gap between these formats is larger than most people expect. On an A100, dense matrix multiplication runs at roughly 19 TFLOPS in FP32 and roughly 150 TFLOPS in FP16 or BF16, an eight-fold difference for a one-line change in dtype. Part of that is available without touching the model at all: enabling TF32 through `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32` routes FP32 matmuls through Tensor Cores while keeping FP32 range. If a training job is compute-bound and still running plain FP32, those two flags are the cheapest speedup on the list.

## Relationship Between GPU Architecture and Model Design

### Why Transformers Map Well onto GPUs

The Transformer architecture aligns perfectly with GPU strengths. Core operations: matrix multiplication (attention and FFN), softmax, normalization: all parallelize well.

The attention mechanism consists of matrix multiplications efficiently accelerated by Tensor Cores. Feed-forward layers are a sequence of matrix multiplications with nonlinearities. LayerNorm parallelizes well.

Researchers design architectures with hardware constraints in mind. Mixture of Experts is designed so that active parameters fit in GPU memory. State Space Models are optimized for sequential processing while retaining the ability for parallel training.

### Model Sizes and GPU Memory

Rule of thumb: each billion parameters requires ~2 GB of memory in FP16 (or 4 GB in FP32). Training requires gradients and optimizer states. Mixed-precision AdamW needs about 16 bytes per parameter, eight times the FP16 weights alone.

A 70B parameter model requires ~140 GB for weights alone in FP16. Model parallelism is necessary, distributing the model across multiple GPUs.

KV-cache for inference grows linearly with sequence length and batch size. For long contexts (100K+ tokens), KV-cache can consume more memory than the model itself.

### Batching and GPU Utilization

GPUs achieve peak performance with sufficient parallelism. A small batch size underutilizes resources. A large batch size may not fit in memory.

For training, the optimal batch size balances memory utilization, training dynamics, and throughput. For inference, continuous batching dynamically forms batches from requests, maximizing utilization.

Gradient accumulation allows emulating a large batch size: gradients are accumulated over several forward passes before updating weights.

## The Frontier Labs Started Building Their Own Silicon

Through 2025 the accelerator question was which NVIDIA part to buy, with AMD as the alternative. During 2026 every major lab moved to designing its own, and the pattern is consistent enough to plan around.

OpenAI revealed **Jalapeño** with Broadcom on June 24, 2026, an inference-only accelerator designed end to end in roughly nine months with help from OpenAI's own models, targeted for deployment by the end of 2026. Anthropic entered early talks with Samsung on custom inference silicon, against compute costs reported around $1.25B per month. AMD acquired **Taalas** on August 6, 2026, which designs inference silicon around a model's dataflow rather than as a general-purpose array, and announced its **Helios** rack-scale system plus a Cerebras partnership that splits inference across both vendors' hardware.

Notice what all of these target: **inference, not training**. Training remains general-purpose work where flexibility matters and the hardware must survive whatever architecture comes next. Inference at scale is the opposite: a stable, repeated workload against a known model family, which is exactly the condition under which specialized silicon beats general silicon.

By October the pattern had moved from announcements to deployment plans. Microsoft said its third-generation **Maia 300** will become the primary inference system across its products, with production deployments from 2027 at multi-gigawatt scale and a claimed 40% more tokens per megawatt than leading GPU systems, every figure its own and none independently tested. OpenAI claimed in late August that its first inference chip beats Blackwell on performance per watt. The split is now explicit at the largest buyers: train on Nvidia and AMD, infer on your own silicon.

The procurement consequence is a trade rather than a win. Purpose-built inference hardware buys latency and energy efficiency, and costs you lock-in, re-certification when the weights change, and a slower path to adopting a new model. If your model changes quarterly, the specialization has nothing to amortize against.

### Memory became the constraint

The scarce resource shifted during 2026, and it is no longer compute. AI data centers are projected to absorb roughly **70% of global memory output** in 2026 as Samsung, SK Hynix, and Micron redirect wafers toward high-bandwidth memory for accelerators. DRAM went from about **$2.80 per GB in 2025 to roughly $12** across that shift. The three manufacturers face a US price-fixing class action over the reallocation, with the allegations unproven. The cost now passes through to finished systems. Nvidia told its largest customers in August that AI server prices would rise more than 15% for systems shipping from early 2027, Vera Rubin configurations included, citing memory, and Micron warned in October that shortages may tighten through 2028, with most of its 2027 capacity already sold.

Two consequences for anyone sizing infrastructure. Memory-bound decisions that looked marginal at 2025 prices now dominate: the quantization and KV-cache techniques in this module buy real money rather than headroom. And a build-versus-rent calculation done before 2026 is stale, because the hardware line item moved while the API prices in Module 01 fell.

## Key Takeaways

**Memory hierarchy dominates performance.** Most ML operations are limited by memory bandwidth. Optimization means minimizing accesses to slow levels and maximizing the use of shared memory.

**Coalesced access patterns are essential.** Warp threads must access sequential addresses. Scattered access degrades performance by an order of magnitude.

**Tensor Cores require proper data formats.** FP16, BF16, TF32, FP8 provide 8-16x speedup compared to FP32, but require proper alignment.

**Model architecture affects hardware efficiency.** Transformers map well onto GPUs. Architectural decisions directly impact hardware utilization. MoE models benefit from large memory (B200's 192 GB, MI300X's 192 GB) for hosting all experts.

**The GPU landscape is no longer NVIDIA-only.** AMD MI300X offers competitive memory capacity and bandwidth. Grace Hopper's unified memory simplifies large-model workloads. Blackwell's dual-GPU modules change the multi-GPU boundary. Choose hardware based on workload: training favors NVIDIA (CUDA ecosystem maturity), inference can benefit from AMD's memory advantage.

**Energy is becoming a constraint, not just a cost.** A single NVL72 rack draws ~120 kW. Data centers are projected to consume ~1,000 TWh by 2026 (IEA). Hyperscalers are procuring nuclear (SMR pipeline: 25 to 45 GW in 16 months) and geothermal to power AI infrastructure. Performance-per-watt (Vera Rubin: 10x over Grace Blackwell) is now an accelerator selection criterion alongside raw throughput. See Module 18, Enterprise AI Adoption for the full energy landscape.

**Profiling reveals real bottlenecks.** Only profiling shows the true picture. torch.profiler and Nsight are essential tools.
