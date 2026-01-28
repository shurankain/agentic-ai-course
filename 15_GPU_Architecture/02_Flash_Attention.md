# Flash Attention: A Revolution in Efficiency

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → GPU Architecture
**Previous:** [[01_GPU_Architecture|GPU Architecture and Memory Hierarchy]]
**Next:** [[03_Triton_Programming|Triton: GPU Programming in Python]]

---

## The Quadratic Complexity Problem of Attention

The attention mechanism in Transformer requires computing interactions between every pair of positions — N² values. At N = 512, that is 262 thousand values. At N = 4096 — 16 million. At N = 32768 — over a billion. At N = 128000 (Claude, GPT-4 Turbo) — 16 billion.

Quadratic complexity created a dual problem: computation time grew quadratically, and memory for storing the attention matrix also grew quadratically. For a 32K token sequence, the attention matrix occupied gigabytes of memory for each layer, for each head.

Researchers proposed sparse attention, linear attention, Longformer, BigBird. Each approach had trade-offs: it lost the ability to model certain dependencies or required architectural changes.

Tri Dao and colleagues (2022) discovered that the main bottleneck was not computation, but memory.

## The Key Insight: IO-Awareness

### Where the Bottleneck Actually Is

The standard attention implementation performs the following steps:
1. Computes the matrix Q × K^T (N × N values)
2. Stores it in HBM
3. Applies softmax to each row
4. Stores the result in HBM again
5. Multiplies the result by V
6. Writes the final result to HBM

Each step requires reading from and writing to HBM — O(N²) memory accesses. With HBM bandwidth of ~2-3 TB/s and matrix sizes in the gigabytes, these transfers determine execution time.

A GPU with teraflops of performance sits idle, waiting for data.

The solution: rewrite the algorithm to minimize HBM accesses, even at the cost of additional computation.

### The Math Stays the Same, IO Changes

The key observation: softmax can be computed incrementally. Instead of obtaining all N² values, data can be processed in blocks, updating the intermediate result.

Softmax admits decomposition. Online softmax — an algorithm known since the 1960s — allows recalculating softmax as new data arrives by correcting through the difference of maxima.

Flash Attention uses online softmax to process small blocks that fit in SRAM:
1. Loads a block of Q into SRAM
2. Iteratively loads blocks of K, V
3. Computes partial attention in SRAM
4. Updates running statistics for online softmax
5. Writes only the final result to HBM

The N × N matrix never exists in full — only small blocks in fast SRAM. IO is reduced from O(N²) to O(N).

## How Flash Attention Works

### Tiling: Partitioning into Blocks

Flash Attention partitions the attention matrix into small tiles, processes each in fast memory, and assembles the result.

Block size is chosen so that blocks of Q, K, V and intermediate computations fit in the SRAM of a single SM. Typical sizes are 64×64 or 128×128 elements.

The outer loop iterates over blocks of Q. For each Q block, the inner loop traverses all blocks of K and V. Each Q element "sees" all K elements — full attention is preserved.

### Online Softmax: The Key Innovation

Softmax is a nonlinear operation that requires all values in a row simultaneously. Online softmax circumvents this limitation.

The algorithm maintains two running statistics for each row:
- m — the maximum value
- l — the sum of exponentials for normalization

When processing a new block:
1. Compute the local maximum of the new block
2. Update the global maximum: m_new = max(m_old, m_local)
3. Recalculate previous sums accounting for the new maximum
4. Add the contribution of the new block

**Online Softmax Mathematics:**

The algorithm processes blocks B₁, B₂, ..., Bₖ sequentially. The maximum m is initialized to negative infinity, the sum l to zero, and the output O to zero.

For each block j, three steps are performed:
1. Maximum update: the new maximum m⁽ʲ⁾ is computed as the maximum between the previous value and the maximum element of the current scores Sⱼ
2. Sum correction: the sum l⁽ʲ⁾ is corrected by multiplying by the exponential of the difference between old and new maxima, then the contribution of the current block is added
3. Output update: the output vector O⁽ʲ⁾ is recalculated accounting for the change in maximum and new data

The final result O⁽ᵏ⁾ is mathematically equivalent to the full computation of softmax(QKᵀ) × V, but obtained without materializing the entire N × N matrix.

The key insight: the exponential multiplier of the maximum difference corrects all previous computations when a new maximum appears.

### Recomputation Instead of Storage

For the backward pass, standard attention stores the softmax matrix — O(N²) memory for each layer.

Flash Attention uses a radical approach: instead of storing this matrix, it recomputes it during the backward pass. Yes, this doubles the computation. But computation is cheap, and memory is expensive.

For long sequences, memory savings far outweigh the additional computation. Even with recomputation, Flash Attention is often faster than the standard implementation — so great is the gain from reducing IO.

## Practical Results

### Memory: Revolutionary Savings

For a 2048-token sequence, standard attention consumes ~16 MB per matrix (one head, FP32). Flash Attention requires only O(N) memory.

At sequence length 4096:
- Standard attention: ~64 MB per head
- Flash Attention: ~32 KB per head (only blocks in SRAM)

A ~2000x savings. This enables:
- Increasing batch size
- Using longer contexts
- Training larger models on fewer GPUs

### Speed: Up to 3x Speedup

Flash Attention not only saves memory — it is faster. On H100 with sequence length 2048, Flash Attention delivers a 2-3x speedup.

Speedup grows with sequence length:
- Seq 512: ~1.5x faster
- Seq 2048: ~2x faster
- Seq 8192: ~3x faster
- Seq 16384+: >3x faster

Counterintuitively: an algorithm that performs more computation (recomputation) runs faster. But computations in SRAM run at ~19 TB/s, while IO to HBM is only ~3 TB/s. Reducing IO outweighs the additional computation.

### Accuracy: Bit-Identical Results

Flash Attention computes mathematically the same result as standard attention. This is not an approximation. Online softmax yields exactly the same values.

In practice, there may be minor differences due to the order of floating-point operations, but at the level of machine epsilon.

## Flash Attention 2: Evolution of the Idea

### Parallelism Optimization

Flash Attention 2 (2023) focuses on better GPU utilization. The first version was memory-efficient but did not fully utilize computational resources.

**Parallelism across sequence length.** FA1 parallelized only across batch and heads. FA2 adds parallelism across sequence blocks, critical for long contexts with small batch sizes.

**Reducing non-matmul operations.** FA2 optimizes softmax scaling, causal masking, and other operations that do not use Tensor Cores.

**Optimizing work partitioning.** Blocks are distributed across SMs more efficiently, reducing load imbalance.

Result: FA2 is 2x faster than FA1, achieving ~70% peak Tensor Core performance.

### New Capabilities

Flash Attention 2 adds native support for:
- Causal masking for autoregressive models without overhead
- Grouped-query attention (GQA) and Multi-query attention (MQA)
- Variable sequence lengths in a single batch
- ALiBi and Rotary Position Embeddings

These optimizations are critical for production LLMs.

## Flash Attention 3: The Next Level

Flash Attention 3 (2024) leverages the capabilities of the Hopper architecture (H100):

**Asynchronous execution.** H100 supports asynchronous operation execution. FA3 overlaps loading the next block with computing the current one, hiding memory latency.

**FP8 support.** Native FP8 support with automatic scaling, using the H100 Transformer Engine.

**Hardware-aware implementation.** FA3 is written specifically for Hopper, using architecture-specific instructions and features.

Result: FA3 achieves ~75-80% peak H100 performance, approaching the theoretical maximum.

---

## Tensor Cores: The Heart of Modern GPU Computing

### What Are Tensor Cores

Tensor Cores are specialized compute units in NVIDIA GPUs, starting from Volta (2017). They perform matrix multiplications significantly faster than CUDA cores.

A standard CUDA core performs one FMA (fused multiply-add) operation per cycle. A Tensor Core performs an entire matrix operation D = A × B + C in a few cycles, where the matrices are 16×16. That is 4096 FMA operations in ~8 cycles — orders of magnitude more efficient.

### WMMA Instructions

WMMA (Warp Matrix Multiply Accumulate) are PTX instructions for programming Tensor Cores. Each warp collaboratively performs a matrix multiplication.

Basic WMMA operations:
- wmma.load — loads a matrix fragment from memory into warp registers
- wmma.mma — performs matrix multiply-accumulate on the Tensor Core
- wmma.store — stores the result from registers back to memory

Supported sizes: 16 × 16 × 16 (primary), 8 × 32 × 16 and 32 × 8 × 16 (alternative).

CUDA provides a C++ API through the nvcuda::wmma namespace. The programmer describes matrix fragments, then invokes synchronous operations for loading, multiplication, and storage. Each thread in the warp automatically receives the required portion of data.

Flash Attention uses these WMMA instructions for the matrix multiplications Q×K and Scores×V, achieving full Tensor Core performance.

### Evolution of Tensor Cores

- Volta (V100) 2017: 125 FP16 TFLOPS, FP16 → FP32
- Turing (T4) 2018: 65 TFLOPS, FP16, INT8, INT4
- Ampere (A100) 2020: 312 TFLOPS, FP16, BF16, TF32, INT8
- Hopper (H100) 2022: 989 TFLOPS, FP8, FP16, BF16, TF32, INT8

Each generation expanded format support and increased performance. The H100 with FP8 achieves nearly a petaflop on a single GPU.

---

## Mixed-Precision Training: Balancing Speed and Accuracy

### Why Mixed-Precision Is Needed

Neural network training historically used FP32. This provides high accuracy but requires 4 bytes per parameter, is slower on Tensor Cores, and consumes more memory.

Mixed-precision uses a combination of formats:
- Computation: FP16/BF16 on Tensor Cores (fast)
- Accumulation: FP32 to preserve accuracy (stable)
- Master weights: FP32 copy of weights for correct updates

Result: ~2x speedup, ~50% memory savings, training quality preserved.

### Formats: FP16, BF16, TF32, FP8

**FP16 (Half Precision):**
- 1 sign bit, 5 exponent bits, 10 mantissa bits
- Range: ~6×10⁻⁵ to 65504
- Problem: narrow dynamic range, risk of overflow/underflow

**BF16 (Brain Float):**
- 1 sign bit, 8 exponent bits, 7 mantissa bits
- Range: same as FP32
- Advantage: rarely requires loss scaling
- Disadvantage: lower precision than FP16

**TF32 (TensorFloat-32):**
- Internal format of Ampere+ Tensor Cores
- 8 exponent bits (like BF16), 10 mantissa bits (like FP16)
- Used automatically for FP32 operations on Tensor Cores

**FP8 (8-bit Float):**
- Two variants: E4M3 (4 exponent bits) and E5M2 (5 bits)
- E4M3 for forward pass (more precision), E5M2 for backward (more range)
- Requires careful scaling

### Loss Scaling

With FP16, gradients can round to zero (underflow). Loss scaling solves this:

1. Multiply the loss by a large number (scale factor, e.g. 1024)
2. The backward pass computes scaled-up gradients
3. Before updating weights, divide gradients by the scale factor

Dynamic loss scaling automatically selects the scale: if inf/nan is detected, decrease the scale; if several iterations pass stably, increase the scale.

PyTorch AMP (Automatic Mixed Precision) implements this automatically through GradScaler. In the training loop, the autocast() context manager is used for automatic precision selection of operations, then the scaler scales the loss before the backward pass.

### Numerical Stability in Attention

Attention is particularly sensitive to numerical stability. Softmax with large values can produce overflow: if max(x) > 88 in FP16, the exponential yields infinity.

The solution — numerically stable softmax: softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

Flash Attention implements stable softmax in its online algorithm, tracking the running maximum.

Additional considerations:
- Scaling Q×K: divide by √d_k before softmax to control magnitude
- Accumulated values: accumulator in FP32 even when operands are in FP16
- Residual connections: FP32 to preserve information during addition

### Practical Recommendations

**For training:**
- Start with BF16 if hardware supports it — simpler, does not require loss scaling
- FP16 + dynamic loss scaling for older hardware
- TF32 as a "free" speedup for existing FP32 code

**For inference:**
- FP16 is usually sufficient and widely supported
- BF16 if the model was trained in BF16
- FP8 for maximum throughput (requires calibration)

**When to be cautious:**
- Layer normalization and softmax — critical for precision
- Embedding layers — can accumulate errors
- Final layers — affect output quality

---

## Integration into the Ecosystem

### PyTorch Native Support

Starting with PyTorch 2.0, Flash Attention is integrated into torch.nn.functional.scaled_dot_product_attention. PyTorch automatically selects the optimal implementation: Flash Attention for long sequences, Memory-efficient attention as a fallback, Math attention for debugging.

Simply use F.scaled_dot_product_attention instead of a manual implementation — PyTorch will choose the best backend.

### Hugging Face Transformers

The Transformers library automatically uses Flash Attention when available. When loading a model via AutoModelForCausalLM, the parameter attn_implementation="flash_attention_2" is specified along with torch_dtype=torch.float16 to enable optimized attention.

### vLLM and Inference Engines

All modern inference engines (vLLM, TensorRT-LLM, TGI) use Flash Attention by default. This is critical for production: memory savings allow serving more concurrent requests, and speedup reduces latency.

## When Flash Attention Does Not Help

**Short sequences:** At sequence length < 256, the overhead from tiling and online softmax can exceed the gain from reduced IO. In practice, this is rarely a problem for LLM tasks.

**Cross-attention with small KV:** In encoder-decoder models, cross-attention accesses fixed-size encodings. If the encoder output size is small (256 tokens), the benefits are minimal.

**Specific architectures:** Some attention variants (e.g., with certain sparse patterns) are not directly supported.

**Hardware limitations:** Flash Attention requires a GPU with sufficient shared memory and Tensor Core support. On older GPUs (pre-Volta), support is limited.

## Key Takeaways

1. **IO-awareness matters more than compute efficiency.** On modern GPUs, the bottleneck is data transfer. An algorithm that minimizes IO can be faster even with additional computation.

2. **Recomputation is a legitimate strategy.** Recomputing can be cheaper than storing and reading.

3. **Kernel fusion is critical.** Combining operations into a single kernel avoids intermediate writes to HBM. Flash Attention is an extreme example.

4. **Tensor Cores are the key to performance.** WMMA instructions execute 16×16×16 matrix operations in a few cycles.

5. **Mixed-precision yields a 2x speedup.** BF16 is simpler, FP16 requires attention to gradient underflow, FP8 is the future of high-throughput inference.

6. **Numerical stability is non-trivial.** Attention requires stable softmax, proper scaling, and FP32 accumulation.

7. **Hardware-aware design.** An efficient algorithm accounts for specific hardware characteristics: SRAM size, bandwidth at each memory level, Tensor Core features.

8. **Practical impact.** Flash Attention enabled long contexts in modern LLMs. Without this optimization, 128K contexts in Claude or GPT-4 would be impractical.

## Practical Application

### Using Flash Attention via PyTorch

To use Flash Attention, test data Q, K, V are created in float16 format on a CUDA device. The first approach is explicitly specifying the backend via the sdpa_kernel(SDPBackend.FLASH_ATTENTION) context manager, useful for testing and performance comparison. The second approach is automatic selection via F.scaled_dot_product_attention, where PyTorch chooses the optimal backend.

The is_causal=True parameter enables causal masking for autoregressive models without requiring explicit creation of an attention mask. Flash Attention efficiently handles causal masking within its kernel.

For benchmarking, CUDA Events with enable_timing=True are used, measuring time between start.record() and end.record(). Flash Attention, Memory-efficient attention, and the standard Math backend are compared.

### Integration into Production Models

For Hugging Face models, attn_implementation="flash_attention_2" is specified when loading via AutoModelForCausalLM.from_pretrained(). Activation is verified via model.config._attn_implementation.

For custom PyTorch models, replace manual attention implementation with F.scaled_dot_product_attention(). PyTorch automatically uses Flash Attention for sequences longer than ~512 tokens.

### Profiling and Optimization

torch.profiler with parameters activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] and profile_memory=True enables performance analysis.

Key metrics:
- CUDA kernel time: the Flash Attention kernel should account for >90% of attention operation time
- Memory footprint: peak usage should scale as O(N), not O(N²)
- Throughput: measured in tokens/second for different batch sizes and sequence lengths
- GPU utilization: target utilization >80% for long sequences

Memory savings grow quadratically with sequence length: for seq_len=4096, savings reach 90%+ compared to a naive implementation. This allows increasing batch size by 2-4x or using significantly longer contexts.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → GPU Architecture
**Previous:** [[01_GPU_Architecture|GPU Architecture and Memory Hierarchy]]
**Next:** [[03_Triton_Programming|Triton: GPU Programming in Python]]
