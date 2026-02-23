# Quantization: Model Compression Without Quality Loss

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → GPU Architecture
**Previous:** [[04_GPU_Profiling|GPU Profiling and Optimization]]
**Next:** [[../16_Distributed_Training/01_Distributed_Training_Basics|Distributed Training Fundamentals]]

---

## Introduction: Why Quantization Is Critical

Modern LLMs are staggering in size. LLaMA-70B requires 140 GB of memory just for weights in FP16. GPT-4 presumably contains over a trillion parameters. Running such models on commodity hardware would seem impossible — if not for quantization.

Quantization is a technique for reducing the precision of number representation: instead of 16 or 32 bits per parameter, we use 8, 4, or even 2 bits. Mathematically, this means information loss. In practice — with the right approach, the model preserves quality while requiring 2-8x less memory and running faster.

The paradox of quantization is that neural networks are redundant. Most weights do not need high precision — only relative proportions and general patterns matter. Quantization exploits this redundancy, preserving the essence of the model while discarding precision noise.

---

## Fundamental Concepts

### What Is Quantization

Quantization maps a continuous set of values (floating point) to a discrete one (fixed point with a limited number of levels).

For K-bit quantization, we have 2^K possible values. At K=8, that is 256 levels; at K=4 — only 16 levels; at K=2 — just 4 levels.

The simplest linear quantization: x_q = round((x - z) / s), where x is the original value in FP32/FP16, x_q is the quantized integer value, s (scale) is the scaling factor, z (zero point) is the zero offset.

Dequantization: x_dq = s * x_q + z

Scale and zero point are chosen to minimize the quantization error on calibration data.

### Symmetric vs Asymmetric

**Symmetric quantization** (z = 0):
- Range is symmetric around zero: [-α, α]
- Scale: s = α / (2^(K-1) - 1)
- Simpler to implement, faster at inference
- Inefficient for asymmetric distributions (e.g., ReLU outputs)

**Asymmetric quantization** (z ≠ 0):
- Range [β_min, β_max]
- Scale: s = (β_max - β_min) / (2^K - 1)
- Zero point: z = round(-β_min / s)
- Better coverage of the actual data range
- Additional computation due to zero point

For LLM weights, symmetric is typically used (distribution is close to symmetric); for activations — asymmetric.

### Per-tensor vs Per-channel vs Per-group

**Per-tensor quantization**: one scale/zero point for the entire tensor
- Minimal overhead
- Coarse approximation, worse quality

**Per-channel quantization**: separate scale for each output channel
- For weight matrix [out_features, in_features]: one scale per row
- Better quality, moderate overhead
- Standard for CNNs

**Per-group quantization**: divide the tensor into groups, one scale per group
- Group is typically 64-128 elements
- Balance between per-tensor and per-element
- Foundation of modern LLM-optimized methods

The finer the granularity, the better the quality, but the greater the overhead for storing scales.

---

## Weight Quantization vs Activation Quantization

### Weight-only quantization (W8A16, W4A16)

In weight-only quantization, weights are stored at low precision but dequantized to FP16/FP32 before computation. Activations remain at full precision.

Advantages:
- Memory savings: weights account for the bulk of model size
- Simplicity: no activation calibration needed
- Quality: activations do not lose precision

Disadvantages:
- Dequantization overhead: every forward pass requires unpacking weights
- Limited speedup: computation is still in FP16

Weight-only quantization is the standard for LLM deployment, where memory bandwidth is the bottleneck.

### Full quantization (W8A8, W4A4)

Both weights AND activations are quantized. Computation occurs in integer arithmetic (INT8 matmul).

Advantages:
- Maximum speedup: INT8 compute is faster than FP16
- Less bandwidth: both weights and activations at low precision

Challenges:
- Activation calibration: representative data is needed
- Outliers in activations: individual large values corrupt the scale
- Quality degradation: activations are more sensitive to quantization

For LLMs, full quantization is harder due to outliers in activations (SmoothQuant addresses this problem).

---

## GPTQ: Intelligent Weight Quantization

### The Optimal Quantization Problem

Naive quantization (rounding each weight independently) yields poor results at low bitwidth. Errors accumulate, degrading quality.

Key insight: quantization errors can be compensated by adjusting weights that have not yet been quantized.

### Optimal Brain Quantization (OBQ)

OBQ — the predecessor to GPTQ — quantizes weights sequentially, compensating for error:

1. Quantize weight w_i
2. Quantization error: δ = w_i - Q(w_i)
3. Distribute the error to remaining weights: w_j += δ · (H^(-1))_{ij} / (H^(-1))_{ii}
4. Repeat for the next weight

Where H is the Hessian of the loss function. Intuition: adjust weights to minimize the increase in loss from quantization.

Problem with OBQ: O(n³) complexity, impractical for large layers.

### GPTQ Algorithm

GPTQ makes OBQ practical for LLMs:
- Block processing: quantize 128 columns at a time rather than one weight at a time
- Lazy batch updates: accumulate updates, apply efficiently
- Cholesky decomposition: efficient update of H^(-1)

GPTQ algorithm: for each block of columns B and each column i in B, quantize the weight, compute the error, distribute the error to remaining columns using the Hessian inverse, apply lazy updates to remaining columns. Output: quantized weights Q, scales, zero points.

GPTQ quantizes LLaMA-65B to 4 bits in ~4 GPU-hours on an A100, preserving nearly original quality.

### Hessian Computation

For a linear layer y = Wx, the error L = ||y - y_q||²

Hessian: H = 2X^T X, where X is the matrix of input activations.

In practice: collect activations on a small calibration dataset (typically 128-512 samples), compute X^T X.

---

## AWQ: Activation-aware Weight Quantization

### The Idea: Not All Weights Are Equal

AWQ observation: a small fraction of weights (1-3%) is critical for model quality. These "salient" weights correspond to channels with large activations.

Instead of uniform quantization, AWQ:
1. Identifies salient channels by activation magnitude
2. Applies per-channel scaling, increasing precision for salient channels
3. Quantizes with scaling taken into account

### AWQ Mathematics

For weight matrix W and input X:
- Find the importance of each input channel: importance_i = ||X[:, i]||
- Compute scaling factors: s_i = importance_i^α (α is typically 0.5)
- Scale: W' = W · diag(s), X' = X · diag(1/s)
- Quantize W' (now salient weights have larger magnitude)

After quantization: W_q · X = (W'_q / s) · (X · s) ≈ W · X

Scaling does not change the result (mathematically), but improves quantization quality by protecting important weights.

### AWQ vs GPTQ

Comparison aspects:
- Approach: GPTQ — optimal error compensation, AWQ — protection of important weights
- Calibration: GPTQ — requires the Hessian, AWQ — requires only activations
- Quantization speed: GPTQ — slower (hours), AWQ — faster (minutes)
- Quality @ 4-bit: roughly equal
- Inference speed: GPTQ — baseline, AWQ — optimized for AutoAWQ

The practical choice is often determined by tooling: GPTQ is supported via AutoGPTQ, AWQ via AutoAWQ. Both deliver comparable quality.

---

## Quantized Model Formats

### GGUF (formerly GGML)

GGUF is a format from llama.cpp, optimized for CPU inference:

**Features:**
- Mixed quantization: different layers with different bitwidths
- Support for Q2, Q3, Q4, Q5, Q6, Q8 (and their variants _K, _0, _1)
- Metadata in the file (model architecture, tokenizer)
- Efficient loading and memory-mapping

**Naming convention:**
- Q4_0: 4-bit, naive quantization, no scaling
- Q4_K_M: 4-bit, K-means based, Medium quality
- Q4_K_S: 4-bit, K-means based, Small (faster)
- Q8_0: 8-bit, baseline quality

Q4_K_M is typically the optimal balance for most use cases.

### GPTQ format

GPTQ models are stored as:
- Quantized weights (typically INT4)
- Scale factors (FP16)
- Zero points (optional)
- Group size configuration

The format is optimized for GPU inference via exllama/exllamav2 or AutoGPTQ.

### AWQ format

Similar to GPTQ, but includes:
- Pre-computed scaling factors
- AWQ-specific metadata
- Optimized for AutoAWQ inference

---

## Quantization-Aware Training (QAT)

### Post-Training vs Training-time

**Post-Training Quantization (PTQ)**: quantization after training
- Fast, does not require retraining
- Quality loss under aggressive quantization

**Quantization-Aware Training (QAT)**: the model is trained to "know" about quantization
- Fake quantization during training: forward pass simulates quantization
- Gradients pass through STE (Straight-Through Estimator)
- The model adapts to quantization noise
- Better quality at low bitwidth

### Straight-Through Estimator

Problem: quantize() is a step function with zero gradient almost everywhere.

STE solution: during the backward pass, pretend that quantize() is an identity function: ∂quantize(x)/∂x ≈ 1

This is a rough approximation, but it works in practice. The model learns to keep weights in "good" ranges for quantization.

### QAT for LLMs

QAT is expensive for LLMs (full training cycle), so it is rarely used. Alternatives:
- QLoRA: fine-tune in 4-bit with LoRA adapters
- QLoRA + GPTQ: quantize, then fine-tune adapters
- Partial QAT: QAT only for sensitive layers

---

## Calibration: The Key to Quality

### Why Calibration Is Needed

Scale and zero point depend on the value range. For weights, the range is known immediately. For activations — the model must be run on representative data.

Poor calibration = poor quantization:
- Too narrow a range: saturation, loss of large values
- Too wide a range: low resolution, everything rounds to ~0

### Calibration Strategies

**Min-Max calibration**:
- Range = [min(x), max(x)] over calibration data
- Simplest method
- Sensitive to outliers

**Percentile calibration**:
- Range = [percentile(x, p), percentile(x, 100-p)]
- p is typically 0.01-1%
- Robust to outliers

**MSE calibration**:
- Select the scale that minimizes MSE between original and dequantized values
- Grid search or gradient-based optimization
- Best quality, more computationally expensive

**Entropy calibration (KL divergence)**:
- Minimize KL(original || dequantized)
- Preserves the distribution shape
- Used in TensorRT

### Amount of Calibration Data

Empirically:
- 128-512 samples is usually sufficient for LLMs
- More data rarely improves results
- Data must be representative (not edge cases)

For GPTQ/AWQ: the same calibration data is used for computing the Hessian/importance.

---

## Trade-offs: Quality vs Speed vs Memory

### Bitwidth spectrum

Format comparison:
- FP16: 2x vs FP32 memory reduction, no quality loss, training and high-quality inference
- INT8: 4x vs FP32 reduction, minimal quality loss, production inference
- INT4: 8x vs FP32 reduction, noticeable quality loss, resource-constrained inference
- INT2-3: 10-16x vs FP32 reduction, significant quality loss, extreme edge cases

### Quality degradation patterns

**Perplexity increase vs bitwidth** (typical for LLMs):
- 8-bit: +0.1-0.5 PPL
- 4-bit GPTQ/AWQ: +0.5-2.0 PPL
- 3-bit: +2-5 PPL
- 2-bit: +5-20 PPL (often unusable)

**Task-specific sensitivity**:
- Summarization: robust down to 4-bit
- Code generation: sensitive to quantization
- Math reasoning: highly sensitive
- Translation: moderately robust

### Inference Speed

Weight-only quantization in 4-bit on GPU:
- ~1.5-2x speedup in memory-bound scenarios
- Speedup comes from lower memory bandwidth
- Compute is still in FP16

Full INT8 quantization:
- ~2-3x speedup in compute-bound scenarios
- Uses INT8 Tensor Cores
- Requires careful quality management

---

## Relationship to Inference Serving

Quantization is critical for production serving:

**Memory savings → larger batches**:
A 4-bit model uses 4x less memory → can serve 4x more concurrent requests → higher throughput.

**Faster prefill**:
Weight loading is the bottleneck of the prefill phase. Smaller weights = faster prefill.

**KV cache competition**:
Memory freed by model quantization = more room for KV cache = support for longer contexts.

**Deployment flexibility**:
A 70B model in 4-bit (~35GB) fits on a single GPU, eliminating tensor parallelism overhead.

---

## Frontier Quantization Methods (2024-2025)

### FP4 on Blackwell

NVIDIA Blackwell (B200/GB200) introduces native FP4 (4-bit floating point) Tensor Cores — a hardware-level shift from integer to floating-point 4-bit computation.

**FP4 format:** E2M1 (2 exponent bits, 1 mantissa bit) with microscaling (block-level scale factors). Only 8 distinct values per element, but floating-point semantics preserve relative magnitudes better than INT4.

**FP4 vs INT4:**
- INT4: uniform quantization levels, optimal for symmetric distributions
- FP4: non-uniform levels (denser near zero), better for the naturally bell-curved weight distributions of LLMs
- FP4 on Blackwell achieves 2x the throughput of FP8 — making it 4x faster than FP16

**Practical implications:** FP4 on Blackwell provides INT4-level compression with FP8-level quality. Early benchmarks show FP4 models matching INT4 GPTQ quality while running at hardware-native speed (no dequantization overhead). This may make GPTQ/AWQ less necessary on Blackwell hardware.

### KV-Cache Quantization

KV-cache grows with sequence length and batch size, often consuming more memory than model weights in production serving. Quantizing the KV-cache is a high-impact optimization.

**KV-cache memory:** For a 70B model serving 32 concurrent requests with 4K context each, the KV-cache consumes ~40 GB in FP16. At 32K context — ~320 GB.

**Approaches:**
- **KV-cache INT8:** 2x compression with negligible quality loss. Supported in vLLM and TensorRT-LLM. Per-token or per-head quantization.
- **KV-cache FP8:** Preferred on H100+ hardware. Native compute support means no dequantization penalty. vLLM supports this out of the box.
- **KV-cache INT4:** 4x compression, slight quality degradation on long contexts. Research active; KIVI (2024) demonstrates effective 2-bit KV-cache via per-channel quantization with residual compensation.

**Impact:** KV-cache quantization from FP16 to INT8 doubles the number of concurrent users a single GPU can serve, or doubles the supported context length — a direct production cost saving.

### BitNet: 1-bit LLMs

**BitNet b1.58** (Microsoft, 2024) trains models with ternary weights: {-1, 0, +1}. Each weight requires only 1.58 bits.

**Key insight:** BitNet does not quantize a trained model — it trains from scratch with ternary weights. The model learns to represent knowledge within the ternary constraint.

**Results:** BitNet b1.58 at 3B parameters matches FP16 LLaMA-3B on many benchmarks. Memory: 8x reduction vs FP16. Compute: matrix multiplication becomes addition/subtraction (no actual multiplications needed). Energy: 71x less energy per token than FP16 at equivalent scale.

**Limitations:** Requires training from scratch (cannot convert existing models), smaller models show more degradation, ecosystem support is limited (no optimized kernels in major frameworks yet). BitNet represents a research direction rather than a production-ready solution as of 2025.

### AQLM: Additive Quantization

**AQLM** (Additive Quantization for Language Models, 2024) applies multi-codebook quantization to LLM weights.

**Approach:** Instead of scalar quantization (one scale per group), AQLM represents each weight vector as a sum of entries from multiple codebooks. This is vector quantization — a richer representation space at the same bit budget.

**Results at 2-bit:** AQLM at 2 bits per weight significantly outperforms GPTQ and AWQ at the same bitwidth. For LLaMA-2-70B at 2-bit, AQLM achieves perplexity within 0.5 of the 4-bit GPTQ result — remarkable for halving the storage.

**Trade-off:** Codebook lookup adds latency compared to simple dequantization. Best suited for memory-constrained deployment where the 2-bit regime is necessary.

### QuIP#: Incoherence Processing

**QuIP#** (Quantization with Incoherence Processing, 2024) achieves state-of-the-art 2-bit quantization through incoherent weight processing.

**Key idea:** Before quantization, apply random orthogonal transformations (Hadamard rotations) to weight matrices. This spreads outlier values across all dimensions, making the weight distribution more uniform and easier to quantize. After quantization, the inverse transformation is applied during inference.

**Results:** QuIP# at 2-bit matches GPTQ at 3-bit and approaches GPTQ at 4-bit for large models (70B+). Combined with lattice codebooks (E8 lattice), it achieves optimal packing of quantization levels.

**Relationship to FA3:** The "incoherent processing" technique in Flash Attention 3's FP8 mode is directly inspired by QuIP#'s approach — both use random rotations to improve quantization quality.

## Key Takeaways

1. **Quantization exploits neural network redundancy**. Most weights do not require high precision — patterns matter, not exact values.

2. **Weight-only quantization is the standard for LLMs**. W4A16 (4-bit weights, 16-bit activations) provides 4x memory savings with minimal quality loss.

3. **GPTQ and AWQ are state-of-the-art methods**. GPTQ compensates errors through the Hessian, AWQ protects important weights through scaling. Both deliver comparable quality.

4. **Calibration is critical**. 128-512 representative samples is usually sufficient, but data must reflect real-world usage.

5. **The trade-off is nonlinear**. Going from 8-bit to 4-bit yields 2x compression with moderate quality loss. Going from 4-bit to 2-bit yields another 2x compression, but quality degradation is disproportionately larger — though AQLM and QuIP# are narrowing this gap.

6. **QAT is better than PTQ at low bitwidth**. But for LLMs, QAT is expensive, so advanced PTQ methods are used instead. BitNet b1.58 shows that training-time ternary weights can match FP16 quality.

7. **Format matters for deployment**. GGUF for CPU (llama.cpp), GPTQ/AWQ for GPU (exllama, vLLM). Format choice is determined by the inference runtime.

8. **KV-cache quantization is a high-impact production optimization**. INT8/FP8 KV-cache doubles concurrent users or context length with minimal quality loss.

9. **FP4 on Blackwell is the next frontier**. Native hardware FP4 provides INT4 compression with better quality and no dequantization overhead. May reduce the need for GPTQ/AWQ on new hardware.

10. **Quantization enables democratization**. Thanks to 4-bit quantization, 70B models are accessible for running on consumer GPUs.

---

## Practical Application

### Using Ready-made Quantization Tools

**AutoGPTQ** is the primary tool for GPTQ quantization. The process consists of creating a quantization configuration (specifying bitwidth of 4 or 8, group size typically 128, symmetry), loading the source model, preparing calibration data (128-512 representative texts from your domain), running the quantization process, and saving the result. For a LLaMA-7B model, the process takes 30-60 minutes on an A100. After quantization, the model is loaded via AutoGPTQForCausalLM.from_quantized() and used as a regular Transformers model, but occupies 4x less memory.

**AutoAWQ** is an analogous tool for AWQ quantization. The interface is similar to AutoGPTQ, but the process is typically faster (10-20 minutes for LLaMA-7B), since AWQ does not require Hessian computation. Configuration is simpler: you specify bitwidth, group size, and version (GEMM for batch inference, GEMV for batch_size=1). AWQ automatically identifies important channels and applies protective scaling. The result is optimized for fast inference through custom CUDA kernels.

**llama.cpp and GGUF** — for CPU inference. Conversion is done through Python scripts from the llama.cpp repository. You choose the quantization type: Q4_K_M — optimal balance for most cases, Q5_K_M — slightly better quality at the cost of size, Q8_0 — minimal quality loss, Q3_K_M — extreme compression. After conversion, the model runs through the llama.cpp CLI or through bindings (llama-cpp-python). The advantage: works on any processor, supports memory mapping for RAM savings.

### Evaluating Quality After Quantization

The primary metric for evaluation is perplexity on a test dataset (typically WikiText-2). The procedure: load the original model in FP16 and the quantized version, run both through the same set of texts, compute cross-entropy loss for each token, average and exponentiate — this is the perplexity. Lower is better. Typical results for 4-bit GPTQ: perplexity increase of 0.5-2.0 points, which is negligible for practical use. It is also important to test on downstream tasks: for summarization and translation, quantization usually causes little harm; for code generation and math reasoning, losses are more noticeable.

An additional check is qualitative evaluation: generate responses to prompts characteristic of your use case with both models and compare. Sometimes perplexity does not correlate with user-perceived quality: the model may preserve quality on simple tasks but degrade on complex reasoning tasks.

### Minimal Per-group Quantization Example

To understand the inner workings, consider a simplified implementation of symmetric per-group quantization:

The quantize_per_group function takes a tensor, bitwidth (default 4), and group size (default 128). The tensor is reshaped into groups via reshape, then for each group the maximum absolute value is found. Scale is computed as max_abs / (2^(bits-1) - 1). Values are quantized by rounding the division by scale, converted to int8, and clamped to the range [-max_int, max_int]. The quantized tensor and the array of scales are returned.

The dequantize_per_group function performs the inverse operation: takes the quantized tensor and scales, reshapes into groups, converts to float, multiplies by the corresponding scales, and returns to the original shape.

A test on a random 1024x1024 tensor shows an MSE error of ~0.001-0.01 for 4-bit quantization and an 8x size reduction (when using efficient INT4 packing into bytes).

This code demonstrates the core mechanics: divide the tensor into groups, find the scale for each group, round and clamp. Real libraries add optimizations: fused kernels, mixed precision compute, efficient INT4 packing into bytes.

### Practical Recommendations for Method Selection

**For production GPU inference**: use GPTQ or AWQ with 4-bit precision. The choice between them is usually determined by compatibility with the inference engine (vLLM supports both, TGI works better with AWQ, exllama2 is optimized for GPTQ). Quality is comparable. Calibration data is critical: use 256-512 examples from your actual data distribution, not generic Wikipedia texts.

**For CPU/edge inference**: GGUF format via llama.cpp. Start with Q4_K_M; if quality is insufficient, move to Q5_K_M. The advantage: memory mapping allows running large models on limited RAM systems.

**For fine-tuning**: QLoRA (4-bit base model + LoRA adapters in FP16). This allows fine-tuning a 70B model on a single 80GB GPU. The base model is quantized to 4-bit via bitsandbytes (nf4 format), adapters are trained at full precision.

**For maximum quality**: stay at 8-bit (W8A8 or W8A16). Quality loss is nearly imperceptible, memory savings are 2x, and all modern frameworks are well optimized for INT8.

**Do not use quantization for**: training (except QAT), critical tasks where the slightest degradation is unacceptable (medical, legal), models smaller than 7B (they are already small; the overhead from quantization is not justified).

For deployment-oriented guidance — hardware decision frameworks, serving framework integration (vLLM, TensorRT-LLM), FP8 format details, and production benchmarks — see [[../17_Production_Inference/04_Model_Quantization|Model Quantization for Production]].

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → GPU Architecture
**Previous:** [[04_GPU_Profiling|GPU Profiling and Optimization]]
**Next:** [[../16_Distributed_Training/01_Distributed_Training_Basics|Distributed Training Fundamentals]]
