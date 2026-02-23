# Quantization: Memory vs Quality

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[03_Speculative_Decoding|Speculative Decoding: Accelerating Generation]]
**Next:** [[05_Inference_Cost_Optimization|Cost Optimization Strategies]]

---

## Why Quantize

LLaMA-70B in FP16 takes 140 GB — more than any consumer GPU and even a datacenter A100-80GB. Inference requires an expensive multi-GPU setup.

Quantization: representing weights with lower precision. FP16 → INT8 cuts size in half. FP16 → INT4 — by four times. LLaMA-70B in INT4 takes ~35 GB — fits on a single A100 or a 48GB consumer GPU.

Trade-off: lower precision means potential quality loss. The art of quantization is balancing compression ratio and quality.

Modern methods (GPTQ, AWQ) allow quantizing down to 4 bits with minimal loss. FP8 has become the standard on H100/Blackwell — 2x speedup with negligible impact on quality.

## Types of Quantization

**Post-Training Quantization (PTQ):**
Applied to a trained model without additional training. Fast (minutes to hours), does not require training infrastructure. Potentially more degradation, requires calibration data.

Process: load the model, run a calibration dataset (128-1024 samples), collect activation statistics, determine optimal parameters (scales, zero-points), convert to target precision.

**Quantization-Aware Training (QAT):**
Incorporates quantization into training. The model adapts to quantization noise. Better quality, but requires training (expensive, time-consuming).

**Static vs Dynamic:**
Static — parameters are fixed after calibration, faster inference. Dynamic — activation ranges are computed dynamically, slower but more accurate. For LLMs, static is typical.

## Formats

**INT8:**
8-bit integer. Symmetric quantization (range [-127, 127]) or asymmetric ([0, 255] or [-128, 127]). 2x size reduction, minimal loss (<1%), hardware acceleration on all modern GPUs.

**INT4:**
4-bit — aggressive compression. Only 16 values, typically [-8, 7] or [0, 15]. 4x size reduction. With GPTQ/AWQ, loss is 1-2%. Requires specialized CUDA kernels.

**FP8:**
8-bit floating point. E4M3 (4 bits exponent, 3 mantissa) or E5M2 (5 exponent, 2 mantissa). Native hardware support on H100+, floating point semantics, 2x speedup, minimal degradation. Becoming the standard.

**GPTQ:**
Hessian-based error compensation for INT4/INT3. 3-4 bit with <1% perplexity increase. Widely available on HuggingFace.

**AWQ:**
Activation-aware Weight Quantization. Identifies and protects "salient" weights via per-channel scaling. Comparable to or better than GPTQ, faster quantization.

**GGUF:**
Format for CPU and hybrid inference (llama.cpp). Supports mixed precision and efficient memory mapping. Q4_K_M (4-bit) is the most popular variant for local deployment.

For mathematical foundations and algorithm internals of these methods, see [[../15_GPU_Architecture/05_Quantization_Deep_Dive|Quantization Deep Dive]].

## Trade-offs

**Benchmark results LLaMA-2-70B:**
- FP16: 140 GB, perplexity 3.12, latency 1.0×
- INT8: 70 GB, perplexity 3.14, latency 0.6×
- FP8: 70 GB, perplexity 3.13, latency 0.5× (H100)
- INT4 AWQ: 35 GB, perplexity 3.18, latency 0.4×
- INT4 GPTQ: 35 GB, perplexity 3.20, latency 0.4×

Perplexity increase FP16 → INT4: ~2-3%. On most tasks, <1% accuracy drop.

**Task-dependent sensitivity:**
Low: text generation, classification, simple Q&A. Medium: summarization, translation, complex reasoning. High: mathematical reasoning, code generation, precise factual recall. For high-sensitivity tasks, INT8/FP8 is recommended over INT4.

**Hardware considerations:**
Consumer GPUs (RTX) — INT4 allows fitting larger models, AWQ/GPTQ optimized, ~50-100 tok/s for 70B INT4. Datacenter (A100/H100) — FP8 on H100 offers the best balance, INT8 on A100 is mature and stable.

## Practical Selection

**Decision framework:**

Quality-critical (research, enterprise) — FP16 or FP8. Production API — INT8 or FP8. Personal use — INT4.

Multi-GPU — FP16/INT8. Single datacenter GPU — INT8/FP8. Consumer 24GB — INT4 for larger models. 12GB — INT4 + smaller model.

H100 — FP8 via TensorRT-LLM/vLLM is optimal. A100 — INT8 via vLLM. Consumer RTX — INT4 via AWQ/GPTQ.

**Recommendations:**

Production API with quality focus: H100 FP8 or A100 INT8.

Cost-optimized: INT4-AWQ via vLLM maximizes requests per GPU.

Local/Edge: GGUF Q4_K_M via llama.cpp for CPU/consumer hardware.

## vLLM and TensorRT-LLM

**vLLM:**
The quantization parameter at load time: "awq", "gptq", "fp8" (H100+). Automatically uses optimized kernels. Pre-quantized models from HuggingFace (TheBloke). FP8 is applied dynamically to FP16 models.

**TensorRT-LLM:**
Maximum performance through deep CUDA integration. Checkpoint conversion with specified precision (weight-only INT8, INT8 KV cache, FP8, SmoothQuant). Compilation of an optimized engine via trtllm-build.

## FP4: Next-Generation Quantization (2025)

**FP4 on Blackwell (B200/GB200):** NVIDIA Blackwell introduces native FP4 Tensor Cores — 4-bit floating point with hardware acceleration. FP4 (E2M1 with microscaling) provides 2x throughput over FP8 and 4x over FP16.

**FP4 vs INT4:** FP4 uses non-uniform quantization levels (denser near zero), better matching LLM weight distributions. Early benchmarks show FP4 matching INT4 GPTQ quality while running at native hardware speed — no dequantization overhead.

**Impact on deployment:** FP4 on Blackwell may reduce the need for software-based quantization methods (GPTQ, AWQ). The hardware handles 4-bit inference natively, simplifying the deployment pipeline.

## KV-Cache Quantization

KV-cache memory scales with batch_size × sequence_length × num_layers × hidden_dim. For production serving with many concurrent users and long contexts, the KV-cache often exceeds model weight memory.

**INT8 KV-cache:** 2x compression with negligible quality impact. Supported in vLLM (`--kv-cache-dtype int8`) and TensorRT-LLM. Per-token quantization with per-head scales.

**FP8 KV-cache:** Preferred on H100+. Native hardware support means no dequantization penalty during attention computation. vLLM supports this via `--kv-cache-dtype fp8`.

**Production impact:** Quantizing KV-cache from FP16 to INT8 doubles the number of concurrent users a single GPU can serve (or doubles the maximum context length). This is one of the highest-ROI optimizations for production serving.

## GGUF Importance Quantization (IQ)

GGUF IQ (Importance Quantization) methods in llama.cpp go beyond standard k-quants:

**IQ1, IQ2, IQ3, IQ4:** Use non-uniform quantization with learned codebooks optimized for LLM weight distributions. IQ2_XXS achieves usable quality at 2.06 bits per weight — previously considered impossible for practical use.

**How IQ works:** Weights are vector-quantized using codebooks derived from lattice structures (E8 lattice for IQ2). The codebook entries are optimized for the actual weight distribution rather than being uniformly spaced. Importance weighting ensures that more critical layers receive higher precision.

**Practical impact:** IQ methods enable running 70B models on 16 GB RAM (IQ2), making frontier models accessible on consumer hardware. Quality degrades gracefully: IQ4_XS ≈ Q4_K_M quality at slightly smaller size, IQ3_XXS enables running models that otherwise wouldn't fit.

## Marlin Kernels

**Marlin** (2024) is a specialized CUDA kernel for mixed-precision inference: INT4 weights × FP16 activations.

**Why Marlin matters:** Standard dequantization-based INT4 inference leaves performance on the table. Marlin fuses dequantization into the matrix multiplication kernel, achieving near-ideal (4x) speedup over FP16 inference.

**Technical approach:** Marlin uses an asynchronous global-to-shared memory pipeline, perfectly overlapping data loading with computation. It exploits the Tensor Core pipeline structure to perform INT4→FP16 dequantization "for free" within the existing data movement.

**Benchmarks:** On A100, Marlin achieves ~3.7x speedup over FP16 for batch_size=1 (near the theoretical 4x). At larger batches, the speedup narrows as inference becomes compute-bound rather than memory-bound.

**Integration:** Marlin kernels are integrated into vLLM and GPTQ inference pipelines. When serving AWQ or GPTQ models in vLLM, Marlin kernels are used automatically when available.

## Key Takeaways

INT4 enables consumer deployment — 70B on a single GPU thanks to 4x compression.

FP8 is the standard on H100 with native support and minimal quality loss. FP4 on Blackwell is the next step — native hardware 4-bit at 2x FP8 throughput.

KV-cache quantization (INT8/FP8) doubles concurrent users or context length — one of the highest-ROI production optimizations.

Method matters — GPTQ, AWQ are significantly better than naive quantization. Marlin kernels extract near-ideal speedup from INT4 weights.

GGUF IQ methods push the frontier to 2-bit quantization for consumer hardware. IQ2 enables 70B models on 16 GB RAM.

Test on your task — sensitivity varies, benchmark on your use cases.

Tools are mature — vLLM, TensorRT-LLM make quantization straightforward.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[03_Speculative_Decoding|Speculative Decoding: Accelerating Generation]]
**Next:** [[05_Inference_Cost_Optimization|Cost Optimization Strategies]]
