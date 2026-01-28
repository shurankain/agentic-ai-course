# Speculative Decoding: Accelerating Generation

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[02_vLLM_internals|vLLM: Modern Inference Engine]]
**Next:** [[04_Model_Quantization|Quantization: Memory vs Quality]]

---

## Idea

Autoregressive generation has a fundamental limitation: each token requires a full pass through the model. The next token depends on the previous one — parallelism is impossible.

Speculative decoding: generate multiple tokens in a single pass of the large model.

**Principle:**
1. A small fast model generates K candidate tokens
2. The large model verifies all K tokens in a single pass
3. Tokens approved by the large model are accepted
4. The rest are rejected, generation continues from the rejection point

With good prediction, K tokens are generated in the time of one pass of the large model plus K passes of the small one.

**Mathematical Correctness:**
Output distribution is identical to pure autoregressive sampling with the large model. This is an exact algorithm, not an approximation. Achieved through rejection sampling.

## How It Works

**Draft generation:**
The small model generates K tokens sequentially. It saves not only the tokens but also the probabilities for acceptance/rejection.

**Target verification:**
The large model receives the prompt + all K draft tokens and performs a single pass. Thanks to causal attention, it computes probabilities for all K positions simultaneously. Instead of K separate passes — one batched pass.

**Acceptance/Rejection:**
For each draft token, probabilities are compared. If the target probability >= draft, the token is accepted. If lower, rejection sampling is triggered: a random number is generated, and the token is accepted with probability target/draft. Upon the first rejection, all subsequent tokens are discarded.

**Bonus token:**
After acceptance/rejection, an additional token is generated. If all K are accepted — from the target distribution. If rejected — from an adjusted distribution that compensates for the rejected probability mass.

## Acceptance Rate and Speedup

**Typical values:**
- Good draft model (same family): 70-90%
- Poor draft model: 30-50%
- EAGLE (feature-level): 85-95%

**Speedup depends on:**
- Acceptance rate
- Speed ratio between models
- Speculation length K

Calculation example: draft generates 4 tokens at 5ms each, target verification takes 50ms. With 80% acceptance, ~3-4 tokens are accepted plus bonus, totaling ~5 tokens in 70ms. Without speculation: 5 × 50ms = 250ms. Speedup 3.5×.

Optimal K is usually 3-8 tokens. Too large a K reduces acceptance (error accumulation). Too small a K underutilizes parallelism.

## Draft Model Selection

**Smaller models from the same family:**
LLaMA-70B + LLaMA-7B. Identical tokenizer (mandatory), similar training, architectural similarity. High acceptance 70-90%, speedup 1.5-2.5×. Trade-off: draft requires GPU memory.

**EAGLE:**
Draft operates at the feature level, not the token level. Target generates hidden states, a lightweight network predicts subsequent states, and tokens are extracted. Acceptance 85-95%, speedup 2-3×. Requires training for each target model. Adds ~5-10% memory overhead.

**Medusa:**
Multiple prediction heads in the target model. Head 1 — next token, Head 2 — one token ahead, and so on. All heads are simple MLPs, running in parallel. Tree-based verification checks combinations. Minimal memory overhead (~2-5%), but requires fine-tuning. Acceptance 60-75%.

## When to Use

**Ideal:**
- Single-user or small-batch (1-4 requests)
- Long-form generation (100+ tokens)
- Significant size difference between models (at least 5-10×)
- Predictable outputs (code, factual QA, structured data)

**Do NOT use:**
- High-throughput batch serving (batch > 8-16)
- Memory-constrained environments
- Very short outputs (<20 tokens)
- Mismatched vocabularies (different tokenizers)

**Real-world speedups:**
- Code generation: 2-3×
- Chat/conversational: 1.5-2.5×
- Creative writing: 1.3-1.8×
- Structured data extraction: 2.5-3.5×

## Trade-offs

**Memory vs Speed:**
Draft adds 10-30% to memory usage. In tight constraints, quantizing the target may be more beneficial.

**Latency vs Throughput:**
Optimizes per-request latency, may reduce aggregate throughput in batch serving.

**Quality guarantees:**
Mathematically preserves output distribution — identical quality is guaranteed.

**System constraints:**
Tokenizer compatibility (identical vocabulary is mandatory), hardware requirements (additional memory), implementation complexity.

## Practice in vLLM

Launch with parameters: --model (target), --speculative-model (draft), --num-speculative-tokens (usually 3-5), --tensor-parallel-size for target.

Key parameters:
- speculative-model: path to draft, identical tokenizer is mandatory
- num-speculative-tokens: 3-5 is optimal, determined experimentally
- speculative-draft-tensor-parallel-size: usually 1

Monitoring via Prometheus:
- spec_decode_draft_acceptance_rate (>70% is good)
- spec_decode_efficiency (actual speedup factor)
- num_spec_tokens (average number of accepted tokens)

Python API is transparent — when initializing LLM, specify speculative_model and num_speculative_tokens, then use standard generation.

## Key Takeaways

Mathematical precision — output is identical to the target model, not an approximation.

Draft model selection is critical — acceptance rate determines speedup.

Optimal for specific workloads — single-user, long-form. Large batch does not benefit.

Trade-off: memory for latency — 10-30% memory overhead for 1.5-3× latency reduction.

Production-ready in vLLM — simple activation, built-in monitoring.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Production Inference
**Previous:** [[02_vLLM_internals|vLLM: Modern Inference Engine]]
**Next:** [[04_Model_Quantization|Quantization: Memory vs Quality]]
