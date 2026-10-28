# Speculative Decoding: Accelerating Generation

## Idea

Autoregressive generation has a fundamental limitation: each token requires a full pass through the model. The next token depends on the previous one: parallelism is impossible.

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
The small model generates K tokens sequentially. It saves the tokens and the probabilities for acceptance/rejection.

**Target verification:**
The large model receives the prompt + all K draft tokens and performs a single pass. Thanks to causal attention, it computes probabilities for all K positions simultaneously. Instead of K separate passes. One batched pass.

**Acceptance/Rejection:**
For each draft token, probabilities are compared. If the target probability >= draft, the token is accepted. If lower, rejection sampling is triggered: a random number is generated, and the token is accepted with probability target/draft. Upon the first rejection, all subsequent tokens are discarded.

**Bonus token:**
After acceptance/rejection, an additional token is generated. If all K are accepted: from the target distribution. If rejected: from an adjusted distribution that compensates for the rejected probability mass.

## Acceptance Rate and Speedup

**Typical values:**
- Good draft model (same family): 70-90%
- Poor draft model: 30-50%
- EAGLE (feature-level): 85-95%

**Speedup depends on:**
- Acceptance rate
- Speed ratio between models
- Speculation length K

Calculation example: draft generates 4 tokens at 5ms each, target verification takes 50ms. With 80% acceptance, ~3-4 tokens are accepted plus bonus, totaling ~5 tokens in 70ms. Without speculation: 5 x 50ms = 250ms. Speedup 3.5x.

Optimal K is usually 3-8 tokens. Too large a K reduces acceptance (error accumulation). Too small a K underutilizes parallelism.

## Draft Model Selection

**Smaller models from the same family:**
LLaMA-70B + LLaMA-7B. Identical tokenizer (mandatory), similar training, architectural similarity. High acceptance 70-90%, speedup 1.5-2.5x. Trade-off: draft requires GPU memory.

**EAGLE and EAGLE-2:**
Draft operates at the feature level, not the token level. Target generates hidden states, a lightweight network predicts subsequent states, and tokens are extracted.

EAGLE (2024): Acceptance 85-95%, speedup 2-3x. Requires training a small autoregression head for each target model. Adds ~5-10% memory overhead.

EAGLE-2 (2024): Introduces **context-aware dynamic draft trees**. Instead of a fixed tree structure, EAGLE-2 builds the draft tree adaptively based on the confidence of each draft token. High-confidence tokens branch further; low-confidence tokens are pruned early. Results: ~10-20% higher acceptance rate than EAGLE-1, and ~3-4x speedup on coding tasks. EAGLE-2 topped the Spec-Bench leaderboard in 2024.

**Medusa:**
Multiple prediction heads in the target model. Head 1: next token, Head 2: one token ahead, and so on. All heads are simple MLPs, running in parallel. Tree-based verification checks combinations. Minimal memory overhead (~2-5%), but requires fine-tuning. Acceptance 60-75%.

**Draftless Speculative Decoding (2024):**
Methods that avoid a separate draft model entirely:

**Jacobi decoding:** Initialize all positions with random tokens, then iteratively refine all positions in parallel using the target model. Each iteration is a single forward pass. When tokens converge (stop changing), generation is complete. No draft model needed, no additional memory. Speedup 1.5-2x for predictable outputs.

**Lookahead decoding (2024):** Combines Jacobi decoding with n-gram caching. During generation, observed n-grams are cached. When the model generates a token matching the start of a cached n-gram, the cached continuation is verified in parallel. Effectively "free" speculation from patterns already seen in the output.

**Token recycling (2024):** Reuses rejected tokens from previous speculation rounds as draft candidates for subsequent rounds. Rejected tokens are not discarded but stored in a pool. When generating the next draft, the pool is consulted first: tokens already computed by the target model are "recycled" as high-quality draft candidates. This is effectively free since the target model's logits for these positions were already computed during verification. Speedup improvement: +10-20% on top of base speculative decoding.

## When to Use

**Ideal:**
- Single-user or small-batch (1-4 requests)
- Long-form generation (100+ tokens)
- Significant size difference between models (at least 5-10x)
- Predictable outputs (code, factual QA, structured data)

**Do NOT use:**
- High-throughput batch serving (batch > 8-16)
- Memory-constrained environments
- Very short outputs (<20 tokens)
- Mismatched vocabularies (different tokenizers)

**Real-world speedups:**
- Code generation: 2-3x
- Chat/conversational: 1.5-2.5x
- Creative writing: 1.3-1.8x
- Structured data extraction: 2.5-3.5x

## Why Batching Cancels Speculation

The rule "do not use above batch 8 to 16" is the most surprising line in this lesson, and the reason behind it explains the whole technique.

Speculative decoding is free because decode is memory-bound. Generating one token at batch size 1 reads the entire KV-cache and performs almost no arithmetic, so the GPU's compute units sit idle waiting on memory. Verifying four draft tokens in that same pass costs almost nothing extra: the memory read already happened, and the extra matrix work fits in capacity that was going to waste. You are spending idle compute to buy latency.

Batching spends that same idle compute. Each additional concurrent request amortizes the weight read across more tokens, which is exactly what raises arithmetic intensity. Somewhere around batch 8 to 16 the workload crosses from memory-bound to compute-bound, and at that point the spare capacity speculation was living on is gone. Verification now competes with real requests for the same units, rejected tokens become pure waste, and aggregate throughput falls.

So the two techniques draw on one budget. A service optimizing per-token latency for a handful of interactive users should speculate. A service maximizing tokens per dollar across a full queue should batch. Running both hard is a configuration that looks thorough and performs worse than either alone.

Two consequences for capacity planning. The acceptance rate is the number to watch, not the speedup: below about 70% the rejected work outweighs the saving, and a draft model that was fine on English prose can fall below that on code or on another language without anything else changing. And speculation raises cost per token even when it lowers latency, because rejected tokens were computed and thrown away. On self-hosted hardware that shows up as reduced concurrency rather than as a line on an invoice, which makes it easy to miss.

## Trade-offs

**Memory vs Speed:**
Draft adds 10-30% to memory usage. In tight constraints, quantizing the target may be more beneficial.

**Latency vs Throughput:**
Optimizes per-request latency, may reduce aggregate throughput in batch serving.

**Quality guarantees:**
Mathematically preserves output distribution: identical quality is guaranteed.

**System constraints:**
Tokenizer compatibility (a shared vocabulary, or a configured vocabulary intersection), hardware requirements (additional memory), implementation complexity.

## Practice in vLLM

Current vLLM releases take every speculative setting in one JSON object, `--speculative-config` on the command line and `speculative_config` in the Python API. The older `--speculative-model` and `--num-speculative-tokens` flags belong to the pre-V1 engine.

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --speculative-config '{
    "method": "draft_model",
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "num_speculative_tokens": 4,
    "draft_tensor_parallel_size": 1
  }'
```

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "method": "draft_model",
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "num_speculative_tokens": 4,
    },
)
outputs = llm.generate(["Explain speculative decoding."], SamplingParams(max_tokens=256))
```

Key parameters:
- `method`: `draft_model` for a separate small model; `eagle3`, `mtp`, `ngram`, and `suffix` select the draftless or head-based methods covered above
- `model`: the draft model or EAGLE head; it should share the target's tokenizer, and a mismatched vocabulary needs `use_heterogeneous_vocab`, which only `draft_model` supports
- `num_speculative_tokens`: 3-5 is the usual range, determined experimentally
- `draft_tensor_parallel_size`: usually 1

Monitoring: the `/metrics` endpoint exports the `vllm:spec_decode_num_draft_tokens_total` and `vllm:spec_decode_num_accepted_tokens_total` counters, and their ratio is the acceptance rate (above 70% is good for a draft model). Starting the server with `--per-request-spec-decode-metrics summary` also adds `mean_acceptance_length` and `draft_acceptance_rate` to each response, an experimental field whose shape may change between releases.

## Key Takeaways

Mathematical precision: output is identical to the target model, not an approximation.

Draft model selection is critical: acceptance rate determines speedup. EAGLE-2 leads with context-aware dynamic trees and ~3-4x speedup.

Draftless methods (Jacobi, lookahead decoding) eliminate the draft model entirely. Token recycling reuses rejected tokens as free draft candidates.

Optimal for specific workloads: single-user, long-form. Large batch does not benefit, and the reason is that speculation and batching spend the same resource. Speculation is cheap only while decode is memory-bound and compute sits idle; batching raises arithmetic intensity until that idle compute is gone, somewhere around batch 8 to 16. Running both hard performs worse than either alone.

Watch the acceptance rate rather than the advertised speedup. Below about 70% the rejected work outweighs the saving, and a draft model that holds up on English prose can fall under that on code or another language with nothing else changed.

Speculation raises cost per token even while lowering latency, since rejected tokens were computed and discarded. On self-hosted hardware that surfaces as reduced concurrency rather than as an invoice line, which is why it goes unnoticed.

Trade-off: memory for latency: 10-30% memory overhead for 1.5-3x latency reduction.

Production-ready in vLLM: simple activation, built-in monitoring.
