## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[05_RLHF_Alternatives|RLHF Alternatives]]
**Next:** [[07_Synthetic_Data|Synthetic Data]]

---

# Test-Time Compute: A New Dimension of Scaling

## Introduction

For decades, AI improvement followed the formula: more data, parameters, training compute. In 2024, a shift occurred: OpenAI o1 demonstrated that computation during inference dramatically improves quality.

Test-Time Compute (TTC) represents a new optimization axis. Instead of a fixed answer in a single forward pass, the model can "think longer" — generating chains of reasoning, verifying hypotheses, correcting errors.

This changes the economics of AI systems. For complex tasks, it is cheaper to use more inference compute than to train a larger model.

## OpenAI o1: A Breakthrough in Reasoning

o1 (September 2024) demonstrated results that seemed impossible:

| Benchmark | GPT-4o | o1-preview | o1 (full) |
|-----------|--------|------------|-----------|
| AIME 2024 | 13.4% | 74.4% | 83.3% |
| Codeforces | 11th percentile | 89th | 93rd |

Architecture: user prompt → hidden chain of reasoning (hundreds/thousands of tokens) → final answer.

Key features: extended thinking, self-correction, verification, backtracking.

o3 (December 2024) went further. o4-mini demonstrates that even small models with sufficient TTC outperform larger ones.

## Scaling Laws for Test-Time Compute

Traditionally: Quality ∝ (Training Compute)^α, where α ≈ 0.05-0.1.

TTC introduces a new dimension: Quality ∝ (Inference Compute)^β, and for complex tasks β can be higher than α.

Trade-off: for a fixed budget C_total = C_train + N × C_inference, one must decide how much to allocate to training and how much to inference reasoning.

The optimum depends on task complexity, number of requests, and the current model level.

| Scenario | Preference | Why |
|----------|------------|-----|
| Math/Logic | TTC | Reasoning helps more than knowledge |
| Factual QA | Training | Knowledge matters more than reasoning |
| Coding | TTC | Debugging, planning are critical |
| Translation | Training | Pattern matching, not reasoning |

## Test-Time Compute Mechanisms

**Process Reward Models (PRMs)** evaluate the quality of each intermediate reasoning step. Advantages: early error detection, search guidance, informative feedback. Training through Monte Carlo estimation or synthetic step labels.

**Best-of-N Sampling** — generate N answers, select the best one via a verifier or self-consistency (majority voting). Trade-offs: linear compute growth, diminishing returns after N ≈ 64-256, requires a good verifier.

**Self-Consistency** — for tasks with a definite answer, generate N chains of reasoning, extract the final answer from each, select the most frequent one. Works when errors are random, not systematic.

**Tree Search** — build a tree of reasoning paths, explore intelligently via beam search, MCTS, or best-first. Advantages over Best-of-N: uses compute more efficiently, can recover from mistakes. Disadvantages: harder to implement, requires a quality value function.

**Budget Forcing** — the model adapts reasoning depth based on compute budget. OpenAI o1 API: reasoning_effort ("low", "medium", "high"). With "low" — minimal tokens (fast, cheap); with "high" — thousands of reasoning tokens (slow, high quality).

## DeepSeek-R1: Open-Source Reasoning

DeepSeek-R1 (January 2025) demonstrated that reasoning can be achieved through pure RL without supervised fine-tuning on reasoning traces.

Results: AIME 15.6% → 71.0%, competitive with o1 at lower cost.

**R1-Zero** — the model independently "discovered" chain-of-thought reasoning without demonstrations. A fundamental result: reasoning emergence through proper incentives.

**Distillation** — reasoning capabilities can be distilled into smaller models. Even a 7B model shows impressive reasoning.

## Compute-Optimal Inference Strategies

TTC follows predictable laws: Quality(C) = a × log(C) + b. Doubling inference compute yields a constant absolute improvement.

**When TTC is more efficient than training:**
1. The task is complex (math, coding, reasoning)
2. A correct answer is required
3. A good verifier is available
4. Inference is sporadic

**When training is more efficient:**
1. The task is simple
2. High latency is unacceptable
3. Massive scale (millions of requests)
4. No good verifier is available

**Optimal strategy can yield 4× improvement:** PRM for early pruning, adaptive budget, diverse generation, early termination, progressive refinement.

**Smaller Models + TTC vs Larger Models:** A 7B model with optimal TTC can demonstrate quality comparable to 70B in single-pass on reasoning tasks. Cost arbitrage, latency vs quality, democratization of reasoning capabilities.

## Practical Implications

**System Architecture:**
- Request Router — classification by complexity
- Simple Path (70% of requests) — single call
- Complex Path — Best-of-N or Tree Search
- Caching Layer — caching expensive computations
- Monitoring — cost, latency, quality, distribution

**Cost Modeling:**
- Easy: 1× base cost
- Medium: 4× base cost
- Hard: 16× base cost
- Extreme: 64× base cost

**Latency vs Quality:** streaming partial results, parallel sampling, early termination, async processing, progressive disclosure.

## Key Takeaways

TTC is a new scaling dimension alongside parameters, data, and training compute.

o1/o3 demonstrate 5-10× improvement on reasoning through TTC.

DeepSeek-R1: reasoning is emergent through pure RL.

4× efficiency gain through optimal TTC strategies.

Smaller models + TTC compete with larger models on complex tasks.

Cost modeling becomes more complex — variable inference cost.

Architectural implications: routing, caching, async are critical.

Verifiers are critical for TTC quality.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[05_RLHF_Alternatives|RLHF Alternatives]]
**Next:** [[07_Synthetic_Data|Synthetic Data]]
