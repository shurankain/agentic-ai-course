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

## OpenAI o-Series: The Reasoning Model Family

o1 (September 2024) demonstrated results that seemed impossible:

| Benchmark | GPT-4o | o1 | o3 | o4-mini |
|-----------|--------|-----|-----|---------|
| AIME 2024 | 13.4% | 83.3% | 96.7% | 93.4% |
| Codeforces | 11th % | 93rd % | 99th % | 93rd % |
| GPQA Diamond | 53.6% | 78.0% | 87.7% | 81.4% |

Architecture: user prompt → hidden chain of reasoning (hundreds/thousands of tokens) → final answer.

Key features: extended thinking, self-correction, verification, backtracking.

**o3** (announced January 2025, initially limited release, broadly available by mid-2025) significantly improved over o1, achieving near-human performance on mathematical competition problems. **o4-mini** (April 2025) demonstrated that even small models with sufficient TTC outperform larger standard models, at a fraction of the cost.

API parameters: `reasoning_effort` ("low", "medium", "high") controls thinking depth. Higher effort = more thinking tokens = better quality but higher cost and latency.

## Claude Extended Thinking

Anthropic introduced extended thinking with Claude 3.5 Sonnet (late 2024), maturing with the Claude 4 family (2025). Unlike OpenAI's hidden reasoning, Claude's thinking is transparent.

**How it works:** The model generates explicit chain-of-thought in `<thinking>` blocks before producing the final answer. These thinking tokens are visible to the developer (though not always shown to end users by default).

**API control:** The `budget_tokens` parameter sets the maximum number of thinking tokens the model can use. Higher budgets allow more thorough reasoning. Thinking tokens are billed at a reduced rate.

**Key differences from o1/o3:**
- **Transparent thinking** — developers can see the full reasoning chain, aiding debugging and trust
- **Explicit budget control** — `budget_tokens` gives fine-grained control (vs. o-series `reasoning_effort` which is coarse: low/medium/high)
- **Same model architecture** — extended thinking is a capability of the same Claude models, not a separate model family

**Results:** Claude Sonnet 4 with extended thinking achieves competitive results with o3 on GPQA, AIME, and coding benchmarks, with the added benefit of transparent reasoning.

## Gemini Thinking

Google introduced thinking capabilities with Gemini 2.0 Flash Thinking (late 2024), expanding to Gemini 2.5 Pro and Flash (2025).

**Approach:** Gemini models generate explicit reasoning in "thought" blocks, similar to Claude's extended thinking. The thinking process is visible in the API response.

**API control:** The `thinking` configuration in the generation config enables thinking mode. Budget control is available through `thinkingBudget` parameter.

**Gemini 2.5 Pro with thinking** achieves strong results on reasoning benchmarks, leveraging Google's long-context capabilities (1M+ tokens) for reasoning over large inputs.

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

**Budget Forcing** — the model adapts reasoning depth based on compute budget. OpenAI o-series API: `reasoning_effort` ("low", "medium", "high"). Claude API: `budget_tokens` (explicit token limit for thinking). Gemini API: `thinkingBudget` (similar explicit control). With low budget — minimal tokens (fast, cheap); with high budget — thousands of reasoning tokens (slow, high quality).

## DeepSeek-R1: Open-Source Reasoning via GRPO

DeepSeek-R1 (January 2025) demonstrated that reasoning can be achieved through pure RL without supervised fine-tuning on reasoning traces.

Results: AIME 15.6% → 71.0%, competitive with o1 at lower cost. Open-weight (MIT license).

**GRPO (Group Relative Policy Optimization)** is the training method that made R1 possible. For each prompt, GRPO samples K outputs from the current policy, scores them with a verifier, then computes group-relative advantages (reward_i - group_mean) / group_std. This replaces PPO's separate value model with simple group statistics. The clipped policy gradient update with KL penalty to the reference policy completes the algorithm.

**R1-Zero** — applying GRPO with only rule-based rewards (correctness for math, format compliance) to the DeepSeek-V3-Base model. Without any SFT on reasoning traces, the model independently "discovered" chain-of-thought reasoning, self-verification, and even "aha moments." A fundamental result: reasoning emergence through proper RL incentives, not supervised demonstrations.

**R1 (full)** — R1-Zero followed by SFT on curated reasoning traces and a second round of GRPO. This produces cleaner, more readable reasoning while maintaining the emergent capabilities.

**Distillation** — reasoning capabilities can be distilled into smaller models. DeepSeek distilled R1 into 1.5B, 7B, 8B, 14B, 32B, and 70B variants, demonstrating that even small models can reason impressively when distilled from a strong reasoning teacher.

**Impact on the field:** GRPO has been widely adopted for training reasoning models. Qwen 3 (hybrid reasoning, 2025) uses a similar approach. The key insight — that reasoning emerges from proper reward signals without needing reasoning demonstrations — has reshaped how labs approach training reasoning models.

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

o3/o4-mini, Claude extended thinking, and Gemini thinking all demonstrate 5-10× improvement on reasoning through TTC.

All major providers now offer reasoning models: OpenAI (o3, o4-mini), Anthropic (Claude with extended thinking), Google (Gemini with thinking), DeepSeek (R1).

DeepSeek-R1 and GRPO: reasoning emerges through pure RL with group-relative rewards — no supervised reasoning demonstrations needed.

Transparent vs. hidden thinking: Claude and Gemini show the reasoning chain; OpenAI hides it. Transparency aids debugging and trust.

4× efficiency gain through optimal TTC strategies.

Smaller models + TTC compete with larger models on complex tasks (o4-mini vs. GPT-4o, R1-distill-7B vs. larger standard models).

Cost modeling becomes more complex — thinking tokens add variable inference cost.

Architectural implications: routing, caching, async are critical. Budget control (reasoning_effort, budget_tokens) enables cost-quality trade-offs per request.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[05_RLHF_Alternatives|RLHF Alternatives]]
**Next:** [[07_Synthetic_Data|Synthetic Data]]
