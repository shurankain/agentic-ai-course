## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[04_RLHF_and_Alignment|RLHF and Alignment]]
**Next:** [[06_Test_Time_Compute|Test-Time Compute]]

---

# RLHF Alternatives: DPO, KTO, ORPO, and SimPO

## Introduction: Evolution of Alignment Methods

RLHF became an industry standard but is complex to implement, requires training a separate reward model, is unstable, and expensive. This led to the emergence of alternative methods.

By 2024, DPO became the de facto standard, KTO demonstrated that paired comparisons are not necessary, and ORPO unified SFT and alignment into a single step. In 2025, GRPO (from DeepSeek R1) demonstrated that even PPO-style RL can be dramatically simplified by removing both the reward model and value model, while OpenAI's Reinforcement Fine-Tuning (RFT) made RL-style training accessible via API.

## Problems of Classical RLHF

**PPO Instability:** sensitive to hyperparameters, reward hacking, requires careful KL divergence tuning.

**Computational cost:** 4 models in memory (policy, reference, reward, value) — ~4× compute compared to SFT.

**Reward model bottleneck:** alignment quality is limited by reward model quality.

**Implementation complexity:** requires RL expertise, many moving parts.

## DPO: Direct Preference Optimization

DPO reformulates RLHF as a supervised learning task, eliminating the need for an explicit reward model and RL.

Key insight: the reward function can be mathematically expressed through the ratio of the policy to the reference policy. When substituted into the Bradley-Terry model, the normalization constant cancels out.

Final DPO loss function: negative log-likelihood that the preferred response is actually preferred. Compares the log-probability ratio of the current and reference policy for preferred and dispreferred responses.

**Advantages:** simplicity (single stage), stability (supervised learning), efficiency (2 models instead of 4).

**Limitations:** reference model dependency, requires paired data, distribution shift.

**Practical recommendations:** β = 0.1 for aggressive optimization, β = 0.5 for conservative. Minimum 10K preference pairs.

## KTO: Kahneman-Tversky Optimization

KTO uses an idea from behavioral economics: people perceive gains and losses asymmetrically (loss aversion).

Works with individual examples labeled as "good" or "bad", without requiring paired comparisons.

KTO loss is asymmetric: for "good" examples it penalizes underestimation, for "bad" ones it penalizes overestimation (with greater weight).

**Advantages:** binary feedback (like/dislike), robust to imbalanced data, comparable to DPO across all scales.

**When to use:** only binary feedback is available, data is highly imbalanced, no resources for collecting paired preferences.

## ORPO: Odds Ratio Preference Optimization

ORPO eliminates the need for a reference model by combining SFT and preference optimization.

ORPO loss: a combination of SFT loss and odds ratio preference loss. Using odds ratio instead of probability ratio naturally penalizes high probability of rejected responses.

**Advantages:** single stage, no reference model (less memory), most efficient method, comparable to DPO.

**Limitations:** less control over individual stages, less studied than DPO.

## SimPO: Simple Preference Optimization

SimPO simplifies DPO by eliminating the reference model and adding length normalization.

**Differences from DPO:**
1. No reference model
2. Length normalization (divide by response length)
3. Margin γ for separation

**Advantages:** maximum simplicity, no reference model, cost-effective, plug-and-play replacement for DPO.

## GRPO: Group Relative Policy Optimization

GRPO (DeepSeek, January 2025) takes a fundamentally different approach from DPO-family methods. Instead of learning from static preference datasets, GRPO is an online RL method — but dramatically simpler than PPO.

**How it works:** For each prompt, sample K outputs (typically K=8-64) from the current policy. Compute a reward for each output using a verifier (rule-based for math, LLM judge for open-ended). Compute group-relative advantages: Advantage_i = (reward_i - mean) / std. Use these advantages to update the policy with a clipped objective (similar to PPO) plus a KL penalty to the reference policy.

**Key innovation:** The group statistics (mean, std of rewards within the sampled group) replace the value/critic model that PPO requires. No separate reward model training is needed — any verifier or scoring function works.

**Advantages:** no reward model, no value model (2 fewer models than PPO), works with any reward signal (rule-based, LLM judge, human), online learning (generates fresh data each iteration), enabled the R1-Zero breakthrough in emergent reasoning.

**Limitations:** requires a reliable reward/verifier signal, higher inference cost during training (sampling K outputs per prompt), less studied outside reasoning tasks.

**When to use:** reasoning tasks with verifiable answers (math, code), when you have a reliable automated verifier, when you want RL-style optimization without PPO complexity.

## Online DPO and Iterative Methods

Standard DPO trains on a static preference dataset collected once. This creates a distribution shift problem: as the model improves, the preference data (generated by the old model) becomes less informative.

**Online DPO** addresses this by generating new responses from the current policy during training, collecting preferences on these fresh outputs, and updating the model iteratively. Each round, the model generates responses → preferences are collected (by humans or AI) → DPO training → repeat.

**Variants:** Online IPO (uses IPO loss instead of DPO), Self-Play Preference Optimization (SPPO, the model generates both responses and judges), iterative DPO with periodic data refresh.

**OpenAI Reinforcement Fine-Tuning (RFT):** OpenAI's API offering (2025) that provides RL-style fine-tuning without requiring users to manage training infrastructure. Users provide a grading function (automated verifier for their domain), and OpenAI applies RL optimization to their models. Particularly effective for domain-specific reasoning: legal analysis, scientific reasoning, medical diagnosis. RFT bridges the gap between simple SFT (easy but limited) and full RLHF (powerful but complex).

## Comparative Analysis of Methods

| Method | Reference Model | Paired Data | SFT Stage | Memory | Complexity |
|--------|-----------------|-------------|-----------|--------|------------|
| RLHF   | ✓ (+ reward + value) | ✓      | ✓         | 4x     | High       |
| DPO    | ✓               | ✓           | ✓         | 2x     | Medium     |
| KTO    | ✓               | ✗           | ✓         | 2x     | Medium     |
| ORPO   | ✗               | ✓           | ✗         | 1x     | Low        |
| SimPO  | ✗               | ✓           | ✓         | 1x     | Low        |
| GRPO   | ✓               | ✗ (online)  | Optional  | 2x     | Medium     |

### Performance

RLHF ≈ DPO when properly tuned.
KTO ≈ DPO even without paired data.
ORPO ≈ DPO at lower cost.
SimPO ≥ DPO on many benchmarks.
GRPO excels on reasoning tasks — enabled R1-level performance.

### Decision Framework

Start by analyzing the task and available data.

Reasoning tasks with verifiable answers (math, code) → GRPO (with rule-based verifiers).
Paired comparisons available → DPO/SimPO/ORPO.
Binary feedback only → KTO.
Need API-based RL without infrastructure → OpenAI RFT.

For paired data: need control → DPO/SimPO. Maximum efficiency → ORPO.

Production deployment: start with DPO (most proven), then experiment with SimPO, ORPO, or GRPO depending on the task.

## Hugging Face TRL Integration

TRL provides a unified API for all methods. Specialized trainers for DPO, KTO, and ORPO. SimPO is implemented via DPOTrainer with parameters.

**Data format:**
- DPO/ORPO/SimPO: prompt, chosen, rejected
- KTO: prompt, completion, binary label (True/False)

## Practical Recommendations

**Production deployment:** start with DPO; if efficiency is needed → ORPO; if only binary feedback is available → KTO.

**Research:** compare multiple methods, SimPO as a baseline for simplicity.

**Common mistakes:** insufficient data (minimum 10K), poor quality, incorrect β, ignoring SFT.

**Iterative approach:** SFT on high-quality data → evaluation → targeted preference data → preference optimization → repeat.

## Key Takeaways

DPO is a practical replacement for RLHF with comparable quality and lower complexity.

KTO works without paired comparisons — critical for binary feedback.

ORPO combines SFT and alignment — maximum efficiency.

SimPO removes the reference model — simplifies deployment.

GRPO eliminates the reward and value models from RL — enabled DeepSeek R1's reasoning breakthrough and showed that reasoning can emerge from pure RL.

Online DPO and OpenAI RFT address the distribution shift problem and make RL-style training more accessible.

Method choice depends on data, task type, and constraints — GRPO for reasoning with verifiers, DPO-family for general preferences.

TRL provides a unified API — easy to experiment.

Data quality matters more than method choice — all methods require high-quality signals (preferences or reward functions).

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[04_RLHF_and_Alignment|RLHF and Alignment]]
**Next:** [[06_Test_Time_Compute|Test-Time Compute]]
