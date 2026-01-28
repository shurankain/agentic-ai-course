## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[04_RLHF_and_Alignment|RLHF and Alignment]]
**Next:** [[06_Test_Time_Compute|Test-Time Compute]]

---

# RLHF Alternatives: DPO, KTO, ORPO, and SimPO

## Introduction: Evolution of Alignment Methods

RLHF became an industry standard but is complex to implement, requires training a separate reward model, is unstable, and expensive. This led to the emergence of alternative methods.

By 2024, DPO became the de facto standard, KTO demonstrated that paired comparisons are not necessary, and ORPO unified SFT and alignment into a single step.

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

## Comparative Analysis of Methods

| Method | Reference Model | Paired Data | SFT Stage | Memory | Complexity |
|--------|-----------------|-------------|-----------|--------|------------|
| RLHF   | ✓ (+ reward)    | ✓           | ✓         | 4x     | High       |
| DPO    | ✓               | ✓           | ✓         | 2x     | Medium     |
| KTO    | ✓               | ✗           | ✓         | 2x     | Medium     |
| ORPO   | ✗               | ✓           | ✗         | 1x     | Low        |
| SimPO  | ✗               | ✓           | ✓         | 1x     | Low        |

### Performance

RLHF ≈ DPO when properly tuned.
KTO ≈ DPO even without paired data.
ORPO ≈ DPO at lower cost.
SimPO ≥ DPO on many benchmarks.

### Decision Framework

Start by analyzing available data. Paired comparisons → wide range of options. Binary feedback → KTO.

For paired data: need control → DPO/SimPO. Maximum efficiency → ORPO.

Production deployment: start with DPO (most proven), then experiment with SimPO or ORPO.

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

Method choice depends on data and constraints — there is no universally best option.

TRL provides a unified API — easy to experiment.

Data quality matters more than method choice — all methods require high-quality preferences.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[04_RLHF_and_Alignment|RLHF and Alignment]]
**Next:** [[06_Test_Time_Compute|Test-Time Compute]]
