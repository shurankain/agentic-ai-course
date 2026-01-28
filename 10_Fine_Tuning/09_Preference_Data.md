# Preference Data: The Foundation of Alignment

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[08_Continued_Pretraining|Continued Pretraining]]
**Next:** [[10_DoRA_and_Beyond|DoRA and Advanced PEFT Methods]]

---

RLHF and DPO rely on preference data — comparisons of "A is better than B". The quality of this data directly determines the quality of the trained model.

## Bradley-Terry Model: The Mathematics of Preferences

The Bradley-Terry model is a classical mathematical framework for modeling pairwise comparisons. The idea: each option has a latent "strength", and the probability of selection is determined by the ratio of strengths.

Probability of preferring A over B = sigma(r_A - r_B), where r is log-strength, sigma is the sigmoid function.

**Connection to RLHF and DPO:**
- The Reward Model in RLHF uses Bradley-Terry likelihood for training
- DPO implicitly uses Bradley-Terry, but without explicitly training a reward model

**Model assumptions:**
- Transitivity: A > B and B > C → A > C
- Independence: preference of A vs B does not depend on other items
- Stability: item strength does not change

In practice, these assumptions are violated.

## Elo Ratings for Models and Responses

**Elo in the LLM arena:** Chatbot Arena by LMSYS uses Elo for its leaderboard. Two responses from different models are shown for a single prompt, the annotator selects the better one, and ratings are updated.

Update formula: new rating = old + K × (actual outcome - expected outcome). K (typically 16-32) determines the rate of rating change.

**Elo for preference data curation:** can be applied for efficient curation within a single model. Instead of N² comparisons for full ranking, O(N log N) strategically selected comparisons suffice.

Advantages: dramatically more efficient, provides full ranking, robust to noise, adaptive pair selection.

## Collecting Preference Data: Practice

**Sources:**
- Human annotation (gold standard) — professional annotators, crowdsourcing, domain experts
- AI evaluation (RLAIF) — a strong model as judge, cheaper and more scalable
- User feedback (implicit) — thumbs up/down, regeneration requests, response copying
- Hybrid — AI for initial filtering, human for final quality check

**Data format:**
- Pairwise comparisons: prompt, chosen, rejected
- Rankings: ranking of N responses
- Ratings: numerical score (1-5) by criteria

**Inter-Annotator Agreement:**

Cohen's Kappa measures agreement adjusted for chance. Interpretation: <0.2 — poor, 0.2-0.4 — fair, 0.4-0.6 — moderate, 0.6-0.8 — substantial, >0.8 — almost perfect.

Typical values: simple tasks >0.7, complex 0.4-0.6, subjective 0.3-0.5.

## Active Learning for Preferences

Collecting preference data is expensive. Active learning selects the most informative comparisons.

**Strategies:**
- Uncertainty sampling — prioritize pairs where the model is least confident
- Diversity sampling — cover diverse regions of the space
- Expected model improvement — select pairs with the maximum expected improvement

**Batch Active Learning:** a combined score balancing importance (uncertainty) and diversity. Diversity weight — 0.2-0.4.

## Data Quality and Its Impact

**Garbage in, garbage out:** 10% noise → noticeable degradation, 30% noise → no better than random, systematic bias is worse than random noise.

**Types of problems:**
- Label noise — random annotator errors
- Systematic bias — preference for longer responses, confident tone, cultural/demographic bias
- Distribution mismatch — preference data does not cover production queries

**Diagnostics:** consistency analysis, analysis of difficult examples, checking for systematic bias.

**Improving quality:** better instructions, annotator calibration, quality control (check questions, agreement thresholds), data cleaning.

## RLAIF: Synthetic Preferences

**When to use AI evaluation:** scale, speed, consistency, cost.

**When human evaluation is necessary:** subjective preferences, safety-critical domains, novel/edge cases, final validation.

**Best practices:**
- Model selection — the judge must be significantly more powerful than the student model
- Prompting for consistency — clear criteria, structured output
- Reducing position bias — compare twice with reversed order

## Key Takeaways

The Bradley-Terry model is the mathematical foundation of preference learning.

Elo ratings are effective for ranking responses and prioritizing annotation. O(N log N) instead of O(N²).

Inter-annotator agreement is a critical data quality metric. κ < 0.4 is a serious problem.

Active learning maximizes the informativeness of each comparison.

Quality > quantity: 10K clean examples are better than 100K noisy ones.

Systematic bias is more dangerous than noise: length bias, verbosity preference, sycophancy.

RLAIF is scalable but requires careful prompting and validation on human data.

A hybrid approach is optimal: AI for speed and scale, human for quality and edge cases.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[08_Continued_Pretraining|Continued Pretraining]]
**Next:** [[10_DoRA_and_Beyond|DoRA and Advanced PEFT Methods]]
