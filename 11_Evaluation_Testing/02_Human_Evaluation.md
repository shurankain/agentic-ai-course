# Human Evaluation of LLMs

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Evaluation and Testing
**Previous:** [[01_Metrics_and_Benchmarks|Metrics and Benchmarks]]
**Next:** [[03_LLM_as_Judge|LLM as Judge]]

---

## The Gold Standard of Evaluation

The end consumer of model outputs is a human. No metric can fully model human perception of usefulness, appropriateness, and quality.

Automated metrics measure proxy characteristics: word overlap, semantic vector similarity. Humans evaluate the holistic impression: whether the answer helped solve the task, whether it was clear, whether it inspired trust. These aspects cannot be reduced to numerical formulas.

Automated metrics require calibration. What does a BERTScore of 0.85 mean? The only way to interpret numbers is to correlate them with human judgments. Human evaluation is the reference point for all other metrics.

**Limitations:** expensive, slow, subjective, scales poorly. The art of building an evaluation system lies in finding the right balance between human and automated evaluation.

## Task Design

The quality of human evaluation critically depends on task formulation. A poorly designed task yields inconsistent, unreliable results.

**Clarity of criteria** — the evaluator must understand exactly what is required. Bad: "Rate the quality of the answer". Better: "Rate the completeness of the answer on a scale of 1-5, where 1 means the answer contains none of the requested information, and 5 means it contains all necessary information without omissions".

**Examples for each scale point** significantly improve consistency. Show what a 2-point answer looks like, what a 4-point answer looks like. This calibrates understanding of the scale.

**Avoid ambiguity** — break down a "good" answer into specific dimensions: correctness, completeness, clarity, usefulness.

**Cognitive load** — an evaluator with 15 criteria will make errors. Break complex evaluation into simple tasks or use a hierarchy.

## Rating Scales

**Binary** (yes/no) — easy to explain, high agreement. But loses nuance: the difference between "almost good" and "completely bad" is erased.

**5-point** — a popular trade-off. Sufficient granularity without excessive complexity.

**Likert with anchors** — "1 — Strongly disagree, 2 — Disagree, 3 — Neutral, 4 — Agree, 5 — Strongly agree" is more reliable than bare numbers 1-5.

**Continuous** (slider 0-100) — maximum granularity, but requires more cognitive effort and may reduce agreement.

**Best-worst scaling** — selecting the best and worst answer from a set. More reliable than absolute ratings, but requires more examples.

## Pairwise Comparison

Comparing pairs instead of assigning absolute scores has significant advantages.

**Easier for the evaluator** — "which answer is better?" is more natural than assigning a numerical rating. People handle comparative judgments better.

**Higher inter-annotator agreement** — evaluators agree more often when comparing pairs than when assigning absolute ratings.

**Less sensitive to bias** — different experts have different "baselines" in absolute ratings. Pairwise comparison neutralizes this difference.

**Elo rating** and similar systems allow building a global ranking based on pairwise comparisons.

**Limitation:** quadratic growth in the number of pairs. For N answers, N(N-1)/2 comparisons are needed. Solutions: sampling, tournament-style evaluations.

## Selecting Evaluators

**Domain experts** provide expert assessment of content. Medical Q&A requires physicians. Limitations: expensive, slow, small pool, specific biases.

**Crowd workers** (MTurk, Prolific, Appen) provide scale: hundreds of evaluations within hours at low cost. Quality is variable; control mechanisms are needed: qualification tests, attention checks, rejection criteria, agreement thresholds.

**Target users** — sometimes the best evaluators. If the model is for students, students will assess clarity better than professors.

**Combined approach:** domain experts for correctness, crowd workers for fluency and helpfulness, target users for real-world usefulness.

## Number of Evaluators

**Minimum three** evaluators per example with majority voting or averaging.

**Inter-annotator agreement (IAA)** measures consistency: Cohen's kappa, Krippendorff's alpha, ICC. Low agreement indicates a difficult task, poor instructions, or underqualified evaluators.

**Power analysis** determines the required sample size to detect an effect of a given magnitude. The smaller the expected difference, the more examples are needed.

**In practice:** for rough evaluation, 50-100 examples × 3 evaluations. For fine-grained comparisons (5-10% difference) — hundreds of examples. For statistical rigor — consult a statistician.

## Bias in Human Evaluation

**Anchoring** — early examples influence the perception of subsequent ones.

**Order effects** — presentation order affects ratings. Randomize the A/B order.

**Length bias** — longer answers are perceived as more complete, even if the extra length is filler.

**Positional bias** — in a list of options, the first and last items attract more attention.

**Fatigue effects** — evaluation quality decreases over time. Long sessions produce less reliable results.

**Mitigation:** order randomization, condition balancing, session length limits, calibration examples, blind evaluation.

## Process Organization

**Preparation:** clear instructions, calibration examples, convenient interface.

**Pilot study** — testing on a small sample reveals problems with instructions and the interface.

**Evaluator training:** task explanation, example walkthrough, calibration session with discussion of disagreements.

**Monitoring:** tracking IAA in real time, checking attention checks, identifying outliers.

**Debriefing:** feedback from evaluators about problems and ambiguities.

**Documentation:** recording decisions, versioning instructions, preserving raw data.

## When to Use Human Evaluation

**Metric development** — calibrating automated metrics requires ground truth from humans. Conduct it once, then use the calibrated automated metrics.

**Critical decisions** — choosing between fundamentally different approaches. Automated metrics may miss important nuances.

**Novel tasks** — for non-standard tasks, no ready-made benchmarks exist.

**Final validation** — before a production release, verify that automated metrics are not misleading.

**Problem investigation** — when users complain but automated metrics look good.

## Practical Implementation

The campaign management service includes creating a campaign with configuration validation, forming tasks from examples, adding attention checks at a specified ratio, selecting qualified evaluators, distributing tasks, and notifying participants.

When retrieving the next task, the evaluator's qualifications are verified, tasks the evaluator has not yet completed are filtered, and assignment is taken into account.

Submitting an evaluation includes validation, checking completion time (excessively fast completion is rejected), and verifying attention checks with possible evaluator blocking upon systematic failures.

Result analysis aggregates evaluations by task. For numerical ratings, the mean, median, and standard deviation are computed. For pairwise comparisons — majority voting with confidence as the proportion of votes for the winner.

Inter-annotator agreement is computed via Krippendorff's alpha: observed disagreement and expected disagreement are calculated, alpha = 1 - observed/expected. Interpretation: >=0.8 — strong agreement, >=0.6 — moderate, >=0.4 — fair, otherwise poor.

The pairwise comparison system implements Elo rating. When creating a pair, the order is randomized to eliminate positional bias. When recording a result, the order is restored, and Elo ratings are updated using the formula: new rating = old + K × (actual outcome - expected outcome). The expected outcome is computed via the formula 1 / (1 + 10^((opponent_rating - own_rating)/400)).

Statistical significance is verified through a binomial test on direct comparisons between systems. With fewer than 10 direct matches — insufficient data. A p-value < 0.05 is considered significant.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Evaluation and Testing
**Previous:** [[01_Metrics_and_Benchmarks|Metrics and Benchmarks]]
**Next:** [[03_LLM_as_Judge|LLM as Judge]]
