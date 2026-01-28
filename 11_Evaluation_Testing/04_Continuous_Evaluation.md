# Continuous Evaluation and A/B Testing

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Evaluation and Testing
**Previous:** [[03_LLM_as_Judge|LLM as Judge]]
**Next:** [[05_RAG_Evaluation|RAG Evaluation]]

---

## From One-Time Evaluation to Continuous Monitoring

One-time evaluation before release is necessary but insufficient. A model can degrade in production due to input data drift, changing user expectations, and external dependency updates.

Continuous evaluation transforms evaluation from a point-in-time event into an ongoing process. Every request is a potential source of quality information. Every system change is a reason for regression testing.

**Goals:** early detection of problems before mass user impact, tracking long-term quality trends, validating hypotheses through A/B tests, accumulating data for training and improvement.

## System Architecture

**Sampling layer** — a mechanism for selecting requests for evaluation. You cannot evaluate every request (too expensive), and you cannot evaluate none. The sampling strategy determines which requests go to evaluation.

**Evaluation pipeline** — the actual evaluation of selected requests: automated metrics, LLM judge, queue for human evaluation.

**Storage layer** — storing results with metadata: timestamp, model version, prompt version, user segment.

**Analysis layer** — aggregation, visualization, statistical analysis. Dashboards, alerts, reports.

**Feedback loop** — using results for improvement: data for fine-tuning, identifying problematic patterns, informing prompt decisions.

## Sampling Strategies

**Random sampling** — randomly selecting X% of traffic. Ensures representativeness but may miss rare important cases.

**Stratified sampling** — selection accounting for categories. If 80% of traffic is simple questions and 20% is complex, random sampling yields few complex examples. Stratified sampling ensures proportional or balanced representation.

**Importance sampling** — prioritizing "important" requests by category (production-critical), user characteristics (enterprise), or request features (long, contains specific words).

**Error-triggered sampling** — increased selection when a problem is suspected: user complaint, latency increase, suspiciously short response.

**Coverage sampling** — targeted selection to fill gaps. If certain request types are underrepresented in evaluation data, their sampling rate is increased.

## A/B Testing

The gold standard for comparing variants in production. Users are randomly divided into groups, each receives its own variant, and metrics are compared.

**Why A/B is better than offline evaluation:** offline tests use historical data and simulations. They do not account for all aspects of real usage: how users react, what follow-up questions they ask, whether the change drives engagement.

**The unit of randomization matters.** Randomizing by request can result in one user seeing different variants within a session, which is confusing. Randomizing by session or user ensures a consistent experience.

**Sample size** determines statistical power. The smaller the expected effect, the more data is needed. A power calculator helps determine the required size before starting the test.

**Duration** must cover natural variability: day of week, time of day, seasonality. A short test yields misleading results.

## Metrics for A/B Tests

**Guardrail metrics** — must not deteriorate: latency, error rate, safety violations. If an experiment improves the target metric but violates guardrail criteria, the experiment is unsuccessful.

**Primary metrics** — the main target metrics of the experiment: user satisfaction, task completion, quality scores.

**Secondary metrics** — additional metrics for understanding the effect. They do not directly determine the decision but help interpret results.

**Relative vs absolute:** "proportion of successful conversations" (relative) vs "number of successful conversations" (absolute). Relative metrics are more interpretable and less sensitive to traffic changes.

## Statistical Significance

**p-value** shows the probability of observing such an extreme or more extreme result given no real difference exists. p < 0.05 is traditionally considered significant (a convention, not a law). We test the null hypothesis (H₀: no difference) against the alternative (H₁: there is a difference).

**Confidence intervals** provide a range of plausible values for the true effect. "Effect of 5% with a 95% confidence interval [2%, 8%]" is more informative than simply "p < 0.05" — it shows not only significance but also magnitude and uncertainty.

**Computing confidence intervals:**

Bootstrapping (recommended for most cases) — resample data with replacement multiple times, compute the metric for each sample, take the 2.5% and 97.5% percentiles of the distribution.

Paired Bootstrap for A/B comparisons — resample pairs (control, treatment), compute the mean difference for each iteration, build the distribution of differences. If the confidence interval does not contain 0, the difference is significant.

Wilcoxon Signed-Rank Test for paired data with non-normal distribution — compute differences between pairs of scores, rank the absolute values of differences, sum the ranks of positive differences, compute the p-value.

**The multiple comparisons problem:** with 20 metrics and a significance level of 0.05, one false positive finding can be expected by chance.

Bonferroni correction — divides the significance level by the number of tests (α/m), but is overly strict.

Holm-Bonferroni correction — less conservative: sorts p-values, applies sequentially decreasing thresholds.

Benjamini-Hochberg FDR method — controls the false discovery rate: sorts p-values, finds the maximum index where p-value ≤ (index/number_of_tests) × α, rejects hypotheses for all smaller indices.

**Practical significance** differs from statistical significance. A statistically significant improvement of 0.1% may not be worth the complexity.

**Effect size** helps assess practical importance. Cohen's d shows the standardized difference: a small effect (~0.2) is statistically significant but practically unnoticeable; medium (~0.5) is a moderate improvement; large (≥0.8) is a substantial change that users will notice.

## Regression Testing in CI/CD

**Golden dataset** — a set of examples with known correct answers. With every change, the golden dataset is run and compared against the baseline.

**Regression thresholds** — acceptable degradation. "Accuracy ≥95% of the baseline level" or "Latency must not increase >10%".

**Blocking vs warning** — serious regressions block deployment, minor ones generate warnings.

**Incremental testing** — for major changes, first test on a portion of traffic (canary deployment), then gradually expand.

## Monitoring and Alerting

**Quality metrics** — regular sampling and evaluation of response quality. Trends, anomaly detection.

**Latency metrics** — response time, broken down by component (model, retrieval, post-processing).

**Error metrics** — types, frequency, correlations with other factors.

**User signals** — explicit feedback (ratings, complaints), implicit signals (abandonment, reformulations).

**Drift detection** — changes in the distribution of input data. If request characteristics change, the model may perform worse.

**Alerting rules** balance sensitivity and specificity. Overly sensitive rules create alert fatigue, insufficiently sensitive ones miss problems.

## Feedback Loops

**Collecting training data** — examples with low scores and corrections become training data.

**Prompt optimization** — error analysis informs prompt changes.

**Model selection** — comparative data helps choose between models.

**Retrieval optimization** — in RAG systems, faithfulness analysis informs retrieval improvements.

**User experience** — understanding failure modes helps improve UX (better fallback options, clear limitations).

## Organizational Aspects

**Ownership** — who is responsible for the evaluation system? A dedicated team or part of the platform team?

**Processes** — how do results influence decisions? Regular review meetings, integration into planning.

**Culture** — evaluation as assistance, not policing. The goal is improvement, not blame.

**Resources** — evaluation requires compute resources, storage, and possibly human evaluators. Budget accordingly.

## Practical Implementation

Experiment management creates an experiment with configuration: variants, traffic allocation, primary and guardrail metrics. Before launch, power analysis is conducted to determine the required sample size based on the expected effect size, baseline metric, desired test power, and significance level.

Variant assignment is determined deterministically based on userId and experimentId through hashing. This ensures consistency — one user always sees one variant. The hash is computed as (userId + experimentId).hashCode(), converted to a 0-1 value, then compared against the cumulative distribution of variants.

Metric collection records each event tied to the experiment, variant, user, and timestamp. Data is aggregated by variant.

Results analysis compares variants on primary metrics, checks guardrail criteria, and computes statistical significance through p-values and confidence intervals. The winner is determined: if any guardrail is violated — control wins; if the primary metric is significantly better for treatment — treatment wins; if data is insufficient — the decision is deferred; if data is sufficient but there is no difference — no effect is recorded.

Continuous production evaluation asynchronously processes requests: the sampling service decides whether to evaluate a request (error-triggered, importance-based, or stratified strategies); if the request is selected — an evaluation chain is launched; the result is saved and checked for anomalies; metrics are updated in real time.

Periodic analysis (hourly) aggregates metrics over the window, compares with the previous week's baseline, computes z-scores, and sends alerts for significant deviations (z > 2.5 for WARNING, z > 3.5 for CRITICAL).

Regression testing in CI/CD on each commit loads the golden dataset, runs the model with the current configuration, generates an evaluation report, compares with the baseline, identifies regressions (degradations exceeding thresholds), and classifies them as blocking or warning. Status: FAILED if there are blocking regressions, WARNING if there are minor issues, PASSED if everything is in order. On successful merge to main, the baseline is updated.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Evaluation and Testing
**Previous:** [[03_LLM_as_Judge|LLM as Judge]]
**Next:** [[05_RAG_Evaluation|RAG Evaluation]]
