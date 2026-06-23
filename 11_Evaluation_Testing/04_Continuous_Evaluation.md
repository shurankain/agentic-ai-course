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

**Practical starting point:** Begin with random sampling at 5-10% of traffic. This is sufficient for detecting major quality degradation. Increase to 100% sampling for high-risk agent actions — financial transactions, medical recommendations, legal advice, any action with material consequences for users. The cost of evaluating every request is justified when the cost of a missed error is high.

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

## Production Evaluation Pipeline

The difference between "we evaluate" and "we have a production evaluation system" is the pipeline — an automated, continuously running process that detects quality issues before users do.

**Sampling strategy for production.** The optimal sampling combines multiple strategies: 1% random (baseline quality monitoring — cheap, continuous, representative), 100% errors (every error is evaluated to understand failure modes and detect systemic issues), 10% edge cases (queries that match known-difficult patterns — long inputs, multi-language, adversarial-looking), and 100% high-risk actions (financial transactions, medical recommendations, any action with material consequences). This gives comprehensive coverage at manageable cost: for a system processing 100K requests/day, this evaluates approximately 2,000-3,000 requests daily.

**Evaluation frequency and alert thresholds.** Real-time: error rate monitoring (alert on >2x baseline), latency monitoring (alert on P95 exceeding SLA). Hourly: aggregated quality metrics from sampled evaluations (alert on >10% quality drop vs 7-day rolling average). Daily: comprehensive quality report with breakdown by query type, model, and feature. Weekly: deep-dive analysis including LLM-as-Judge evaluation on a curated sample, trend review, and comparison with human evaluation on a 5% subset.

**From Braintrust merge-blocking gates to weekly quality reports.** Braintrust (and similar platforms like DeepEval) support merge-blocking quality gates — PRs that introduce prompt or model changes are blocked if evaluation quality drops below a configured threshold. This integrates evaluation directly into the development workflow, not just the production monitoring workflow. The pattern: developer changes prompt → CI runs evaluation suite → if score drops below baseline - N%, the PR is blocked with a clear report of which test cases regressed → developer fixes the regression before merging.

## Regression Detection After Provider Updates

One of the most insidious quality issues in LLM applications: the model provider updates the model without notice, and your system's quality changes — sometimes improving, sometimes degrading. OpenAI retired GPT-5.2 on June 12, 2026 and migrated all conversations to GPT-5.5 automatically, with no advance notice. Developers whose prompts were tuned for GPT-5.2's specific behavior patterns saw unexpected changes.

**Shadow evaluation.** Run every Nth request through both the current model and a "shadow" model (the candidate or the provider's latest version). Compare quality metrics between the two without exposing the shadow model's output to users. When the shadow model consistently outperforms the current model — upgrade with confidence. When it underperforms — stay on the current version and investigate.

**A/B quality comparison for model changes.** When a provider updates a model (GPT-4o → GPT-5, Claude Opus 4.7 → 4.8), do not switch production immediately. Run a canary: 5% of traffic on the new model, 95% on the old. Compare quality metrics over 24-48 hours. Only expand when the new model meets or exceeds the old model's quality on your specific workload. Generic benchmarks (SWE-bench, MMLU) do not predict how the model will behave on your prompts.

**What to do when model behavior changes silently.** Provider-side model updates (without version change) can alter behavior. Detection: monitor quality metrics on the golden dataset daily. If scores change significantly with no code change on your side — the provider updated something. Response: re-run the golden dataset evaluation, compare with the historical baseline, identify which test cases regressed, and file a support ticket with the provider if the regression is significant. Longer term: maintain a version-pinned model configuration where possible (many providers offer dated model versions — e.g., `claude-sonnet-4-6-20250923`) and upgrade explicitly rather than tracking the latest alias. Note: Anthropic retired `claude-sonnet-4-20250514` on June 15, 2026 with no grace period — a real-world example of why version-pinned configurations need monitoring for deprecation notices.

## Regression Testing in CI/CD

**Golden dataset** — a set of examples with known correct answers. With every change, the golden dataset is run and compared against the baseline. The golden dataset should grow over time: every production bug that is root-caused becomes a new test case. A golden dataset of 200-500 examples covering all major query types provides strong regression coverage.

**Regression thresholds** — acceptable degradation. "Accuracy ≥95% of the baseline level" or "Latency must not increase >10%".

**Blocking vs warning** — serious regressions block deployment, minor ones generate warnings. Braintrust and DeepEval support this as CI/CD integration.

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

## Simulation Testing with Synthetic Personas

Automated evaluation catches regressions, and A/B testing validates improvements. But neither systematically explores the space of possible user interactions. **Simulation testing** fills this gap by generating hundreds of synthetic multi-turn conversations to find edge cases that are impossible to discover through manual testing or random sampling.

**The pattern:**
1. Define 5-10 synthetic user personas with different interaction styles: aggressive (demanding, impatient), confused (unclear queries, frequent topic changes), adversarial (intentionally trying to break the agent), verbose (long rambling inputs), non-native speaker (grammatical errors, unusual phrasing), domain expert (uses technical jargon, tests depth)
2. For each persona, generate 20-50 multi-turn conversation scenarios covering typical workflows and known edge cases
3. Run all conversations against the agent, measuring: task completion rate per persona, guardrail trigger rate, response quality scores, failure modes

**Tools:** Maxim AI provides end-to-end simulation with synthetic personas and multi-turn trajectory testing — it simulates realistic user behavior including follow-up questions, corrections, and escalation. DeepEval (open-source) offers synthetic conversation generation with configurable personas and automated evaluation.

**When to use:** Before any major deployment or model change for user-facing agents. The investment (setting up personas and scenarios) pays for itself by catching failures that would otherwise reach production users. Particularly valuable for customer service agents, onboarding assistants, and any agent where user dissatisfaction has direct business cost.

---

## CuP: Completion under Policy

An emerging evaluation metric that combines task success with compliance: a task is considered complete **only if it was accomplished without policy violations.**

**The problem it solves:** Traditional success metrics score a customer service agent that resolves a ticket as a success — even if the agent leaked PII, used inappropriate language, or violated a regulatory requirement during the resolution. CuP scores this as 0, not 1. The ticket was resolved, but the resolution violated policy.

**Definition:** CuP(task) = 1 if and only if:
- The task was completed successfully (goal achieved)
- AND no policy violations occurred during execution (no PII leaks, no safety violations, no unauthorized actions, no regulatory breaches)

**Why this matters for enterprise agents:** In regulated industries (finance, healthcare, legal), a task that succeeds while violating policy is worse than a task that fails cleanly — the failure can be retried, but the policy violation may have legal consequences. See the Air Canada case study ([[../18_AI_Governance/07_Enterprise_AI_Adoption|Enterprise AI Adoption]]) for an example where an agent's "successful" response created a legal liability.

**Expected adoption:** CuP is expected to become an enterprise standard metric by 2026-2027, particularly in industries where regulatory compliance is non-negotiable. It aligns agent evaluation with what enterprise buyers actually care about: reliable outcomes within policy boundaries.

## Key Takeaways

- **Continuous evaluation transforms quality assurance from a one-time gate into an ongoing process.** Models degrade in production due to data drift, provider updates, and changing user expectations -- one-time evaluation before release is necessary but insufficient.

- **Sampling strategy determines evaluation coverage and cost.** Start with 5-10% random sampling for baseline monitoring, but increase to 100% for high-risk agent actions where the cost of a missed error exceeds the cost of evaluation.

- **A/B testing is the gold standard for comparing variants in production.** Randomize by user (not request), define guardrail metrics that must not deteriorate, and run tests long enough to cover natural variability.

- **Regression detection after silent provider updates is critical.** Shadow evaluation and canary deployments catch quality changes before full rollout -- generic benchmarks do not predict how a model update will behave on your specific prompts.

- **CuP (Completion under Policy) combines task success with compliance into a single gate metric.** A task that succeeds while violating policy scores zero -- essential for regulated industries where policy violations carry legal consequences.

- **Simulation testing with synthetic personas systematically explores the interaction space.** Generating multi-turn conversations with aggressive, confused, and adversarial personas catches edge cases that random sampling and manual testing miss.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Evaluation and Testing
**Previous:** [[03_LLM_as_Judge|LLM as Judge]]
**Next:** [[05_RAG_Evaluation|RAG Evaluation]]
