# AI Risk Management

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Governance
**Previous:** [[01_Regulatory_Landscape|Regulatory Landscape]]
**Next:** [[03_Red_Teaming|Red Teaming]]

---

## Introduction: Why AI Systems Require a Special Approach to Risk

Traditional risk management methodologies were developed for deterministic systems — software either works correctly or contains a bug that can be found and fixed. AI systems break this paradigm. A model can produce excellent results on 99% of inputs but fail catastrophically on the remaining percent — and it is impossible to determine in advance which inputs will cause the failure.

This fundamental uncertainty gives rise to an entire class of risks that traditional frameworks cannot address. When GPT-4 hallucinates nonexistent court precedents, it is not a bug in the code — it is emergent behavior of a system trained to predict probable tokens. When a recommendation algorithm systematically discriminates against certain groups, it is not the result of malicious intent by developers — it is a manifestation of bias in training data. Managing such risks requires a fundamentally new toolkit.

## AI Risk Taxonomy

### Technical Risks

The first category covers risks associated with the technology itself. **Hallucinations** — generation of plausible-sounding but false information — remain an unsolved problem even for frontier models. In a medical context, a hallucination can lead to an incorrect diagnosis; in a legal context, it can lead to accusing an innocent person.

**Adversarial attacks** represent an even more insidious threat. Minimal changes to input data, imperceptible to humans, can fundamentally alter a model's output. A sticker on a road sign causes an autopilot to fail to recognize a STOP sign. A specially crafted prompt bypasses safety guardrails and forces the model to generate harmful content.

**Distribution shift** occurs when real-world data differs from training data. A model trained on 2020 data can fail catastrophically in 2024 because user behavior patterns, economic conditions, or even word meanings have changed. COVID-19 broke thousands of forecasting models that had never seen data from a global pandemic.

**Model degradation** — gradual quality deterioration without apparent cause. Data drifts, the environment changes, and a model that worked perfectly yesterday produces suboptimal results today. Without continuous monitoring, this process can go unnoticed for months.

### Ethical Risks

**Algorithmic bias** is one of the most studied and still unsolved problems. AI systems absorb and amplify biases contained in training data. A credit scoring system discriminates by race not because race is included as a feature, but because ZIP code correlates with race, and the model learned this proxy.

The classic COMPAS study showed that the recidivism assessment algorithm erroneously assigned high risk to African Americans twice as often as to white individuals. The system's developers had no intention of creating a discriminatory algorithm — the bias arose from historical arrest data, which itself reflected systemic racism in the law enforcement system.

**The trade-off between fairness and accuracy** creates an ethical dilemma: improving fairness by one metric can worsen it by another. Equalized odds, demographic parity, and individual fairness are incompatible goals, and the choice among them is not a technical but a values-based question.

**Privacy risks** in AI go beyond traditional threats. Membership inference attacks allow determining whether a specific example was in the training set. Model extraction attacks create functional copies of proprietary models. Training data extraction can recover confidential data directly from model weights — GPT-2 "memorized" and reproduced credit card numbers from the training corpus.

### Operational Risks

**Single point of failure** arises from excessive dependence on a single AI provider or model. When the OpenAI API was unavailable for several hours in March 2023, thousands of applications worldwide stopped working. Market concentration around a few frontier models creates systemic risk for the entire industry.

**Vendor lock-in** exacerbates the problem. An application built on GPT-4-specific features (vision, function calling, JSON mode) is difficult to migrate to Claude or Gemini. Fine-tuned models are tied to the provider's platform. Prompts optimized for one model often perform worse on others.

**Cost unpredictability** is a characteristic of token-based pricing. Unlike the fixed cost of traditional software, AI API expenses depend on usage patterns that are difficult to predict. A viral product can lead to a tenfold increase in AI costs within days.

**Compliance complexity** grows as regulators worldwide adopt AI-specific legislation. The EU AI Act, Chinese generative AI regulations, industry standards from the FDA and financial regulators — all impose different requirements on the same AI system.

### Strategic Risks

**Competitive displacement** — AI can devalue a company's core competencies faster than it can adapt. Stack Overflow lost 50% of its traffic within a year after ChatGPT launched, because users gained an alternative way to solve coding problems.

**Talent concentration** — AI specialists are extremely unevenly distributed. Google, OpenAI, Anthropic, and Meta absorb top talent, leaving other companies with a difficult choice between expensive poaching and working with less experienced teams.

**Technical debt accumulation** — AI systems generate a special kind of technical debt. Models become outdated faster than traditional code. The need for retraining on new data, migrating to new architectures, and adapting to new APIs creates constant resource pressure.

## Risk Assessment Frameworks

### NIST AI Risk Management Framework

The NIST AI RMF, released in January 2023, represents the most mature framework for systematic AI risk management. Its four functions form a continuous cycle:

**GOVERN** establishes the organizational culture and structures for AI risk management. This includes defining roles and responsibilities, creating policies, and ensuring accountability at the C-suite level. Critically: AI governance is not an IT function but an enterprise-wide capability.

**MAP** — identification and analysis of the AI usage context. What decisions does the system make? Who is affected by those decisions? What are the consequences of errors? At this stage, a comprehensive picture of AI usage across the entire organization is created. Many companies discover "shadow AI" — employee use of AI tools without the IT department's knowledge.

**MEASURE** — risk assessment using quantitative and qualitative methods. NIST emphasizes the importance of measuring not only technical characteristics (accuracy, latency) but also trustworthiness attributes: bias, explainability, privacy, robustness. Measurement must be continuous, not a one-time event.

**MANAGE** — making decisions about risks and their mitigation. The four classic strategies — avoid, mitigate, transfer, accept — apply to AI risks as well, but with specific considerations. Some AI risks cannot be fully mitigated, and the organization must explicitly accept residual risk.

### ISO/IEC 23894: AI Risk Management

The international standard ISO/IEC 23894 (2023) provides guidelines for integrating AI risks into enterprise risk management. Key principles:

**Proportionality** — the level of risk management effort should correspond to the level of risk. A FAQ chatbot requires less oversight than an AI system for credit decisions.

**Lifecycle approach** — risks are managed at all stages: design, development, deployment, operation, decommissioning. Risks change at each stage, and mitigations must evolve accordingly.

**Stakeholder engagement** — those affected by an AI system should have a voice in its governance. This is especially important for AI systems that make decisions about people.

### Industry-Specific Frameworks

**Financial services** have developed the most mature practices thanks to strict regulatory oversight. SR 11-7 from the Federal Reserve establishes requirements for model risk management for US banks. EBA guidelines require European banks to provide special oversight for AI/ML models.

**Healthcare** follows the FDA's proposed regulatory framework for AI/ML-based Software as a Medical Device (SaMD). The key requirement is a Total Product Lifecycle (TPLC) approach, including pre-market review and post-market monitoring.

**Automotive** uses ISO 21448 (SOTIF — Safety of the Intended Functionality) for autonomous driving. The framework addresses unknown unsafe scenarios — situations the developer did not foresee but that could lead to an accident.

## AI Risk Assessment Methodology

### Risk Identification

The process begins with comprehensive mapping of all AI use cases in the organization. This includes:

**AI System Inventory** — a centralized registry of all AI systems, including vendor solutions, custom models, and shadow AI. For each system, the following is recorded: purpose, input/output, users, decisions affected, data used, model type.

**Impact Assessment** — for each system, the potential impact of errors is determined. The severity scale can range from "inconvenience" (recommending an uninteresting movie) to "critical" (an incorrect medical diagnosis).

**Likelihood Estimation** — assessing the probability of risk materialization. For AI systems, this is especially difficult due to emergent behaviors and distribution shift. Historical data on failures helps but does not guarantee the absence of novel failures.

### Quantitative Methods

**Failure Mode and Effects Analysis (FMEA)** is adapted for AI with the addition of AI-specific failure modes: hallucination, bias, adversarial vulnerability, data poisoning.

**Stress testing** includes evaluation on edge cases, out-of-distribution inputs, and adversarial examples. The goal is to find the boundaries of the model's capabilities before users do.

**A/B testing** for AI systems requires special caution. If AI makes decisions affecting people (credit, hiring), an A/B test can create unfair treatment of control vs. treatment groups.

### Qualitative Methods

**Expert judgment** remains critically important, especially for novel risks. Inviting external experts (ethicists, domain specialists, affected communities) provides perspective that the internal team may lack.

**Red teaming** — a systematic attempt to "break" an AI system, finding ways to misuse or manipulate it. Anthropic, OpenAI, and other frontier labs practice extensive red teaming before releasing new models.

**Scenario analysis** explores "what if" questions: what if training data leaked? What if a competitor copied our model? What if a regulator banned our use case?

## AI Risk Mitigation

### Technical Mitigations

**Model monitoring** — continuous tracking of model performance in production. Metrics include accuracy, latency, throughput, but also fairness metrics and anomaly detection. Drift detection warns of distribution shift before it becomes critical.

**Human-in-the-loop (HITL)** — maintaining human oversight for high-stakes decisions. Automation bias — the tendency to rely on AI recommendations — requires explicit design for meaningful human review.

**Graceful degradation** — the system must fail safely. If the AI component is unavailable or unreliable, the system should fall back to rule-based logic or human processing rather than crash.

**Multi-model ensembles** — using multiple models for critical tasks. Disagreement between models signals the need for human review.

### Organizational Mitigations

**AI Ethics Board** — an independent body for reviewing high-risk AI use cases. The board should include diverse perspectives, not only technical specialists.

**Training and awareness** — all employees working with AI must understand its limitations. This is especially important for business users who may over-trust AI outputs.

**Incident response** — a documented procedure for responding to AI failures. Includes detection, containment, eradication, recovery, and lessons learned.

**Audit trails** — logging of all AI decisions with sufficient detail for post-hoc analysis. Regulatory requirements for traceability are becoming increasingly strict.

### Contractual Mitigations

When working with AI vendors:

**SLAs** should include AI-specific metrics: model performance guarantees, maximum latency, uptime for critical components.

**Data rights** — clarity on who owns training data, fine-tuned models, and inference logs. Especially important for sensitive domains.

**Liability allocation** — who bears responsibility for AI-induced damages? The vendor or the customer? This is a key negotiation point.

**Exit provisions** — the right to export data and models when switching vendors. Model portability is not yet standardized, which complicates exits.

## Key Takeaways

AI risks fundamentally differ from traditional IT risks. The stochastic nature of AI systems, emergent behaviors, and the ability to absorb biases from data require specialized approaches to risk management.

The NIST AI RMF provides a comprehensive framework with four functions: GOVERN, MAP, MEASURE, MANAGE. It emphasizes that AI governance is an enterprise-wide capability, not exclusively a technical function.

The AI risk taxonomy covers technical (hallucinations, adversarial attacks, drift), ethical (bias, privacy), operational (vendor lock-in, cost unpredictability), and strategic (competitive displacement) categories.

Mitigations should be multi-layered: technical (monitoring, HITL, ensembles), organizational (ethics boards, training, incident response), and contractual (SLAs, liability, exit provisions).

Continuous monitoring is critically important for AI systems. Unlike traditional software that operates deterministically, AI models degrade over time and require constant observation.

## Practical Code Examples

### AI System Risk Assessment Framework

A comprehensive AI risk assessment framework is based on NIST AI RMF and ISO/IEC 23894 standards. The system provides a structured approach to identifying, analyzing, and managing AI system risks.

**Risk Classification:**

The system uses a hierarchical risk classification by categories. Technical risks include hallucinations, data leakage, adversarial vulnerabilities, and model drift. Ethical risks cover algorithmic bias and fairness issues. Operational risks include vendor lock-in, cost unpredictability, and regulatory compliance issues. Strategic risks concern competitive displacement and long-term business viability.

Risk severity is assessed on a four-level scale: low, medium, high, and critical. The likelihood of risk materialization is measured on a five-level scale: rare, unlikely, possible, likely, and almost certain.

**Risk Model:**

Each identified risk has a unique identifier, name, and detailed description. The risk is classified by category and assessed by severity and likelihood. Affected stakeholders are recorded. Mitigations — risk reduction measures — can be defined for each risk.

Inherent risk is calculated as the product of severity and likelihood before mitigations are applied. Residual risk accounts for the effect of mitigations and shows the risk remaining after their application. Each risk has an owner responsible for managing it and a date for the next review.

The risk level is categorized automatically: critical (score 16 and above), high (9-15), medium (4-8), low (below 4). This helps prioritize risk management efforts.

**AI System Description:**

To conduct a risk assessment, the system must be fully described. The description includes a unique identifier, name, and purpose. The model type is specified (large language model, classification, regression, etc.). It is critically important to document what decisions the system makes and who is affected by those decisions.

Data sources for training and inference, the deployment environment, and the level of human oversight are described. The oversight level can be: none, monitoring, approval required, or veto power. The list of risks is associated with the system and can be expanded during the assessment process.

A risk summary is automatically generated for the system: the total number of identified risks, distribution by category, distribution by severity level, a list of risks without mitigations, and the highest risk score. This provides a quick overview of the system's risk profile.

**Standard Risk Catalog:**

The framework includes a catalog of standard risks for AI systems. Hallucinations (TECH-001) — high severity, likely occurrence — generation of plausible but incorrect information. Algorithmic bias (ETH-001) — critical severity, possible occurrence — systematic discrimination against groups.

Training data leakage (TECH-002) — critical severity, unlikely occurrence — reproduction of confidential information from the training set. Adversarial vulnerability (TECH-003) — high severity, possible occurrence — success of specially crafted attacks.

Vendor lock-in (OPS-001) — medium severity, likely occurrence — excessive dependence on a single AI provider. Cost unpredictability (OPS-002) — medium severity, likely occurrence — unexpected growth in API expenses.

Model drift (TECH-004) — high severity, almost certain occurrence — gradual quality degradation due to changes in data distribution. Regulatory non-compliance (OPS-003) — critical severity, possible occurrence — violation of AI legislation.

Competitive displacement (STRAT-001) — high severity, possible occurrence — AI competitors devalue the organization's core competencies.

**Risk Assessment Process:**

The assessment begins by applying relevant standard risks from the catalog to a specific system. For each applied risk, a copy is created preserving all parameters. This forms the baseline risk profile of the system.

Then, for each significant risk, mitigations are developed and documented. When adding a mitigation, a new severity and/or likelihood assessment can be specified, reflecting the mitigation's effectiveness. Residual risk is recalculated automatically.

**Risk Matrix:**

Risk visualization is performed through a risk matrix with severity on the vertical axis and likelihood on the horizontal axis. Each cell in the matrix shows the identifiers of risks with the corresponding parameter combination. The matrix immediately reveals which risks are in the critical zone (high severity and high likelihood).

**Report Generation:**

The full risk assessment report includes the assessment date, key system characteristics (identifier, name, purpose, model type, oversight level), and a risk summary by category and level.

Critical and high risks are highlighted separately with their scores and mitigation status. Recommendations are generated based on the risk profile: if there are critical risks — address immediately before deployment; if there are risks without mitigations — develop measures; if human oversight is absent for high-stakes decisions — add controls; if there are ethical risks — conduct a fairness audit.

The next review date is calculated based on the profile: critical risks — in 30 days, high risks — in 90 days, otherwise — in 180 days.

**Practical Application:**

For a credit scoring system, a full description is created: purpose — automating initial scoring, model type — gradient boosting ensemble, decisions made — credit approval, interest rate, credit limit, affected groups — credit applicants and existing customers, data sources — credit bureau, application data, transaction history, environment — production AWS, oversight level — human approval for large loans.

The following risks are applied to the system: algorithmic bias, model drift, regulatory non-compliance, adversarial attacks. For bias, a mitigation is added — regular fairness audits using the Aequitas toolkit, which reduces severity to medium and likelihood to unlikely. For drift — weekly monitoring with automatic alerts, reducing likelihood to possible.

A final report is generated in JSON format with complete information on risks, mitigations, and recommendations. A risk matrix is visualized for quick overview. This structured approach ensures systematic risk management throughout the entire AI system lifecycle.

### Continuous Monitoring for AI Systems

A continuous monitoring framework for AI systems provides real-time performance tracking, data drift detection, and decision fairness control. This is critically important because AI models degrade over time, unlike traditional software.

**Metrics Collection:**

The monitoring system uses ring buffers to store metric history with a limited size (typically the last 1000 data points). Each data point includes a timestamp, metric value, and optional metadata. This allows efficient time series tracking without unbounded memory growth.

For each metric, recent values over a specified period (e.g., the last hour) can be retrieved, and statistics can be computed: count, mean, median, standard deviation, minimum, and maximum. These statistics are used to detect anomalies and deviations from normal behavior.

**Alert Management:**

The alert system tracks threshold violations and other issues. Each alert has a unique identifier, severity level (informational, warning, critical), metric name, message, timestamp, current value, threshold value, and an acknowledgment flag.

The alert manager supports handler registration — functions that are called when a new alert is created. This allows integrating notifications through various channels: email, Slack, PagerDuty, etc. Active (unacknowledged) alerts can be queried for display in dashboards.

**Drift Detection:**

The drift detector tracks changes in data distribution. The algorithm works as follows: first, a baseline is established based on the first data window (typically 1000 points), computing the mean and standard deviation. Then for each new detection window (typically 100 points), the mean is computed.

The normalized deviation from the baseline is calculated as a z-score: the difference between the current mean and the baseline mean, divided by the baseline standard deviation. If the z-score exceeds the threshold (typically corresponding to 3 standard deviations), a drift warning is generated.

Drift information includes the type (distribution drift), baseline mean, current mean, z-score, and severity level (critical if z-score > 3, otherwise warning). After drift is confirmed, the baseline can be reset based on new data.

**Fairness Monitoring:**

The fairness monitor tracks equality metrics for protected attributes (gender, age, race, etc.). The system records decision outcomes for each group: positive and total counts.

Demographic parity is computed as the positive outcome rate for each group. Disparity is the difference between the maximum and minimum rates. The system is considered fair if the disparity is less than 10%.

Equalized odds require equal true positive rates (TPR) and false positive rates (FPR) across all groups. For each group, TPR (the proportion of correct positive predictions among actual positive cases) and FPR (the proportion of incorrect positive predictions among actual negative cases) are computed.

**Main Monitor:**

The central monitoring class integrates all components. For each metric, a buffer, drift detector, and optional thresholds are registered. Thresholds define warning and critical levels, as well as direction (exceeding or falling below the threshold).

When recording a metric value, several checks are performed. The value is added to the buffer. Threshold values are checked: if the critical threshold is exceeded, a critical alert is created; if the warning threshold is exceeded, a warning alert is created. The drift detector is updated with the new value, and if drift is detected, a corresponding alert is created.

For fairness monitoring, predictions are recorded with the group and positive/negative outcome. The fairness monitor updates group statistics.

**Dashboard:**

Dashboard data includes the system identifier, timestamp, statistics for all metrics (for the last hour), fairness summary for all monitors, and a list of active alerts with their identifiers, severity, metric, message, and timestamp.

**Practical Application:**

For a credit scoring system, the following metrics are registered: prediction latency (with warning threshold of 500ms and critical threshold of 1000ms), model confidence (with warning threshold of 0.6 and critical threshold of 0.4, direction — below), and approval rate (no thresholds, tracking only).

Fairness monitors are registered for gender and age groups. During operation, predictions are simulated with random latency and confidence values. If values cross thresholds, alerts are generated.

To demonstrate a fairness issue, the approval rate for men is set higher (70%) than for women (50%). This creates a bias detected by the monitor. The dashboard shows a fairness violation with a 20% difference in approval rates, exceeding the acceptable 10% threshold.

This framework enables detection of model degradation, technical issues, and ethical violations in real time, ensuring rapid response to problems before they affect users.

### Risk Register and Audit Trail

The risk register and audit trail system provides centralized risk management with full traceability of all actions. A tamper-proof audit trail based on hash chains is used.

**Event Types:**

The system tracks key events in the risk management lifecycle: risk creation, risk update, risk closure, mitigation addition, assessment completion, risk decision, incident report, and review completion.

**Audit Trail:**

Each audit event is immutable and contains a unique identifier, event type, timestamp, actor (who performed the action), system identifier, event details, and the hash of the previous event to ensure chain integrity.

The event hash is computed from all event fields including the previous hash, using SHA-256. This creates a chain where altering any past event breaks all subsequent hashes, making tampering evident.

Integrity verification iterates through all events, recomputes hashes, and compares them with recorded values. It also verifies that the previous event hash in each record matches the actual hash of the preceding record. Any discrepancy indicates tampering.

Events can be filtered by system or type. Audit export provides events for a specified period in a format suitable for external auditors, including all hashes for independent verification.

**Risk Record:**

Each entry in the risk register contains an identifier, name, description, category, and system identifier. The assessment includes inherent and residual likelihood and impact. Management includes the risk owner, response owner, status (open, mitigating, accepted, closed), response plan, and list of mitigations.

Tracking includes the creation date, last review date, next review date, and a list of related incidents. The inherent risk score is calculated as the product of likelihood and impact. The residual score accounts for the effect of mitigations.

**Risk Register:**

The register manages a centralized repository of all AI risks, integrated with the audit trail. Risk creation generates a unique identifier, creates a record, and logs an event to the audit trail with details including the risk score.

Adding a mitigation updates the risk's mitigation list, can update residual likelihood and impact, and logs an event with information about the risk score reduction. Recording a decision documents the response plan and rationale, logging the decision with the residual score.

Reporting an incident creates a unique incident identifier, links it to the risk, and logs complete information including description and severity. Completing a review updates review dates, logs findings, and schedules the next review.

**Risk Dashboard:**

The dashboard provides the total number of risks, distribution by status, distribution by category, a list of high risks (score >= 16) sorted by score, and a list of overdue reviews with the number of days overdue.

**Text Report:**

The report includes general information (generation date, system filter), a summary (total count, critical/high count), and details for each risk: identifier, name, category, status, inherent and residual scores, owner, number of mitigations, and number of incidents.

**Practical Example:**

For a credit scoring system, the following risks are created: algorithmic bias in lending (ethical category, likelihood 3, impact 5, owner ML lead) and model performance degradation (technical category, likelihood 4, impact 3, owner ML ops).

For bias, mitigations are added: fairness constraints implemented during training (reduces likelihood to 2) and monthly fairness audits with Aequitas (reduces impact to 3). A decision is made: accept the residual risk with continuous monitoring, as the residual score of 6 is within tolerance.

For degradation, an incident is reported: the approval rate dropped 15% over the last week, severity medium. A bias review is conducted with findings: bias metrics are within tolerance, 0 incidents, recommendation to continue monitoring.

Audit trail integrity verification confirms that the hash chain is intact. The dashboard shows the current state of all risks. The report provides complete documentation for management and regulators.

Exporting audit events for the last day shows the full chronology of all risk actions with timestamps, event types, and details. This ensures complete traceability and accountability in AI risk management.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Governance
**Previous:** [[01_Regulatory_Landscape|Regulatory Landscape]]
**Next:** [[03_Red_Teaming|Red Teaming]]
