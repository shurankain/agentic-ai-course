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

## Further Reading

For production implementation of AI risk assessment frameworks, see the NIST AI RMF Playbook and ISO/IEC 23894 standard. The concepts above (risk scoring, continuous monitoring, audit trails) translate directly to code — the risk scoring formula, monitoring thresholds, and hash-chain audit patterns described in this lesson provide the specification that your implementation should follow.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Governance
**Previous:** [[01_Regulatory_Landscape|Regulatory Landscape]]
**Next:** [[03_Red_Teaming|Red Teaming]]
