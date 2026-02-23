# AI Regulatory Landscape: EU AI Act and Global Initiatives

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Governance
**Previous:** [[../17_Production_Inference/07_SGLang_and_Alternatives|SGLang and Alternatives]]
**Next:** [[02_AI_Risk_Management|AI Risk Management]]

---

## Introduction

In August 2024, the EU AI Act came into force — the world's first comprehensive law regulating artificial intelligence. Fines up to 7% of global company turnover, entire categories of AI systems banned, mandatory certification for high-risk applications. For an AI architect, the regulatory landscape represents constraints that directly affect architectural decisions.

## EU AI Act: Structure and Timeline

### Risk-Based Framework

The EU AI Act classifies AI systems into four risk levels:

**UNACCEPTABLE RISK** — fully prohibited practices: social scoring, real-time biometrics in public spaces.

**HIGH RISK** — strict requirements and certification: HR systems, credit scoring, law enforcement, healthcare.

**LIMITED RISK** — transparency requirements: chatbots, emotion recognition, deepfakes.

**MINIMAL RISK** — no special requirements: spam filters, video games, warehouse management.

### Enforcement Timeline (Updated February 2026)

- **August 2024** — the law entered into force ✓
- **February 2025** — ban on unacceptable risk practices ✓ (NOW ENFORCED — social scoring, real-time public biometrics, emotion recognition in workplaces/schools are actively prohibited with fines)
- **August 2025** — obligations for GPAI models ✓ (NOW ENFORCED — all GPAI providers must provide technical documentation, training data summaries, copyright compliance)
- **August 2026** — full enforcement for high-risk systems (UPCOMING — certification requirements for HR, credit scoring, law enforcement, healthcare AI)
- **August 2027** — high-risk in Annex I (safety products)

### Prohibited Practices

As of February 2025, the following are fully prohibited: evaluating people based on social behavior, manipulation of vulnerable groups, facial recognition in public spaces (with exceptions for terrorism, missing persons, serious crimes), emotion recognition in the workplace and educational institutions, biometric categorization by race or political views, building facial recognition databases from the internet or CCTV.

### High-Risk Categories

Mandatory certification is required for: biometric identification, critical infrastructure (energy, water, transport), education and vocational training, employment, essential services (credit, insurance, social benefits), law enforcement, migration and borders, the judicial system.

### Eight Mandatory Requirements for High-Risk Systems

High-risk systems must comply with: risk management (a continuous system throughout the entire lifecycle), data governance (quality, relevance, representativeness), technical documentation (full description of the system and architecture), record-keeping (automatic logging for audit), transparency (clear instructions for users), human oversight (mechanisms for operator intervention), accuracy and robustness (resilience to errors), cybersecurity (protection against adversarial attacks).

## General Purpose AI (GPAI)

### GPAI Definition

General Purpose AI Model — a model trained on a large volume of data with self-supervision, capable of performing a broad range of tasks. Examples: GPT-4, Claude, Gemini, Llama, Mistral, open-source LLMs.

### Obligations for All GPAI Providers (from August 2025)

Technical documentation: model description, training process, data sources, compute resources, known limitations. Transparency for downstream: information for deployers, integration instructions, risk mitigation guidance. Copyright compliance: policy for compliance with EU copyright law, summary of training data. Cooperation: responding to regulator requests, providing documentation on demand.

### GPAI with Systemic Risk

For models exceeding 10^25 FLOPS during training, additional requirements apply: model evaluation (adversarial testing, red teaming), incident reporting (serious incidents and corrective measures), cybersecurity (adequate protection, vulnerability management), energy consumption (documentation and reporting).

### Codes of Practice

The EU AI Office developed the first GPAI Code of Practice (finalized 2025) with participation from OpenAI, Google, Meta, Mistral, and Anthropic. The Code covers: transparency requirements for model capabilities and limitations, safety evaluation procedures (including red teaming protocols), systemic risk assessment methodologies, and incident reporting procedures. Compliance with the Code is considered sufficient to demonstrate adherence to GPAI obligations — it serves as a "safe harbor" for providers.

## NIST AI Risk Management Framework

### The Shifting US Approach

The US regulatory landscape shifted significantly in 2025. **President Biden's Executive Order on AI (October 2023)** — which established safety standards, reporting requirements for frontier models, and directed federal agencies to develop AI guidelines — was **rescinded by President Trump in January 2025**. This removed federal-level AI safety reporting requirements and shifted the US approach further toward voluntary industry self-regulation.

However, NIST AI RMF 1.0 (January 2023) remains in effect as a voluntary, practical, flexible, risk-based framework.

### NIST AI RMF Structure

Four core functions:

**GOVERN:** Organizational culture, roles and responsibilities, policies and processes.

**MAP:** Establishing context, identifying risks, categorization.

**MEASURE:** Risk analysis, testing and evaluation, quantitative assessment.

**MANAGE:** Risk prioritization, treatment selection, monitoring.

NIST provides the AI RMF Playbook (practical actions), Profiles (adaptations for specific use cases), companion resources (crosswalks, case studies).

## Sector-Specific Requirements

### Healthcare: FDA and HIPAA

FDA Guidance for AI/ML medical devices: Class I (low risk) — general controls, Class II (moderate) — 510(k) clearance, Class III (high) — PMA approval. HIPAA for AI: protection of PHI in training data, de-identification requirements, Business Associate Agreements.

### Finance: SEC and FINRA

SEC AI Guidance (2024): disclosure requirements for AI-driven trading, risk management expectations, model governance. FINRA Expectations: oversight of AI tools, testing and validation, record-keeping. EU DORA: ICT risk management, third-party risks, incident reporting.

### Government: FedRAMP

AI in federal systems: FedRAMP authorization, FISMA compliance, agency requirements. Note: The Biden Executive Order on AI (October 2023) was rescinded in January 2025. Federal AI policy is now primarily driven by individual agency guidance and procurement requirements rather than a centralized executive mandate.

### US State-Level AI Regulation

With the federal executive order rescinded, **US states have become the primary source of AI regulation**:

**Colorado AI Act (SB 24-205, signed May 2024):** The first comprehensive US state AI law. Requires deployers of "high-risk AI systems" to use reasonable care to avoid algorithmic discrimination. Mandates risk management policies, impact assessments, and consumer notification when AI is used in consequential decisions (employment, finance, housing, insurance, education). Effective February 2026.

**California SB-1047 (vetoed September 2024):** Would have required safety testing for large AI models (>$100M training cost or >10^26 FLOPS). Though vetoed by Governor Newsom, it influenced national debate and set a template for future legislation. Key provisions that may resurface: mandatory safety evaluations, kill switch requirements, whistleblower protections.

**Other state activity (2024-2025):** Texas, Illinois, Connecticut, and several other states have introduced or passed AI-related legislation focusing on: algorithmic discrimination in employment (Illinois AI Video Interview Act), deepfake disclosure requirements, AI in insurance underwriting. The trend is toward sector-specific state regulation in the absence of federal legislation.

**Implication for AI architects:** Multi-state deployment in the US now requires tracking a patchwork of state laws, similar to how GDPR compliance was needed for EU deployment. The strictest applicable state law effectively sets the floor.

## Global Differences

### Comparison of Approaches

**EU:** Regulation of all AI, strong enforcement, GPAI rules in place.

**USA:** Voluntary federal framework (NIST) after Biden EO rescission (Jan 2025), growing state-level regulation (Colorado AI Act), sector-specific federal rules.

**China:** Comprehensive regulation focused on safety and ethics, state enforcement, GPAI rules in place.

**UK:** Pro-innovation approach, principles-based foundation, sandbox for experimentation, evolving GPAI rules.

### China: Comprehensive Regulation

Key regulations: Algorithm Recommendation (2022), Deep Synthesis (2023) for deepfakes, Generative AI Measures (2023). Requirements: content moderation, registration, user consent, safety assessment.

### UK: Pro-Innovation Approach

AI Regulation White Paper (2023): sector-specific regulation, existing regulators, principles-based approach. Five principles: safety and robustness, transparency and explainability, fairness, accountability and governance, contestability and redress.

## Practical Implications

### Compliance Architecture

The compliance architecture includes two layers:

**AI Governance Layer:** Risk assessment (continuous identification and analysis), compliance monitoring (tracking requirement fulfillment), audit and logging (collecting and storing logs for traceability).

**AI System Layer:** Model management (versioning, testing, validation), data management (quality control, access management), human oversight (interfaces for intervention).

### Architect Checklist

**For any AI system:** risk level defined per EU AI Act, applicable sector-specific regulations identified, documentation requirements understood, data governance in place, human oversight mechanisms designed.

**For high-risk:** conformity assessment planned, technical documentation prepared, logging and record-keeping implemented, accuracy and robustness validated, cybersecurity measures deployed.

**For GPAI:** training data documented, copyright compliance ensured, information prepared for downstream providers, incident reporting process ready.

### Multi-jurisdiction Strategy

When deploying across multiple jurisdictions, a unified approach covering the strictest requirements is necessary. For the EU: EU AI Act with risk level determination. For the USA: NIST AI RMF plus sector-specific regulations. For China: Generative AI Measures with mandatory content moderation. For the UK: principles-based approach through existing regulators.

## Key Takeaways

The EU AI Act is actively enforced: prohibited practices banned since February 2025, GPAI obligations in force since August 2025, high-risk requirements coming August 2026. The GPAI Code of Practice provides a compliance safe harbor.

The US landscape shifted: Biden's AI Executive Order was rescinded (January 2025), making NIST AI RMF the primary federal guidance (voluntary). US states are filling the gap — Colorado AI Act (2026 enforcement), with Texas, Illinois, and others following.

GPAI obligations now affect all foundation model providers: technical documentation, training data summaries, copyright compliance, and (for systemic risk models) adversarial testing and incident reporting.

Global differences are significant — a multi-jurisdiction strategy is needed. The EU is the strictest, the US is fragmented across states, China requires registration and content moderation. Compliance requires architectural decisions: logging, human oversight, documentation.

## Practical Example: Determining Compliance Requirements

When developing an AI system, the process of determining requirements includes several stages:

**Risk level classification:** Determine the category per the EU AI Act depending on the use case. Systems for recruiting, credit scoring, critical infrastructure — high-risk. Chatbots and emotion recognition systems — limited risk. Social scoring and mass biometrics — prohibited.

**Determining GPAI requirement applicability:** If the model is a General Purpose AI (trained on large data, solves a broad range of tasks), additional requirements apply. The 10^25 FLOPS threshold indicates systemic risk: adversarial testing, incident reporting, energy consumption reporting.

**Accounting for deployment geography:** For systems in the EU, the EU AI Act applies with fines up to 7% of turnover. In the USA, the voluntary NIST AI RMF with sector-specific regulations is in effect. In China, registration and content moderation are mandatory.

**Compiling the requirements list:** For high-risk: risk management system, data governance, technical documentation, automatic logging, human oversight, accuracy validation, cybersecurity, conformity assessment, registration in the EU database. For limited risk: informing users about interaction with AI, labeling AI-generated content. For GPAI: model technical documentation, information for downstream providers, copyright policy, training data summary.

For risk classification, the following logic applies: if the use case is on the prohibited list (social scoring, public biometrics) — UNACCEPTABLE. If on the high-risk list (employment, credit scoring, law enforcement, education, critical infrastructure) — HIGH. If on the limited-risk list (chatbot, emotion recognition, deepfake) — LIMITED. Otherwise — MINIMAL. This approach ensures systematic assessment and understanding of the timeline: prohibited practices cannot be used from February 2025, GPAI obligations from August 2025, full requirements for high-risk from August 2026.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Governance
**Previous:** [[../17_Production_Inference/07_SGLang_and_Alternatives|SGLang and Alternatives]]
**Next:** [[02_AI_Risk_Management|AI Risk Management]]
