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

### Enforcement Timeline (Updated May 2026 — Digital Omnibus)

- **August 2024** — the law entered into force ✓
- **February 2025** — ban on unacceptable risk practices ✓ (NOW ENFORCED — social scoring, real-time public biometrics, emotion recognition in workplaces/schools are actively prohibited with fines)
- **August 2025** — obligations for GPAI models ✓ (NOW ENFORCED — all GPAI providers must provide technical documentation, training data summaries, copyright compliance)
- **December 2026** — watermarking requirements (Article 50) — delayed from August 2026 by the Digital Omnibus (+4 months)
- **December 2027** — full enforcement for standalone high-risk systems (Annex III) — delayed from August 2026 by the Digital Omnibus (+16 months). Covers HR, credit scoring, law enforcement, healthcare AI
- **August 2028** — high-risk AI in regulated products (Annex I — safety products) — delayed from August 2027 by the Digital Omnibus (+12 months)

**Digital Omnibus (May 7, 2026):** The EU Council and European Parliament reached a provisional political agreement to delay high-risk enforcement deadlines as part of a broader simplification package. The deal has provisional status — formal adoption is expected before August 2, 2026, with amendments becoming binding after publication in the Official Journal. The requirements themselves are unchanged — only the deadlines shifted. This is additional time for compliance, not a relaxation of obligations.

**New prohibition — AI-generated NCII/CSAM (Article 5 amendment):** The Digital Omnibus adds a new entry to the list of prohibited AI practices: AI systems that generate or manipulate non-consensual intimate imagery (NCII) and child sexual abuse material (CSAM), including so-called "nudifier" tools. Effective December 2, 2026. This is the first new prohibited practice added to the AI Act since its adoption — signaling that the prohibition list is not static and will expand as new harms emerge (as of late May 2026).

**Additional Digital Omnibus changes (as of late May 2026):**
- **AI regulatory sandbox deadline extended** to August 2, 2027 (from the original earlier date). Member states have more time to establish national AI sandboxes for controlled testing of AI systems before market deployment.
- **Transparency grace period shortened** from 6 months to 3 months — new deadline December 2, 2026 (was later under the original schedule). Providers of AI systems with transparency obligations (chatbots, emotion recognition, deepfakes) now have less time to comply after the formal adoption.
- **AI Office competence clarified:** The EU AI Office supervises GPAI-model-based systems where the same provider develops both the model and the downstream system. Exceptions remain under national authority: law enforcement, border management, judicial systems, and financial institutions — these continue to be supervised by national market surveillance authorities.
- **GPAI Code of Practice signatories:** Amazon, Anthropic, Google, Microsoft, and OpenAI have signed the GPAI Code of Practice, which provides a presumption of compliance ("safe harbor") with GPAI obligations.

### Enforcement Reality (as of May 2026)

No fines have been issued yet under the EU AI Act. The first enforcement wave (prohibited practices, February 2025) is being monitored, but national market surveillance authorities are still building enforcement capacity. The GPAI Code of Practice (finalized 2025) provides a "safe harbor" — signatories receive a presumption of compliance and face fewer enforcement investigations. The practical reality: the Digital Omnibus delay (May 2026) extended the high-risk compliance window by 16-24 months, but the requirements are unchanged. Organizations should use the additional time to prepare — not to postpone. AI Literacy obligations (February 2025) and GPAI obligations (August 2025) remain fully in force.

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

### Bilateral AI Safety Testing (as of late May 2026)

Microsoft signed evaluation agreements with U.S. CAISI (Consortium for AI Safety and Innovation) and U.K. AISI (AI Safety Institute) on May 5, 2026, for testing national security and public safety risks of AI models. This is the first bilateral AI safety testing agreement involving a major AI company — establishing a precedent for government-industry cooperation on frontier model evaluation across jurisdictions. The agreement allows both institutes to test Microsoft's frontier models before and after deployment, with a focus on national security applications and public safety scenarios. This voluntary model — where companies proactively submit to government testing — may become the de facto standard in the absence of mandatory federal AI safety legislation.

### US State-Level AI Regulation

With the federal executive order rescinded, **US states have become the primary source of AI regulation**:

**Colorado AI Act (SB 24-205, signed May 2024):** Originally the first comprehensive US state AI law requiring reasonable care to avoid algorithmic discrimination, with risk management policies, impact assessments, and consumer notification for consequential decisions. The original effective date (February 2026) was delayed to June 30, 2026, then **delayed again to January 1, 2027** by SB 189 (signed May 14, 2026). More significantly, SB 189 **narrowed the scope**: the risk-based framework was replaced with a transparency-only approach — eliminating the duty of care, deployer obligations for impact assessments, and most reporting requirements to the Attorney General. The first comprehensive US state AI law was substantially weakened before ever being enforced — a cautionary signal about the political durability of AI regulation.

**California SB-1047 (vetoed September 2024):** Would have required safety testing for large AI models (>$100M training cost or >10^26 FLOPS). Though vetoed by Governor Newsom, it influenced national debate. California subsequently passed **SB-53** (signed September 2024) — a narrower alternative requiring frontier model developers to publish transparency reports about safety testing. California also passed 18+ additional AI-related bills in 2024 addressing AI literacy, election integrity, deepfakes, and AI watermarking, making it the most active US state on AI regulation.

**Other state activity (2024-2025):** Texas, Illinois, Connecticut, and several other states have introduced or passed AI-related legislation focusing on: algorithmic discrimination in employment (Illinois AI Video Interview Act), deepfake disclosure requirements, AI in insurance underwriting. The trend is toward sector-specific state regulation in the absence of federal legislation.

**Implication for AI architects:** Multi-state deployment in the US now requires tracking a patchwork of state laws, similar to how GDPR compliance was needed for EU deployment. The strictest applicable state law effectively sets the floor.

### Federal AI Legislation Attempts (as of mid-June 2026)

**Great American Artificial Intelligence Act (June 4, 2026):** A 269-page bipartisan discussion draft from Reps. Obernolte (R-CA) and Trahan (D-MA) — the most comprehensive federal AI bill proposed to date. Four titles: Frontier AI Governance (mandatory safety frameworks for models costing >$500M to train, third-party audits), Workforce (retraining programs), Cybersecurity, and Research/Development/International Cooperation. The most significant provision: a **3-year preemption of state AI development laws** — states retain the right to regulate AI *use* within their borders but lose the ability to legislate how AI systems are *built*. If enacted, this would override the Colorado AI Act and similar state laws on AI development. $100M/year authorized for a Center for AI Standards and Innovation. Status: discussion draft only — not yet introduced as a formal bill. The preemption clause is highly contested.

**Trump AI Executive Order (as of mid-June 2026):** Establishes a voluntary 30-day model review framework. CISA, NSA, and Treasury to define "qualifying models" subject to review within 60 days. The voluntary nature contrasts sharply with the EU's mandatory approach and with the June 13 export ban on Fable 5/Mythos 5 — the administration simultaneously promotes voluntary cooperation and wields export controls unilaterally when it deems necessary.

**First model-level export controls (June 13, 2026):** The Fable 5/Mythos 5 export ban (see [[../14_Security_Safety/03_Agent_Security|Agent Security]]) sets a precedent: AI models can now be export-controlled independently of hardware. This goes beyond the chip-focused CHIPS Act framework and creates a new category of AI governance — model-level access control by government directive.

### China: National AI Agent Framework (as of mid-June 2026)

**China issued the first national policy framework for AI agents** (May 8, 2026) — a joint publication by the Cyberspace Administration, National Development and Reform Commission, and Ministry of Industry and Information Technology. The framework introduces **risk-tiered governance** based on application scenarios: healthcare, transport, media, and public safety applications require mandatory filing, safety testing, and recall procedures. The document enumerates 19 application scenarios spanning scientific research, manufacturing, energy, agriculture, finance, education, healthcare, and public safety.

Key differences from the EU approach: China's framework specifically addresses agentic AI (agents that take autonomous actions) as a distinct governance category — not just a subcategory of general-purpose AI. Decision boundaries, behavioral controls, risk warnings, traceability, human oversight, and multi-agent coordination are all explicitly regulated. This is the most detailed national framework for agent governance globally and may influence how other jurisdictions approach agentic AI regulation.

## Global Differences

### Comparison of Approaches

**EU:** Regulation of all AI, strong enforcement, GPAI rules in place.

**USA:** Voluntary federal framework (NIST) after Biden EO rescission (Jan 2025), growing state-level regulation (Colorado weakened May 2026), Great American AI Act draft (June 2026), model-level export controls (June 2026). Fragmented but increasingly active.

**China:** Comprehensive regulation including the first national AI agent framework (May 2026). Risk-tiered: 19 application scenarios with mandatory filing for healthcare/transport/media. State enforcement, GPAI rules in place.

**UK:** Pro-innovation approach, principles-based foundation, sandbox for experimentation, evolving GPAI rules.

### China: Comprehensive Regulation

Key regulations: Algorithm Recommendation (2022), Deep Synthesis (2023) for deepfakes, Generative AI Measures (2023), **AI Agent Framework (May 2026)** — the first national agentic AI policy (see above). Requirements: content moderation, registration, user consent, safety assessment. The agent framework adds: decision boundary controls, behavioral monitoring, risk-tiered filing, and mandatory recall procedures for high-risk agent applications.

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

The EU AI Act is actively enforced: prohibited practices banned since February 2025, GPAI obligations in force since August 2025. High-risk requirements were delayed by the Digital Omnibus (May 2026): standalone high-risk (Annex III) now December 2027, product-embedded high-risk (Annex I) now August 2028. The GPAI Code of Practice provides a compliance safe harbor.

The US landscape is in flux (as of mid-June 2026): Biden's EO rescinded (Jan 2025), Colorado AI Act weakened before enforcement (SB 189, May 2026), but the **Great American AI Act** draft (June 4) proposes the first comprehensive federal framework with 3-year state preemption. The **Fable 5/Mythos 5 export ban** (June 13) established model-level export controls as a new governance tool. Bilateral safety testing (Microsoft-CAISI/AISI) may become the de facto standard. China published the **first national AI agent framework** (May 8) with risk-tiered governance for agentic AI — the most detailed agent-specific regulation globally.

GPAI obligations now affect all foundation model providers: technical documentation, training data summaries, copyright compliance, and (for systemic risk models) adversarial testing and incident reporting.

Global differences are significant — a multi-jurisdiction strategy is needed. The EU is the strictest, the US is fragmented across states, China requires registration and content moderation. Compliance requires architectural decisions: logging, human oversight, documentation.

## Practical Example: Determining Compliance Requirements

When developing an AI system, the process of determining requirements includes several stages:

**Risk level classification:** Determine the category per the EU AI Act depending on the use case. Systems for recruiting, credit scoring, critical infrastructure — high-risk. Chatbots and emotion recognition systems — limited risk. Social scoring and mass biometrics — prohibited.

**Determining GPAI requirement applicability:** If the model is a General Purpose AI (trained on large data, solves a broad range of tasks), additional requirements apply. The 10^25 FLOPS threshold indicates systemic risk: adversarial testing, incident reporting, energy consumption reporting.

**Accounting for deployment geography:** For systems in the EU, the EU AI Act applies with fines up to 7% of turnover. In the USA, the voluntary NIST AI RMF with sector-specific regulations is in effect. In China, registration and content moderation are mandatory.

**Compiling the requirements list:** For high-risk: risk management system, data governance, technical documentation, automatic logging, human oversight, accuracy validation, cybersecurity, conformity assessment, registration in the EU database. For limited risk: informing users about interaction with AI, labeling AI-generated content. For GPAI: model technical documentation, information for downstream providers, copyright policy, training data summary.

For risk classification, the following logic applies: if the use case is on the prohibited list (social scoring, public biometrics) — UNACCEPTABLE. If on the high-risk list (employment, credit scoring, law enforcement, education, critical infrastructure) — HIGH. If on the limited-risk list (chatbot, emotion recognition, deepfake) — LIMITED. Otherwise — MINIMAL. This approach ensures systematic assessment and understanding of the timeline: prohibited practices cannot be used from February 2025, GPAI obligations from August 2025, full requirements for standalone high-risk from December 2027 (delayed from August 2026 by the Digital Omnibus).

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Governance
**Previous:** [[../17_Production_Inference/07_SGLang_and_Alternatives|SGLang and Alternatives]]
**Next:** [[02_AI_Risk_Management|AI Risk Management]]
