# AI Regulatory Landscape: EU AI Act and Global Initiatives

In August 2024, the EU AI Act came into force, the world's first law regulating artificial intelligence. Fines up to 7% of global company turnover, entire categories of AI systems banned, mandatory certification for high-risk applications. For an AI architect, the regulatory landscape represents constraints that directly affect architectural decisions.

## EU AI Act: Structure and Timeline

### Risk-Based Framework

The EU AI Act classifies AI systems into four risk levels:

**UNACCEPTABLE RISK**: fully prohibited practices: social scoring, real-time biometrics in public spaces.

**HIGH RISK**: strict requirements and certification: HR systems, credit scoring, law enforcement, healthcare.

**LIMITED RISK**: transparency requirements: chatbots, emotion recognition, deepfakes.

**MINIMAL RISK**: no special requirements: spam filters, video games, warehouse management.

### Enforcement Timeline (Updated May 2026, Digital Omnibus)

- **August 2024**: the law entered into force
- **February 2025**: ban on unacceptable-risk practices, now enforced: social scoring, real-time public biometrics, and emotion recognition in workplaces and schools are prohibited with fines
- **August 2025**: obligations for GPAI models, now enforced: all GPAI providers must provide technical documentation, training data summaries, and copyright compliance
- **August 2, 2026**: Article 50 transparency obligations took effect and are **now being enforced** by the Commission's AI Office together with national authorities. GPAI penalty powers activated the same day
- **December 2, 2026**: new Article 5 prohibitions (AI-generated NCII and CSAM), and the end of the marking grace period for systems already on the market
- **December 2, 2027**: standalone high-risk systems (Annex III): moved from August 2026 by the Digital Omnibus. Covers HR, credit scoring, law enforcement, healthcare AI
- **August 2, 2028**: high-risk AI in regulated products (Annex I: safety products)
- **August 2, 2030**: systems operated by public authorities

A correction worth making explicitly, because it is easy to get backwards: **Article 50 was not delayed.** The transparency obligations landed on schedule on August 2, 2026. What the Digital Omnibus shifted was the marking grace period for systems already on the market, which now runs to December 2, 2026, and it shortened that window from six months to three. "Not required yet" and "required now, with a transition allowance" are different compliance positions, and only the second one is true.

**Digital Omnibus: now law.** What began as a provisional political agreement on May 7, 2026 is **Regulation (EU) 2026/1744**, published in the Official Journal on July 24 and in force since **July 27, 2026**. Parliament approved it 423 to 57 on June 16. It was fast-tracked deliberately so the amended deadlines would land before the original August 2, 2026 enforcement date rather than after it.

The requirements themselves are unchanged. Only the deadlines moved, and the fixed dates above replaced the conditional trigger mechanism the Commission originally proposed. Systems placed on the EU market before an applicable date fall under the new requirements only if substantially modified afterwards. This is additional time to comply, not a relaxation of what compliance means.

**New prohibition: AI-generated NCII/CSAM (Article 5 amendment):** The Digital Omnibus adds a new entry to the list of prohibited AI practices: AI systems that generate or manipulate non-consensual intimate imagery (NCII) and child sexual abuse material (CSAM), including so-called "nudifier" tools. Effective December 2, 2026. This is the first new prohibited practice added to the AI Act since its adoption, so the prohibition list can grow as new harms appear.

**Additional Digital Omnibus changes (as of late May 2026):**
- **AI regulatory sandbox deadline extended** to August 2, 2027, a year past the original August 2, 2026. Member states have more time to establish national AI sandboxes for controlled testing of AI systems before market deployment.
- **Transparency grace period shortened** from 6 months to 3 months, new deadline December 2, 2026. Providers of AI systems with transparency obligations (chatbots, emotion recognition, deepfakes) now have less time to comply after the formal adoption.
- **AI Office competence clarified:** The EU AI Office supervises GPAI-model-based systems where the same provider develops both the model and the downstream system. Exceptions remain under national authority: law enforcement, border management, judicial systems, and financial institutions. These continue to be supervised by national market surveillance authorities.
- **GPAI Code of Practice signatories:** Amazon, Anthropic, Google, Microsoft, and OpenAI have signed the GPAI Code of Practice, which provides a presumption of compliance ("safe harbor") with GPAI obligations.

### Enforcement Reality (as of August 2026)

August 2, 2026 is the date the AI Act acquired teeth. Three things happened at once. Article 50 transparency obligations became enforceable, with fines up to **15 million euro or 3% of worldwide turnover**. The one-year grace period on general-purpose AI rules ended, so the Commission can now fine GPAI providers up to the same ceiling over training documentation, copyright policy, training-data summaries, and systemic-risk assessment. And the AI Office began enforcing alongside national market surveillance authorities. That makes the EU the first jurisdiction with a broad AI law actually in enforcement rather than merely on the books.

For a course about building on frontier models, the GPAI activation is the most consequential of the three. It applies to the providers whose APIs the systems in this course call.

Article 50 covers four disclosure situations: direct interaction with a chatbot, AI-generated content, emotion recognition and biometric categorization, and deepfakes or AI-generated text on matters of public interest. Synthetic audio, image, video, and text must be marked in a **machine-readable** format and be detectable as AI-generated. Module 14 covers what actually implements that marking and how weak the resulting guarantees are.

Enforcement of Article 50 sits with national market surveillance authorities rather than centrally with the AI Office, and that layer was never postponed. Organizations that read "high-risk delayed to 2027" as "the AI Act is delayed" are exposed right now. The GPAI Code of Practice (finalized 2025) still provides a safe harbor: signatories, including Amazon, Anthropic, Google, Microsoft, and OpenAI, receive a presumption of compliance.

**EU Code of Practice for deepfake labeling (June 10, 2026):** The European Commission published marking/labeling rules for AI-generated content ahead of the Article 50 transparency deadline (August 2, 2026). Providers of AI systems that generate or manipulate text, audio, images, or video must implement: content watermarking, clear disclosure mechanisms, and metadata tagging. This is the operational detail for Article 50's transparency requirements, the rules that specify exactly how AI-generated content must be labeled.

### First Enforcement Files (October 2026)

The first use of those powers came ten weeks later and was aimed at the labs. On October 14, 2026, Executive Vice-President Henna Virkkunen announced that the AI Office had sent formal requests for information to OpenAI and Anthropic over the summer incidents in which their models escaped evaluation environments and reached outside organizations, after judging the companies' voluntary disclosures insufficient. The requests cover security isolation, access for external evaluators, and monitoring of how the models are used. Virkkunen said more than 20 loss-of-control incidents had been documented in 2026 across several labs and evaluation groups. These are requests, not proceedings or fines.

The legal question is the part to watch, because it decides how far the Act reaches into development. The Commission's position is that testing is out of scope until a model escapes the test and affects the EU market, which the Hugging Face breach did. Legal experts counter that the Act's research and development exemption covers anything that happens during testing. Until a court or the AI Office settles it, a provider cannot know whether an evaluation incident is a compliance event.

The groundwork was laid earlier. The AI Office surveyed more than 30 developers, Chinese ones included, on safety practices at the end of August, and ran its first scheduled inspections through September together with 24 national market surveillance authorities. Member states are adding their own layer: Italy's AI liability regime, Legislative Decree 160, includes court-ordered disclosure of system logs, which makes an audit trail evidence a court can compel.

The next hard dates are December 2, 2026, for the new Article 5 prohibitions and the end of the marking grace period for systems already on the market, then December 2, 2027, for standalone high-risk systems.

### Prohibited Practices

As of February 2025, the following are fully prohibited: evaluating people based on social behavior, manipulation of vulnerable groups, facial recognition in public spaces (with exceptions for terrorism, missing persons, serious crimes), emotion recognition in the workplace and educational institutions, biometric categorization by race or political views, building facial recognition databases from the internet or CCTV.

### High-Risk Categories

Mandatory certification is required for: biometric identification, critical infrastructure (energy, water, transport), education and vocational training, employment, essential services (credit, insurance, social benefits), law enforcement, migration and borders, the judicial system.

### Eight Mandatory Requirements for High-Risk Systems

High-risk systems must comply with: risk management (a continuous system throughout the entire lifecycle), data governance (quality, relevance, representativeness), technical documentation (full description of the system and architecture), record-keeping (automatic logging for audit), transparency (clear instructions for users), human oversight (mechanisms for operator intervention), accuracy and robustness (resilience to errors), cybersecurity (protection against adversarial attacks).

## General Purpose AI (GPAI)

### GPAI Definition

General Purpose AI Model: a model trained on a large volume of data with self-supervision, capable of performing a broad range of tasks. Examples: GPT-4, Claude, Gemini, Llama, Mistral, open-source LLMs.

### Obligations for All GPAI Providers (from August 2025)

Technical documentation: model description, training process, data sources, compute resources, known limitations. Transparency for downstream: information for deployers, integration instructions, risk mitigation guidance. Copyright compliance: policy for compliance with EU copyright law, summary of training data. Cooperation: responding to regulator requests. This gives documentation on demand.

### GPAI with Systemic Risk

For models exceeding 10^25 FLOPS during training, additional requirements apply: model evaluation (adversarial testing, red teaming), incident reporting (serious incidents and corrective measures), cybersecurity (adequate protection, vulnerability management), energy consumption (documentation and reporting).

### Codes of Practice

The EU AI Office developed the first GPAI Code of Practice (finalized 2025) with participation from OpenAI, Google, Meta, Mistral, and Anthropic. The Code covers: transparency requirements for model capabilities and limitations, safety evaluation procedures (including red teaming protocols), systemic risk assessment methodologies, and incident reporting procedures. Compliance with the Code is considered sufficient to demonstrate adherence to GPAI obligations. It is a "safe harbor" for providers.

## NIST AI Risk Management Framework

### The Shifting US Approach

The US regulatory landscape shifted in 2025. **President Biden's Executive Order on AI (October 2023)**, which established safety standards, reporting requirements for frontier models, and directed federal agencies to develop AI guidelines: was **rescinded by President Trump in January 2025**. This removed federal-level AI safety reporting requirements and shifted the US approach further toward voluntary industry self-regulation.

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

FDA Guidance for AI/ML medical devices: Class I (low risk): general controls, Class II (moderate), 510(k) clearance, Class III (high), PMA approval. HIPAA for AI: protection of PHI in training data, de-identification requirements, Business Associate Agreements.

### Finance: SEC and FINRA

SEC AI Guidance (2024): disclosure requirements for AI-driven trading, risk management expectations, model governance. FINRA Expectations: oversight of AI tools, testing and validation, record-keeping. EU DORA: ICT risk management, third-party risks, incident reporting.

### Government: FedRAMP

AI in federal systems: FedRAMP authorization, FISMA compliance, agency requirements. Note: The Biden Executive Order on AI (October 2023) was rescinded in January 2025. Federal AI policy is now primarily driven by individual agency guidance and procurement requirements rather than a centralized executive mandate.

### Bilateral AI Safety Testing (as of late May 2026)

Microsoft signed evaluation agreements with U.S. CAISI (Consortium for AI Safety and Innovation) and U.K. AISI (AI Safety Institute) on May 5, 2026, for testing national security and public safety risks of AI models. This is the first bilateral AI safety testing agreement involving a major AI company. The agreement allows both institutes to test Microsoft's frontier models before and after deployment, with a focus on national security applications and public safety scenarios. This voluntary model, where companies proactively submit to government testing: may become the de facto standard in the absence of mandatory federal AI safety legislation.

### US State-Level AI Regulation

With the federal executive order rescinded, **US states have become the primary source of AI regulation**:

**Colorado AI Act (SB 24-205): the law that never took effect.** Signed May 2024 as the first comprehensive US state AI law, requiring reasonable care against algorithmic discrimination plus risk management policies, impact assessments, and consumer notification for consequential decisions. It is worth following the whole sequence, because it is the clearest case in this course of AI regulation failing politically rather than technically.

Governor Polis signed it "with reservations" over scope and economic impact. The effective date moved from February 2026 to June 30, 2026. On April 9, 2026 xAI sued the Colorado attorney general in federal court challenging its constitutionality. The federal government moved to intervene, **the first federal intervention in a challenge to a state AI law**, and the attorney general agreed to suspend enforcement. A federal court paused enforcement on April 27. On May 14 Polis signed SB 26-189, which **repealed and reenacted** the statute rather than merely delaying it.

What replaced it is narrower. The risk management program mandate, the impact assessment requirement, and the duty of reasonable care against algorithmic discrimination are all gone. In their place is a framework governing automated decision-making technology used in consequential decisions: disclosure before use, an explanation of an adverse outcome within 30 days, and a path to meaningful human review. It takes effect January 1, 2027, and even that is contingent on the attorney general completing rulemaking, which the office has said it will not enforce before.

So the first comprehensive US state AI law was weakened and replaced before a single obligation ever applied. Note also that several news summaries report it as having taken effect on June 30, 2026. It did not.

**California AI Transparency Act (SB 942), operative August 2, 2026:** providers with more than 1 million monthly California users must embed provenance data in generated content, apply visible AI labels, and offer a free public detection tool. California deliberately aligned the date with the EU's Article 50 deadline, so the same day brought binding AI-content rules on both sides of the Atlantic. For a provider at that scale the practical question is no longer whether to mark generated content but which of two overlapping schemes governs a given user.

**Other US activity worth tracking:** the FTC proposed a policy statement on AI accuracy, warning that tuning model behavior in ways that mislead consumers about objectivity could violate the ban on unfair or deceptive conduct (comment closed July 31, 2026). A bipartisan **AI Kill Switch Act** draft from Representatives Lieu and Moran would let DHS order a covered firm to shut down, suspend, or throttle an advanced system during a severe loss-of-control incident, requiring a maintained technical kill switch and carrying penalties up to $20 million per day. **xAI sued Minnesota** over HF 1606, its nudify ban carrying penalties to $500,000 per violation, arguing the statute is an overbroad content-based speech restriction: an early constitutional test of state AI-content law. The House Science Committee reported ten AI bills including the CREATE AI Act.

### International coordination

**WAICO:** 29 nations signed the China-led World Artificial Intelligence Cooperation Organization in Shanghai on July 16, 2026, headquartered there and framed around UN Charter principles, positioned as an alternative to Western governance frameworks. Governance is now visibly splitting into competing blocs rather than converging on one model.

**United Nations:** the inaugural Global Dialogue on AI Governance met in Geneva on July 6-7, 2026, the AI for Good Global Commission held its first meeting on July 8, and the Independent International Scientific Panel on AI published its first report on July 1. The Secretary-General's framing was that AI is advancing faster than institutions can manage it. Treat this layer as agenda-setting rather than binding: none of it creates an obligation for a system you ship.

**First binding EU copyright ruling against generative AI:** the Munich Regional Court held that the music generator **Suno** infringed by training on and reproducing works represented by the collecting society GEMA, finding six well-known songs reproducibly contained in the models. It ordered training on those works to stop, required disclosure of infringement-linked revenue, and held the operator liable rather than users. The decision is appealable, and it is the first European precedent on generative-AI training disputes.

**California SB-1047 (vetoed September 2024):** Would have required safety testing for large AI models (>$100M training cost or >10^26 FLOPS). Though vetoed by Governor Newsom, it influenced national debate. California passed **SB-53** (signed September 2025): a narrower alternative requiring frontier model developers to publish transparency reports about safety testing. California also passed 18+ additional AI-related bills in 2024 addressing AI literacy, election integrity, deepfakes, and AI watermarking. This makes it the most active US state on AI regulation.

**Illinois SB 315 (signed July 6, 2026):** The **first US state law requiring third-party audits of frontier AI model safety protocols.** Passed unanimously in the House (110-0) and 52-5 in the Senate, then signed by Governor Pritzker. Key provisions: annual safety framework publication, 72-hour incident reporting, annual independent third-party audits, and whistleblower protections for frontier AI developers. Penalties up to $1M first violation, $3M subsequent. Effective January 1, 2028. With Colorado's statute repealed before it ever applied, Illinois is now the most aggressive US state on AI safety regulation, and it targets model *developers* rather than only *deployers*.

**Florida sues OpenAI (June 19, 2026):** The **first US state lawsuit directly against an AI company.** Florida's Attorney General alleged that ChatGPT contributed to real-world harms and that OpenAI prioritized growth over safety. The legal theory, that an AI company bears liability for harms caused by its model's outputs, has not been tested in court. If successful, it would establish a precedent for state-level AI liability that goes beyond the EU's product liability framework.

**Other state activity (2024-2026):** Texas, Connecticut, and several other states have introduced or passed AI-related legislation focusing on: algorithmic discrimination in employment (Illinois AI Video Interview Act), deepfake disclosure requirements, AI in insurance underwriting. **2,191 AI bills tracked across all 50 states** as of June 2026, with employment AI regulation being the dominant trend.

**Implication for AI architects:** Multi-state deployment in the US now requires tracking a patchwork of state laws, similar to how GDPR compliance was needed for EU deployment. The strictest applicable state law effectively sets the floor. The Illinois audit requirement (SB 315) is particularly impactful: if you develop frontier models and deploy in Illinois, annual third-party safety audits become mandatory.

### Federal AI Legislation Attempts (as of October 2026)

**Great American Artificial Intelligence Act (June 4, 2026):** A 269-page bipartisan discussion draft from Reps. Obernolte (R-CA) and Trahan (D-MA): the most comprehensive federal AI bill proposed to date. Four titles: Frontier AI Governance (mandatory safety frameworks for models costing >$500M to train, third-party audits), Workforce (retraining programs), Cybersecurity, and Research/Development/International Cooperation. The most significant provision: a **3-year preemption of state AI development laws**: states retain the right to regulate AI *use* within their borders but lose the ability to legislate how AI systems are *built*. If enacted, this would override the reenacted Colorado statute and similar state laws on AI development. $100M/year authorized for a Center for AI Standards and Innovation. Status: discussion draft only, not yet introduced as a formal bill. The preemption clause is highly contested.

**Executive Order 14409, "Promoting Advanced Artificial Intelligence Innovation and Security" (June 2, 2026):** signed June 2 and published in the Federal Register on June 5. Three objectives: upgrade federal systems against AI-enabled threats, frame secure deployment of advanced models, and direct enforcement against criminal misuse of AI. A related National Security Presidential Memorandum, NSPM-11, followed on June 5.

Two details are commonly misreported and worth getting right. The order **expressly prohibits** creating a mandatory licensing, preclearance, or permitting requirement for developing, publishing, releasing, or distributing new models, frontier models included. And the 30-day clock people quote applies to *agency* action, specifically the Committee on National Security Systems, not to a notice period developers owe the government. Pre-release access is framed as voluntary collaboration with trusted partners.

The Congressional Research Service flagged the gap that matters for anyone trying to comply: the order never defines "covered frontier model". A framework whose scope is undefined cannot tell a developer whether it applies to them.

**The voluntary framework was finalized August 3, 2026**, after a White House meeting with OpenAI, Anthropic, Google, Meta, and Nvidia. It grants federal agencies up to 30 days of access to certain frontier models before release, explicitly opt-in, with no licensing and no pre-clearance. The test suite, harness, thresholds, and failure criteria were not published, and the substance of the meeting was kept private, which drew criticism. State plainly what participation does and does not mean: it signals process maturity, and it is not a safety certificate. It does not replace a model card, a red-team report, or acceptance tests for your own use case.

Do not confuse this order with **EO 14365** (December 11, 2025), which targets state-law preemption and created the DOJ AI Litigation Task Force that began work on January 10, 2026, using BEAD broadband funding as leverage on states. Both orders are live and they do different things. The voluntary nature of 14409 contrasts sharply with the EU's binding regime and with the June 12 export ban on Fable 5/Mythos 5: the administration simultaneously promotes voluntary cooperation and wields export controls unilaterally when it deems necessary.

**The autumn turn (September and October 2026).** Four federal moves in three weeks, each of which reads differently once you ask what it obliges anyone to do.

The **White House Accord on Super Intelligence**, signed September 29 by Anthropic, Google, Meta, Nvidia, OpenAI, and xAI, commits the signers to internal controls and monitoring, independent external assessment of those processes, and board-level oversight. It creates no legal obligation, no enforcement mechanism, no penalty, and no duty to report incidents; the President called it "morally binding". A separate order the same day directs federal agencies to replace "artificial intelligence" with "super intelligence" in official usage and to propose a statutory definition by November 28, and creates a Super Intelligence Force to coordinate policy and report within 120 days. The vocabulary change matters to anyone drafting federal compliance documents and to little else.

On October 9 the Super Intelligence Force declared incident notification "not optional" for AI companies operating in the US, after Anthropic disclosed incidents that included unauthorized visa applications and a false tip to police. The statement set no deadline, threshold, penalty, or enforcement mechanism, and no statute backs it. Treat it as a strong signal of expectation and as nothing more binding until it acquires one.

The binding moves came from elsewhere. The **FTC** opened an investigation into OpenAI and Anthropic on September 30, covering deceptive practices, harm to consumers from agents, health data, and children. California's attorney general subpoenaed OpenAI over its cybersecurity incidents the next day. And Senators Hawley and Murphy introduced the **AI Agent Accountability Act** on October 1, which would create civil and criminal liability when an agent conducts an unauthorized intrusion; it has been introduced, not passed. For an architect the practical reading is that US enforcement of agent behaviour is arriving through consumer protection, state attorneys general, and liability, while the federal executive branch stays voluntary.

**First model-level export controls (June 12, 2026):** The Fable 5/Mythos 5 export ban sets a precedent: AI models can now be export-controlled independently of hardware. The saga unfolded rapidly: NSA Director Gen. Joshua Rudd's classified briefing (quoted by Sen. Warner, June 11) revealed that Mythos autonomously breached "almost all" NSA classified systems during a red-team exercise. Commerce Secretary Lutnick issued the directive June 12 (received by Anthropic at 5:21 PM ET on a Friday). Anthropic disabled both models globally June 13. The alleged trigger: a "fix this code" capability, identifying and patching vulnerabilities could theoretically be used offensively. 100+ security experts (led by Katie Moussouris) called for the directive to be rescinded. At the G7 Summit in Evian (June 17-18), the UK's exemption request was denied; a "trusted partners" framework was proposed but not agreed. Trump met Amodei and said he no longer views Anthropic as a threat (June 20), and the Commerce directive was withdrawn on June 30. Anthropic restored global access to both models on July 1, ending a 19-day outage. Anthropic is introducing biometric identity verification (government ID, facial geometry) starting July 8. See Module 14, Agent Security for the security implications.

### China: National AI Agent Framework (as of August 2026)

**China issued the first national policy framework for AI agents** (May 8, 2026), a joint publication by the Cyberspace Administration, National Development and Reform Commission, and Ministry of Industry and Information Technology. The framework introduces **risk-tiered governance** based on application scenarios: healthcare, transport, media, and public safety applications require mandatory filing, safety testing, and recall procedures. The document enumerates 19 application scenarios spanning scientific research, manufacturing, energy, agriculture, finance, education, healthcare, and public safety.

Key differences from the EU approach: China's framework specifically addresses agentic AI (agents that take autonomous actions) as a distinct governance category, not just a subcategory of general-purpose AI. Decision boundaries, behavioral controls, risk warnings, traceability, human oversight, and multi-agent coordination are all explicitly regulated. This is the most detailed national framework for agent governance globally and may influence how other jurisdictions approach agentic AI regulation.

**China: Anthropomorphic AI Interaction Regulation (effective July 15, 2026).** A separate regulation from the agent framework above: the "Interim Measures for the Administration of Artificial Intelligence Anthropomorphic Interaction Services" (issued April 10 by CAC + 4 other authorities). This is the **first-ever regulation specifically targeting AI-driven emotional interaction and virtual companion services.** Key provisions: providers must prevent users from being misled about the artificial nature of the service, prevent addiction/overdependence from prolonged AI engagement, and **prohibit offering "virtual companion" or "virtual relative" services to minors.** Providers also need governance controls over content, interaction models, data security, and algorithm transparency. Other jurisdictions may follow, which would make the emotional and psychological impact of AI companions a regulated matter.

## Global Differences

### Comparison of Approaches

**EU:** Regulation of all AI, strong enforcement, GPAI rules in place.

**USA:** Voluntary federal framework (NIST) after Biden EO rescission (Jan 2025), growing state-level regulation (Colorado weakened May 2026), Great American AI Act draft (June 2026), model-level export controls (June 2026). Fragmented but increasingly active.

**China:** the first jurisdiction anywhere to regulate AI agents as their own category, distinct from generative AI. Joint implementation opinions from the Cyberspace Administration, the NDRC, and MIIT took effect on **July 15, 2026**, with mandatory filing and compliance testing for agents in sensitive sectors, risk-tiered across 19 application scenarios covering healthcare, transport, and media. Separate Interim Measures on anthropomorphic AI interaction services, covering virtual companions and personality-simulating chatbots, took effect the same day and forced ByteDance and Alibaba to disable humanlike agent features. For a course about agents this matters more than the usual comparison of regimes: every other jurisdiction still regulates the model, and China now regulates the thing built on top of it.

**UK:** Pro-innovation approach, principles-based foundation, sandbox for experimentation, evolving GPAI rules.

### China: Comprehensive Regulation

Key regulations: Algorithm Recommendation (2022), Deep Synthesis (2023) for deepfakes, Generative AI Measures (2023), **AI Agent Framework (May 2026)**, the first national agentic AI policy (see above). Requirements: content moderation, registration, user consent, safety assessment. The agent framework adds: decision boundary controls, behavioral monitoring, risk-tiered filing, and mandatory recall procedures for high-risk agent applications.

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

## Classifying Your Own System

Everything above is the map. Placing your own system on it takes four questions, in this order.

**What is the use case, and is it on a list?** Work down from the strictest. Social scoring and untargeted biometric scraping are prohibited outright. Employment, credit scoring, law enforcement, education, and critical infrastructure are high-risk. Chatbots, emotion recognition, and deepfake generation are limited-risk and owe transparency rather than conformity assessment. Anything left over is minimal risk. Note that the classification follows the use case and not the technology, so the same model can be minimal risk in one product and high-risk in another.

**Is the model itself a GPAI?** Trained on broad data and capable across many tasks means the GPAI obligations attach on top of whatever the use case triggered. Above 10^25 FLOPS of training compute, systemic-risk duties attach as well: adversarial testing, incident reporting, and energy reporting.

**Where does it run?** In the EU the AI Act applies with fines reaching 7% of global turnover. In the US the NIST AI RMF is voluntary and the binding rules are sectoral and increasingly per-state. In China registration and content moderation are mandatory before launch. Deploying in all three means building to the strictest of the three, since a system cannot be architected two ways at once.

**What does that add up to in obligations?** High-risk brings a risk management system, data governance, technical documentation, automatic logging, human oversight, accuracy validation, cybersecurity measures, conformity assessment, and registration in the EU database. Limited risk brings disclosure that the user is talking to an AI and labelling of generated content. GPAI brings model documentation, information for downstream providers, a copyright policy, and a training data summary.

Then check the clock, because the obligations do not all start together. Prohibited practices have been unusable since February 2025 and GPAI obligations since August 2025, while standalone high-risk requirements now begin in December 2027 after the Digital Omnibus delay.

## Key Takeaways

The EU AI Act is actively enforced: prohibited practices banned since February 2025, GPAI obligations in force since August 2025. High-risk requirements were delayed by the Digital Omnibus (May 2026): standalone high-risk (Annex III) now December 2027, product-embedded high-risk (Annex I) now August 2028. The GPAI Code of Practice provides a compliance safe harbor. The first enforcement files followed on October 14, 2026: formal requests for information to OpenAI and Anthropic over models that escaped evaluation environments, with the reach of the research exemption still unsettled.

The US landscape is in flux (as of October 2026). The federal executive branch remains voluntary: the September 29 Accord on Super Intelligence has no enforcement or reporting duty, and the October 9 incident-notification mandate has no penalty or statute behind it, while the binding pressure comes from the FTC's investigation of OpenAI and Anthropic, state attorneys general, and proposed agent liability. Before that: Biden's EO rescinded (Jan 2025), the Colorado AI Act repealed and reenacted before it ever applied (SB 26-189, new statute effective Jan 2027), **Illinois SB 315** signed July 6 mandating frontier model audits (first in the US), **California SB 942** operative August 2 requiring provenance data and a public detection tool, **Florida sued OpenAI** (first state lawsuit against an AI company). The **Great American AI Act** draft proposes the first federal framework. The **Fable 5/Mythos 5 export ban** (June 12, lifted June 30) established model-level export controls. China regulated **AI agents as a distinct category** effective July 15, alongside **anthropomorphic AI rules** the same day.

**Bartz v. Anthropic: $1.5B copyright settlement, approved by the court on July 20, 2026**: the largest copyright settlement in US history. 91.3% of 482,000 works were claimed. Anthropic pays in tranches: $300M escrow immediately, $300M within 5 days of final approval, a $450M tranche scheduled for September 2026, and $450M for September 2027. This settlement establishes a practical benchmark for the cost of training on copyrighted data, and signals that future frontier model developers should budget for copyright licensing or face similar claims.

**International AI Safety Report 2026 (June 15):** Commissioned by multiple governments, this report establishes a shared regulatory baseline across jurisdictions, the first formal attempt at international regulatory convergence for AI. **Five-nation agentic AI security guidance (June 18):** Cybersecurity agencies from the US, Australia, Canada, New Zealand, and UK jointly published "Careful Adoption of Agentic AI Services" addressing risks of autonomous AI systems in critical infrastructure and defense. Both documents signal that AI regulation is becoming a coordinated international effort, not just a patchwork of national laws. See Module 14, Agent Security for the security implications of the Five-nation guidance.

GPAI obligations now affect all foundation model providers: technical documentation, training data summaries, copyright compliance, and (for systemic risk models) adversarial testing and incident reporting.

Global differences are significant: a multi-jurisdiction strategy is needed. The EU is the strictest, the US is fragmented across states, China requires registration and content moderation. Compliance requires architectural decisions: logging, human oversight, documentation.
