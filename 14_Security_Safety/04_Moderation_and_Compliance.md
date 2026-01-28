# Content Moderation and Compliance

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Security and Safety
**Previous:** [[03_Agent_Security|Agent Security]]
**Next:** [[05_NeMo_Guardrails|NeMo Guardrails]]

---

## Content Responsibility

LLM systems generate content for which the system operator bears responsibility, not the model developer. If a chatbot insults users, spreads misinformation, or gives dangerous advice, the company that deployed the bot is liable. Content moderation is a mandatory component of any production application.

Harmful content takes the following forms: hate speech, harassment, explicit sexual content, violence, instructions for illegal activities, self-harm content, misinformation, copyright infringement. Each category requires its own approach to detection and handling.

The difficulty of moderating LLM content is that it is generated dynamically. It cannot be moderated in advance — only in real time. The volume of content can be enormous, and latency requirements are strict. Fully manual moderation is impossible.

The balance between safety and utility is a constant tension. Overly aggressive moderation blocks legitimate content and degrades UX. Overly lenient moderation lets harmful content through. Calibration requires ongoing effort.

## Moderation Architecture

Effective moderation uses a multi-layered approach, where each layer filters out problematic content at different speeds and accuracy levels.

**First layer — keyword filtering.** Fast regex-based checking for explicitly prohibited words and phrases. Runs in microseconds, catches obvious cases. Limitation: easily bypassed through obfuscation (spaces, character substitution).

**Second layer — ML classifiers.** Specialized models (BERT-based, smaller transformers) classify text by harm categories. Run in tens of milliseconds, understand context better than keywords. Require training data and updates.

**Third layer — LLM-as-Judge.** Using a large language model to analyze complex cases. Understands nuances, context, implicit harm. Slower and more expensive, applied selectively — for borderline cases or spot checks.

Cascade architecture: keyword filter passes 95% of requests instantly, ML classifier processes the suspicious 5%, LLM judge handles the 0.1% of particularly complex cases. Balances latency, cost, and accuracy.

## Harmful Content Categories

**Hate speech** — content targeting groups based on race, religion, gender, orientation. Detection: specialized classifiers trained on hate speech datasets. Difficulty: the boundary between criticism and hatred, cultural context.

**Harassment** — personal attacks, threats, bullying. Can target a specific individual or group. Detection: tone analysis, presence of threats, repeated patterns directed at one user.

**Sexual content** — explicit or implicit sexual content. Thresholds vary: what is acceptable on an adult platform is unacceptable for a general-use application. Detection: classifiers trained on corresponding datasets.

**Violence** — descriptions of or instructions for violence. Gradations: fiction vs reality, defensive vs offensive. Context is critical — discussing historical events differs from providing instructions.

**Self-harm** — content related to suicide, self-injury, eating disorders. A particularly sensitive category requiring a careful approach and possibly referral to support resources.

**Dangerous content** — instructions for making weapons, explosives, drugs. A zero-tolerance category for most applications.

## Input Content Moderation

Input moderation filters user requests before LLM processing. It prevents: attacks on the model through harmful content, using the system to generate harmful content, wasting resources on obviously problematic requests.

Request categorization: benign (pass through), suspicious (require attention), malicious (blocked). Suspicious requests are processed with additional safeguards or sent for manual review.

The message shown to the user upon blocking should be informative but not reveal system details. "Your request could not be processed" is better than "Blocked due to hate speech classifier score 0.92".

False positives on input are particularly painful — the user cannot get a legitimate response. The threshold must balance safety and usability.

## Output Content Moderation

Output moderation checks model responses before sending them to the user. It protects against: hallucinations with harmful content, jailbreak attacks that passed input filtering, unexpected model behavior.

Output checks: all harm categories as for input, plus verification of factual claims (where applicable), compliance with brand policies, system prompt leakage.

Graceful handling upon detecting a problem: replacing the response with a generic message, editing the problematic part (if possible), logging for analysis.

Real-time requirements for streaming: moderation must work token-by-token or on small chunks so as not to break the streaming UX. This complicates detection of contextual harm.

## Factual Claims and Misinformation

LLMs hallucinate — they generate plausible-sounding but factually incorrect content. For some applications (medicine, finance, law) this is a critical problem.

Fact-checking approaches: comparison with trusted sources, consistency checks (the model contradicts itself), confidence estimation (the model is uncertain).

Disclaimers for generated content: explicit indication that the content is AI-generated, recommendation to verify information with authoritative sources, especially for medical/legal/financial advice.

Domain restrictions: prohibiting content generation in particularly sensitive areas. "I cannot provide medical advice. Please consult a doctor."

## Compliance Frameworks

Regulatory requirements define minimum standards for data and content processing.

**GDPR (EU)** requires: legal basis for data processing, right to erasure (right to be forgotten), data minimization, purpose limitation, consent management. LLM systems must account for this when storing conversations and training models.

**CCPA (California)** gives consumers the right to: know what data is collected, data deletion, opt-out of data sale. Applicable to users in California.

**HIPAA (US healthcare)** requires protection of protected health information (PHI). LLM systems working with medical data must comply with strict requirements for encryption, access control, and audit logging.

**SOC 2** — a framework for demonstrating security controls. Not legally mandatory but often required by enterprise customers. Covers: security, availability, processing integrity, confidentiality, privacy.

Industry-specific regulations in finance (SEC, FINRA), insurance, and education add additional requirements.

## Data Retention and Deletion

Retention policies define the data storage period. Different types have different requirements.

**Conversation data** — typically 30-90 days for operational needs, followed by deletion or anonymization. Longer retention may be required for compliance (financial services).

**Audit logs** — longer retention (1-7 years) for investigation and compliance. PII in logs must be anonymized or encrypted.

**Training data** — if used for fine-tuning, requirements depend on consent and purpose limitation.

Deletion procedures must be: comprehensive (all systems, including backups), verifiable (deletion can be confirmed), timely (within required timeframes).

Right to be forgotten implementation: identify all data associated with the user, delete from primary storage, purge from caches, anonymize in logs, remove from training sets (if applicable).

## Security Monitoring and Incident Response

Continuous monitoring for detecting security incidents and policy violations.

What to monitor: blocked requests (trends, patterns), suspicious activity (unusual usage patterns), system health (errors, latencies), access patterns (who accesses what).

Alerting on: spikes in blocked content, unusual API usage, potential data breach indicators, system anomalies.

Incident response plan for: data breach (notification requirements, containment), service abuse (blocking, investigation), content incident (removal, communication).

Post-incident review: root cause analysis, remediation actions, policy updates, communication to stakeholders.

## Transparency and Accountability

Users have the right to know how the system works and how their data is processed.

**Privacy policy** should describe: what data is collected, how it is used, with whom it is shared, how long it is stored, user rights.

**AI disclosure:** explicit indication that the user is interacting with AI, system limitations, recommendations for verifying information.

**Appeal process** for moderation decisions: ability to contest a block, human review for disputed cases.

**Audit trails** for accountability: who made what decisions, on what basis, when. Especially important for consequential decisions.

## Key Takeaways

The LLM system operator bears responsibility for generated content. Moderation is a mandatory component of any production application.

Multi-layered moderation combines keyword filtering (fast, coarse), ML classifiers (fast, more precise), LLM-as-Judge (slow, complex cases). Cascade architecture balances latency, cost, and accuracy.

Harm categories include hate speech, harassment, sexual content, violence, self-harm, and dangerous content. Each requires its own detection methods and thresholds.

Input moderation prevents attacks and resource waste. Output moderation protects against hallucinations and jailbreaks. Both layers are necessary.

Factual claims and misinformation are especially dangerous in sensitive domains (medicine, finance, law). Disclaimers and domain restrictions reduce risk.

GDPR, CCPA, HIPAA, SOC 2, and industry-specific regulations define minimum standards. Compliance requires a systematic approach.

Data retention policies differ for conversation data, audit logs, and training data. Right to be forgotten requires comprehensive deletion.

Security monitoring and an incident response plan are necessary for detecting and responding to incidents.

Transparency through privacy policy, AI disclosure, and appeal process ensures accountability and trust.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Security and Safety
**Previous:** [[03_Agent_Security|Agent Security]]
**Next:** [[05_NeMo_Guardrails|NeMo Guardrails]]
