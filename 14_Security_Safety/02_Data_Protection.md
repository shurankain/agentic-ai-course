# Data Protection in LLM Systems

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Security and Safety
**Previous:** [[01_Prompt_Injection|Prompt Injection]]
**Next:** [[03_Agent_Security|Agent Security]]

---

## New Dimensions of the Privacy Problem

LLM applications create unique challenges for data protection. Every request potentially contains confidential information: personal data, business secrets, financial details. Leakage channels: the LLM provider itself gains access to the context, the model's response may contain sensitive information, logs and metrics retain data.

Traditional measures (encryption at rest/in transit, access control, audit logging) are necessary but insufficient. LLM adds a new vector: the model can "memorize" information from training data and "leak" it in responses. Even if the data was not in the training set, the model makes inferences based on patterns.

Example: a user asks a support chatbot about order status, providing an order number and email. The data enters the prompt, gets sent to the API provider, and is stored in logs. Each point is a potential leak. With RAG, the volume of transmitted data increases dramatically.

## PII Detection

Personally Identifiable Information is data that allows identifying a specific individual. It is critically important to detect and protect PII before sending it to the model provider.

PII categories by sensitivity level. Direct identifiers: full name, email, phone number, SSN/TIN — directly identify an individual. Indirect identifiers: date of birth, address, workplace — in combination also identify an individual. Sensitive categories: medical information, financial data, biometrics — require special protection under regulatory requirements.

Detection combines methods:

**Regex-based detection** quickly finds structured data: email addresses, phone numbers, card numbers, SSN. Patterns are stable and easily recognized. It cannot handle names, addresses, or unstructured data.

**NER (Named Entity Recognition)** models recognize named entities: people's names, organization names, geographic locations. Modern NER models (spaCy, Flair, Presidio) achieve high accuracy and require fine-tuning for specific domains.

**Contextual analysis** considers the surrounding context. The number "1234567890" could be a phone number or a random number. Context ("call me at 1234567890") helps classify it.

## Masking and Anonymization

Three main approaches: masking (replacement with a placeholder), anonymization (irreversible removal), pseudonymization (reversible replacement preserving relationships).

**Masking** replaces PII with a placeholder [EMAIL_1234], [PHONE_5678], preserving the data type. The model understands the context ("send the information to [EMAIL_1234]") without access to actual data. When needed, the placeholder is replaced back in the response.

**Consistent masking** is critical. If the same email appears multiple times, it must be masked with the same placeholder. Otherwise, the model loses the connection between mentions. A session-level mapping table is used.

**Anonymization** completely removes or irreversibly alters data. SSN is replaced with [SSN_REDACTED] without recovery. Stricter than masking, but does not allow de-anonymizing the response.

**Pseudonymization** generates fake data of the same type. Email john@company.com becomes user123@example.com. It preserves data format but requires caution — fake data must not correspond to real entries.

## System Prompt Protection

The system prompt contains business logic, constraints, and application context. Leakage reveals competitive advantages, system vulnerabilities, and internal information.

**Instructive protection** explicitly directs the model not to reveal instructions. It works against naive attacks but is vulnerable to sophisticated prompt injection.

**Minimizing sensitive information** is a reliable approach. The system prompt should not contain: API keys and credentials, internal URLs and endpoints, infrastructure details, employee names. All of this is moved to backend logic.

**Leak detection** in post-processing compares the response with the system prompt. Metrics: text similarity, embedding similarity, key phrase overlap. If the threshold is exceeded, the response is blocked.

**Splitting into public/private parts** stores sensitive instructions separately. The public part can be revealed without harm ("I am an assistant of company X"). The private part contains sensitive logic.

## Data Minimization and Retention

The data minimization principle: collect and store only the necessary minimum. For LLM: do not include unnecessary information in the context, limit conversation history, do not save full prompts if aggregated metrics are sufficient.

**Context windowing** limits the volume of history in each request. Instead of full history of 100 messages — the last 10 or a smart summary. This reduces the volume of potentially sensitive data.

**Retention policies** define storage duration. Conversations — 30-90 days, then deletion or anonymization. Detailed logs — 7-30 days. Aggregated metrics — 1+ year. Policies must comply with GDPR.

**Right to be forgotten** implements complete deletion of user data upon request: conversation history, embeddings in vector store, logs with user ID, caches. The complexity lies in data being distributed across many systems.

## Encryption and Key Practices

**Encryption at rest** protects data in storage. Conversations, embeddings, and logs are stored in encrypted form. Modern databases and object storage support transparent encryption.

**Encryption in transit** — HTTPS for all communications, including LLM API calls. Protection from interception at the network level.

**Application-level encryption** adds a layer for highly sensitive data. Conversations are encrypted with the user's key. If the database is compromised, the attacker obtains only encrypted text.

**Key management** determines access to encryption keys. Separation of duties: database administrators do not have access to keys. Hardware Security Modules (HSM) for storing master keys. Regular key rotation.

**Field-level encryption** encrypts only specific fields. Email and phone number are encrypted; metadata remains in plaintext for search and analytics.

## Audit Logging for Compliance

The audit log records all actions for analysis and compliance: who made the request, what data was in the prompt (hash or categories, not raw data), what response was received, what PII was detected, whether any operations were blocked.

**Immutability** of audit logs is critical — logs are not modifiable after being written. Append-only storage, write-once media, blockchain-based approaches.

**Tamper detection** identifies modification attempts. Cryptographic signatures on records, hash chains (each record includes the hash of the previous one), regular integrity checks.

**Retention** for audit logs is longer than for operational data — 1-7 years depending on regulations. PII in logs is anonymized or encrypted.

**Access** to audit logs is strictly restricted. Only security and compliance teams. Access to logs is also logged (meta-audit).

## Integration with External Providers

When using cloud LLM APIs (OpenAI, Anthropic), data leaves the perimeter. Additional risks require attention.

**Data Processing Agreements (DPA)** are signed with the provider. The DPA defines data protection obligations, audit rights, and breach responsibilities.

**Data usage policies** are critical. Is data used for training? How long is it stored? Who has access? Many providers offer options to opt out of training data usage (OpenAI API, Anthropic API does not use data for training by default).

**Residency requirements** may mandate that data does not leave the jurisdiction. This limits the choice of providers or requires using regional endpoints.

**Self-hosted models** eliminate the data transfer problem but create new ones: responsibility for infrastructure, updates, and patching falls entirely on your side.

## Key Takeaways

LLM systems create new challenges: every request potentially contains sensitive information transmitted to the model provider.

PII detection combines regex for structured data, NER for named entities, and contextual analysis for complex cases.

Masking replaces PII with placeholders while preserving data type. Consistent masking maintains connections between mentions.

The system prompt requires protection: minimizing sensitive information, instructive restrictions, leak detection in post-processing.

Data minimization limits the volume of collected and stored data. Context windowing, retention policies, and right to be forgotten are practical implementations.

Encryption at rest, in transit, and at the application level protects data at all stages. Key management and field-level encryption for sensitive fields.

Audit logging records all actions for compliance. Immutability and tamper detection ensure log reliability.

When using external LLM providers, the following are critical: DPA, data usage policies, and residency requirements. Self-hosted models eliminate data transfer but create operational obligations.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Security and Safety
**Previous:** [[01_Prompt_Injection|Prompt Injection]]
**Next:** [[03_Agent_Security|Agent Security]]
