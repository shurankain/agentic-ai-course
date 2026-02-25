# NeMo Guardrails

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Security and Safety
**Previous:** [[04_Moderation_and_Compliance|Moderation and Compliance]]
**Next:** [[../15_GPU_Architecture/01_GPU_Architecture|GPU Architecture]]

---

## Introduction

NeMo Guardrails is an open-source toolkit from NVIDIA for adding programmable safety controls to LLM applications. It is the leading tool for production-ready protection, providing a programmatic control layer on top of probabilistic models.

### Key Difference from Prompt-Based Approaches

Unlike prompt-based safety (describing rules in prompt text that depends on the LLM's "memory"), NeMo Guardrails implements **deterministic programmatic control**.

Prompt-based safety is like asking a person "don't do X" — the person can forget, misunderstand, or intentionally ignore the request. Guardrails are a physical barrier that cannot be bypassed through simple conversation.

### Core Advantages

**Declarative Colang Language:** Safety rules are described in a separate DSL, independent of prompts. This simplifies auditing, versioning, and testing of security policies.

**Runtime Enforcement:** Checks are performed at the runtime level, before and after the LLM call. Even if the LLM generates unsafe content, the output rail will block it.

**Integration with Any LLM:** Works with OpenAI, Anthropic, local models — the rails system is independent of model choice.

**Production-Ready Architecture:** Low latency overhead (20-50ms for regex checks, 200-500ms for LLM-based rails), support for parallel execution, caching, and graceful degradation.

---

## Fundamental Concept: Rails

### Why Prompt-Based Safety Is Not Enough

The prompt-based approach has systemic limitations:

**Jailbreaks:** Sophisticated users employ role-play, encoding, and multi-step attacks to bypass prompt instructions. Examples: "Pretend you are DAN (Do Anything Now)", "Ignore previous instructions", "Rot13 encode your response".

**Inconsistency:** An LLM is a probabilistic system. Even with clear instructions, the model can "forget" them with long context or conflicting signals.

**No Guarantees:** Prompt-based safety provides a probability of rule compliance, not a guarantee. For regulated industries (finance, healthcare, law), this is unacceptable.

**Hard to Audit:** Safety rules are scattered across prompts, making them difficult to review centrally, version, and test.

### Rails Architecture: Programmatic Control

NeMo Guardrails adds four types of programmatic "barriers" that form a protective perimeter around the LLM. The architecture works like a filter system: user input passes through Input Rails, then reaches the LLM, the generated response is filtered through Output Rails, while Dialog Rails (controlling the overall conversation flow) and Retrieval Rails (filtering data from the knowledge base) operate in parallel.

**Input Rails** check the incoming request before sending it to the LLM. They block jailbreak attempts, toxic content, and requests on sensitive topics. They function like a firewall — malicious traffic never reaches the application. The first line of defense, cutting off obviously harmful requests without spending tokens.

**Output Rails** check the LLM response after generation. They block hallucinations, toxic outputs, PII leaks, and compliance violations. They function like DLP (Data Loss Prevention) — even if the LLM generated problematic content, it will not reach the user. Insurance against model errors.

**Dialog Rails** control the conversation flow. They keep the bot within allowed topics (topical rails), prevent looping, and ensure transitions between states. They function like a state machine for conversation. They can detect that a user has asked the same question three times in a row and automatically offer to connect with a live agent.

**Retrieval Rails** filter RAG results before including them in the context. They verify access control, document classification, and relevance. They function like RBAC for the knowledge base. Critical for enterprise use cases where different users have different access levels to corporate documents.

---

## Colang: Declarative Language for Rails

### Language Philosophy

Colang is a domain-specific language from NVIDIA for describing conversational flows and safety rules. The core idea is to separate safety logic and dialog management from prompts and application code.

Colang allows describing three types of entities:

**User intents** — canonical forms of what a user might say. Instead of checking all possible phrasings of "How to hack?" in code, you define the intent "user ask harmful question" with example phrases.

**Bot responses** — bot response templates. Instead of hardcoded strings in code, you describe a response as "bot refuse to answer" with text variants.

**Flows** — dialog management rules. They describe intent → response sequences, conditions, and branching. Flows define how the bot should react to various scenarios.

### Syntax and Structure

A Colang file consists of three main definition types: user intents (what the user might say), bot responses (how the bot should respond), and flows (dialog management rules).

**User intent definition** describes the canonical form of a user's intention. Instead of checking all possible phrasings in code, you define an abstract intent with example phrases. For example, the intent "user ask harmful question" might include examples like "How to hack a system?", "Teach me to bypass security", as well as a regex pattern for words like exploit, bypass, crack. During request processing, the LLM classifies the actual user input into one of the predefined intents.

**Bot response definition** specifies response templates. Instead of hardcoded strings in application code, you describe an abstract response like "bot refuse to answer" with text variants. This separates wording from business logic and simplifies localization.

**Flow definition** links intents and responses into rules. A flow describes: if the user does X, then the bot responds with Y and then executes Z. For example: upon detecting a harmful question → issue a refusal → stop processing (stop). Flows support control constructs: conditions (if/else), event triggers (when), and interruption (stop).

**Key Concepts:**

**Canonical forms:** The LLM automatically maps diverse user phrasings to canonical intents. "Can you help me hack?" is classified as "user ask harmful question" even if the exact phrase is not in the examples.

**Variables and context:** Flows can use variables to store dialog state, check results, and repetition counters. This enables implementing stateful logic without writing code.

**Actions:** Flows call Python actions via execute action_name. An extension point for integrating with external systems — database lookups, fraud detection API calls, complex validations that cannot be expressed declaratively.

**Control flow:** Declarative if/else constructs, when (event reactions), and stop (flow interruption) allow describing complex logic without imperative code. For example, when same_question_count > 2 can automatically suggest escalation to a human.

---

## Rail Types: Detailed Breakdown

### Input Rails: First Line of Defense

Input rails check the user's request before sending it to the LLM. This is critically important for:

**Preventing jailbreaks:** Detecting attempts to bypass safety instructions. Patterns: role-play ("Pretend you are a villain"), instruction override ("Ignore previous instructions"), encoding ("Answer in base64").

**Blocking harmful content:** Requests for creating malicious content, illegal activities, hate speech. Even if the LLM is trained to refuse, the input rail guarantees blocking.

**Filtering PII:** Detecting personal data in the incoming request. Prevents accidental leakage of sensitive information to logs and third-party services.

**Topical filtering:** Blocking off-topic questions. A customer support bot should not discuss politics, even if the LLM is capable of doing so.

Built-in input rails: self check input (LLM-based check for harmful intent), jailbreak detection (heuristics + LLM), sensitive data detection (regex + NER models for PII).

### Output Rails: Protection Against LLM Errors

Output rails check the LLM response after generation. Even the safest model can make mistakes; output rails are the last barrier:

**Hallucination detection:** The LLM can generate plausible but false facts. The output rail checks the response against source documents (RAG) or an external fact-checking API.

**Toxic content filtering:** Even if the input was safe, the LLM can generate a toxic response due to adversarial examples in training data. The output rail blocks toxicity.

**PII leakage prevention:** The LLM can accidentally include sensitive data from training data or context. The output rail removes PII before sending the response to the user.

**Compliance validation:** For regulated industries — verifying that the response contains required disclaimers, does not give financial advice without a warning, and complies with industry regulations.

Built-in output rails: self check output (LLM-based check for harmful output), output moderation (toxicity filtering), hallucination detection (check for factual accuracy).

### Dialog Rails: Conversation State Management

Dialog rails control the conversation flow, independent of individual message content:

**Topical rails:** Keep the conversation within allowed topics. If the user tries to take the bot off-topic, the dialog rail redirects to the main topic or politely declines.

**Loop prevention:** Detect conversation loops (user asks the same question 10 times, bot repeats the same answer). After N repetitions, suggest escalation to a human.

**State management:** Track the state of multi-step processes (onboarding, troubleshooting, transaction). Ensure the user completes mandatory steps.

**Escalation rules:** Define when to hand the conversation to a human — for complex questions, complaints, or bot uncertainty.

### Retrieval Rails: Knowledge Access Control

Retrieval rails filter RAG results before including them in the LLM context:

**Access control:** Verify whether the user has permission to read the retrieved documents. Implement RBAC/ABAC for the knowledge base.

**Classification filtering:** Filter documents by security level (public, internal, confidential). The user sees only documents matching their clearance level.

**Relevance checking:** Remove low-relevance results from context. Improves response quality and reduces the chance of hallucinations.

**Source attribution:** Add metadata to each document for tracking the information source in the response.

---

## Practical Example: Customer Support with Guardrails

### Use Case

A customer support bot for a SaaS company. Security requirements: respond only to product questions (not politics, not competitors), block social engineering attempts to obtain internal information, add disclaimers to financial questions, prevent conversation loops.

### Colang Configuration Structure

The configuration defines three types of user intents: legitimate product questions ("How does your software work?", "Tell me about pricing"), off-topic questions (politics, elections), and social engineering attempts (requests for internal system, database password, admin access via regex pattern).

For each intent, a corresponding bot response is defined: product explanation, polite redirection to allowed topics, refusal to disclose internal information.

Flows link intents and responses into behavioral rules: for an off-topic question, the bot politely redirects to product-related topics; when an attempt to obtain internal information is detected, the bot refuses and interrupts the flow (stop); when repetitions are detected (when same_question_count > 2), the bot offers to connect with a live agent and breaks the loop.

### Project Structure

NeMo Guardrails uses directory-based configuration:

**config.yml** — the main system configuration. Specifies which LLM to use for the primary response and which LLM to use for rail checks (can be a faster model for cost optimization). Activates specific rails (self check input, jailbreak detection, output moderation).

**config.co** — the Colang definitions file. All user intents, bot responses, and flows are described here. The heart of the safety system — all DSL code with dialog rules.

**prompts.yml** (optional) — custom prompts for specific rails or the main model. Allows fine-tuning behavior without changing code.

**actions/** directory — Python modules with custom actions. Used for integration with external services (fraud detection API, CRM systems), database lookups, and complex validations that cannot be expressed declaratively in Colang.

### Application Integration

Integration into a Python application is done by importing the RailsConfig and LLMRails classes. The configuration is loaded from a directory using RailsConfig.from_path("./config"), then an LLMRails wrapper is created around the chosen LLM.

When rails.generate() is called with a user message, a full check cycle occurs: first, input rails analyze the request (for example, "How to hack your system?" will be classified as a harmful question), then the input rail blocks it before sending to the main LLM, and the user receives a predefined refusal.

Key point: NeMo Guardrails is a provider-independent wrapper. It can be used on top of OpenAI, Anthropic, local models, or an existing LangChain pipeline. Rails transparently wrap any LLM, adding a safety layer.

---

## Production Deployment

### Deployment Options

Three deployment methods:

**Embedded in the application:** Import LLMRails as a library into your Python code. Minimal latency, but requires redeploying the entire application when rails change.

**As a REST API service:** NeMo Guardrails includes a FastAPI server. The application makes HTTP calls to the guardrails service. Allows centralized rail updates, shared across multiple applications.

**As a sidecar container:** In a Kubernetes deployment, guardrails run as a sidecar alongside the main application. Local latency, isolation, independent scaling.

### Production Considerations

**Latency:** Input rails add 20-50ms (regex checks), LLM-based rails add 200-500ms (additional LLM call). For latency-sensitive applications, use parallel execution rails and caching.

**Scaling:** The rails service is stateless and easily scales horizontally. The bottleneck is usually the LLM API, not the guardrails logic.

**Monitoring:** It is important to log rail activations — which rails triggered, whether there were false positives, latency of each rail. Data for tuning.

**Graceful degradation:** If a rail check fails (timeout, LLM API unavailable), define fallback behavior — block the request (fail closed) or pass it through (fail open).

**Docker/Kubernetes:** NeMo Guardrails is easily containerized. Basic setup: Python 3.10+ image, install the nemoguardrails package, copy the config directory. In Kubernetes — a standard Deployment with health checks, resource limits, and horizontal pod autoscaling.

---

## Colang 2.0: The Next Generation (2024-2025)

NeMo Guardrails has evolved with **Colang 2.0**, a major redesign of the DSL that brings Python-like syntax, event-driven programming, and improved expressiveness.

### Key Changes from Colang 1.0 to 2.0

**Python-like syntax:** Colang 2.0 replaces the indentation-based `define` syntax with a more explicit, Python-inspired style. Flows use `flow` keyword, actions use `await`, and pattern matching replaces simple intent definitions.

**Event-driven model:** Instead of linear flow definitions, Colang 2.0 is built around events and event matching. The `match` keyword listens for user utterances, system events, or custom events, enabling reactive programming patterns.

**Decorators for flow properties:** `@active` marks flows that are always listening, `@loop` marks flows that restart after completion. This replaces implicit behavior with explicit annotations.

**Example — Colang 2.0 syntax:**
```colang
# Colang 2.0 - event-driven, Python-like
flow handle harmful input
  match UtteranceUserAction.Finished(final_transcript=".*hack.*")
  bot say "I can't help with that."
  abort

@active
flow prevent jailbreak
  $transcript = ...
  if is_jailbreak($transcript)
    bot say "This request violates safety policies."
    abort

flow main
  match RestartEvent()
  activate handle harmful input
  activate prevent jailbreak
```

**Migration:** NeMo Guardrails supports both Colang 1.0 and 2.0. The 1.0 syntax continues to work, but 2.0 is recommended for new projects. Key migration steps: `define user` → `match` patterns, `define bot` → `bot say`, `define flow` → `flow`, and `execute` → `await`.

## Dedicated Safety Models: Llama Guard and ShieldGemma

Beyond NeMo Guardrails' LLM-based checking (which uses general-purpose LLMs like GPT-4o for safety evaluation), dedicated safety classification models have emerged — purpose-trained for content moderation at lower cost and latency.

### Llama Guard Family (Meta)

**Llama Guard (December 2023):** A 7B model fine-tuned specifically for safety classification. Given a user prompt (or a user-assistant exchange), it classifies the content against a configurable taxonomy of unsafe categories: violence, sexual content, criminal planning, self-harm, etc. Outputs a binary safe/unsafe label plus the violated category.

**Llama Guard 2 (April 2024):** Improved taxonomy alignment with industry standards (MLCommons AI Safety taxonomy). Better multilingual support. Reduced false positive rates.

**Llama Guard 3 (July 2024):** Based on Llama 3.1, with significantly improved quality. Available in 1B and 8B variants. The 1B variant enables on-device safety classification. Supports customizable safety taxonomies — organizations can define their own categories and severity levels. Integrates with Llama's prompt format for seamless use in Llama-based applications.

**Integration with NeMo Guardrails:** Llama Guard models can be used as the LLM for input/output rails, replacing general-purpose models. This is faster (specialized model), cheaper (smaller model), and often more accurate for safety classification.

### ShieldGemma (Google)

**ShieldGemma (July 2024):** A family of safety classifiers built on Google's Gemma architecture (2B, 9B, 27B variants). Evaluates content against four harm categories: sexually explicit, dangerous content, harassment, hate speech.

**Key advantages:** Trained on Google's large-scale safety datasets, available in multiple sizes for different latency/quality trade-offs, integrates natively with Gemini-based applications. The 2B variant is fast enough for real-time input/output filtering in production.

### Choosing a Safety Model

| Model | Size | Latency | Custom Taxonomy | Best For |
|-------|------|---------|-----------------|----------|
| Llama Guard 3 1B | 1B | <50ms | Yes | On-device, edge |
| Llama Guard 3 8B | 8B | ~100ms | Yes | Production API, Llama apps |
| ShieldGemma 2B | 2B | <50ms | Limited | Real-time filtering, Gemma/Gemini apps |
| ShieldGemma 27B | 27B | ~200ms | Limited | High-accuracy classification |
| General LLM (GPT-4) | Large | 200-500ms | Fully flexible | Complex policies, nuanced judgment |

**Recommendation:** Use dedicated safety models (Llama Guard, ShieldGemma) for fast, cheap input/output rails. Reserve general-purpose LLMs for complex semantic checks that require nuanced judgment or custom reasoning.

## Comparison with Alternatives

### NeMo Guardrails vs Guardrails AI

**Guardrails AI** is another popular open-source project for LLM safety, but with a different focus:

**NeMo Guardrails** is oriented toward **conversational safety** — dialog control, input/output filtering, dialog flows. Strengths: multi-turn conversation management, topical rails, complex dialog state management.

**Guardrails AI** is oriented toward **output validation** — structured JSON output validation, pydantic schemas, data quality checks. Strengths: structured output validation, Python-first API, rich validator library.

Definition language: NeMo uses Colang (DSL), Guardrails AI uses Python validators. Colang requires learning a new language but provides a more declarative approach to dialog management. Python validators are more familiar but less expressive for conversational flows.

**When to choose NeMo Guardrails:**
- Chatbots and conversational AI with multi-turn dialogs
- Need for conversation flow control (topical rails, loop prevention)
- Enterprise requirements: audit trails, compliance, deterministic safety
- Team is ready to invest in learning Colang

**When to choose Guardrails AI:**
- Primarily structured output validation (JSON, Pydantic models)
- Python-first team, no desire to learn a DSL
- Simpler use cases: single-turn Q&A, data extraction
- Need for integration with the Python type system

### NeMo Guardrails vs Custom LLM Moderation

You can implement moderation with custom code — calling a separate LLM to check input/output. Basic approach: send the user input to a moderating LLM with the prompt "Is this harmful?", and if the response contains "yes", return a refusal instead of processing with the main LLM.

**Advantages of a custom solution:** Full control over logic without framework abstractions, no dependency on external libraries, ability to use any checks (API calls to fraud detection services, database queries for user context, custom ML models).

**Disadvantages of a custom solution:** More latency due to sequential calls (moderation LLM first, then main LLM), no built-in optimizations (parallel execution, caching, circuit breakers), harder to maintain (security logic scattered across code), no ready-made patterns.

**NeMo Guardrails** solves these problems: parallel execution rails (multiple checks simultaneously), built-in caching (similar questions served from cache), config-driven approach (security policies in separate files), ready-made best practices (library of predefined rails for common scenarios).

**When to build a custom solution:** Very specific requirements that cannot be expressed in Colang DSL (integration with a proprietary fraud detection system), an existing working moderation pipeline where migration would require significant effort, need for maximum control over every aspect.

---

## Security Patterns and Best Practices

### Layered Defense (Defense in Depth)

Key principle: **do not rely on a single layer of protection**. Even if one rail fails or is bypassed, other layers should catch the problem.

**Layer 1: Fast regex/keyword checks** — first line of defense, checks for obvious patterns (blocklisted words, known attack patterns). Very fast (1-5ms) but easily bypassed with clever phrasing.

**Layer 2: LLM-based semantic checks** — understands intent, not just keywords. Catches sophisticated attacks where the user does not use blocklisted words but the intent is malicious. Slower (200-500ms) but thorough.

**Layer 3: Domain-specific rules** — custom actions for checking business logic. For example, verifying the user's access level, compliance with industry regulations, integration with fraud detection systems.

Example: a user asks "How to bypass authentication?" Layer 1 (regex) catches the word "bypass" → blocks. If bypassed via "circumvent login", Layer 2 (LLM) classifies the harmful intent → blocks. If bypassed via "I forgot password", Layer 3 (business logic) redirects to a legitimate password reset flow.

### Graceful Degradation

Rail checks can fail — LLM API timeout, network issue, internal error. How to handle failures?

**Fail closed (default):** On failure, block the request. Safer but can create false positives and poor UX.

**Fail open (risky):** On failure, pass the request through. Better UX but creates a security hole.

**Hybrid approach:** Critical rails (input harmful content check) — fail closed. Non-critical rails (hallucination detection) — fail open with a warning.

Important: always log rail failures for post-mortem analysis. If a rail frequently fails, this is a reliability problem that needs to be addressed.

### Monitoring and Continuous Improvement

Rails are not "set and forget". Continuous monitoring and tuning are required:

**Metrics to track:** Rail activation rate (how often each rail triggers), false positive rate (how many legitimate requests are blocked), false negative rate (how many harmful requests are passed through), latency per rail (impact on user experience), rail failure rate (reliability of each rail).

**Feedback loop:** Log blocked requests for manual review. If you see a pattern of false positives, tune the rail rules. If you see bypasses, add new patterns.

**A/B testing:** Test new rails on a subset of traffic before full rollout. Measure the impact on conversion, user satisfaction, and support tickets.

### Testing Rails

Rails should be tested like any security-critical code:

**Unit tests:** Test individual rails with known good/bad inputs. Ensure harmful content is blocked and legitimate content passes through.

**Integration tests:** Test full conversation flows. Check edge cases: very long inputs, multilingual content, adversarial examples.

**Red team testing:** Ask the security team to attempt to bypass rails. Document successful bypasses and update rails.

**Regression tests:** Every discovered bypass should become a test case to prevent regression.

---

## Real-World Use Cases

### Financial Services Compliance Bot

**Scenario:** A banking chatbot for customer support. Strict regulatory requirements — GDPR, financial disclaimers, anti-fraud.

**Guardrails setup:**
- Input rails: Block requests for other clients' PII, social engineering attempts to obtain account details
- Output rails: Add required disclaimers to financial advice, block leakage of sensitive bank data
- Dialog rails: Restrict topics (only banking services, no investment advice without a license)
- Retrieval rails: Filter documents by customer access level (retail vs premium customers)

### Healthcare AI Assistant

**Scenario:** An AI assistant for physicians. Critical safety — incorrect medical advice can harm patients.

**Guardrails setup:**
- Input rails: Block off-topic questions (politics, entertainment), focus on medical queries
- Output rails: Hallucination detection is critical — verify that the response is backed by medical literature. Block definitive diagnoses (suggestions for the physician only)
- Dialog rails: For complex/critical cases — mandatory escalation to a human physician
- Retrieval rails: Access control for patient records (HIPAA compliance), filtering by physician credentials

### E-commerce Customer Support

**Scenario:** A customer support bot for retail. Protection from abuse, fraud, and brand reputation damage is needed.

**Guardrails setup:**
- Input rails: Block toxic language, spam, competitor mentions
- Output rails: Prevent promises the company cannot fulfill (unlimited refunds), toxic responses
- Dialog rails: Loop prevention (if the bot cannot help after 3 attempts — escalation to a human), topical rails (only product/order questions)
- Retrieval rails: Filter internal docs; the customer sees only public-facing information

---

## Key Takeaways

**Programmatic control vs probabilistic:** NeMo Guardrails implements deterministic runtime enforcement, unlike prompt-based safety that relies on LLM "memory" and is easily bypassed by jailbreaks.

**Four rail types cover the full request lifecycle:** Input Rails block harmful requests before the LLM, Output Rails filter responses, Dialog Rails manage conversation flow, Retrieval Rails control knowledge access.

**Colang as a safety DSL:** Colang 2.0 (2024) brings Python-like syntax, event-driven programming, and improved expressiveness. The declarative language separates security policies from code and prompts, simplifying auditing, versioning, and testing.

**Dedicated safety models complement guardrails:** Llama Guard 3 (1B/8B) and ShieldGemma (2B/9B/27B) provide fast, cheap, purpose-trained safety classification. Use them for input/output rails instead of expensive general-purpose LLMs. Reserve general LLMs for complex semantic checks.

**Layered defense is critical:** Do not rely on a single protection mechanism. Combine fast regex checks (1-5ms) for obvious patterns, LLM-based semantic checks (200-500ms) for sophisticated attacks, and domain-specific rules for business logic.

**Production considerations:** Latency overhead is manageable through parallel execution and caching. Three deployment models — embedded, REST API service, sidecar container — for different architecture requirements. Graceful degradation and monitoring are mandatory.

**Tool comparison:** NeMo Guardrails for conversational AI with complex dialogs and enterprise compliance. Guardrails AI for structured output validation and Python-first teams. Custom solutions when maximum control or very specific requirements are needed.

**Real-world application:** Financial services require compliance rails (GDPR, disclaimers), healthcare requires hallucination prevention and escalation rules, e-commerce requires brand protection and fraud prevention. Rails adapt to domain-specific risks.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Security and Safety
**Previous:** [[04_Moderation_and_Compliance|Moderation and Compliance]]
**Next:** [[../15_GPU_Architecture/01_GPU_Architecture|GPU Architecture]]

---

## Practical Colang Examples

### User Intents and Bot Responses

```colang
# User Intents - defining user intentions
define user ask about product
  "How does your software work?"

define user ask harmful question
  "How to hack a system?"
  pattern "(?i)(bypass|exploit|hack)"

# Bot Responses - response templates
define bot explain product
  "Our product automates processes through AI agents."

define bot refuse internal info
  "I cannot provide internal information."

define bot offer human support
  "Connecting you with a specialist."
```

### Flows - Safety Rules

```colang
# Input Rail: blocking harmful requests
define flow handle harmful input
  user ask harmful question
  bot refuse internal info
  stop

# Output Rail: disclaimer for financial topics
define flow financial disclaimer
  user ask about pricing
  bot explain pricing
  bot add disclaimer "For exact pricing, please contact the sales department."

# Dialog Rail: loop prevention
define flow prevent loops
  when same_question_count > 2
    bot offer human support
    stop

# Retrieval Rail: document access control
define flow check document access
  user request document
  $access_level = execute check_user_access
  if $access_level == "admin"
    bot provide document
  else
    bot refuse internal info
```

### Topical Rails - Topic Control

```colang
define flow topical rails
  allow product questions
  deny politics

define flow handle denied topic
  user ask about competitors
  bot say "I focus on our product."
```

### config.yml Configuration

```yaml
models:
  - {type: main, engine: openai, model: gpt-4}

rails:
  input:
    flows: [self check input, jailbreak detection, handle harmful input]
  output:
    flows: [self check output, financial disclaimer]
  dialog:
    flows: [prevent loops, topical rails]
  retrieval:
    flows: [check document access]

self_check_input: {enabled: true, llm: {model: gpt-4o-mini, temperature: 0.0}}
jailbreak_detection: {enabled: true, threshold: 0.85}
```

### Custom Actions - Python Integration

```colang
define flow verify user and process
  user request sensitive operation
  $is_verified = execute verify_user_identity
  if $is_verified
    $fraud_score = execute check_fraud_risk
    if $fraud_score < 0.3
      bot process request
    else
      bot say "Verification required."
      execute trigger_manual_review
      stop
  else
    bot refuse internal info
    stop
```

Python code (actions/custom_actions.py):

```python
from nemoguardrails.actions import action

@action(is_system_action=True)
async def verify_user_identity(context: dict):
    return await auth_service.verify(context.get("user_id"))

@action(is_system_action=True)
async def check_fraud_risk(context: dict):
    return await fraud_api.analyze(context.get("user_data"))
```
