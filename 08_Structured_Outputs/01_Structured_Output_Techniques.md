# Structured Output Techniques

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Structured Output
**Previous:** [[../07_Frameworks/05_AWS_Strands_Agents|AWS Strands Agents]]
**Next:** [[02_Data_Extraction|Extracting Data from Unstructured Text]]

---

## Introduction: When Text Is Not Enough

Consider an automated resume processing system. A candidate uploads a document, and the system must extract the name, contacts, experience, education, skills, and store them in a database for filtering and analysis. GPT-4o returns a response: "John Smith is an experienced developer with 10 years of experience. He worked at Google and Microsoft..." Excellent text for a human. But how do you structure it for programmatic processing?

This is a fundamental problem of integrating language models into software systems. Models generate natural text — sentences, paragraphs, narratives. Programs operate on structures — objects, arrays, numbers. Structured output is the bridge between these worlds.

## The Problem with Unstructured Text

Free-form text is flexible and expressive but creates serious challenges for machine processing.

**Parsing ambiguity.** The phrase "I started working about five years ago" does not provide an exact date. When exactly — May 2019 or June 2020? "About" — how approximate? And if the model writes "I joined the company in twenty nineteen" — what regex covers all variants? Free-form text creates a combinatorial explosion of representations. Each field can be expressed in dozens of ways, ordering is arbitrary, delimiters vary. A reliable parser is practically impossible.

**Unpredictable structure.** The model adds introductory sentences, concluding remarks, and breaks information into sub-items. Each request is a format lottery. Even with temperature=0, structure changes depending on context. For database integration, predictability is required: the "age" field must contain a number, not "thirty-two years of youth and enthusiasm."

**Structure hallucinations.** The model hallucinates not only facts but also structure. You requested three fields — it returned five. You expected an array — you got an object. You wanted a number — you got a string with explanations. Code crashes with an exception, or worse — silently accepts incorrect data, breaking downstream processes.

## Levels of Structured Output Guarantees

Structured output methods range on a scale from "hoping for the best" to "guaranteed schema."

**Level 1: Prompt engineering.** You ask the model to return JSON: "Please respond with a JSON object containing fields name, age, and occupation." Modern models understand JSON thanks to training on code and documentation. It works often but without guarantees. The model adds comments, uses single quotes, forgets brackets. This is a request, not a contract.

**Level 2: JSON Mode.** Providers (OpenAI, Anthropic) offer a mode that guarantees valid JSON. The system forcibly closes brackets and fixes syntax. The result always parses. However, only syntactic correctness is guaranteed — it could be {"random": "data"} instead of {"name": "John", "age": 30}. Structure is not controlled.

**Level 3: JSON Schema / Function Calling.** An exact result schema is specified. In OpenAI — via Function Calling, in Anthropic — via Tool Use. It is guaranteed through **constrained decoding** — at each generation step, the model's logit probabilities are masked so that only tokens that are valid continuations of the schema can be sampled. A finite automaton tracks the current position in the schema: after an opening brace, only key tokens or a closing brace are allowed; after a key — a colon; after a colon — the start of a value of the correct type. The model literally cannot produce invalid JSON because invalid tokens have zero probability.

This guarantees **syntactic** correctness but not **semantic** correctness. A JSON object can be perfectly valid structurally — all types correct, all required fields present — yet contain hallucinated values (a fictional address, an invented product name, a wrong date). This distinction matters for validation design: constrained decoding eliminates parsing errors, but business-rule validation (Level 4) is still needed for semantic correctness.

Implementations: Outlines (Python, FSM-based, works with vLLM and HuggingFace Transformers), Guidance (Microsoft, template-based), OpenAI Structured Outputs (built into the API with `strict: true`). Note: OpenAI reports a 200-500ms cold start for schema preprocessing on the first structured output call with a new schema.

You describe a schema: fields, types, required properties. The model returns JSON with the correct structure. Function Calling was originally designed for invoking functions but became a mechanism for structured output — the model "calls a function" with the required arguments, and you intercept them.

**Level 4: Schema + Validation + Retry.** Even with a guaranteed structure, values can be incorrect: negative age, email without @. Business rule validation, range checks, and format verification are added. On errors — feedback to the model requesting corrections. More API calls, but maximum reliability for critical applications.

## Choosing a Method

Different tasks require different levels of guarantees. Excessive complexity creates problems: additional API calls, increased code complexity, higher latency.

**Simple tasks:** 1-3 simple fields, human review, non-critical errors — prompt engineering is sufficient. Add a format example, request "Return only JSON, no other text." For prototypes and internal tools, this is enough.

**Integration tasks:** Results go into code, databases, APIs — Function Calling or JSON Schema is needed. You get type-safe data without parsing and guessing.

**Critical applications:** Finance, healthcare, legal documents — add validation and retry. Three requests with a correct result are better than one with an error that costs millions.

## Frameworks for Java

**LangChain4j AI Services** — a declarative approach through interfaces with methods that return Java objects. The framework automatically generates prompts, parses responses, and handles errors. Type safety at compile time. The most convenient approach for Java developers.

**Spring AI Output Converters** — integration into the Spring ecosystem. BeanOutputConverter generates instructions from Java classes, MapOutputConverter works with dynamic structures. Configuration through beans and configuration files.

**Direct API usage** — full control through OpenAI or Anthropic APIs with manual schema construction and parsing. More code, more flexibility. For non-standard scenarios and optimizations.

## Practical Implementation Details

**Prompt engineering** asks the model to return JSON with specific fields and a format example. Problems: the model wraps JSON in markdown blocks, adds preambles, uses single quotes, forgets brackets, adds comments. The solution is a response cleaning layer (remove markdown, extract JSON, parse). Works in 80-90% of cases, but the code is fragile. For prototypes and non-critical experiments.

**JSON Mode** guarantees valid JSON — the system monitors brackets, quotes, commas. In OpenAI, this is the responseFormat("json_object") parameter; in Anthropic — a flag in the request. Syntax is guaranteed, but not structure: it could be {"random": "stuff"} instead of {"name": "John"}. For cases with flexible data schemas, such as extracting arbitrary facts.

**Function Calling** describes an exact JSON Schema — fields, types, required properties, descriptions. In LangChain4j, a ToolSpecification with parameters is created. The model "calls a function" with the required arguments, and you intercept them. For example, for an invoice: invoiceNumber (string), vendorName (string), totalAmount (number), currency (string), dueDate (ISO string). The model must return exactly these fields. If information is unavailable — it returns null, but the structure is preserved. For integrations with databases, APIs, and financial systems.

**LangChain4j AI Services** uses a declarative approach through interfaces. A method is annotated with @UserMessage containing a prompt and returns the desired structure. The framework automatically: analyzes the return type, generates JSON Schema, creates a ToolSpecification, sends a request with Function Calling, parses the arguments, creates an object, and returns a type-safe result. Type safety at compile time, IDE autocompletion for fields, safe refactoring. The most convenient approach for Java applications.

**Spring AI** integrates with ChatClient. The .entity(CustomerInfo.class) method automatically analyzes the class structure, generates instructions, converts the response, and handles errors. It supports ParameterizedTypeReference for collections: List<OrderItem>, Map<String, Product>. A natural choice for Spring applications with @Configuration setup and dependency injection.

## Provider Comparison for Structured Outputs

| Provider | Mechanism | Schema Format | Guarantee Level | Notes |
|----------|-----------|--------------|-----------------|-------|
| **OpenAI** | Structured Outputs (`strict: true`) | JSON Schema | Syntactic + structural | Constrained decoding built into API. 200-500ms cold start per new schema |
| **Anthropic** | Tool Use with schema | JSON Schema (via tool parameters) | Syntactic + structural | Uses the tool calling mechanism — the model "calls a tool" with structured arguments |
| **Google** | Controlled generation | Pydantic schema or JSON Schema | Syntactic + structural | Available on Gemini via `response_schema` parameter |
| **Open-source (Outlines)** | Constrained decoding via FSM | JSON Schema, regex, CFG | Syntactic + structural | Works with vLLM, HuggingFace Transformers, llama.cpp. No vendor lock-in |
| **Open-source (Guidance)** | Template-based generation | Custom template language | Syntactic + structural | Microsoft-developed. Templates mix text and structured sections |

**Recommendation:** For API-based systems, use the provider's native structured output (OpenAI Structured Outputs, Anthropic Tool Use). For self-hosted models, Outlines with vLLM provides equivalent guarantees without vendor dependency. All approaches guarantee structure; none guarantee semantic correctness — validation remains your responsibility.

---

## Structured Extraction Tooling

Beyond provider APIs, three libraries have emerged as the standard toolkit for structured extraction in production:

**Instructor** — wraps any LLM provider with Pydantic schema validation and automatic retry on validation failure. Define a Pydantic model, call the LLM, get a validated object or an automatic retry with the validation error as feedback. Minimal setup, excellent for prototyping and development. The most popular choice for Python applications.

**BAML** (Boundary AI Markup Language) — compile-time contracts for structured output. Schema violations are caught before deployment, not at runtime. Generates client code in multiple languages (Python, TypeScript, Ruby). For production systems where a runtime parsing failure is unacceptable.

**Outlines** — constrained decoding via finite state machine, running directly on the inference server (vLLM, HuggingFace). Guarantees valid output at the token level with zero retry overhead. Best for cost-critical self-hosted inference where every wasted token matters.

**Decision guide:** Instructor for development and rapid iteration. BAML for production systems with strict reliability requirements. Outlines + vLLM for high-volume self-hosted inference where cost optimization is paramount.

**Token overhead awareness:** JSON structural tokens (braces, quotes, colons, commas) cost approximately 24 tokens per object. At scale, this overhead is material: field name compression (e.g., `customer_satisfaction_score` → `css`) can save 3-5 tokens per field. For internal agent-to-agent communication where human readability is not required, minimal-delimiter formats reduce overhead significantly — see TOON below.

---

## Token-Optimized Output Notation (TOON)

JSON is the standard structured format, but its overhead is significant: braces, quotes, colons, commas, and verbose key names consume tokens that carry no semantic information. For high-volume applications processing millions of requests per day, this overhead can exceed $100K per year.

**TOON** and similar minimal-delimiter approaches achieve approximately **69% token reduction** compared to JSON by using pipe-separated values, newline-delimited records, and positional fields instead of named keys.

| Format | Tokens for 10-field record | Relative Cost |
|--------|---------------------------|---------------|
| JSON | ~45 tokens | 1.0x |
| TOON / minimal delimiters | ~14 tokens | 0.31x |

**When to use JSON:** External APIs, persistent storage, human-readable logs, any boundary where interoperability with other systems matters. JSON is the universal interchange format — do not replace it where interoperability is required.

**When to use TOON / minimal formats:** Internal agent-to-agent communication, intermediate tool results consumed by the next LLM call, structured output that will be immediately parsed and never stored in its serialized form. The key criterion: will this data ever leave the agent system? If no — optimize for tokens. If yes — use JSON.

See [[../../02_Prompt_Engineering/05_Context_Engineering|Context Engineering]] for the broader discussion of token budget management and output format optimization.

---

## Key Takeaways

Structured output is the bridge between language model text and program data structures. Without it, LLM integration into enterprise systems is fragile and unreliable.

Guarantee levels: prompt engineering (hope only), JSON Mode (syntax), Function Calling/JSON Schema (structure), Schema + Validation + Retry (maximum reliability with business rule validation).

The choice depends on criticality: prototypes — prompts, integrations — Function Calling, critical systems — full stack with validation.

Java ecosystem: LangChain4j AI Services (declarative approach), Spring AI (Spring integration). They hide the complexity of working with JSON Schema and parsing.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Structured Output
**Previous:** [[../07_Frameworks/05_AWS_Strands_Agents|AWS Strands Agents]]
**Next:** [[02_Data_Extraction|Extracting Data from Unstructured Text]]

---
