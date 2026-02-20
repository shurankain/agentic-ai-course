# LLM Providers and Working with APIs

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[04_Generation_Parameters|Generation Parameters: Controlling Model Behavior]]
**Next:** [[06_Integration_Patterns|LLM Integration Patterns]]

---

Consider the cloud computing market in the early 2010s: AWS dominated, Azure and Google Cloud were emerging, each offering their services with different APIs, pricing, and capabilities. Today we observe a similar — but far more dynamic — landscape in the world of large language models. OpenAI, Anthropic, Google, Meta, DeepSeek, xAI, and dozens of other providers offer their models through APIs, and choosing the right provider becomes a strategic decision for any project involving AI agents.

Understanding the LLM provider landscape is not simply a matter of picking the "best model." It is a question of balancing quality, cost, speed, reliability, and compliance with security requirements. The market has also split into two paradigms: **standard models** (optimized for speed and general tasks) and **reasoning models** (optimized for complex problem-solving via extended thinking). For agent developers, this knowledge shapes system architecture, fallback strategy, and project budget.

---

## Major Providers and Their Models

### OpenAI

OpenAI popularized LLMs through ChatGPT and continues to set the pace of the industry. Their API remains the de facto standard that other providers emulate.

**Frontier models:**
- **GPT-5** — the latest flagship, natively multimodal (text, images, audio). Unified standard and reasoning capabilities. 128K–1M context.
- **GPT-4o** — the previous-generation flagship. Remains widely deployed due to its strong cost/quality ratio. 128K context.
- **GPT-4.1** — optimized specifically for coding and instruction following (released April 2025).

**Reasoning models:**
- **o3** — frontier reasoning model. Generates internal "thinking tokens" before the final answer. Excels on math, science, and complex multi-step problems.
- **o4-mini** — cost-efficient reasoning model. Strong on STEM tasks at a fraction of o3's cost.

**Budget models:**
- **GPT-4o-mini** — lightweight and inexpensive, ideal for classification, extraction, and simple Q&A.
- GPT-3.5 Turbo is fully deprecated.

**OpenAI highlights:**
- Most mature ecosystem and documentation
- Broad support across SDKs and frameworks
- Native function calling / tool use with parallel calls and strict mode
- Structured outputs (guaranteed JSON via constrained decoding)
- Batching API for bulk processing at a 50% discount
- Prompt caching for savings on repeated prefixes
- Responses API (successor to Assistants API) and Agents SDK for multi-agent orchestration

### Anthropic

Anthropic, founded by former OpenAI researchers, positions itself as a company focused on AI safety. Their Claude models are distinguished by excellent instruction following, strong code generation, and long context capabilities.

**Claude model family (latest generation):**
- **Claude Opus 4** — most powerful model for the hardest tasks. Deep reasoning, complex analysis, extended autonomous workflows.
- **Claude Sonnet 4** — the primary workhorse. Excellent balance of intelligence, speed, and cost. Benchmark leader in code generation.
- **Claude Haiku 4.5** — fast and cost-effective model for high-volume, simpler tasks.

**Extended thinking:** Claude models support an "extended thinking" mode — Anthropic's approach to reasoning. The model produces a chain-of-thought in a dedicated thinking block before generating the final response. Controlled via the `budget_tokens` parameter.

**Anthropic highlights:**
- Context window — 200K tokens (1M in beta)
- "Constitutional AI" — models trained to follow ethical principles
- Excellent performance in long document analysis and code generation
- **MCP (Model Context Protocol)** — Anthropic-created open standard for tool integration, now an industry standard governed by AAIF
- Tool use API compatible with the OpenAI format
- Computer use capability — Claude can interact with desktop applications via screenshots

### Google

Google offers models through Vertex AI (enterprise) and Google AI Studio (for developers).

**Gemini family:**
- **Gemini 2.5 Pro** — multimodal model with 1M token context and built-in thinking capabilities. Strong across reasoning, coding, and multimodal tasks.
- **Gemini 2.5 Flash** — optimized for speed and cost with configurable "thinking budgets," ideal for real-time applications.
- **Gemini 2.0 Flash** — budget model for high-volume workloads.

The "Gemini Ultra" branding has been abandoned; the Pro tier is the flagship.

**Google advantages:**
- Record context sizes — 1M tokens standard, 2M in preview
- Native multimodal understanding and generation (text, images, audio, video)
- Grounding with Google Search for up-to-date information
- Native integration with Google Cloud services
- Competitive pricing, especially for Flash models

### xAI

Founded by Elon Musk, xAI emerged as a significant provider with competitive models and aggressive pricing.

**Models:**
- **Grok 3** — frontier model with strong reasoning capabilities
- **Grok 3 Mini** — efficient model for standard tasks

**xAI highlights:**
- Competitive pricing aimed at undercutting incumbents
- Real-time access to X (Twitter) data
- OpenAI-compatible API format

### Mistral AI

A French company offering efficient open-weight models. Attractive for European companies due to GDPR compliance.

**Models:**
- **Mistral Large 3** — flagship model, competitive with frontier proprietary models
- **Mistral Small 3** — compact and efficient for simpler tasks
- **Magistral** — reasoning-focused models
- **Devstral 2** — specialized for code generation
- **Ministral** — edge-friendly compact models

**Mistral highlights:**
- Models available with open weights (can be self-hosted)
- Competitive pricing with good quality
- GDPR compliance out of the box (EU company)
- Strong multilingual support for European languages

### Other Notable Providers

**Amazon Bedrock** — access to models from multiple providers (Anthropic, Meta, Mistral, Cohere) through a unified AWS API with enterprise guarantees.

**Azure OpenAI** — OpenAI models with enterprise SLA, compliance, and integration with the Azure ecosystem.

**Cohere** — specializes in enterprise search and RAG. Embed models for embeddings, Command for generation, Rerank for re-ranking search results.

---

## Reasoning Models: A New API Paradigm

### What Are Reasoning Models?

Starting with OpenAI's o1 (September 2024) and rapidly expanding through o3, o4-mini, DeepSeek-R1, Claude extended thinking, and Gemini thinking modes, reasoning models represent a fundamentally new API paradigm.

Unlike standard models that generate responses token by token, reasoning models produce **thinking tokens** — an internal chain of reasoning — before generating the visible response. This dramatically improves performance on complex tasks: math, science, multi-step planning, and code.

### How the API Differs

**Thinking tokens:**
- Invisible to the user (in most implementations) but billed as output tokens
- Can represent 80-90% of total output for complex queries
- Controlled via parameters like `reasoning_effort` (OpenAI) or `budget_tokens` (Anthropic)

**Parameter restrictions:**
- Temperature is typically fixed at 1.0 (OpenAI reasoning models) or has limited ranges
- System prompts may be handled differently (some models use "developer" messages)
- Streaming behavior differs — thinking tokens may not stream

**Cost structure:**
- Thinking tokens are billed at the output token rate
- A simple question might cost 10x more through a reasoning model due to thinking overhead
- The `reasoning_effort` parameter lets you trade quality for cost (low/medium/high)

### When to Use Reasoning Models

**Use reasoning models for:** complex math and logic, multi-step planning, scientific analysis, hard coding problems, tasks where accuracy matters more than speed.

**Use standard models for:** simple Q&A, classification, extraction, creative writing, real-time chat, high-volume tasks where cost and latency matter.

**The hybrid approach is optimal:** route simple tasks to standard models, escalate complex ones to reasoning models. This can cut costs by 70%+ while maintaining quality where it matters.

---

## Open-Source Models: In-Depth Comparison

The open-source landscape has transformed. In 2025, open-source models achieved parity with — and in some domains surpassed — proprietary ones. DeepSeek R1, Llama 4, and Qwen 3 are the three flagship families.

### DeepSeek

A Chinese lab that has twice disrupted the industry — first with V3's training efficiency, then with R1's reasoning breakthrough.

**DeepSeek V3** — 671B parameters, 37B active (Mixture of Experts). Revolutionized training economics.

**Key innovations:**
- **Multi-head Latent Attention (MLA)**: KV-cache compression via low-rank projections — 93% memory savings
- **Multi-Token Prediction (MTP)**: Predicting 2 tokens per step — 1.8x inference speedup
- **FP8 Training**: Full training in 8-bit floating point — the first successful example at this scale
- **Training cost**: $5.5M — an order of magnitude cheaper than competitors

**DeepSeek R1** — the reasoning breakthrough. Achieved frontier-level reasoning through pure reinforcement learning (GRPO — Group Relative Policy Optimization), without supervised fine-tuning on reasoning traces. R1-Zero showed that reasoning can emerge from RL alone.

- Open-weight with MIT-like license
- Distilled versions available from 1.5B to 70B parameters
- Competitive with o3 on math and science benchmarks
- R1-0528: improved version with better instruction following

**When to choose DeepSeek:**
- Mathematical and algorithmic tasks (leader)
- Self-hosting with a limited GPU budget (MoE efficiency)
- Reasoning tasks where cost matters (R1 distilled variants)
- Research on MoE architectures and RL-based reasoning

### Meta Llama 4

Meta's latest generation represents a fundamental architectural shift from dense models to Mixture of Experts with native multimodality.

**Models:**
- **Llama 4 Scout** — 16 experts, 10M token context window (the largest of any production model). Designed for long-context tasks.
- **Llama 4 Maverick** — 128 experts, ~400B total parameters with 17B active. The performance flagship.
- **Llama 4 Behemoth** — ~2T parameters, the largest open model. Teacher model for distillation.

**Key changes from Llama 3:**
- **MoE architecture** (previously dense) — dramatically better inference efficiency
- **Native multimodality** — trained from scratch on text, images, and video (not bolt-on adapters)
- **Massive context** — 10M tokens in Scout, far beyond any predecessor
- **Interleaved attention** — architectural innovation for efficiency

**When to choose Llama 4:**
- Production self-hosting (strongest ecosystem, mature tooling)
- Extremely long context requirements (Scout 10M)
- Multimodal applications
- When a permissive license matters (Llama license allows broad commercial use)

### Qwen 3

Alibaba's latest series, leading many benchmarks and offering the broadest size range.

**Models:**
- **Qwen 3 235B** — MoE flagship (22B active parameters), 119 languages
- **Qwen 3 72B** — dense flagship
- **Qwen 3 32B / 14B / 8B** — range of dense models for different GPU budgets
- **Qwen 3 0.6B / 1.7B / 4B** — edge and mobile-friendly

**Key innovations:**
- **Hybrid reasoning**: can switch between "thinking mode" (extended reasoning) and "non-thinking mode" (fast) within the same model, controlled via system prompt
- **119 language support** — the most multilingual open model
- **Apache 2.0 license** — fully permissive

**When to choose Qwen 3:**
- Multilingual applications (broadest language coverage)
- When you need a single model with both reasoning and fast modes
- Asian language markets
- Edge deployment (strong small models down to 0.6B)

### Open-Source Comparison Table

| Criterion | DeepSeek R1 | Llama 4 Maverick | Qwen 3 235B |
|----------|-------------|------------------|-------------|
| **Architecture** | MoE (671B/37B) | MoE (400B/17B) | MoE (235B/22B) |
| **Context** | 128K | 1M+ | 128K |
| **Licensing** | MIT-like | Llama License | Apache 2.0 |
| **Commercial use** | ✅ | ✅ (with restrictions) | ✅ |
| **Reasoning** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ (hybrid) |
| **Coding** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Math** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Multilingual** | ⭐⭐ (CN focus) | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ (119 langs) |
| **Multimodal** | Text only (V3) | ✅ Native | ✅ |

### Practical Recommendations

**For a startup with a limited budget:** start with Qwen 3 in a 14B or 32B configuration — good balance between quality and resources. Use hybrid reasoning mode to avoid needing separate models. For pure reasoning tasks, DeepSeek R1 distilled variants (7B-70B) deliver outstanding performance.

**For enterprise self-hosting:** Llama 4 Maverick or Scout are the optimal choices thanks to the mature ecosystem, strong tooling (vLLM, TGI), and clear licensing. Scout's 10M context window eliminates many RAG complexity needs. For reasoning-heavy workloads, pair with R1 distilled models.

**For research:** DeepSeek R1 is of particular interest for studying GRPO and RL-based reasoning emergence. Qwen 3's hybrid reasoning mode offers insights into mode-switching architectures. All three families provide full model weights for analysis.

### Interview Questions: Open-Source Models

1. **Compare MoE architectures across DeepSeek V3, Llama 4, and Qwen 3. What are the trade-offs?**
   - All three use MoE, but with different expert counts and routing strategies
   - DeepSeek: 256 experts, auxiliary-loss-free balancing, MLA for KV-cache efficiency
   - Llama 4: 16-128 experts, interleaved attention, native multimodality
   - Qwen 3: traditional MoE with hybrid reasoning mode
   - More experts = finer specialization but more complex routing

2. **When is an open-source model preferable to a proprietary one?**
   - Data privacy requirements (healthcare, finance)
   - Predictable costs at high volume
   - Need for fine-tuning (especially RL-based, like GRPO)
   - Offline/edge deployment
   - Vendor independence

3. **What made DeepSeek R1 significant for the field?**
   - Demonstrated reasoning can emerge from pure RL (GRPO) without supervised reasoning traces
   - R1-Zero showed reasoning emergence without any SFT
   - Open-weight release enabled distillation into smaller models
   - Training cost transparency challenged industry assumptions

---

## API Structure: Common Principles

Despite differences between providers, most LLM APIs follow similar principles. Understanding this common structure allows you to switch between providers easily and write provider-agnostic code.

### Message Format

Virtually all modern APIs use a "messages" format — an array of messages with roles:

- **system** (or **developer** in some reasoning model APIs) — instructions for the model defining its behavior, style, and constraints
- **user** — user messages
- **assistant** — previous model responses
- **tool** — results of tool invocations

This format reflects the conversation structure and allows the model to understand dialog context. The system prompt is typically placed first and defines the assistant's "persona."

### Core Request Parameters

A typical API request includes:
- **model** — model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514", "gemini-2.5-pro")
- **messages** — array of messages
- **temperature** — degree of randomness in responses (0-2; note: some reasoning models fix this)
- **max_tokens** — maximum response length
- **top_p** — nucleus sampling
- **stop** — character sequences that trigger generation stop
- **tools** — descriptions of available tools (for agents)
- **reasoning_effort** — for reasoning models: low/medium/high (OpenAI) or budget_tokens (Anthropic)

### Response Format

The response typically contains:
- **content** — generated text
- **usage** — token usage statistics (input, output, total; reasoning models also report thinking tokens)
- **finish_reason** — reason for stopping (stop, length, tool_calls, content_filter)
- **tool_calls** — requested tool invocations (if the model decided to use a tool)

---

## Pricing Comparison

Cost is a critical factor when choosing a provider and model. All providers charge by token count, with input and output tokens priced differently. Prices are quoted **per 1M tokens** (the industry convention since 2025).

### Pricing Tiers (approximate, early 2026)

**Reasoning models:**

| Model | Input | Output | Notes |
|--------|-------|--------|-------|
| o3 | $10.00 | $40.00 | Thinking tokens billed at output rate |
| o4-mini | $1.10 | $4.40 | Cost-efficient reasoning |
| Claude Opus 4 (extended thinking) | $15.00 | $75.00 | Thinking via budget_tokens |
| Gemini 2.5 Pro (thinking) | $1.25-2.50 | $10.00-15.00 | Configurable thinking budget |

**Standard models:**

| Model | Input | Output |
|--------|-------|--------|
| GPT-5 | ~$2.00-5.00 | ~$8.00-15.00 |
| GPT-4o | $2.50 | $10.00 |
| Claude Sonnet 4 | $3.00 | $15.00 |
| Gemini 2.5 Pro | $1.25 | $10.00 |
| Mistral Large 3 | $2.00 | $6.00 |
| Grok 3 | $3.00 | $15.00 |

**Budget models:**

| Model | Input | Output |
|--------|-------|--------|
| GPT-4o-mini | $0.15 | $0.60 |
| Claude Haiku 4.5 | $0.80 | $4.00 |
| Gemini 2.0 Flash | $0.10 | $0.40 |
| Mistral Small 3 | $0.10 | $0.30 |

**Note:** Reasoning model costs can be deceptive. A query that costs $0.01 with a standard model might cost $0.10-0.50 with a reasoning model due to thinking tokens. Always account for thinking token overhead in cost estimates.

### Factors Affecting Actual Cost

**Input/output ratio:** Output tokens are typically 3-5x more expensive than input tokens. For tasks with long prompts and short responses (classification, data extraction), this works in your favor. For generating long texts, output becomes the primary expense.

**Thinking token overhead:** Reasoning models may generate 5-50x more thinking tokens than visible output tokens. A 100-token answer might require 5,000 thinking tokens. The `reasoning_effort` parameter helps control this.

**Prompt caching:** OpenAI, Anthropic, and Google offer discounts when the same prefixes are reused. An agent's system prompt repeats in every request — caching can save 50-90% on its cost.

### Prompt Caching: How It Works

Prompt caching is a mechanism where the provider saves the processing results (KV-cache) of the initial portion of the prompt and reuses them in subsequent requests.

**How it works:**
1. On the first request, the model computes the KV-cache for the entire prompt
2. The provider saves the KV-cache for the repeating **prefix** (beginning of the prompt)
3. On the next request with the same prefix — cache hit, only the new portion is computed
4. TTL is typically 5-10 minutes — after that the cache is invalidated

**Conditions for a cache hit:**
- The prefix must be **byte-identical**
- System prompt + beginning of messages must match
- Minimum prefix length: 1024-2048 tokens (depends on the provider)

**Savings by provider:**

| Provider | Cache hit discount | Min prefix length | TTL |
|-----------|-------------------|-------------------|-----|
| OpenAI | 50% | 1024 tokens | 5-10 min |
| Anthropic | 90% | 1024 tokens | 5 min |
| Google | Varies | 32K tokens | - |

**Optimal prompt structure for caching:**

The ideal prompt organization for maximum caching efficiency consists of four layers. At the top is the system prompt — a long, unchanging block of instructions that is fully cached. Next come few-shot examples — multiple request-response examples that teach the model the desired format, also fully cached. Then comes the context — documents or RAG results that may be partially cached if they repeat across requests. Finally, the user message — unique per request and never cached.

**When prompt caching is effective:**
- Long system prompt (>2000 tokens) with detailed instructions
- Many few-shot examples in each request
- Shared context reused across requests
- High frequency of similar requests (>1 req/min)

**When it is NOT effective:**
- Short prompts (<1024 tokens)
- Highly varying context (RAG with different documents)
- Infrequent requests (cache expires)

### Batching API: Architecture and Economics

**Batching:** Deferred processing of requests in batches (not real-time) often yields a 50% discount. Suitable for analytical tasks, document processing, and content generation.

**Batching API architecture:**

The batch processing architecture follows an asynchronous pattern: the client submits requests to a Batch Queue, from which they enter the Batch Processor, which executes them at low priority. Results are saved in Results Storage, and the client is notified either via periodic Status Polling or through webhook/callback notifications upon completion.

**How it works (using OpenAI as an example):**
1. **Submit batch** — you send a JSONL file with requests (up to 50,000 requests)
2. **Queue** — requests enter a low-priority queue
3. **Processing** — execution within 24 hours (SLA)
4. **Results** — you retrieve results via polling or webhook

**Batching economics:**
- 50% discount on all tokens (input and output)
- Trade-off: latency (up to 24 hours) for cost reduction
- Minimum volume: typically none, but overhead makes sense at >100 requests

**Use cases for batching:**
- Daily processing of accumulated documents
- Generating embeddings for a large collection
- A/B testing prompts on historical data
- Evaluating model quality on a test set
- Content moderation backlog

**Client-side code architecture:**

A proper client-side batching implementation follows a specific pattern. First, requests accumulate in a buffer until a threshold is reached — this can be either a certain number of requests or a time interval. When the threshold is met, a batch is formed and submitted to the API, which returns a unique batch identifier (batch_id). The client then either periodically polls the processing status every few minutes or awaits a webhook notification. When processing completes, results are retrieved from storage and used to update downstream systems.

---

## Choosing a Model for Different Tasks

Different tasks require different models. Using a reasoning model for simple classification wastes money and adds latency. Using a budget model for complex reasoning produces unreliable results.

### Complex Reasoning and Analysis

**Recommended models:**
- o3 / o4-mini (OpenAI reasoning)
- Claude Opus 4 or Sonnet 4 with extended thinking (Anthropic)
- Gemini 2.5 Pro with thinking (Google)

**Typical tasks:** multi-step planning, complex document analysis, mathematical proofs, scientific reasoning, architectural decisions, research tasks.

### Code Generation

**Recommended models:**
- Claude Sonnet 4 (consistently strong on coding benchmarks)
- GPT-4.1 (optimized specifically for code)
- GPT-4o
- Devstral 2 (Mistral, open-weight)

**Typical tasks:** writing new code, refactoring, debugging, writing tests, code review, documentation.

### Simple Tasks and Classification

**Recommended models:**
- GPT-4o-mini
- Claude Haiku 4.5
- Gemini 2.0 Flash

**Typical tasks:** text categorization, entity extraction, sentiment analysis, simple format conversions, filtering.

### Long Documents

**Recommended models:**
- Gemini 2.5 Pro (1M+ token context)
- Claude Sonnet 4 (200K tokens, 1M beta)
- GPT-5 (128K-1M context)

**Typical tasks:** analyzing books, lengthy reports, codebases, legal documents, scientific papers.

### Real-Time Applications

**Recommended models:**
- Gemini 2.5 Flash
- GPT-4o-mini
- Claude Haiku 4.5

**Typical tasks:** chatbots with instant responses, autocomplete, interactive assistants.

---

## Tool Use / Function Calling

The ability to invoke external functions is a key capability for AI agents. It transforms an LLM from a text generator into an interactive assistant capable of interacting with the real world.

### How It Works

1. **Tool descriptions:** You provide the model with a JSON schema of available tools — their names, descriptions, and parameters
2. **Request analysis:** The model analyzes the user's request and decides whether a tool is needed
3. **Call generation:** If a tool is needed, the model generates structured JSON with the function name and arguments
4. **Execution:** Your code executes the function and obtains the result
5. **Continuation:** The result is returned to the model, which uses it to form the final response

### Comparing Provider Implementations

**OpenAI:** The most mature implementation. Support for parallel calls (multiple tools at once), strict mode with guaranteed JSON schema, excellent documentation.

**Anthropic:** Compatible format, high reliability in selecting the correct tool. Claude is particularly strong in complex scenarios with many tools. MCP (Model Context Protocol) provides a standardized way to expose tools.

**Google:** Full-featured function calling support, integration with Google Cloud Functions for serverless execution.

**Mistral:** Tool support exists, but the implementation is less mature compared to the leaders.

### Structured Outputs

An important capability — guaranteed conformance of the response to a JSON schema. OpenAI implemented this through constrained decoding: at the token generation level, the model is physically unable to produce invalid JSON. Anthropic and Google offer similar capabilities.

This is critical for agents, where parsing reliability must be 100%. Previously, you had to handle parsing errors and retry — now that is unnecessary.

---

## Reliability Strategies

Production systems cannot depend on a single provider. APIs go down, rate limits are hit, and models become temporarily unavailable. Reliability requires a well-designed architecture.

### Multi-Provider Architecture

A good practice is to have a fallback to an alternative provider:

1. **Primary request** to the preferred provider (e.g., Anthropic)
2. **On failure** — automatic switch to the fallback (OpenAI)
3. **Monitoring** of success rates for each provider
4. **Automatic rotation** when the primary degrades

Important: models from different providers are not identical. The same prompt may behave differently. Test your fallback scenarios.

### Handling Rate Limits

All providers have limits on requests per minute and tokens per minute. Working strategies:

- **Exponential backoff** on receiving 429 errors — wait and retry with increasing intervals
- **Request queue** with rate control — do not send more than N requests per second
- **Load distribution** across multiple API keys
- **Respect the Retry-After header** — if the API says wait 30 seconds, wait 30 seconds

### Timeouts and Retries

LLM requests can take from one second to a minute (reasoning models can take several minutes). Rules:

- **Reasonable timeouts:** 30-60 seconds for standard generation, 10-20 seconds for classification, 2-5 minutes for reasoning models
- **Retry with exponential backoff:** start at 1 second, double up to a maximum
- **Jitter:** add randomness to the delay to avoid thundering herd
- **Circuit breaker:** after N consecutive errors, stop trying and return an error immediately

### SLA and Reliability Engineering for LLM APIs

When designing production systems, understanding the actual provider guarantees and planning accordingly is critically important.

**Typical SLA metrics:**

| Metric | Description | Typical Values |
|---------|----------|-------------------|
| **Availability** | Service uptime | 99.9% - 99.99% |
| **Latency P50** | Median response time | 1-3 sec (standard), 5-30 sec (reasoning) |
| **Latency P95** | 95th percentile | 5-15 sec (standard), 30-120 sec (reasoning) |
| **Latency P99** | 99th percentile | 15-60 sec |
| **Error rate** | Percentage of 5xx errors | <0.1% |

**Latency percentiles — why they matter:**

Average latency is deceptive. If P50 = 2 sec but P99 = 30 sec, every hundredth user waits half a minute. For interactive applications, this is unacceptable.

A typical latency distribution looks like this: P50 is approximately 2.1 seconds, meaning half of all requests complete faster. P75 increases to 3.5 seconds, P90 to 6.2 seconds, P95 to 9.8 seconds. Critical values are P99 at 24.3 seconds (every hundredth request) and P99.9 at 58 seconds (every thousandth request). These distribution "tails" determine the worst-case user experience.

**Capacity planning with latency in mind:**

System throughput is calculated by the formula: Throughput = Concurrency / Latency. For example, if you need to handle 100 requests per second with a median latency of 2 seconds, you need 200 concurrent connections. However, to handle the distribution tail at P99 latency of 20 seconds, a buffer of 2000 connections is needed to prevent system degradation under peak loads.

**SLA monitoring:**

What to track:
1. **Latency distribution** — not just average, but P50/P95/P99
2. **Error rate by type** — 429 (rate limit), 500 (server), 503 (overload)
3. **Token throughput** — tokens/second for capacity planning
4. **Cost per request** — for budget alerting
5. **Thinking token ratio** — for reasoning models, monitor thinking-to-output ratio

**Degradation strategies:**

| Load Level | Strategy |
|------------------|-----------|
| Normal | All requests to primary provider |
| High | Shedding low-priority requests |
| Overload | Fallback to faster/cheaper model |
| Emergency | Graceful degradation (cached responses, simplified prompts) |

**Practical recommendations:**

1. **Never rely on a single provider** — even with 99.9% SLA, that is 8.7 hours of downtime per year
2. **Measure your own metrics** — provider SLA ≠ your user experience
3. **Plan for burst capacity** — LLM load is often spiky (start of business day)
4. **Budget for latency** — if P99 = 30s, your timeout should be >30s

---

## Integration via LangChain4j

LangChain4j is a popular Java framework that abstracts working with different LLM providers. It provides a unified interface, simplifying switching between models.

### Benefits of a Unified Interface

Instead of learning each provider's API separately, you work with a single `ChatLanguageModel` interface. Switching between providers means changing a single configuration line. The code remains the same.

### Supported Providers

LangChain4j supports out of the box:
- OpenAI and Azure OpenAI
- Anthropic
- Google Vertex AI and Google AI
- Mistral
- Amazon Bedrock
- Ollama (local models)
- HuggingFace
- Cohere
- And many others

### Additional Framework Capabilities

- **Automatic retry** and rate limiting
- **Moderators** for content filtering
- **Response caching** out of the box
- **Integration with vector stores** for RAG
- **AI Services** — a declarative way to create agents through interfaces

---

## Local Models

Cloud providers are not always the right choice. In some cases, data cannot be sent to external APIs, full control over infrastructure is needed, or predictable costs are required.

### When to Consider Local Models

- **Strict data privacy:** medical data, financial documents, government secrets
- **Predictable cost:** fixed infrastructure expenses instead of per-token billing
- **Offline scenarios:** edge computing, working without internet
- **Customization:** full fine-tuning (including RL-based methods like GRPO) on proprietary data
- **Independence:** avoiding vendor lock-in and dependency on external services

### Popular Solutions for Self-Hosting

**Ollama** — the simplest way to run models locally. One-command installation, support for Llama, Mistral, Qwen, DeepSeek, and other open-weight models. Ideal for development and testing.

**vLLM** — a high-performance inference server for production. Continuous batching, speculative decoding, PagedAttention for efficient GPU memory utilization. OpenAI-compatible API.

**SGLang** — optimized for multi-turn conversations via RadixAttention. Excellent for chatbots and agents. Strong structured output support.

**Text Generation Inference (TGI)** — a production-ready solution from Hugging Face. Good documentation, enterprise features, native HF ecosystem integration.

### Hardware Requirements

Modern LLMs require significant resources:

| Model Size | GPU RAM (FP16) | GPU RAM (INT4) | Examples |
|---------------|---------|---------|---------|
| 7-8B | 16 GB | 4-6 GB | Llama 4 8B, Qwen 3 8B, Mistral 7B |
| 14B | 28 GB | 8-10 GB | Qwen 3 14B |
| 32B | 64 GB | 16-20 GB | Qwen 3 32B |
| 70B | 140 GB | 35-40 GB | Llama 3.3 70B, Qwen 3 72B, DeepSeek R1 70B distill |
| MoE 100B+ active | 200+ GB | 80+ GB | Llama 4 Maverick, Qwen 3 235B |

**Quantization** reduces requirements by 2-4x with a small quality loss. 4-bit quantization (GPTQ, AWQ, GGUF) allows running a 70B model on a single 24GB GPU.

---

## Key Takeaways

1. **The LLM provider market is diverse and fast-moving.** OpenAI, Anthropic, Google, xAI, Mistral, DeepSeek, Meta — each has its own strengths. There is no universally "best" provider.

2. **Reasoning models are a new paradigm.** o3, Claude extended thinking, Gemini thinking — they trade latency and cost for dramatically better performance on complex tasks. Use them selectively.

3. **Model choice depends on the task.** Complex reasoning requires reasoning models; standard tasks require standard models; simple tasks require budget models. The right routing strategy saves 70%+ on costs.

4. **Cost is determined by tokens.** Output tokens are 3-5x more expensive than input. Thinking tokens in reasoning models can dominate costs. Caching, batching, and routing can reduce costs by several times.

5. **Open-source models have reached parity.** DeepSeek R1, Llama 4, and Qwen 3 compete with proprietary models. MoE architecture dominates the frontier.

6. **Tool use / function calling** is the standard for agents. All major providers support this capability with compatible formats.

7. **Reliability requires a multi-provider approach.** Fallback, retry with backoff, circuit breakers — mandatory elements of a production system.

8. **Local models** are suitable for specific requirements: privacy, predictable cost, offline operation, full fine-tuning control. They require infrastructure investment.

---

## Brief Code Example

### Basic Integration via LangChain4j

Working with different providers through LangChain4j is unified thanks to the shared `ChatLanguageModel` interface. To create an OpenAI client, use `OpenAiChatModel.builder()` specifying the API key, model name (e.g., "gpt-4o"), temperature, and maximum token count. Anthropic clients are created similarly via `AnthropicChatModel.builder()` with model "claude-sonnet-4-20250514", Google via `GoogleAiGeminiChatModel.builder()` with model "gemini-2.5-pro", and even local models via `OllamaChatModel.builder()` specifying the URL and a model such as "llama4:maverick".

**Multi-provider architecture with fallback:** To ensure reliability, a client is created that maintains a list of providers in priority order. When generating a response, it sequentially attempts each provider starting with the first. If a provider returns an error (e.g., rate limit or server error), the client automatically switches to the next one. To handle transient failures, a retry mechanism with exponential backoff is implemented: the first attempt after 1 second, the second after 2 seconds, the third after 4 seconds, with random jitter added to avoid synchronized retry storms. Only after exhausting all attempts across all providers is a final error thrown.

**Routing by task type:** An efficient architecture routes requests to optimal models depending on task complexity. For simple tasks like classification or data extraction, economical models such as GPT-4o-mini or Claude Haiku 4.5 are used. For complex reasoning and analysis, reasoning models like o3 or Claude Opus 4 with extended thinking are applied. Code generation works best with Claude Sonnet 4 or GPT-4.1, which lead coding benchmarks. For real-time applications where speed is critical, the fastest models such as Claude Haiku 4.5 or Gemini 2.5 Flash are selected.

**Tool Use via AI Services:** LangChain4j provides a declarative way to create agents with tools. Tools are defined as regular Java methods annotated with `@Tool`, which contains the function description. Method parameters are annotated with `@P` including a description for the model. Then an agent interface is created with a `chat(String message)` method, and `AiServices.builder()` is used to automatically generate the implementation. The model independently analyzes the user's request, decides which tools are needed, invokes them with the correct parameters, and forms the final response based on the results.

**Structured Output:** To guarantee structured data output, the `responseFormat("json_schema")` mode is used with OpenAI models. A Java record is defined with the required fields (e.g., Product with fields name, category, price, features), an interface is created with an extraction method annotated with `@UserMessage`, and AI Services automatically handles parsing the response into a typed object. Thanks to constrained decoding, the model is physically unable to generate invalid JSON, providing 100% parsing reliability.

**Cost calculation:** Monitoring LLM API expenses is critically important. Cost is calculated by the formula: (inputTokens / 1,000,000) × inputPrice + (outputTokens / 1,000,000) × outputPrice. For reasoning models, add thinking tokens: (thinkingTokens / 1,000,000) × outputPrice. For a typical workload, monthly expenses can be projected by multiplying the cost per request by the number of requests per day and by 30 days. For example, at 10,000 requests per day with an average of 500 input / 200 output tokens, GPT-4o would cost approximately $600/month, whereas GPT-4o-mini would be only about $27/month — a 22x difference with comparable quality for simple tasks. A single complex reasoning query with o3 might use 5,000 thinking tokens, costing $0.20 per request — making selective routing essential.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[04_Generation_Parameters|Generation Parameters: Controlling Model Behavior]]
**Next:** [[06_Integration_Patterns|LLM Integration Patterns]]
