# LLM Providers and Working with APIs

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[04_Generation_Parameters|Generation Parameters: Controlling Model Behavior]]
**Next:** [[06_Integration_Patterns|LLM Integration Patterns]]

---

Consider the cloud computing market in the early 2010s: AWS dominated, Azure and Google Cloud were emerging, each offering their services with different APIs, pricing, and capabilities. Today we observe a similar landscape in the world of large language models. OpenAI, Anthropic, Google, Mistral, and dozens of other providers offer their models through APIs, and choosing the right provider becomes a strategic decision for any project involving AI agents.

Understanding the LLM provider landscape is not simply a matter of picking the "best model." It is a question of balancing quality, cost, speed, reliability, and compliance with security requirements. For agent developers, this knowledge shapes system architecture, fallback strategy, and project budget.

---

## Major Providers and Their Models

### OpenAI: Pioneer and Market Leader

OpenAI is the company that popularized LLMs through ChatGPT and remains one of the key market players. Their API has become the de facto standard that other providers emulate.

**Flagship models:**
- **GPT-4o** — multimodal model (text, images, audio). Faster and cheaper than previous GPT-4 versions, with comparable quality.
- **GPT-4 Turbo** — extended context up to 128K tokens, optimized for long documents.
- **o1** — model with a "reasoning mode," specialized for complex logical tasks.

**Budget options:**
- **GPT-4o-mini** — lightweight version, significantly cheaper with good quality for simple tasks.
- **GPT-3.5 Turbo** — legacy but very inexpensive model for straightforward tasks.

**OpenAI highlights:**
- Most mature ecosystem and documentation
- Broad support across SDKs and frameworks
- Native support for function calling / tool use
- Structured outputs (guaranteed JSON via constrained decoding)
- Batching API for bulk processing at a 50% discount
- Prompt caching for savings on repeated prefixes

### Anthropic: Focus on Safety

Anthropic, founded by former OpenAI researchers, positions itself as a company focused on AI safety. Their Claude models are distinguished by excellent instruction following and long context.

**Claude model family:**
- **Claude 3.5 Sonnet** — the primary workhorse, excellent balance of quality and speed. Leader in code generation benchmarks.
- **Claude 3 Opus** — flagship for the most complex tasks.
- **Claude 3.5 Haiku** — fast and cost-effective model for simple tasks.

**Anthropic highlights:**
- Massive context window — up to 200K tokens
- "Constitutional AI" — models trained to follow ethical principles
- Excellent performance in long document analysis
- Less prone to hallucinations in certain scenarios
- Tool use API compatible with the OpenAI format
- Particularly strong for code generation and analysis

### Google: Multimodality and Scale

Google offers models through Vertex AI (enterprise) and Google AI Studio (for developers).

**Gemini family:**
- **Gemini 1.5 Pro** — multimodal model with a record context of up to 1M+ tokens
- **Gemini 1.5 Flash** — optimized for speed and cost, ideal for real-time applications
- **Gemini Ultra** — the most powerful model for complex reasoning

**Google advantages:**
- Record context sizes — entire books can be loaded
- Native integration with Google Cloud services
- Strong multimodality (video, audio, images)
- Grounding with Google Search for up-to-date information
- Competitive pricing

### Mistral AI: The European Alternative

A French startup offering efficient open-weight models. Attractive for European companies due to GDPR compliance.

**Models:**
- **Mistral Large** — flagship model, GPT-4 competitor
- **Mistral Medium** — balance of performance and cost
- **Mistral Small / Ministral** — compact models for simple tasks
- **Codestral** — specialized for code generation

**Mistral highlights:**
- Models can be run locally (open weights)
- Competitive pricing with good quality
- GDPR compliance out of the box (EU company)
- Sliding window attention for efficient handling of long contexts

### Other Notable Providers

**Cohere** — specializes in enterprise search and RAG. Embed models for embeddings, Command for generation, Rerank for re-ranking search results.

**Meta (Llama)** — fully open-source models that can be self-hosted. Llama 3 70B/405B competes with proprietary models on quality.

**Amazon Bedrock** — access to models from multiple providers (Anthropic, Meta, Mistral) through a unified AWS API with enterprise guarantees.

**Azure OpenAI** — OpenAI models with enterprise SLA, compliance, and integration with the Azure ecosystem.

---

## Open-Source Models: In-Depth Comparison

In 2024-2025, open-source models reached parity with proprietary ones. DeepSeek V3, Llama 3.3, and Qwen 2.5 are the three flagships, each with its own strengths.

### DeepSeek V3

A Chinese model that revolutionized training efficiency. 671B parameters, but only 37B active (Mixture of Experts).

**Key innovations:**
- **Multi-head Latent Attention (MLA)**: KV-cache compression via low-rank projections — 93% memory savings
- **Multi-Token Prediction (MTP)**: Predicting 2 tokens per step — 1.8x inference speedup
- **FP8 Training**: Full training in 8-bit floating point — the first successful example at this scale
- **Auxiliary-loss-free Load Balancing**: Even load distribution across experts without additional loss terms

**Training cost**: $5.5M — an order of magnitude cheaper than competitors (GPT-4 is estimated at $100M+).

**Performance (benchmarks):**
| Benchmark | DeepSeek V3 | GPT-4o | Claude 3.5 Sonnet |
|-----------|-------------|--------|-------------------|
| MMLU | 88.5 | 88.7 | 88.7 |
| HumanEval | 82.6 | 90.2 | 92.0 |
| MATH-500 | 90.2 | 74.6 | 78.3 |
| Codeforces | 51.6% | 23.0% | 20.3% |

**When to choose DeepSeek V3:**
- Mathematical and algorithmic tasks (leader)
- Self-hosting with a limited GPU budget (MoE efficiency)
- Tasks in Chinese
- Research on MoE architectures

### Llama 3.3 70B

The latest model from Meta, optimized for accessibility. A single 70B model matches the quality of the previous generation's 405B.

**Features:**
- **Dense architecture**: Traditional architecture, simpler for deployment
- **128K context window**: Long context out of the box
- **Optimized inference**: Carefully tuned for popular inference frameworks
- **Strong multilingual**: 8 major languages with high quality

**Performance:**
| Benchmark | Llama 3.3 70B | Llama 3.1 405B | GPT-4o-mini |
|-----------|---------------|----------------|-------------|
| MMLU | 86.0 | 87.3 | 82.0 |
| HumanEval | 88.4 | 89.0 | 87.2 |
| GSM8K | 91.1 | 96.8 | 87.0 |
| MGSM | 91.1 | 91.6 | 86.5 |

**When to choose Llama 3.3:**
- Production self-hosting (mature ecosystem)
- When predictability and stability matter
- European multilingual deployment
- Integration with existing Meta AI infrastructure

### Qwen 2.5

A model series from Alibaba, leading many benchmarks in its size class.

**Size range:**
- **Qwen 2.5 72B** — flagship, GPT-4 competitor
- **Qwen 2.5 32B** — sweet spot for a single GPU
- **Qwen 2.5 14B** — efficient for edge deployment
- **Qwen 2.5-Coder** — specialized for code

**Features:**
- **Long context**: Up to 128K tokens
- **Strong reasoning**: Improved chain-of-thought capabilities
- **Best in code among open-source** (Qwen-Coder variants)
- **Excellent multilingual**: Particularly strong in Asian languages

**Qwen 2.5 72B performance:**
| Benchmark | Qwen 2.5 72B | Llama 3.1 70B | GPT-4o |
|-----------|--------------|---------------|--------|
| MMLU | 86.1 | 83.6 | 88.7 |
| HumanEval | 86.6 | 80.5 | 90.2 |
| MATH | 83.1 | 68.0 | 76.6 |
| GPQA | 49.0 | 46.7 | 53.6 |

**When to choose Qwen 2.5:**
- Coding tasks (Qwen-Coder leads)
- Asian languages and markets
- When a full size range matters (from 0.5B to 72B)
- Research and experimentation

### Comparison Table

| Criterion | DeepSeek V3 | Llama 3.3 70B | Qwen 2.5 72B |
|----------|-------------|---------------|--------------|
| **Architecture** | MoE (671B/37B) | Dense | Dense |
| **Context** | 128K | 128K | 128K |
| **Licensing** | MIT-like | Llama 3.3 License | Apache 2.0 |
| **Commercial use** | ✅ | ✅ (with restrictions) | ✅ |
| **GPU for 16-bit** | 8x80GB | 2x80GB | 2x80GB |
| **GPU for 4-bit** | 2x80GB | 1x80GB | 1x80GB |
| **Inference speed** | Fast (MoE) | Medium | Medium |
| **Reasoning** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Coding** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Math** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Multilingual** | ⭐⭐ (CN focus) | ⭐⭐⭐ | ⭐⭐⭐ |

### Practical Recommendations

**For a startup with a limited budget:** start with Qwen 2.5 in a 14B or 32B parameter configuration — this provides a good balance between quality and resources. As the project grows, you can gradually migrate to the 72B version. For compute-intensive mathematical tasks, consider DeepSeek V3, which shows leading results in this area.

**For enterprise self-hosting:** the optimal choice is Llama 3.3 70B thanks to its mature ecosystem and clear licensing. For inference, use specialized solutions such as vLLM or Text Generation Inference (TGI) that deliver high performance. For specific code generation tasks, apply Qwen-Coder variants optimized for this workload.

**For research:** DeepSeek V3 is of particular interest for studying Mixture of Experts (MoE), Multi-head Latent Attention (MLA), and Multi-Token Prediction (MTP) architectures. The Qwen 2.5 series is well suited for experiments across different model sizes. For comparative analysis, working with all three families simultaneously provides a complete understanding of their strengths and weaknesses.

### Interview Questions: Open-Source Models

1. **Compare the architectures of DeepSeek V3 and Llama 3.3. What are the trade-offs?**
   - DeepSeek: MoE — fewer active parameters, faster inference, more complex deployment
   - Llama: Dense — predictable, simple deployment, requires more memory per parameter
   - MoE wins when there is sufficient memory to load all experts

2. **When is an open-source model preferable to a proprietary one?**
   - Data privacy requirements (healthcare, finance)
   - Predictable costs at high volume
   - Need for fine-tuning
   - Offline/edge deployment
   - Vendor independence

3. **How would you choose a model for a multilingual RAG system?**
   - Analyze the language distribution in the corpus
   - For European languages: Llama 3.3
   - For Asian languages: Qwen 2.5
   - For mixed: benchmarks on the target languages
   - Consider tokenizer efficiency for different languages

---

## API Structure: Common Principles

Despite differences between providers, most LLM APIs follow similar principles. Understanding this common structure allows you to switch between providers easily and write provider-agnostic code.

### Message Format

Virtually all modern APIs use a "messages" format — an array of messages with roles:

- **system** — instructions for the model defining its behavior, style, and constraints
- **user** — user messages
- **assistant** — previous model responses
- **tool** — results of tool invocations

This format reflects the conversation structure and allows the model to understand dialog context. The system prompt is typically placed first and defines the assistant's "persona."

### Core Request Parameters

A typical API request includes:
- **model** — model identifier ("gpt-4o", "claude-3-5-sonnet-20241022")
- **messages** — array of messages
- **temperature** — degree of randomness in responses (0-2)
- **max_tokens** — maximum response length
- **top_p** — nucleus sampling
- **stop** — character sequences that trigger generation stop
- **tools** — descriptions of available tools (for agents)

### Response Format

The response typically contains:
- **content** — generated text
- **usage** — token usage statistics (input, output, total)
- **finish_reason** — reason for stopping (stop, length, tool_calls, content_filter)
- **tool_calls** — requested tool invocations (if the model decided to use a tool)

---

## Pricing Comparison

Cost is a critical factor when choosing a provider and model. All providers charge by token count, with input and output tokens priced differently.

### Pricing Tiers (approximate prices per 1M tokens, end of 2024)

**Premium models:**

| Model | Input | Output |
|--------|-------|--------|
| GPT-4o | $2.50 | $10.00 |
| Claude 3.5 Sonnet | $3.00 | $15.00 |
| Claude 3 Opus | $15.00 | $75.00 |
| Gemini 1.5 Pro | $1.25 | $5.00 |
| Mistral Large | $2.00 | $6.00 |

**Budget models:**

| Model | Input | Output |
|--------|-------|--------|
| GPT-4o-mini | $0.15 | $0.60 |
| Claude 3.5 Haiku | $0.25 | $1.25 |
| Gemini 1.5 Flash | $0.075 | $0.30 |
| Mistral Small | $0.10 | $0.30 |

### Factors Affecting Actual Cost

**Input/output ratio:** Output tokens are typically 3-5x more expensive than input tokens. For tasks with long prompts and short responses (classification, data extraction), this works in your favor. For generating long texts, output becomes the primary expense.

**Prompt caching:** OpenAI, Anthropic, and others offer discounts when the same prefixes are reused. An agent's system prompt repeats in every request — caching can save 50-90% on its cost.

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

Different tasks require different models. Using GPT-4 for simple classification wastes money and adds latency. Using GPT-3.5 for complex reasoning produces unreliable results and wastes time correcting errors.

### Complex Reasoning and Analysis

**Recommended models:**
- Claude 3 Opus / Claude 3.5 Sonnet
- GPT-4o / o1
- Gemini 1.5 Pro

**Typical tasks:** multi-step planning, complex document analysis, code review, architectural decisions, research tasks.

### Code Generation

**Recommended models:**
- Claude 3.5 Sonnet (benchmark leader)
- GPT-4o
- Codestral (Mistral)

**Typical tasks:** writing new code, refactoring, debugging, writing tests, documentation.

### Simple Tasks and Classification

**Recommended models:**
- GPT-4o-mini
- Claude 3.5 Haiku
- Gemini 1.5 Flash

**Typical tasks:** text categorization, entity extraction, sentiment analysis, simple format conversions, filtering.

### Long Documents

**Recommended models:**
- Gemini 1.5 Pro (1M+ token context)
- Claude 3.5 Sonnet (200K tokens)
- GPT-4 Turbo (128K tokens)

**Typical tasks:** analyzing books, lengthy reports, codebases, legal documents, scientific papers.

### Real-Time Applications

**Recommended models:**
- Gemini 1.5 Flash
- GPT-4o-mini
- Claude 3.5 Haiku

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

**Anthropic:** Compatible format, high reliability in selecting the correct tool. Claude is particularly strong in complex scenarios with many tools.

**Google:** Full-featured function calling support, integration with Google Cloud Functions for serverless execution.

**Mistral:** Tool support exists, but the implementation is less mature compared to the leaders.

### Structured Outputs

An important newer capability — guaranteed conformance of the response to a JSON schema. OpenAI implemented this through constrained decoding: at the token generation level, the model is physically unable to produce invalid JSON.

This is critical for agents, where parsing reliability must be 100%. Previously, you had to handle parsing errors and retry — now that is unnecessary.

---

## Reliability Strategies

Production systems cannot depend on a single provider. APIs go down, rate limits are hit, and models become temporarily unavailable. Reliability requires a well-designed architecture.

### Multi-Provider Architecture

A good practice is to have a fallback to an alternative provider:

1. **Primary request** to the preferred provider (e.g., OpenAI)
2. **On failure** — automatic switch to the fallback (Anthropic)
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

LLM requests can take from one second to a minute. Rules:

- **Reasonable timeouts:** 30-60 seconds for generation, 10-20 seconds for classification
- **Retry with exponential backoff:** start at 1 second, double up to a maximum
- **Jitter:** add randomness to the delay to avoid thundering herd
- **Circuit breaker:** after N consecutive errors, stop trying and return an error immediately

### SLA and Reliability Engineering for LLM APIs

When designing production systems, understanding the actual provider guarantees and planning accordingly is critically important.

**Typical SLA metrics:**

| Metric | Description | Typical Values |
|---------|----------|-------------------|
| **Availability** | Service uptime | 99.9% - 99.99% |
| **Latency P50** | Median response time | 1-3 sec |
| **Latency P95** | 95th percentile | 5-15 sec |
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
- **Customization:** fine-tuning on proprietary data
- **Independence:** avoiding vendor lock-in and dependency on external services

### Popular Solutions for Self-Hosting

**Ollama** — the simplest way to run models locally. One-command installation, support for Llama, Mistral, and other open-weight models. Ideal for development and testing.

**vLLM** — a high-performance inference server for production. Continuous batching, speculative decoding, PagedAttention for efficient GPU memory utilization.

**Text Generation Inference (TGI)** — a solution from Hugging Face. Production-ready, supports many models, good documentation.

**LocalAI** — an OpenAI-compatible API for local models. Drop-in replacement for the OpenAI SDK.

### Hardware Requirements

Modern LLMs require significant resources:

| Model Size | GPU RAM | Examples |
|---------------|---------|---------|
| 7B | 8-16 GB | Mistral 7B, Llama 3 8B |
| 13B | 16-24 GB | Llama 2 13B |
| 34B | 40-48 GB | CodeLlama 34B |
| 70B | 80+ GB | Llama 3 70B (requires 2+ GPUs) |

**Quantization** reduces requirements by 2-4x with a small quality loss. 4-bit quantization allows running a 70B model on a single 24GB GPU.

---

## Key Takeaways

1. **The LLM provider market is diverse.** OpenAI, Anthropic, Google, Mistral — each has its own strengths. There is no universally "best" provider.

2. **Model choice depends on the task.** Complex reasoning requires powerful models; simple tasks require economical ones. The right choice saves money and time.

3. **Cost is determined by tokens.** Output tokens are 3-5x more expensive than input. Caching, batching, and prompt optimization can reduce costs by several times.

4. **Tool use / function calling** is the standard for agents. All major providers support this capability with compatible formats.

5. **Reliability requires a multi-provider approach.** Fallback, retry with backoff, circuit breakers — mandatory elements of a production system.

6. **Local models** are suitable for specific requirements: privacy, predictable cost, offline operation. They require infrastructure investment.

---

## Brief Code Example

### Basic Integration via LangChain4j

Working with different providers through LangChain4j is unified thanks to the shared `ChatLanguageModel` interface. To create an OpenAI client, use `OpenAiChatModel.builder()` specifying the API key, model name (e.g., "gpt-4o"), temperature, and maximum token count. Anthropic clients are created similarly via `AnthropicChatModel.builder()` with model "claude-3-5-sonnet-20241022", Google via `GoogleAiGeminiChatModel.builder()` with model "gemini-1.5-pro", and even local models via `OllamaChatModel.builder()` specifying the URL and a model such as "llama3:70b".

**Multi-provider architecture with fallback:** To ensure reliability, a client is created that maintains a list of providers in priority order. When generating a response, it sequentially attempts each provider starting with the first. If a provider returns an error (e.g., rate limit or server error), the client automatically switches to the next one. To handle transient failures, a retry mechanism with exponential backoff is implemented: the first attempt after 1 second, the second after 2 seconds, the third after 4 seconds, with random jitter added to avoid synchronized retry storms. Only after exhausting all attempts across all providers is a final error thrown.

**Routing by task type:** An efficient architecture routes requests to optimal models depending on task complexity. For simple tasks like classification or data extraction, economical models such as GPT-4o-mini or Claude Haiku are used. For complex reasoning and analysis, powerful models like GPT-4o or Claude Sonnet are applied. Code generation works best with Claude 3.5 Sonnet, which leads coding benchmarks. For real-time applications where speed is critical, the fastest models such as Claude Haiku or Gemini Flash are selected.

**Tool Use via AI Services:** LangChain4j provides a declarative way to create agents with tools. Tools are defined as regular Java methods annotated with `@Tool`, which contains the function description. Method parameters are annotated with `@P` including a description for the model. Then an agent interface is created with a `chat(String message)` method, and `AiServices.builder()` is used to automatically generate the implementation. The model independently analyzes the user's request, decides which tools are needed, invokes them with the correct parameters, and forms the final response based on the results.

**Structured Output:** To guarantee structured data output, the `responseFormat("json_schema")` mode is used with OpenAI models. A Java record is defined with the required fields (e.g., Product with fields name, category, price, features), an interface is created with an extraction method annotated with `@UserMessage`, and AI Services automatically handles parsing the response into a typed object. Thanks to constrained decoding, the model is physically unable to generate invalid JSON, providing 100% parsing reliability.

**Cost calculation:** Monitoring LLM API expenses is critically important. Cost is calculated by the formula: (inputTokens / 1,000,000) × inputPrice + (outputTokens / 1,000,000) × outputPrice. For a typical workload, monthly expenses can be projected by multiplying the cost per request by the number of requests per day and by 30 days. For example, at 10,000 requests per day with an average of 500 input / 200 output tokens, GPT-4o would cost approximately $600/month, whereas GPT-4o-mini would be only about $27/month — a 22x difference with comparable quality for simple tasks.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[04_Generation_Parameters|Generation Parameters: Controlling Model Behavior]]
**Next:** [[06_Integration_Patterns|LLM Integration Patterns]]
