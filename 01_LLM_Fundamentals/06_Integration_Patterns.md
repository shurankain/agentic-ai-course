# LLM Integration Patterns

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[05_LLM_Providers_and_API|LLM Providers and API Usage]]
**Next:** [[07_Modern_Architectures|Modern LLM Architectures]]

---

Imagine building a bridge between two banks of a river. One side is your application with its requirements for speed, reliability, and scalability. The other is the LLM API with its characteristics: long-running requests, unpredictable latency, rate limits, and potential errors. Integration patterns are the structural design of this bridge, determining how request flows will move between the two sides.

Integrating LLMs into production systems is fundamentally different from experimenting in a Jupyter notebook. In the real world, requests arrive concurrently, users expect fast responses, errors must be handled gracefully, and the API budget must remain under control. This chapter covers patterns that address these challenges.

---

## Synchronous vs Asynchronous Requests

### The Problem with Synchronous Approach

The simplest integration method is a synchronous request: send the prompt, wait for the response, return the result. The code is straightforward, but it scales poorly.

An LLM request takes from 1 to 30+ seconds depending on complexity and response length. During this entire time, your application thread is blocked, waiting for the response. With 100 concurrent users, you need 100 threads simply waiting for an API response. With 1000 users, 1000 threads. This is inefficient and quickly hits system limits.

### Asynchronous Approach

Asynchronous execution is the standard for production LLM integration. Instead of blocking the thread, return a `Future` or `Promise`, allowing the system to process other requests while waiting for the response.

In Java, this is implemented through `CompletableFuture`:
- The request is dispatched on a separate thread pool
- The main thread is released immediately
- The result is processed via callback when ready

With proper implementation, 20-30 threads can serve hundreds of concurrent LLM requests.

### Reactive Approach

For systems under very high load, reactive programming (Project Reactor, RxJava) is a good fit. Reactive streams enable:
- Processing request streams with backpressure control
- Easy composition and transformation of asynchronous operations
- Built-in operators for timeout, retry, fallback

This is particularly useful when integrating with reactive web frameworks (Spring WebFlux).

### Backpressure in Reactive Streams with LLMs

**Problem:** LLM requests take seconds. If incoming requests arrive faster than the system can process them, a backpressure mechanism is needed.

**What is backpressure:**
Backpressure is a signal from consumer to producer saying "slow down, I can't keep up." In reactive streams, this is a built-in mechanism: the subscriber requests only as many elements as it can process.

**Backpressure strategies for LLMs:**

There are three main strategies for managing overload in reactive systems with LLMs:

1. **BUFFER** — buffering with a limit. Incoming requests accumulate in a fixed-size buffer (e.g., 100 requests). When the buffer fills up, new requests are either rejected or evict older ones. This strategy works well for handling temporary load spikes but requires memory.

2. **DROP** — discarding under overload. If the system cannot keep up, incoming requests are simply dropped with logging. Suitable for non-critical operations where losing some requests is acceptable.

3. **LATEST** — keeping only the most recent request. Under overload, all intermediate requests are discarded, and only the freshest one is processed. Ideal for real-time scenarios where recency matters more than completeness.

**Implementation with Flux/Mono:**

When integrating LLMs into a reactive pipeline, it is critical not to block the event loop. The model call is wrapped in `Mono.fromCallable()` and executed on a `boundedElastic` scheduler — a dedicated thread pool for blocking operations. The pipeline includes timeout operators to limit wait time (typically 30-60 seconds), retryWhen for automatic retries on rate limit errors with exponential backoff, and onErrorResume for graceful handling of timeouts and other errors.

**Key operators for LLMs:**

| Operator | Purpose |
|----------|---------|
| `flatMap(fn, concurrency)` | Limiting parallel calls |
| `subscribeOn(boundedElastic)` | Moving blocking calls off the event loop |
| `timeout(duration)` | Time limit per operation |
| `retryWhen(spec)` | Configurable retry |
| `onBackpressureBuffer/Drop` | Overload control |
| `delayElements(duration)` | Rate limiting via delay |

**Pattern: Controlled Concurrency Pipeline**

A typical production pipeline for LLMs combines all of these mechanisms: buffering with eviction of old requests (DROP_OLDEST), concurrency limiting via flatMap (e.g., maximum 20 simultaneous LLM calls), error handling that returns an empty result instead of breaking the stream, and collection of success/failure metrics via doOnNext and doOnError.

---

## Streaming Responses

### Why Streaming Is Needed

When a model generates a long response, the user may wait 10-20 seconds before the first character appears. This creates a perception of the system "hanging" and results in poor UX.

Streaming solves this problem: the response is sent token by token as it is generated. The first words appear within 200-500ms, and the user sees that the system is working.

Streaming is especially important for:
- Chatbots and interactive assistants
- Long responses (articles, code, analysis)
- Real-time applications

### How Streaming Works

All major providers support streaming via Server-Sent Events (SSE):

1. The client opens an HTTP connection
2. The server (provider API) sends tokens as they are generated
3. The connection remains open until the response is complete
4. Each token is a separate SSE event

LangChain4j provides `StreamingChatLanguageModel` with a callback interface for token processing.

### Streaming in Web Applications

The following approaches are used to deliver a streaming response to the client:

**Server-Sent Events (SSE):** A unidirectional stream from server to client. Simple protocol with good browser support. Ideal for chat interfaces.

**WebSocket:** Bidirectional communication. Required when the client sends data during streaming (e.g., to cancel generation).

In Spring Boot, an SSE endpoint is created by returning a `Flux<String>` with the `text/event-stream` content type.

---

## Rate Limiting and Retry

### Understanding Rate Limits

All LLM providers impose limits:
- **RPM (Requests per Minute):** number of requests per minute
- **TPM (Tokens per Minute):** number of tokens per minute
- **RPD (Requests per Day):** daily limit

When the limit is exceeded, the API returns 429 Too Many Requests. Continuing to bombard the API with requests is pointless — you need to wait.

### Exponential Backoff with Jitter

The standard pattern for handling rate limits:

1. On receiving a 429 error — wait and retry
2. Wait time increases exponentially: 1s → 2s → 4s → 8s...
3. Add randomness (jitter) to the delay

Jitter is critical: without it, during a mass rate limit event, all clients will retry simultaneously, creating a "thundering herd" — a flood of requests at the same moment.

### Respect Retry-After

Many APIs include a `Retry-After` header in the 429 response, indicating how many seconds to wait. Always check this header — it provides the exact time after which you can retry the request.

### Token Bucket for Rate Control

For proactive control (instead of waiting for a 429, limit requests in advance), the Token Bucket algorithm is used:

- A "bucket" holds a certain number of tokens
- Each request "spends" one token
- Tokens are replenished at a fixed rate
- If no tokens are available — the request waits in the queue

This allows even distribution of load and avoids spikes.

### Queueing Theory for LLM Serving

For proper capacity planning of LLM systems, understanding the basics of queueing theory is useful.

**Little's Law — the fundamental law:**

The formula `L = λ × W` connects three key metrics:
- `L` — average number of requests in the system (queue + processing)
- `λ` — request arrival rate (requests/sec)
- `W` — average time in the system (queue time + processing time)

**Application example:**

If you have 10 requests per second and each takes 3 seconds to process, then L = 10 × 3 = 30 requests are simultaneously in the system. This means you need at least 30 concurrent connections. But this is the average case. To handle peaks (P99 latency = 15s), you need peak = 10 × 15 = 150 connections. Therefore, capacity planning should target percentile latency, not the average.

**M/M/c Model for Capacity Planning:**

The classic queueing model helps determine system utilization using the formula `Utilization (ρ) = λ / (c × μ)`, where μ is the service rate (requests per second per worker), and c is the number of workers.

**Utilization rules:**
- ρ > 1 → the queue grows infinitely (system is overloaded)
- ρ > 0.8 → sharp latency increase (danger zone)
- ρ = 0.6-0.7 → optimal load

**Practical capacity planning:**

Suppose you need to handle 100 requests per second, and the LLM latency is 3 seconds on average. Service rate per worker = 1/3 = 0.33 requests per second. Minimum workers: 100 / 0.33 = 300. But accounting for a target utilization of ρ = 0.7 for headroom: 300 / 0.7 ≈ 430 concurrent connections. In practice, plan for ~450 connections to account for burst load.

**Why this matters for LLMs:**
- LLM latency is high (seconds) and variable — unlike microsecond REST APIs
- Cost of waiting is high — users leave after a few seconds of waiting
- API rate limits create an artificial ceiling that cannot be overcome simply by adding workers
- Load spikes require buffer capacity, otherwise the queue overflows

---

## Caching

Caching is one of the most effective ways to optimize LLM integration. It reduces costs, decreases latency, and alleviates API load.

### Exact Match Caching

The simplest form of caching: store a "prompt → response" pair and return the stored response when an identical prompt is received again.

**When effective:**
- Deterministic tasks (classification, data extraction)
- Temperature = 0 (identical prompt produces identical response)
- Popular recurring queries

**When it does not work:**
- Creative tasks (diversity is needed)
- Unique queries (low hit rate)
- Queries with temporal context

Implementation is trivial: hash the prompt as the key, response as the value. Caffeine, Redis, any cache will do.

### Semantic Caching

A more advanced approach: instead of looking for an exact match, search for semantically similar queries.

"What's the weather in Moscow?" and "Moscow weather today" are different strings but semantically identical. A semantic cache will find the match.

**How it works:**
1. Create an embedding (vector) for each query
2. Search the vector store for the nearest stored query
3. If similarity is above the threshold — return the stored response
4. Otherwise — generate a new response and store it

**Trade-offs:**
- Requires an embedding model (additional latency and cost)
- Requires a vector store
- Choosing the similarity threshold — too high = few hits, too low = inaccurate responses

### Similarity Thresholds: Theory and Practice

Choosing the similarity threshold is a key decision in semantic caching:

| Threshold (cosine) | Hit Rate | False Positive Risk | Use Case |
|---------------------|----------|---------------------|----------|
| 0.99+ | Very low | Minimal | Nearly identical only |
| 0.95-0.98 | Low | Low | Conservative cache |
| 0.90-0.95 | Medium | Moderate | Balanced (recommended) |
| 0.85-0.90 | High | Significant | Aggressive cache |
| <0.85 | Very high | High | Risky |

**How to choose the threshold:**
1. Start with a conservative value (0.95)
2. Collect metrics: hit rate, false positive rate
3. Gradually lower it to increase hit rate
4. Stop when false positives become noticeable

**Dependency on embedding model:**
- Different models produce different similarity distributions
- text-embedding-3-small: more "concentrated" distributions
- sentence-transformers: more "spread out" distributions
- Calibrate the threshold for the specific model

### Cache Invalidation

"There are only two hard things in Computer Science: cache invalidation and naming things."

For LLM caches, invalidation is especially important:

**Time-based invalidation (TTL):**
- The simplest approach — expire after N minutes/hours
- Suitable for FAQ-style queries
- Not suitable for dynamic content

**Event-based invalidation:**

Cache invalidation is triggered by system events:
- Knowledge base update → invalidate related queries
- Business logic change → invalidate affected categories
- New model release → consider full invalidation

**Versioning:**

Add a version tag to the cache key. When prompts change, use a new version, and the old cache will be automatically ignored. For example, the cache key is formed as hash(query + prompt_version), where prompt_version is incremented with each prompt template change.

**Selective invalidation:**

A tag-based grouping pattern enables invalidating entire categories of queries. When saving a response, tags are added (e.g., "weather" and "moscow"), and then all entries with the "weather" tag can be invalidated with a single command. This is convenient for topic-specific data updates.

### Provider-Level Prompt Caching

OpenAI and Anthropic offer built-in caching of system prompts. If the prefix of your request (system prompt) repeats, you receive a discount on input tokens.

For agents, this is particularly beneficial: the system prompt and tool descriptions are the same in every request and can amount to thousands of tokens.

---

## Cost Optimization

### Complexity-Based Routing

Not all queries require a powerful model. A simple question like "How many days are in a week?" does not need GPT-4 — GPT-4o-mini handles it just as well, but at 15x lower cost.

Model Router pattern:
1. Classify query complexity (using a cheap model)
2. Route to the appropriate model
3. Simple → cheap model, Complex → powerful model

The classifier itself requires an LLM call, but this pays off at high query volumes.

### Prompt Compression

Longer prompts = more input tokens = higher cost. Compression strategies:

**Remove redundancy:** multiple spaces, blank lines, repeated instructions — all of these are tokens.

**Compress context:** if a long document needs to be included, extract only the relevant parts first.

**Concise examples:** in few-shot prompting, select the shortest examples that preserve the intent.

### Batching

Many providers offer a Batch API with a 50% discount: you submit a batch of requests, they are processed outside of real-time, and results are available within hours.

Suitable for:
- Document processing
- Content generation
- Analytical tasks
- Anything that does not require an immediate response

---

## Error Handling

### Error Types

LLM APIs can return different types of errors:

**Rate Limit (429):** request limit exceeded. Solution: backoff and retry.

**Timeout:** request took too long. Solution: retry with possibly lower max_tokens.

**Invalid Request (400):** incorrect prompt or parameters. Solution: fix the request; retry will not help.

**Content Filter:** response blocked by moderation. Solution: rephrase the request or handle as a content error.

**Server Error (500, 503):** issues on the provider side. Solution: retry or fallback to a different provider.

### Circuit Breaker

The Circuit Breaker pattern prevents cascading failures:

**Closed (normal state):** requests pass through to the API

**Open (failure state):** after N consecutive errors, the circuit "opens" — requests immediately return an error without attempting to reach the API

**Half-Open (probe state):** periodically allows a single request through to check availability. If successful — returns to Closed; if not — remains Open.

This protects the system: when the API is unavailable, resources are not wasted on futile attempts, and errors are returned to the user quickly.

### Fallback Strategies

**Fallback to an alternative provider:** if OpenAI is unavailable, try Anthropic.

**Graceful degradation:** if the LLM is unavailable, return a pre-prepared response or a simplified version of the functionality.

**Stale cache:** on error, return an outdated cached response. Old data is better than no data.

---

## Monitoring and Metrics

### Key Metrics

**Latency:** time from request to complete response. Track p50, p90, p99.

**Token Usage:** number of input and output tokens. Directly impacts cost.

**Cost:** approximate or exact cost of requests. Aggregate by model, by endpoint, by user.

**Error Rate:** percentage of failed requests. Break down by error type.

**Cache Hit Rate:** caching effectiveness.

### Alerting

Set up alerts for:
- Error rate > 5% — something is broken
- P99 latency > 30s — performance has degraded
- Daily cost > budget — budget overrun
- Cache hit rate dropped — possible cache issues

### Distributed Tracing

For agents with many LLM calls, it is important to see the full picture:
- Which calls occurred within a single user request
- How long each one took
- Where the error occurred

LangChain4j integrates with OpenTelemetry, LangSmith, and other tracing systems.

---

## Key Takeaways

1. **Asynchronous execution is mandatory** for production. Synchronous calls do not scale — 20 threads cannot serve 200 users.

2. **Streaming** improves UX — the user sees the response immediately rather than after 10 seconds of waiting.

3. **Rate limiting and retry** are not optional but essential. Exponential backoff with jitter is the industry standard.

4. **Caching** (exact and semantic) significantly reduces costs and latency. With temperature=0, exact caching is highly effective.

5. **Complexity-based routing** allows using cheap models for simple tasks, saving budget.

6. **Circuit breaker and fallback** ensure reliability. The system must function even during provider outages.

7. **Monitoring** covers latency, tokens, cost, and errors. Without metrics, you are flying blind.

---

## Brief Code Example

Below is a minimal example demonstrating key LLM integration patterns in a single class:

A **production-ready LLM client** combines several critical patterns. The class is initialized with a model and a thread pool for asynchronous execution. Internally, it creates an LRU cache with a size limit (e.g., 1000 entries) and a TTL (1 hour via Caffeine), a rate limiter to control request frequency (e.g., 10 per second via Guava), and a circuit breaker to prevent cascading failures with a 50% error threshold and a 30-second wait time.

The asynchronous generation method first checks the cache by prompt hash, returning the stored result on a match. If there is no cache hit, the request is executed asynchronously via CompletableFuture: first, the rate limiter is acquired to respect limits, then the circuit breaker executes the request with retry logic and exponential backoff, and finally the result is stored in the cache before being returned.

This approach provides asynchronous execution for scalability, caching for cost reduction, rate limiting for API compliance, circuit breaker for fault tolerance, and retry for handling transient failures.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[05_LLM_Providers_and_API|LLM Providers and API Usage]]
**Next:** [[07_Modern_Architectures|Modern LLM Architectures]]
