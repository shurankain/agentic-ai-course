# LLM Application Deployment Strategies

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Deployment
**Previous:** [[../12_Observability/04_AgentOps|AgentOps]]
**Next:** [[02_Containerization_and_Kubernetes|Containerization and Kubernetes]]

---

## Deployment Specifics in the World of Language Models

Deploying LLM applications differs fundamentally from traditional web services. Classic microservices process requests in milliseconds, consume predictable resources, and scale linearly. LLM systems break all of these assumptions: a single request takes seconds or minutes, resource consumption varies depending on context length, and scaling costs are determined not only by infrastructure but also by the price of API calls.

Response time increases from 50 milliseconds to 5 seconds — a hundredfold. Traditional approaches to load balancing, timeouts, and circuit breakers require a complete rethink. Dependence on external APIs (OpenAI, Anthropic, Google) adds a layer of complexity: providers have their own rate limits, experience temporary outages, and change model behavior without notice. The system must ensure graceful degradation instead of catastrophic failures.

## LLM Application Architecture

A typical architecture consists of several layers. The API Gateway provides authentication, rate limiting, and basic routing — the first line of defense. The Application Layer implements business logic: Chat API for conversations, Agent API for autonomous agents, Completion API for generation. Each service is specialized, but all use a shared LLM Orchestration Layer — the heart of the system, where prompt management, retry logic, fallback between providers, caching, and cost monitoring take place. At the lowest level — LLM providers accessed through a unified abstraction.

## Deployment Patterns

**Synchronous pattern** is suitable for short operations under 30 seconds: classification, entity extraction, short responses. The critical requirement is predictable execution time. The fundamental limitation: HTTP connections have timeouts at all levels. Synchronous requests occupy threads throughout the entire processing, limiting concurrency. Each subsequent stack level must have a timeout slightly longer than the previous one.

**Asynchronous pattern with polling** solves the timeout problem. The client sends a request, receives a task ID, then polls the status until completion. The HTTP connection closes immediately, processing continues in the background. Requirements: task storage (database or Redis), a queue (RabbitMQ, Kafka, SQS), workers. The Retry-After header hints at the polling interval for the client: 5-10 seconds for pending, less for processing with known progress.

**Server-Sent Events (SSE)** is critical for interactive applications. The user sees tokens as they are generated. A 500 ms delay to the first token feels better than a 5-second wait for the full response. SSE runs over HTTP, simplifying infrastructure. It requires end-to-end support: provider, application server, load balancer, reverse proxy with buffering disabled. Errors are transmitted as special events.

**WebSocket** provides full-duplex communication. Ideal for interactive agents requiring user input during execution. Client message types: start_task, provide_input, cancel. Server message types: thought, action, tool_result, need_input, complete, error. Requires a sticky sessions mechanism or centralized state storage. A heartbeat mechanism keeps the connection active.

## Fault Tolerance

**Circuit Breaker** prevents cascading failures. When the error threshold is reached (50% over 10 calls), the circuit opens, and subsequent requests immediately return an error. After 30 seconds — a probe call; on success, the circuit closes again.

**Fallback strategies**: switching to an alternative provider (GPT-4o → Claude), degrading to a cheaper model (GPT-4o → GPT-4o-mini), cached response for similar queries, honest unavailability message. The choice depends on quality and availability requirements. A multi-provider setup requires a unified interface.

### Multi-Provider Failover Architecture

Production systems should not depend on a single LLM provider. A robust failover architecture uses a **unified abstraction layer** — tools like LiteLLM provide a unified API for 100+ LLM providers with cost routing, budget enforcement, and spend tracking behind a single interface. The application code calls LiteLLM; LiteLLM routes to the configured provider.

**Failover chain:** Primary provider (e.g., Anthropic) → Secondary provider (e.g., OpenAI) → Degraded mode (cached responses or simpler model). Each transition is triggered by circuit breaker thresholds. Provider health is monitored continuously — latency percentiles, error rates, and availability.

**Cost-aware routing** extends failover into an optimization strategy: route simple queries (classification, extraction) to cheap models, complex queries (reasoning, analysis) to expensive ones. This is not just failover — it is intelligent routing that reduces cost by 50-70% during normal operation while providing automatic failover during outages. See [[../03_AI_Agents_Core/10_Resource_Optimization|Resource Optimization]] for routing patterns.

**Timeouts** are differentiated by endpoint: Completion 30 seconds, Chat 60 seconds, Agent 5 minutes. At the LLM call level, expected generation length is taken into account. Retry logic considers idempotency: completion is safe, agent workflows are riskier due to side effects. Exponential backoff with jitter: second attempt after 1 second, third after 2-4 seconds with randomization.

## Caching

Exact match caching for deterministic operations with temperature=0. Cache key: model, system prompt, user message, generation parameters. Semantic caching uses embeddings: similar queries share a common cache. Effective for FAQ scenarios where different phrasings yield the same answer. TTL depends on data nature: static information is cached for hours, dynamic data requires a short TTL. Caching is not suitable for creative generation where variability is a desired property.

## Central Orchestrator

The LLM Orchestration Layer coordinates all calls through a single interface. It manages multiple providers (OpenAI, Anthropic, local models), registering each under an identifier. It implements multi-level caching: computes a hash key, looks up the cache, and registers hits. It applies Circuit Breaker per provider: on persistent errors, the circuit opens. The built-in monitoring system measures latency and tracks errors with detailed breakdowns. The fallback strategy defines an alternative model on exception and logs switchovers.

The cache key is formed via SHA-256 hashing of all parameters: model, message roles and content, temperature, max tokens. This ensures uniform distribution and eliminates collisions.

## Asynchronous Processing

The controller accepts a POST request, generates a UUID, creates a task with PENDING status, saves it to the repository, and places it in the queue. It returns 202 Accepted with a Location header and the task ID. A GET status request returns details: status, progress percentage, current step, result or error, timestamps. Retry-After header: 5 seconds for PENDING, 2 seconds for PROCESSING.

The worker subscribes to the queue, updates the status to PROCESSING, and invokes the agent with callbacks: onProgress updates the percentage and step, onThought and onAction add entries to the log. On completion, it saves the result and updates the status to COMPLETED. On exception — FAILED with a message. The architecture ensures fault tolerance: a worker crash leaves the task in the queue for processing by another worker.

## Streaming via SSE

The controller uses the reactive Flux type for the event stream. Each chunk is wrapped in a ServerSentEvent with an event type. Content-type is set to text/event-stream. Error handling via onErrorResume transforms the exception into an error-type event. The service loads the conversation context, adds the new message, and initiates a streaming request to the LLM. As tokens arrive, it accumulates the full response and increments the counter. On completion, it saves the response to the repository. The client subscribes via the EventSource API and automatically reconnects on disconnection.

## WebSocket for Agents

The handler creates an AgentSession when a connection is established and stores it in a concurrent map. Incoming messages are routed by type: start_task initiates execution, provide_input delivers a response, cancel aborts. The onThought callback sends reasoning to the client, onAction sends an execution indicator, onToolResult sends the tool result. onNeedInput switches the session to a waiting state, saves the request ID, and blocks the agent until a response arrives. provide_input unblocks execution through a resolution mechanism. When the connection closes, the session is removed and the current task is canceled.

## Serverless Deployment

Serverless platforms (AWS Lambda, Google Cloud Functions, Azure Functions) can host LLM applications, but with significant constraints that differ from traditional web workloads.

**Constraints:** Cold start latency (10-30 seconds for loading dependencies, problematic for interactive agents), execution time limits (15 minutes max on Lambda — insufficient for long-running agent loops), memory limits (10GB max — insufficient for local model inference). These constraints make serverless unsuitable for real-time streaming or autonomous agents with multi-step tool use.

**Where serverless works well:** Async batch processing (document classification, data extraction triggered by S3 uploads), event-triggered agents (process Slack message, handle webhook), low-traffic endpoints where maintaining always-on infrastructure is wasteful, scheduled tasks (daily report generation, periodic evaluation runs).

**Where serverless does NOT work:** Real-time streaming (SSE/WebSocket require persistent connections), long-running agent loops (Devin-style sessions can last hours), self-hosted model inference (memory and timeout constraints), high-throughput production APIs (cold starts degrade P99 latency).

**Hybrid approach:** Use serverless for event-triggered and batch workloads, containers (ECS, GKE, AKS) for real-time and long-running agents. The LLM Orchestration Layer abstracts the difference — the same business logic runs in both environments.

---

## Agent-Specific Deployment Considerations

Deploying AI agents differs from deploying standard LLM applications in several critical ways:

**Long-running sessions.** A chatbot request completes in seconds. An agent session can last minutes (code review), hours (Devin working on an issue), or even days (research agents). Infrastructure must support persistent connections and state management across these timescales. WebSocket keepalives, session storage, and timeout configuration all need adjustment.

**State management across tool calls.** Each tool call in an agent loop may modify external state (files written, APIs called, databases updated). If the agent crashes mid-loop, the system must handle partial state: which actions were completed? Which need rollback? Checkpointing after each successful tool call is essential — see [[../03_AI_Agents_Core/02_Agent_Architectures|Agent Architectures]] for loop essentials.

**Graceful termination.** Agents must not be killed mid-action. A SIGTERM during a database write or API call can leave systems in an inconsistent state. On Kubernetes, set `terminationGracePeriodSeconds: 180-300` (3-5 minutes) to give agents time to reach a safe checkpoint. The agent's shutdown handler should: complete the current tool call, save state for resumption, release any locks or resources.

**Resource unpredictability.** A standard LLM app makes one API call per request — resource usage is predictable. An agent may make 1 or 50 LLM calls per task, depending on the complexity of the problem it encounters. This makes capacity planning harder — provision for the P99 case, not the average. Autoscaling should monitor token throughput and active agent sessions, not just CPU.

**Shared memory for inference.** Self-hosted models via vLLM require large shared memory allocations (`shm-size: 16Gi` in Docker/Kubernetes) for the KV-cache. Default container shared memory (64MB) is insufficient and will cause silent failures or crashes.

---

## Key Takeaways

LLM applications require rethinking traditional approaches. High latency, dependence on external APIs, and variable costs affect every decision. Pattern selection depends on operation characteristics: synchronous for short, asynchronous for long, streaming for interactivity, WebSocket for bidirectional communication. The LLM Orchestration Layer centralizes management: retry, fallback, caching, monitoring. Circuit Breaker and fallback strategies protect against cascading failures. Timeouts are differentiated by operation type and coordinated across all levels. Caching reduces cost and latency for repeated queries.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Deployment
**Previous:** [[../12_Observability/04_AgentOps|AgentOps]]
**Next:** [[02_Containerization_and_Kubernetes|Containerization and Kubernetes]]
