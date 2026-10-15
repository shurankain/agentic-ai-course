# LLM Application Deployment Strategies

## Deployment Specifics in the World of Language Models

Deploying LLM applications differs fundamentally from traditional web services. Classic microservices process requests in milliseconds, consume predictable resources, and scale linearly. LLM systems break all of these assumptions: a single request takes seconds or minutes, resource consumption varies depending on context length, and scaling costs are determined by infrastructure and by the price of API calls.

Response time increases from 50 milliseconds to 5 seconds: a hundredfold. Traditional approaches to load balancing, timeouts, and circuit breakers require a complete rethink. Dependence on external APIs (OpenAI, Anthropic, Google) adds a layer of complexity: providers have their own rate limits, experience temporary outages, and change model behavior without notice. The system must ensure graceful degradation instead of catastrophic failures.

## LLM Application Architecture

A typical architecture consists of several layers. The API Gateway provides authentication, rate limiting, and basic routing, the first line of defense. The Application Layer implements business logic: Chat API for conversations, Agent API for autonomous agents, Completion API for generation. Each service is specialized, but all use a shared LLM Orchestration Layer, the heart of the system, where prompt management, retry logic, fallback between providers, caching, and cost monitoring take place. At the lowest level: LLM providers accessed through a unified abstraction.

## Deployment Patterns

**Synchronous pattern** is suitable for short operations under 30 seconds: classification, entity extraction, short responses. The critical requirement is predictable execution time. The fundamental limitation: HTTP connections have timeouts at all levels. Synchronous requests occupy threads throughout the entire processing, limiting concurrency. Each subsequent stack level must have a timeout slightly longer than the previous one.

**Asynchronous pattern with polling** solves the timeout problem. The client sends a request, receives a task ID, then polls the status until completion. The HTTP connection closes immediately, processing continues in the background. Requirements: task storage (database or Redis), a queue (RabbitMQ, Kafka, SQS), workers. The Retry-After header hints at the polling interval for the client: 5-10 seconds for pending, less for processing with known progress.

**Server-Sent Events (SSE)** is critical for interactive applications. The user sees tokens as they are generated. A 500 ms delay to the first token feels better than a 5-second wait for the full response. SSE runs over HTTP, simplifying infrastructure. It requires end-to-end support: provider, application server, load balancer, reverse proxy with buffering disabled. Errors are transmitted as special events.

**WebSocket** provides full-duplex communication. Ideal for interactive agents requiring user input during execution. Client message types: start_task, provide_input, cancel. Server message types: thought, action, tool_result, need_input, complete, error. Requires a sticky sessions mechanism or centralized state storage. A heartbeat mechanism keeps the connection active.

## Fault Tolerance

**Circuit Breaker** prevents cascading failures. When the error threshold is reached (50% over 10 calls), the circuit opens, and subsequent requests immediately return an error. After 30 seconds: a probe call; on success, the circuit closes again.

**Fallback strategies**: switching to an alternative provider (GPT-6, then Claude), degrading to a cheaper tier (GPT-6 Sol, then GPT-6 Luna), cached response for similar queries, honest unavailability message. The choice depends on quality and availability requirements. A multi-provider setup requires a unified interface.

### Multi-Provider Failover Architecture

Production systems should not depend on a single LLM provider. A robust failover architecture uses a **unified abstraction layer**: tools like LiteLLM provide a unified API for 100+ LLM providers with cost routing, budget enforcement, and spend tracking behind a single interface. The application code calls LiteLLM; LiteLLM routes to the configured provider. **Security caveat:** LiteLLM CVE-2026-42271 (added to CISA Known Exploited Vulnerabilities catalog, June 9) demonstrated that proxy libraries inherit and amplify security risk: a vulnerability in the proxy affects every application behind it. Pin proxy library versions, monitor security advisories, and apply patches within hours of disclosure. See Module 14, Agent Security for the full CVE chain and the Mercor breach case study.

**Failover chain:** Primary provider (e.g., Anthropic), then Secondary provider (e.g., OpenAI), then Degraded mode (cached responses or simpler model). Each transition is triggered by circuit breaker thresholds. Provider health is monitored continuously: latency percentiles, error rates, and availability.

**Cost-aware routing** extends failover into an optimization strategy: route simple queries (classification, extraction) to cheap models, complex queries (reasoning, analysis) to expensive ones. This is not just failover. It is intelligent routing that reduces cost by 50-70% during normal operation while providing automatic failover during outages. See Module 03, Resource Optimization for routing patterns.

**Timeouts** are differentiated by endpoint: Completion 30 seconds, Chat 60 seconds, Agent 5 minutes. At the LLM call level, expected generation length is taken into account. Retry logic considers idempotency: completion is safe, agent workflows are riskier due to side effects. Exponential backoff with jitter: second attempt after 1 second, third after 2-4 seconds with randomization.

## Caching

Exact match caching for deterministic operations with temperature=0. Cache key: model, system prompt, user message, generation parameters. Semantic caching uses embeddings: similar queries share a common cache. Effective for FAQ scenarios where different phrasings yield the same answer. TTL depends on data nature: static information is cached for hours, dynamic data requires a short TTL. Caching is not suitable for creative generation where variability is a desired property.

## Releasing a Change

Classic release strategy assumes one artefact: you built a binary, you ship it gradually, you watch the error rate, you roll back if it spikes. An LLM system has four release surfaces and only the first is a binary.

The application code is one. The prompt is a second, and it changes far more often. The model version is a third, and it can change without you doing anything at all when a provider updates an alias. The retrieval index is a fourth, and reindexing changes answers without touching a line of code. A release process that only gates the first surface will let the other three reach production unreviewed, which is how a system that passed every test starts giving different answers on Tuesday.

**Pin the model, never the alias.** `claude-opus-4-5` style aliases resolve to whatever the provider currently considers that model. Pin the dated identifier, record it alongside every stored response, and treat a version bump as a release. Then when quality moves you can ask which of the four surfaces moved, instead of guessing.

**Canary by user, not by request.** Routing 5% of requests to the new configuration gives a clean sample and a broken experience: the same user gets the old prompt and then the new one in the same conversation, and the inconsistency reads as the system being unreliable. Hash the user identifier and route by that, so a user stays on one side for the whole session. The cost is that you need more traffic before the sample means anything.

**Shadow before canary when the metric is quality.** Send a copy of live traffic to the new configuration, discard its output, and compare offline. No user is exposed, which matters because a quality regression cannot be detected from a single response the way an error can. This is the only safe way to evaluate a model upgrade on real traffic, and it costs a second inference on every shadowed request, so sample rather than mirror everything.

**Rollback triggers have to include quality, and quality is slow.** Error rate and latency are visible within minutes, so wire them to automatic rollback in the usual way. A quality regression is silent: the system returns confident, well-formed, wrong answers at the same latency and the same cost. Detecting it needs the continuous evaluation pipeline from Module 11, and it needs hours rather than minutes. Plan the canary window around the slower signal, and keep the previous configuration warm for the full window rather than tearing it down once the error rate looks clean.

**Blue-green is awkward here, and the reason is instructive.** Blue-green assumes the two environments are equivalent apart from your change. When the model runs on the provider's infrastructure, both environments call the same model, so the switch does not isolate the variable you most want to isolate. Blue-green earns its keep for self-hosted inference, where the model weights genuinely live in the environment you are swapping.

## Central Orchestrator

Every provider call goes through one layer, and that layer owns five things the application code should never see: which provider serves this request, whether the answer is already cached, whether this provider's circuit is open, how long the call took, and what to do when it fails. Scattering those five across call sites is how a codebase ends up with three different retry policies and no way to answer "what did we spend on Claude last week".

Build the cache key by hashing every input that can change the output: model identifier, each message with its role, temperature, and max tokens. Omitting temperature is the classic bug, because a cached deterministic answer then gets served to a request that asked for variation. Hash rather than concatenate, since prompts are long and a key that contains the prompt makes the cache store the corpus twice.

Keep the circuit breaker per provider rather than global. One provider degrading should shift traffic to another, and a global breaker turns a single vendor's bad afternoon into a full outage.

## Asynchronous Processing

The contract is worth stating precisely, because clients get written against it. The submit endpoint returns 202 with a task identifier and a `Location` header, never the result. The status endpoint returns the state, a progress percentage, the current step, and either the result or the error. Set `Retry-After` to about 5 seconds while the task is queued and about 2 seconds once it is running, since a client that polls at a fixed 1 second will generate more load than the work itself on a queue that is backed up.

Durability comes from where the task lives, not from the worker. Write the task to storage before acknowledging the request, and have the worker take it from the queue rather than from an in-memory handoff. Then a worker that dies leaves the task claimable by another one, which is the entire reason for choosing this pattern over synchronous processing with a long timeout.

Progress reporting is what makes an asynchronous agent bearable to use. Emit the step and the percentage on every meaningful transition, and record the agent's reasoning and actions to the task log as they happen. A task that sits at PENDING for four minutes and then returns an answer is indistinguishable from a hung one.

## Streaming via SSE

Stream at the token level, tag every chunk with an event type, and set the content type to `text/event-stream`. The part that breaks in production is error handling: an exception thrown mid-stream cannot become an HTTP 500, because the status line was sent minutes ago. Convert it into a terminal error event on the same stream, and make sure the client treats an unexpected stream end as a failure rather than as completion.

Two infrastructure details defeat more SSE deployments than any application bug. Response buffering must be off at every hop, since a reverse proxy that buffers will hold the whole response and deliver it at once, which looks exactly like streaming not being implemented. And the accumulated response has to be persisted server-side on completion, because the client that reconnects mid-stream has no way to recover what it missed.

## WebSocket for Agents

Use WebSocket when the agent needs to ask the user something mid-run, and not otherwise, since SSE covers one-way streaming at a fraction of the operational cost.

The message vocabulary carries the design. Inbound: start a task, supply input, cancel. Outbound: a thought, an action being taken, a tool result, a request for input, completion, and error. Keep them as distinct types rather than a single message with a flag, because the client renders each one differently and a flag invites the renderer to guess.

The request-for-input path is where implementations go wrong. The agent must genuinely suspend at that point, with the pending request keyed by an identifier, and resume only when the matching reply arrives. An implementation that polls for input or times out into a default answer produces an agent that quietly decides on the user's behalf. On disconnect, cancel the running task rather than orphaning it, and decide explicitly whether the session is resumable, because an agent still burning tokens for a browser tab that closed is a cost leak nobody watches.

## Serverless Deployment

Serverless platforms (AWS Lambda, Google Cloud Functions, Azure Functions) can host LLM applications, but with significant constraints that differ from traditional web workloads.

**Constraints:** Cold start latency (10-30 seconds for loading dependencies, problematic for interactive agents), execution time limits (15 minutes max on Lambda, insufficient for long-running agent loops), memory limits (10GB max: insufficient for local model inference). These constraints make serverless unsuitable for real-time streaming or autonomous agents with multi-step tool use.

**Where serverless works well:** Async batch processing (document classification, data extraction triggered by S3 uploads), event-triggered agents (process Slack message, handle webhook), low-traffic endpoints where maintaining always-on infrastructure is wasteful, scheduled tasks (daily report generation, periodic evaluation runs).

**Where serverless does NOT work:** Real-time streaming (SSE/WebSocket require persistent connections), long-running agent loops (Devin-style sessions can last hours), self-hosted model inference (memory and timeout constraints), high-throughput production APIs (cold starts degrade P99 latency).

**Hybrid approach:** Use serverless for event-triggered and batch workloads, containers (ECS, GKE, AKS) for real-time and long-running agents. The LLM Orchestration Layer abstracts the difference: the same business logic runs in both environments.

---

## Agent-Specific Deployment Considerations

Deploying AI agents differs from deploying standard LLM applications in several critical ways:

**Long-running sessions.** A chatbot request completes in seconds. An agent session can last minutes (code review), hours (Devin working on an issue), or even days (research agents). Infrastructure must support persistent connections and state management across these timescales. WebSocket keepalives, session storage, and timeout configuration all need adjustment.

**State management across tool calls.** Each tool call in an agent loop may modify external state (files written, APIs called, databases updated). If the agent crashes mid-loop, the system must handle partial state: which actions were completed? Which need rollback? Checkpointing after each successful tool call is essential, see Module 03, Agent Architectures for loop essentials.

**Graceful termination.** Agents must not be killed mid-action. A SIGTERM during a database write or API call can leave systems in an inconsistent state. On Kubernetes, set `terminationGracePeriodSeconds: 180-300` (3-5 minutes) to give agents time to reach a safe checkpoint. The agent's shutdown handler should: complete the current tool call, save state for resumption, release any locks or resources.

**Resource unpredictability.** A standard LLM app makes one API call per request: resource usage is predictable. An agent may make 1 or 50 LLM calls per task, depending on the complexity of the problem it encounters. This makes capacity planning harder, provision for the P99 case, not the average. Autoscaling should monitor token throughput and active agent sessions, not just CPU.

**Shared memory for inference.** Self-hosted models via vLLM require large shared memory allocations (`shm-size: 16Gi` in Docker/Kubernetes) for the KV-cache. Default container shared memory (64MB) is insufficient and will cause silent failures or crashes.

## Agent State Persistence and Recovery

When an agent crashes mid-task: provider timeout, OOM kill, infrastructure failure: what happens to the work in progress? Without state persistence, the entire task restarts from scratch. For a 30-minute research agent session, that is 30 minutes of lost work and wasted tokens.

**Checkpoint after each tool call.** The minimum viable persistence strategy: after every successful tool call, save the agent's state (conversation history, tool results, plan progress) to durable storage. If the agent crashes, it resumes from the last checkpoint rather than the beginning. LangGraph implements this natively through its checkpoint mechanism (PostgreSQL for production, SQLite for development). For custom agents, the checkpoint is a serialized state object stored in Redis (fast, TTL-based expiry) or PostgreSQL (durable, queryable).

**Database schema for agent sessions.** A production agent needs at minimum: a sessions table (session_id, user_id, agent_type, status, created_at, updated_at), a checkpoints table (checkpoint_id, session_id, state_json, step_number, created_at), and a tool_results table (result_id, session_id, tool_name, input_json, output_json, duration_ms, created_at). The checkpoint state_json contains everything needed to resume: conversation messages, plan state, accumulated results, and current step. Index by session_id and step_number for fast recovery.

**Idempotency for resumable agents.** When an agent resumes from a checkpoint, it must not repeat side effects. If the agent sent an email before crashing, it should not send it again on resume. Implementation: each tool call is tagged with a unique idempotency key (session_id + step_number). Tools check whether an action with this key was already completed before executing. This is the same idempotency pattern used in payment systems: apply it to agent tool calls.

**Session recovery flow:** On agent startup, check for an incomplete session for this user/task. If found, load the latest checkpoint, verify external state consistency (did the email actually send? did the file actually write?), and resume from the next step. If external state is inconsistent, fall back to the last known-good checkpoint.

## Load Testing Methodology for AI Agents

Load testing AI systems differs fundamentally from load testing web APIs. A web API request takes 50ms; an agent session takes 5-60 seconds (or minutes for autonomous agents). Traditional tools configured for millisecond latencies produce misleading results.

**The challenge.** A conventional load test might spawn 1,000 concurrent requests and measure throughput. For agents, 1,000 concurrent sessions means 1,000 simultaneous LLM API calls (hitting rate limits immediately), 1,000 active WebSocket connections (if streaming), and potentially 10,000-50,000 LLM calls in total (each agent session makes 10-50 calls). The load test saturates provider rate limits long before it stresses the application.

**Locust/k6 configuration for agents.** Use long-lived "user" sessions that simulate realistic agent interactions: start a session, send a task, wait for progressive results (5-60 seconds), send a follow-up, wait again. Set think times between interactions to 5-30 seconds (not milliseconds). Ramp up slowly: add 1-2 concurrent users per minute, not 100 per second. Monitor provider rate limit responses (HTTP 429) as the primary saturation signal.

**What to measure.** Concurrent agent session capacity (how many simultaneous sessions before degradation), time-to-first-response under load, session completion rate under load (do agents start failing to complete tasks?), provider rate limit hit rate, memory consumption growth over time (agent state accumulates), and cost per session under load (does cost increase due to retries?).

**Concurrent session limits.** Determine your system's practical limit: start with 10 concurrent sessions, increase by 10, and observe when P95 latency exceeds SLA or session completion rate drops below 95%. This number, not requests-per-second, is the capacity metric for agent systems. Plan production capacity at 60-70% of this limit to handle bursts.

---

## Key Takeaways

1. **Pick the pattern from the operation, not from habit.** Synchronous under 30 seconds, asynchronous with polling for anything longer, SSE when the user should see progress, WebSocket only when the agent has to ask a question mid-run. WebSocket costs sticky sessions or shared state, so do not reach for it to get one-way streaming.

2. **One orchestration layer owns provider choice, caching, breaking, timing, and failure.** Spread those five across call sites and you get three retry policies and no answer to what you spent per provider. Keep the circuit breaker per provider, since a global one turns one vendor's bad afternoon into your outage.

3. **An LLM system has four release surfaces: code, prompt, model version, and retrieval index.** Only the first is a binary, and the third can change without you acting when a provider updates an alias. Pin dated model identifiers, record the version with every stored response, and gate all four.

4. **Quality regressions are silent, so plan the canary window around the slow signal.** Error rate and latency surface in minutes and can drive automatic rollback. A model that returns confident wrong answers at normal latency and normal cost surfaces in hours, through the evaluation pipeline in Module 11. Canary by user rather than by request, and shadow before canary whenever the metric is quality.

5. **Agents break the assumptions the rest of this list rests on.** A session runs for minutes to days, makes 1 or 50 model calls for the same nominal task, mutates external state as it goes, and must not be killed mid-action. That means checkpointing after every tool call, idempotency keys so a resumed agent does not send the email twice, a termination grace period of 3 to 5 minutes, and capacity planned at P99 rather than the mean.

6. **Capacity for agent systems is measured in concurrent sessions, not requests per second.** Ramp a load test by 1 or 2 users per minute, watch for HTTP 429 as the real saturation signal, and plan production at 60% to 70% of the point where completion rate falls below 95%.
