# MCP Client Integration

If MCP servers are bridges to external systems, then clients are orchestrators that coordinate interaction between the user, the language model, and multiple servers. Understanding the client side of MCP is critically important for building fully functional AI applications.

An MCP client performs several key functions. First, it manages connections to servers, establishing, maintaining, and restoring them after failures. Second, it aggregates capabilities of all connected servers and presents them to the language model in an understandable format. Third, it routes model requests to the appropriate servers and returns results.

In the MCP ecosystem, the client occupies a central position. It ties together the user interface, the AI model, and external services, creating a cohesive intelligent application.

## Client Application Architecture

### Layered Structure

A well-designed MCP client has a clear layered architecture. At the bottom level sits the transport layer, responsible for physical data transmission. Above it is the protocol layer, implementing JSON-RPC and MCP specifics. Higher still is the server management layer, coordinating multiple connections. At the top is the AI integration layer, which links MCP capabilities with the language model.

This separation allows each layer to be developed and tested independently. The transport layer can be replaced without changing business logic. The protocol layer abstracts message format details. The management layer provides a unified interface regardless of the number of servers.

### State Management

The client must track the state of each server connection. A connection can be in the "connecting", "ready", "reconnecting", or "disconnected" state. Clients written for specifications before 2026-07-28 also had an "initializing" state for the handshake, which no longer exists. These states determine which operations are available.

Beyond connection state, the client caches information about server capabilities. Lists of resources, tools, and prompts are requested when the connection opens and updated upon receiving change notifications.

Subscription state also requires tracking. The client remembers which resources it is subscribed to and processes incoming notifications.

### Asynchrony and Concurrency

The nature of MCP implies asynchronous interaction. Requests to servers can take a long time to execute, and the client must not block while waiting for a response.

Modern client implementations use non-blocking I/O and reactive patterns. Requests are sent asynchronously, and responses are processed as they arrive. This enables efficient management of many parallel operations.

Concurrency requires attention to synchronization. Updating shared state (the capability cache, the list of active requests) must be thread-safe.

## Connecting to Servers

### Server Launch Methods

An MCP client can connect to servers in two ways, over stdio or Streamable HTTP. The most common is launching a server as a child process with data exchange via stdio. The client specifies the launch command and arguments, then establishes the connection.

For remote servers, **Streamable HTTP** is the standard network transport (the earlier SSE transport was deprecated in March 2025). The client connects to the specified URL and begins message exchange.

Some implementations support "embedded" servers implemented within the client process. This is useful for simple integrations that do not require a separate process.

### Connection Procedure

There is no initialization step. The 2026-07-28 specification removed the `initialize` request and the `initialized` notification, so a client opens a transport and issues real requests immediately.

Each request carries what used to be negotiated: the protocol version and the client's capabilities in `_meta` (`io.modelcontextprotocol/protocolVersion`, `io.modelcontextprotocol/clientCapabilities`), plus client identity that the client should send on every request. Results come back with the server's identity in their own `_meta`. A version the server cannot serve returns `UnsupportedProtocolVersionError` for that request rather than tearing down a connection.

A client that wants to know what it is talking to before committing may call `server/discover`, which every server must implement. It returns supported protocol versions, capabilities, and identity. On stdio it doubles as a backward-compatibility probe against older servers.

### Capability Discovery

Once connected, the client queries the server for available components. The `resources/list` method returns the list of resources. The `tools/list` method returns the list of tools. The `prompts/list` method returns the list of prompts.

This information is cached by the client and provided to the language model. The model sees all available tools and can select the appropriate ones for solving the task.

Caching is no longer left to the client's judgement. List results now carry two required fields: `ttlMs`, a freshness hint in milliseconds, and `cacheScope`, either `"public"` or `"private"`, which says whether a shared intermediary may cache the response. Servers should also return tools in a deterministic order, because a stable ordering keeps the client's cache valid and improves prompt-cache hit rates on the model side.

For change notifications the client opens `subscriptions/listen`, a single long-lived POST-response stream, and opts into the types it cares about (`toolsListChanged`, `promptsListChanged`, `resourcesListChanged`, `resourceSubscriptions`). The older per-resource `resources/subscribe` and `resources/unsubscribe` methods and the HTTP GET endpoint were removed.

### Handling Disconnections

Network connections are unreliable. A server can crash, the network can drop, a process can hang. The client must gracefully handle such situations.

The reconnection strategy typically includes exponential backoff. After the first failure, the client waits several seconds, then tries to reconnect. If the attempt fails, the interval increases. This prevents overload during system issues.

Reconnection got simpler and more dangerous at the same time. Simpler because there is no initialization to redo: the client just starts sending requests again, and any instance can serve them. Dangerous because stream resumability was removed. `Last-Event-ID` and SSE event IDs are gone, so a broken response stream loses the in-flight request and the client **must** re-issue it as a new request with a new request ID.

That re-issue is a real retry against a server that may already have done the work. Before a client retries anything with a side effect, it needs an idempotency key the server honours, or an explicit check of whether the operation completed. A client that blindly retries a payment or a merge on connection loss will duplicate it. Subscriptions also need re-establishing through `subscriptions/listen`, since the stream carried them.

## Multi-Server Aggregation

### Theory of Multi-Server Aggregation

The problem of aggregating tools from multiple servers is a classic distributed systems integration challenge. Key theoretical aspects include:

**Tool namespace.** Tools from different servers form a shared namespace. A fundamental question arises: how to ensure identifier uniqueness and predictability?

There are four main naming strategies:

**Flat namespace** uses simple names without prefixes, such as read_file, query_db, send_email. This approach is maximally simple and clear but creates a high risk of conflicts when different servers declare tools with identical names.

**Prefix naming with server identifier** adds a unique prefix to each tool. For example, filesystem:read_file, database:query_db. This guarantees name uniqueness since each server has its own namespace. The prefix explicitly indicates the tool's source, which simplifies debugging and system comprehension.

**Capability-based naming** groups tools by semantic categories using a hierarchical structure: file.read, db.query, email.send. This approach provides logical tool organization regardless of which server provides them. It is useful when multiple servers can offer similar capabilities.

**URI-style** uses full URIs for tool identification, for example mcp://filesystem/tools/read_file. This ensures global uniqueness and extensibility. This allows inclusion of versioning, hosts, and other metadata. This approach is most versatile for large distributed systems.

### Multi-Server Architecture

Real-world AI applications often use multiple MCP servers simultaneously. A file system server, a database server, an external API server: each provides its own capabilities.

The client aggregates all these capabilities into a unified interface. The language model sees a combined tool list without needing to know which server is responsible for what.

When a tool is invoked, the client routes the request to the appropriate server. Routing can be based on name prefixes, explicit registration, or metadata.

### Conflict Resolution and Prioritization

During aggregation, name conflicts are possible. Two servers may declare tools with identical names but different semantics. This requires a formal resolution approach.

**Conflict resolution algorithms:**

There are five main strategies for resolving tool name conflicts:

The **First-wins** strategy means the first registered tool wins, and subsequent tools with the same name are ignored. This is a simple and predictable approach, suitable for basic scenarios where the server loading order is stable.

The **Last-wins** strategy allows the last registered tool to override previous ones. This is useful for hot-reloading configurations or updating servers on the fly, when a new version should replace the old one.

The **Priority-based** approach uses explicitly assigned server priorities. Each server is given a numeric priority, and during a conflict, the tool from the server with the higher priority is selected. This provides full control in enterprise or managed environments.

The **Merge** strategy combines all conflicting tools by renaming them with unique identifiers appended. This ensures maximum availability of all capabilities but complicates naming.

The **Fail** strategy strictly prohibits conflicts, raising an error upon detection. This is the safest approach for critical systems where ambiguity is unacceptable.

**Tool prioritization.** Beyond conflict resolution, prioritization is important for optimizing tool selection by the model.

Each tool's priority is computed as a function of several factors. Relevance is determined by semantic similarity between the user's request and the tool description. Quality is assessed through execution success rate and user satisfaction. Cost includes token consumption and computational expenses. Latency reflects the average response time.

For example, when comparing three data-reading tools, filesystem:read may have high relevance of 0.9 and quality of 0.95 but average speed of 0.8, yielding an overall priority of 0.88. An S3 tool may be slightly less relevant (0.85) with good quality (0.9) but low speed (0.6), resulting in 0.78. An HTTP tool may be less relevant (0.7) but very fast (0.9), achieving 0.80.

A common practical solution is using namespaces. Tool names are augmented with the server identifier: filesystem:read_file, database:read_file. This eliminates ambiguity at the naming level.

An alternative approach is dynamic prioritization based on context. The system analyzes the current task, the history of successful invocations of specific tools, and their current latency, selecting the optimal option for each situation.

### Fault Tolerance

A single server failure must not block the entire system. The client must continue functioning with the available servers.

This requires isolating the state of each connection. An error communicating with one server must not affect other connections.

When a server is unavailable, its tools are temporarily excluded from the list. The user is informed about the degraded functionality.

## Integration with Language Models

### Tool Representation

Language models with function calling support expect descriptions of available functions in a specific format. The client transforms MCP tool descriptions into this format.

The description includes the function name, its purpose, and the parameter schema. The quality of this description critically affects the model's ability to use tools correctly.

A good practice is to supplement descriptions with usage examples. The model better understands a tool's semantics when it sees concrete use cases.

### Call Processing

When the model decides to call a tool, it generates a structured request with the tool name and arguments. The client intercepts this request and routes it to the appropriate MCP server.

The call result is returned to the model in the next dialog message. The model analyzes the result and decides whether additional actions are needed.

This cycle (call generation, execution, result return) can repeat multiple times within a single user request.

### Handling servers that need something back

A server can no longer call into a client. Sampling is deprecated as of 2026-07-28, and the pattern it served is now MRTR: the server returns an `InputRequiredResult` with `resultType: "input_required"` whose `inputRequests` field lists what it needs, and the client retries the original request with `inputResponses` attached.

For a client this changes the shape of the work. Instead of listening for inbound server requests, the client checks `resultType` on every result and drives a retry loop when it sees `"input_required"`. A result from an older server that omits the field must be treated as `"complete"`.

The control point survives the redesign. A client still decides what reaches the model or the user, can filter or modify what a server asks for, and can refuse. It just enforces that policy while preparing a retry rather than while answering a callback.

### Context Limitations

Language models have context size limitations. With a large number of tools, their descriptions can occupy a significant portion of the context window. A typical tool definition consumes 200-500 tokens (name, description, JSON Schema for parameters). With 100 connected tools, this alone can consume 20,000-50,000 tokens, a significant portion of even a 200K context window, leaving less room for conversation history and resource content.

**Dynamic tool filtering:** The client selects a subset of relevant tools based on the current conversation. Approaches include: keyword matching (the user mentions "database" becomes include SQL tools), semantic similarity (embed the user message and tool descriptions, select top-K matches), or LLM-based routing (a fast, cheap model classifies the query and selects a tool category).

**Hierarchical representation:** Instead of exposing hundreds of tools directly, the client presents a meta-tool like `select_toolset(category)` that returns the tools for that category. The model first selects "database tools" or "file tools," then sees only the 5-10 tools in that category. This reduces initial context usage to a few hundred tokens regardless of total tool count.

**Description truncation:** For tools with verbose descriptions, the client can truncate to essential information (name, one-line summary, required parameters) during initial presentation, expanding the full description only when the model selects a tool.

**Tool count budgeting:** A practical guideline: keep active tool definitions under 10% of the total context window. For a 128K context model, this means ~12,800 tokens for tools, accommodating roughly 25-60 tools at full detail or 100+ tools with truncated descriptions.

## User Interface

### Action Transparency

The user must understand what is happening in the system. When the AI calls a tool, it should be visible. Call results should be displayed explicitly.

Such transparency is important for trust. The user knows what actions the AI is performing and can intervene if necessary.

Transparency helps diagnose problems. If the result does not meet expectations, the call chain can be traced to find the cause.

### Confirmation of Dangerous Operations

Destructive operations (data deletion, message sending, financial transactions) require user confirmation.

The client can use tool annotations to identify dangerous operations. A tool with the `destructive` flag requests confirmation before execution.

The confirmation level can be configurable. In trusted automation scenarios, confirmations can be disabled; in interactive ones, they can be enabled.

### Error Handling

Errors must be presented to the user in an understandable way. A technical message like "Connection reset by peer" is less informative than "Failed to connect to the database server".

The client can transform technical errors into user-friendly messages. The original information is preserved for diagnostics.

Some errors allow retrying the operation. The client can offer the user a retry or automatically repeat the request.

## Production MCP Client Patterns

Building an MCP client that works in development is straightforward. Building one that works reliably in production requires handling failure modes that rarely appear during testing.

**Connection pooling.** A production client typically connects to 5-20 MCP servers simultaneously. Each connection consumes resources: a child process for stdio servers, a persistent HTTP connection for remote servers. Connection pooling manages this: maintain a pool of initialized connections, reuse them across requests, and limit the total number of concurrent connections. When a request needs a tool from a server with no available connection, queue the request rather than spawning unlimited connections.

**Server health monitoring.** Not all connected servers are equally healthy. A server may be technically connected but responding slowly, returning errors, or timing out. Health monitoring tracks: response latency (P50, P95), error rate over a sliding window, and last successful response time. When a server's health degrades beyond a threshold: mark it unhealthy, stop routing requests to it, and attempt recovery in the background. This prevents a single degraded server from dragging down the entire system.

**Graceful degradation when servers are unavailable.** When a server goes down, the client must continue operating with reduced functionality, not crash entirely. The pattern: maintain a registry of which tools come from which servers. When a server is unavailable, remove its tools from the model's tool list for subsequent requests. If the model attempts to call an unavailable tool, return a clear error message ("The GitHub integration is temporarily unavailable") rather than a cryptic failure. Optionally, suggest alternative tools that provide similar functionality.

**Retry logic for tool calls.** Tool calls can fail transiently (network timeout, rate limit, server restart). Retry with exponential backoff: first retry after 1 second, second after 2 seconds, third after 4 seconds, then give up. Idempotency matters: retrying a "read" operation is safe, but retrying a "send email" operation may cause duplicates. The client should check tool metadata for idempotency guarantees before retrying write operations.

## Client-Side Security

MCP security is a shared responsibility between client and server. The client's security posture determines whether the agent is protected from malicious or compromised servers.

### Where policy can see an MCP call

Three parties see different parts of an MCP call, and no single one of them can enforce policy alone. The network sees the route and the metadata: since 2026-07-28 the required `Mcp-Method` and `Mcp-Name` headers let a proxy or firewall classify a call without opening the body, and Cloudflare ships detection of MCP traffic by protocol headers on TLS-inspected connections. The client knows the user's intent and which tool the model chose. The server knows what the arguments mean and what side effect they will cause.

A policy built at only one of those layers has a blind spot. The same tooling also flags **Shadow MCP**, meaning servers reachable outside the managed portal, which is the MCP form of a problem every platform team already knows: the inventory you govern is not the inventory that exists.

### Server Trust Levels

Not all servers deserve equal trust. The client should categorize servers into trust tiers: **first-party servers** (built by your team, full trust, unrestricted capabilities), **verified third-party servers** (from MCP registries with review processes, medium trust, restrict sensitive operations), **unverified third-party servers** (unknown provenance, low trust, strict sandboxing required). Tool descriptions from untrusted servers are a prompt injection vector: a malicious server can embed instructions in tool descriptions that redirect the agent's behavior. See Module 14, Agent Security for the CurXecute CVE and Tool Poisoning attack taxonomy.

### OAuth 2.1 Integration

MCP's authorization framework (added to the spec in 2025) uses OAuth 2.1 with PKCE for secure credential delegation. The client obtains scoped access tokens on behalf of the user, with limited permissions and expiration. Production implementation: use short-lived tokens (minutes, not hours), request minimum scopes for each tool call, implement token rotation for long-running agent sessions, and log all token grants for audit. Protected Resource Metadata (PRM) allows servers to advertise their authorization requirements: the client discovers what credentials are needed before attempting access.

### STDIO Vulnerability Awareness

The STDIO transport (MCP servers running as child processes communicating via stdin/stdout) has a critical architectural vulnerability (CVSS 9.8, May 2026): it enables arbitrary OS command execution by design. Approximately 200,000 servers are affected. For production deployments with untrusted servers, prefer Streamable HTTP transport over STDIO, run STDIO servers in sandboxed environments (containers, MicroVMs), and audit server code before granting STDIO access.

### Audit and Logging

All interactions with servers should be logged: who called which tool, when, with what parameters, and with what result. These logs serve security (detecting suspicious tool usage patterns), debugging (tracing failures across the client-server boundary), compliance (demonstrating what actions the agent took and why), and cost attribution (tracking which servers consume the most resources). Sensitive data in logs must be masked, see Module 12, Tracing and Logging for PII masking patterns.

## Key Takeaways

MCP client integration is a broad task requiring attention to many aspects. The architecture must be modular and extensible. This allows the addition of new servers and transports.

Connection management lost its initialization step in the 2026-07-28 specification and gained a harder recovery problem. Without stream resumability, a dropped connection means re-issuing the request rather than resuming it, so a client that retries a write without an idempotency key will duplicate it.

Multi-server aggregation creates a unified interface for the AI model. Conflict resolution and request routing occur transparently to the upper layers.

Integration with language models requires proper tool representation and efficient call processing. The model's context limitations must be considered during design.

The user interface provides transparency and control. The user understands what the AI is doing and can influence the process.

Security is a cross-cutting concern of the client implementation. From server validation to operation audit: every aspect requires attention.
