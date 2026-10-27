# Model Context Protocol (MCP) Basics

## A New Standard for AI Interaction

Every application, service, and database uses its own API. Integrating an AI assistant with a calendar requires one integration code, email requires another, and the file system requires a third. Developers create unique adapters for each tool.

Model Context Protocol (MCP) is an open standard: originally created by Anthropic (November 2024) and now governed by the Agentic AI Foundation (AAIF) under the Linux Foundation, that defines a universal way for AI models to interact with external systems. Instead of NxM integrations (N applications x M services), only N + M implementations are needed (N clients + M servers).

MCP has rapidly become the cross-industry standard for AI tool integration. As of mid-June 2026, the ecosystem includes **97M+ monthly SDK downloads**, **9,400+ verified servers** (~9,652 in the official registry by late May), and native support from all major AI platforms (Claude, ChatGPT, Gemini, Copilot, Cursor, VS Code, and many others). Enterprise adoption: **28% of Fortune 500** companies have deployed dedicated MCP servers for production AI workflows. For context: 80% of F500 use AI agents in production generally: MCP is the dominant but not the only integration mechanism. **41% of surveyed software organizations** are in limited or broad MCP production (Stacklok 2026 report). ~1,000 new servers appear monthly. The official MCP Registry, now governed by the Linux Foundation's Agentic AI Foundation (AAIF, co-founded by Anthropic, Block, and OpenAI), grew from roughly 2,000 servers at its September 2025 launch to about 9,652 by late May 2026, with namespace verification via GitHub OAuth/OIDC and DNS/HTTP domain verification to prevent impersonation. MCP has transitioned from experiment to enterprise infrastructure.

**SDK version warning (as of August 2026):** the 2026-07-28 specification landed with support in all four Tier 1 SDKs on publication day, and the Python SDK v2 line carries the breaking changes: FastMCP renamed to `MCPServer`, no handshake, no session pinning. Migrating is a real project, so **pin your dependency** and move deliberately. The TypeScript SDK v1.29.0 dropped Zod from peerDependencies in favor of Standard Schema support (Zod v4, Valibot, ArkType) and added a Fastify middleware adapter.

This chapter covers MCP architecture, the problems it solves, and its impact on AI application development.

## The Problem MCP Solves

To appreciate the significance of MCP, it is important to understand the problem that existed before it.

### Integration Chaos

A modern AI assistant must be capable of many things: searching the internet, working with files, querying databases, managing calendars, sending emails, and interacting with APIs of services. Each of these capabilities requires an integration.

The traditional approach: function calling or tool use: only partially solves this problem. Yes, the model can call functions. But each function is implemented individually. A Slack integration looks different from a Discord integration. Working with PostgreSQL differs from working with MongoDB. Every new tool requires new code.

This creates scaling problems. If you have ten AI applications and fifty tools, you potentially need five hundred integrations. Each one must be written, tested, documented, and maintained.

### Duplicated Effort

Even worse, different teams solve the same problems in parallel. One team writes a Claude integration with GitHub. Another team writes the same integration for GPT. A third does it for Gemini. The code is nearly identical, but it gets written three times because there is no common standard.

The open-source community suffers from fragmentation. Dozens of libraries do roughly the same thing but in slightly different ways. Developers must choose between them, learn the specifics of each, and switch when changing tools.

### Security and Control

Without a standard, it is difficult to ensure uniform security. Each integration implements authentication and authorization differently. Auditing becomes a nightmare. Each adapter must be reviewed separately. A vulnerability in one integration does not imply a vulnerability in others, which seems positive, but it also means that fixing one vulnerability does not fix similar ones elsewhere.

## What MCP Is

Model Context Protocol is an open standard that defines how AI applications (clients) interact with external data sources and tools (servers).

### The USB Analogy

A useful analogy is USB. Before USB, every device required its own unique connector and driver. A printer used one cable, a scanner another, a camera a third. USB unified the physical interface and the communication protocol. Now any USB device works with any USB port.

MCP does the same for AI. Any MCP client can work with any MCP server. By writing a GitHub integration once as an MCP server, you automatically gain compatibility with all MCP clients: Claude Desktop, VS Code with AI extensions, and any custom applications that support the protocol.

### Relationship with Language Server Protocol (LSP)

It is interesting to compare MCP with another successful standardization protocol: LSP (Language Server Protocol) from Microsoft. Both systems solve a similar MxN integration problem by introducing a middleware layer that decouples clients from providers. The architectural parallels are striking, though the domains differ: LSP targets language tooling while MCP targets AI-service integration. The following table highlights the key structural similarities and differences.

| Aspect | LSP | MCP |
|--------|-----|-----|
| **Problem** | Each IDE x each language | Each AI client x each service |
| **Solution** | Language Server as middleware | MCP Server as middleware |
| **Transport** | JSON-RPC over stdio/TCP | JSON-RPC over stdio/Streamable HTTP |
| **Initialization** | Capability negotiation | Capability negotiation |
| **Dynamics** | Workspace events, file changes | Resource subscriptions |

**Principles borrowed from LSP:**
- Client/server separation with clearly defined roles
- Explicit capability negotiation
- Support for multiple transports
- Notifications for push updates

**Differences between MCP and LSP:**
- MCP is oriented toward AI interaction (tools, prompts, sampling)
- LSP focuses on language semantics (completions, diagnostics, symbols)
- MCP supports Sampling, callbacks to the model
- LSP is more mature (since 2016); MCP launched in 2024 and has rapidly matured with AAIF governance and broad industry adoption

### Protocol Design Principles

MCP developers followed several fundamental principles that shaped the protocol's architecture:

**Loose Coupling Principle.** The client and server know about each other only through the abstract protocol interface. The server does not know which model runs on the client side. The client does not know the server's implementation details. This allows either side to be replaced without affecting the other.

**Explicit Declaration Principle.** All capabilities are declared explicitly through capabilities, tools, resources, and prompts. There are no hidden functions or implicit agreements. This ensures predictability and simplifies debugging.

**Safe Failure Principle.** The protocol is designed so that failures are explicit and recoverable. If a client does not understand a capability, it ignores it. If a server does not support a method, it returns an error. The system degrades gracefully.

**Extensibility Principle.** The protocol supports experimental capabilities and custom extensions. New features can be added without breaking compatibility with existing implementations.

**Symmetry Principle Where Appropriate.** Although MCP follows the client-server model, some aspects are symmetric: both sides can send notifications. Sampling used to reverse the roles outright, letting the server initiate a call back into the client. The 2026-07-28 specification deprecated Sampling and replaced server-initiated requests with MRTR, where the server asks for what it needs in its result and the client retries. The symmetry is now in the message shapes rather than in who may call whom.

These principles reflect lessons learned from other protocols: LSP, gRPC, GraphQL. MCP does not reinvent the wheel but applies proven patterns to a new challenge, integrating AI with the external world.

### Core Concepts

MCP is built around several key concepts.

**Client**: the AI application that needs access to external data and functions. Typical clients include chat interfaces (Claude Desktop), IDEs with AI capabilities (Cursor), and autonomous agents.

**Server**: the service that provides data and functionality. A server can expose access to the file system, a database, an API, or any other resource. A single server typically focuses on one domain: there are servers for GitHub, for Slack, for PostgreSQL.

**Host**: the application in which the client runs. The host manages the client's lifecycle and can define security policies. For example, Claude Desktop is the host for an MCP client.

**Transport**: the mechanism for transmitting messages between client and server. MCP supports several transports: stdio (standard input/output, for local servers) and Streamable HTTP (for remote servers). The earlier SSE-based HTTP transport was deprecated in March 2025 in favor of Streamable HTTP. The reason: SSE is unidirectional (server, then client only), requiring a separate HTTP endpoint for client, then server messages. Streamable HTTP provides true bidirectional communication over a single HTTP connection, simplifying deployment. It originally carried server-initiated requests such as Sampling and Elicitation; since the 2026-07-28 specification those travel as MRTR results on ordinary responses.

**STDIO transport security warning (May 2026):** A critical vulnerability (CVSS 9.8) was discovered in the STDIO transport mechanism. This enables arbitrary OS command execution. The vulnerability affects all MCP SDK implementations and puts an estimated 200,000+ servers at risk. This is a fundamental design issue in the transport layer, not an implementation bug in a specific SDK. A separate NGINX integration flaw (also CVSS 9.8) was discovered around the same time. These are the first serious architectural vulnerabilities in a protocol that had already become an industry standard: a classic example of adoption outpacing security review. Production deployments using STDIO transport should implement defense-in-depth: sandboxing, network policies, and input validation at every boundary. See Module 14, Agent Security for detailed MCP security analysis.

## MCP Architecture

MCP architecture is designed with flexibility and extensibility in mind. It follows the separation of concerns principle: each component does one thing well.

### Three-Tier Model

At the top level are AI applications: chatbots, agents, IDE assistants. They formulate natural language requests and interpret results.

The middle level consists of MCP clients. They translate high-level requests into specific protocol calls, manage connections to servers, and handle security and caching.

The bottom level consists of MCP servers. They interact directly with external systems: reading files, executing SQL queries, and calling APIs.

### Message Protocol

MCP uses JSON-RPC 2.0 for message exchange. This is a mature, well-understood protocol with clear semantics for requests, responses, and notifications.

Each message has a standard structure. A request contains a method (what to do), parameters (with what data), and an identifier (for matching with the response). A response contains either a result or an error. Notifications are one-way messages that do not require a response.

This structure ensures reliability: it is always clear what was requested and what was received in response. Errors are handled explicitly rather than silently ignored.

### Capability Negotiation

Until the 2026-07-28 specification, connecting meant a three-phase handshake: the client sent `initialize` with its protocol version and capabilities, the server answered with its own, and the client confirmed with an `initialized` notification. That handshake is gone.

Capabilities now travel inline. Every request carries the protocol version and the client's capabilities in its `_meta` field, and every result carries the server's identity in the same place. A server must also implement `server/discover`, which a client may call once for up-front version selection, though it is not required to.

Negotiation still works on the intersection principle: a feature is active only if both sides support it, and unknown capabilities are ignored so that older clients and newer servers keep working. What changed is where the agreement lives. It used to be connection state held by both parties. It is now a property of each individual message, which is precisely what allows any request to reach any server instance.

## Benefits of MCP

Adopting a standardized protocol brings benefits.

### Universality and Reusability

An MCP server, once written, works with all MCP clients. A Jira integration implemented as an MCP server is automatically available in Claude, in Cursor, and in any other MCP-compatible application. There is no need to write three different plugins for three different tools.

This radically reduces the total cost for the ecosystem. Instead of NxM integrations (N applications x M services), only N + M implementations are needed (N clients + M servers). At large scale, the savings are enormous.

### Standardized Security

MCP defines standard authentication and authorization mechanisms. All servers follow the same patterns, simplifying auditing and certification. A vulnerability found in the protocol implementation is fixed centrally, automatically protecting all users.

Hosts can apply uniform policies to all connected servers. Prohibiting writes to the file system is a single setting that applies to all servers with file access.

### Ecosystem and Community

The standard creates a foundation for an ecosystem. Catalogs of MCP servers emerge, along with tools for developing and testing them and ready-made solutions for common tasks. Developers can focus on unique business logic while using ready-made components for standard functions.

The openness of the standard encourages community contributions. Anyone can create an MCP server for a needed service and share it with the world.

## MCP in the Context of Agentic Systems

MCP is particularly important for AI agents. Agents by definition must interact with the external world, reading data, executing actions, and verifying results. MCP provides a standard way to do so.

### Data Access

An agent often needs information that is not in its training data: current news, the contents of specific documents, database states. MCP server resources provide this data in a unified format.

The agent queries the MCP client for a list of available resources, selects the ones it needs, and reads their contents. All through a standard protocol, regardless of where the data physically originates: a file, a database, or an external API.

### Action Execution

An agent must be able to read and act: creating files, sending messages, placing orders. MCP server tools provide these capabilities.

Each tool is described by a schema: what parameters it accepts, what it returns, and what errors are possible. The agent can analyze available tools, select the appropriate ones for the task, and call them with the correct parameters.

### Coordinating Multiple Sources

A real-world task often requires data from multiple sources and actions across multiple systems. You might need to read a task from Jira, find related code in GitHub, analyze logs in Elasticsearch, and create a pull request.

MCP allows connecting multiple servers simultaneously. The agent sees a combined set of resources and tools from all servers and can combine them to solve complex tasks.

## MCP Apps (January 2026)

An evolution beyond data and actions: MCP tools can now return **interactive UI components**: forms, charts, filtered tables, maps, approval buttons: rendered directly in the conversation. When a tool returns an MCP App, the host application (Claude, an IDE, a custom agent UI) renders the component inline rather than displaying raw data.

This blurs the line between a tool call and an application. An MCP server for a database is no longer limited to returning query results as text. It can return an interactive table with sorting, filtering, and pagination. A monitoring MCP server can return a live chart. A workflow MCP server can return an approval form with Accept/Reject buttons.

MCP Apps represent the evolution of MCP from a data protocol to an **application platform**. The implications for agent UX are significant: agents can provide rich, interactive responses rather than text-only outputs, without requiring custom frontend development for each integration.

## MCP Server Cards and Discovery

As the MCP ecosystem grew to 9,400+ verified servers (mid-June 2026), discovering the right server for a task became a challenge. **MCP Server Cards** address this by providing a standard for exposing structured server metadata via a `.well-known` URL. A browser, crawler, or registry can discover a server's capabilities: what tools it provides, what resources it exposes, what authentication it requires, without connecting to it or starting a session.

This is analogous to OpenAPI specification files for REST APIs: a machine-readable description of capabilities that enables automated discovery and cataloging. The **MCP Registry** (launched in the second half of 2025) is the single source of truth for available MCP servers, supporting both public servers and private sub-registries that organizations can customize for their specific needs.

## MCP Tasks: Long-Running Operations

**MCP Tasks** provide support for long-running asynchronous operations, needed wherever a tool call takes minutes or hours rather than milliseconds. As of the 2026-07-28 specification, Tasks live in the official `io.modelcontextprotocol/tasks` extension rather than in the core protocol, so a simple synchronous server stays lightweight and does not implement them at all.

A Task represents an operation with a lifecycle: created, then running, then completed (or failed or cancelled). The client polls with `tasks/get` rather than blocking on a `tasks/result` call, sends input back through `tasks/update`, and cancels with `tasks/cancel`. Servers may hand back a task handle unsolicited, without the client opting in per request. There is no `tasks/list`: enumeration was removed, so a client tracks the handles it holds.

**Why this matters for agents:** Without Tasks, an agent calling a long-running tool (generating a report, running a CI pipeline, processing a large document set) must either block (wasting context window time) or implement custom polling logic. Tasks make this a protocol-level concern, the agent issues a tool call, receives a task ID, and can continue other work while monitoring progress asynchronously.

## MCP 2026-07-28: the stateless specification

The largest revision since MCP launched was published as final on **July 28, 2026**, replacing 2025-11-25 after a ten-week release-candidate period. All four Tier 1 SDKs shipped support on publication day. If you run an MCP server in production, this section is a list of what breaks.

**Stateless core (SEP-2575).** The `initialize` and `notifications/initialized` handshake is gone. Every request now carries its own protocol version and client capabilities in `_meta`, under `io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities`. Clients should identify themselves on each request, servers should identify themselves in each result's `_meta`, and a version mismatch returns `UnsupportedProtocolVersionError`.

**Sessions removed (SEP-2567).** The `Mcp-Session-Id` header is retired and list endpoints no longer vary per connection. A server that needs state across calls now mints an explicit handle and passes it back as an ordinary tool argument. That is the migration path, and it is where most existing servers break: state that used to live implicitly in a session must become an explicit value the client carries.

**`server/discover`.** Servers **must** implement this RPC, advertising supported protocol versions, capabilities, and identity. Clients **may** call it before anything else for version selection, or use it as a backward-compatibility probe on stdio. Mandatory server-side, optional client-side.

**Subscriptions redesigned.** The HTTP GET endpoint and the `resources/subscribe` and `resources/unsubscribe` methods are replaced by `subscriptions/listen`, a single long-lived POST-response stream. Clients opt into specific notification types (`toolsListChanged`, `promptsListChanged`, `resourcesListChanged`, `resourceSubscriptions`) and the server tags notifications with `io.modelcontextprotocol/subscriptionId`. Request-scoped notifications such as `notifications/progress` still flow on the response stream of the request they belong to.

**Removed outright:** `ping`, `logging/setLevel`, and `notifications/roots/list_changed`. Log level is now set per request through `io.modelcontextprotocol/logLevel` in `_meta`, and a server must not emit `notifications/message` for a request that omitted that field.

**Tasks moved to an extension (SEP-2663).** Tasks left the core protocol and became the official `io.modelcontextprotocol/tasks` extension. The blocking `tasks/result` method is replaced by polling through `tasks/get`, `tasks/update` was added for client-to-server input, `tasks/list` was removed, and servers may now return task handles unsolicited rather than requiring per-request opt-in.

**MRTR replaces server-initiated requests (SEP-2322).** Multi Round-Trip Requests remove the need for a server to call back into the client. Instead of `roots/list`, `sampling/createMessage`, or `elicitation/create`, the server returns an `InputRequiredResult` with `resultType: "input_required"` whose `inputRequests` field lists what it needs. The client retries the original request with `inputResponses` attached. Every result now carries a required `resultType`, either `"complete"` or `"input_required"`, and clients must treat a missing field from an older server as `"complete"`.

**SSE resumability removed.** `Last-Event-ID` and SSE event IDs are gone from Streamable HTTP. A broken response stream loses the in-flight request, and the client must re-issue it as a new request with a new request ID.

That last change has a consequence worth stating plainly, because the specification will not state it for you. **Retry safety is now entirely your application's problem.** A re-issued `create_invoice`, payment, or merge will execute twice unless you prevent it. Statelessness at the protocol layer removes session bookkeeping; it does not remove the need for idempotency keys, deduplication, and transaction boundaries above it. A protocol that makes retries easy makes idempotency mandatory.

**Caching becomes mandatory (SEP-2549).** Results from `tools/list`, `prompts/list`, `resources/list`, `resources/read`, and `resources/templates/list` must carry `ttlMs` (a freshness hint in milliseconds) and `cacheScope` (`"public"` or `"private"`, controlling whether shared intermediaries may cache the response).

**Routing headers (SEP-2243).** `Mcp-Method` and `Mcp-Name` are now required on Streamable HTTP POSTs, so a proxy, firewall, or rate limiter can classify a call without parsing the request body. Combined with the stateless core, this is what lets MCP sit behind a plain round-robin load balancer.

**OpenTelemetry trace context (SEP-414).** Conventions for `traceparent`, `tracestate`, and `baggage` in `_meta` are now documented. This is the first standardized way to propagate trace context across an MCP boundary, and Module 12 covers what to do with it.

**Deprecated with a twelve-month minimum window (SEP-2577, SEP-2596).** Roots, Sampling, and Logging are deprecated with documented migrations: pass directories through tool parameters, resource URIs, or server configuration instead of Roots; integrate directly with a provider API instead of Sampling; log to `stderr` or use OpenTelemetry instead of Logging. Also deprecated: the HTTP+SSE transport, the `includeContext` values `"thisServer"` and `"allServers"`, and OAuth 2.0 Dynamic Client Registration in favour of Client ID Metadata Documents.

**A formal feature lifecycle.** The specification now defines Active, Deprecated, and Removed states, a minimum twelve-month deprecation window, and a public registry of deprecated features. For a protocol two years old and carrying production traffic, adopting a deprecation policy is a larger signal of maturity than any individual feature in this list.

**MCP Apps and Enterprise Managed Authorization** join Tasks as official extensions under the same framework, which is what makes the core small enough to be stateless.

**Status as of October 2026.** 2026-07-28 remains the current specification; no newer version has been published. The core maintainers released an updated roadmap on August 22 naming five priorities for review time: agentic messaging primitives, HTTP transport unification and hardening, agent identity and enterprise security, improved primitives, and SDK developer experience. Under the new lifecycle rules, anything those produce is likely to arrive first as an opt-in extension rather than as a change to the core.

---

## Why MCP Won: The Adoption Story

MCP's dominance was not inevitable. When Anthropic launched the protocol in November 2024, it was one of several competing approaches to AI tool integration: alongside OpenAI's function calling, Google's Vertex AI tools, and custom frameworks. By mid-2026, MCP became the universal standard. Understanding why illuminates what makes a protocol succeed.

**The USB moment.** MCP solved a real pain point at the right time. By late 2024, every team building AI agents was writing custom integration code, different formats for LangChain tools, OpenAI function schemas, Anthropic tool_use, and framework-specific interfaces. The same GitHub integration was written dozens of times in slightly different ways. MCP provided "write once, use everywhere": a single server works with Claude, ChatGPT, Gemini, Copilot, Cursor, and any custom agent. The value proposition was immediately obvious.

**Critical mass through openness.** Anthropic open-sourced MCP from day one and transferred governance to the Agentic AI Foundation (AAIF) under the Linux Foundation, co-founded with Block and OpenAI and joined by Google, Microsoft, and Amazon as members. No single company controls the protocol. This removed the adoption barrier: competitors adopted MCP because they were co-owners, not just consumers. When OpenAI added MCP support to ChatGPT (mid-2025), the last major holdout disappeared.

**Network effects.** Each new MCP server makes the ecosystem more valuable for every client. Each new client makes it more worthwhile to build servers. By mid-2026: 9,400+ verified servers, 97M monthly SDK downloads, ~1,000 new servers per month. The flywheel is self-sustaining, building a new AI tool without MCP support is now a competitive disadvantage.

## MCP vs Function Calling vs A2A: When to Use Which

Three mechanisms for connecting AI to the outside world serve different purposes. Choosing correctly avoids overengineering.

**Function calling (provider-native tool use)**: the simplest option. The model receives JSON Schema descriptions of available functions, decides when to call one, and generates structured arguments. The application code executes the function and returns results. No protocol, no server process, no transport: just a function in your codebase. Use when: the tools are application-specific (not reusable across projects), the number of tools is small (<10), and there is no need for tool discovery or sharing. Example: a chatbot that queries its own database and sends emails.

**MCP**: the standardized tool integration protocol. Tools are defined once as MCP servers and work with any MCP-compatible client. Use when: tools should be reusable across applications (a GitHub integration used by Claude Code, Cursor, and your custom agent), the tool ecosystem is large or growing, tools are developed by different teams or third parties, or you need standardized security (OAuth 2.1), discovery, and lifecycle management. Example: enterprise tool platform where 50+ integrations serve multiple AI applications.

**A2A**: agent-to-agent coordination protocol. Enables agents built on different frameworks to discover each other and delegate tasks. Use when: agents from different organizations or frameworks need to coordinate, cross-organizational delegation is required (your travel agent talks to an airline's booking agent), or you need long-running task management across agent boundaries. Example: an enterprise orchestration platform where Salesforce agents coordinate with custom internal agents.

**Decision shortcut:** If the tool is a function in your code, then function calling. If the tool should be reusable across AI applications, then MCP. If agents need to find and delegate to other agents, then A2A. Most production systems use function calling for simple cases AND MCP for the tool ecosystem. They are not mutually exclusive.

## The Protocol Landscape: MCP, A2A, and ANP

MCP exists alongside two complementary protocols at different layers of the agent communication stack. Each protocol addresses a distinct communication boundary: tool access, inter-agent coordination, and open-network discovery. Understanding which layer a given integration problem falls into prevents overengineering and ensures the right protocol is applied.

| Layer | Protocol | Purpose | Analogy | Maturity (early 2026) |
|-------|----------|---------|---------|----------------------|
| **Agent to Tool** | **MCP** | Agent accesses external data and actions | USB | Production (97M+ downloads) |
| **Agent to Agent** | **A2A** | Agents coordinate with each other across organizations | LAN | Early production (150+ orgs, v1.0) |
| **Agent to Open Network** | **ANP** | Peer-to-peer agent discovery in open networks | Internet | Experimental (watch, don't invest) |

**A2A (Agent-to-Agent Protocol)**: created by Google, now under Linux Foundation governance. Enables agents from different teams, organizations, or frameworks to communicate. Uses Agent Cards (JSON-LD) for capability advertisement and supports long-running tasks (hours/days). See A2A Protocol for full coverage.

**ANP (Agent Network Protocol)**: an early-stage protocol for peer-to-peer agent communication in open, untrusted networks. Uses W3C Decentralized Identifiers (DIDs) for agent identity and discovery. Conceptually important, it addresses how agents will find and trust each other on the open internet, but not yet ready for production investment. Monitor progress; do not build on it today.

The three protocols are complementary: MCP for what an agent can do (tools), A2A for who an agent can work with (other agents in known organizations), and ANP for how agents will discover each other (open network, future). A production agent in 2026 uses MCP (mandatory), may use A2A (if cross-organization coordination is needed), and does not yet need ANP.

---

## Key Takeaways

Model Context Protocol has become the industry standard for AI tool integration. Instead of chaotic individual integrations, MCP provides a common language for interaction between AI applications and external services: now adopted by all major AI platforms and backed by cross-industry governance under the AAIF.

MCP architecture follows proven principles: separation of clients and servers, a standard message protocol (JSON-RPC 2.0), and capability declaration. The 2026-07-28 specification made the core stateless, removing the handshake and sessions so that any request can be served by any instance. The trade is explicit: the protocol stopped carrying state, and the application now carries it, including the idempotency that retries demand. Capability declaration keeps clients and servers of different versions compatible.

Key benefits of standardization: a radical reduction in integration costs (N + M instead of N x M), uniform security, and the ability to form an ecosystem with reusable components.

For agentic systems, MCP is particularly valuable. This gives unified access to data (resources) and the ability to execute actions (tools) through a standard interface.

MCP is the established foundation on which AI tools combine and interact, much like USB devices work with any computer.
