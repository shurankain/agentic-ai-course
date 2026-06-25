## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[../04_Multi_Agent_Systems/05_MAS_Consensus_and_Reliability|MAS Consensus and Reliability]]
**Next:** [[02_MCP_Components|MCP Components]]

---

# Model Context Protocol (MCP) Basics

## Introduction: A New Standard for AI Interaction

Every application, service, and database uses its own API. Integrating an AI assistant with a calendar requires one integration code, email requires another, and the file system requires a third. Developers create unique adapters for each tool.

Model Context Protocol (MCP) is an open standard — originally created by Anthropic (November 2024) and now governed by the Agentic AI Foundation (AAIF) under the Linux Foundation — that defines a universal way for AI models to interact with external systems. Instead of N×M integrations (N applications × M services), only N + M implementations are needed (N clients + M servers).

MCP has rapidly become the cross-industry standard for AI tool integration. As of mid-June 2026, the ecosystem includes **97M+ monthly SDK downloads**, **9,400+ verified servers** (~9,652 in the official registry by late May), and native support from all major AI platforms (Claude, ChatGPT, Gemini, Copilot, Cursor, VS Code, and many others). Enterprise adoption: **28% of Fortune 500** companies have deployed dedicated MCP servers for production AI workflows. For context: 80% of F500 use AI agents in production generally — MCP is the dominant but not the only integration mechanism. **41% of surveyed software organizations** are in limited or broad MCP production (Stacklok 2026 report). ~1,000 new servers appear monthly. The official MCP Registry — now governed by the Linux Foundation's Agentic AI Foundation (AAIF, with Block and OpenAI as co-stewards) — catalogs 500+ verified public servers with namespace verification via GitHub OAuth/OIDC and DNS/HTTP domain verification to prevent impersonation. MCP has transitioned from experiment to enterprise infrastructure.

**SDK version warning (as of late June 2026):** MCP Python SDK **v2.0.0a1** was published June 11 — a breaking alpha. FastMCP has been renamed to `MCPServer`. The protocol becomes stateless at the protocol layer (no handshake, no session pinning). Beta is targeted for June 30, stable v2 for July 27. **Pin your dependency to `mcp>=1.27,<2`** for production stability. The TypeScript SDK v1.29.0 dropped Zod from peerDependencies in favor of Standard Schema support (Zod v4, Valibot, ArkType) and added a Fastify middleware adapter.

This chapter covers MCP architecture, the problems it solves, and its impact on AI application development.

## The Problem MCP Solves

To appreciate the significance of MCP, it is important to understand the problem that existed before it.

### Integration Chaos

A modern AI assistant must be capable of many things: searching the internet, working with files, querying databases, managing calendars, sending emails, and interacting with APIs of various services. Each of these capabilities requires an integration.

The traditional approach — function calling or tool use — only partially solves this problem. Yes, the model can call functions. But each function is implemented individually. A Slack integration looks different from a Discord integration. Working with PostgreSQL differs from working with MongoDB. Every new tool requires new code.

This creates scaling problems. If you have ten AI applications and fifty tools, you potentially need five hundred integrations. Each one must be written, tested, documented, and maintained.

### Duplicated Effort

Even worse, different teams solve the same problems in parallel. One team writes a Claude integration with GitHub. Another team writes the same integration for GPT. A third does it for Gemini. The code is nearly identical, but it gets written three times because there is no common standard.

The open-source community suffers from fragmentation. Dozens of libraries do roughly the same thing but in slightly different ways. Developers must choose between them, learn the specifics of each, and switch when changing tools.

### Security and Control

Without a standard, it is difficult to ensure uniform security. Each integration implements authentication and authorization differently. Auditing becomes a nightmare — each adapter must be reviewed separately. A vulnerability in one integration does not imply a vulnerability in others, which seems positive, but it also means that fixing one vulnerability does not fix similar ones elsewhere.

## What MCP Is

Model Context Protocol is an open standard that defines how AI applications (clients) interact with external data sources and tools (servers).

### The USB Analogy

A useful analogy is USB. Before USB, every device required its own unique connector and driver. A printer used one cable, a scanner another, a camera a third. USB unified the physical interface and the communication protocol. Now any USB device works with any USB port.

MCP does the same for AI. Any MCP client can work with any MCP server. By writing a GitHub integration once as an MCP server, you automatically gain compatibility with all MCP clients: Claude Desktop, VS Code with AI extensions, and any custom applications that support the protocol.

### Relationship with Language Server Protocol (LSP)

It is interesting to compare MCP with another successful standardization protocol — LSP (Language Server Protocol) from Microsoft. Both systems solve a similar M×N integration problem by introducing a middleware layer that decouples clients from providers. The architectural parallels are striking, though the domains differ: LSP targets language tooling while MCP targets AI-service integration. The following table highlights the key structural similarities and differences.

| Aspect | LSP | MCP |
|--------|-----|-----|
| **Problem** | Each IDE × each language | Each AI client × each service |
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
- MCP supports Sampling — callbacks to the model
- LSP is more mature (since 2016); MCP launched in 2024 and has rapidly matured with AAIF governance and broad industry adoption

### Protocol Design Principles

MCP developers followed several fundamental principles that shaped the protocol's architecture:

**Loose Coupling Principle.** The client and server know about each other only through the abstract protocol interface. The server does not know which model runs on the client side. The client does not know the server's implementation details. This allows either side to be replaced without affecting the other.

**Explicit Declaration Principle.** All capabilities are declared explicitly through capabilities, tools, resources, and prompts. There are no hidden functions or implicit agreements. This ensures predictability and simplifies debugging.

**Safe Failure Principle.** The protocol is designed so that failures are explicit and recoverable. If a client does not understand a capability, it ignores it. If a server does not support a method, it returns an error. The system degrades gracefully.

**Extensibility Principle.** The protocol supports experimental capabilities and custom extensions. New features can be added without breaking compatibility with existing implementations.

**Symmetry Principle Where Appropriate.** Although MCP follows the client-server model, some aspects are symmetric: both can send notifications, both participate in the handshake. Sampling even "reverses" the roles — the server becomes the call initiator.

These principles reflect lessons learned from other protocols: LSP, gRPC, GraphQL. MCP does not reinvent the wheel but applies proven patterns to a new challenge — integrating AI with the external world.

### Core Concepts

MCP is built around several key concepts.

**Client** — the AI application that needs access to external data and functions. Typical clients include chat interfaces (Claude Desktop), IDEs with AI capabilities (Cursor), and autonomous agents.

**Server** — the service that provides data and functionality. A server can expose access to the file system, a database, an API, or any other resource. A single server typically focuses on one domain — there are servers for GitHub, for Slack, for PostgreSQL.

**Host** — the application in which the client runs. The host manages the client's lifecycle and can define security policies. For example, Claude Desktop is the host for an MCP client.

**Transport** — the mechanism for transmitting messages between client and server. MCP supports several transports: stdio (standard input/output, for local servers) and Streamable HTTP (for remote servers). The earlier SSE-based HTTP transport was deprecated in March 2025 in favor of Streamable HTTP. The reason: SSE is unidirectional (server → client only), requiring a separate HTTP endpoint for client → server messages. Streamable HTTP provides true bidirectional communication over a single HTTP connection, simplifying deployment and enabling server-initiated messages (Sampling, Elicitation) to flow naturally.

**STDIO transport security warning (May 2026):** A critical vulnerability (CVSS 9.8) was discovered in the STDIO transport mechanism, enabling arbitrary OS command execution. The vulnerability affects all MCP SDK implementations and puts an estimated 200,000+ servers at risk. This is a fundamental design issue in the transport layer, not an implementation bug in a specific SDK. A separate NGINX integration flaw (also CVSS 9.8) was discovered around the same time. These are the first serious architectural vulnerabilities in a protocol that had already become an industry standard — a classic example of adoption outpacing security review. Production deployments using STDIO transport should implement defense-in-depth: sandboxing, network policies, and input validation at every boundary. See [[../14_Security_Safety/03_Agent_Security|Agent Security]] for detailed MCP security analysis.

## MCP Architecture

MCP architecture is designed with flexibility and extensibility in mind. It follows the separation of concerns principle: each component does one thing well.

### Three-Tier Model

At the top level are AI applications — chatbots, agents, IDE assistants. They formulate natural language requests and interpret results.

The middle level consists of MCP clients. They translate high-level requests into specific protocol calls, manage connections to servers, and handle security and caching.

The bottom level consists of MCP servers. They interact directly with external systems: reading files, executing SQL queries, and calling APIs.

### Message Protocol

MCP uses JSON-RPC 2.0 for message exchange. This is a mature, well-understood protocol with clear semantics for requests, responses, and notifications.

Each message has a standard structure. A request contains a method (what to do), parameters (with what data), and an identifier (for matching with the response). A response contains either a result or an error. Notifications are one-way messages that do not require a response.

This structure ensures reliability: it is always clear what was requested and what was received in response. Errors are handled explicitly rather than silently ignored.

### Capability Negotiation

Upon connection, a three-phase handshake occurs. The client sends an initialize message with the protocol version and its capabilities. The server responds with its capabilities and metadata. The client confirms readiness with an initialized notification.

Capability negotiation works on the intersection principle: a feature is active only if both sides support it. The client can declare readiness for sampling or experimental features. The server declares its resources, tools, and prompts with optional subscription and notification flags.

Protocol versioning ensures compatibility. If versions do not match, the connection is terminated. The mechanism allows older clients to work with newer servers and vice versa by ignoring unknown capabilities.

## Benefits of MCP

Adopting a standardized protocol brings numerous benefits.

### Universality and Reusability

An MCP server, once written, works with all MCP clients. A Jira integration implemented as an MCP server is automatically available in Claude, in Cursor, and in any other MCP-compatible application. There is no need to write three different plugins for three different tools.

This radically reduces the total cost for the ecosystem. Instead of N×M integrations (N applications × M services), only N + M implementations are needed (N clients + M servers). At large scale, the savings are enormous.

### Standardized Security

MCP defines standard authentication and authorization mechanisms. All servers follow the same patterns, simplifying auditing and certification. A vulnerability found in the protocol implementation is fixed centrally, automatically protecting all users.

Hosts can apply uniform policies to all connected servers. Prohibiting writes to the file system is a single setting that applies to all servers with file access.

### Ecosystem and Community

The standard creates a foundation for an ecosystem. Catalogs of MCP servers emerge, along with tools for developing and testing them and ready-made solutions for common tasks. Developers can focus on unique business logic while using ready-made components for standard functions.

The openness of the standard encourages community contributions. Anyone can create an MCP server for a needed service and share it with the world.

## MCP in the Context of Agentic Systems

MCP is particularly important for AI agents. Agents by definition must interact with the external world — reading data, executing actions, and verifying results. MCP provides a standard way to do so.

### Data Access

An agent often needs information that is not in its training data: current news, the contents of specific documents, database states. MCP server resources provide this data in a unified format.

The agent queries the MCP client for a list of available resources, selects the ones it needs, and reads their contents. All through a standard protocol, regardless of where the data physically originates — a file, a database, or an external API.

### Action Execution

An agent must be able to not only read but also act: creating files, sending messages, placing orders. MCP server tools provide these capabilities.

Each tool is described by a schema: what parameters it accepts, what it returns, and what errors are possible. The agent can analyze available tools, select the appropriate ones for the task, and call them with the correct parameters.

### Coordinating Multiple Sources

A real-world task often requires data from multiple sources and actions across multiple systems. You might need to read a task from Jira, find related code in GitHub, analyze logs in Elasticsearch, and create a pull request.

MCP allows connecting multiple servers simultaneously. The agent sees a combined set of resources and tools from all servers and can combine them to solve complex tasks.

## MCP Apps (January 2026)

An evolution beyond data and actions: MCP tools can now return **interactive UI components** — forms, charts, filtered tables, maps, approval buttons — rendered directly in the conversation. When a tool returns an MCP App, the host application (Claude, an IDE, a custom agent UI) renders the component inline rather than displaying raw data.

This blurs the line between a tool call and an application. An MCP server for a database is no longer limited to returning query results as text — it can return an interactive table with sorting, filtering, and pagination. A monitoring MCP server can return a live chart. A workflow MCP server can return an approval form with Accept/Reject buttons.

MCP Apps represent the evolution of MCP from a data protocol to an **application platform**. The implications for agent UX are significant: agents can provide rich, interactive responses rather than text-only outputs, without requiring custom frontend development for each integration.

## MCP Server Cards and Discovery

As the MCP ecosystem grew to 9,400+ verified servers (mid-June 2026), discovering the right server for a task became a challenge. **MCP Server Cards** address this by providing a standard for exposing structured server metadata via a `.well-known` URL. A browser, crawler, or registry can discover a server's capabilities — what tools it provides, what resources it exposes, what authentication it requires — without connecting to it or starting a session.

This is analogous to OpenAPI specification files for REST APIs: a machine-readable description of capabilities that enables automated discovery and cataloging. The **MCP Registry** (launched in the second half of 2025) serves as the single source of truth for available MCP servers, supporting both public servers and private sub-registries that organizations can customize for their specific needs.

## MCP Tasks: Long-Running Operations

**MCP Tasks** (formalized as SEP-1686) provide first-class support for long-running asynchronous operations — a critical capability for agent orchestration where tool calls may take minutes or hours rather than milliseconds.

A Task represents an operation with a lifecycle: created → running → completed (or failed/cancelled). During execution, the server sends **progress updates** (percentage complete, status messages) and can produce **partial results** (intermediate outputs available before final completion). The client can **cancel** a running task if it is no longer needed.

**Why this matters for agents:** Without Tasks, an agent calling a long-running tool (generating a report, running a CI pipeline, processing a large document set) must either block (wasting context window time) or implement custom polling logic. Tasks make this a protocol-level concern — the agent issues a tool call, receives a task ID, and can continue other work while monitoring progress asynchronously.

## MCP 2026-07-28 Release Candidate (May 21, 2026)

The biggest revision since MCP's launch was published as a Release Candidate on May 21, 2026, with the final specification targeted for July 28, 2026. Key changes (as of late May 2026):

**Stateless core.** The protocol moves to HTTP-level operation without mandatory sessions. Servers no longer need to track client session state, enabling horizontal scaling behind standard load balancers without sticky sessions. This is the most architecturally significant change — it transforms MCP from a session-oriented protocol into a stateless one suitable for cloud-native deployment.

**MCP Apps.** Servers can now return interactive HTML UI rendered in a sandboxed iframe within the host application. A database MCP server can return a filterable table; a monitoring server can return a live chart; a workflow server can return an approval form. This formalizes and extends the MCP Apps concept introduced earlier.

**Tasks as extension.** The Tasks primitive (formalized from SEP-1686) becomes an official protocol extension rather than a core requirement. Servers that need long-running operation support opt into the Tasks extension; simple synchronous servers remain lightweight.

**Formal deprecation policy.** The specification now includes explicit rules for deprecating protocol features, transport mechanisms, and capability versions — providing stability guarantees for production deployments.

**OAuth/OIDC alignment.** Authentication aligns with standard OAuth 2.0 and OpenID Connect flows, simplifying integration with enterprise identity providers.

**Transport default: stateless Streamable HTTP.** MCP is moving to stateless Streamable HTTP as the default transport. This enables horizontal scaling without sticky sessions — critical for production deployments where MCP servers must handle thousands of concurrent clients behind load balancers. The STDIO transport remains supported for local development.

---

## Why MCP Won: The Adoption Story

MCP's dominance was not inevitable. When Anthropic launched the protocol in November 2024, it was one of several competing approaches to AI tool integration — alongside OpenAI's function calling, Google's Vertex AI tools, and various custom frameworks. By mid-2026, MCP became the universal standard. Understanding why illuminates what makes a protocol succeed.

**The USB moment.** MCP solved a real pain point at the right time. By late 2024, every team building AI agents was writing custom integration code — different formats for LangChain tools, OpenAI function schemas, Anthropic tool_use, and framework-specific interfaces. The same GitHub integration was written dozens of times in slightly different ways. MCP provided "write once, use everywhere" — a single server works with Claude, ChatGPT, Gemini, Copilot, Cursor, and any custom agent. The value proposition was immediately obvious.

**Critical mass through openness.** Anthropic open-sourced MCP from day one and transferred governance to the Agentic AI Foundation (AAIF) under the Linux Foundation — with Google, OpenAI, Microsoft, Amazon, Meta, and Block as co-stewards. No single company controls the protocol. This removed the adoption barrier: competitors adopted MCP because they were co-owners, not just consumers. When OpenAI added MCP support to ChatGPT (mid-2025), the last major holdout disappeared.

**Network effects.** Each new MCP server makes the ecosystem more valuable for every client. Each new client makes it more worthwhile to build servers. By mid-2026: 9,400+ verified servers, 97M monthly SDK downloads, ~1,000 new servers per month. The flywheel is self-sustaining — building a new AI tool without MCP support is now a competitive disadvantage.

## MCP vs Function Calling vs A2A: When to Use Which

Three mechanisms for connecting AI to the outside world serve different purposes. Choosing correctly avoids overengineering.

**Function calling (provider-native tool use)** — the simplest option. The model receives JSON Schema descriptions of available functions, decides when to call one, and generates structured arguments. The application code executes the function and returns results. No protocol, no server process, no transport — just a function in your codebase. Use when: the tools are application-specific (not reusable across projects), the number of tools is small (<10), and there is no need for tool discovery or sharing. Example: a chatbot that queries its own database and sends emails.

**MCP** — the standardized tool integration protocol. Tools are defined once as MCP servers and work with any MCP-compatible client. Use when: tools should be reusable across applications (a GitHub integration used by Claude Code, Cursor, and your custom agent), the tool ecosystem is large or growing, tools are developed by different teams or third parties, or you need standardized security (OAuth 2.1), discovery, and lifecycle management. Example: enterprise tool platform where 50+ integrations serve multiple AI applications.

**A2A** — agent-to-agent coordination protocol. Enables agents built on different frameworks to discover each other and delegate tasks. Use when: agents from different organizations or frameworks need to coordinate, cross-organizational delegation is required (your travel agent talks to an airline's booking agent), or you need long-running task management across agent boundaries. Example: an enterprise orchestration platform where Salesforce agents coordinate with custom internal agents.

**Decision shortcut:** If the tool is a function in your code → function calling. If the tool should be reusable across AI applications → MCP. If agents need to find and delegate to other agents → A2A. Most production systems use function calling for simple cases AND MCP for the tool ecosystem — they are not mutually exclusive.

## The Protocol Landscape: MCP, A2A, and ANP

MCP exists alongside two complementary protocols at different layers of the agent communication stack. Each protocol addresses a distinct communication boundary: tool access, inter-agent coordination, and open-network discovery. Understanding which layer a given integration problem falls into prevents overengineering and ensures the right protocol is applied.

| Layer | Protocol | Purpose | Analogy | Maturity (early 2026) |
|-------|----------|---------|---------|----------------------|
| **Agent → Tool** | **MCP** | Agent accesses external data and actions | USB | Production (97M+ downloads) |
| **Agent → Agent** | **A2A** | Agents coordinate with each other across organizations | LAN | Early production (150+ orgs, v1.0) |
| **Agent → Open Network** | **ANP** | Peer-to-peer agent discovery in open networks | Internet | Experimental (watch, don't invest) |

**A2A (Agent-to-Agent Protocol)** — created by Google, now under Linux Foundation governance. Enables agents from different teams, organizations, or frameworks to communicate. Uses Agent Cards (JSON-LD) for capability advertisement and supports long-running tasks (hours/days). See [[05_A2A_Protocol|A2A Protocol]] for full coverage.

**ANP (Agent Network Protocol)** — an early-stage protocol for peer-to-peer agent communication in open, untrusted networks. Uses W3C Decentralized Identifiers (DIDs) for agent identity and discovery. Conceptually important — it addresses how agents will find and trust each other on the open internet — but not yet ready for production investment. Monitor progress; do not build on it today.

The three protocols are complementary: MCP for what an agent can do (tools), A2A for who an agent can work with (other agents in known organizations), and ANP for how agents will discover each other (open network, future). A production agent in 2026 uses MCP (mandatory), may use A2A (if cross-organization coordination is needed), and does not yet need ANP.

---

## Key Takeaways

Model Context Protocol has become the industry standard for AI tool integration. Instead of chaotic individual integrations, MCP provides a common language for interaction between AI applications and external services — now adopted by all major AI platforms and backed by cross-industry governance under the AAIF.

MCP architecture follows proven principles: separation of clients and servers, a standard message protocol (JSON-RPC 2.0), and a capability negotiation mechanism for compatibility.

Key benefits of standardization: a radical reduction in integration costs (N + M instead of N × M), uniform security, and the ability to form an ecosystem with reusable components.

For agentic systems, MCP is particularly valuable, providing unified access to data (resources) and the ability to execute actions (tools) through a standard interface.

MCP is no longer an emerging standard — it is the established foundation on which AI tools freely combine and interact, much like USB devices work with any computer.


---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[../04_Multi_Agent_Systems/05_MAS_Consensus_and_Reliability|MAS Consensus and Reliability]]
**Next:** [[02_MCP_Components|MCP Components]]
