## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[../04_Multi_Agent_Systems/05_MAS_Consensus_and_Reliability|MAS Consensus and Reliability]]
**Next:** [[02_MCP_Components|MCP Components]]

---

# Model Context Protocol (MCP) Basics

## Introduction: A New Standard for AI Interaction

Every application, service, and database uses its own API. Integrating an AI assistant with a calendar requires one integration code, email requires another, and the file system requires a third. Developers create unique adapters for each tool.

Model Context Protocol (MCP) is an open standard from Anthropic that defines a universal way for AI models to interact with external systems. Instead of N×M integrations (N applications × M services), only N + M implementations are needed (N clients + M servers).

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

It is interesting to compare MCP with another successful standardization protocol — LSP (Language Server Protocol) from Microsoft. Both systems solve a similar M×N integration problem:

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

**Transport** — the mechanism for transmitting messages between client and server. MCP supports several transports: stdio (standard input/output, for local servers) and Streamable HTTP (for remote servers). The earlier SSE-based HTTP transport was deprecated in March 2025 in favor of Streamable HTTP, which supports bidirectional communication over a single HTTP connection.

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

## Key Takeaways

Model Context Protocol represents an important step toward standardizing the AI tooling ecosystem. Instead of chaotic individual integrations, MCP offers a common language for interaction between AI applications and external services.

MCP architecture follows proven principles: separation of clients and servers, a standard message protocol (JSON-RPC 2.0), and a capability negotiation mechanism for compatibility.

Key benefits of standardization: a radical reduction in integration costs (N + M instead of N × M), uniform security, and the ability to form an ecosystem with reusable components.

For agentic systems, MCP is particularly valuable, providing unified access to data (resources) and the ability to execute actions (tools) through a standard interface.

MCP is not just a technical standard — it is an investment in a future where AI tools freely combine and interact, much like USB devices work with any computer.


---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[../04_Multi_Agent_Systems/05_MAS_Consensus_and_Reliability|MAS Consensus and Reliability]]
**Next:** [[02_MCP_Components|MCP Components]]
