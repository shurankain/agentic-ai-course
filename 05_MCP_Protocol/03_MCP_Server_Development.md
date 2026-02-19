## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[02_MCP_Components|MCP Components]]
**Next:** [[04_MCP_Client_Integration|MCP Client Integration]]

---

# MCP Server Development

## Introduction to Server Development

An MCP server provides communication between an AI model and external systems (databases, APIs, file systems). An effective server exposes system capabilities in a format optimized for the language model.

## Architectural Principles

### Single Responsibility Principle

Each MCP server should have a clear and bounded area of responsibility. A server for file system operations should not simultaneously manage a database and send emails. This separation simplifies development, testing, and maintenance.

When a server focuses on a single domain, its API becomes intuitive. A user connecting a GitHub server expects to see tools for managing repositories, commits, and pull requests — and nothing extraneous.

This principle also promotes security. A minimal set of capabilities means a minimal attack surface. A server that can only read files cannot accidentally or maliciously delete critical data.

### Transparency Principle

A server should clearly document its capabilities and limitations. Descriptions of resources, tools, and prompts are not a formality but critically important information for the language model.

A good description answers the questions: what does this component do? When should it be used? What side effects does it have? What parameters does it accept and what values does it return?

Transparency extends to error handling as well. The server should return informative error messages that help understand the cause of a problem and how to resolve it.

### Idempotency Principle

Where possible, server operations should be idempotent — a repeated call with the same parameters should not create additional side effects. This is especially important in unreliable network conditions where requests may be retried.

For read operations, idempotency is natural. For write operations, it can often be achieved through unique identifiers. Instead of "create a record," one can implement "create or update a record with ID X."

If an operation cannot be idempotent by nature (for example, sending a message), this should be explicitly stated in the documentation and tool annotations.

### Principle of Least Privilege

A server should request only the access rights necessary to perform its functions. If a server only needs to read data, it should not have write permissions.

This principle protects both the user from accidental damage and the system from malicious use. Even if a server is compromised, its capabilities are limited to the granted privileges.

## Server Lifecycle

### Initialization

On startup, the server performs a series of preparatory operations. Configuration is loaded, connections to external systems are established, and internal data structures are initialized.

It is important for the server to properly handle initialization errors. If a critical resource is unavailable, the server should report this rather than silently continue operating in an inoperable state.

After successful initialization, the server is ready to accept client connections. Each new connection begins with a capability negotiation procedure.

### Capability Negotiation

When a client connects to the server, an exchange of information about supported capabilities takes place. The server declares which primitives it provides: resources, tools, prompts, Sampling support.

The client, in turn, reports its capabilities: support for subscriptions, file system roots, experimental features. This negotiation allows both sides to adapt their behavior.

Protocol versioning also occurs at this stage. The client and server agree on the MCP version to use, ensuring compatibility.

### Request Processing

After negotiation, the server enters request processing mode. Each incoming request is a JSON-RPC message with a method and parameters.

The server routes requests to the appropriate handlers: resource read requests go to resource handlers, tool calls go to tool handlers, and so on.

It is important to ensure correct handling of parallel requests. MCP does not guarantee execution order, and the server must be prepared to process multiple operations concurrently.

### Notifications and Events

In addition to request-response interactions, MCP supports notifications — one-way messages that do not require a response. The server can send notifications about changes in the list of resources, tools, or prompts.

If the server supports resource subscriptions, it must track changes and notify subscribed clients. This requires mechanisms for observing data sources.

### Shutdown

Graceful server shutdown includes notifying connected clients, completing in-progress operations, and releasing resources. The server should handle termination signals (SIGTERM, SIGINT) gracefully.

Unexpected connection interruptions should be handled properly. The server should not crash due to a disconnected client.

## Transport Mechanisms

### Standard Input/Output (stdio)

The most common transport for local MCP servers is standard input/output. The client launches the server as a child process and exchanges data with it via stdin and stdout.

This mechanism is simple to implement and does not require network infrastructure. It is ideal for command-line tools, IDE integrations, and local automation.

The limitation of stdio is the inability to provide remote access. The server must be installed on the same machine as the client.

### Server-Sent Events (SSE)

For remote access, MCP supports HTTP-based transport with Server-Sent Events. The client connects to the server via HTTP and receives events in real time.

SSE provides a unidirectional data flow from server to client. To send requests, the client uses separate HTTP calls. This makes SSE compatible with most proxies and firewalls.

### WebSocket

A full-duplex WebSocket connection provides bidirectional data exchange with minimal overhead. This is the optimal choice for intensive interaction with frequent requests and responses.

WebSocket requires infrastructure support but provides better performance for complex scenarios.

### Choosing a Transport

The choice of transport depends on the use case. For local tools, stdio is the obvious choice. For cloud services, SSE or WebSocket depending on interactivity requirements.

A well-designed server abstracts the transport layer, allowing the same code to be used with different transports.

### Transport Layer Security

For remote connections, transport security is critical. MCP does not prescribe specific mechanisms but follows standard practices:

**TLS for encryption.** All remote connections should use TLS 1.2 or higher. This provides data confidentiality through encryption, integrity through change detection, and authentication through server identity verification.

**Certificate validation.** The client should verify the certificate expiration date, the chain to a trusted certificate authority, and the hostname match in SAN extensions or the CN field.

**Mutual authentication (mTLS).** For high-security scenarios, mutual authentication is used where the client also presents a certificate. The process works as follows: the server sends its certificate to the client, the client sends its certificate to the server, both parties verify the received certificates, and only after successful mutual verification is an encrypted communication channel established.

**Why does MCP use JSON-RPC rather than gRPC?** This is an interesting architectural choice. JSON-RPC uses text-based JSON serialization and can operate over any transport (stdio, HTTP, WebSocket), whereas gRPC requires binary Protocol Buffers and HTTP/2. Debugging JSON-RPC is easier since messages are human-readable, while gRPC requires specialized tools. Schema in JSON-RPC is optional, whereas in gRPC it is mandatory via .proto files. The JSON-RPC ecosystem is broader and does not require code generation.

JSON-RPC was chosen for MCP for four key reasons: simplicity of working over stdio (gRPC requires HTTP/2, which is more complex for processes), human readability for easier debugging and protocol evolution, minimal dependencies (no protobuf compiler needed), and schema flexibility allowing protocol evolution without recompilation.

## Error Handling

### Error Categories

MCP distinguishes several error categories. Protocol errors arise from malformed messages — incorrect format, missing fields, unknown methods.

Application errors relate to server logic — resource not found, insufficient permissions, invalid parameters. These errors are domain-specific.

Internal errors indicate problems in the server code — unhandled exceptions, connection failures to external systems.

### Informative Messages

Error messages should be informative yet safe. They should help diagnose the problem but not reveal the internal system structure or confidential data.

A good error message: "Document with ID 'abc123' not found in the database." A bad one: "SQLException: connection to database at 192.168.1.100:5432 failed with password authentication error."

### Error Recovery

The server should be resilient to errors. A failure in processing one request should not affect other requests or the overall server stability.

Transient issues — network timeouts, connection pool exhaustion — can be resolved through retries with exponential backoff.

### Error Resilience Patterns

When developing MCP servers, it is useful to apply proven resilience patterns:

**Retry with exponential backoff.** This pattern involves repeated attempts with increasing intervals: the first attempt occurs immediately, the second after one second, the third after two seconds, the fourth after four seconds, the fifth after eight seconds. Adding jitter (random variance within ±20%) prevents "thundering herd" behavior where many clients retry requests simultaneously, creating load spikes.

**Circuit Breaker.** This pattern protects against cascading failures through a state machine with three states. In the CLOSED state, normal operation occurs and all requests pass through. After a certain number of errors, the state machine transitions to the OPEN state, where fast failure occurs without attempting to reach the faulty backend, protecting it. After a timeout, the state machine transitions to the HALF-OPEN state, where test requests are allowed through to check for recovery. On success, it returns to CLOSED; on failure, it goes back to OPEN.

**Bulkhead (compartment isolation).** This pattern limits resources per operation by creating separate thread pools. For example, a pool of 5 threads is allocated for heavy operations, and a pool of 20 threads for lightweight operations. This prevents a situation where slow operations block fast ones.

**Timeout at all levels.** Timeouts must be set at every level of interaction: Connect timeout (typically 5 seconds) for establishing a connection, Read timeout (typically 30 seconds) for waiting for data, Request timeout (typically 60 seconds) for the full request cycle, and Idle timeout (typically 5 minutes) for inactive connections.

Applying these patterns makes the server resilient to partial dependency failures and prevents problems from propagating throughout the entire system.

## Security

### Input Validation

All input data from the client must be validated. Tool parameters are checked against the schema. Resource URIs are checked for valid format and scope.

Special attention should be paid to injection protection. If parameters are used in SQL queries, shell commands, or other execution contexts, they must be appropriately escaped or parameterized.

### Authorization and Authentication

The MCP specification now includes an OAuth 2.1 authorization framework with PKCE, Protected Resource Metadata (RFC 9728), and support for enterprise-managed authorization. For simpler deployments, the server may also use tokens, certificates, or other mechanisms.

Authorization determines which operations are permitted for the client. The server can restrict access to specific resources or tools based on client identity.

### Isolation and Sandboxing

Where possible, server operations should be executed in an isolated environment. A server for code execution should use containers or sandboxes. A server for file operations should restrict access to specific directories.

Isolation protects both the host system from malicious actions and different requests from each other.

## Testing MCP Servers

### Unit Testing

Individual server components — resource, tool, and prompt handlers — are tested in isolation. External dependencies are mocked, and edge cases are verified.

Unit tests are fast and stable. They codify the contract of each component and help prevent regressions.

### Integration Testing

Integration tests verify component interactions and protocol correctness. A test client simulating real interaction is used.

These tests connect to the server, execute request sequences, and verify responses. They are slower but catch integration issues.

### End-to-End Testing

Full testing involves real external systems and real clients. This is the most accurate verification but also the most complex to set up.

E2E tests are especially important before release, when it is necessary to confirm the entire system is working correctly.

## Monitoring and Observability

### Logging

The server should maintain detailed logs for problem diagnosis. Incoming requests, executed operations, and errors that occur are logged.

Log levels control verbosity: DEBUG for development, INFO for production, ERROR for critical issues.

### Metrics

Key server metrics include: request count, response time, error rate, and resource utilization. Metrics help understand the load and detect performance degradation.

### Tracing

Distributed tracing allows tracking a request through the entire system — from client through server to external systems. This is critically important for diagnosing complex scenarios.

---

## Modern MCP Ecosystem

### FastMCP: The De Facto Standard

FastMCP is a library created by Jeremiah Lowin (founder of Prefect) that became the de facto standard for developing MCP servers in Python. In late 2024, FastMCP was integrated into the official MCP SDK under the name `mcp.server.fastmcp`.

**Advantages of FastMCP:**
FastMCP provides declarative syntax with decorators, automatic JSON Schema generation from Python type hints, built-in parameter validation via Pydantic, and minimal boilerplate code. Developers do not need to manually describe tool schemas — they are automatically generated from type annotations.

**FastMCP Server Architecture:**
A server is created through an instance of the FastMCP class with a specified name. Components are then registered via decorators: tools via `@mcp.tool()`, resources via `@mcp.resource(uri)`, and prompts via `@mcp.prompt()`. Each function automatically becomes the corresponding MCP primitive. For example, a function with the `@mcp.tool()` decorator and type annotations `def add(a: float, b: float) -> float` automatically registers a tool with two float parameters and a float return value. The function's docstring becomes the tool description for the language model.

**Key Decorators:**
The `@mcp.tool()` decorator registers a function as a tool — a function that performs actions. The `@mcp.resource(uri)` decorator registers a function as a resource — a data source for reading with the specified URI. The `@mcp.prompt()` decorator registers a function as a prompt — a template for interacting with the language model.

### MCP Inspector

MCP Inspector is the official tool for testing and debugging MCP servers. It allows interactive communication with the server, viewing available components, and testing calls.

**Installation and Launch:**
Inspector is installed via npx with the command `npx @anthropic-ai/mcp-inspector` for Node.js environments, or via pip with the command `pip install mcp-inspector` for the Python wrapper. It is critically important to use version 0.14.1 or higher, as earlier versions contained a serious security vulnerability (CVE-2025-49596) that allowed remote code execution.

**Inspector Capabilities:**
Inspector provides viewing of the complete list of registered tools, resources, and prompts on the server. It allows interactive invocation of tools with various parameters to test their behavior. It displays all JSON-RPC messages for debugging protocol interactions. It performs schema validation, checking that parameters and responses conform to declared types.

### Current Ecosystem State

**Official SDKs:**
For Python, the `mcp` package is available via PyPI version 1.25.0 and above. For TypeScript, the `@modelcontextprotocol/sdk` package version 1.0.0 and above. For Kotlin, `mcp-kotlin-sdk` version 0.5.0 exists, though it is less mature.

**Supporting Clients:**
Claude Desktop is in production and is the first official client. VS Code received native MCP support starting from version 1.102. Cursor, a popular AI editor, also supports MCP in production. ChatGPT added support in beta mode in late 2024. The minimalist editor Zed supports MCP in production.

**Key Ecosystem Events:**
In November 2024, Anthropic launched MCP in production. In December 2024, the protocol was transferred to the Linux Foundation under the governance of the AI Alliance Infrastructure Forum to ensure independence and openness. That same month, OpenAI, Microsoft, and Google announced plans to support MCP. In 2025, MCP is expected to become a cross-industry standard for AI agent integration.

### Security Considerations

When developing MCP servers, consider several critical security aspects:

**Input Validation:**
All tool parameters must undergo strict validation. For example, for a file reading function, it is necessary to use a Pydantic Field with a regular expression restricting allowed characters in the path. This protects against path injection attacks where a malicious user may attempt to access system files through constructs like `../../../etc/passwd`.

**Sandboxing:**
Restrict server access to the file system to only the necessary directories. Use Docker containers or other isolation mechanisms for executing potentially dangerous operations. Apply the principle of least privilege — the server should have only those rights that are absolutely necessary for its operation.

**Rate Limiting:**
Implement overload protection by limiting the frequency of tool calls. For example, one can track timestamps of recent calls and reject requests if they occur more frequently than a defined threshold (for example, no more than one call per second). This protects against both accidental loops and malicious exploitation of server resources.

### Integration with Claude Desktop

MCP server configuration for Claude Desktop is done through a JSON file containing an `mcpServers` object with server descriptions. Each server is defined by a launch command (for example, `python`), command-line arguments (for example, `-m my_mcp_server` to run a Python module), and optional environment variables for passing API keys and other confidential data.

**Configuration Location:**
The `claude_desktop_config.json` configuration file is located in different places depending on the operating system. On macOS, it is `~/Library/Application Support/Claude/claude_desktop_config.json`. On Windows, it is `%APPDATA%\Claude\claude_desktop_config.json`. On Linux, it is `~/.config/Claude/claude_desktop_config.json`.

### Best Practices for Production

1. **Use type hints** — FastMCP generates JSON Schema from type hints
2. **Write docstrings** — they become descriptions for the LLM
3. **Validate input data** — Pydantic Fields for constraints
4. **Log everything** — MCP supports structured logging
5. **Test via Inspector** — before integrating with clients
6. **Version your API** — for backwards compatibility

## Key Takeaways

MCP server development is a balance between functionality, security, and usability. A good server follows the principles of single responsibility and least privilege, providing a clear and documented API.

The server lifecycle includes initialization, capability negotiation, request processing, and graceful shutdown. Each stage requires attention to detail and error handling.

The choice of transport depends on the use case. Stdio is suitable for local tools; SSE and WebSocket are suitable for remote access.

Security is not an option but a mandatory requirement. Input validation, authorization, and isolation protect the system from abuse.

Testing and monitoring ensure server quality and reliability in production. Investment in these areas pays off with stability and ease of maintenance.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[02_MCP_Components|MCP Components]]
**Next:** [[04_MCP_Client_Integration|MCP Client Integration]]
