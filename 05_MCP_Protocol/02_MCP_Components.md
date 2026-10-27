# MCP Components: Resources, Tools, Prompts, and Sampling

Model Context Protocol defines a set of primitives for client-server interaction. The three original primitives (Resources, Tools, and Prompts) form the foundation, while Sampling, Elicitation, and Tasks extend the protocol's capabilities.

Resources answer the question "what can be read?", Tools: "what can be done?", Prompts: "how to formulate a request?", Elicitation, "how to ask the user?", Tasks: "how to track long-running work?". This architecture separates concerns between components, similar to the MVC pattern.

## Resources: Data and Content Access

### Resource Philosophy

Resources in MCP represent an abstraction for accessing any data that may be useful to a language model. Conceptually, a resource is a unit of content accessible by a unique URI identifier. This model intentionally resembles web architecture, where each document has its own unique address.

The key property of resources is their passivity. A resource does not perform actions and does not change system state. It simply provides data for reading. This is the fundamental distinction from tools, which can actively interact with the external world.

Why is this separation so important? Consider a situation where an AI agent analyzes business data. Reading a sales report is an operation without side effects that can be performed multiple times without consequences. But submitting an order to a supplier is an action with real consequences that requires explicit confirmation. MCP formalizes this distinction at the protocol level.

### Resource Structure and Identification

Each resource in MCP is identified by a URI: a universal resource identifier. The URI scheme is defined by the server and can be arbitrary. This gives flexibility in namespace organization. For example, a file system server might use the `file://` scheme, a database server, `db://`, and an API server, `api://`.

A resource contains data and metadata. The MIME type indicates the content format: text, JSON, image, or another type. The name and description help the client and user understand the resource's purpose. This metadata is critically important for proper information processing and presentation.

MCP supports two types of resource content: text and binary. Text content is transmitted directly, while binary content is encoded in base64. This allows working with documents and code, with images, audio, and other media files.

### URI Design for Resources

A URI scheme is an architectural decision, since clients and servers both depend on it. MCP follows the principles of RFC 3986 but adapts them for the needs of AI integrations.

**URI Design Principles:**

Well-designed URIs make the resource system intuitive for both developers and AI models. A predictable URI structure reduces integration friction and simplifies debugging, since a human or model can infer the URI for a resource without consulting documentation. The following principles guide effective URI design in MCP servers.

| Principle | Description | Example |
|-----------|-------------|---------|
| **Hierarchy** | URI reflects data structure | `db://mydb/users/123` |
| **Readability** | URI is human-readable | `file:///docs/readme.md` |
| **Stability** | URI does not change when implementation changes | `api://v1/orders` instead of `api://server-2/orders` |
| **Predictability** | URI pattern can be guessed | `github://owner/repo/issues/42` |

**Scheme Recommendations:**
- `file://`: for the local file system
- `db://`: for databases (tables, records, queries)
- `api://`: for REST/GraphQL endpoints
- `git://`: for Git repositories
- Custom schemes for specific domains: `jira://`, `slack://`, `notion://`

**URI Templates (RFC 6570).** MCP uses URI template syntax for resource templates:
- `{var}`: simple substitution
- `{+var}`: reserved expansion (preserving `/`)
- `{#var}`: fragment expansion
- `{?var}`: query parameter

Example: `db://tables/{tableName}/rows/{+rowId}` allows addressing any row in any table.

**URI Normalization.** For correct URI comparison, MCP recommends normalization: converting scheme and host to lowercase, removing redundant slashes, decoding percent-encoded characters where safe.

### Resource Templates

Resource Templates play a special role in the resource system. Instead of enumerating all possible resources, which may be impossible for dynamic data sources: the server declares URI templates with parameters.

A template describes a family of resources with a shared structure. For example, the template `db://tables/{tableName}/rows/{rowId}` describes access to any row in any table in a database. The client can substitute specific parameter values to obtain the desired resource.

This concept is borrowed from web development, where URL templates are widely used in REST APIs and routing. Using a familiar model lowers the entry barrier and simplifies integration.

### Subscriptions and Notifications

MCP provides a mechanism for subscribing to resource changes, useful for monitoring log files, tracking database changes, or observing system metrics.

The 2026-07-28 specification reworked how those notifications arrive. The HTTP GET endpoint and the `resources/subscribe` and `resources/unsubscribe` methods are gone, replaced by `subscriptions/listen`: a single long-lived POST-response stream. The client opts into specific notification types (`toolsListChanged`, `promptsListChanged`, `resourcesListChanged`, `resourceSubscriptions`), the server acknowledges, and each notification carries an `io.modelcontextprotocol/subscriptionId` tag. Notifications scoped to one request, such as `notifications/progress`, still travel on that request's own response stream rather than the shared one.

The subscription mechanism keeps MCP from being purely request-response: a server can inform a client about a change without waiting to be asked. What changed is that one stream now carries every subscription instead of one endpoint per resource.

## Tools: Performing Actions

### The Nature of Tools

If resources represent passive data access, then Tools embody active interaction with the world. A tool is an executable function that the model can invoke to perform a specific action.

The philosophy of tools in MCP is closely tied to the concept of function calling in modern language models. The model analyzes available tools, their descriptions and parameters, and then decides to invoke the appropriate tool to solve the user's task.

It is critically important to understand that tools are intended to be invoked by the model, not directly by a human. This means tool descriptions must be understandable to AI, not just to programmers. A good tool description explains what it does and when it should be used.

### Tool Parameterization

Each tool has a parameter schema described in JSON Schema format. This schema defines what arguments the tool accepts, their types, constraints, and default values.

JSON Schema provides a rich language for describing data structures. You can specify required and optional fields, enumerations of allowed values, numeric ranges, string patterns, and much more. The more precisely the schema is described, the better the model understands how to use the tool.

Modern language models handle schema interpretation and valid argument generation well. However, the quality of the result directly depends on the quality of the description. A tool with a vague description and minimal schema will be used incorrectly.

### Call Semantics

A tool invocation in MCP is a synchronous operation with clearly defined semantics. The client sends a request with the tool name and arguments, the server performs the action and returns the result.

The invocation result can be successful or erroneous. On success, content is returned: text, data, or an execution confirmation. On error, a problem description is returned that the model can use to adjust its actions.

An interesting feature of MCP is support for the `isError` flag in the invocation result. Even a successfully executed request can return a result marked as an error. This allows the server to signal business logic problems without resorting to protocol errors.

### Annotations and Metadata

Tools in MCP can have annotations that provide additional behavioral information. Annotations related to side effects are especially important.

The `readOnly` annotation indicates that the tool does not modify system state. Such tools are conceptually close to resources but differ in their dynamic nature or the need for parameterization.

The `destructive` annotation warns about dangerous operations: data deletion, irreversible changes, and similar actions. The client can use this information for additional confirmation before execution.

The `idempotent` annotation indicates that repeated invocation with the same arguments will not cause additional changes. This is important for handling connection loss situations and request retransmission.

## Prompts: Interaction Templates

### The Concept of Prompts in MCP

Prompts in MCP are predefined message templates that the server provides for typical usage scenarios. Unlike resources and tools, prompts are intended for use by humans, not by the model.

The idea behind prompts is simple yet powerful. Many tasks require a specific request format to achieve quality results. Instead of each user independently formulating complex prompts, the server can provide ready-made templates.

Consider an example: a code analysis server can provide a "code_review" prompt with parameters for programming language and detail level. The user simply selects the prompt and specifies parameters, receiving a professionally formulated request.

### Prompt Structure

A prompt in MCP consists of a name, description, parameter list, and message generator. Parameters are described similarly to tool parameters: with a name, description, and required flag.

When a prompt is invoked, the server generates a sequence of messages based on the provided parameters. Messages have roles, user or assistant, which allows forming requests and example dialogues.

This structure enables creating complex multi-step prompts with context and examples. The few-shot learning technique, where the model is shown examples of desired behavior, is easily implemented through MCP prompts.

### Dynamic Prompts

Dynamic prompts, which can include resource content, are especially valuable. A code analysis prompt can automatically include the current file from a repository. A documentation generation prompt can pull in existing examples from a knowledge base.

This integration between prompts and resources creates powerful compositional capabilities. An interaction template is combined with current data, forming a context-dependent request.

## Sampling: Reverse Requests to the Model

### The Uniqueness of the Sampling Mechanism

**Sampling is deprecated as of the 2026-07-28 specification** (SEP-2577), with a twelve-month minimum removal window. It remains functional, and new implementations should not adopt it. The suggested migration is to integrate directly with a provider API instead. Read this section to understand existing servers and the design idea; do not build on it.

While resources, tools, and prompts follow the "client requests, server responds" model, Sampling reversed it. The server asked the client to perform a language model call, so the server could use AI capabilities without direct access to a model, delegating the reasoning to whoever held the connection.

Why does this matter? Imagine an MCP server that needs to analyze text, generate a response, or make a decision based on unstructured data. Without Sampling, the server would have to integrate directly with a language model API, manage authentication, handle limits and billing.

With Sampling, the server simply formulates a request and receives a response through the client, which already has a configured connection to the model. This creates an elegant architecture where AI capabilities become a reusable resource.

### Control and Security

The Sampling mechanism includes important control aspects. The client has full control over which Sampling requests to allow. It can modify prompts, filter unwanted requests, or entirely reject Sampling from untrusted servers.

This is a "human-in-the-loop" architecture where the client is an intermediary between the server and the model. No request to the model can pass without the client's knowledge and consent.

Sampling parameters include the model, maximum token count, temperature, and other generation settings. However, the client can override these parameters in accordance with its policies.

### Sampling Applications

Typical Sampling use cases are diverse. Agentic workflows can use Sampling for decision-making at each step. Data analysis systems can delegate result interpretation. Chatbots can use Sampling to generate response variants.

Sampling is especially useful for building complex systems from simple components. Each MCP server focuses on its specialization (data access, action execution) and uses Sampling for intellectual operations.

### Detailed Sampling Request Flow

Understanding the complete Sampling request cycle is important for secure implementation. The process is as follows:

**Server Initiation:** The server sends a `sampling/createMessage` request to the client containing a message array and generation parameters (model, maximum token count, temperature). At this stage, the server formulates what it needs from the language model but has no direct access to it.

**Client Processing:** The client receives the request and performs several critically important validation and modification steps. First, it checks whether the server has the right to perform Sampling according to the agreed-upon capabilities. Then the client can modify the request: change the system prompt for additional security, add context, restrict generation parameters, or even completely block the call.

**Model Interaction:** Only after all checks and modifications does the client perform the actual API call to the language model. This is the single point of contact with the LLM in MCP architecture. The model processes the request and returns a response to the client.

**Filtering and Return:** After receiving the response from the model, the client can filter or modify it once more before sending it to the server. This is the second layer of protection, preventing sensitive information leakage. Finally, the filtered response is sent back to the server in the `SamplingResponse` format.

**Processing Stages on the Client Side:**

The first stage is request reception. The client receives the `sampling/createMessage` message from the server containing all necessary parameters for the model call.

The second stage is validation. The client verifies that the server has the right to perform Sampling according to the capabilities agreed upon during connection establishment. If the server did not declare this capability or was explicitly blocked, the request is rejected.

The third stage is modification. The client applies its security policies: it can change the system prompt, add warnings, limit the token count, lower the temperature for more deterministic responses.

The fourth stage is model invocation. The client performs the actual API call to the LLM (for example, to the Claude API) with modified parameters. This is the only moment when the language model is contacted.

The fifth stage is filtering. The response received from the model is analyzed for sensitive information, inappropriate content, or other policy violations. The client can edit or block the response.

The sixth stage is result return. The final, filtered response is sent to the server as a `SamplingResponse` structure containing the selected model, generated content, and the generation stop reason.

**Sampling Security Model:**

The threat of Prompt Injection from the server is mitigated by the client's full control over the system prompt. This allows it to add instructions that prevent manipulation.

Excessive token consumption is controlled by the client through strict limits on the `maxTokens` parameter, regardless of what the server requests.

Access to sensitive data is prevented by the client filtering responses before passing them to the server. The client can edit or completely block responses containing confidential information.

Uncontrolled invocations are prevented by the client's ability to require explicit user confirmation (human approval) before each Sampling request, especially from untrusted servers.

**Important:** The server never has direct access to the model. The client is the sole intermediary controlling all aspects of the call. This is the fundamental security principle of MCP. This ensures that the user retains full control over the use of AI capabilities.

## Elicitation: Requesting User Input

### The Need for User Interaction

Elicitation addresses a real gap: servers sometimes need information only the user can provide. It arrived in the June 2025 specification as a server-initiated request, and the 2026-07-28 specification changed the mechanism while keeping the need. Read the scenarios below for what Elicitation is for, and the MRTR section for how it now works.

Consider an MCP server connecting to a database. During the connection process, it may need the user to select a specific schema, confirm a destructive migration, or provide a one-time password. Without Elicitation, the server would either have to require all such information upfront in configuration or fail with an error asking the user to restart with additional parameters.

### How Elicitation Works

Until 2026-07-28 the server sent an `elicitation/create` request to the client. That method is gone, along with `notifications/elicitation/complete` and the `elicitationId` correlation field, because the protocol no longer lets a server call into a client at all.

The replacement is **MRTR, Multi Round-Trip Requests** (SEP-2322). When a server cannot finish a request without more information, it returns an `InputRequiredResult` with `resultType: "input_required"`, whose `inputRequests` field carries the questions and their JSON Schemas. The client asks the user, then **retries the original request** with `inputResponses` attached. A server that needs to correlate an out-of-band interaction across retries encodes its own identifier in `requestState`.

The schema still supports text, numeric, boolean confirmations, and enumerated choices, and the client still owns the user experience. The structural change is that the interaction is now a retry loop on one request rather than a callback, which is what allows the whole protocol to stay stateless.

**Security considerations:** The client can reject elicitation requests, modify the presentation, or require user approval before showing them. Servers must not assume elicitation will always succeed, the user may dismiss the request, the client may not support it, or the host policy may prohibit it.

### Practical Elicitation Scenarios

Elicitation is critical for production MCP servers that need to handle ambiguous or sensitive situations:

- **Authentication flows:** A database server needs a one-time password or MFA confirmation before executing a query
- **Destructive operations:** A file management server asks "Delete 47 files matching *.log? [yes/no]" before proceeding
- **Disambiguation:** A search server finds multiple matching entities and asks the user to select the correct one
- **Configuration at runtime:** A deployment server asks which environment (staging/production) to target

Without Elicitation, servers either fail with an error requiring restart with new configuration, or make assumptions that may be wrong. Elicitation keeps the human in the loop for decisions that should not be automated.

### Elicitation vs. Sampling

Elicitation and Sampling serve different purposes. Sampling asks the AI model to generate a response, a machine-to-machine interaction. Elicitation asks the human user for input, a machine-to-human one. Elicitation is for decisions that need human judgment; Sampling is for work that needs model reasoning.

Both used to reverse the request flow, with the server initiating. Neither does now. Sampling is deprecated in favour of integrating directly with a provider API, and both mechanisms route through MRTR, where the server states what it needs and the client comes back with it.

## Tasks: Long-Running Operations

### The Problem of Asynchronous Work

Tasks address long-running operations that cannot return a result inside a single request-response cycle. Before Tasks, MCP tool calls were synchronous and the client simply waited.

Tasks arrived in the November 2025 specification as a core feature. The 2026-07-28 revision moved them out of the core and into the official `io.modelcontextprotocol/tasks` extension (SEP-2663), so a server that only does fast synchronous work never implements them.

Many real-world operations take significant time: running a CI/CD pipeline, processing a large dataset, training a model, generating a complex report, or waiting for external approval. Forcing these into a synchronous model either required long timeouts or artificial workarounds.

### Task Lifecycle

A Task goes through a defined lifecycle. The server creates one when a long-running operation begins and returns a handle. Under the redesigned extension it may return that handle unsolicited, without the client opting in on the request. The client then drives the task: `tasks/get` to poll status, `tasks/update` to send input back, `tasks/cancel` to stop it. The blocking `tasks/result` method was removed in favour of polling, and so was `tasks/list`, so a client tracks the handles it holds rather than asking the server to enumerate them.

Task states include `pending` (created but not started), `running` (in progress), `completed` (finished successfully), `failed` (finished with error), and `cancelled` (aborted by client request).

### Task Capabilities

Tasks support cancellation: the client can request that a running task be stopped. They support progress reporting: the server sends structured updates about completion percentage and current step. They support partial results: intermediate outputs can be returned before the task completes.

This primitive is especially valuable for agentic systems where an agent may launch multiple long-running operations in parallel and needs to monitor their progress, cancel them if the strategy changes, or collect partial results to inform subsequent decisions.

### Tasks and Multi-Agent Coordination

Tasks transform MCP from a synchronous request-response protocol into an asynchronous coordination layer. In multi-agent systems, this enables patterns that were previously difficult:

- **Parallel execution with progress:** An orchestrator agent launches CI builds, test suites, and deployment checks via different MCP servers simultaneously, monitoring each task's progress and making decisions as results arrive
- **Adaptive cancellation:** If an early task reveals a critical error, the agent cancels remaining tasks rather than wasting compute on doomed work
- **Human-in-the-loop for long operations:** Tasks can pause for approval at critical checkpoints (e.g., before deploying to production), resuming when the human approves via Elicitation
- **Partial result streaming:** A data processing task returns intermediate results (e.g., rows processed so far). This allows the agent to start analysis before the full dataset is ready

Tasks, combined with Elicitation for human checkpoints, give MCP servers the building blocks for long-running workflows. Since the 2026-07-28 specification both checkpoints and model calls travel through MRTR results instead of server-initiated requests, and a server that needs a model calls a provider API directly.

## OAuth 2.1 Authorization

### Authentication and Authorization in MCP

The MCP specification includes an OAuth 2.1 authorization framework as a standard mechanism for securing remote server connections. This is particularly important for Streamable HTTP transport, where servers are accessible over the network.

### Key Components

**Protected Resource Metadata (RFC 9728):** Remote MCP servers advertise their authorization requirements through a standardized metadata endpoint. Clients discover what authorization server to use and what scopes are required.

**PKCE (Proof Key for Code Exchange):** All authorization code flows require PKCE, preventing authorization code interception attacks. This is mandatory, not optional.

**Dynamic Client Registration (RFC 7591):** Clients can register with the authorization server dynamically, without pre-registration. It solved the problem of new clients appearing constantly, and the 2026-07-28 specification deprecated it in favour of Client ID Metadata Documents, where the client ID is an HTTPS URL pointing to the client's own metadata.

**Third-Party Authorization:** MCP supports delegated authorization where the authorization server is separate from the MCP server. This allows enterprises to use their existing identity providers (Okta, Auth0, Azure AD) for MCP server access control.

### Authorization Flow

The client discovers the server's authorization requirements via Protected Resource Metadata. It then performs the OAuth 2.1 authorization code flow with PKCE against the designated authorization server. Upon receiving tokens, the client includes the access token in subsequent MCP requests. Token refresh happens transparently when tokens expire.

This standardized approach replaces the ad-hoc authentication patterns (API keys in environment variables, custom token headers) that characterized early MCP deployments.

### Choosing an Authentication Approach

The appropriate authentication mechanism depends on the deployment context: where the server runs, who the clients are, and what compliance constraints apply. Simpler approaches work well for local development, while production remote servers demand standards-based authorization. The following table maps common deployment scenarios to recommended authentication strategies.

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Local stdio server (same machine) | No auth / filesystem permissions | The server runs as a local process; OS-level access control is sufficient |
| Internal remote server (corporate network) | API keys via environment variables | Simple, sufficient when network is trusted; rotate keys regularly |
| Public remote server (SaaS integration) | OAuth 2.1 with PKCE | Standard, user-delegated authorization; tokens are scoped and short-lived |
| Server-to-server (backend services) | OAuth 2.1 client credentials or mTLS | No user interaction; mutual authentication ensures both parties are verified |
| Enterprise with existing IdP | OAuth 2.1 + third-party authorization server | Reuse Okta/Auth0/Azure AD; centralized policy enforcement |

**Practical rule:** Start with the simplest approach that meets your security requirements. Local development servers do not need OAuth. Production remote servers should always use OAuth 2.1. Reserve mTLS for high-security server-to-server scenarios where certificate management infrastructure already exists.

## Component Interaction

### Composability

The true power of the MCP component model manifests in composition. Resources can reference each other. Tools can read resources and create new ones. Prompts can include resources and use tool results.

This composability allows building complex integrations from simple primitives. A database server can provide resources for reading the schema, tools for executing queries, and prompts for typical analytical tasks, all within a single conceptual framework.

### Packaging capability: Agent Plugins

MCP standardizes how an agent reaches a tool. It says nothing about how a capability is distributed, so teams shipping internal capability have been inventing their own layouts: a repository here, an MCP server config there, instructions pasted into a system prompt.

**Agent Plugins 1.0.0**, announced August 6, 2026 by AWS, Cursor, Microsoft, OpenAI, Vercel, and Google, puts those three things in one directory:

- **`plugin.json`** at the root: a required manifest carrying name, version, description, author, and an `extensions` object for client-specific data.
- **`skills/`**: `SKILL.md` files conforming to the separate Agent Skills specification, which supply knowledge and instructions.
- **`mcp.json`** (optional): declares MCP servers over stdio, Streamable HTTP, or the legacy HTTP+SSE transport.

Client-specific extensions are isolated behind a reverse-domain namespace, so one client's additions cannot collide with another's. GitHub added support on August 12 across VS Code, Copilot CLI, the SDK, and the Copilot app.

For a platform team the value is versioning: knowledge, instructions, and tool endpoints ship as one package that attaches to any compatible agent, instead of three artifacts that drift apart.

**Read the version number carefully.** The specification's status was Working Draft as of mid-August 2026, and "1.0.0" describes the format rather than its stability. Several things a plugin ecosystem needs are explicitly outside v1: commands, hooks, agents, rules, and LSP servers were judged too client-specific for a portable contract. Also absent are a marketplace, an installation flow, signing, permission UX, and enterprise approval.

That last group is the part to think about before adopting. **A plugin can carry executable code or introduce a new MCP server, which makes it a supply-chain dependency in the ordinary sense.** It needs provenance, an SBOM, an allowlist, and review of updates, and the format itself provides none of that yet. The LiteLLM compromise covered in Module 14 is the precedent: a widely trusted package in the AI tooling chain, and the breach reached the companies that installed it.

### WebMCP: tools declared by the page

Remote MCP puts a tool behind a network service with its own identity and policy. **WebMCP**, shown by Cloudflare in developer preview on August 6, 2026, does something different: a web page declares its own tools through the experimental `document.modelContext` browser API, and the tool executes inside the page's origin using the current browser session.

The gain is precision. An agent working through a page today reconstructs intent from the DOM, screen coordinates, and screenshots, which costs vision tokens and breaks whenever the markup changes. A declared tool states what the page can do and what arguments it takes, so neither cost applies.

Keep the two straight, because they have different security properties. WebMCP lives beside the page and inherits the user's session, so a tool call runs with whatever the logged-in user can do. Remote MCP is a separate service with its own identity, its own credentials, and its own policy enforcement point. The browser API is still experimental, which makes WebMCP a thing to watch rather than a contract to build production on.

### Design Recommendations

When designing an MCP server, it is important to properly distribute functionality among components. General recommendations are as follows:

Use resources for static or slowly changing data that can be read multiple times. Configuration, documentation, reference data: these are ideal candidates for resources.

Use tools for operations with side effects or dynamic queries with parameters. Create, update, delete, and complex computations. These belong to the domain of tools.

Use prompts to standardize typical scenarios. If users frequently perform similar tasks, a prompt will save their time and ensure quality.

Sampling is deprecated, so treat this as guidance for maintaining existing servers. Each Sampling call is a request to a language model, which can be slow and costly, and new servers that need a model should call a provider API directly.

## Lifecycle and Discovery

### Capabilities and Negotiation

There is no connection establishment left to negotiate during. Since 2026-07-28 the client's capabilities ride in `_meta` on every request, under `io.modelcontextprotocol/clientCapabilities`, and the server's identity comes back in each result's `_meta`. A server must implement `server/discover`, which a client may call once to learn supported protocol versions and capabilities up front, or use as a compatibility probe on stdio.

Both sides also declare extensions. `ClientCapabilities` and `ServerCapabilities` gained an `extensions` field, which is how Tasks, MCP Apps, and Enterprise Managed Authorization are advertised now that they sit outside the core protocol.

### Dynamic Discovery

MCP supports dynamic component discovery. The client can request a list of available resources, tools, or prompts. Each element is accompanied by a description and schema sufficient for use.

The server can notify about changes to the component list. If the server adds a new tool or removes an outdated resource, the client receives a notification and can update its representation.

### MCP Registry and Discovery

The MCP ecosystem has matured rapidly. The official MCP Registry launched in September 2025 with roughly 2,000 verified servers and reached about 9,652 by late May 2026. Counting beyond the registry, over 10,000 community MCP servers exist across GitHub, npm, and PyPI.

**Server Discovery Models:**

The first model is static configuration. The client (for example, Claude Desktop or Claude Code) uses a configuration file such as `claude_desktop_config.json`, where available servers are manually listed with their launch paths and parameters. This remains the most common approach for local stdio servers.

The second model is the MCP Registry. The official registry provides a searchable catalog with server metadata: descriptions, capabilities, versions, dependencies, licenses, and verification status. Clients can query the registry API to discover servers by capability, category, or keyword.

The third approach is package manager distribution. MCP servers are distributed as npm or pip packages with standardized manifests. The `npx` and `uvx` runners allow launching servers without local installation, simplifying adoption.

The fourth model is enterprise registries. Organizations deploy private registries for internal MCP servers, with optional federation to the public registry. This addresses corporate security, compliance, and governance requirements.

**Registry Components:**

Catalog: the central database of servers with metadata, search, and categorization. Verification: a tiered trust system with verified publishers (official, audited), community trusted (reviewed by the community), and unverified (new, experimental). Versioning: semantic versioning with MCP protocol compatibility tracking. Security: automated static analysis, vulnerability scanning, and code signing by trusted publishers.

## Key Takeaways

The MCP component model is a carefully designed architecture for integrating AI systems with the external world. The core primitives (Resources, Tools, and Prompts) cover the fundamental aspects of interaction: data access, action execution, and request templating.

Resources embody the philosophy of passive information access. They are safe, predictable, and ideally suited for providing context to the language model. URI addressing and template support ensure flexibility in data organization.

Tools represent active interaction with side effects. Strict typing through JSON Schema and a rich annotation system allow the model to make informed decisions about tool invocation.

Prompts standardize typical scenarios, lowering the entry barrier for users and ensuring interaction quality.

Sampling and Elicitation both let a server ask for something it could not produce alone, and neither works by calling into the client any more. As of 2026-07-28 Sampling is deprecated in favour of integrating with a provider API directly, and server-initiated requests are replaced by MRTR: the server returns `input_required` with what it needs, and the client retries. Elicitation still covers the case where servers request user input at runtime for decisions requiring human judgment. Tasks support long-running asynchronous operations with progress tracking and cancellation.

OAuth 2.1 authorization provides a standardized security framework for remote MCP connections, replacing ad-hoc authentication patterns.

Agent Plugins 1.0 packages skills, instructions, and MCP server configuration as one versioned unit, which is how capability now travels between teams and clients. Treat an installed plugin as a supply-chain dependency and as a bulk addition to the agent's tool surface: it can carry executable code, and v1 deliberately omits signing, permission UX, and enterprise approval.

WebMCP lets a page declare its own tools instead of forcing an agent to infer them from the DOM or a screenshot. It is experimental, it runs inside the page origin with the user's session, and it is a different security model from remote MCP, which is a separate service with its own identity.

Understanding these components and their proper use is the key to building effective MCP servers that elegantly integrate into the AI ecosystem.
