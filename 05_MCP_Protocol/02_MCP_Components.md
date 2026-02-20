## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[01_MCP_Basics|MCP Basics]]
**Next:** [[03_MCP_Server_Development|MCP Server Development]]

---

# MCP Components: Resources, Tools, Prompts, and Sampling

## Introduction to the MCP Component Model

Model Context Protocol defines a set of primitives for client-server interaction. The three original primitives — Resources, Tools, and Prompts — form the foundation, while Sampling, Elicitation, and Tasks extend the protocol's capabilities.

Resources answer the question "what can be read?", Tools — "what can be done?", Prompts — "how to formulate a request?", Elicitation — "how to ask the user?", Tasks — "how to track long-running work?". This architecture separates concerns between components, similar to the MVC pattern.

## Resources: Data and Content Access

### Resource Philosophy

Resources in MCP represent an abstraction for accessing any data that may be useful to a language model. Conceptually, a resource is a unit of content accessible by a unique URI identifier. This model intentionally resembles web architecture, where each document has its own unique address.

The key property of resources is their passivity. A resource does not perform actions and does not change system state. It simply provides data for reading. This is the fundamental distinction from tools, which can actively interact with the external world.

Why is this separation so important? Consider a situation where an AI agent analyzes business data. Reading a sales report is an operation without side effects that can be performed multiple times without consequences. But submitting an order to a supplier is an action with real consequences that requires explicit confirmation. MCP formalizes this distinction at the protocol level.

### Resource Structure and Identification

Each resource in MCP is identified by a URI — a universal resource identifier. The URI scheme is defined by the server and can be arbitrary, providing flexibility in namespace organization. For example, a file system server might use the `file://` scheme, a database server — `db://`, and an API server — `api://`.

A resource contains not only data but also metadata. The MIME type indicates the content format — text, JSON, image, or another type. The name and description help the client and user understand the resource's purpose. This metadata is critically important for proper information processing and presentation.

Notably, MCP supports two types of resource content: text and binary. Text content is transmitted directly, while binary content is encoded in base64. This allows working not only with documents and code but also with images, audio, and other media files.

### URI Design for Resources

Choosing a URI scheme is not merely a technical decision but an architectural one. MCP follows the principles of RFC 3986 but adapts them for the needs of AI integrations.

**URI Design Principles:**

| Principle | Description | Example |
|-----------|-------------|---------|
| **Hierarchy** | URI reflects data structure | `db://mydb/users/123` |
| **Readability** | URI is human-readable | `file:///docs/readme.md` |
| **Stability** | URI does not change when implementation changes | `api://v1/orders` instead of `api://server-2/orders` |
| **Predictability** | URI pattern can be guessed | `github://owner/repo/issues/42` |

**Scheme Recommendations:**
- `file://` — for the local file system
- `db://` — for databases (tables, records, queries)
- `api://` — for REST/GraphQL endpoints
- `git://` — for Git repositories
- Custom schemes for specific domains: `jira://`, `slack://`, `notion://`

**URI Templates (RFC 6570).** MCP uses URI template syntax for resource templates:
- `{var}` — simple substitution
- `{+var}` — reserved expansion (preserving `/`)
- `{#var}` — fragment expansion
- `{?var}` — query parameter

Example: `db://tables/{tableName}/rows/{+rowId}` allows addressing any row in any table.

**URI Normalization.** For correct URI comparison, MCP recommends normalization: converting scheme and host to lowercase, removing redundant slashes, decoding percent-encoded characters where safe.

### Resource Templates

Resource Templates play a special role in the resource system. Instead of enumerating all possible resources — which may be impossible for dynamic data sources — the server declares URI templates with parameters.

A template describes a family of resources with a shared structure. For example, the template `db://tables/{tableName}/rows/{rowId}` describes access to any row in any table in a database. The client can substitute specific parameter values to obtain the desired resource.

This concept is borrowed from web development, where URL templates are widely used in REST APIs and routing. Using a familiar model lowers the entry barrier and simplifies integration.

### Subscriptions and Notifications

MCP provides a mechanism for subscribing to resource changes. A client can subscribe to a specific resource and receive notifications when it is updated. This is especially useful for monitoring log files, tracking database changes, or observing system metrics.

The subscription mechanism transforms MCP from a simple request-response protocol into a system with support for reactive patterns. The server can proactively inform the client about important changes without waiting for a request.

## Tools: Performing Actions

### The Nature of Tools

If resources represent passive data access, then Tools embody active interaction with the world. A tool is an executable function that the model can invoke to perform a specific action.

The philosophy of tools in MCP is closely tied to the concept of function calling in modern language models. The model analyzes available tools, their descriptions and parameters, and then decides to invoke the appropriate tool to solve the user's task.

It is critically important to understand that tools are intended to be invoked by the model, not directly by a human. This means tool descriptions must be understandable to AI, not just to programmers. A good tool description explains not only what it does but also when it should be used.

### Tool Parameterization

Each tool has a parameter schema described in JSON Schema format. This schema defines what arguments the tool accepts, their types, constraints, and default values.

JSON Schema provides a rich language for describing data structures. You can specify required and optional fields, enumerations of allowed values, numeric ranges, string patterns, and much more. The more precisely the schema is described, the better the model understands how to use the tool.

Modern language models handle schema interpretation and valid argument generation well. However, the quality of the result directly depends on the quality of the description. A tool with a vague description and minimal schema will be used incorrectly.

### Call Semantics

A tool invocation in MCP is a synchronous operation with clearly defined semantics. The client sends a request with the tool name and arguments, the server performs the action and returns the result.

The invocation result can be successful or erroneous. On success, content is returned — text, data, or an execution confirmation. On error, a problem description is returned that the model can use to adjust its actions.

An interesting feature of MCP is support for the `isError` flag in the invocation result. Even a successfully executed request can return a result marked as an error. This allows the server to signal business logic problems without resorting to protocol errors.

### Annotations and Metadata

Tools in MCP can have annotations that provide additional behavioral information. Annotations related to side effects are especially important.

The `readOnly` annotation indicates that the tool does not modify system state. Such tools are conceptually close to resources but differ in their dynamic nature or the need for parameterization.

The `destructive` annotation warns about dangerous operations — data deletion, irreversible changes, and similar actions. The client can use this information for additional confirmation before execution.

The `idempotent` annotation indicates that repeated invocation with the same arguments will not cause additional changes. This is important for handling connection loss situations and request retransmission.

## Prompts: Interaction Templates

### The Concept of Prompts in MCP

Prompts in MCP are predefined message templates that the server provides for typical usage scenarios. Unlike resources and tools, prompts are intended for use by humans, not by the model.

The idea behind prompts is simple yet powerful. Many tasks require a specific request format to achieve quality results. Instead of each user independently formulating complex prompts, the server can provide ready-made templates.

Consider an example: a code analysis server can provide a "code_review" prompt with parameters for programming language and detail level. The user simply selects the prompt and specifies parameters, receiving a professionally formulated request.

### Prompt Structure

A prompt in MCP consists of a name, description, parameter list, and message generator. Parameters are described similarly to tool parameters — with a name, description, and required flag.

When a prompt is invoked, the server generates a sequence of messages based on the provided parameters. Messages have roles — user or assistant — which allows forming not only requests but also example dialogues.

This structure enables creating complex multi-step prompts with context and examples. The few-shot learning technique, where the model is shown examples of desired behavior, is easily implemented through MCP prompts.

### Dynamic Prompts

Dynamic prompts, which can include resource content, are especially valuable. A code analysis prompt can automatically include the current file from a repository. A documentation generation prompt can pull in existing examples from a knowledge base.

This integration between prompts and resources creates powerful compositional capabilities. An interaction template is combined with current data, forming a context-dependent request.

## Sampling: Reverse Requests to the Model

### The Uniqueness of the Sampling Mechanism

Sampling is the most innovative component of MCP. While resources, tools, and prompts follow the "client requests — server responds" model, Sampling reverses this model.

With Sampling, the server requests the client to perform a language model call. This means the server can use AI capabilities without direct access to the model, delegating intellectual operations to the client.

Why does this matter? Imagine an MCP server that needs to analyze text, generate a response, or make a decision based on unstructured data. Without Sampling, the server would have to integrate directly with a language model API, manage authentication, handle limits and billing.

With Sampling, the server simply formulates a request and receives a response through the client, which already has a configured connection to the model. This creates an elegant architecture where AI capabilities become a reusable resource.

### Control and Security

The Sampling mechanism includes important control aspects. The client has full control over which Sampling requests to allow. It can modify prompts, filter unwanted requests, or entirely reject Sampling from untrusted servers.

This is a "human-in-the-loop" architecture where the client acts as an intermediary between the server and the model. No request to the model can pass without the client's knowledge and consent.

Sampling parameters include the model, maximum token count, temperature, and other generation settings. However, the client can override these parameters in accordance with its policies.

### Sampling Applications

Typical Sampling use cases are diverse. Agentic workflows can use Sampling for decision-making at each step. Data analysis systems can delegate result interpretation. Chatbots can use Sampling to generate response variants.

Sampling is especially useful for building complex systems from simple components. Each MCP server focuses on its specialization — data access, action execution — and uses Sampling for intellectual operations.

### Detailed Sampling Request Flow

Understanding the complete Sampling request cycle is important for secure implementation. The process is as follows:

**Server Initiation:** The server sends a `sampling/createMessage` request to the client containing a message array and generation parameters (model, maximum token count, temperature). At this stage, the server formulates what it needs from the language model but has no direct access to it.

**Client Processing:** The client receives the request and performs several critically important validation and modification steps. First, it checks whether the server has the right to perform Sampling according to the agreed-upon capabilities. Then the client can modify the request: change the system prompt for additional security, add context, restrict generation parameters, or even completely block the call.

**Model Interaction:** Only after all checks and modifications does the client perform the actual API call to the language model. This is the single point of contact with the LLM in MCP architecture. The model processes the request and returns a response to the client.

**Filtering and Return:** After receiving the response from the model, the client can additionally filter or modify it before sending it to the server. This is the second layer of protection, preventing sensitive information leakage. Finally, the filtered response is sent back to the server in the `SamplingResponse` format.

**Processing Stages on the Client Side:**

The first stage is request reception. The client receives the `sampling/createMessage` message from the server containing all necessary parameters for the model call.

The second stage is validation. The client verifies that the server has the right to perform Sampling according to the capabilities agreed upon during connection establishment. If the server did not declare this capability or was explicitly blocked, the request is rejected.

The third stage is modification. The client applies its security policies: it can change the system prompt, add warnings, limit the token count, lower the temperature for more deterministic responses.

The fourth stage is model invocation. The client performs the actual API call to the LLM (for example, to the Claude API) with modified parameters. This is the only moment when the language model is contacted.

The fifth stage is filtering. The response received from the model is analyzed for sensitive information, inappropriate content, or other policy violations. The client can edit or block the response.

The sixth stage is result return. The final, filtered response is sent to the server as a `SamplingResponse` structure containing the selected model, generated content, and the generation stop reason.

**Sampling Security Model:**

The threat of Prompt Injection from the server is mitigated by the client's full control over the system prompt, allowing it to add instructions that prevent manipulation.

Excessive token consumption is controlled by the client through strict limits on the `maxTokens` parameter, regardless of what the server requests.

Access to sensitive data is prevented by the client filtering responses before passing them to the server. The client can edit or completely block responses containing confidential information.

Uncontrolled invocations are prevented by the client's ability to require explicit user confirmation (human approval) before each Sampling request, especially from untrusted servers.

**Important:** The server never has direct access to the model. The client is the sole intermediary controlling all aspects of the call. This is the fundamental security principle of MCP, ensuring that the user retains full control over the use of AI capabilities.

## Elicitation: Requesting User Input

### The Need for User Interaction

Elicitation, introduced in the June 2025 MCP specification update, addresses a fundamental gap: servers sometimes need information that only the user can provide. Before Elicitation, a server had no standard way to ask the user a question at runtime.

Consider an MCP server connecting to a database. During the connection process, it may need the user to select a specific schema, confirm a destructive migration, or provide a one-time password. Without Elicitation, the server would either have to require all such information upfront in configuration or fail with an error asking the user to restart with additional parameters.

### How Elicitation Works

The server sends an `elicitation/create` request to the client, specifying a message (the question to present to the user) and a JSON Schema describing the expected response format. The client displays this to the user and returns their input.

The response schema supports text input, numeric input, boolean (yes/no) confirmations, and enumerated choices. The client has full control over the user experience — it can present the elicitation as a dialog, inline prompt, or any appropriate UI element.

**Security considerations:** The client can reject elicitation requests, modify the presentation, or require user approval before showing them. Servers must not assume elicitation will always succeed — the user may dismiss the request, the client may not support it, or the host policy may prohibit it.

### Elicitation vs. Sampling

Elicitation and Sampling serve different purposes. Sampling asks the AI model to generate a response — it is a machine-to-machine interaction. Elicitation asks the human user for input — it is a machine-to-human interaction. Both "reverse" the typical request flow (server initiates instead of client), but they address different needs. Elicitation is for decisions that require human judgment, while Sampling is for tasks that require AI reasoning.

## Tasks: Long-Running Operations

### The Problem of Asynchronous Work

Tasks, introduced in the November 2025 MCP specification update, address long-running operations that cannot return results within a single request-response cycle. Before Tasks, MCP tool calls were synchronous — the client waited for the server to complete the operation and return a result.

Many real-world operations take significant time: running a CI/CD pipeline, processing a large dataset, training a model, generating a complex report, or waiting for external approval. Forcing these into a synchronous model either required long timeouts or artificial workarounds.

### Task Lifecycle

A Task goes through a defined lifecycle: the server creates a task when a long-running operation begins, providing a unique task ID. The client can poll for status or subscribe to updates. The server sends progress notifications with percentage complete, status messages, and partial results.

Task states include: `pending` (created but not started), `running` (in progress), `completed` (finished successfully), `failed` (finished with error), and `cancelled` (aborted by client request).

### Task Capabilities

Tasks support cancellation — the client can request that a running task be stopped. They support progress reporting — the server sends structured updates about completion percentage and current step. They support partial results — intermediate outputs can be returned before the task completes.

This primitive is especially valuable for agentic systems where an agent may launch multiple long-running operations in parallel and needs to monitor their progress, cancel them if the strategy changes, or collect partial results to inform subsequent decisions.

## OAuth 2.1 Authorization

### Authentication and Authorization in MCP

The MCP specification includes an OAuth 2.1 authorization framework as a standard mechanism for securing remote server connections. This is particularly important for Streamable HTTP transport, where servers are accessible over the network.

### Key Components

**Protected Resource Metadata (RFC 9728):** Remote MCP servers advertise their authorization requirements through a standardized metadata endpoint. Clients discover what authorization server to use and what scopes are required.

**PKCE (Proof Key for Code Exchange):** All authorization code flows require PKCE, preventing authorization code interception attacks. This is mandatory, not optional.

**Dynamic Client Registration (RFC 7591):** Clients can register with the authorization server dynamically, without pre-registration. This is essential for the MCP ecosystem where new clients frequently appear.

**Third-Party Authorization:** MCP supports delegated authorization where the authorization server is separate from the MCP server. This allows enterprises to use their existing identity providers (Okta, Auth0, Azure AD) for MCP server access control.

### Authorization Flow

The client discovers the server's authorization requirements via Protected Resource Metadata. It then performs the OAuth 2.1 authorization code flow with PKCE against the designated authorization server. Upon receiving tokens, the client includes the access token in subsequent MCP requests. Token refresh happens transparently when tokens expire.

This standardized approach replaces the ad-hoc authentication patterns (API keys in environment variables, custom token headers) that characterized early MCP deployments.

## Component Interaction

### Composability

The true power of the MCP component model manifests in composition. Resources can reference each other. Tools can read resources and create new ones. Prompts can include resources and use tool results.

This composability allows building complex integrations from simple primitives. A database server can provide resources for reading the schema, tools for executing queries, and prompts for typical analytical tasks — all within a single conceptual framework.

### Design Recommendations

When designing an MCP server, it is important to properly distribute functionality among components. General recommendations are as follows:

Use resources for static or slowly changing data that can be read multiple times. Configuration, documentation, reference data — these are ideal candidates for resources.

Use tools for operations with side effects or dynamic queries with parameters. Create, update, delete, and complex computations — these belong to the domain of tools.

Use prompts to standardize typical scenarios. If users frequently perform similar tasks, a prompt will save their time and ensure quality.

Use Sampling cautiously. Each Sampling call is a request to a language model, which can be slow and costly. Sampling is justified when the server genuinely needs intellectual capabilities.

## Lifecycle and Discovery

### Capabilities and Negotiation

During connection establishment, the client and server exchange information about supported capabilities. The server declares which components it provides: resources, tools, prompts, Sampling.

The client also declares its capabilities — for example, support for resource subscriptions or willingness to handle Sampling requests. This negotiation ensures that both sides understand the available functionality.

### Dynamic Discovery

MCP supports dynamic component discovery. The client can request a list of available resources, tools, or prompts. Each element is accompanied by a description and schema sufficient for use.

Moreover, the server can notify about changes to the component list. If the server adds a new tool or removes an outdated resource, the client receives a notification and can update its representation.

### MCP Registry and Discovery

The MCP ecosystem has matured rapidly. The official MCP Registry launched in September 2025, providing a centralized catalog of approximately 2,000 verified servers. By early 2026, over 10,000 community MCP servers exist across GitHub, npm, and PyPI.

**Server Discovery Models:**

The first model is static configuration. The client (for example, Claude Desktop or Claude Code) uses a configuration file such as `claude_desktop_config.json`, where available servers are manually listed with their launch paths and parameters. This remains the most common approach for local stdio servers.

The second model is the MCP Registry. The official registry provides a searchable catalog with server metadata: descriptions, capabilities, versions, dependencies, licenses, and verification status. Clients can query the registry API to discover servers by capability, category, or keyword.

The third approach is package manager distribution. MCP servers are distributed as npm or pip packages with standardized manifests. The `npx` and `uvx` runners allow launching servers without local installation, simplifying adoption.

The fourth model is enterprise registries. Organizations deploy private registries for internal MCP servers, with optional federation to the public registry. This addresses corporate security, compliance, and governance requirements.

**Registry Components:**

Catalog — the central database of servers with metadata, search, and categorization. Verification — a tiered trust system with verified publishers (official, audited), community trusted (reviewed by the community), and unverified (new, experimental). Versioning — semantic versioning with MCP protocol compatibility tracking. Security — automated static analysis, vulnerability scanning, and code signing by trusted publishers.

## Key Takeaways

The MCP component model is a carefully designed architecture for integrating AI systems with the external world. The core primitives — Resources, Tools, and Prompts — cover the fundamental aspects of interaction: data access, action execution, and request templating.

Resources embody the philosophy of passive information access. They are safe, predictable, and ideally suited for providing context to the language model. URI addressing and template support ensure flexibility in data organization.

Tools represent active interaction with side effects. Strict typing through JSON Schema and a rich annotation system allow the model to make informed decisions about tool invocation.

Prompts standardize typical scenarios, lowering the entry barrier for users and ensuring interaction quality.

Sampling adds a reverse model invocation capability, allowing servers to delegate intellectual operations to the client. Elicitation enables servers to request user input at runtime for decisions requiring human judgment. Tasks support long-running asynchronous operations with progress tracking and cancellation.

OAuth 2.1 authorization provides a standardized security framework for remote MCP connections, replacing ad-hoc authentication patterns.

Understanding these components and their proper use is the key to building effective MCP servers that elegantly integrate into the AI ecosystem.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[01_MCP_Basics|MCP Basics]]
**Next:** [[03_MCP_Server_Development|MCP Server Development]]
