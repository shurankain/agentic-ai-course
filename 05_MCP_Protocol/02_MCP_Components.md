## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[01_MCP_Basics|MCP Basics]]
**Next:** [[03_MCP_Server_Development|MCP Server Development]]

---

# MCP Components: Resources, Tools, Prompts, and Sampling

## Introduction to the MCP Component Model

Model Context Protocol defines three types of primitives for client-server interaction. Each primitive — Resources, Tools, and Prompts — addresses a specific task.

Resources answer the question "what can be read?", Tools — "what can be done?", Prompts — "how to formulate a request?". This architecture separates concerns between components, similar to the MVC pattern.

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

### MCP Registry and Discovery Patterns

As the MCP ecosystem grows, the need for centralized server discovery arises. Although at the time of writing MCP does not have an official registry, patterns and best practices are forming.

**Server Discovery Models:**

The first model is static configuration. The client (for example, Claude Desktop) uses a configuration file such as `claude_desktop_config.json`, where available servers are manually listed with their launch paths and parameters. This is the simplest approach requiring minimal infrastructure but inconvenient for dynamic environments.

The second approach is a local registry through package managers. MCP servers are distributed as npm packages or pip packages with special metadata. The client scans installed packages and automatically discovers MCP servers by their manifests. This approach integrates well with existing development ecosystems.

The third model is network discovery through protocols such as mDNS or DNS-SD. Servers on the local network announce their presence through broadcast messages. The client listens for these announcements and dynamically discovers available servers. This approach is useful for corporate environments with many local services.

The fourth model is a centralized registry, analogous to the npm registry or Docker Hub. Servers are published to a central catalog, and clients search for and download needed servers through an API. This approach provides the best user experience but requires infrastructure and governance.

**Components of a Future MCP Registry:**

Catalog — a central database of available servers with detailed metadata: descriptions, capabilities, versions, dependencies, licenses. Similar to npm or crates.io for Rust.

Search — a powerful search system enabling server discovery by capabilities (tools, resources, prompts), categories (database, filesystem, API), tags, and keywords. Support for filters and sorting by popularity or rating.

Verification — a mechanism for verifying server authenticity and security. May include code signing, community audits, and automated security analysis. Servers can have statuses: verified (checked by the publisher), community (verified by the community), unknown (not verified).

Versioning — management of server versions and their compatibility with different MCP protocol versions. Semantic versioning, backward compatibility policies, migration guides.

Rating — a user review and rating system. Helps evaluate server quality, reliability, and usefulness before installation.

**Open Ecosystem Questions:**

How to verify MCP servers? Possible approaches: cryptographic signatures from trusted publishers, community code audits, automated static analysis, sandboxes for testing.

How to manage trust levels? A gradation is needed: verified publishers (official publishers), community trusted (approved by the community), experimental (not verified). Each level with corresponding warnings and restrictions.

How to handle name conflicts between servers? Use namespaces such as `@vendor/server-name`, similar to npm scoped packages. Or centralized name registration with uniqueness checks.

How to organize federated discovery? Corporations may want private registries for internal servers, but users should also have access to public ones. A priority system and source aggregation are needed.

The community is already creating unofficial catalogs: awesome-mcp-servers on GitHub, integrations into package managers. Anthropic or the community will likely propose an official solution as the ecosystem matures.

## Key Takeaways

The MCP component model is a carefully designed architecture for integrating AI systems with the external world. The three main primitives — Resources, Tools, and Prompts — cover different aspects of interaction: data access, action execution, and request templating.

Resources embody the philosophy of passive information access. They are safe, predictable, and ideally suited for providing context to the language model. URI addressing and template support ensure flexibility in data organization.

Tools represent active interaction with side effects. Strict typing through JSON Schema and a rich annotation system allow the model to make informed decisions about tool invocation.

Prompts standardize typical scenarios, lowering the entry barrier for users and ensuring interaction quality.

Sampling adds a unique reverse model invocation capability, allowing servers to delegate intellectual operations to the client.

Understanding these components and their proper use is the key to building effective MCP servers that elegantly integrate into the AI ecosystem.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[01_MCP_Basics|MCP Basics]]
**Next:** [[03_MCP_Server_Development|MCP Server Development]]
