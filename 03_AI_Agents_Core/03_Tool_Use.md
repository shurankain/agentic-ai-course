# Tool Use

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[02_Agent_Architectures|Agent Architectures]]
**Next:** [[04_Planning|Agent Planning]]

---

## Introduction

A language model on its own is a scholar locked in an ivory tower. It knows many facts, can reason and generate text, but it is cut off from the real world. Its knowledge is frozen at the time of training. It cannot check the current weather, look up a stock price, or send an email.

Tools are the windows and doors that open this tower to the outside world. Each tool gives the agent a new capability: a search tool enables finding up-to-date information, a calculator enables precise computations, an API client enables interaction with external services.

But tools are not merely "additional features." They fundamentally change the nature of the system. With tools, an LLM stops being a passive source of text and becomes an active agent capable of acting in the world.

---

## The Function Calling Concept

### From Text to Structured Calls

Traditionally, language models generate unstructured text. To the question "What is the weather in Kyiv?" a model might respond: "Unfortunately, I do not have access to current weather data." This is an honest but useless answer.

Function Calling (also known as Tool Use) is a mechanism that allows the model to generate structured function calls instead of plain text. The model is trained to recognize situations where an external tool is needed to answer, and to generate the correct call to that tool with the right parameters.

With Function Calling, given the same weather question, the model generates not a text response but a structured request: "call the function get_weather with parameter city='Kyiv'." The system executes this function, obtains the result, and the model formulates the final answer based on the received data.

### Anatomy of an Interaction

Interaction with tools occurs in several stages. First, the user sends a request. The model analyzes it and decides: can it answer from its own knowledge, or is an external tool needed?

If a tool is needed, the model generates a function call. This is not arbitrary text but a structured object: the tool name and arguments in a specified format. The system intercepts this call and passes it to the tool executor.

The executor validates the call, performs the actual operation, and returns the result. This result is passed back to the model as part of the context. The model analyzes the received data and formulates the final answer for the user.

An important point: the model does not execute tools itself. It only generates instructions for calling them. Actual execution happens in a controlled environment where security and correctness of operations can be managed.

### The Bridge Between Language and Action

Function Calling solves a fundamental problem: how to transition from fuzzy natural language to precise programmatic calls. The user says "find information about the latest Apple news" — this is an ambiguous request. What exact search query should be used? How many results are needed? What time range?

A model trained on Function Calling transforms this ambiguity into a specific call with specific parameters. It "understands" the user's intent and translates it into a precise specification.

This works in both directions. A function's result is structured data (JSON, table, text). The model must interpret this data and convert it into a response the user can understand. Not simply output raw data, but make sense of it and present it in the context of the original question.

### How Function Calling Is Implemented in Model Training

To understand the capabilities and limitations of tool use, it helps to know how models are **trained** to use tools.

The ability of models to use tools evolved gradually. Early approaches relied on zero-shot and few-shot prompting — models received tool descriptions in the prompt and "hoped" to understand the format. Quality was low, with many JSON format errors.

The breakthrough came with supervised fine-tuning specifically for tool use. Models were fine-tuned on datasets with examples of correct tool calls. Special tokens were introduced for marking calls and results. This significantly improved accuracy.

Modern models use RLHF (reinforcement learning from human feedback) to optimize tool selection and usage. A reward model evaluates how correctly the model selected a tool and filled in parameters.

**What the model learns to do:**

1. **Tool Selection**: recognize when a tool is needed versus when it can answer directly
2. **Format Compliance**: generate syntactically correct calls in JSON format
3. **Parameter Filling**: extract required arguments from the user's request
4. **Result Interpretation**: understand the execution result and use it in the response

**Practical implications for design:**

Models were predominantly trained on simple, flat tool schemas like `get_weather(city: string)` or `search(query: string, limit: int)`. These schemas therefore work best. Complex nested structures, callbacks, binary data — all of these require more detailed descriptions and examples, since the model saw few such cases in training data.

Training on "good" examples via RLHF makes models conservative — they tend toward safe, proven choices. Unusual use cases require explicit mention in the tool description.

Function calling is a specialized form of instruction following. Models with high instruction-following quality (high IFEval score) typically use tools well, since both abilities are formed through pre-training, supervised fine-tuning, and RLHF.

---

## Anatomy of a Tool

### Four Required Components

Every tool consists of four inseparably linked parts: name, description, parameter schema, and execution function.

**Name** — a unique identifier for the tool. The model uses it when generating a call. The name should be short but informative: `web_search`, `send_email`, `query_database`. Avoid overly generic names (`do_task`) or overly long ones (`search_the_web_for_information`).

A good tool name is self-documenting and reflects its purpose. Use snake_case for compatibility with various LLM APIs. The name must be unique within your toolset to avoid confusion during selection. If you have several similar tools, use prefixes for categorization: `db_query`, `db_insert`, `file_read`, `file_write`.

**Description** — a critically important element that is often underestimated. It is an instruction for the model: when to use this tool, what tasks it is designed for, and what tasks it is not designed for. A good description contains examples of appropriate use and explicit limitations.

The tool description is your primary communication channel with the model. Through it you explain not only what the tool does, but also when to apply it. A poor description leads to the model either ignoring a useful tool, using it incorrectly, or applying it in situations it was not designed for.

An effective description follows this structure: brief purpose (one sentence), typical use cases (2-3 examples), explicit limitations (what the tool does NOT do), important details (input data format, expected output format). Use language close to user queries — this improves semantic matching during tool selection.

**Parameter schema** defines what arguments the tool accepts. This is not just a list of names — it is a full specification with types, descriptions, default values, and constraints. The more precise the schema, the more correctly the model will fill in parameters.

The parameter schema works as a two-sided contract: it helps the model understand what data to provide and simultaneously serves as a basis for validation on the execution side. A well-designed schema minimizes the number of validation errors, saving tokens and time on retries.

For each parameter, specify: the exact data type (string, number, boolean, array, object), a semantic description (not just "city" but "City name in English, e.g., 'Kyiv', 'London'"), whether it is required (required: true/false), a default value if applicable, and constraints (enum for fixed values, minimum/maximum for numbers, pattern for strings).

Enum parameters are especially useful: they explicitly restrict the model's choices to predefined options, preventing typos and incorrect values. Instead of "unit: string" use "unit: enum ['celsius', 'fahrenheit']".

**Execution function** — the actual code that does the work. It receives validated arguments and returns a result. Reliability, security, and predictability are key here.

The execution function should be deterministic to the extent possible: identical inputs should produce identical results. This simplifies debugging and makes agent behavior predictable. Handle all expected errors gracefully, returning clear messages instead of exceptions. The model should receive error information in a format that allows it to either correct the call or try an alternative approach.

Set reasonable timeouts for operations that may run for a long time. If a tool makes an external API request, do not wait indefinitely — limit the wait time and return an informative error on timeout. Log all calls for auditing and debugging, but be careful with sensitive data in logs.

### The Art of Tool Descriptions

A tool description is a prompt for the model. Like any prompt, it requires careful crafting. A poor description causes the model to use the tool incorrectly or not use it at all.

An effective description answers three questions. First: what does the tool do? This should be clear from the first sentence. Second: when should it be used? List typical scenarios. Third: when should it NOT be used? This is often more important — the model needs to know the boundaries of applicability.

For example, for a search tool it is insufficient to write "searches the internet." You need to clarify: for current news — yes, for historical facts — no (the model knows those itself). For checking fresh prices — yes, for basic definitions — no.

Parameter descriptions are equally important. The "query" parameter for search — what should go into it? The user's full question or keywords? In what language? With what level of detail? All of this should be explicitly stated.

When writing descriptions, consider the psychology of a language model. Models respond better to specific examples than to abstract rules. Instead of "use for temporal data" write "use for exchange rates, stock prices, weather." Models are sensitive to wording: the phrase "read-only data" may be ignored, while "never modifies data" is received better.

Description length is a balancing act. A description that is too short does not give the model enough context for correct selection. One that is too long dilutes key information and consumes context tokens. The optimal length is 2-4 sentences for the tool itself and 1 sentence for each parameter.

Test descriptions empirically. Check how the model behaves with your wording on typical queries. If the model systematically selects the wrong tool, the problem is often in the description, not in the model's capabilities.

### Structured Outputs for Tool Calls: Guaranteed JSON

Traditionally, the model generates tool calls as text, which is then parsed. But what if the text is invalid JSON? Or the parameters do not match the schema? The request must be retried, wasting tokens and time.

**Structured Outputs** is a revolutionary solution to this problem. It is a generation mode in which the model **guarantees** valid JSON conforming to a given schema. Not "tries to" — guarantees.

How does this work under the hood? It uses constrained decoding — a technique where at each generation step the model selects only those tokens that do not violate JSON validity. An automaton of valid JSON strings is constructed, and token selection is constrained by this automaton.

OpenAI introduced this technology in August 2024 for GPT-4o. When the `strict: true` mode is enabled for a tool, the model physically cannot generate invalid JSON or omit required fields.

**Advantages:**
- Zero parsing error rate — JSON is always valid
- Zero schema mismatch rate — all fields are correct
- No retry-loop overhead — works on the first attempt
- Predictable behavior — determinism instead of probability

**Limitations:**
- All fields must be required (or nullable) — simplifies constrained decoding
- additionalProperties must be false — strict validation
- Nesting depth restrictions — performance overhead
- Recursive schemas are limited — automaton complexity

Structured Outputs is a paradigm shift from "the model tries to generate correctly" to "the model cannot generate incorrectly." This is especially valuable for production systems where reliability is critical.

---

## Types of Tools

### Information Retrieval Tools

The most common category — tools that fetch data from external sources. Web search, database queries, file reading, API calls.

Their distinguishing characteristic is that they do not change the state of the external world. They only read, never write. This makes them relatively safe: even if used incorrectly, they cannot cause irreversible damage.

However, they have their own risks. Inefficient queries can overload external systems. Careless data handling can lead to leaks of confidential information. It is important to control what data the tool can extract and how it presents that data.

When designing information retrieval tools, consider: the size of returned data (an overly large response will overflow the model's context), data format (structured JSON is better than unformatted text), caching (repeated requests for the same data are wasteful), and rate limiting (protection against overloading data sources).

Information retrieval tools are the foundation of most agents. They allow overcoming the primary limitation of LLMs: outdated knowledge frozen at the time of training. With properly designed information retrieval tools, the agent always has access to current data.

### Action Execution Tools

The second category — tools that change the state of the world. Sending emails, creating files, publishing posts, executing transactions.

These tools require heightened attention to security. An erroneous call can have irreversible consequences: a deleted file cannot be restored, a sent email cannot be recalled.

The typical practice is to require user confirmation before executing critical actions. The agent plans the action, shows it to the user, and executes it only after explicit approval.

A criticality gradient helps decide what requires confirmation. Reading a file — low criticality, executed automatically. Creating a file — medium criticality, may require confirmation depending on policy. Deleting a file — high criticality, always requires confirmation. Financial transactions — critical operation, requires confirmation and additional authorization.

Idempotency is an important property for action tools. An idempotent operation can be executed multiple times with the same result as a single execution. This simplifies error handling and retries. Creating a file with overwrite — idempotent. Appending a line to a file — not idempotent.

### Computation Tools

The third category — tools that perform precise computations. Calculators, statistical analysis, code execution.

Language models are notoriously error-prone in arithmetic. They can incorrectly multiply numbers or make mistakes in complex calculations. Computation tools solve this problem — the model formulates the task, and specialized code executes it.

A special case is tools for executing arbitrary code. They are powerful but dangerous. Code may contain bugs or be malicious. Such tools require isolation (sandboxing) and strict limitations.

Designing computation tools requires a balance between flexibility and security. A calculator for simple mathematical expressions is safe and flexible enough for most tasks. A Python REPL with full library access is maximally flexible but requires serious isolation. A restricted Python interpreter with a whitelist of allowed modules — a trade-off between flexibility and security.

Computation tools are especially useful for data science agents, financial analysts, scientific research — anywhere computational accuracy is critical and the language model alone is insufficiently reliable.

---

## Tool Selection Strategies

### Theory of Tool Selection: How the Model Makes Decisions

Before discussing practical strategies, it is important to understand: **how does the model decide which tool to use?**

When the model "sees" tool descriptions, it processes them as part of the context. Tool selection is essentially a classification task solved by the attention mechanism. The model determines the semantic match between the user's request and each tool's capabilities, then generates the most probable call.

**Factors influencing selection:**

Position in the list matters — the first and last tools are selected more frequently due to positional bias. Place the most important or frequently used tools at the beginning of the list.

Description length and detail affect selection accuracy. More detailed descriptions help the model better understand the purpose, but overly long descriptions dilute key information. Balance is critical.

Semantic similarity between the query and description is the key factor. Use language in descriptions that is close to actual user queries. If users ask "what's the weather," use the word "weather" in the description rather than "meteorological conditions."

Recent use creates recency bias — tools from recent context are selected more frequently. Account for this during long sessions.

Name uniqueness reduces confusion. Avoid similar names like `search`, `search_web`, `web_search` — the model may get confused.

**Typical selection errors:**

Tool confusion arises when two tools have similar descriptions. Symptom: the model systematically selects the wrong tool from a similar pair. Solution: strengthen the differences in descriptions, explicitly state when to use one versus the other.

Capability overestimation — the model overestimates a tool's capabilities. Symptom: calls with parameters the tool does not support. Solution: explicitly list limitations in the description.

No-tool bias — the model ignores tools and tries to answer from its own knowledge. Symptom: outdated information instead of current data. Solution: strengthen usage triggers in the description, explicitly state "use this tool for..."

### The Scale Problem

When an agent has three tools, the choice is obvious. But what happens when there are thirty? Or three hundred?

As the number of tools grows, two problems arise. First — context size. Descriptions of all tools may not fit in the model's context window. Second — selection quality. The more options there are, the harder it is for the model to choose correctly.

Strategies for filtering and prioritizing tools are needed.

### Semantic Tool Search

The idea is simple: instead of showing the model all tools, find the most relevant ones for the current request.

Each tool's description is converted into a vector via an embedding model. When a user request arrives, it is also converted into a vector. Then the tools with the closest vectors are found.

This allows efficiently filtering hundreds of tools down to the few most suitable ones. The model works only with relevant options, which improves selection quality and conserves context.

### Categorization and Hierarchy

An alternative approach is to organize tools into a hierarchy. At the top level — categories: data operations, communications, computations, system operations. Within each category — specific tools.

Selection occurs in two stages. First, the model determines the task category. Then it selects a specific tool from that category. This reduces cognitive load: instead of choosing from a hundred options — choosing from five categories, then from twenty tools.

Categorization also helps with permissions. Access to entire categories can be restricted for certain users or scenarios.

### Adaptive Selection

Advanced systems learn to select tools based on experience. If a particular combination of tools successfully solves a class of tasks, the system remembers this pattern.

When a similar task arrives, the system immediately suggests the proven set of tools. This speeds up resolution and reduces errors.

This approach requires feedback mechanisms: how does the system know the selection was successful? Typically, a combination of explicit signals (user approved the result) and implicit signals (task completed without errors) is used.

### Parallel Tool Calls: When You Can and When You Cannot

Modern LLM APIs (GPT-4, Claude, Gemini) support **parallel tool calls** — the model can generate multiple tool calls in a single response. This significantly speeds up execution but requires understanding dependencies.

**Types of dependencies between tools:**

Data Dependency — when one tool needs the output of another. For example, `search_user(email)` → `get_orders(user_id)`. The second call requires user_id from the first; parallel execution is not possible.

Temporal Dependency — when execution order matters for the result. For example, `create_file(path)` → `write_to_file(path, content)`. The file must exist before writing; sequential execution is required.

Resource Dependency — when tools compete for the same resource. For example, two calls to `update_config(key, value)` can create a race condition. Caution is required.

No Dependency (independent tools) — when tools are completely independent. For example, `get_weather("Kyiv")`, `get_stock_price("AAPL")`, `search_news("AI")` access different sources with no overlap. They can be safely executed in parallel.

**How the model decides about parallelism:**

Models are trained to recognize independent requests. For the question "What is the weather in Kyiv and the current Bitcoin price?" the model will generate two parallel calls. For the question "Find the user by email and show their orders," the model will execute sequentially — first the search, then the order query with the obtained user_id.

**Practical recommendations:**

Allow parallelism for obviously independent requests and read-only operations without overlap. Execute write operations on the same resource sequentially. When dependencies are unclear, be conservative — sequential is better than risking a race condition. For high-latency tools (web APIs), maximize parallelism to reduce overall execution time.

---

## Execution and Error Handling

### Executor as a Protective Layer

Between the model and actual tool execution sits a component called the Executor or Tool Runner. Its job is not simply to call a function but to ensure safe and predictable execution.

The Executor verifies that the requested tool exists and is available. It validates arguments against the tool's schema. It sets a timeout — a tool cannot run indefinitely. It catches exceptions and converts them into clear error messages.

The Executor can also log all operations for auditing, check user permissions, and limit call frequency (rate limiting).

The Executor acts as a circuit breaker for tools. If a particular tool systematically fails, the Executor can temporarily disable it, preventing cascading errors. This is especially important when interacting with external APIs that may be unstable.

Execution metrics are collected by the Executor automatically: latency of each call, success rate, error types, result sizes. These metrics are critical for monitoring system health and optimizing performance.

### Graceful Degradation

Tools can fail. API unavailable, file not found, request exceeded a limit. The agent must be able to handle failures.

The first level of defense — retries. Many errors are temporary: network timeout, overloaded server. Several retries with exponential backoff often resolve the issue.

Exponential backoff is critical for avoiding overloading an already struggling service. First attempt immediately, second after 1 second, third after 2 seconds, fourth after 4 seconds. With the addition of jitter (random variation) to prevent the thundering herd problem.

The second level — alternative tools. If one search API is unavailable, try another. If a file cannot be read directly, perhaps a copy exists.

The fallback tool strategy should be explicitly defined in the system configuration. For each critical tool, specify a list of alternatives in order of preference. The model should not decide which alternative to use on its own — this decision is made by the Executor based on predefined rules.

The third level — strategy change. Perhaps the task can be solved without this tool. The model must be able to adapt its approach when resources are unavailable.

Adaptability requires the model to receive an informative error message. Not just "tool unavailable" but "the search tool is temporarily unavailable, try formulating an answer based on your knowledge or suggest the user try again later." Such a message gives the model context for making a reasonable decision.

### Result Formatting

The result of tool execution needs to be passed to the model for further processing. But results vary: JSON object, table, long text, binary data.

The model works best with text of moderate length. If the result is too large, it needs to be trimmed. If too structured, converted into a readable format.

Smart truncation preserves important information. For JSON — structure and key fields. For tables — headers and representative rows. For long text — beginning, end, and an indication of omitted middle content.

Truncation strategies depend on data type. For search result lists: show the first 3-5, add "N more results omitted." For JSON objects: show the first level in full, truncate nested objects. For text: first 500 characters, last 200 characters, indication of omitted middle section.

Some results require format conversion. Tabular data is better presented as a markdown table than CSV. HTML needs to be stripped of tags while preserving structure. XML should be converted to more readable JSON.

It is also important to include meta-information about the result: original data size, how much information was omitted, data retrieval timestamp, and data source. This helps the model correctly interpret the result and formulate an accurate answer for the user.

---

## Tool Security

### Principle of Least Privilege

Each tool should have only the permissions necessary for its operation. A search tool should not have access to the file system. A file-reading tool should not be able to delete files.

This limits damage in case of errors or attacks. Even if an attacker manages to manipulate the agent, they will be limited to the capabilities of available tools.

In practice, this means creating separate tools for different operations instead of a single "super-tool." A database read tool is separated from a database write tool. A file viewing tool is separated from a file modification tool.

The granularity of privilege separation is a critical design decision. Too coarse a separation (one tool for all file operations) leaves broad potential for abuse. Too fine a separation (a separate tool for each operation) complicates the system and makes model selection harder.

A reasonable approach is to group operations by risk level. Read operations can be in a single tool since they do not change state. Write operations are separated by object: creating files, modifying files, deleting files — three different tools with different permission levels.

Use Role-Based Access Control (RBAC) for tools. Define user roles (viewer, editor, admin) and assign each role a permitted set of tools. An agent operating on behalf of a viewer physically does not have access to modification tools, even if the model attempts to call them.

### Sandboxing for Code Execution

Tools that execute arbitrary code require special isolation. Code may contain infinite loops, excessive memory consumption, or attempts to access system resources.

The typical solution is execution in an isolated container with restrictions: CPU and memory limits, disabled networking, read-only file system, and execution timeout.

A container is created for each execution and destroyed afterward. This guarantees that one call cannot affect another.

Modern approaches to sandboxing include multiple layers of protection. The first layer — containerization (Docker, containerd) isolates the process from the host system. The second layer — virtual machines (gVisor, Firecracker) add an additional isolation layer with minimal overhead. The third layer — kernel-level restrictions (seccomp, AppArmor) block dangerous system calls.

For interpreted languages (Python, JavaScript), specialized sandboxing libraries exist that restrict available modules and functions. For example, RestrictedPython for Python allows prohibiting imports of dangerous modules (os, subprocess, socket) and restricting access to builtins.

It is critically important to set resource limits: maximum execution time (typically 30-60 seconds), maximum memory (256-512 MB for typical tasks), maximum output size (prevents memory bombs via stdout), and prohibition on creating child processes.

### Input Validation

The model may generate incorrect or malicious calls — intentionally (during prompt injection) or accidentally. Tools must defend themselves.

A SQL tool must block modifying queries if only reads are permitted. A file tool must verify that the path does not escape the allowed directory (path traversal attack). An email tool must validate recipient addresses.

Validation occurs at the Executor level before passing to the execution function. Any suspicious call is blocked with a clear error message.

Types of validation by priority: schema validation (JSON Schema conformance — this is the minimum, always performed automatically), semantic validation (values make sense: date not in the future, email is valid, number is in a reasonable range), security validation (no SQL injection, path traversal, or command injection patterns), and business validation (user has the right to this operation, limits are not exceeded).

Special attention should be paid to string parameters used in dangerous contexts. File paths must undergo canonicalization and boundary escape checking. SQL parameters must be used via prepared statements, never through string concatenation. Shell commands should not be constructed from user input without strict validation — or better yet, not at all.

### Audit and Monitoring

All tool calls must be logged: who called, when, with what parameters, and what result was received. This is necessary for debugging, incident analysis, and anomaly detection.

A monitoring system can identify suspicious patterns: too many failed calls, unusual parameters, access to sensitive resources. Upon detecting an anomaly, an alert is sent to the security team.

It is also important to log what the model "saw" in the call result. If the tool returned confidential data, this needs to be known for risk assessment.

Structured logging is critical for effective auditing. Each log entry should contain: timestamp with millisecond precision, user/session ID for attribution, tool name and version (for tracking changes), full input parameters (with sensitive data masking), execution result (success/error), execution latency, and correlation ID for linking to the call chain.

Logs should be immutable (write-only, append-only) to prevent tampering. Use a centralized logging system (ELK stack, Splunk, CloudWatch) for aggregation and analysis. Set up automatic alerts for anomalies: spike in error count, unusual access patterns, attempts to call non-existent tools, and repeated validation errors (a sign of prompt injection attempts).

Conduct regular audit reviews of logs. Look for abuse patterns: path traversal attempts, SQL injection in parameters, unusual temporal patterns (calls during off-hours), and anomalous data volumes. Use machine learning for detecting anomalies in tool usage patterns.

---

## Integration with LangChain4j

### Declarative Approach with Annotations

LangChain4j offers a convenient way to create tools through annotations. An ordinary Java method becomes a tool by adding the `@Tool` annotation. Method parameters automatically become tool parameters.

This approach minimizes boilerplate code. There is no need to create classes, implement interfaces, or describe schemas manually. The framework extracts all necessary information from the method signature and annotations.

The `@P` annotation on parameters adds descriptions that the model uses for correct argument filling. The better these descriptions are, the more accurately the model will call tools.

Advantages of the annotation approach: minimal boilerplate (focus on logic rather than infrastructure), automatic JSON Schema generation from Java types, type safety (the compiler checks types), and ease of testing (tools are ordinary Java methods).

Annotations in LangChain4j also support additional parameters: description for a detailed tool description, name for explicitly specifying the name (if the method name is unsatisfactory), and return value descriptions for explaining the result format.

### Automatic Integration

When creating an agent via `AiServices.builder()`, tools are registered automatically. The framework analyzes the object with annotated methods, extracts schemas and descriptions, and formats them into a format the model understands.

When the model generates a tool call, the framework automatically finds the corresponding method, converts arguments from JSON to Java types, calls the method, and formats the result back.

The developer can focus on the business logic of tools without being distracted by infrastructure details.

LangChain4j handles edge cases automatically: exceptions inside tool methods are caught and formatted into error messages understandable to the model, null values are handled correctly (if the parameter is nullable), and collections and complex objects are serialized to JSON automatically.

The framework also supports advanced scenarios: async tools (methods can return CompletableFuture), streaming results (for long-running operations), and context injection (access to chat memory or user context inside a tool).

Integration with Spring Boot is especially convenient — tools can be Spring beans with dependency injection, which simplifies access to services, repositories, and other application components.

---

## Model Context Protocol (MCP): Standardized Tool Interface

### The Problem MCP Solves

Before MCP, every agent framework defined tools differently. LangChain had its tool format, OpenAI had function calling schemas, Anthropic had its own tool_use format, and custom agents used bespoke interfaces. An MCP server built once works with every MCP-compatible host — Claude Code, Cursor, Windsurf, Zed, custom agents, and more.

### MCP Architecture

MCP follows a client-server architecture:

**MCP Host** — the application running the agent (IDE, CLI tool, custom app). It manages one or more MCP clients.

**MCP Client** — maintains a 1:1 connection with an MCP server. Handles protocol negotiation, capability discovery, and message transport.

**MCP Server** — exposes capabilities to the client through three primitives:
- **Tools** — executable functions the model can call (equivalent to function calling)
- **Resources** — data the model can read (files, database records, API responses)
- **Prompts** — reusable prompt templates for specific tasks

**Transport:** MCP supports two transport mechanisms — **stdio** (for local servers, spawned as child processes) and **Streamable HTTP** (for remote servers, over HTTP with optional SSE streaming).

### MCP Tools vs Native Function Calling

| Aspect | Native Function Calling | MCP Tools |
|--------|------------------------|-----------|
| **Definition** | Per-provider JSON schema | Standardized MCP schema |
| **Discovery** | Static, defined at request time | Dynamic, discovered at runtime |
| **Reusability** | Framework-specific | Cross-framework, cross-host |
| **Ecosystem** | Provider-locked | Open ecosystem of servers |
| **Transport** | In-process | Local (stdio) or remote (HTTP) |

### MCP and Tool Security

MCP introduces specific security considerations (see [[../14_Security_Safety/03_Agent_Security|Agent Security]] for details):

- **Tool descriptions from untrusted MCP servers are a prompt injection vector.** A malicious server can craft tool descriptions that manipulate the agent's behavior.
- **Trust levels matter:** first-party servers (high trust), verified third-party (medium trust), unverified (low trust, strict sandboxing required).
- **OAuth 2.1 with PKCE** for credential delegation — agents obtain scoped, short-lived tokens instead of receiving raw user credentials.

### Practical Impact

MCP has become the de facto standard for agent tool integration in 2025. Major adopters include: Anthropic (Claude Code, Claude Desktop), Cursor, Windsurf, Zed, Sourcegraph, and numerous custom agent frameworks. The ecosystem has grown to thousands of community-built MCP servers providing access to databases, APIs, SaaS tools, and local system capabilities.

For tool designers, MCP means: build your tool as an MCP server once, and it works everywhere. For agent architects, MCP means: access to a vast ecosystem of pre-built tools without custom integration work.

## Tool-Integrated Reasoning (2024-2025)

Reasoning models (o1, o3, o4-mini, Claude extended thinking, DeepSeek R1) introduce a new paradigm: **tool use as part of the reasoning process**, not just as an execution step.

### Traditional vs Reasoning-Integrated Tool Use

**Traditional agent loop:** Think → Select tool → Execute → Observe → Think → ... The model alternates between reasoning and tool use in discrete steps.

**Reasoning-integrated tool use:** The model plans multiple tool calls as part of a single extended reasoning chain. It reasons about which tools to call, in what order, how to combine results — all within the thinking phase before producing any output.

OpenAI's o3 and o4-mini can interleave tool calls within their chain-of-thought reasoning. The model might: reason about the problem → call a search tool → reason about the results → call a calculator → reason about the combined information → produce a final answer — all as a single integrated reasoning trace.

### Implications for Tool Design

**Fewer, more powerful tools:** Reasoning models are better at composing simple tools into complex workflows. Instead of creating a specialized `analyze_sales_data` tool, provide `query_database` and `calculate` — the reasoning model will compose them correctly.

**Better error recovery:** When a tool call fails mid-reasoning, the model can adjust its plan within the same reasoning chain rather than requiring a separate retry loop.

**Reduced need for explicit orchestration:** Traditional agents need frameworks (LangGraph, CrewAI) to manage complex multi-tool workflows. Reasoning models can often handle the orchestration internally through extended thinking.

### When to Use Reasoning-Integrated vs Traditional Tool Use

**Reasoning-integrated tool use works best when:** the task requires combining information from multiple tools in a single coherent analysis, error recovery needs to happen within the reasoning flow, and the number of tool calls is moderate (5-15 per task).

**Traditional ReAct loops work best when:** the task involves many sequential tool calls with unpredictable branching, external state changes between calls matter (e.g., database writes), or fine-grained observability of each step is required for compliance or debugging.

**Token cost comparison:** A traditional ReAct loop over 10 tool calls might require 10 separate LLM calls (each with growing context). A reasoning model can plan and execute the same 10 calls within a single extended reasoning pass — fewer total tokens but higher per-token cost for thinking tokens. For tool-heavy workflows with 20+ calls, the traditional approach with a cheaper model may be more economical.

### Tool Schema Design for Reasoning Models

Reasoning models benefit from **descriptive tool schemas**: detailed parameter descriptions, explicit constraints, and examples of valid inputs. The model uses this information during its reasoning phase to plan tool calls more accurately, reducing failed calls and retries. Invest in schema quality over quantity — fewer well-documented tools outperform many poorly described ones.

## Key Takeaways

1. **Tools transform an LLM into an agent** capable of acting in the real world. Without them, the model is a closed system; with them, it is an active participant in external processes.

2. **The tool description is more important than the code**. The model makes decisions based on the description. Poor description = incorrect usage.

3. **The parameter schema is a contract**. The more precise the schema, the fewer errors during calls. Use types, enums, and default values.

4. **Scaling requires a strategy**. With a large number of tools, semantic search, categorization, or hierarchical selection is needed.

5. **The Executor is a protective layer**. It validates calls, sets timeouts, handles errors, and logs operations.

6. **Security is a priority**. Least privilege, sandboxing for code, input validation, and auditing of all operations.

7. **Error handling must be graceful**. Retries, alternative tools, strategy adaptation — the agent must be able to handle failures.

8. **MCP is the standard for tool integration (2025)**. Build tools as MCP servers for cross-framework compatibility. Be aware of security implications — tool descriptions from untrusted servers are an attack vector.

9. **Reasoning models change tool use patterns**. Tool-integrated reasoning enables planning and execution within a single thinking chain, reducing the need for explicit orchestration frameworks.

---

## Practical Code Examples

Below is a minimal but complete example of a search tool demonstrating all key components: name, description with applicability boundaries, detailed parameter schema with enum values, and safe execution with error handling.

**Example of a search tool structure:**

Consider the organization of a typical web search tool. The WebSearchTool class represents a full-featured tool with all necessary components.

The getName method returns the tool's string identifier — in this case "web_search". This name is used by the model when generating calls.

The getDescription method provides a detailed description of the tool's purpose. The description explicitly states what the tool is intended for: searching for current information, fresh news, current prices, and events after the model's training date. The description also contains explicit limitations — the tool should NOT be used for historical facts or stable documentation that the model already knows.

The getParametersSchema method defines the tool's parameter structure. The schema is built via the builder pattern and includes three parameters:

The first parameter "query" has type "string" and is marked as required (required: true). Its description hints to the model: "Search query. Be specific and concise." This guides the model toward forming precise queries.

The second parameter "num_results" has type "integer" and a description "Number of results (1-10)". It has a default value of 3, meaning: if the model does not explicitly specify a count, three results will be used.

The third parameter "time_range" demonstrates the use of enum values. The type is "string", but possible values are restricted to: "day", "week", "month", "year", "all". The default value is "all". Using an enum prevents typos and incorrect values from the model.

The execute method accepts a map of arguments and performs the actual work. First, parameters are extracted from the map: query as a string and num_results with a fallback to the default value of 3. Then, in a try block, the external search API (searchApi.search) is called with these parameters. If the search succeeds, results are formatted and returned as ToolResult.success. If an exception occurs, it is caught and ToolResult.error is returned with a clear error message. This is graceful error handling — the model receives an informative message instead of a raw exception.

**Using LangChain4j annotations:**

LangChain4j provides a declarative approach through annotations that significantly simplifies tool creation. Instead of explicitly defining schemas and metadata, you simply annotate ordinary Java methods.

Consider an example of a Tools class with a computation method. The calculate method has the @Tool annotation with the description "Calculates mathematical expressions." The method parameter — the string expression — is annotated with @P with the description "Expression to evaluate." The method returns a double — the computation result obtained through the expression parser ExpressionParser.evaluate.

Tool registration occurs when building the agent via AiServices.builder. A builder is created for the Assistant interface, the language model is specified via chatLanguageModel(model), and an instance of the Tools class is passed via the .tools(new Tools()) method. Calling build() completes the configuration and creates the ready agent.

The framework automatically discovers all annotated methods, extracts names, descriptions, parameter types from them, and creates corresponding schemas. When the model generates a call to the calculate tool with some expression, the framework automatically routes the call to the corresponding method, performs JSON argument to Java type conversion, and returns the result back to the model. The developer simply writes business logic, and all infrastructure is generated automatically.



---

## Navigation
**Previous:** [[02_Agent_Architectures|Agent Architectures]]
**Next:** [[04_Planning|Agent Planning]]

---

## Practical Code Examples

Below are full implementation examples of tools in Java using LangChain4j. The examples demonstrate three categories of tools: information retrieval, database operations, and file operations.

### Example 1: Web Search Tool

```java
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.P;

public class WebSearchTools {

    private final SearchApiClient searchClient;

    public WebSearchTools(SearchApiClient searchClient) {
        this.searchClient = searchClient;
    }

    @Tool("Searches for up-to-date information on the internet. Use for fresh news, " +
          "current prices, events after 2024. Do NOT use for historical facts.")
    public String searchWeb(
            @P("Search query in English, specific and concise")
            String query,

            @P("Number of results to return (1-10)")
            int maxResults) {

        try {
            // Input parameter validation
            if (query == null || query.trim().isEmpty()) {
                return "Error: search query cannot be empty";
            }

            if (maxResults < 1 || maxResults > 10) {
                maxResults = 3; // Default value
            }

            // Execute search with timeout
            List<SearchResult> results = searchClient.search(query, maxResults);

            // Format results for the model
            StringBuilder response = new StringBuilder();
            response.append(String.format("Found %d results for query '%s':\n\n",
                                         results.size(), query));

            for (int i = 0; i < results.size(); i++) {
                SearchResult result = results.get(i);
                response.append(String.format("%d. %s\n", i + 1, result.getTitle()));
                response.append(String.format("   URL: %s\n", result.getUrl()));
                response.append(String.format("   %s\n\n", result.getSnippet()));
            }

            return response.toString();

        } catch (TimeoutException e) {
            return "Error: search API response timed out. " +
                   "Try rephrasing the query.";
        } catch (ApiException e) {
            return "Error: search service is temporarily unavailable. " +
                   "Try using your own knowledge.";
        } catch (Exception e) {
            // Logging for debugging
            logger.error("Unexpected error during search: {}", e.getMessage());
            return "An unexpected error occurred while performing the search.";
        }
    }
}
```

### Example 2: Database Tool

```java
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.P;
import java.sql.*;
import java.util.*;

public class DatabaseTools {

    private final DataSource dataSource;
    private final Set<String> allowedTables = Set.of("users", "orders", "products");

    public DatabaseTools(DataSource dataSource) {
        this.dataSource = dataSource;
    }

    @Tool("Executes a SQL SELECT query against the database. Read-only access ONLY. " +
          "Available tables: users, orders, products. INSERT/UPDATE/DELETE queries are forbidden.")
    public String queryDatabase(
            @P("SQL SELECT query. Use only SELECT, no data modifications")
            String sqlQuery) {

        try {
            // Security validation: only SELECT queries allowed
            String normalizedQuery = sqlQuery.trim().toLowerCase();
            if (!normalizedQuery.startsWith("select")) {
                return "Error: only SELECT queries are allowed. " +
                       "Modifying operations (INSERT/UPDATE/DELETE) are forbidden.";
            }

            // Check for usage of allowed tables
            boolean containsAllowedTable = allowedTables.stream()
                .anyMatch(table -> normalizedQuery.contains(table));

            if (!containsAllowedTable) {
                return String.format("Error: query must use one of the allowed tables: %s",
                                   String.join(", ", allowedTables));
            }

            // Execute query with time limit
            try (Connection conn = dataSource.getConnection();
                 Statement stmt = conn.createStatement()) {

                stmt.setQueryTimeout(30); // 30 seconds maximum

                // Limit row count to prevent context overload
                String limitedQuery = sqlQuery;
                if (!normalizedQuery.contains("limit")) {
                    limitedQuery += " LIMIT 50";
                }

                try (ResultSet rs = stmt.executeQuery(limitedQuery)) {
                    return formatResultSet(rs);
                }
            }

        } catch (SQLTimeoutException e) {
            return "Error: query is taking too long to execute. " +
                   "Try simplifying the conditions or adding indexes.";
        } catch (SQLException e) {
            return String.format("SQL error: %s. Check the query syntax.",
                               e.getMessage());
        } catch (Exception e) {
            logger.error("Error executing database query: {}", e.getMessage());
            return "An error occurred while accessing the database.";
        }
    }

    private String formatResultSet(ResultSet rs) throws SQLException {
        ResultSetMetaData metaData = rs.getMetaData();
        int columnCount = metaData.getColumnCount();

        StringBuilder result = new StringBuilder();

        // Column headers
        for (int i = 1; i <= columnCount; i++) {
            result.append(metaData.getColumnName(i));
            if (i < columnCount) result.append(" | ");
        }
        result.append("\n");
        result.append("-".repeat(50)).append("\n");

        // Data (maximum 50 rows)
        int rowCount = 0;
        while (rs.next() && rowCount < 50) {
            for (int i = 1; i <= columnCount; i++) {
                result.append(rs.getString(i));
                if (i < columnCount) result.append(" | ");
            }
            result.append("\n");
            rowCount++;
        }

        result.append(String.format("\n(Rows displayed: %d)", rowCount));
        return result.toString();
    }
}
```

### Example 3: File Tools

```java
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.P;
import java.nio.file.*;
import java.io.IOException;

public class FileTools {

    private final Path allowedDirectory;

    public FileTools(String allowedDirectoryPath) {
        this.allowedDirectory = Paths.get(allowedDirectoryPath).toAbsolutePath().normalize();
    }

    @Tool("Reads the contents of a text file. Read-only, does not modify the file.")
    public String readFile(
            @P("Relative path to the file within the allowed directory")
            String relativePath) {

        try {
            // Protection against path traversal attacks
            Path requestedPath = allowedDirectory.resolve(relativePath).normalize();

            if (!requestedPath.startsWith(allowedDirectory)) {
                return "Security error: attempted access outside the allowed directory.";
            }

            if (!Files.exists(requestedPath)) {
                return String.format("Error: file '%s' not found.", relativePath);
            }

            if (!Files.isRegularFile(requestedPath)) {
                return "Error: the specified path is not a file.";
            }

            // Check file size (maximum 1 MB)
            long fileSize = Files.size(requestedPath);
            if (fileSize > 1_000_000) {
                return String.format("Error: file is too large (%d bytes). " +
                                   "Maximum size: 1 MB.", fileSize);
            }

            // Read contents
            String content = Files.readString(requestedPath);

            return String.format("File contents '%s' (%d characters):\n\n%s",
                               relativePath, content.length(), content);

        } catch (IOException e) {
            return String.format("Error reading file: %s", e.getMessage());
        }
    }

    @Tool("Writes text to a file. WARNING: overwrites the existing file!")
    public String writeFile(
            @P("Relative path to the file to write")
            String relativePath,

            @P("Text content to write to the file")
            String content) {

        try {
            Path targetPath = allowedDirectory.resolve(relativePath).normalize();

            // Security validation
            if (!targetPath.startsWith(allowedDirectory)) {
                return "Security error: attempted write outside the allowed directory.";
            }

            // Create parent directories if needed
            Path parentDir = targetPath.getParent();
            if (parentDir != null && !Files.exists(parentDir)) {
                Files.createDirectories(parentDir);
            }

            // Write the file
            Files.writeString(targetPath, content, StandardOpenOption.CREATE,
                            StandardOpenOption.TRUNCATE_EXISTING);

            return String.format("File '%s' written successfully (%d characters).",
                               relativePath, content.length());

        } catch (IOException e) {
            return String.format("Error writing file: %s", e.getMessage());
        }
    }
}
```

### Tool Registration and Usage

```java
import dev.langchain4j.service.AiServices;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;

public class AgentApplication {

    public static void main(String[] args) {
        // Initialize the language model
        ChatLanguageModel model = OpenAiChatModel.builder()
            .apiKey(System.getenv("OPENAI_API_KEY"))
            .modelName("gpt-4")
            .temperature(0.7)
            .build();

        // Create tools
        WebSearchTools webTools = new WebSearchTools(new SearchApiClient());
        DatabaseTools dbTools = new DatabaseTools(createDataSource());
        FileTools fileTools = new FileTools("/safe/workspace");

        // Register the agent with tools
        Assistant assistant = AiServices.builder(Assistant.class)
            .chatLanguageModel(model)
            .tools(webTools, dbTools, fileTools)
            .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
            .build();

        // Use the agent
        String response = assistant.chat(
            "Find the latest AI news and save a summary to the file ai_news.txt"
        );

        System.out.println(response);
    }

    interface Assistant {
        String chat(String userMessage);
    }
}
```
