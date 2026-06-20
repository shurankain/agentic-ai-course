# Project: MCP Server for a Knowledge Base

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Practical Projects
**Previous:** [[02_Multi_Agent_System|Multi-Agent System]]
**Next:** [[04_Fine_Tuning_Pipeline|Fine-Tuning Pipeline]]

---

## Project Overview

An MCP (Model Context Protocol) server provides an AI model with access to a corporate knowledge base. It is a standard protocol by Anthropic for connecting external data and tools to LLM applications.

### Why MCP?

Every AI provider (OpenAI, Anthropic, Google) has its own format for connecting data and tools. MCP standardizes the process — you write one server, and it works with any MCP-compatible client (Claude Desktop, Cursor, custom application).

### Three Types of Capabilities

**Resources** — read-only data (knowledge base articles). URI identification (kb://articles/oauth-guide), MIME type, discovery (the client learns about available data), templates for dynamic content.

**Tools** — actions that change state (search, create, update records). JSON Schema parameter descriptions allow the model to understand usage.

**Prompts** — ready-made templates for common scenarios. Simplify onboarding — instead of crafting a query, the user selects a prompt like "Summarize article" or "Answer question from knowledge base".

### Advantages

Portability — one server works with any MCP client. Modularity — multiple servers connected to one client (knowledge base + GitHub + Jira). Discovery — the client automatically learns about capabilities. Type safety — JSON Schema validation. Standardized errors — a uniform error handling format.

## Basic Structure

### Communication Protocol

JSON-RPC 2.0: jsonrpc (version "2.0"), id (unique request identifier), method (method name: tools/list, resources/read), params (call parameters). The response contains the same id and either result (success) or error (failure).

### Transport Mechanisms

**stdio (Standard Input/Output)** — the server runs as a subprocess, reads JSON from stdin, writes to stdout. Secure (no network access), simple, ideal for local servers. Claude Desktop uses stdio.

**Streamable HTTP (2025)** — the current recommended HTTP transport, replacing the earlier HTTP+SSE approach. A single HTTP endpoint handles both request-response and streaming. The client sends POST requests with JSON-RPC payloads. The server responds with either a regular JSON response (for simple operations) or an SSE stream (for long-running operations and notifications). This simplifies deployment — one endpoint instead of separate POST and SSE endpoints. Backwards-compatible: servers can support both old SSE and new Streamable HTTP during migration.

**HTTP with SSE (deprecated)** — the original HTTP transport. Separate endpoints for POST requests and SSE streams. Still supported for backwards compatibility but new implementations should use Streamable HTTP.

### Lifecycle

Initialization — the client sends an initialize request with capabilities, the server responds with its own capabilities. Confirmation — the client sends an initialized notification. Working phase — resource requests, tool calls, prompt retrieval. Termination — shutdown request, resource cleanup.

## Resources

### Concept

Read-only data for the model. Each resource has: URI (unique identifier kb://articles/getting-started), Name (human-readable title), Description (brief summary), MIME type (text/markdown, application/json, text/plain).

### Discovery

resources/list returns a list of available resources with metadata. Only descriptions, not the data itself — efficient for large collections.

### Resource Templates

For dynamic content: kb://articles/{article_id}. The client substitutes the parameter and requests the needed resource. Useful for large collections, on-the-fly generated resources, parameterized queries (logs://{service}/{date}).

### Reading

resources/read with a specific URI. The server returns a contents array: URI, MIME type, data (text/blob). Metadata with caching hints and last-modified timestamp.

## Tools

### Concept

Actions via the server: create, update, delete, query, API calls.

Components: Name (unique identifier, snake_case, verbs: search_articles, create_user), Description (critically important — how the model understands when to use it), Input Schema (JSON Schema of parameters, types, required fields, validation).

### Design for a Knowledge Base

search_articles — search by keywords (query, limit). create_article — create an article (title, content, tags). update_article — update an article (article_id, new values). get_stats — statistics (count, views, popular tags).

### Execution

tools/call request with name and arguments. The server: validates against the schema, performs the operation, returns the result in a content array.

### Error Handling

Graceful handling: isError: true, text description. Article not found, insufficient parameters, service unavailable, rate limit. Good error messages help the model correct its behavior.

## Prompts

### Concept

Ready-made templates for common scenarios. Solve the "cold start" problem. Name, Description, Arguments (user-provided parameters).

prompts/get returns ready-made messages in conversation format. For example, summarize_article with {article_id, length} generates "Please provide a short summary of this article: [text]".

### For a Knowledge Base

summarize_article — summary generation (ID, length short/medium/long). answer_question — answer using the knowledge base (user question, auto-search for relevant articles). write_article — writing assistance (topic, loading existing articles for reference).

### Advantages

Lower the entry barrier, demonstrate capabilities, ensure best practices are followed, speed up routine tasks, guarantee proper formatting.

## Building the Server: Step-by-Step Walkthrough

The knowledge base MCP server is built in stages, each adding a capability that can be tested independently.

**Stage 1: Minimal viable server with one resource.** Start with a single resource — `kb://articles/list` that returns a JSON list of article titles and IDs from a PostgreSQL database. Use the Python MCP SDK (`mcp` on PyPI) or the TypeScript SDK (`@modelcontextprotocol/sdk`). The server reads from the database, formats the response, and returns it. Test by connecting to Claude Desktop or the MCP Inspector. If the client can discover and read the resource, the foundation works.

**Stage 2: Add tools for search and CRUD.** Define `search_articles(query: str, limit: int)` with JSON Schema for parameters. The model sees the schema, understands the parameters, and calls the tool with structured arguments. Add `create_article(title: str, content: str, tags: list[str])` and `update_article(article_id: str, ...)` for mutations. Each tool validates inputs against the schema before executing. Test with progressively complex queries: simple keyword search, multi-word queries, edge cases (empty query, special characters).

**Stage 3: Add prompts for common workflows.** Define `summarize_article(article_id)` and `answer_question(question)`. The `answer_question` prompt internally calls `search_articles` to find relevant content, then formats a prompt that includes the retrieved articles as context. This transforms the server from a data source into a knowledge assistant. Test by asking questions that require synthesizing information across multiple articles.

**Stage 4: Add error handling and edge cases.** What happens when: the article does not exist (return a clear error, not a crash), the database is down (return a service unavailable error with a retry hint), the search returns no results (return an empty result with a suggestion to broaden the query), the input contains SQL injection attempts (parameterized queries prevent this, but validate inputs explicitly). Each error path should return a human-readable message that helps the model (and the user) understand what went wrong and what to try next.

**Integration testing with Claude Code.** Configure Claude Code to connect to the server via stdio. Ask Claude to "search the knowledge base for authentication articles" and verify it calls the right tool. Ask it to "create an article about OAuth best practices" and verify the article appears in the database. Ask a question that requires cross-referencing multiple articles. This end-to-end test validates that the server works as part of a real agent workflow, not just in isolation.

## Production Hardening

Moving from a working prototype to a production-ready MCP server requires addressing reliability, security, and operational concerns.

**Authentication and authorization.** For servers exposed over the network (Streamable HTTP transport), implement OAuth 2.1 as described in the OAuth 2.1 Authorization section below. For stdio servers running locally, authentication is implicit — the server runs in the user's process. Never expose a stdio server over the network without wrapping it in an authenticated HTTP layer.

**Rate limiting.** Prevent abuse and control costs. Implement per-user rate limits (e.g., 100 tool calls per minute, 1000 resource reads per hour) using a sliding window counter in Redis. Return a clear rate limit error with a Retry-After hint. For search operations that hit the database, add a separate query rate limit to protect the backend.

**Input validation beyond JSON Schema.** JSON Schema validates types and required fields, but not business logic. Additional validation: string length limits (prevent 1MB article titles), allowed characters (prevent injection), referential integrity (article_id must exist), and content sanitization (strip scripts from user-provided content). Fail fast with a descriptive error — do not let invalid data reach the database.

**Monitoring and alerting.** Log every tool call with: timestamp, user/session ID, tool name, input parameters (PII-masked), execution time, result status (success/error). Alert on: error rate exceeding 5% over a 5-minute window, P95 latency exceeding 2 seconds, rate limit violations (potential abuse), and tool calls from unexpected clients (security incident). Integrate with the observability stack — see [[../12_Observability/01_Tracing_and_Logging|Tracing and Logging]] for patterns.

**Graceful degradation.** When the database is slow or unavailable, the server should: return cached results for read operations (with a staleness warning), queue write operations for retry, and return a service degradation error for operations that cannot be cached or queued. The model (and user) should receive a clear message: "The knowledge base is temporarily operating in read-only mode. New articles cannot be created until the issue is resolved."

## Production Features

### Logging

notifications/message with a level (debug, info, warning, error) and text. The client displays it in the UI or writes to a file. Useful for debugging — shows executed queries, processed data, errors.

### Progress Reporting

Progress notifications for long-running operations (indexing, document loading). progressToken, current progress, total. The client displays a progress bar. Improves UX, prevents premature cancellation.

### Caching

Caching hints in metadata: maxAge (how many seconds the data remains valid), etag (content version). The client caches data and does not re-request it while the cache is valid. Speeds up operation, reduces load.

### Security

Input validation — parameter and URI checks for tools, JSON Schema validation. Content sanitization — preventing injection and sensitive data leaks. Rate limiting — restricting call frequency. Audit logging — recording mutating operations with timestamp and user context.

### OAuth 2.1 Authorization (2025)

The MCP specification added OAuth 2.1 as the standard authorization mechanism for remote MCP servers:

**Server as Resource Server:** The MCP server validates OAuth 2.1 access tokens on incoming requests. Tokens carry scopes that map to specific tools and resources — a token with scope `kb:read` can access resources but not mutating tools.

**Authorization flow:** The MCP client discovers the server's authorization requirements during the initialize handshake. If authorization is required, the client redirects the user through the standard OAuth 2.1 authorization code flow (with PKCE). The client stores the refresh token for transparent token renewal.

**Scopes and permissions:** Define granular scopes for your knowledge base server: `kb:read` (read resources), `kb:search` (use search tools), `kb:write` (create/update articles), `kb:admin` (statistics, management). Map OAuth scopes to MCP capabilities.

**Token validation in practice:** Every incoming MCP request over Streamable HTTP must carry a Bearer token. The server validates the token before processing the request — verifying the signature (JWT) or introspecting with the authorization server (opaque tokens), checking expiration, and confirming that the token's scopes match the requested operation.

The validation flow for a tool call: extract the Bearer token from the Authorization header → validate the token (signature, expiration, issuer) → extract scopes from the token claims → check that the required scope for the requested tool is present (e.g., `kb:write` for `create_article`) → if valid, execute the tool; if not, return a JSON-RPC error with code `-32001` and message "Insufficient permissions."

**Scope-to-capability mapping example:** `kb:read` permits `resources/read` and `resources/list`. `kb:search` permits `tools/call` for `search_articles`. `kb:write` permits `tools/call` for `create_article` and `update_article`. `kb:admin` permits `tools/call` for `get_stats` and management operations. Unmapped operations are denied by default.

**Why OAuth 2.1 over API keys:** OAuth 2.1 provides token expiration, scope-based access control, token revocation, and audit trails. API keys are static, unscoped, and difficult to rotate. For production MCP servers exposed over the network, OAuth 2.1 is the recommended approach.

## Key Takeaways

Standardization changes everything — one MCP server instead of separate connectors for each AI provider.

Three capabilities cover most scenarios — Resources for read-only, Tools for actions, Prompts for UX.

Metadata quality determines operational quality — good descriptions, detailed schemas, clear names are critical.

Transport evolution — stdio for local integrations, Streamable HTTP (2025) for remote servers. Streamable HTTP replaces the older HTTP+SSE transport with a simpler single-endpoint design.

OAuth 2.1 authorization enables secure multi-tenant MCP servers with scoped access control, token expiration, and revocation.

Production-readiness requires more than a basic implementation — logging, progress reporting, caching, security measures.

Extensibility — easy to add new resources and tools. Start with the basics (search, read), then add features.

---

## Navigation
**Previous:** [[02_Multi_Agent_System|Multi-Agent System]]
**Next:** [[04_Fine_Tuning_Pipeline|Fine-Tuning Pipeline]]
