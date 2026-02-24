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
