# Agent Security

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Security and Safety
**Previous:** [[02_Data_Protection|Data Protection]]
**Next:** [[04_Moderation_and_Compliance|Moderation and Compliance]]

---

## Autonomy as a Source of Risk

Autonomous agents represent a fundamentally new level of risk. While a chatbot can give a wrong answer, an agent can perform a wrong action: delete data, send an email, execute code, spend money. The consequences of errors are material and often irreversible.

Examples: an agent with access to corporate email could send spam on behalf of the company under prompt injection. An agent with filesystem access could delete important documents. An agent with access to a payment system API could initiate transactions. Every tool is a potential attack vector.

The problem is compounded by the fact that LLMs are not reliable decision-makers. They hallucinate, misinterpret instructions, and are vulnerable to manipulation. Delegating autonomous decisions requires multiple layers of protection and constraints.

## Principle of Least Privilege

Principle of Least Privilege (PoLP) is a fundamental security principle, especially critical for agents. An agent should have exactly the permissions necessary to perform the task, and not a single permission more.

In practice: if an agent needs to read files from a specific directory, it should not have access to the entire filesystem. If it needs to send email, it should not have access to the inbox or contacts. If it needs to execute SQL queries, it should not have permissions to modify the schema.

Permissions should be granular and context-specific. Instead of a "database access" permission, use "SELECT from customers, orders tables for the current user's user_id". Instead of "send email", use "send email to pre-approved domains with a rate limit of 10 per day".

Dynamic privilege escalation on demand is safer than granting all permissions upfront. The agent starts with minimal permissions, and the system requests escalation only when truly needed.

## Permission System for Tools

Each agent tool must have explicitly defined permissions: which operations are allowed, which parameters are acceptable, what the limits are, and whether human approval is required.

**Whitelist approach** is safer than blacklist. Instead of "prohibit dangerous operations", use "allow only these specific operations". Everything not explicitly allowed is denied by default.

Permission levels: READ_ONLY (read only), LIMITED_WRITE (write with restrictions), FULL_ACCESS (full access, dangerous), HUMAN_APPROVAL (requires approval before each action).

Permission checks occur before every tool invocation. Validated: operation type, parameters, usage limits, invocation context. Any violation blocks the operation and is logged.

## Sandboxing Code Execution

Code execution is the most dangerous capability. Generated code can do anything the environment permits: file access, network calls, system operations. Sandboxing isolates execution, limiting potential damage.

**Container-based sandboxing** uses Docker or other container runtimes for isolation. Code runs in an ephemeral container with a minimal image, no network access, and limited resources. After execution, the container is destroyed.

**Resource limits** prevent denial-of-service: CPU limits (cpu quota), memory, execution time. Infinite loops or memory bombs do not bring down the system.

**Network isolation** (--network=none) prevents data exfiltration and connections to external resources. If code requires network access, it must be explicitly allowed and restricted by a whitelist.

**Filesystem restrictions** through read-only root filesystem, mounting only necessary directories, and prohibiting writes outside designated areas.

## Static Code Analysis

Before executing generated code, it is prudent to perform static analysis for dangerous operations. This does not replace sandboxing but adds an additional layer of protection.

Dangerous operation patterns depend on the language. For Python: import os, import subprocess, eval(), exec(), open() with write, socket operations. For JavaScript: require('child_process'), eval(), fs.writeFile.

**Whitelist imports** are safer than blacklist. Allow only a specific set of modules (math, json, datetime), block everything else.

**AST analysis** allows understanding code structure without execution. It finds function calls, imports, and dangerous constructs at the syntax tree level.

**Severity levels** help make decisions. High severity (system calls, network) — block. Medium (file operations) — warn or require approval. Low (safe operations) — allow.

## Human-in-the-Loop

Human-in-the-loop (HITL) is a pattern where critical actions require human approval before execution. It balances agent autonomy and human control.

Which actions require HITL depends on the domain and risk appetite. Common categories: financial operations above a threshold, modification of production data, communication with external parties, irreversible actions, access to sensitive information.

HITL implementation: pause execution, notify the human with action context, wait for approve/reject, timeout with default action (usually reject).

UX is critical for adoption. If approval requests are too frequent or uninformative, people start approving everything automatically (approval fatigue). Context should be sufficient for decision-making but not overwhelming.

Async approval allows a human to approve an action later. The agent saves state and can continue after receiving approval. Important for non-real-time scenarios.

## Rate Limiting and Anomalies

Rate limiting restricts operation frequency, preventing abuse and reducing damage in case of compromise.

**Per-user limits** protect against abuse by a single user. **Per-tool limits** restrict usage of individual tools. **Global limits** protect the system as a whole.

**Burst vs sustained rates** allow short-term spikes while limiting long-term usage. Token bucket or leaky bucket algorithms implement this.

**Daily/monthly quotas** restrict total usage over a period. Especially important for expensive operations (LLM API, external services).

**Anomaly detection** identifies unusual behavior: sudden activity spikes, atypical usage patterns, access at unusual times. Anomalies may indicate an attack or compromise.

## Logging and Action Auditing

Every agent action must be logged for subsequent analysis and audit. Logging is not only for debugging but also for security.

What to log: requested action, parameters (sanitized), result, timing, context (user, session, conversation). Do not log sensitive data in raw form.

**Immutable logs** prevent tampering. Append-only storage, cryptographic signatures, forwarding to an external SIEM.

**Real-time monitoring** for critical events. Alerts on suspicious activity, blocked operations, rate limit violations.

**Retention** sufficient for forensics and compliance. 90 days for detailed logs, 1+ year for aggregated logs.

## Graceful Degradation

When problems are detected, the system should degrade gracefully rather than fail completely.

**Fallback mode** — transition to a less autonomous mode. Upon suspicion of an attack — switch to a mode with mandatory HITL for all actions.

**Circuit breaker** on repeated errors. If a tool consistently fails, temporarily disable it.

**User notification** about limited functionality. The user should understand why the agent cannot perform an action.

**Recovery procedures** for returning to normal mode after the problem is resolved.

## MCP Security Boundaries

The Model Context Protocol (MCP) introduces a standardized tool interface for agents, but also creates specific security challenges that require architectural attention.

### MCP Trust Model

MCP operates with an implicit trust hierarchy: the **host application** (e.g., Claude Code, an IDE) trusts the **MCP client** it manages, which connects to **MCP servers** that provide tools, resources, and prompts. The critical security question is: **how much should the client trust each server?**

**Server trust levels:**
- **First-party servers** (built by the application developer) — high trust, full capability access
- **Verified third-party servers** (from MCP registries with review processes) — medium trust, capability restrictions advisable
- **Unverified third-party servers** — low trust, strict sandboxing required
- Tool descriptions from untrusted servers are a prompt injection vector (see CVE-2025-54135 CurXecute)

### OAuth 2.1 Delegation for Agents

MCP's authorization framework (OAuth 2.1 with PKCE, added to the spec in 2025) enables secure delegation of user credentials to agents:

**The delegation problem:** An agent acting on behalf of a user needs access to external services (databases, APIs, SaaS tools). Giving the agent the user's credentials directly is insecure — the agent could be compromised, and credentials cannot be scoped.

**OAuth 2.1 solution:** The MCP server acts as a resource server, the agent obtains scoped access tokens via OAuth flows, tokens have limited scope (only the permissions needed for the task), tokens expire and can be revoked, the user approves the delegation through a standard consent flow.

**Protected Resource Metadata (PRM):** MCP servers advertise their authorization requirements, enabling clients to discover what credentials are needed before attempting access. This prevents trial-and-error authentication and enables better UX.

**Security best practices for MCP OAuth:**
- Use short-lived tokens (minutes, not hours)
- Request minimum scopes — never request broader access than the current task requires
- Implement token rotation for long-running agent sessions
- Log all token grants and usage for audit

### Multi-Agent Security

When multiple agents collaborate (via frameworks like OpenAI Agents SDK, LangGraph, CrewAI), new security challenges emerge:

**Agent-to-agent trust:** When Agent A delegates a subtask to Agent B, what permissions does B inherit? The naive approach (B inherits all of A's permissions) violates least privilege. The correct approach: B receives only the permissions needed for its specific subtask, with explicit scope reduction at each delegation boundary.

**Privilege escalation across agents:** A compromised or manipulated inner agent could attempt to use its delegator's permissions to access resources beyond its scope. Defense: each agent maintains its own permission boundary, delegation explicitly reduces (never increases) the permission set, and cross-agent calls are logged and auditable.

**Confused deputy attacks:** An outer agent asks an inner agent to perform a task. The inner agent, acting on the outer agent's authority, accesses resources it shouldn't. This is analogous to the classic confused deputy problem in computer security. Defense: capabilities should be explicitly passed (not ambient), and each tool invocation should include the effective permission context.

**Shared context poisoning:** In multi-agent systems, agents often share context (conversation history, tool results). A compromised agent can inject malicious content into shared context, affecting all downstream agents. Defense: sanitize cross-agent context, mark trust boundaries in shared state.

## Agent Security Testing

Security testing for agents requires specific approaches.

**Adversarial testing** attempts to make the agent perform prohibited actions through prompt injection, social engineering, and edge cases.

**Permission boundary testing** verifies that restrictions actually work. Attempts to execute prohibited operations, exceed limits, and bypass HITL.

**Chaos testing** for resilience. What happens when tools fail, timeouts occur, or unexpected responses arrive?

**Red teaming** with external specialists for independent security assessment.

## Key Takeaways

Autonomous agents carry fundamentally greater risk. Their actions are material and often irreversible.

Principle of Least Privilege is critical: an agent should have only the minimum necessary permissions, specific to the task context.

Permission system defines allowed operations, parameters, limits, and HITL requirements for each tool. Whitelist approach is safer than blacklist.

Sandboxing isolates generated code execution through containers, resource limits, network isolation, and filesystem restrictions.

Static code analysis identifies dangerous operations before execution: system calls, network access, file write, dynamic code execution.

Human-in-the-loop for critical actions balances autonomy and control. UX is critical for preventing approval fatigue.

Rate limiting prevents abuse and limits damage in case of compromise. Anomaly detection identifies atypical behavior.

Comprehensive logging and real-time monitoring provide visibility and forensics. Immutable logs prevent tampering.

Graceful degradation during problems: transition to fallback mode with reduced autonomy, circuit breakers, user notification.

Adversarial testing, permission boundary testing, and red teaming are necessary for validating agent security.

MCP security requires explicit trust boundaries: verify server sources, audit tool descriptions (a prompt injection vector), use OAuth 2.1 for scoped credential delegation, and apply different trust levels to first-party vs third-party servers.

Multi-agent systems introduce privilege escalation risks: enforce scope reduction at delegation boundaries, prevent confused deputy attacks through explicit capability passing, and sanitize shared context between agents.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Security and Safety
**Previous:** [[02_Data_Protection|Data Protection]]
**Next:** [[04_Moderation_and_Compliance|Moderation and Compliance]]
