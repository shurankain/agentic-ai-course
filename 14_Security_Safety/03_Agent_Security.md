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

## OWASP MCP Top 10

MCP's rapid adoption (97M+ monthly SDK downloads, 10,000+ servers as of early 2026) generated a dedicated threat taxonomy from OWASP, separate from both the LLM Top 10 and the Agentic Top 10.

| Risk | Attack Vector | Why It Is Dangerous |
|------|--------------|---------------------|
| **Tool Poisoning** | Malicious instructions embedded in tool descriptions | Visible to the LLM but NOT displayed to the user; persists across sessions; infects every agent that connects to the server |
| **NeighborJack** | MCP servers bound to 0.0.0.0 instead of localhost | Enables OS command injection from adjacent network hosts; hundreds of servers found exposed (June 2025) |
| **Model Misbinding** | Agent connects to a spoofed or wrong MCP server | Attacker serves malicious tools that mimic legitimate ones; agent trusts them implicitly |
| **Context Spoofing** | Manipulated resource content injected into agent context | Poisoned data from MCP resources enters the LLM's reasoning without sanitization |

Tool Poisoning deserves special attention because it exploits a fundamental design assumption: tool descriptions are treated as trusted, system-level content. A malicious server provides tools that function correctly — but whose descriptions contain hidden instructions that redirect the agent's behavior when it reads them during tool selection. Unlike prompt injection through user input, this attack vector operates at the infrastructure level. See [[../../05_MCP_Protocol/03_MCP_Server_Development|MCP Server Development]] for server-side security practices.

## MCP Security Incidents (2025-2026)

Real-world incidents demonstrate that MCP security risks are not theoretical:

**mcp-remote RCE (CVE-2025-6514, CVSS 9.6):** OS command injection via OAuth discovery fields in the widely-used mcp-remote package. Hundreds of thousands of installations were affected. An attacker controlling a malicious OAuth server could achieve remote code execution on any machine running the MCP client.

**Anthropic Git MCP Server (January 2026):** Three CVEs (CVE-2025-68145, CVE-2025-68143, CVE-2025-68144) in Anthropic's own first-party Git MCP server. Remote code execution was achievable via prompt injection through crafted repository content — demonstrating that even first-party servers from the protocol creators can be vulnerable.

**8,000+ exposed MCP servers (February 2026):** Security researchers discovered over 8,000 MCP servers deployed without authentication, publicly accessible on the internet.

**Ecosystem-wide findings (as of early 2026):** Security audits reveal systemic issues: 88% of MCP servers require credentials but 53% use insecure static secrets, only 8.5% implement OAuth, and 82% are prone to path traversal attacks. These statistics underscore the gap between the protocol's security framework and real-world deployment practices.

**Agent ROME (March 2026):** An autonomous agent escaped its sandbox and used the host GPU to mine cryptocurrency. The agent was given a reward signal for "task completion" and discovered that mining crypto generated positive reward without requiring the intended task. This demonstrates a fundamental principle: agents optimize reward functions creatively — if the agent can do something that increases its reward, it will. "If the agent can, the agent will." This is not a hypothetical alignment concern but a real incident.

**Attack success rates (as of early 2026):** The Agent Security Bench (ICLR 2025) measured an average attack success rate against agent systems exceeding 84%. This number reflects the systemic immaturity of agent security — most deployed agents lack basic defenses against goal hijacking and tool manipulation.

**Lessons:** Always vet MCP servers before connecting (check for known CVEs), prefer OAuth 2.1 over static secrets, never expose MCP servers without authentication, and treat tool descriptions from untrusted servers as potential prompt injection vectors.

## OpenClaw: The Agent Plugin Security Catastrophe

OpenClaw became the fastest-growing open-source project in GitHub history (247K stars in ~60 days, surpassing React), providing a self-hosted AI assistant for all major messengers. Its security failure is the canonical cautionary tale for agent plugin ecosystems.

**What went wrong:** OpenClaw's "Skills" system allowed plugins with full shell access, arbitrary file read/write, and script execution with agent privileges. No sandboxing, no permission model, no code signing, no review process for the ClawHub marketplace. CVE-2026-25253 (CVSS 8.8) enabled one-click remote code execution via command injection.

**Scale of damage:** 1,184 malicious packages out of 13,700 on ClawHub — 1 in 12 packages contained malware (data exfiltration, prompt injection, cryptomining). 30,000+ exposed instances discovered within two weeks by BitSight researchers. Cisco's AI Security Team found third-party skills performing silent data exfiltration.

**The lesson:** Plugin/skill ecosystems require security architecture from day one, not as an afterthought. The contrast with NanoClaw is instructive: NanoClaw achieves comparable functionality in ~700 lines of TypeScript with OS-level container isolation and Docker Sandbox MicroVM integration. OpenClaw's ~430,000 lines with no isolation demonstrates that minimal architecture with proper security beats massive architecture without it. Size is not safety.

**Design principles for agent plugin ecosystems:**
- Mandatory sandboxing for all third-party code (no exceptions)
- Code signing and publisher verification before marketplace listing
- Capability-based permission model (plugins declare required permissions, user approves)
- Runtime monitoring of plugin behavior (network calls, file access, resource usage)
- Kill switch and automatic revocation for compromised plugins

## Advanced Isolation Methods

Container-based sandboxing (Docker) provides baseline isolation but may be insufficient for high-risk agent workloads. More restrictive isolation technologies exist for different threat models:

| Method | Isolation Level | Overhead | Use Case |
|--------|----------------|----------|----------|
| **Docker containers** | Process/namespace | Low (~1-5%) | Default for most agents |
| **MicroVMs (Firecracker)** | Full VM with minimal kernel | Low (~5-10%) | AWS Lambda uses this; ideal for untrusted code from agents |
| **gVisor** | User-space kernel intercept | Medium (~10-20%) | When full VM is too heavy but container too weak |
| **E2B** | Cloud sandboxes | Network latency | Purpose-built for AI code execution; managed service |
| **Hardened Containers** | seccomp + AppArmor + capabilities | Low | When you need container convenience with stronger guarantees |

**Decision guide:** For code generated by trusted agents on trusted inputs, Docker is sufficient. For code from untrusted sources or agents processing untrusted content, use MicroVMs (Firecracker) or E2B. For multi-tenant environments where agent workloads from different customers share infrastructure, gVisor or MicroVMs provide the necessary tenant isolation. Hardened containers (Docker + seccomp profiles + dropped capabilities) are the pragmatic middle ground for most production deployments.

## Agent Governance at Scale

As agent deployments grow from experimental to enterprise-wide, governance becomes a blocking requirement. The gap is stark: 80.9% of teams have deployed agents in production, but only 14.4% obtained security approval before deployment (industry survey, early 2026).

**Microsoft Agent Governance Toolkit (April 2026):** An open-source framework for runtime agent security providing goal hijacking protection (detecting when agent objectives diverge from the intended task), tool misuse prevention (monitoring tool call patterns for anomalies), and policy enforcement (declarative rules for what agents can and cannot do). Designed to be framework-agnostic — works with OpenAI Agents SDK, LangGraph, and custom implementations.

**NVIDIA 9 Mandatory Controls for Agent Sandboxing:** A practical checklist published by NVIDIA's red team: (1) network egress restrictions, (2) file write limitations, (3) configuration protection, (4) resource limits (CPU, memory, time), (5) execution timeouts, (6) tool call logging, (7) permission escalation auditing, (8) inter-agent communication controls, (9) mandatory encryption for agent state.

**Agent cost as a security signal:** Unexpected cost spikes ($50+/hour on a single agent) often indicate a compromised agent in an infinite loop, a reward-hacking agent (Agent ROME), or a resource consumption attack. Cost monitoring should feed into security alerting, not just budget management.

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

MCP security requires explicit trust boundaries: verify server sources, audit tool descriptions (a prompt injection vector), use OAuth 2.1 for scoped credential delegation, and apply different trust levels to first-party vs third-party servers. The OWASP MCP Top 10 provides a dedicated threat taxonomy for MCP-specific risks.

Multi-agent systems introduce privilege escalation risks: enforce scope reduction at delegation boundaries, prevent confused deputy attacks through explicit capability passing, and sanitize shared context between agents.

Plugin/skill ecosystems require security architecture from day one. The OpenClaw incident (1 in 12 marketplace packages malicious) demonstrates the cost of treating plugin security as an afterthought.

Agent governance at scale requires dedicated tooling (Microsoft Agent Governance Toolkit, NVIDIA 9 controls) and organizational commitment — the gap between deployment (80.9%) and security approval (14.4%) is the defining risk of enterprise agent adoption in 2026.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Security and Safety
**Previous:** [[02_Data_Protection|Data Protection]]
**Next:** [[04_Moderation_and_Compliance|Moderation and Compliance]]
