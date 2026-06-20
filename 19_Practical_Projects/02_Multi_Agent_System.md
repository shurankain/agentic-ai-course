# Project: Multi-Agent Code Review System

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Practical Projects
**Previous:** [[01_RAG_Chatbot|RAG Chatbot]]
**Next:** [[03_MCP_Server|MCP Server]]

---

## Project Overview

A multi-agent system for automated code review — a team of specialized agents, each focusing on its own aspect: security, performance, style, architecture.

### Why a Multi-Agent Approach?

A single agent with the prompt "review this code" suffers from:

**Cognitive overload** — attention diluted across security, performance, readability, and architecture. Each aspect receives only a shallow analysis.

**Inconsistent depth** — deep analysis of one file, shallow analysis of another, depending on what happened to catch the model's attention.

**Missing perspectives** — a generalist misses specific patterns. SQL injection is obvious to a security specialist but gets overlooked by a general reviewer.

A multi-agent architecture addresses this through specialization and parallelization.

## Architecture

The Orchestrator coordinates the work, distributes code, collects results, and manages the workflow.

Specialized reviewers:
- Security — SQL injection, XSS, path traversal, hardcoded secrets, insecure dependencies
- Performance — algorithmic complexity, memory usage, N+1 queries, caching
- Style — code style, naming conventions, readability
- Architecture — design decisions, coupling, cohesion, SOLID, patterns
- Test Coverage — test presence, quality, edge cases
- Documentation — comments, docstrings, API docs

The Synthesizer combines findings, eliminates duplicates, prioritizes issues, and produces the report.

## Framework Selection

This project can be implemented with several modern agent frameworks:

**LangGraph** — graph-driven orchestration with explicit state machines. The code review workflow is well-defined (parse → route → review → synthesize), making explicit control valuable. LangGraph's checkpointing enables resumable reviews for large PRs. Best when: you need fine-grained control over the execution flow, custom error handling per step, or observability via LangSmith.

**CrewAI** — role-based multi-agent coordination with built-in delegation. The specialized reviewer pattern maps naturally to CrewAI's role/goal/backstory model. CrewAI's sequential and hierarchical process types match the orchestrator-reviewer architecture. Best when: you want to get a working prototype quickly with minimal boilerplate, and the workflow is straightforward enough that implicit orchestration suffices.

**AWS Strands Agents** — model-driven approach with Agents-as-Tools pattern. The orchestrator delegates to specialized reviewer agents exposed as tools. Minimal boilerplate but less explicit control over the review workflow. Best when: you are already in the AWS ecosystem and want deep integration with AWS services.

**Recommendation:** Use LangGraph when the workflow has complex branching, error recovery, or requires production observability (checkpointing, LangSmith tracing). Use CrewAI when the agent roles are the primary abstraction and you want faster development with less code — it works well for production with straightforward multi-agent workflows. Both are production-viable; the choice depends on whether you need explicit flow control (LangGraph) or role-based simplicity (CrewAI).

## MCP Tool Integration

Specialized review agents benefit from external tools accessed via MCP:

**Code analysis tools:** MCP servers wrapping linters (ESLint, Pylint), static analyzers (SonarQube, Semgrep), and type checkers (mypy, tsc). The Security Reviewer agent connects to a Semgrep MCP server for SAST (Static Application Security Testing) findings.

**Repository context:** An MCP server providing Git history, blame information, related PRs, and issue tracker context. Agents use this to understand the change context and avoid flagging intentional patterns.

**Documentation:** An MCP server exposing project-specific coding standards, architecture decision records (ADRs), and style guides. The Style and Architecture reviewers reference these for project-specific recommendations.

**Configuration:** Each agent declares which MCP servers it requires. The orchestrator initializes the appropriate MCP connections when spawning reviewer agents.

## System Design: From Requirements to Architecture

Before writing code, the critical decisions are architectural. A multi-agent code review system requires answering three questions that determine the entire implementation.

**How many agents and what specializations?** Start with the minimum viable set: Security + Performance + Style + Orchestrator = 4 agents. Add Architecture and Test Coverage reviewers only if the team explicitly values these dimensions and is willing to pay the token cost. Each additional agent multiplies cost by ~1.3x (not 2x — the orchestrator and synthesis overhead is shared). A 4-agent system costs approximately 4x a single-agent review; a 6-agent system costs approximately 5.5x.

**Which communication pattern?** For code review, the Fan-Out/Fan-In pattern is optimal (see [[../04_Multi_Agent_Systems/02_MAS_Patterns|MAS Patterns]]). The orchestrator fans out the PR to all reviewers in parallel, then fans in the results for synthesis. This is preferred over sequential (too slow — each reviewer adds 10-30 seconds) or debate (unnecessary — reviewers analyze different dimensions, not opposing positions). The parallel execution means total latency equals the slowest reviewer, not the sum.

**How to manage state?** Two options. Shared state (LangGraph): all reviewers read from and write to a shared state object containing the PR diff, metadata, and accumulated findings. Simple to implement, single source of truth, but requires reducer logic for concurrent writes. Isolated state with message passing: each reviewer maintains its own context, the orchestrator collects results via messages. Cleaner separation but more boilerplate. For code review, shared state is simpler — the reviewers read the same PR and write non-overlapping findings.

**Model selection per agent.** Not all reviewers need the same model. The Security Reviewer benefits from a frontier model (Claude Opus 4.8 or Fable 5) because security analysis requires deep reasoning about attack vectors. The Style Reviewer works well with a cheaper model (Claude Sonnet 4.6 or Haiku 4.5) because style checks are pattern-matching. The Orchestrator can use a fast model (Haiku) — it routes and synthesizes, it does not analyze. This tiered approach reduces cost by 40-60% compared to using a frontier model for all agents.

## Evaluation: How to Know If Multi-Agent Is Worth It

Building a multi-agent system is more work than a single-agent solution. The evaluation must prove that the additional complexity delivers measurable value.

**A/B comparison methodology.** Run the same set of 50-100 PRs through both a single-agent reviewer and the multi-agent system. For each PR, measure: number of findings (recall — does the multi-agent system catch issues the single agent misses?), precision (are the findings actionable or noise?), developer acceptance rate (do developers actually fix the flagged issues?), review latency, and cost per review. The multi-agent system must show a measurable improvement on at least one dimension without significant regression on others. If the single agent catches 70% of issues and the multi-agent catches 85%, the 15% improvement may justify the 4x cost increase. If it catches 72%, it does not.

**Cost-quality analysis.** Calculate the cost per actionable finding. If the single agent costs $0.02/review and produces 3 actionable findings, the cost per finding is $0.007. If the multi-agent system costs $0.08/review and produces 5 actionable findings, the cost per finding is $0.016 — more expensive per finding, but catching 2 additional issues per PR may prevent bugs that cost thousands to fix in production. Frame the ROI in terms of bugs prevented, not review cost.

**When to revert to single-agent.** If the multi-agent system's precision drops below 60% (more than 40% false positives), developers will ignore all findings — worse than no review. If latency exceeds 5 minutes per PR, it blocks the development workflow. If cost exceeds $0.50/review for a team making 100+ PRs/day, the $50/day cost may exceed the budget. In any of these cases, a well-tuned single agent with a frontier model is the better choice.

## Core Infrastructure

### Agent Base Class

The base class defines: agent name, role, specialized system prompt, an analyze method that accepts a ReviewContext (code diff, metadata, history, configuration) and returns ReviewFindings (issues with severity/location/description, suggestions, positive observations, confidence scores).

### Specialized Prompts

The Security Reviewer focuses on the OWASP Top 10, credentials in code, authentication/authorization, input validation, and cryptography. For each finding it specifies severity (CRITICAL/HIGH/MEDIUM/LOW), CWE ID, location, exploitation scenario, and recommended fix. Confidence level is provided when uncertain.

The Performance Reviewer analyzes algorithmic complexity, memory leaks, and N+1 queries. The Style Reviewer checks naming conventions, formatting, and readability. The Architecture Reviewer evaluates SOLID principles, design patterns, and coupling. Each prompt is tuned for narrow expertise.

## Orchestration and Communication

### Orchestrator Logic

1. Parse input — determine scope (changed files, affected dependencies)
2. Analyze structure — identify languages, frameworks, file types
3. Route to agents — send relevant files to the corresponding agents
4. Parallel execution — run agents in parallel
5. Collect results — gather findings, handle errors
6. Resolve conflicts — handle contradictory recommendations
7. Synthesize report — produce a unified report

### Communication Protocol

Structured messages: sender ID, receiver ID, type (Finding/Question/Clarification/Consensus), content, timestamp. Ensures traceability and decision-making flow.

### Conflict Resolution

Strategies for contradictions:

1. Priority-based — Security > Performance > Style
2. Voting — majority of agents decides
3. Escalation — Synthesizer makes the final decision
4. Human-in-the-loop — disputed cases go to human review

## GitHub Integration

Workflow: PR created/updated → webhook triggers review → agents analyze → results posted as PR comments → status check updated (pass/fail blocks merge).

Comment formatting: severity indicator (🔒 Security, ⚡ Performance), severity level, location, CWE ID, description, recommended fix with examples.

### Incremental Review

Review only changed lines (not the entire file), context-aware (surrounding code), delta reports (new vs. fixed issues), focus on regressions (new issues vs. existing tech debt).

## Learning and Improvement

### Feedback Loop

Explicit feedback — "helpful"/"not helpful" for fine-tuning. Implicit feedback — PR merged without fixing a finding (possible false positive). Correction tracking — the fix differs from the suggestion (improving suggestions).

### Metrics

Precision — percentage of findings actually fixed (few false positives). Recall — percentage of real issues detected. Time savings — comparison of AI vs. manual review. Coverage — percentage of codebase that has undergone review.

A dashboard visualizes trends and identifies areas for improvement.

## Key Takeaways

A multi-agent approach outperforms a single agent through specialization and depth of analysis.

Modern frameworks (LangGraph, CrewAI, Strands) provide ready-made multi-agent orchestration patterns — use them instead of building from scratch.

MCP integration gives review agents access to external tools (linters, analyzers, repo context) for deeper analysis than LLM reasoning alone.

Separation of roles ensures comprehensive coverage without cognitive overload.

The Orchestrator coordinates parallel execution and conflict resolution for consistent results.

GitHub integration makes AI review part of the natural workflow.

A feedback loop and analytics ensure improvement over time, adapting to the codebase and preferences.

---

## Navigation
**Previous:** [[01_RAG_Chatbot|RAG Chatbot]]
**Next:** [[03_MCP_Server|MCP Server]]
