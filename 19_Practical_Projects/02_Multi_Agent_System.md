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

**LangGraph** — graph-driven orchestration with explicit state machines. Best for this use case: the code review workflow is well-defined (parse → route → review → synthesize), making explicit control valuable. LangGraph's checkpointing enables resumable reviews for large PRs.

**CrewAI** — role-based multi-agent coordination with built-in delegation. Natural fit for the specialized reviewer pattern: each agent has a role, goal, and backstory. CrewAI's sequential and hierarchical process types map directly to the orchestrator-reviewer architecture.

**AWS Strands Agents** — model-driven approach with Agents-as-Tools pattern. The orchestrator delegates to specialized reviewer agents exposed as tools. Minimal boilerplate but less explicit control over the review workflow.

**Recommendation:** LangGraph for production (explicit flow control, checkpointing, observability via LangSmith) or CrewAI for rapid prototyping (less code, intuitive role-based design).

## MCP Tool Integration

Specialized review agents benefit from external tools accessed via MCP:

**Code analysis tools:** MCP servers wrapping linters (ESLint, Pylint), static analyzers (SonarQube, Semgrep), and type checkers (mypy, tsc). The Security Reviewer agent connects to a Semgrep MCP server for SAST (Static Application Security Testing) findings.

**Repository context:** An MCP server providing Git history, blame information, related PRs, and issue tracker context. Agents use this to understand the change context and avoid flagging intentional patterns.

**Documentation:** An MCP server exposing project-specific coding standards, architecture decision records (ADRs), and style guides. The Style and Architecture reviewers reference these for project-specific recommendations.

**Configuration:** Each agent declares which MCP servers it requires. The orchestrator initializes the appropriate MCP connections when spawning reviewer agents.

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
