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

Separation of roles ensures comprehensive coverage without cognitive overload.

The Orchestrator coordinates parallel execution and conflict resolution for consistent results.

GitHub integration makes AI review part of the natural workflow.

Structured communication enables requesting clarification and reaching consensus.

A feedback loop and analytics ensure improvement over time, adapting to the codebase and preferences.

---

## Navigation
**Previous:** [[01_RAG_Chatbot|RAG Chatbot]]
**Next:** [[03_MCP_Server|MCP Server]]
