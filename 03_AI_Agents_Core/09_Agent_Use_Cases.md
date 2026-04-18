# Practical AI Agent Use Cases

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[08_Process_Reward_Models|Process Reward Models]]
**Next:** [[10_Resource_Optimization|Resource-Aware Optimization]]

---

## Introduction: From Theory to Practice

AI agents have been solving real-world problems in production environments since 2021. This chapter provides a practical map of applications: major use case categories, key players, technology maturity levels, and deployment examples.

Understanding agent capabilities helps you correctly identify areas for adoption and avoid applying complex agentic systems where a simple API call would suffice.

---

## Taxonomy of AI Agent Applications

### Maturity vs Potential

Not all agent applications are at the same maturity level. Some already generate billions of dollars in revenue; others remain promising prototypes. Understanding this distinction is essential when planning projects.

| Category | Maturity | Example Companies | Market Size |
|-----------|----------|------------------|-------------|
| **Coding Agents** | Production | Cursor, GitHub Copilot, Claude Code, Devin | Dominant: Cursor $2B ARR, Claude Code $500M ARR, Devin $10.2B valuation (as of early 2026). Terminal agents (Claude Code, Codex CLI, Aider) emerged as a new subcategory |
| **Agentic RAG** | Production | Perplexity, Glean, You.com | $9B Perplexity valuation, 22M+ users (as of mid-2025) |
| **Workflow Automation** | Production | n8n, Make, Zapier AI | 200K+ businesses |
| **Voice Agents** | Production | ElevenLabs, VAPI, Retell | 75ms latency achieved |
| **Computer Use Agents** | Beta | Anthropic, ChatGPT agent (formerly Operator) | Active development |
| **Browser Automation** | Production | Browser Use, Stagehand, Playwright MCP | 50-78K+ stars Browser Use (as of early 2026) |
| **Automated Research** | Emerging | AutoResearch (Karpathy), custom agents | Universal optimization pattern |
| **Deep Research** | Production | Gemini Deep Research, OpenAI, OWL | OWL 69.09% on GAIA (#1 open-source, Apache 2.0) |
| **Code Review** | Production | CodeRabbit, Sourcery, custom | Integrated into CI/CD |
| **Data Analysis** | Production | Julius AI, Code Interpreter | Natural language → SQL/viz |
| **Internal Employee** | Production | Glean, custom RAG agents | Corporate knowledge + tools |
| **No-Code/Low-Code Builders** | Production | n8n, Dify, Lovable, Bolt.new | n8n 150K+ stars, Dify $180M valuation, Lovable $100M ARR (as of early 2026) |

---

## Coding Agents: The Dominant Category

### Why Coding Is the Ideal Use Case for Agents

Coding agents have become the most successful category of AI agents, and this is no coincidence:

**Formal verifiability.** Code can be verified automatically — compilation, tests, static analysis. This provides a clear feedback loop for the agent: success or failure.

**Rich tooling.** Agents can use the same tools as developers: editors, terminals, debuggers, version control systems.

**Clear decomposition.** Programming tasks naturally break down into subtasks: understand requirements → find relevant files → make changes → test.

**High ROI.** Development acceleration directly converts into cost savings, justifying investment in the technology.

### Key Players

**Cursor** — an AI-first IDE built on VS Code. Integrates full project context, supports multi-file editing, understands codebase structure. Has become the standard for many developers.

**GitHub Copilot** — the category pioneer, integrated into most IDEs. Evolved from autocomplete to a full-featured agent with Copilot Workspace.

**Devin (Cognition)** — an autonomous software engineer valued at $4B+ (as of mid-2025). Operates in its own development environment and can execute tasks end-to-end.

**Roo Code (Roo Cline)** — an open-source alternative with 20K+ stars (as of early 2026). Runs in VS Code and supports various LLMs.

**Aider** — a CLI-based coding agent (39-42K stars, as of early 2026). Minimalist approach with good integration into existing workflows. Pioneered the Architect/Editor dual-role pattern.

**Developer cost reality (as of early 2026):** Code review agents cost $0.001-$2.00 per PR. Claude Code averages ~$6/day per developer. Cursor Auto mode adds $50-150/month above the subscription. Claude Code on API billing runs ~$120-180/month for active developers. BYOK tools (Cline, Aider) cost $30-100/month depending on model usage.

### Architectural Patterns of Coding Agents

Successful coding agents typically follow several key patterns:

**1. Codebase-aware RAG.** Indexing the entire codebase to understand project structure, dependencies, and coding conventions.

**2. Multi-step reasoning.** Task decomposition: analysis → planning → search → editing → testing.

**3. Tool integration.** Calling LSP (Language Server Protocol), running tests, interacting with Git.

**4. Self-verification.** Running code after changes, analyzing compilation errors, iterative fixes.

---

## Agentic RAG: Intelligent Search

### From Retrieval to Reasoning

Classic RAG (Retrieval-Augmented Generation) is a simple pipeline: find relevant documents → add to context → generate a response. Agentic RAG makes this process intelligent:

- The agent decides which query to formulate
- Evaluates relevance of retrieved results
- Can reformulate the query if results are unsatisfactory
- Synthesizes information from multiple sources
- Verifies facts through cross-referencing

### Key Players

**Perplexity** — the category flagship valued at $9B with 22 million users (as of mid-2025). Combines web search with LLM reasoning. Transparently displays sources.

**Glean** — enterprise search with AI. Indexes a company's internal systems (Confluence, Slack, email) and answers employee questions.

**You.com** — web search with citation and multi-source synthesis.

### Agentic RAG Patterns

**Router Pattern.** The LLM decides which data source to use for a specific query:
- Vector knowledge base
- Full-text search (BM25)
- SQL query against structured data
- Real-time web search

**Multi-Agent Pattern.** Multiple specialized agents work in parallel, each with its own data source. Results are synthesized by a coordinator agent.

---

## Workflow Automation: Agents in Business Processes

### The No-Code/Low-Code Revolution

Workflow automation is the area where AI agents became accessible to a mass audience through no-code platforms. This democratizes the technology, enabling business users to automate processes without programming.

### Key Players

**n8n** — self-hosted workflow automation. 200K+ businesses use the platform. Native MCP support since 2024, 100+ integrations. Open-source, deployable on-premise.

**Make (formerly Integromat)** — visual automation. An AI assistant helps build scenarios.

**Zapier AI** — AI Actions enable agents to perform actions across 6000+ applications.

### Typical Use Cases

**Customer Service.** Automated processing of incoming requests: classification, routing, FAQ responses, escalation of complex cases.

**Sales Pipeline.** Lead enrichment with data, automated follow-up, scoring, and prioritization.

**Document Processing.** Data extraction from documents, automated form filling, cross-system synchronization.

---

## Voice Agents: The Natural Interface

### The Latency Breakthrough

Voice agents long remained a "tomorrow's technology" due to high latency. A 2-3 second delay between utterances made conversations unnatural. In 2024 a breakthrough occurred: ElevenLabs achieved 75ms latency, making conversations virtually indistinguishable from human interaction.

### Technical Stack

**Speech-to-Text.** Whisper (OpenAI), Deepgram, AssemblyAI. Real-time transcription with low latency.

**LLM Processing.** Fast models — Claude Haiku 4.5, GPT-4o mini. Time-to-first-token is critical.

**Text-to-Speech.** ElevenLabs, PlayHT, LMNT. High-quality synthesis with expression control.

**Voice Activity Detection (VAD).** Detecting the start and end of user speech for natural turn-taking.

### Key Players

**ElevenLabs** — leader in voice synthesis quality. 75ms latency in Conversational AI.

**VAPI** — voice agent infrastructure. API-first approach, integration with any LLM.

**Retell AI** — enterprise voice agents. Focus on call center automation.

### Use Cases

**Customer Support.** First-line support, handling routine requests, routing complex cases to human operators.

**Appointment Scheduling.** Automated booking: availability checks, confirmations, reminders.

**Outbound Calls.** Surveys, confirmation calls, gentle reminders.

---

## Computer Use Agents: Full Control

### Concept

Computer Use Agents interact with a computer the same way a human does: they see the screen, move the mouse, press keys. This is a universal interface — the agent can work with any application, even without an API.

### Key Players

**Anthropic Computer Use** — the category pioneer. Claude can see screenshots and generate actions (clicks, text input). Available in beta since October 2024.

**ChatGPT agent** (formerly OpenAI Operator, rebranded July 2025) — OpenAI's browser automation capability, now integrated directly into ChatGPT rather than offered as a standalone product.

### Technical Challenges

**Latency.** Each action requires a screenshot → analysis → decision cycle. Cycles take seconds, not milliseconds.

**Positioning accuracy.** Clicking on the correct element requires precise understanding of UI layout.

**Error Recovery.** If something goes wrong (popup, loading screen), the agent must adapt.

### When to Use

Computer Use is justified when:
- No API exists for the required action
- Working with legacy software is necessary
- The task requires visual verification

Not justified when:
- A programmatic API is available
- Execution speed matters
- High reliability is required

---

## Browser Automation Agents

### Web Specialization

Browser Automation Agents are a subcategory of Computer Use, specialized for web browsers. They are optimized for DOM manipulation, form filling, and navigation.

### Key Players

**Browser Use** — open-source (50-78K+ GitHub stars, as of early 2026). Full AI-reasoning on every browser step — the agent re-reasons on every action rather than caching selectors. Model-agnostic (OpenAI, Anthropic, Gemini, Ollama). Best for unpredictable pages and exploratory tasks.

**Stagehand 2.0** (Browserbase) — extends Playwright with three AI methods: `act()`, `extract()`, `observe()`. Deterministic-first philosophy: uses standard Playwright selectors when possible, falls back to AI only for dynamic elements. Key innovation: **auto-caching of selectors** — first run is expensive (AI figures out selectors), subsequent runs are near-free (cached selectors).

**Playwright MCP** (Microsoft) — uses **accessibility snapshots** instead of screenshots, eliminating the need for a vision model entirely. Reduces token usage by 4x (114K→27K tokens via `@playwright/cli`). Acts as a bridge between coding agents (Claude Code, Cursor) and browser automation.

**Amazon Nova Act** — part of Amazon Nova models. Optimized for web automation in the AWS ecosystem with deterministic action replay.

### The 80/20 Production Pattern

Production browser automation converges on a pragmatic hybrid: **80% deterministic Playwright** for known, stable UI elements (login forms, navigation, standard buttons) and **20% AI** for dynamic or unpredictable elements (changing layouts, A/B test variants, new UI components). The principle: "Don't use AI for a button with a known ID."

**Evolution path:** Prototype (100% Browser Use — fast to build, expensive to run) → MVP (Stagehand — caching reduces cost) → Production (80/20 hybrid — maximum reliability at minimum cost).

---

## Automated Research Agents (The Karpathy Loop)

In April 2026, Andrej Karpathy released **AutoResearch** — 630 lines of Python that launched agents on automated ML experiments. The result: 700 experiments in two days on one GPU, 20 significant optimizations discovered autonomously, 11% training speedup.

The numbers matter less than the pattern they reveal. Karpathy (and subsequently Fortune magazine) coined the term **"The Karpathy Loop"** — a universal optimization pattern with three ingredients:

1. **Agent with file access** — the agent can read and modify code, config files, or any text-based artifact
2. **One objectively testable metric** — a number that can be measured automatically (training loss, test pass rate, latency, cost)
3. **Fixed time limit** — the agent runs for N hours, explores the search space, and reports the best result

The agent autonomously modifies code or configuration, measures the result, and iterates. No human in the loop during the search — only at the beginning (define the metric) and end (evaluate the results).

**Beyond ML:** The pattern is universal. Any domain with a measurable metric and an automatable feedback loop is a candidate:
- **CI/CD optimization** — agent modifies pipeline config, measures build time, iterates
- **A/B testing** — agent generates page variants, measures conversion, iterates
- **Infrastructure tuning** — agent adjusts Kubernetes resource limits, measures latency/cost, iterates
- **SQL query optimization** — agent rewrites queries, measures execution time, iterates
- **Prompt engineering** — agent modifies prompts, measures eval scores, iterates (this is what DSPy automates)

AutoResearch represents a philosophical shift from "agents that help humans do research" to "agents that conduct research independently." The human role shifts from doing to directing — defining the metric, constraining the search space, and evaluating whether the discovered optimizations are genuinely useful or merely overfitting the metric.

21K GitHub stars within weeks of release (as of April 2026) (as of April 2026). Vision: SETI@home-style distributed communities where AI agents conduct research across volunteer compute.

---

## Deep Research Agents: Autonomous Investigation

### Concept

Deep Research Agents are autonomous systems for conducting in-depth investigations. The agent receives a question or topic, then independently formulates sub-questions, searches for information, analyzes sources, and synthesizes findings.

This is not simply "search and summarize" — it is a full research process involving:
- Iterative deepening into the topic
- Verification through multiple sources
- A structured final report
- Transparent source citations

### Key Players

**Gemini Deep Research** (Google) — integrated into Gemini. Creates a research plan, performs multi-step research, and generates structured reports.

**OpenAI Deep Research** — integrated into ChatGPT Pro. Now connects to MCP servers, enabling research over private enterprise data. Supports trusted site whitelisting for enterprise deployments.

**OWL** (CAMEL-AI) — the leading open-source research agent. 69.09% on the GAIA benchmark (#1 open-source, surpassing OpenAI Deep Research by 2.34%). Apache 2.0 license, fully self-hostable. Multi-tool architecture for web browsing, file manipulation, and multi-step reasoning.

### Difference from Agentic RAG

| Aspect | Agentic RAG | Deep Research |
|--------|-------------|---------------|
| Goal | Answer a specific question | In-depth investigation of a topic |
| Time | Seconds | Minutes to hours |
| Output | Brief answer | Structured report |
| Iteration | 1-3 cycles | Many iterations |
| Scope | Focused | Exploratory |

---

## Internal Enterprise Agents

### Code Review Agents

Specialized multi-agent systems that review pull requests for security vulnerabilities, performance issues, style violations, and architectural concerns. Each agent focuses on a narrow domain (see [[../19_Practical_Projects/02_Multi_Agent_System|Multi-Agent Code Review Project]] for a detailed implementation).

**Maturity:** Production-ready. GitHub, GitLab, and third-party tools (CodeRabbit, Sourcery) offer automated AI code review. Enterprise teams deploy custom agents tuned to their coding standards.

**Key pattern:** Multi-agent specialization — one agent per concern (security, performance, style) — outperforms a single "review everything" agent by avoiding cognitive overload.

### Data Analysis Agents

Agents that accept natural language questions about datasets and autonomously write SQL/Python, execute queries, generate visualizations, and interpret results. The agent iterates: if a query returns unexpected results, it investigates further.

**Key Players:** Julius AI, Code Interpreter (OpenAI), Gemini Code Execution. Enterprise deployments use custom agents connected to internal data warehouses via MCP.

**Why agents work well here:** Data analysis has a natural feedback loop — query results are immediately verifiable, and the agent can self-correct by examining the data. The challenge is ensuring the agent does not hallucinate insights from statistical noise.

### Internal Employee Agents

AI assistants embedded in corporate workflows: answering HR policy questions, helping with expense reports, onboarding new employees, searching internal documentation, and routing requests to the right team.

**Architecture:** Typically agentic RAG over internal knowledge bases (Confluence, SharePoint, Notion) with tool access to corporate systems (HRIS, ticketing, calendar). MCP servers provide standardized access to each corporate system.

**Key difference from customer-facing agents:** Higher trust level (employees understand limitations), access to sensitive internal data (requires strict authorization), and integration with SSO/identity systems for audit trails.

---

## No-Code/Low-Code Agent Builders

A rapidly growing segment that allows non-developers (and developers seeking rapid iteration) to build agent workflows through visual interfaces rather than code. These platforms are complementary to the developer frameworks covered in [[../../07_Frameworks/00_Frameworks_Overview|Module 07]] — they target different audiences and use cases.

### Workflow Automation Platforms

**n8n** — 150K+ GitHub stars (as of early 2026), 400+ integrations, open-source with self-hosted option. n8n 2.0 (January 2026) added sandboxed execution, persistent agent memory, 70+ AI nodes, and full traceability for every reasoning step. The self-hosted option is critical for enterprise scenarios with data sovereignty requirements. Native MCP support connects agents to the growing MCP server ecosystem.

**Dify** — 93K+ GitHub stars, $30M raised at $180M valuation, 1.4M installations (as of early 2026). Drag-and-drop visual canvas for building agentic workflows with built-in RAG pipeline, model management, and observability. Self-hostable. Occupies the middle ground between pure no-code (Flowise) and full frameworks (LangGraph) — visual building with code escape hatches.

**Flowise** — 30K+ GitHub stars. A transparent LangChain visual builder where every chain node is visible and editable. Best for teams that want the visual metaphor but need to understand (and modify) the underlying LangChain constructs.

**Langflow** — visual canvas with a live chat pane for instant testing while building. Useful for rapid iteration — modify the graph, test immediately in the same window.

### AI App Builders

A parallel trend: platforms that generate full applications (not just agent workflows) from natural language descriptions.

**Lovable** — $100M ARR in 8 months (fastest-growing European startup ever, as of early 2026). Generates clean React code from descriptions. $20/month. No built-in database — relies on Supabase for persistence.

**Bolt.new** — $40M ARR in 6 months. Built on WebContainer technology (Node.js running in the browser). Token-based pricing (10M tokens/month). Supports multiple frameworks (Next.js, Remix, Astro, SvelteKit).

**v0** (Vercel) — evolved from a component generator to a full-stack platform with built-in databases and integrated deployment. Uses shadcn/ui as the default component library.

**Replit Agent** — the most autonomous of the app builders. 30+ integrations, built-in PostgreSQL/SQLite/KV store, multi-language support (Python, JS, Go, Java). Can create bots, CLIs, and data pipelines — not just web apps.

**Key market split:** Built-in databases (v0, Replit) vs external Supabase dependency (Bolt, Lovable). This is a critical friction point for non-technical users — built-in databases reduce setup steps.

**Security concern for generated code:** AI-generated applications commonly contain SQL injection vulnerabilities, XSS, tokens in client-side code, and missing rate limiting. A human security review before production deployment is mandatory. The correct strategy: vibe-code a prototype → gather stakeholder feedback → human code review → decide whether to refactor or rewrite.

---

## Choosing the Right Use Case

### Criteria for Successful Agent Deployment

When selecting an area for AI agent adoption, several factors should be evaluated:

**1. Formal verifiability of results**

Can you automatically verify that the agent completed the task correctly? Code can be compiled and tested. A factual answer can be checked against sources. Creative writing is subjective.

**2. Tolerance to errors**

What is the cost of failure? For a coding agent, incorrect code simply fails tests. For an agent sending emails to clients — reputational damage. For a medical agent — a potential health risk.

**3. Availability of a feedback loop**

Can the agent receive feedback on the results of its actions? The best use cases are those where the agent can observe outcomes and self-correct.

**4. Economic viability**

What does it cost to have a human perform the task vs. an agent? How many tasks of this type exist? What is the expected ROI?

### Red Flags

Exercise caution if:

- "We need AI for everything" — scope is too broad
- No clear success criteria exist
- Results cannot be verified automatically
- The cost of failure is very high
- No data is available for training/fine-tuning

---

## Key Takeaways

1. **Coding Agents** — the most mature category thanks to the formal verifiability of code

2. **Agentic RAG** — production-ready for enterprise search and Q&A systems

3. **Workflow Automation** — no-code platforms make agents accessible to business users

4. **Voice Agents** — the latency breakthrough unlocked production use cases

5. **Computer Use and Browser Automation** — emerging categories, justified when no API exists

6. **Deep Research** — autonomous multi-hour investigation

7. **Key selection criteria**: verifiability, tolerance to errors, feedback loop, ROI

8. **"Start simple"** — begin with clearly scoped use cases and expand as experience accumulates

---

## Related Materials

- [[01_What_is_AI_Agent|What Is an AI Agent]]
- [[02_Agent_Architectures|Agent Architectures]]
- [[07_Code_Generation_Agents|Coding Agents: Code Generation]]
- [[06_Computer_Use_Agents|Computer Use Agents]]
- [[../06_RAG/05_Advanced_RAG|Advanced RAG]]

---

## Navigation
**Previous:** [[08_Process_Reward_Models|Process Reward Models]]
**Next:** [[10_Resource_Optimization|Resource-Aware Optimization]]
