# Context Engineering

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Prompt Engineering
**Previous:** [[04_Prompt_Optimization|Prompt Optimization]]
**Next:** [[../03_AI_Agents_Core/01_What_is_AI_Agent|What is an AI Agent]]

---

## Evolution: from Prompt Engineering to Context Engineering

In 2025, Andrej Karpathy proposed the term **Context Engineering**, which was quickly adopted by the industry as a more accurate description of working with LLMs. Harrison Chase (CEO of LangChain) formalized the shift further: "When agents mess up, they mess up because they don't have the right context; when they succeed, they succeed because they have the right context."

This is not merely a rebranding. Research shows that **70% of LLM application errors stem from incomplete, irrelevant, or poorly structured context — not from model limitations.** The same Claude Opus 4.5 model differs by 17 tasks on SWE-bench Verified depending solely on the scaffolding and context harness wrapping it. Investing in better context engineering yields more improvement than upgrading the model.

### Why Context, not Prompt?

**Prompt Engineering** focuses on: instruction formulation, example selection (few-shot), query structure.

**Context Engineering** encompasses: the entire context window, dynamic information management, persistent memory across sessions, coordination between agents. It manages seven layers of information: system prompt, conversation history, retrieved documents, tool definitions, memory, task context, and world model.

> "Prompt Engineering is like writing a good letter. Context Engineering is like organizing the entire correspondence."

---

## Karpathy's Analogy: LLM as CPU

Karpathy proposed a powerful analogy for understanding LLMs:

- **CPU (processor)** → **LLM (model weights)** — processing
- **RAM (random-access memory)** → **Context Window** — working memory
- **HDD/SSD (storage)** → **Vector DB** — long-term storage
- **OS (operating system)** → **Agent Loop** — resource management

### Key Insight

The context window is the only "working memory" of an LLM. Everything the model "knows" at generation time must be present in the context. This creates a fundamental constraint and defines working strategies.

---

## The Lost-in-the-Middle Problem

Before diving into strategies, it is essential to understand a counterintuitive property of LLMs: they process information at the **beginning and end** of the context window significantly better than in the middle. This "Lost-in-the-Middle" effect (Liu et al., 2023) means that position in the context influences model attention more than semantic relevance.

The practical implication is architectural: critical instructions belong at the start of the context (system prompt), key constraints should be repeated at the end (the "sandwich" pattern), and important retrieved documents should not be buried between less relevant ones. When assembling context for an agent, the ordering of information is not cosmetic — it directly affects the quality of reasoning.

This effect intensifies with context length. At 4K tokens, the middle is still reasonably attended. At 128K tokens, information in the middle may effectively be invisible. For long-context applications, hierarchical summarization or progressive disclosure (loading detail on demand) outperforms simply concatenating all available text.

---

## Four Context Engineering Strategies

Karpathy and his followers identified four core context management strategies — **WSCI** (Write, Select, Compress, Isolate).

### 1. WRITE — Persisting Outside the Context

**Problem:** The context window is limited (8K–200K tokens); information is lost between sessions.

**Solution:** Write important information to external storage.

**What to persist:** user facts (preferences, history), intermediate work results (drafts), decisions made and their rationale, errors and how they were resolved.

**WRITE Tools (from simplest to most complex):**

- **File-based memory** (e.g., Claude Code's CLAUDE.md) — plain text files the agent reads and writes. Human-readable, git-versioned, zero infrastructure. Effective for developer tools with modest memory scale
- **Scratchpads** — temporary agent notes for complex computations
- **MCP memory servers** — standardized read/write interface, backend-agnostic (can use files, SQL, or vector DB behind the protocol)
- **Vector DBs** — semantic search over memory for RAG systems, scales to millions of items
- **MemGPT / Letta** — one of the early approaches (2023) to automatic paged memory management for long dialogues, inspired by OS virtual memory. The concept (self-directed memory via tool calls, tiered storage) has been broadly influential, though simpler patterns often suffice in production — see [[../03_AI_Agents_Core/05_Memory_Systems|Memory Systems]] for a detailed comparison

---

### 2. SELECT — Choosing Relevant Context

**Problem:** There is a lot of information, but not all of it is relevant to the current query.

**Solution:** Intelligent selection of the needed information.

**SELECT Techniques:**

- **Semantic search** — meaning-based search via embeddings (primary method for RAG)
- **Keyword search** — exact term matching (technical terms, codes)
- **Hybrid search** — combination of semantic and keyword search (production systems)
- **Metadata filtering** — filtering by attributes (temporal data, categories)
- **Reranking** — re-ranking results (improving precision)

**SELECT Process:**
1. Receive the user query
2. Find candidates via semantic/hybrid search
3. Filter by metadata (date, category, source)
4. Rerank via cross-encoder
5. Select top-K for context

---

### 3. COMPRESS — Compressing Information

**Problem:** There is too much information, even relevant information.

**Solution:** Compression without loss of meaning.

**COMPRESS Techniques:**

- **Summarization** — ratio 5–10x, loss of details, applied to long documents
- **Fact extraction** — ratio 10–20x, loss of context, for structured data
- **Sliding window** — ratio 2–3x, loss of older content, for dialogues
- **Hierarchical compression** — ratio 3–5x, loss of lower levels, for structured documents

**Hierarchical Compression:**
- Level 1: Full document (10K tokens)
- Level 2: Summary (2K tokens)
- Level 3: Key facts (500 tokens)
- Level 4: Single sentence (50 tokens)

Depending on available space in the context, the appropriate level of detail is used.

---

### 4. ISOLATE — Isolation via Multi-Agent

**Problem:** A single agent cannot hold everything needed for a complex task in its context.

**Solution:** Splitting context across specialized agents.

**ISOLATE Advantages:** each agent fully utilizes its own context window, specialization reduces noise in the context, parallel processing is possible, easier to debug and improve.

**Isolation Patterns:**

- **Sequential pipeline** — agents work sequentially (linear workflows)
- **Hierarchical** — Orchestrator + Executors (complex tasks)
- **Debate/Critique** — Proposer + Critic (quality improvement)
- **Parallel specialists** — parallel processing (independent subtasks)

---

## Context Engineering for Different Tasks

### Customer Support Agent

- **WRITE**: Preferences → Vector DB, History → Redis
- **SELECT**: Hybrid search over FAQ + Customer history
- **COMPRESS**: Sliding window of last 10 messages
- **ISOLATE**: Intent → Retrieval → Response → QA

### Code Assistant

- **WRITE**: Changes → Git, Decisions → Scratchpad
- **SELECT**: Semantic + Keyword over codebase index
- **COMPRESS**: Signatures + Docstrings (without function bodies)
- **ISOLATE**: Planner → Coder → Tester → Reviewer

---

## Context Engineering Metrics

- **Context Utilization** — % of context window used → target 70–90%
- **Relevance Score** — relevance of retrieved context → above 0.8
- **Memory Hit Rate** — % of queries where memory helped → above 60%
- **Compression Ratio** — compression without quality loss → 3–5x
- **Token Efficiency** — useful tokens / total tokens → above 0.7

---

## Context Engineering Anti-Patterns

### Context Stuffing
Cramming everything into the context without filtering. Leads to: low response relevance, high token costs, loss of important information in the "noise".

### Static Context
Using the same context for all queries. Problems: irrelevant information, stale data, inefficient token usage.

### Lack of Memory Strategy
Processing every query from scratch. Consequences: loss of interaction history, repeated questions to the user, inability to learn from experience.

### Correct Approach: Dynamic Context Management
Adaptive context assembly for each query, balance between completeness and relevance, persistent memory for important facts.

---

## MCP as Context Engineering Infrastructure (2024-2025)

The **Model Context Protocol (MCP)**, introduced by Anthropic in late 2024 and widely adopted through 2025, is fundamentally a context engineering protocol. MCP maps directly to the WSCI framework:

**MCP Resources → SELECT:** MCP resources provide a standardized way for agents to discover and retrieve context from external systems. Instead of custom RAG pipelines for each data source, MCP resources expose databases, file systems, APIs, and knowledge bases through a uniform interface. The agent (or the host application) selects which resources to load into context based on the current task.

**MCP Tools → WRITE:** MCP tools enable agents to write information to external systems — creating files, updating databases, sending messages. This is the WRITE strategy formalized as a protocol: the agent persists important results and decisions outside the context window.

**MCP Prompts → COMPRESS/SELECT:** MCP prompt templates provide pre-structured context for specific tasks. Instead of the agent assembling context from scratch, a prompt template defines what information to include and how to structure it — a form of compression and selection baked into the protocol.

**Why MCP matters for context engineering:** Before MCP, every agent framework implemented its own approach to tool definitions, resource access, and context assembly. MCP standardizes this, enabling interoperable context engineering across tools, IDEs, and agent frameworks. The same MCP server providing access to a Postgres database works identically in Claude Code, Cursor, Windsurf, or any custom agent.

**MCP's context engineering impact by the numbers (early 2026):** 97M+ monthly SDK downloads, 10,000+ community servers, native support in all major AI platforms. MCP is no longer emerging infrastructure — it is the industry-standard context engineering protocol, governed by the Agentic AI Foundation under the Linux Foundation.

**MCP primitives as WSCI enablers:**

| WSCI Strategy | MCP Primitive | Example |
|---------------|---------------|---------|
| **WRITE** | Tools | Agent writes analysis results to a database via `db_insert` tool |
| **SELECT** | Resources + Resource Templates | Agent discovers and reads relevant documents via `docs://project/{topic}` |
| **COMPRESS** | Prompts | Server provides pre-structured prompt templates that include only essential context |
| **ISOLATE** | Tasks + Sampling | Long-running operations run asynchronously (Tasks); sub-agents delegate reasoning via Sampling |

The WSCI framework and MCP are complementary: WSCI describes *what* to do with context (write, select, compress, isolate), while MCP provides the *how* — standardized protocol primitives that implement these strategies across any AI platform.

### Agentic Search as Context Engineering

A significant development in 2024-2025 is the emergence of **agentic search** — where the agent itself manages the SELECT strategy dynamically.

Traditional RAG: fixed pipeline (query → embed → retrieve → generate). The SELECT strategy is static — the same retrieval pipeline runs for every query.

Agentic search: the agent reasons about what information it needs, formulates targeted queries, evaluates retrieved results, and iterates if the results are insufficient. The agent applies Chain-of-Thought to the SELECT strategy itself:

1. Analyze the question — what information is needed?
2. Formulate a search strategy — which sources, what queries?
3. Execute retrieval — run searches in parallel where possible
4. Evaluate results — is the retrieved context sufficient?
5. Iterate if needed — refine queries, try different sources

This is the SELECT strategy elevated from a pipeline component to an agent capability. Examples: Perplexity's search agent, OpenAI's deep research, Claude's web search integration — all implement agentic search patterns.

## Karpathy's Context Hierarchy

A practical framework for thinking about the four levels of context that an agent manages, each with different update frequencies and management strategies:

| Level | Analogy | Content | Update Frequency | Management |
|-------|---------|---------|-----------------|------------|
| **Level 1: System Prompt** | Constitution | Agent role, rules, constraints | Rarely (per deployment) | Static, human-authored |
| **Level 2: Retrieved Context** | Working desk | Documents, tool results, RAG output | Per query | SELECT + COMPRESS strategies |
| **Level 3: Conversation Context** | Working memory | Current dialog, recent messages | Per turn | Sliding window, summarization |
| **Level 4: Agent Memory** | Long-term experience | Facts learned, user preferences, past decisions | Across sessions | WRITE strategy, vector DB, Letta |

The hierarchy implies a management strategy: Level 1 is stable and cacheable (ideal for prefix caching). Level 2 is dynamic but scoped to the current task. Level 3 requires active compression as conversations grow. Level 4 requires a persistence layer (see [[../03_AI_Agents_Core/05_Memory_Systems|Memory Systems]]).

---

## Self-Directed Context Engineering

The most significant trend in 2026 context engineering is the shift from static pipelines to **self-directed context management** — where the agent itself decides what to see, what to retrieve, what to summarize, and what to evict.

In traditional RAG, the context assembly pipeline is fixed: query → embed → retrieve → inject into prompt. The same pipeline runs for every query regardless of what the agent actually needs. Self-directed context engineering gives the agent control over its own context.

**Letta's LLM-as-OS paradigm** takes this furthest: the agent manages a three-tier memory system (Core Memory always in prompt, Archival Memory out-of-context, agent-editable via tool calls). The agent decides when to write to memory, when to search its archives, and when to forget — through explicit tool calls, not pipeline logic. See [[../03_AI_Agents_Core/05_Memory_Systems|Memory Systems]] for details.

**Claude Code's compaction pattern** illustrates a more practical approach: when the conversation context exceeds approximately 60% of the maximum window, the system automatically summarizes earlier turns into a compressed representation. The agent continues working with the summary + recent context, enabling effectively infinite-length sessions within a finite window.

**Agent Teams** decompose a complex task so that each sub-agent receives only the context relevant to its subtask. Instead of one agent with an overloaded context, multiple agents each have focused, high-quality context. This is the ISOLATE strategy applied to context engineering — see [[../04_Multi_Agent_Systems/01_MAS_Basics|Multi-Agent Basics]].

The common thread: context engineering is shifting from something engineers build around the model to something the agent manages for itself. The agent is becoming its own context engineer.

---

## Token Budget Management

In an agentic loop where the LLM is called 10-20 times per task, every wasted token compounds. Token budget management is the discipline of controlling what enters the context and how much of the window it consumes.

**Just-in-time tool loading** is the highest-impact optimization. Rather than loading all tool descriptions into every LLM call (a common pattern in early agent frameworks), a lightweight classifier first determines which tools are relevant to the current step, and only those descriptions are included. This reduces tool-related context from 77K tokens (for a large tool set) to approximately 8.7K tokens — an **85% reduction**. The classifier can be rule-based (keyword matching on the query), semantic (embed the query, match against tool description embeddings), or LLM-based (a cheap model routes to relevant tools).

**Tool description budgeting:** Each tool description consumes 200-500 tokens. A rule of thumb: tool descriptions should not exceed **10% of the available context window**. For a 128K context model, that is 12.8K tokens — roughly 25-60 tools before tool descriptions alone crowd out actual task context. Beyond that threshold, dynamic filtering becomes mandatory.

**Dynamic context filtering** selects which context elements to include based on the current step:

| Filter Type | Mechanism | Speed | Quality |
|-------------|-----------|-------|---------|
| **Keyword** | Regex/string match on query | <1ms | Low (misses synonyms) |
| **Semantic** | Embedding similarity | ~10ms | Medium |
| **LLM-based** | Cheap model classifies relevance | ~200ms | High (but adds latency and cost) |

For most production systems, semantic filtering is the sweet spot — fast enough for real-time and accurate enough to avoid loading irrelevant context.

---

## Production Context Patterns

Five patterns that consistently reduce context costs and improve quality in real agent deployments:

**1. Header Deduplication.** In batch processing scenarios, the same system prompt, instructions, and formatting headers appear in every request. Deduplicating these shared headers can save 20K+ tokens per batch. The pattern: extract common headers into a shared prefix, rely on provider prefix caching to avoid re-processing.

**2. Dynamic Tool Loading.** Classify the user's intent with a fast model or heuristic, then load only the 3-5 tools relevant to that intent. This is just-in-time loading applied at the framework level. Reduces the constant overhead of tool descriptions from every LLM call.

**3. Conversation Compaction.** When conversation context exceeds a threshold (typically 60-70% of the window), automatically summarize earlier turns. The summary preserves key decisions and facts while discarding verbatim dialogue. Claude Code implements this as automatic compaction; custom agents can implement it as a periodic summarization step.

**4. Semantic Caching.** Before calling the LLM, embed the current query and search a cache of recent query-response pairs. If a cached query has similarity above a threshold (typically >0.95), return the cached response. This can reduce LLM calls by 30-40% for workloads with repeated patterns (customer support, FAQ-style queries). Not suitable for creative tasks where variety is desired.

**5. Progressive Disclosure.** Instead of loading the full context for a task upfront (which might be 500K tokens of codebase or documentation), load a hierarchical summary first (10-15K tokens). The agent requests detail on specific sections as needed. This mirrors how humans work with large information sets — overview first, detail on demand.

---

## Token-Optimized Output Formats

JSON is the default structured output format, but it carries significant overhead: braces, quotes, colons, commas, and key names consume tokens that carry no semantic information. For high-volume applications, this overhead is material.

**TOON (Token-Optimized Output Notation)** is an emerging approach that achieves **69% token reduction** compared to JSON by using minimal delimiters (pipe-separated values, newline-delimited records). The savings come from eliminating repeated key names and structural characters.

| Format | Tokens for 10-field record | Relative Cost |
|--------|---------------------------|---------------|
| JSON | ~45 tokens | 1.0x |
| TOON/minimal delimiters | ~14 tokens | 0.31x |

**When to use JSON:** external APIs, persistent storage, human-readable logs, any boundary where interoperability matters.

**When to use TOON or minimal formats:** internal agent-to-agent communication, intermediate tool results that will be consumed by another LLM call, high-volume structured output where every token costs money. At 1M+ requests per day, the difference between JSON and a minimal format can exceed $100K per year.

In practice, many production systems use JSON at system boundaries (API inputs/outputs) and minimal formats internally (agent-to-agent, tool results).

---

## The LLM Wiki Pattern

In April 2026, Karpathy published **LLM Wiki** — a concept that quickly went viral as an "anti-RAG" pattern. The core idea: instead of performing retrieval from raw documents on every query, **have an LLM compile the knowledge into a structured wiki once, then maintain it incrementally**.

**Three-layer architecture:**

1. **Raw sources** — immutable documents (articles, papers, data). Never modified.
2. **Wiki** — markdown files maintained by the LLM: summaries, entity descriptions, concept explanations, cross-references. The LLM creates and updates these.
3. **Schema** — configuration (analogous to CLAUDE.md) defining structure, categories, and maintenance workflows.

**Three operations:**

- **Ingest** — process a new source, update relevant wiki pages with new information
- **Query** — search the wiki + synthesize an answer with citations back to sources
- **Lint** — health-check: find contradictions, outdated data, orphan pages, coverage gaps

**When LLM Wiki beats RAG:** For small-to-medium knowledge bases (hundreds to low thousands of documents) where questions require synthesis across multiple sources. A compiled wiki with pre-computed cross-references enables answers that naive RAG — which retrieves individual chunks without understanding their relationships — simply cannot produce. Karpathy's key insight: "The tedious part of maintaining a knowledge base is not the reading or the thinking — it's the bookkeeping." LLMs do not tire of bookkeeping.

**When RAG still wins:** Large corpora (millions of documents), real-time data that changes continuously, cases where provenance and citation to exact source passages are required, and any scenario where the knowledge base does not fit in the context window even as a compiled wiki. See [[../06_RAG/05_Advanced_RAG|Advanced RAG]] for the full taxonomy.

The two approaches are complementary: LLM Wiki for curated, synthesis-heavy knowledge; RAG for large-scale, real-time retrieval. Some production systems use both — a wiki for core domain knowledge and RAG for long-tail queries.

---

## Related Topics

- [[04_Prompt_Optimization|Prompt Optimization]] — foundational prompt techniques
- [[../06_RAG/01_RAG_Basics|RAG Basics]] — the SELECT strategy in detail
- [[../04_Multi_Agent_Systems/01_MAS_Basics|Multi-Agent Basics]] — the ISOLATE strategy
- [[../03_AI_Agents_Core/05_Memory_Systems|Memory Systems]] — the WRITE strategy
- [[../05_MCP/01_MCP_Basics|MCP Basics]] — standardized context engineering protocol

---

## Key Takeaways

1. **Context engineering provides more ROI than model upgrades.** 70% of errors come from context, not model quality. The same model scores 17 tasks differently on SWE-bench depending on scaffolding alone. Invest in better context before switching models.

2. **4 WSCI Strategies:** **W**RITE (persist outside context), **S**ELECT (choose only relevant), **C**OMPRESS (reduce without losing meaning), **I**SOLATE (split across agents). Every context management decision maps to one of these.

3. **Lost-in-the-Middle:** LLMs attend to context beginnings and ends better than middles. Place critical instructions at the start, repeat constraints at the end. This worsens with context length.

4. **Self-directed context engineering** is the 2026 trend: agents managing their own context (Letta's agent-editable memory, Claude Code's compaction, Agent Teams). The agent is becoming its own context engineer.

5. **Token budget management** is mandatory for production agents. Just-in-time tool loading (85% reduction), 10% context budget for tool descriptions, semantic filtering for dynamic context selection.

6. **LLM Wiki is the anti-RAG pattern** for curated, synthesis-heavy knowledge bases. RAG wins for large corpora and real-time data. Both are complementary.

7. **MCP is the industry-standard context engineering protocol.** 97M+ monthly downloads, 10,000+ servers. WSCI maps directly to MCP primitives.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Prompt Engineering
**Previous:** [[04_Prompt_Optimization|Prompt Optimization]]
**Next:** [[../03_AI_Agents_Core/01_What_is_AI_Agent|What is an AI Agent]]
