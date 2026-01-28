# Context Engineering

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Prompt Engineering
**Previous:** [[04_Prompt_Optimization|Prompt Optimization]]
**Next:** [[../03_AI_Agents_Core/01_What_is_AI_Agent|What is an AI Agent]]

---

## Evolution: from Prompt Engineering to Context Engineering

In 2025, Andrej Karpathy proposed the term **Context Engineering**, which was quickly adopted by the industry as a more accurate description of working with LLMs.

### Why Context, not Prompt?

**Prompt Engineering** focuses on: instruction formulation, example selection (few-shot), query structure.

**Context Engineering** encompasses: the entire context window, dynamic information management, persistent memory across sessions, coordination between agents.

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

## Four Context Engineering Strategies

Karpathy and his followers identified four core context management strategies — **WSCI** (Write, Select, Compress, Isolate).

### 1. WRITE — Persisting Outside the Context

**Problem:** The context window is limited (8K–200K tokens); information is lost between sessions.

**Solution:** Write important information to external storage.

**What to persist:** user facts (preferences, history), intermediate work results (drafts), decisions made and their rationale, errors and how they were resolved.

**WRITE Tools:**

- **Scratchpads** — temporary agent notes for complex computations
- **MemGPT (Letta)** — automatic paged memory management for long dialogues
- **LangMem** — persistent LangChain memory for production agents
- **Vector DBs** — semantic search over memory for RAG systems

**MemGPT Architecture:**
MemGPT implements the virtual memory concept from operating systems. The context is divided into: Core Memory (always in context: system prompt, active facts), Archival Memory (long-term storage in a vector DB), Recall Memory (dialogue history). The agent automatically moves information between levels depending on relevance.

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

## Related Topics

- [[04_Prompt_Optimization|Prompt Optimization]] — foundational prompt techniques
- [[../06_RAG/01_RAG_Basics|RAG Basics]] — the SELECT strategy in detail
- [[../04_Multi_Agent_Systems/01_MAS_Basics|Multi-Agent Basics]] — the ISOLATE strategy
- [[../03_AI_Agents_Core/05_Memory_Systems|Memory Systems]] — the WRITE strategy

---

## Key Takeaways

1. **Context Engineering > Prompt Engineering** — managing the entire context matters more than prompt wording

2. **4 WSCI Strategies:**
   - **W**RITE — persist important data outside the context
   - **S**ELECT — choose only what is relevant
   - **C**OMPRESS — compress without losing meaning
   - **I**SOLATE — split context across agents

3. **Context = RAM for an LLM** — a limited resource that requires optimization

4. **2024–2025 Tools:** MemGPT (Letta), LangMem, vector DBs with reranking

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Prompt Engineering
**Previous:** [[04_Prompt_Optimization|Prompt Optimization]]
**Next:** [[../03_AI_Agents_Core/01_What_is_AI_Agent|What is an AI Agent]]
