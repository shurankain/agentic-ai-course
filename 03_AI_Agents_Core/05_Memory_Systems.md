# Agent Memory Systems

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[04_Planning|Agent Planning]]
**Next:** [[06_Computer_Use_Agents|Computer Use Agents]]

---

## Introduction

Memory is what transforms a set of isolated interactions into a coherent experience. Without memory, an agent is like a person with amnesia: every conversation starts from scratch, every mistake is repeated, every lesson is instantly forgotten.

Imagine an assistant that asks your name every day, does not remember your preferences, and repeats the same mistakes. Formally it functions, but its usefulness is minimal. Now imagine an assistant that remembers the context of your projects, knows your preferences, learns from its mistakes, and accumulates expertise. The difference is memory.

This chapter covers how to build a memory system for an AI agent that enables it to retain context, learn from experience, and personalize interactions.

---

## Types of Memory

### Cognitive Science of Memory: Theoretical Foundations

AI agent memory systems are directly inspired by human memory research in cognitive psychology. Understanding these foundations helps design more effective systems.

**Atkinson-Shiffrin Model (1968)**

The classic model describes memory as a sequence of stores with different retention durations. The sensory register holds information for fractions of a second, short-term memory for seconds and minutes, long-term memory for years and decades.

For AI agents, this model transforms into three levels: incoming tokens (instant processing), context window (session), vector store (persistence). Information flows through these levels, and at each stage a decision is made about what to retain and what to discard.

**Encoding and Retrieval**

**Encoding** — transforming information for storage. Encoding quality determines retrieval quality. Poor encoding (text without context) — hard to retrieve. Good encoding (embeddings + metadata + relationships) — easily found.

**Retrieval** — accessing information from storage. Recognition (similarity matching) vs Recall (generation based on retrieved data).

**Levels of Processing Theory**

Depth of processing matters more than time. Three levels: shallow (raw text), intermediate (embedding), deep (embedding + metadata + relationships + categorization).

Conclusion: investing in rich encoding pays off in retrieval quality.

**Memory Effects**

**Primacy effect** — first items are remembered better, requiring explicit preservation when the window overflows.

**Recency effect** — recent items are easier to recall, important for ranking long-term memory.

**Von Restorff effect** — unusual items stand out, justifying importance scoring.

**Context-dependent memory** — easier to recall in a similar context, explaining the effectiveness of similarity-based retrieval.

These principles explain why certain architectural decisions work.

### Memory Types and Their Roles

**Short-term memory** — a buffer for the current session, storing conversation history and temporary data. Typical implementation is a sliding window of the last N messages. Problem: important information can "fall off" the window. Solution — summarization: old messages are compressed into a summary, preserving key information at a smaller size.

**Long-term memory** — persistent storage across sessions, typically implemented via vector databases. Items are converted to embeddings for semantic search. Requires active management: prioritizing the important and "forgetting" the outdated.

**Episodic memory** — stores complete interaction histories: task context, agent actions, outcomes, and lessons learned. Valuable for learning: when facing a new task, the agent finds similar past episodes and applies accumulated experience.

**Semantic memory** — generalized knowledge and stable patterns abstracted from specific events. Accumulates gradually: if the agent observes a pattern multiple times, it converts it into a general rule.

**Working memory** — a structured representation of the current task: goal, active subtasks, variables, constraints. Differs from short-term memory in its organization. Cleared upon task completion; important items are transferred to long-term memory.

---

## Integrated Memory Architecture

An effective memory system combines all types. For each query, context is assembled from four sources: working memory (current state), semantic (applicable knowledge), episodic (relevant experience), short-term (recent history). Order matters — the context window is limited, and priority is typically given to current state and recent history.

After task completion, consolidation occurs — transferring important items from short-term to long-term memory. The process includes: recording the episode, extracting important facts, updating semantic knowledge, clearing working memory. Consolidation can be synchronous (immediate) or asynchronous (in the background).

### Consolidation Theory

Consolidation is the process of stabilizing memory after encoding. Three types with AI analogs:

**Synaptic** (minutes-hours) — immediate filtering of important items at the end of a session.

**Systemic** (days-weeks) — periodic reorganization: analyzing episodes, identifying patterns, deduplication.

**Reconsolidation** (upon retrieval) — updating importance when accessed.

**Four phases of consolidation:**

1. **Extraction** — evaluating items against criteria (explicit importance, semantic significance, mention frequency), discarding those below threshold.

2. **Integration** — finding related items, reinforcing existing ones or creating new items with explicit connections.

3. **Pruning** — deduplicating similar items, removing low-priority ones, maintaining compactness.

4. **Abstraction** — converting recurring situations into generalized rules (semantic memory).

**Background consolidation** — analogous to sleep, runs during low load or on a schedule. Episode replay, pattern discovery, knowledge abstraction, cleanup. Does not block the agent but is critical for memory quality.

---

## MemGPT: Virtual Context Management (Historical Reference)

MemGPT (UC Berkeley, 2023) introduced an influential concept: giving an LLM the illusion of "infinite" memory through context virtualization, inspired by virtual memory in operating systems.

**Core idea:** The model's context window is "RAM" (fast but limited), and external storage (vector DB) is "disk" (slower but unlimited). A Memory Manager dynamically "swaps" information between context and storage — loading relevant data in, offloading stale data out.

**Architecture:** Main Context (system prompt + working context + recent messages) and External Storage (archival memory via vector DB + recall memory for key facts). The LLM calls memory_read/memory_write functions to manage its own memory — self-directed rather than developer-programmed.

**Key innovations:** Self-directed memory management (the LLM decides what to save and retrieve via tool calls), a heartbeat mechanism for proactive background consolidation, and the separation of archival vs. recall memory tiers.

**Legacy:** While MemGPT as a specific system saw limited production adoption (the Letta project that grew out of it pivoted to a broader agent platform), its core ideas — self-directed memory, tiered storage, and tool-based memory operations — became foundational patterns. Modern production memory systems (Claude Code's file-based memory, MCP-based memory servers) are simpler but embody the same principles: the agent manages its own persistence through explicit operations rather than hoping the context window is large enough.

---

## Memory Storage Decision Guide

Choosing the right storage backend depends on scale, access patterns, and infrastructure constraints:

| Scenario | Recommended Storage | Why |
|----------|-------------------|-----|
| Developer tools, CLI agents (hundreds of items, human-readable) | **File-based** (CLAUDE.md pattern) | Git-versioned, debuggable, zero infrastructure, works offline |
| Enterprise agent with 10K-100K memories, existing PostgreSQL | **PostgreSQL + pgvector** | Single database for relational + vector data, ACID transactions, familiar tooling |
| High-scale production, 100K+ memories, sub-10ms latency required | **Dedicated vector DB** (Qdrant, Weaviate, Pinecone) | Purpose-built ANN indexes, horizontal scaling, managed SaaS options |
| Structured data (user profiles, preferences, config) | **SQL database** (PostgreSQL, SQLite) | Schema enforcement, complex queries, joins, aggregations |
| Multi-agent system, backend-agnostic | **MCP-exposed memory** | Any backend behind a standard protocol, composable, shareable across agents |
| Embedded / edge / local-first agents | **LanceDB** or **SQLite + vector extension** | No server, file-based, embeddable, good for desktop/mobile agents |
| Prototype / proof-of-concept | **JSON files** or **in-memory** | Zero setup, iterate fast, migrate later |

**Decision flow:** Start with the simplest option that meets your scale. File-based or SQLite for prototypes → pgvector when you need SQL + vectors → dedicated vector DB when you hit scale limits. Use MCP as the interface layer so storage can be swapped without changing agent code.

---

## Letta: From MemGPT Concept to Production Memory

The MemGPT concept described above was influential but saw limited direct adoption. **Letta** (15K+ GitHub stars, as of early 2026) is its production evolution — a platform that takes the core ideas (self-directed memory, tiered storage, tool-based operations) and makes them work at scale.

**The LLM-as-Operating-System paradigm:** Letta treats the LLM as a CPU managing its own memory hierarchy. Three tiers of memory map to hardware analogy: **Core Memory** (always loaded in prompt — analogous to CPU registers), **Recall Memory** (searchable conversation history — analogous to RAM), and **Archival Memory** (large-scale vector-indexed storage — analogous to disk). The agent accesses all three tiers through explicit tool calls.

**The key differentiator: agent editability.** Unlike systems where memory is written by the application and read by the agent, Letta agents actively **update, delete, and reorganize** their own memory blocks. The agent decides what is worth remembering, what is outdated, and how to structure its knowledge. This is self-directed context engineering at the memory level — see [[../../02_Prompt_Engineering/05_Context_Engineering|Context Engineering]].

**Core Memory** consists of named blocks (persona, human, goals, preferences) that are version-controlled. The agent modifies these blocks as it learns. For example, after discovering a user prefers concise answers, the agent updates its "human" block. This update persists across sessions and influences all future interactions.

**Production validation:** Letta-based agents ranked #1 among open-source agents on Terminal-Bench and #4 overall (as of early 2026), demonstrating that self-directed memory translates to measurable task performance.

**Alternative approaches to agent memory:**

**Mem0** takes a different path — middleware for automatic fact extraction. Instead of the agent managing its own memory, Mem0 intercepts conversations and automatically extracts facts, preferences, and patterns. This is simpler to integrate (plug-and-play) but gives the agent less control over what gets remembered.

**Amazon Bedrock AgentCore** introduces **episodic memory** — agents remember specific situations, decisions, and outcomes (not just facts). When a similar situation arises, the agent can recall how it handled it before. This goes beyond semantic memory (knowing facts) to experiential memory (knowing what worked).

The three approaches represent different points on the automation-control spectrum: Mem0 (fully automated, no agent control), Letta (agent-directed), and Bedrock episodic memory (experience-based, automated but structured).

---

## Memory Escalation Ladder

Start simple. Add complexity only when the current approach fails. Each step on the ladder addresses a specific failure mode:

| Level | Approach | When to Escalate |
|-------|----------|-----------------|
| 1 | **Sliding window** | Default starting point — keep last N messages |
| 2 | **Conversation summarization** | When conversations exceed context limits → compress older turns |
| 3 | **Mem0** (auto-fact extraction) | When users complain the system "doesn't remember" across sessions |
| 4 | **Letta** (agent-editable memory) | When the agent repeats mistakes or fails to learn from experience |
| 5 | **Agentic RAG** (active research) | When accuracy drops below 50% on complex multi-source queries |
| 6 | **Graph RAG** (entity relationships) | When users ask about relationships between entities ("how is X connected to Y?") |

**Escalation signals in production:**
- Users complain "you forgot" or "we already discussed this" → move from Level 1 to Level 3
- Agent makes the same mistake twice on similar inputs → move to Level 4 (agent needs to learn from experience)
- Accuracy on multi-hop questions is below 50% → move to Level 5 (retrieval is not adaptive enough)
- Domain is relationship-heavy (legal, healthcare, finance) → consider Level 6

This is not a hierarchy of quality — it is a hierarchy of complexity. Level 1 is correct for many production agents. Escalate only when you have evidence that the current level is insufficient.

---

## Retrieval-Augmented Memory

Retrieval-Augmented Memory (RAM) is a pattern where the agent dynamically retrieves relevant memory before each action, instead of keeping everything in context. Analogous to how a person "recalls" what is needed when a task arises.

Technically: the current query is converted to a vector, compared against memory item vectors, and the closest matches are added to context. This solves the limited context problem: memory can contain millions of items, but only relevant ones are used at any given moment.

Challenges: quality depends on the quality of embeddings and search. Poor search produces "noise." The "cold start" problem: at the beginning of a conversation there is little context, the query is vague, and the needed memories are not found.

### Memory and RAG: A Unified Pattern

Memory systems and RAG are the same pattern with different data. Both: embedding-based indexing → similarity search → context augmentation → generation.

**Key differences:**

- **Source**: RAG — external documents, Memory — the agent's own experience
- **Updates**: RAG — periodic/batch, Memory — continuous/real-time
- **Structure**: RAG — uniform chunks, Memory — episodes/facts/preferences
- **Metadata**: RAG — source/date, Memory — importance/recency/success
- **Goal**: RAG — information accuracy, Memory — contextual coherence

**Memory-specific optimizations:**

**Recency weighting** — score = similarity × decay^days. RAG has no temporal component.

**Importance scoring** — important facts are prioritized regardless of similarity.

**Episodic grouping** — retrieving entire episodes, not just the matching chunk.

**Multi-Source Retrieval**: effective agents combine documentation (RAG) + past experience (episodes) + facts (semantic) + preferences. RAG infrastructure can be reused for memory.

---

## Memory Prioritization

The LLM context window is limited — it is impossible to pass everything the agent has memorized. A selection mechanism is needed.

Priority is determined by four factors: **relevance** (semantic proximity to the query via cosine similarity), **recency** (recent is more important than old), **importance** (subjective assessment of significance), **usage frequency** (frequently retrieved items are likely useful).

### Selection Algorithm

Score = w₁ × relevance + w₂ × recency + w₃ × importance + w₄ × frequency. The top N are included in context.

**Task-specific tuning:** personal assistant (importance + recency), analytical agent (relevance + frequency), learning agent (importance + frequency).

**Dynamic adaptation:** weights change based on query context, turning prioritization into a nuance-sensitive process.

---

## Forgetting Mechanisms

Forgetting is not a bug but a feature. Outdated data is harmful: it clutters search, takes up space, and misleads. Human memory actively forgets as an adaptation mechanism. AI agents should do the same.

### Decay — Fading Over Time

The simplest forgetting mechanism is decay — a gradual reduction in importance over time. A memory item that has not been used for a long time loses priority.

Mathematically, this is implemented as multiplying importance by a factor less than one: new_importance = old_importance × decay_factor^days_since_last_use. If decay_factor = 0.95, then after 10 days without use, importance drops to 60% of the original, and after 30 days — to 21%.

This imitates the Ebbinghaus forgetting curve: information that is not used gradually becomes inaccessible. With each day of non-use, importance drops. When it falls below a threshold (e.g., 0.1), the item can be deleted or archived.

However, decay should not be uniform for all memory types. Semantic facts (general knowledge) decay more slowly than episodes (specific events). Important user preferences may not undergo decay at all or may have an extremely slow rate.

**Active deletion** — an explicit decision to remove an item: by user instruction, upon contradiction (new overrides old), upon storage overflow. Requires caution — "soft deletion" (marking as inactive) is often used.

**Archival** — a trade-off: old items go to "cold" storage. They do not participate in regular search but can be restored when needed.

---

## Persistence and Storage

### Storage Selection: Architectural Trade-offs

**Vector DBs** (Pinecone, Weaviate, Milvus, Qdrant) — specialized for ANN search, scale to millions of items. Cost: additional infrastructure, paid SaaS, complex metadata filtering. For production with 100k+ items.

**PostgreSQL + pgvector** — a hybrid: ACID transactions, complex queries, plus vector search. Unifying memory in a single database. Widely adopted (pgvector has 15K+ GitHub stars (as of early 2026)) and often the simplest production choice when PostgreSQL is already in the stack. Performance is adequate for most workloads up to ~100K-500K vectors with HNSW indexes.

**LanceDB** — an embedded vector database (no server). Stores data in Lance columnar format, supports hybrid search (vector + full-text), and runs in-process. Ideal for desktop agents, edge deployment, and local-first applications where running a database server is impractical.

**SQLite** — a file-based DB with no server. For prototypes, local agents, embedded systems. Limited by performance and concurrent writes. With sqlite-vec extension, basic vector search is also possible.

**MongoDB/Elasticsearch** — schema flexibility, full-text search. Trade-off: heavyweight systems, overkill for simple cases. For complex heterogeneous structures.

**JSON files** — zero infrastructure, easy to debug. Limitation: linear scanning, no indexes. For prototypes (< 1000 items).

### Data Structure

A memory item contains: **content** (verbatim text), **embedding** (vector representation for search), **metadata** (type, timestamp, lastAccessed, importance, source, tags), **relationships** (references to related items, turning memory into a graph).

Metadata is critical: without it, search is binary (relevant/not), with it — multidimensional (relevance + importance + recency + frequency = composite score).

### Performance and Scale

Memory grows unboundedly — hundreds of thousands of items within a month. Critical techniques:

**Indexes** — vector (HNSW/IVF) for similarity search, metadata (B-tree) for filtering. Trade-off: they consume space and slow down writes.

**Sharding** — partitioning by user/time/type at millions of items. Adds complexity: cross-shard queries are slow.

**Caching** — LRU cache for embeddings and search results. Exploiting locality.

**Background processing** — consolidation, decay, deduplication run asynchronously without blocking the agent.

**Monitoring** — metrics on size, growth, latency, result quality. Memory problems manifest as degradation in agent quality.

---

## Production Memory Patterns (2024-2025)

### Claude Code Memory: File-Based Persistent Memory

Claude Code (Anthropic's CLI agent) implements a pragmatic memory system that has become an influential pattern for production agents.

**CLAUDE.md files** serve as persistent memory across sessions:
- **Project-level** (`CLAUDE.md` in the project root) — shared across all sessions in the project. Contains: project conventions, architectural decisions, common patterns, known issues.
- **User-level** (`~/.claude/CLAUDE.md`) — personal preferences and workflow patterns.
- **Auto-memory** (`~/.claude/projects/<project>/memory/MEMORY.md`) — automatically accumulated insights from working on the project.

**Why this works:** The memory is stored as plain text files in the filesystem — human-readable, version-controllable (via git), and trivially debuggable. The agent reads these files at session start, writes updates during work, and accumulates knowledge over time. No vector database, no embedding model, no complex infrastructure.

**Pattern:** Memory-as-files is effective for developer tools where: memory items are relatively few (hundreds, not millions), human readability matters, version control integration is valuable, and the agent operates in a file-system context.

### MCP as a Memory Interface

The Model Context Protocol provides a standardized interface for memory operations, separating the memory abstraction from storage implementation:

**MCP Resources for memory retrieval:** An MCP server can expose agent memories as resources — the host application or the agent itself can read stored memories through the standard resource interface. This enables memory to come from any backend (vector DB, SQL, files, APIs) through a uniform protocol.

**MCP Tools for memory writes:** Memory storage operations (save a fact, record an episode, update a preference) are exposed as MCP tools. The agent calls these tools during conversations to persist important information.

**Benefits of MCP-based memory:**
- **Backend-agnostic:** Switch from SQLite to Postgres to Pinecone without changing the agent code
- **Composable:** Combine multiple memory MCP servers (one for user preferences, one for project knowledge, one for episodic memory)
- **Shareable:** The same memory server can serve multiple agents in a multi-agent system
- **Standardized:** Any MCP-compatible agent can use the memory server

This pattern represents the evolution from framework-specific memory implementations (LangChain memory, LlamaIndex storage) to protocol-based memory that is interoperable across the agent ecosystem.

## Key Takeaways

1. **Memory makes an agent useful**. Without memory, every interaction is isolated. With memory, the agent accumulates context, learns, and personalizes.

2. **Different memory types for different purposes**. Short-term for the session, long-term for persistence, episodic for experience, semantic for knowledge, working for the current task.

3. **Integration matters more than individual components**. A memory system is valuable when all types work together, forming a complete context for each query.

4. **Retrieval-Augmented Memory** solves the limited context problem. Instead of keeping everything in the prompt, relevant items are retrieved on demand.

5. **Prioritization is necessary**. Not everything is equally important. A combination of relevance, recency, importance, and frequency determines what enters the context.

6. **Forgetting is a feature, not a bug**. Outdated information is harmful. Decay and active deletion mechanisms keep memory clean and current.

7. **File-based memory works for developer tools**. Claude Code's CLAUDE.md pattern demonstrates that plain text files — human-readable, git-versioned — are an effective memory system when the memory scale is modest.

8. **MCP standardizes memory interfaces**. Exposing memory as MCP resources (read) and tools (write) enables backend-agnostic, composable, and shareable memory across agents.

---

## Practical Example: Integrated Memory System

To illustrate the concepts, consider a simplified implementation of an integrated agent memory system.

**Integrated memory system architecture:**

The AgentMemorySystem class unifies all agent memory types and manages their interactions. As dependencies, the class contains: embeddingModel for converting text to vectors, vectorStore for long-term memory storage, shortTermBuffer — a list of short-term memory messages, and workingMemory — a map for storing the current task state.

**The buildContext method — context construction:**

The buildContext method takes the current task description and assembles a complete context from all memory sources. It uses a StringBuilder for efficient construction of the resulting string.

The first step adds the "Current State" section from working memory. The formatWorkingMemory() method is called, which converts the structured data of the current task (goals, variables, constraints) into readable text. This guarantees the agent always knows its current state.

The second step is retrieving relevant memories from long-term memory. The current task is converted to an embedding via embeddingModel.embed(currentTask).content(). Then vectorStore.findRelevant() performs a search for the five most semantically similar memory items. If relevant memories are found, a "Relevant Knowledge" section is added, and each memory goes through the prioritize method for ranking by composite score, then formatted as a list.

The third step adds short-term memory. The "Recent Conversation" section is populated with the last ten messages from shortTermBuffer via the formatRecentMessages method. This ensures dialog continuity.

The assembled context is returned as a single string, ready for inclusion in the model prompt.

**The prioritize method — memory prioritization:**

The prioritize method implements multi-factor memory evaluation. For each memory item, a final score is computed as the product of three components: similarity (semantic proximity, already computed during search), recencyScore (freshness assessment based on timestamp), and importance (the item's predetermined importance).

The process uses the stream API: each item is mapped to a ScoredMemory with a computed score, the list is sorted by descending score (negative comparator for reverse order), then mapped back to a list of MemoryItem and collected. The result is a sorted list where the most relevant, recent, and important memories come first.

**The consolidate method — memory consolidation:**

The consolidate method transfers important information from short-term to long-term memory. It iterates over all messages in shortTermBuffer. For each message, estimateImportance() is called — a method that evaluates the message's significance (e.g., based on length, presence of keywords, emotional tone). If importance exceeds the threshold of 0.5, the message is worth saving.

A new MemoryItem is created with the message content, current time, and computed importance. The content is converted to an embedding via embeddingModel, and the pair (embedding, item) is added to vectorStore. This makes the item available for future semantic search.

After processing all messages, short-term memory and working memory are cleared — the session is complete, the state is reset.

**The applyDecay method — forgetting mechanism:**

The applyDecay method implements temporal importance decay for memory items. It accepts a decayFactor parameter (typically close to 1.0, e.g., 0.95) and applies it to all items in storage via updateAll.

For each memory item, the number of days since last access (daysSinceLastUse) is calculated via Duration.between from lastAccessed to the current moment. Then importance is multiplied by decayFactor raised to the power of the number of days: the more time has passed, the greater the importance drop. With decayFactor = 0.95 and 10 days without use, importance drops to approximately 60% of the original.

If importance falls below the threshold of 0.1, the item is marked for archival via markForArchival() — it is moved to "cold" storage or deleted, freeing space for more current information. The updated item is returned to storage.

This mechanism keeps memory clean and relevant, imitating the natural forgetting of unused information.

This example demonstrates the key patterns: building context from different memory types, prioritization through scoring, consolidation from short-term to long-term, and forgetting via decay.



---

## Navigation
**Previous:** [[04_Planning|Agent Planning]]
**Next:** [[06_Computer_Use_Agents|Computer Use Agents]]
