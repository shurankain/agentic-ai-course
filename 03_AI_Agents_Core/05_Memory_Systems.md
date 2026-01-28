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

## MemGPT: Virtual Context Management

### The MemGPT Concept

MemGPT (UC Berkeley, 2023) is an architecture that gives an LLM the illusion of "infinite" memory through context virtualization. Inspired by virtual memory in operating systems.

**OS Analogy**

In operating systems, virtual memory solves a fundamental problem: programs need more memory than the physical RAM available. The solution is an illusion: the program "sees" a large address space, but the OS dynamically moves data between fast RAM and slow disk.

MemGPT applies the same idea to LLMs. The model's context window (e.g., 128K tokens) is the "RAM": fast but limited. External storage (vector DB) is the "disk": slower but practically unlimited.

The Memory Manager acts as the OS virtual memory: it dynamically "swaps" information between context and storage. Important and relevant data is loaded into context. Stale or unused data is offloaded to external storage. The LLM operates under the full illusion that all information is available.

### MemGPT Architecture

MemGPT divides memory into two levels:

**Main Context** — the limited window visible to the LLM. Consists of three parts:
- System Prompt (fixed) — instructions and operating rules
- Working Context (dynamic) — current state and active data
- FIFO Messages (recent) — latest conversation messages

**External Storage** — unlimited storage outside the context. Divided into two types:
- Archival Memory (vector DB) — everything ever saved: past conversations, learned facts, user preferences. Unlimited volume, search via embeddings.
- Recall Memory (structured) — key memories about the user and important facts. Smaller in volume but always readily accessible.

The connection between levels is made through memory_read and memory_write functions, which the LLM calls as needed. The agent itself decides when to load information from External Storage into Main Context and when to save new information.

### Key MemGPT Functions

**Self-directed memory management**

The main innovation of MemGPT is that the LLM itself decides when and what to save or retrieve. The model is provided a set of functions for working with memory: archival_memory_insert (save important data), archival_memory_search (find archived memories), core_memory_append (add to core knowledge about the user), core_memory_replace (update existing knowledge).

The agent does not passively save everything. It analyzes the conversation and decides: "This is an important fact — I'll save it," "I need additional information — I'll search the archive," "This is a preference update — I'll replace the old one." This removes the burden from the developer: no need to manually program saving rules.

**Heartbeat mechanism**

A traditional agent works reactively: receives a message → processes → responds → waits. MemGPT adds proactivity: the agent continues to "think" even without incoming messages.

The heartbeat mechanism is a periodic internal call, even when the user is silent. The agent checks: do recent memories need consolidation? Is there relevant context that should be preloaded? Is it time to update working memory?

This internal activity is invisible to the user but critical for quality. The agent does not wait for a problem to manifest ("I forgot the context") but prevents it in advance.

**Practical advantages**

Without MemGPT, long conversations quickly lose context: after 20-30 messages, information from the beginning of the dialog falls out of the window. With MemGPT, it is automatically saved to archival storage and can be retrieved when needed.

Personalization without MemGPT is limited to the current session: the agent "knows" about the user only what was said in this conversation. With MemGPT, knowledge accumulates across sessions: the agent remembers preferences, work style, past tasks.

Relevant context without MemGPT requires manual RAG setup: the developer must explicitly program when and what to retrieve. MemGPT makes this self-directed: the agent itself decides what information is needed.

Context usage changes from "all or nothing" to "intelligent loading": instead of passing the entire dialog history, the agent loads only relevant parts, saving tokens and focusing the model's attention.

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

**PostgreSQL + pgvector** — a hybrid: ACID transactions, complex queries, plus vector search. Unifying memory in a single database. Limitation: performance up to ~100k vectors. For medium volumes with integration into existing infrastructure.

**SQLite** — a file-based DB with no server. For prototypes, local agents, embedded systems. Limited by performance and concurrent writes.

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

## Key Takeaways

1. **Memory makes an agent useful**. Without memory, every interaction is isolated. With memory, the agent accumulates context, learns, and personalizes.

2. **Different memory types for different purposes**. Short-term for the session, long-term for persistence, episodic for experience, semantic for knowledge, working for the current task.

3. **Integration matters more than individual components**. A memory system is valuable when all types work together, forming a complete context for each query.

4. **Retrieval-Augmented Memory** solves the limited context problem. Instead of keeping everything in the prompt, relevant items are retrieved on demand.

5. **Prioritization is necessary**. Not everything is equally important. A combination of relevance, recency, importance, and frequency determines what enters the context.

6. **Forgetting is a feature, not a bug**. Outdated information is harmful. Decay and active deletion mechanisms keep memory clean and current.

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
