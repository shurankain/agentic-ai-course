# Context Windows and Managing Them

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[02_Tokenization|Tokenization: How Text Becomes Numbers]]
**Next:** [[04_Generation_Parameters|Generation Parameters: Controlling Model Behavior]]

---

Imagine you are having a conversation through a note that can hold only a certain number of words. Anything that does not fit is forgotten. Each time you respond, you can only rely on the information that fits on that note. This is exactly how a context window works in language models — it is the AI's "working memory," limited by the number of tokens the model can process at once.

The context window is one of the key constraints of LLMs, directly affecting the architecture of AI agents. Understanding this constraint, its causes, and ways to work with it is critically important for building effective agentic systems.

---

## What Goes Into the Context Window

It is important to understand that the context window is not just what you send to the model. It includes:

**System prompt** — baseline instructions that define the model's behavior. For an agent, this might be a description of its role, available tools, and response format.

**Conversation history** — all previous user messages and model responses. With each exchange, the history grows.

**Current user query** — what the model needs to respond to right now.

**And most importantly — the model's response.** This is often overlooked. The context limit applies not only to input data but also to the generated response. If a model has a context of 8,000 tokens and 7,000 are already occupied by the prompt and history, the model can generate at most 1,000 tokens of response.

---

## Evolution of Context Sizes

The history of context windows is a story of rapid growth:

| Model | Year | Context Window |
|--------|-----|------------------|
| GPT-2 | 2019 | 1,024 tokens |
| GPT-3 | 2020 | 4,096 tokens |
| GPT-3.5 | 2022 | 4K / 16K tokens |
| GPT-4 | 2023 | 8K / 32K / 128K tokens |
| Claude 2 | 2023 | 100,000 tokens |
| Claude 3 | 2024 | 200,000 tokens |
| Gemini 1.5 | 2024 | 1,000,000+ tokens |
| GPT-5 | 2025 | 128K–1M tokens |
| Claude 4.6 | 2026 | 200K (1M beta) |
| Gemini 2.5 Pro | 2025 | 1,000,000 tokens |
| Llama 4 Scout | 2025 | 10,000,000 tokens |

What do these numbers mean in practice? 4,000 tokens is roughly 8 pages of text or 3,000 words. Enough for a short conversation, but not enough for analyzing a long document.

128K tokens is already around 250–300 pages. You can load a small book in its entirety.

1M+ tokens in Gemini is essentially a library. Theoretically, you could load "War and Peace" and ask questions about it.

But bigger is not always better. Increasing context comes at a cost.

---

## Why Context Is Limited

### Quadratic Complexity of the Attention Mechanism

The main reason for the limitation is architectural. The Self-Attention mechanism in the Transformer has quadratic complexity with respect to sequence length: O(n²).

What does this mean? For each token, the model must compute "attention" to all other tokens. At 1,000 tokens, that is one million operations. At 10,000 — already 100 million. At 100,000 — 10 billion.

Memory also grows quadratically. The attention matrix has size n×n, where n is the sequence length. At 128K tokens with 32-bit numbers, the attention matrix alone requires approximately 64 GB of memory. And that is for a single layer — the model has dozens.

### KV-Cache: Linear but Significant Memory

During text generation, the model caches Key and Value tensors for all processed tokens. This avoids recomputing them at every generation step.

The KV-cache size grows linearly with context length, but the coefficient is substantial. For a model with 96 layers, 96 attention heads, and a dimension of 128 per head, the cache for 128K tokens can consume tens of gigabytes of GPU memory.

This creates a practical ceiling: even if the model can algorithmically handle a long context, GPU memory may run out first.

### KV-Cache Compression Methods

Several KV-cache compression techniques have been developed for working with long contexts:

**PagedAttention** (vLLM, 2023): Manages the KV-cache like virtual memory — allocates blocks on demand, avoiding fragmentation. Allows serving more requests simultaneously but does not reduce per-token cache size.

**GQA/MQA** (see [[01_LLM_Basics|LLM Basics]]): Multiple Query heads share a single Key-Value pair. Llama 2 70B uses GQA with 8 KV-heads instead of 64, reducing the cache by 8x.

**Multi-head Latent Attention (MLA)** — a breakthrough by DeepSeek (2024): Instead of storing full Key and Value matrices, it stores their **compressed latent representations**. During processing, it projects them back into the full space.

How MLA works: instead of saving full Key and Value matrices for each attention head (which have dimensionality d), the system saves a compact latent vector with dimensionality 4–8x smaller. When needed, this latent vector is projected back into the full space to compute Key and Value. This is similar to image compression: instead of storing each pixel individually, we save a compact representation and decompress it when viewing.

MLA in DeepSeek-V2 compresses the KV-cache by **5–10x** without noticeable quality loss. This allows a model with 236B parameters to process a 128K context on hardware that otherwise could not handle it.

**Quantized KV-Cache**: Stores KV in INT8 or INT4 instead of FP16. Achieves 2–4x compression with minimal quality degradation.

### Computational Economics

Longer context = more computation = longer processing = higher cost.

When context doubles, processing cost increases roughly 4x (due to quadratic complexity). LLM providers factor this into their pricing.

For users, this means a direct dependency: loading an entire book and asking a question about it can cost tens of times more than loading only the relevant chapters.

---

## The "Lost in the Middle" Problem

Even models with massive context windows have an interesting characteristic: they do not use information equally well from different parts of the context.

Research demonstrates the "Lost in the Middle" phenomenon — information at the beginning and end of the context is utilized better than information in the middle.

### Connection to Cognitive Psychology

This phenomenon is not unique to LLMs — it mirrors well-studied effects from cognitive psychology:

**Primacy Effect**: Information presented first is remembered better. In information processing theory, this is explained by the fact that initial items receive more attention and transfer to long-term memory.

**Recency Effect**: Recent information is also remembered well, as it is still in "working memory" and has not been displaced by subsequent information.

**The Middle**: Information in the middle competes with both the initial material (proactive interference) and subsequent material (retroactive interference). The result — it is remembered worst of all.

In LLMs the mechanism is different, but the effect is similar:
- **Positional embeddings**: The model is trained on specific patterns. The beginning of the sequence has a characteristic "attention profile," as does the end.
- **Causal masking**: During processing, each token "sees" everything preceding it. Tokens at the end have access to the full context and can form richer representations.
- **Gradient flow during training**: Initial and final tokens may receive stronger gradients.

Think of the context as a stack of documents. The model "remembers" the top and bottom documents well, but documents in the middle of the stack may be "forgotten" or used less effectively.

This has practical implications for prompt design and RAG systems. If you load 20 documents into the context and ask a question, the model is more likely to find the answer in the first and last documents, even if the most relevant one is somewhere in the middle.

Mitigation strategy: place critically important information at the beginning of the context (right after the system prompt) or at the end (immediately before the user's query).

---

## Context Extension Techniques

### Sliding Window Attention

Instead of allowing each token to attend to all others, the field of view is restricted. A token "sees" only the N nearest tokens — a sliding window.

Advantages: complexity becomes linear O(n), memory as well. Very long sequences can be processed.

Disadvantages: direct connections between distant parts of the text are lost. A token at the beginning of the document cannot directly "look at" a token at the end.

Mistral models and some versions of Llama use sliding window in a subset of layers, combining it with full attention in other layers.

### Sparse Attention

A hybrid approach: part of the attention is local (sliding window), part is global (special tokens that see the entire context).

Longformer from AllenAI uses this scheme. The [CLS] token at the beginning has global attention — it sees the entire document. Other tokens have local attention plus the ability to see global tokens.

This is a good trade-off for tasks like document classification or question answering, where information needs to be aggregated from the entire text.

### Advanced Positional Encodings

Traditional positional encodings are fixed — the model is trained on sequences of a specific length and extrapolates poorly to longer ones.

**RoPE (Rotary Position Embedding)** solves this problem through vector rotation. The relative position between tokens is encoded as a rotation angle. This allows a model trained on 4K tokens to work with 16K or even 32K — with some quality loss, but without catastrophic failure.

**ALiBi (Attention with Linear Biases)** adds a simple linear penalty based on distance between tokens. The farther apart the tokens, the less "attention" they receive. An elegant solution that does not require positional embeddings at all.

Modern long-context models (Claude, Gemini) use combinations of these techniques, enabling effective processing of hundreds of thousands of tokens.

### How 1M+ Context Is Achieved: Ring Attention and Sequence Parallelism

Gemini 1.5 and other models with million-token contexts use advanced distributed computing techniques:

**Ring Attention** (2023): Distributes attention computation across multiple GPUs organized in a "ring." Imagine four GPUs connected in a cycle: the first is linked to the second, the second to the third, the third to the fourth, and the fourth loops back to the first. Each GPU holds only a portion of the KV-cache and its own portion of Query requests. During computation, KV data blocks are sequentially passed around the ring from one GPU to the next, allowing each processor to compute attention with all tokens in the sequence. This is like a relay race: each runner passes the baton to the next, and ultimately all participants contribute to achieving the common goal. The technology enables processing sequences that physically do not fit in the memory of a single GPU.

**Sequence Parallelism**: The long sequence is split into segments, each processed by a separate GPU. Results are synchronized. It differs from Ring Attention in how the splitting and communication are handled.

**Important limitation**: These techniques require **many GPUs** and **fast interconnect** (NVLink, InfiniBand). A 1M context on a consumer GPU is not feasible — this is datacenter infrastructure.

### Long Context vs RAG: When to Use Which

Having a 1M context does not mean you should always fill it. There is a trade-off between "load everything" and "find what is relevant":

| Criterion | Long Context | RAG |
|----------|------------------|-----|
| **Latency** | Higher (processing everything) | Lower (only relevant data) |
| **Cost** | High (all tokens) | Lower (fewer tokens) |
| **Quality (needle)** | Lost in the Middle | Precise retrieval |
| **Quality (synthesis)** | Better (sees everything) | Depends on chunking |
| **Complexity** | Simple to load | Requires a retrieval pipeline |

**When long context is better:**
- Synthesis of information across the entire document is needed
- The document is structured and order matters (code, legal documents)
- The number of documents is small (1–3)

**When RAG is better:**
- The knowledge base is large (>1M tokens)
- Precise facts are needed (needle-in-haystack)
- Cost/latency is important
- Data is frequently updated

**Hybrid approach**: Use RAG to select the top-K documents, then load them in full into a long context for deep analysis.

---

## Context Management in AI Agents

For agent developers, context management is a critical skill. Without it, the agent will either lose important information or exceed the context limit and stop functioning.

### The Problem of History Accumulation

Imagine an agent that helps a user with work tasks. Each message exchange adds tokens to the history:
- User message: ~50–200 tokens
- Agent response: ~100–500 tokens
- Tool call and result: ~100–1,000 tokens

After 10 exchanges, the history can occupy 5,000–15,000 tokens. After 50, it can easily exceed any reasonable context.

### Truncation Strategy (FIFO)

The simplest approach: when the context fills up, remove old messages while keeping recent ones.

Advantages: simplicity, predictability.

Disadvantages: critically important information from the beginning of the conversation may be lost. "Remind me what we agreed on at the start" — but that information has already been removed.

### Summarization Strategy

A smarter approach: instead of deleting old messages, compress them into a brief summary.

"Previously we discussed: the user wants to build an expense tracking application, chose React for the frontend, PostgreSQL for the database..."

The summary takes far fewer tokens but preserves key information. Recent messages remain in full, providing context for the current query.

### Prioritization Strategy

An even more advanced approach: evaluate the importance of each message and retain the most important ones, even if they are old.

Importance criteria:
- Recency: newer messages are more important
- Relevance to the current query (semantic similarity)
- Message type: tool results may be more important than small talk
- Explicit markers: "remember this," "this is important"

More complex to implement but yields better results for long interactions.

---

## Working with Long Documents

Agents frequently work with documents that exceed the context size. What do you do if you need to answer a question about a 500-page book, but the context only holds 50?

### Chunking: Splitting into Fragments

The document is split into fragments (chunks), each of which fits into the context. The question is asked against each fragment, and results are aggregated.

Critical decisions:
- **Fragment size**: too small loses context, too large is inefficient
- **Overlap**: fragments overlap to avoid losing information at boundaries
- **Semantic splitting**: splitting along semantic boundaries (paragraphs, sections) is better than splitting by a fixed character count

### Map-Reduce for Long Documents

A pattern from the big data world, adapted for LLMs:

**Map**: each document fragment is processed independently. "Extract key facts about the budget from this fragment."

**Reduce**: processing results from all fragments are combined. "Here are budget facts from different parts of the document. Compose an overall summary."

Advantage: documents of any size can be processed. Disadvantage: expensive (many LLM calls) and connections between distant parts of the document are lost.

### Retrieval-Augmented Generation (RAG)

Instead of loading the entire document, load only relevant fragments.

1. The document is split into fragments in advance
2. Each fragment is indexed via embedding
3. When a question arrives, semantically similar fragments are retrieved
4. Only the retrieved fragments are loaded into the context
5. The model answers the question with access to the relevant information

RAG is the standard approach for working with large knowledge bases. It allows "scaling" the agent's knowledge beyond the context window.

---

## Optimizing Context Usage

### Compact Prompts

The system prompt is a constant cost on every request. Optimize it:

Instead of:
"You are a helpful assistant that is designed to help users with their questions. You should always try to provide accurate and helpful responses to the best of your ability."

Write:
"You are a helpful, accurate assistant."

Same meaning, far fewer tokens.

### Minimalist Tool Descriptions

In agents with tools, each tool description consumes tokens. A dozen tools with verbose descriptions means thousands of wasted tokens.

An optimized tool description format should include only the function name, parameter types, return type, and a brief one-line description of what the function does. For example, a web search function can be described as: takes a query string, returns a list of results, searches for information on the web. Avoid detailed explanations, usage examples, or repeating obvious information. Every unnecessary word in a tool description is multiplied by the number of requests to the agent.

### Information Prioritization

Structure the context in descending order of importance:

1. Critical information (system prompt, key instructions)
2. Relevant context (data needed for the current query)
3. Historical information (previous messages)
4. Reference information (if space remains)

When truncation is necessary, the least important parts will be affected.

---

## Key Takeaways

1. **The context window is a hard constraint.** It includes both input data and the model's response. Exceeding it is impossible — the request simply will not execute.

2. **Context size has grown exponentially**, from 1K tokens in 2019 to 1M+ in 2024. But a large context means large computational and monetary costs.

3. **Quality degrades on long contexts.** The "Lost in the Middle" problem means the model utilizes the beginning and end of the context better than the middle.

4. **Context management is a critical skill** for agent development. Truncation, summarization, and prioritization strategies enable effective operation within constraints.

5. **Long documents require specialized techniques:** chunking, Map-Reduce, RAG. Simply loading the entire document is rarely optimal.

6. **Prompt optimization saves context.** Compact system prompts and tool descriptions leave more room for useful information.

---

## Practical Implementation of Context Management Strategies

### Implementing the History Truncation Strategy (FIFO)

The "first in, first out" strategy is implemented through a component that tracks three key parameters: the model's maximum context window size, the number of tokens reserved for the model's future response, and a token counter for measuring message sizes.

The logic is built on the principle of reverse traversal: the algorithm starts from the end of the message history and moves toward the beginning, sequentially adding messages to the result list as long as space remains available. Before starting the traversal, the available capacity is calculated: the system prompt size and the response reserve are subtracted from the maximum context size. Then, for each message from the end of the history, it checks whether the message fits in the remaining space. If yes — the message is prepended to the result; if no — the traversal stops, and all older messages are discarded.

The key advantage of this approach is simplicity and predictability. The disadvantage is the potential loss of important information from the beginning of the conversation, which can lead to context loss during long sessions.

### Implementing the Summarization Strategy

The summarization strategy works as an intelligent memory archiver for the agent. The component receives the message history and the available token limit as input, then decides whether compression is needed.

The algorithm works in several stages. First, the total history size is counted in tokens. If it fits within the limit, the history is returned unchanged. If it exceeds the limit, the algorithm calculates how many tokens need to be freed (with a small 500-token buffer).

Next, the history is divided into two parts: old messages for summarization and recent messages to preserve in full. The algorithm iterates through the history from the beginning, adding messages to the summarization list until enough tokens have been accumulated for freeing. The remaining messages are marked as "preserve in full."

The old messages are combined into a textual conversation representation (each message in "role: content" format), which is sent to a language model with a request to create a brief summary in 2–3 sentences. The resulting summary is wrapped in a special system message labeled "Summary of previous conversation," placed at the beginning of the final list, followed by all recent messages in full.

This approach preserves key information from the entire interaction history while consuming significantly fewer tokens than the full history would. The disadvantage is an additional LLM call to create the summary, which increases latency and cost.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[02_Tokenization|Tokenization: How Text Becomes Numbers]]
**Next:** [[04_Generation_Parameters|Generation Parameters: Controlling Model Behavior]]
