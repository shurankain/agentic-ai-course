# Information Retrieval Methods

## The Role of Retrieval in RAG

Retrieval is the heart of a RAG system. No matter how powerful the generative model is, it is helpless without high-quality context. The "garbage in, garbage out" principle applies here in full: if the retriever finds irrelevant documents, even GPT-6 Astra cannot produce a useful answer.

The task of retrieval seems simple at first glance: find documents most relevant to the user's query. In practice, this is a complex problem. Relevance is subjective and context-dependent. The same document may be relevant to one aspect of a question and irrelevant to another. The user may be looking for a specific fact or a general understanding of a topic.

Effective retrieval requires understanding the technical aspects of search and query semantics, document structure, and domain-specific characteristics. The following covers methods and techniques for building a high-quality information retrieval system.

## Basic Search Methods

### Dense retrieval

Dense retrieval uses dense vector representations (embeddings) for search. Each document and each query is transformed into a fixed-dimension vector. The search reduces to finding documents with vectors close to the query vector.

The advantage of dense retrieval is semantic understanding. The query "problems connecting to the server" will find documents about "connection errors" and "network issues," even if the exact words from the query are not present. The model has learned to associate conceptually related terms.

The limitation is handling rare terms and specialized vocabulary. If the model has not encountered a particular abbreviation or product during training, it cannot correctly represent them in the vector space. The query "error XYZ-12345" may not find a document containing the exact error code.

### Sparse retrieval

Sparse retrieval is based on lexical similarity. Classical algorithms BM25 and TF-IDF score documents by the frequency of matching terms, accounting for their rarity in the corpus.

BM25 effectively finds documents containing the exact terms from the query. If the user searches for a specific error code, function name, or technical abbreviation, BM25 will find documents with those terms.

The limitation is the absence of semantic understanding. "Automobile" and "car" are different words for BM25. A document about "vehicle maintenance" will not be found by the query "car repair."

### Hybrid Search

The hybrid approach combines the strengths of both methods. Dense retrieval provides semantic understanding, sparse retrieval provides exact term matching.

A typical implementation runs both searches in parallel and merges the results. Merging methods vary: simple rank fusion, weighted score combination, Reciprocal Rank Fusion (RRF).

Hybrid search is particularly effective for technical domains where both conceptual understanding and exact technical terms matter.

## Advanced Retrieval Techniques

### Query expansion

Query expansion augments the original query with additional terms or phrasings. The goal is to increase recall by finding documents that are relevant but do not contain the exact words from the query.

One approach uses a thesaurus or WordNet to add synonyms. "Buy" is expanded to "buy OR purchase OR order."

A more advanced approach uses an LLM to generate alternative phrasings. The model generates several versions of the query, and the search is executed against all versions. This is especially useful when the user phrases the query suboptimally.

### Query decomposition

Complex questions can be broken down into sub-questions. The question "Compare the performance of PostgreSQL and MongoDB for analytical queries" decomposes into: "PostgreSQL performance for analytics," "MongoDB performance for analytics," "SQL vs NoSQL comparison for analytics."

Searching for each sub-question finds specific documents. Results are aggregated to form a complete answer.

Decomposition is especially useful for comparative questions and questions requiring information from different sources.

### Hypothetical Document Embeddings (HyDE)

HyDE is an innovative approach where an LLM first generates a hypothetical answer to the question, and then that answer is used for search. The idea is that the answer is semantically closer to relevant documents than the question itself.

**Why hypothetical documents work:**

Consider the Query-Document Gap problem. When a user asks "How does GC work in Java?", an embedding is created from a short, general, interrogative text. Meanwhile, the document contains a declarative, detailed description: "The Garbage Collector in Java uses a generational approach with Young and Old Gen..." These different text styles land in different regions of the vector space. This makes retrieval harder.

The HyDE solution: the question is first passed to an LLM, which generates a hypothetical answer in a documentation style ("GC in Java uses a generational..."). The embedding of this hypothetical answer ends up much closer to the embeddings of the actual documents, improving search quality.

**Theoretical justification:**

Embedding models were trained predominantly on declarative texts: articles, documentation, descriptions. Questions represent a different domain in the vector space. HyDE performs a kind of "translation" of the question into the document domain, bridging the semantic gap between them.

**When HyDE is most effective:**

The method shows the best results when user questions differ in style from the documents in the knowledge base. It is especially useful for technical knowledge bases where documentation is written in formal language, while users phrase queries informally, in a conversational style.

The main limitation is that the additional LLM call for generating the hypothetical answer increases system latency and cost. The quality gains must be weighed against these costs.

### Step-back prompting

Step-back prompting generates a more abstract question before searching. Instead of the specific "Why does service X crash after upgrading to version 2.3?", a general question is first generated: "Typical causes of service crashes after upgrades."

Searching with the abstract question finds documents about general principles that may shed light on the specific problem. This can then be supplemented with a specific search.

### Multi-query retrieval

Multi-query generates several variants of the query and performs a search for each. Results are merged, duplicates are removed.

The advantage is increased recall. Different phrasings can find different relevant documents. The disadvantage is the increased number of queries to the index.

## Reranking: Refining Results

### Why reranking is needed

Initial retrieval is optimized for speed: processing millions of documents in milliseconds. This requires trade-offs in accuracy. The result is an approximately correct but not ideal ranking.

Reranking is a second stage where a small set of candidates (typically 20-100) is ranked by a more accurate but slower model. The two-stage approach delivers both speed and quality.

### Cross-encoder vs Bi-encoder: Architectural Differences

Understanding the difference between these architectures is critical for building an effective retrieval system.

**Bi-encoder (Dual Encoder):**

The bi-encoder architecture processes the query and document independently of each other. The query "How does..." passes through its encoder and becomes a query_emb vector. The document "GC uses..." passes through the encoder (the same or a separate one) and becomes a doc_emb vector. These two vectors are then compared via cosine similarity to obtain a relevance score.

Key advantage: document embeddings can be precomputed once and stored in the index. Approximate nearest neighbor (ANN) indexes such as HNSW make search roughly logarithmic in corpus size. This makes bi-encoders ideal for initial search across millions of documents.

Critical drawback: no interaction between the query and document at the model level. The encoder does not see the document text when processing the query, which limits accuracy.

**Cross-encoder:**

The cross-encoder combines the query and document into a single input: "[CLS] Query [SEP] Document [SEP]". This combined sequence passes through a transformer, where the cross-attention mechanism allows query tokens to interact with document tokens directly. The final [CLS] token then passes through an MLP layer to produce a relevance score.

This full interaction between query and document provides higher accuracy compared to the bi-encoder. The model sees both texts simultaneously and can capture subtle nuances of correspondence.

Critical limitation: O(n) complexity: a separate forward pass through the neural network is required for each document. With millions of documents, this makes cross-encoders unsuitable for initial search but ideal for reranking a small set of candidates.

**Practical comparison:**

For indexing, a bi-encoder processes each document in a single pass; a cross-encoder is not applicable for indexing. Search speed for a bi-encoder is constant with an ANN index; for a cross-encoder, it is linear in the number of documents. Ranking quality for a bi-encoder is good; for a cross-encoder, it is excellent. A bi-encoder scales to millions of documents; a cross-encoder is applicable only for reranking small sets.

Popular cross-encoder models include ms-marco-MiniLM and reranking models from the sentence-transformers library.

### ColBERT: Late Interaction (2020-2024)

ColBERT represents a trade-off between bi-encoder and cross-encoder through the concept of "late interaction."

**Architectural idea:**

Instead of creating a single vector for the entire document, ColBERT creates a separate embedding for each token. Query tokens q1, q2, q3 are converted into embeddings e1, e2, e3. Document tokens d1, d2, d3, d4, d5 are converted into embeddings f1, f2, f3, f4, f5. These embeddings are not pooled into a single vector but remain separate.

At the search stage, the MaxSim (Maximum Similarity) mechanism is applied: for each query token qi, the maximum cosine similarity with the document tokens is found. The final score is the sum of these maximums across all query tokens. In effect, each query token searches for its best match among the document tokens.

**Key advantages:**

Precomputing document embeddings is possible, as with a bi-encoder. They can be created in advance and stored in the index. Token-level matching provides finer-grained correspondence than comparing single vectors. ColBERT turns out to be substantially more accurate than a classic bi-encoder, as it captures fine-grained correspondence between specific words of the query and document.

**Evolution of ColBERT v2 (2022-2024):**

The second version brought significant improvements. Compression via residual compression allows compressing token-level document embeddings, reducing memory requirements. Distillation, training on data from cross-encoder models improves quality without increasing model size. The optimized Plaid indexing system speeds up search over token embeddings.

**Practical recommendations for choosing:**

For collections exceeding 10 million documents or when minimal latency is critical, use a classic bi-encoder. For medium-sized databases of 100K to 10M documents where quality is more important than speed, ColBERT is optimal. Use a cross-encoder only for collections under 10K documents or exclusively for reranking.

### LLM-based reranking

An LLM can be used for ranking by asking it to evaluate the relevance of each document or compare pairs of documents. More powerful models yield higher-quality ranking.

The advantage is understanding nuances and the ability to follow complex relevance criteria. You can specify what matters more: freshness, authoritativeness, or completeness.

The disadvantage is cost and latency. Calling an LLM for dozens of documents increases system cost.

### Ensemble reranking

Combining multiple rankers improves robustness. One ranker may fail on a certain type of document; another can compensate.

Results from different rankers are combined through voting, averaging, or a learned combination.

## Contextual Retrieval

### The Problem of Lost Context

When chunking a document, a fragment may lose important context. The sentence "This leads to an error" without the preceding explanation of "this" is meaningless. A function without indication of which class it belongs to is harder to identify.

Contextual retrieval solves this problem by enriching each fragment with contextual information.

### Contextual Embeddings

One approach is to prepend meta-information to the chunk text before embedding. Instead of "This leads to an error," the indexed text becomes "Document: Troubleshooting Guide. Section: Network Issues. Content: This leads to an error."

The embedding now includes context, and search takes it into account.

### Parent Documents

The parent document retrieval technique: small chunks are indexed for precise search, but upon retrieval, the parent, larger fragment is returned.

The small chunk precisely matches the query. The parent fragment provides context for generation. The best of both worlds.

### Sentence Window Retrieval

A similar approach: individual sentences are indexed, but upon retrieval, a window of N sentences around the found one is returned.

This provides precise search (a sentence is an atomic unit of meaning) with adequate context (surrounding sentences).

## Filtering and Metadata

### Using Metadata

Document metadata is a powerful tool for improving relevance. Publication date allows filtering out outdated documents. Category restricts the search to a relevant topic. Author or source accounts for authoritativeness.

Pre-filtering (filtering before the vector search) reduces the search space and can speed it up. Post-filtering (after the search) is applied to the already retrieved results.

### Dynamic Filtering

The system can extract filtering criteria from the query. "What changed in the API in version 3.0?" implies a version filter. "Latest security news" implies a date and category filter.

An LLM can parse the query and generate filters automatically.

### Boosting and Penalizing

In addition to hard filtering, you can softly influence ranking. Recent documents receive a score bonus. Documents from certain sources get a bonus or penalty. Long documents may be penalized for "diluteness."

Boosting parameters are typically tuned empirically.

## Iterative and Multi-Hop Retrieval

### The Problem of Complex Questions

Some questions require information distributed across multiple documents. "Which companies that invested in AI in 2023 also work on quantum computers?" requires: finding AI-investing companies, then finding information about their activity in quantum computing.

A single query will not find all the needed information.

### Self-RAG and CRAG

Self-RAG evaluates the quality of the retrieved context and decides whether additional search is needed. If the found documents are insufficient, the system iteratively refines the query and searches again.

CRAG (Corrective RAG) goes further: if the context is irrelevant, the system can rephrase the question or turn to external sources.

### Multi-hop reasoning

Multi-hop retrieval performs a chain of search queries where each subsequent query depends on the results of the previous one.

Example: "Who was the finance minister when law X was adopted?" First, search for the date the law was adopted. Then search for the minister at that date.

The LLM coordinates the chain, formulating intermediate queries based on the information obtained.

## Reranker Evolution (2024-2025)

The reranker landscape has evolved, with new models and approaches becoming available.

### Modern Reranker Models

**Cohere Rerank v3 (2024):** A hosted reranking API supporting 100+ languages. Optimized for RAG pipelines, accepts a query and a list of documents, returns relevance scores. Low latency (~50ms for 25 documents), easy integration. Trade-off: hosted service, data leaves your infrastructure.

**Jina Reranker v2 (2024):** Open-weight reranker with multilingual support. Available for self-hosting. Competitive with Cohere on benchmarks while allowing on-premise deployment for data-sensitive applications.

**BGE Reranker v2.5 (BAAI, 2024):** Open-source cross-encoder reranker in multiple sizes (small to large). Strong performance on MTEB benchmarks. Popular choice for self-hosted reranking.

**LLM-based rerankers, RankGPT pattern (2024):** Using LLMs (GPT-6, Claude) as listwise rerankers. Instead of scoring documents individually, the LLM receives the full list and returns a ranking. More expensive but captures inter-document relationships. Best used as a final refinement stage for the top 5-10 candidates.

### Speculative Retrieval

**Concept:** Inspired by speculative decoding in LLM inference. A small, fast retriever proposes candidates, and a larger, more accurate model validates them. The key insight: validation is cheaper than generation, checking whether a document is relevant is faster than searching for relevant documents from scratch.

**Two-phase speculative retrieval:**
1. **Speculate:** A lightweight retriever (BM25, small bi-encoder) quickly proposes top-K candidates
2. **Verify:** A powerful cross-encoder or LLM verifies each candidate's relevance, potentially triggering a refined search if verification rejects too many

**Adaptive retrieval depth:** Instead of always retrieving a fixed top-K, speculative retrieval adjusts depth dynamically. If the first 5 candidates all pass verification: stop early. If none pass: expand the search or reformulate the query. This reduces average latency while maintaining quality.

**Connection to Agentic RAG:** Speculative retrieval is a building block for agentic search patterns. The agent's reasoning loop naturally implements speculate-verify: retrieve, then evaluate relevance, then decide whether to search more or proceed with generation.

## Key Takeaways

High-quality retrieval is the key to a successful RAG system. Techniques allow improving basic vector search.

Hybrid search combines the semantic understanding of dense retrieval with the precision of sparse retrieval. Query transformation (expansion, decomposition, HyDE) improves query formulation.

Reranking refines initial results with more powerful models. Cross-encoders and LLM-based ranking improve the quality of top-K results. Modern rerankers (Cohere v3, Jina v2, BGE v2.5) provide production-ready options for both hosted and self-hosted deployment.

Contextual retrieval addresses the problem of context loss during chunking. Parent documents and sentence windows provide both precision and context.

Speculative retrieval adapts search depth dynamically, reducing latency while maintaining quality. It connects naturally to agentic search patterns.

Metadata and filtering allow incorporating business logic and data structure. Iterative retrieval handles complex multi-step questions.

The choice of techniques depends on the specifics of the task, data, and constraints on latency and cost. Experiments on real data determine the optimal combination.
