# Advanced RAG Techniques

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → RAG
**Previous:** [[04_Retrieval_Methods|Information Retrieval Methods]]
**Next:** [[06_Late_Interaction_Retrieval|Late Interaction Models]]

---

## Beyond Basic RAG

Basic RAG works well for simple scenarios, but production applications demand advanced techniques: handling complex questions that require synthesis from multiple sources, adapting to data changes, and continuously improving quality.

Advanced RAG includes techniques for all pipeline stages: data preparation, retrieval, generation, and quality evaluation. This section focuses on architectural decisions and the applicability of different approaches.

## Taxonomy of RAG Architectures

Before diving into details, it is useful to understand the landscape of RAG approaches. Each architecture addresses specific problems of the basic "retrieve-read" pattern:

| Architecture | Key Idea | When to Use |
|-------------|---------------|-------------------|
| **Naive RAG** | Simple retrieve → generate | Prototypes, simple use cases |
| **Retrieve-and-Rerank** | Two-stage retrieval with reranker | When high precision is needed |
| **Hybrid RAG** | Dense (embeddings) + Sparse (BM25) | Terminology, exact matches matter |
| **Graph RAG** | Knowledge graph + vector search | Multi-hop reasoning, entity relationships |
| **Multimodal RAG** | Text + images/tables | Documents with visual content |
| **Agentic RAG (Router)** | LLM selects data source | Multiple heterogeneous sources |
| **Agentic RAG (Multi-Agent)** | Multiple agents for retrieval | Complex research tasks |

This taxonomy is not exhaustive — in practice architectures are combined. For example, Agentic RAG can use Graph RAG internally, and Hybrid Search is applicable almost everywhere.

**Evolution of complexity:**

RAG architectures evolve from the simplest (Naive RAG) toward more complex and autonomous systems. Each evolutionary stage adds new capabilities: Hybrid RAG improves search accuracy, Reranking improves result quality, Graph and Multimodal add understanding of structure and different modalities, Agentic approaches give the system autonomy in planning and decision-making.

Choosing an architecture is a trade-off between complexity and capability. Start simple (Naive + Hybrid), add complexity only when the basic approach falls short.

## Agentic RAG

### From Passive Search to Active Exploration

Traditional RAG is passive: receive a query, find documents, generate an answer. Agentic RAG transforms the system into an active researcher capable of planning, iterating, and adapting.

The agent analyzes the question and determines a search strategy. It decomposes complex questions into parts. When information is insufficient, it refines the query and searches again. When data is contradictory, it seeks additional sources for resolution.

Agentic RAG uses the LLM not only for answer generation but also for managing the search process. The model decides which tools to use, what queries to formulate, and when to stop.

### Planning and Reasoning

The agent can apply reasoning techniques before searching. Chain-of-thought analysis of the question identifies the required information. The agent forms a plan: "First I will find information about X, then clarify Y, after which I can answer Z."

This plan can be adjusted during execution. If the first search does not yield the needed information, the agent revises its strategy. Unlike a static RAG pipeline, an agentic system adapts to context and intermediate results.

### Tools and Integrations

Agentic RAG goes beyond simple vector search. The agent can have access to multiple tools: search across different indexes, SQL queries to databases, API calls to external services, a calculator for computations.

Tool selection is determined by the nature of the question. A statistics question may require an SQL query. A question about current events — a news search. A question about internal documentation — vector search in the knowledge base.

**Examples of tools in Agentic RAG:**
- Vector search over the corporate knowledge base
- BM25 search for exact term matching
- SQL queries to analytics databases
- Web search for up-to-date information
- Calculator for mathematical computations
- Code interpreter for data analysis

The agent selects tools dynamically based on query analysis. For example, the question "How many customers bought product X last quarter?" requires SQL, while "What is known about company Y?" requires vector search and possibly web search.

## Corrective RAG (CRAG)

### Self-Correction and Verification

CRAG adds a layer of critical evaluation of retrieved documents. Instead of blindly using search results, the system evaluates their relevance and reliability.

If documents are irrelevant, CRAG can: reformulate the query and try again, turn to alternative sources (e.g., web search), or honestly acknowledge that the information was not found.

### Knowledge Refinement

After obtaining candidates, CRAG filters and refines the information. Irrelevant parts of documents are trimmed. Contradictory information is flagged. High-quality fragments receive priority.

This refinement process can use a separate critic model or the same LLM with a different prompt.

### Fallback Strategies

CRAG defines fallback scenarios for various situations:
- Documents are partially relevant: use with caveats
- Documents are irrelevant: try a different approach
- No information available: honestly inform the user
- Contradictory data: present all viewpoints

**CRAG Process in Detail:**

1. **Retrieve** — obtain candidates from the vector store
2. **Evaluate** — assess the relevance of each document (0-1 score)
3. **Decide** — make a decision based on scores:
   - All scores high → use the documents
   - Scores moderate → filter out irrelevant parts
   - Scores low → fall back to web search or another source
4. **Refine** — for partially relevant documents, extract only the relevant fragments
5. **Generate** — generate an answer factoring in source quality assessment

The critical difference from basic RAG: the system does not assume that retrieval always works. It verifies the assumption and adjusts strategy when necessary.

## Self-RAG

### Adaptive Retrieval

Self-RAG dynamically decides whether retrieval is needed for a specific question. Simple factual questions within the model's knowledge may not require search. Complex specific questions do.

The model is trained to generate special tokens signaling the need for retrieval, the relevance of retrieved documents, and whether the answer is supported by sources.

### Self-Reflection

Self-RAG includes a self-assessment mechanism. The model not only generates an answer but also evaluates its quality. If the evaluation is low, the system can try a different approach.

**Formal Decision Model of Self-RAG:**

Self-RAG operates on a multi-step decision model. In the first stage, the model analyzes the incoming query Q and decides whether retrieval is needed, generating a special token [Retrieve] or [No Retrieve]. If retrieval is needed, the system obtains a set of documents D.

Next, each document d undergoes a relevance check — the model generates an ISREL score, classifying the document as relevant or irrelevant. For each relevant document, an answer is generated and then checked for source support via the ISSUP token (full, partial, or no support).

In the final stage, each generated answer receives a usefulness score ISUSE on a scale from 1 to 5. The best answer is selected based on the combined score of all these evaluations.

**When retrieval is needed vs. not:**

| Question Type | Retrieve? | Reason |
|-------------|-----------|---------|
| General knowledge ("What is Python?") | No | LLM knows |
| Specific facts ("Version of library X?") | Yes | May be outdated |
| Reasoning ("Why does 2+2=4?") | No | Logic, not facts |
| Current events | Yes | LLM does not know |
| Internal documentation | Yes | Private data |

Reflection tokens:
- ISREL: is the document relevant to the query?
- ISSUP: is the answer supported by the document?
- ISUSE: is the answer useful to the user?

### Critical Generation

During generation, Self-RAG can create multiple answer variants and select the best one based on internal evaluation. This is similar to best-of-n sampling, but with semantically meaningful criteria.

**Key innovation of Self-RAG:** The model is trained to generate special reflection tokens that are invisible to the user but control the process. For example, the [Retrieve] token signals the need for search, [IsRel] evaluates relevance, [IsSup] checks whether a claim is supported by sources.

This allows the model to "think aloud" about the quality of its work and correct it in real time. Unlike post-hoc evaluation, reflection is embedded in the generation process.

## Graph RAG

### Limitations of a Flat Index

Traditional RAG treats documents as independent units. But real knowledge has structure: entities are connected by relationships, concepts form hierarchies, events form chains.

A flat index loses this structure. A question about relationships between entities requires multiple documents that may not be found by a single query.

### Knowledge Graph Integration

Graph RAG builds a knowledge graph on top of documents. Entities (people, companies, concepts) become nodes. Relationships between them become edges. Documents are linked to the entities they mention.

Search in Graph RAG combines vector similarity with graph traversal. After finding a relevant node, the system can explore its neighbors — related entities and facts about them.

### Advantages of the Graph Approach

Multi-hop reasoning becomes natural. The question "Which companies compete with those that X invested in?" requires: find X → find their investments → find competitors of those companies. On a graph, this is a chain of relationships.

Information aggregation is simplified. The graph knows all mentions of an entity across different documents and can assemble a complete picture.

### Graph Construction

Graph construction involves several processing stages:

1. **Entity extraction (NER)** — identifying people, organizations, locations, products in text
2. **Relation extraction** — determining relationships between entities (works_at, founded, acquired)
3. **Coreference resolution** — understanding that "John Smith", "J. Smith", and "he" refer to the same entity
4. **Entity Linking** — linking to external knowledge bases (Wikidata, corporate directories)

**Relation extraction methods:**

| Method | Description | When to Use |
|-------|----------|-------------------|
| **Rule-based** | Patterns like "X CEO of Y" | Simple, predictable relationships |
| **Supervised ML** | Trained classification model | Labeled data available |
| **Distant Supervision** | Automatic alignment with KB | No labeled data |
| **LLM-based** | Prompt "Extract relations from..." | Flexibility over speed |

Modern systems often use LLMs for graph extraction: the model is shown the entire document and asked to extract entities and relationships in a structured format (JSON). This is expensive, but quality is high and does not require training specialized models.

**Entity Disambiguation** is a critical step. The word "Apple" can mean the company, the fruit, or the record label. The system analyzes context ("announced new iPhone") and selects the correct interpretation. This allows linking the mention to a node in the knowledge graph.

### Microsoft Graph RAG (2024)

Microsoft introduced GraphRAG — a production-ready approach to Graph RAG that became the de facto standard.

**Key innovations:**

| Component | Description |
|-----------|----------|
| **Entity Extraction** | LLM extracts entities and relationships from chunks |
| **Hierarchical Clustering** | Leiden algorithm for community detection |
| **Community Summaries** | LLM generates a summary for each community |
| **Global vs Local Search** | Two retrieval modes |

**Indexing architecture:**

1. **Chunking** — splitting documents into chunks
2. **Entity & Relationship Extraction** — LLM extracts a graph from each chunk
3. **Graph Construction** — merging into a unified graph
4. **Community Detection** — Leiden clustering creates a hierarchy of communities
5. **Summary Generation** — LLM summarizes each community
6. **Indexing** — chunks + entities + communities + summaries → index

**Global vs Local Search:**

| Mode | Description | Use Case |
|-------|----------|----------|
| **Local Search** | Entity-centric, graph traversal from found entities | Specific questions about entities |
| **Global Search** | Community summaries + map-reduce | High-level questions about the entire corpus |

**Global Search pipeline:**
1. Query → all community summaries (parallel)
2. Each summary + query → partial answer (map)
3. Partial answers → final answer (reduce)

**When to use:**
- Questions like "What are the main themes in this corpus?"
- Trend and pattern analysis
- When an overview is needed rather than a specific fact

**Microsoft's results:**
- +30% quality on complex queries vs naive RAG
- Global search is especially effective for summarization tasks
- Trade-off: expensive indexing (many LLM calls)

## RAPTOR: Recursive Abstractive Processing (2024)

### The Long Document Problem

Traditional RAG indexes chunks at a single level of detail. The question "What is the main idea of the book?" requires high-level understanding absent from individual chunks. The question "What code is on page 47?" requires a detailed chunk.

**RAPTOR solves this through hierarchical indexing:**

RAPTOR creates a multi-level abstraction structure of the document. The base level (Level 0) contains the original document. The first level (Level 1) contains individual chunks — the result of splitting the document into fragments.

The system then clusters semantically similar chunks and summarizes each cluster, creating the second level (Level 2) of summaries. This process repeats recursively: Level 2 summaries are again clustered and summarized, forming the third level (Level 3) — an overall summary of the entire document.

All levels are indexed into a single vector store, allowing retrieval to find both detailed chunks for specific questions and high-level summaries for overview queries.

### RAPTOR Algorithm

1. **Chunking** — splitting the document into base chunks (Level 1)
2. **Embedding** — obtaining embeddings for each chunk
3. **Clustering** — clustering similar chunks (GMM or k-means)
4. **Summarization** — LLM summarizes each cluster
5. **Recursion** — repeat steps 2-4 for summaries, creating the next level
6. **Indexing** — index ALL levels into a single vector store

### Advantages of RAPTOR

| Question Type | Level | Example |
|-------------|---------|--------|
| Detailed | Level 1 | "What is the config file format?" |
| Medium | Level 2 | "How does the authorization module work?" |
| High-level | Level 3 | "What is the philosophy of this framework?" |

**Retrieval in RAPTOR:**
- Search across ALL levels simultaneously
- Results can include chunks from different levels
- High-level summaries provide context
- Detailed chunks provide specifics

RAPTOR is especially effective for:
- Long documents (books, specifications)
- Questions at different levels of abstraction
- When both overview and details matter

## Multimodal RAG

### Beyond Text

Documents contain more than text: images, diagrams, tables, charts. Basic RAG ignores this information or processes it superficially.

Multimodal RAG integrates different modalities. Images receive descriptions or are directly used by multimodal models. Tables are converted to structured data. Charts are described in text or analyzed visually.

### Processing Strategies

For images:
- Generating descriptions (captioning) and indexing the text
- Direct use of vision models when answering
- Indexing images in a specialized CLIP-like space

For tables:
- Conversion to text descriptions
- SQL-like queries to structured data
- Indexing individual rows/cells with context

For diagrams:
- OCR + structure recognition
- Description generation by a multimodal model

### Multimodal Embeddings

Models like CLIP create a shared space for text and images. A text query can find a relevant image, and vice versa.

This enables search by description: "sales growth chart" will find a visualization without a text caption. How it works: an image encoder and a text encoder are trained so that semantically related images and text have close embeddings.

**Practical applications:**
- Searching for diagrams and charts by text description
- Answering questions about image content
- Cross-modal search: text → image or image → text
- Indexing presentation slides, infographics, schematics

### ColPali Pipeline: Vision-Language Retrieval

ColPali (ICLR 2025) is a breakthrough approach to Multimodal RAG that completely eliminates OCR from the pipeline.

**ColPali Architecture:**

ColPali uses the vision-language model PaliGemma-3B to process documents directly as images. A document page (PDF or image) is fed into a Vision Encoder based on patch tokenization — the document is divided into visual patches of 14×14 pixels.

Each patch becomes a separate token embedding, creating a multi-vector representation of the document (typically 1024+ tokens per page). A Projection Layer maps these patch tokens into a shared embedding space where they can be compared with text queries.

The result is a multi-vector representation of the document where each vector corresponds to a visual fragment of the page, preserving spatial structure and visual information without OCR losses.

**Key components:**

| Component | Description | Role |
|-----------|----------|------|
| **PaliGemma** | Vision-Language model (SigLIP + Gemma 2B) | Base encoder |
| **Patch Tokenization** | Document → visual patches 14×14 | Preserves spatial layout |
| **Multi-Vector Output** | Each patch → separate embedding | Late interaction |
| **MaxSim Scoring** | Max similarity between query and page tokens | Retrieval metric |

**OCR-Free Workflow:**

**Traditional approach vs ColPali:**

The traditional pipeline passes through several stages with information loss: the PDF is processed by OCR for text extraction, the text is split into chunks, and chunks are converted to embeddings and indexed. Problems arise at each stage: OCR makes recognition errors, layout information is lost, and tables become unstructured text.

The ColPali pipeline is much simpler and lossless: the PDF is rendered to an image, the image is processed by the Vision Encoder directly, creating a multi-vector index. The result: no OCR errors (text is not recognized at all), layout is fully preserved in the visual representation, and tables and diagrams are processed natively as visual objects.

**Late Interaction for Visual Content:**

ColPali uses the same Late Interaction principle as ColBERT:

1. **Query encoding:** Text query → token embeddings
2. **Document encoding:** Page → patch embeddings (multi-vector)
3. **MaxSim scoring:**
   $$\text{Score}(Q, D) = \sum_{i} \max_{j} \text{sim}(q_i, d_j)$$

   Each query token finds the most similar patch on the page.

**Advantages for specific document types:**

| Document Type | Traditional Approach | ColPali |
|---------------|---------------------|---------|
| **Tables** | OCR loses structure | Structure preserved |
| **Diagrams** | Cannot extract relationships | Visual relationships preserved |
| **Formulas** | LaTeX/MathML errors | Native understanding |
| **Multi-column layout** | Incorrect reading order | Spatial awareness |
| **Handwritten text** | OCR often fails | Vision model handles it |

**Practical results:**

Benchmarks on ViDoRe (Visual Document Retrieval):
- ColPali: **81.3** nDCG@5
- Best OCR+embedding baseline: **67.8** nDCG@5
- Improvement: **+20%**

**ColPali limitations:**

1. **GPU requirements:** Requires GPU for inference (vision encoder is heavy)
2. **Index size:** Multi-vector = more storage space
3. **Preprocessing:** Requires rendering PDF → images
4. **Query latency:** Late interaction is more expensive than dense retrieval

**When to use ColPali:**

| Scenario | Recommendation |
|----------|--------------|
| PDFs with tables/diagrams | ✅ ColPali |
| Scanned documents | ✅ ColPali |
| Pure text content | ⚠️ OCR+embedding may suffice |
| Millions of documents | ⚠️ Consider hybrid (ColPali for complex ones) |
| Edge deployment without GPU | ❌ Needs cloud inference |

**Integration into RAG pipeline:**

ColPali integrates into the RAG pipeline as follows: the user query goes to the ColPali Retriever, which finds relevant document pages as images. These page images are passed directly to a Vision-LLM (GPT-4V, Claude 3, Gemini), which generates the final answer by visually "reading" the images without intermediate text representation. This allows the model to see the document just as a human would — with all visual structure, formatting, and graphical elements.

## Temporal RAG

### Temporal Dynamics of Data

Information has a temporal dimension. A document about the "current situation" becomes outdated. Historical data is important for context but not for the current state. Temporal RAG accounts for this dynamic.

**Key strategies:**

**Time filtering:** A question about "current rules" should find only up-to-date documents. A question about "development history" should find historical ones. The system analyzes temporal indicators in the query ("now", "in 2020", "historically") and applies corresponding filters.

**Temporal weighting:** More recent documents receive a relevance bonus if the question is not explicitly historical. For example, a decay function: score × (1 / (1 + age_days)) boosts the rank of fresh documents.

**Versioning:** For changing documents (policies, API documentation), versions are stored. The system can return the version that was current at a specific point in time. This is critical for compliance and audit — one needs to know what the policy stated at the time a decision was made.

**Change detection:** The system tracks changes and updates the index. When a document is updated, the old version is archived, the new one is indexed with a timestamp, dependent entities in the graph are updated, and caches are invalidated.

**Practical example:** In legal tech, RAG must find the version of a contract that was in effect on the signing date, not the current revision. Temporal metadata enables "search in the past."

## Evaluation and Quality Improvement

### Quality Metrics

Evaluating a RAG system involves several independent dimensions. It is important to understand that high retrieval quality does not guarantee a good answer, and vice versa — all pipeline stages need to be evaluated.

**Context Relevance** (Relevance) — how relevant are the retrieved documents to the question? Measures retrieval quality. Low relevance means the retriever finds irrelevant results.

**Context Recall** (Recall) — were all relevant documents found? Complements relevance: one can find relevant results but miss important ones. Requires ground truth annotations.

**Answer Faithfulness** (Groundedness) — is the answer based on the retrieved documents? Critical for preventing hallucinations. An answer can be correct but not grounded in the context — this is a problem.

**Answer Relevance** — does the answer address the question asked? The model can generate truthful text that does not answer the question.

**How to measure:** Each metric can be evaluated by an LLM-as-judge (a prompt with evaluation criteria) or by a human on a test set. LLM evaluation scales but requires validation on a subset manually.

### Automated Evaluation

**LLM-as-judge:** A strong model (GPT-4, Claude) is shown the question, context, answer, and evaluation criteria. It returns a score (0-1 or 1-5) and justification. This enables rapid evaluation of thousands of examples without manual effort.

**Frameworks:** RAGAS, TruLens, DeepEval provide ready-made metrics and pipelines. They standardize evaluation and allow comparing different versions of the system. RAGAS has become the de facto standard — it includes faithfulness, answer relevancy, context precision, and context recall.

### Test Sets

A quality test set is the foundation for improvement. It should cover diverse scenarios:
- Typical user questions (80% of volume)
- Complex multi-hop questions (test reasoning)
- Questions with no answer in the knowledge base (the system should admit it)
- Questions with multiple possible answers (test nuance)
- Edge cases (corner cases that break the system)

Test sets can be generated synthetically from documents (an LLM generates questions from text), but manual curation of critical cases is necessary to cover real user problems.

### Feedback Loop

A production system collects two types of feedback:

**Explicit:** The user rates the answer (thumbs up/down, star rating). This is the most valuable signal, but it is not obtained for every answer.

**Implicit:** The user reformulates the question (meaning the first answer did not help), copies the answer (positive signal), or ignores the answer (negative signal). These signals are noisy but available for all interactions.

Feedback is used for continuous improvement: expanding the test set with problematic cases, fine-tuning components on real data, tuning hyperparameters (chunk size, top-k, reranking threshold).

## Performance Optimization

### Caching Strategies

Caching is critical for production performance. It is applied at several levels:

**Embedding cache:** Identical queries are not vectorized repeatedly. Text hash → cached embedding. Saves time and money on API calls for hosted embeddings.

**Retrieval cache:** Identical queries do not perform search repeatedly. Especially effective for FAQs — frequently repeated questions are cached aggressively (TTL = hours/days).

**Response cache:** Full answers to frequent questions. The most aggressive cache with the greatest savings. But requires caution: invalidation upon data updates is critical.

**Invalidation strategy:** When a document is updated, all caches related to that document are invalidated. This can be expensive — an alternative: TTL-based cache with a reasonable time-to-live (1-24 hours).

### Streaming

Streaming generation improves perceived latency. The user sees the first tokens within ~500ms, even though the full answer takes 5-10 seconds to generate. This radically improves UX.

**Architectural details:** Retrieval is performed in full, then streaming generation begins. Sources and citations can be shown before the text (the user sees that the system found results), during (inline citations), or after (a list of sources at the end).

### Parallelization

Independent operations are executed in parallel to reduce latency:

- **Search across multiple indexes** — if the database is split into indexes (by document type, by date), search in parallel
- **Batch embedding** — if multiple queries arrive, the embedding model processes them as a batch
- **Parallel reranking** — split the top-100 documents into chunks, rerank each chunk in parallel

### Precomputation

Offline preprocessing reduces online latency:

**Document clustering** — for RAPTOR and hierarchical search. Performed during indexing, not during search.

**Summary precomputation** — for large documents, generate summaries in advance. During search, return the summary + a link to the full document.

**Aggregation materialization** — in Graph RAG, frequently queried patterns (all relationships of an entity) are materialized into the index.

---

## Research Frontiers: Trainable RAG Systems

Current research focuses on training RAG components end-to-end rather than using pre-trained models as-is.

### From Frozen Retriever to Learned Retrieval

Standard RAG uses a "frozen" retriever — the embedding model is trained separately and used as-is. Learned retrieval trains the retriever together with the generator, optimizing for final answer quality. The retriever learns to find documents that help the generator, not just semantically similar ones.

**REPLUG (2023):** Trains the retriever via KL-divergence between answer probabilities with and without the document. Documents that increase the probability of the correct answer receive a reward. Works with any LLM as a black box.

**Atlas (2022):** Meta showed that retrieval-augmented models require less data for fine-tuning. Atlas 11B with retrieval competes with PaLM 540B on knowledge-intensive tasks thanks to joint training of the retriever and generator.

### Fusion-in-Decoder

FiD is an architectural pattern for efficiently working with multiple documents. Each document is processed by the encoder separately, then the decoder applies cross-attention to all encoder outputs simultaneously. This yields linear complexity in the number of documents instead of quadratic with naive concatenation.

### Retrieval as Attention

A radical idea: retrieval is a form of attention at the document level instead of the token level. Approaches like **Memorizing Transformer** (KV cache = retrieved memories), **RETRO** (cross-attention to retrieved chunks), **kNN-LM** (next token = weighted sum of retrieved tokens) integrate retrieval at the model architecture level.

**Challenges of end-to-end training:** Top-K selection is non-differentiable (no gradients through argmax), memory is needed for embeddings of all documents, and the index becomes stale during training. Solutions include Gumbel-Softmax for relaxed selection, contrastive learning, async index updates, and in-batch negatives.

---

## Long Context vs RAG: Trade-offs

Modern models support contexts of 100K-1M+ tokens. Is RAG still needed?

### When to Choose What

**Long context wins:** The entire knowledge base fits in context, non-obvious connections between parts are needed (the retriever would miss them), conversation history is needed in full, and the document is needed in its entirety (chunking breaks meaning).

**RAG wins:** Scale (millions of documents), cost (100K tokens per request = hundreds of thousands of dollars/month), latency (prefill 100K tokens = 5-10 seconds vs RAG 0.5-1 second), freshness (new documents are added constantly), precision (relevant fragments are needed).

**Quantitatively:** Long context (100K tokens) = $0.30/request. RAG (2.5K tokens) = $0.0075/request. Savings of 40×. Latency: 10-20× difference.

### Hybrid Approach: RAG + Long Context

The optimal strategy combines both approaches: RAG filters the corpus (1M docs) down to 100 → reranker narrows to top-20 → all 20 are loaded into long context (40K tokens) → LLM synthesizes the answer. This provides the precision of RAG and the depth of understanding from long context.

**"Lost in the middle" problem:** LLMs have worse recall for information in the middle of long context (U-shaped attention). Solution: place important content at the beginning and end, use explicit citations, chunking with overlap.

---

## RAG System Design: Interview Preparation

RAG System Design is a standard question in ML System Design interviews. Interviewers evaluate understanding of ML systems: data pipelines, embedding models, retrieval quality, and critically — the ability to analyze trade-offs.

### Structured Approach

**1. Clarifying Questions (5 minutes):** Scale (documents, QPS)? Latency requirements (<1s)? Accuracy vs recall trade-off? Data types (text/multimodal)? Regulatory constraints (GDPR, HIPAA)?

**2. High-Level Design (10 minutes):** Core components — Ingestion Pipeline → Vector Store → Retrieval Service → Generation Service → Evaluation Pipeline. Key decisions: sync vs async ingestion, hosted vs self-hosted embeddings, vector DB selection, single-stage vs multi-stage retrieval.

**3. Deep Dive (20 minutes):**

**Ingestion:** Document → Preprocessing → Chunking → Embedding → Indexing. Chunking strategy — a critical trade-off:
- Fixed-size (512 tokens): simple, predictable, but breaks semantic units
- Semantic (paragraphs): preserves meaning, but variable size
- Hierarchical: multi-resolution search, complex implementation
- Sliding window: overlap helps with boundaries, but redundant storage

**Retrieval:** Query → Query Processing → Retrieval → Reranking → Context Assembly. Hybrid search (dense + BM25) for terminology. Reranking (cross-encoder) for top-K refinement.

**4. Trade-offs Discussion (10 minutes):** The most important part. Framework: identify trade-off → quantify impact → context-dependent decision → mitigation.

Examples: Chunk size (small = precision, large = context), Retrieval (hybrid wins for terminology), Reranking (add if latency budget > 200ms), Context (depends on question complexity).

### Vector Database Selection

| DB | Best For |
|----|----------|
| **Pinecone** | Fully managed, strict SLAs, enterprise features |
| **Milvus** | Large scale (100M+ vectors), GPU acceleration, on-prem |
| **Qdrant** | Best price/performance, payload filtering, self-hosted |
| **Weaviate** | GraphQL API, multi-modal, Kubernetes-native |
| **Chroma** | Development/prototyping, embedded, local-first |

### Evaluation Framework

**Retrieval metrics:** Precision@K, Recall@K, NDCG.

**Generation metrics:** Faithfulness (grounded in context?), Answer Relevance (addresses question?), Context Relevance (useful context?), Hallucination Rate.

**RAGAS framework** — the de facto standard for automated evaluation. LLM-as-judge for scalable evaluation.

**Production monitoring:** Online metrics (latency, error rate), Offline evals (RAGAS, golden set), Automated testing (regression, A/B tests), Human review (spot checks, flagged responses).

### Red Flags in Interviews

❌ "Just use ChatGPT with all documents in context" — lack of understanding of cost/latency
❌ "We'll use the best embedding model" — no domain-specific understanding
❌ "Vector search will find everything" — lack of understanding of semantic search limitations
❌ "We'll evaluate with accuracy" — lack of understanding of RAG-specific metrics
❌ Ignoring production concerns — no production experience

✅ **Correct approach:** Analyze trade-offs, consider domain, combine dense + BM25, use specific metrics (faithfulness, relevance), address monitoring/fallbacks/cost.

---

## GraphRAG Evolution (2024-2025)

Microsoft's GraphRAG (2024) established the category, but its high indexing cost (many LLM calls for entity extraction and community summarization) spurred significant research into more efficient alternatives.

### LazyGraphRAG (Microsoft, 2024)

**Problem:** Standard GraphRAG requires expensive upfront indexing — extracting entities, building graphs, computing community summaries. For large corpora this can cost thousands of dollars in LLM API calls before a single query is answered.

**Solution:** LazyGraphRAG defers expensive computation to query time. Instead of precomputing the full graph and all community summaries at indexing, it builds a lightweight index (BM25 + minimal entity extraction) and performs graph construction and summarization on-demand, only for the subgraph relevant to the current query.

**Key properties:**
- Indexing cost reduced by **~100x** compared to standard GraphRAG
- Query cost scales with query complexity, not corpus size
- Combines best-of-both-worlds: local search (entity-centric) and global search (community summaries) computed lazily
- No quality degradation on local queries; slight trade-off on global queries

**When to use:** Large or frequently changing corpora where full GraphRAG indexing is prohibitively expensive; scenarios where queries are diverse (only a small fraction of the graph is relevant to any given query).

### LinearRAG (2025)

**Problem:** GraphRAG's community detection and hierarchical summarization add complexity. For many use cases, the graph structure provides diminishing returns over simpler approaches.

**Solution:** LinearRAG simplifies the pipeline by replacing graph-based community detection with linear document chains. Documents are organized by topical similarity into linear sequences, and summarization follows these chains rather than graph communities.

**Key insight:** For many question-answering tasks, the hierarchical community structure of GraphRAG is over-engineered. A simpler linear organization of documents by topic provides 80-90% of the benefit at a fraction of the complexity.

### GraphRAG Recommendations (2025)

| Approach | Indexing Cost | Query Latency | Best For |
|----------|--------------|---------------|----------|
| **Standard GraphRAG** | High (many LLM calls) | Low (precomputed) | Static corpora, frequent queries |
| **LazyGraphRAG** | Very low | Medium (on-demand) | Large/dynamic corpora, diverse queries |
| **LinearRAG** | Low | Low | Simpler use cases, cost-sensitive |
| **No graph (Hybrid RAG)** | Minimal | Low | When entity relationships don't matter |

**Practical guidance:** Start without a graph (Hybrid RAG + Contextual Retrieval). Add GraphRAG only when you have multi-hop questions that require connecting entities across documents. If you add it, start with LazyGraphRAG to avoid upfront indexing costs. Graduate to full GraphRAG only for stable corpora with high query volumes where precomputation amortizes well.

## Latest Techniques (2024)

RAG continues to evolve. Several techniques from 2024 significantly change the approach to building systems.

### CAG: Cache-Augmented Generation

**Radical idea:** Load the entire knowledge base into the model's extended context window, completely bypassing the retrieval pipeline.

**When it works:** The knowledge base fits in context (< 200K tokens), non-obvious connections between documents are needed, retrieval systematically misses relevant content.

**When it does not fit:** Large knowledge base (millions of documents), frequent updates, latency is critical (prefill is slow), limited budget.

**Hybrid approach:** RAG filters the corpus (1M docs) down to 100 → CAG loads all 100 into context → LLM synthesizes.

### Late Chunking (Jina AI)

**Problem:** Embedding models see only an isolated chunk. "He was a great scientist" in isolation is meaningless — who is "he"?

**Solution:** Invert the process — first run the entire document through the embedding model (before pooling), obtain token-level embeddings with full context, then split into chunks and apply pooling.

**Result:** Each chunk "remembers" the document's context. Pronouns, references, and abbreviations retain their meaning. +7.8% retrieval quality (Jina AI benchmark).

**Limitations:** Requires a model with long context (8K tokens), expensive preprocessing.

### Contextual Retrieval (Anthropic)

**Solution:** Before embedding, an LLM generates a contextual annotation for each chunk. Example: "The company reported revenue of $4.2B" → "This is Apple's Q3 2024 quarterly report. The company reported revenue of $4.2B."

**Results:** -35% retrieval errors, -67% with hybrid search (BM25).

**Trade-off:** Requires an LLM call for each chunk during indexing. Expensive, but indexing is a one-time operation and quality is critical.

### ColPali: Vision-Language for RAG

**Problem:** OCR makes errors, layout is lost, table formatting is lost.

**Solution:** Indexes visual patches of the document directly, without OCR. The vision encoder creates embeddings for each patch, and the text query is matched against visual patches via late interaction.

**Results:** +15% recall on documents with tables vs OCR-based.

### Comparison and Recommendations

| Technique | Solves | Trade-off | Use When |
|---------|--------|-----------|-----------------|
| **CAG** | Retrieval misses | Cost, latency | Small knowledge base, full context |
| **Late Chunking** | Context isolation | Preprocessing cost | Pronouns, references |
| **Contextual Retrieval** | Chunk ambiguity | LLM calls | Critical precision |
| **ColPali** | OCR errors | GPU | PDFs with tables |

**Recommendation:** Start with Contextual Retrieval + Hybrid Search (highest ROI). Add ColPali for visual documents, Late Chunking for texts with cross-references.

## Key Takeaways

Advanced RAG goes beyond simple "find and answer." Main directions of development:

**Architectural approaches:**
- **Agentic RAG** transforms the system into an active researcher with planning and adaptation
- **CRAG and Self-RAG** add critical evaluation, self-correction, and fallback strategies
- **Graph RAG** structures knowledge for multi-hop reasoning through entity graphs
- **Multimodal RAG** works with images, tables, diagrams (ColPali without OCR)
- **Temporal RAG** accounts for data dynamics, versioning, and freshness

**Production requirements:**
- Systematic evaluation (RAGAS: faithfulness, relevance, context precision/recall)
- Feedback loop for continuous improvement
- Performance optimization (caching, streaming, parallelization)
- Monitoring and observability

**New techniques in 2024:**
- Contextual Retrieval (Anthropic) — contextual annotation of chunks (-35% errors)
- Late Chunking (Jina) — embeddings with full document context
- ColPali — vision-language retrieval without OCR (+15% recall on tables)

**Choosing an approach:** Depends on question complexity, data types, quality requirements, and performance. The best systems combine multiple approaches: Hybrid Search (dense + BM25) + Reranking + Contextual Retrieval as the baseline, adding Graph/Multimodal/Agentic for specific scenarios.

---

## Practical Code Examples

In production, use frameworks (LangChain, LlamaIndex) instead of writing from scratch. Below is a minimal example for understanding the architecture.

### Hybrid RAG with Reranking

An advanced RAG system is implemented through a class encapsulating the full query processing pipeline. During system initialization, four key components are configured: vector_store for semantic search via embeddings, bm25_index for lexical keyword search, reranker for precise candidate reranking, and llm for final answer generation. These components are stored as instance attributes for subsequent use.

The query processing method takes a user question and two parameters: top_k determines the number of candidates for initial retrieval (default 20), and rerank_k sets the final number of documents after reranking (default 5).

The first stage — hybrid retrieval — performs parallel search across two channels. Dense retrieval via vector_store finds top_k/2 semantically similar documents (10 in this case) using cosine similarity of embeddings. Simultaneously, sparse retrieval via bm25_index finds top_k/2 documents (10) with maximum lexical term overlap. Results from both searches are combined by the merge_results method, which removes duplicates and can apply weighting coefficients to scores from different sources, forming a unified candidate list.

The second stage — reranking — passes all candidates along with the question to the reranker, which typically uses a cross-encoder model. The cross-encoder processes each (question, candidate document) pair jointly through a transformer, obtaining a precise relevance score. This is significantly more accurate than simply comparing vectors but is feasible only for a small set of candidates due to computational complexity. The result is the top rerank_k most relevant documents (5).

The third stage — generation — formats the reranked documents into context via format_context, which can combine document texts with their metadata. This context along with the original question is passed to the LLM for answer generation. The LLM synthesizes information from the provided context, creating a coherent answer.

The final result is returned as a structured object with three fields: answer contains the LLM-generated response text, sources includes metadata of the documents used (titles, URLs, dates) for transparency and verifiability, and confidence represents the system's confidence in the answer, computed based on the relevance scores of reranked documents (e.g., high scores of top documents indicate high confidence).

**Conceptual explanation of the remaining approaches:**

**Agentic RAG** is implemented through a planning system where the LLM analyzes the question, creates a search plan in JSON format with steps and tools, then iteratively executes each step. The agent has access to a dictionary of tools (vector_search, SQL, web_search) and dynamically selects the appropriate one based on query type. When results are insufficient, the agent revises the plan and adds new steps.

**Corrective RAG (CRAG)** adds a relevance evaluation stage after retrieval. Each document receives a score via an evaluator (which can be a separate model or LLM-as-judge). A decision is made based on scores: if all scores are high (>0.8) — use documents as-is; if moderate (0.5-0.8) — filter and refine irrelevant parts; if low (<0.5) — fall back to alternative sources (web search, another index).

**Graph RAG** works through extracting entities from the question, searching for those entities in the knowledge graph, traversing the graph 1-2 hops to obtain related entities and facts, retrieving documents linked to this subgraph, and generating an answer considering the graph structure. This enables answering multi-hop questions that require connecting multiple entities.

**Evaluation with RAGAS** is implemented by computing four key metrics for each example: faithfulness (is the answer based on the context?), answer relevance (does the answer address the question?), context precision (is the context relevant?), and context recall (was all necessary context found?). Metrics are computed using the LLM-as-judge approach — a prompt with evaluation criteria returning a score from 0 to 1. Results are aggregated for an overall system assessment.

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → RAG
**Previous:** [[04_Retrieval_Methods|Information Retrieval Methods]]
**Next:** [[06_Late_Interaction_Retrieval|Late Interaction Models]]
