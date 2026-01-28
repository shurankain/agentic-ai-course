# Embeddings and Vector Stores

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → RAG
**Previous:** [[02_Chunking_Strategies|Document Chunking Strategies]]
**Next:** [[04_Retrieval_Methods|Information Retrieval Methods]]

---

## The Nature of Embeddings

Embeddings are numerical representations of text in a multidimensional space. Converting text into vectors enables semantic search in RAG.

Semantically similar texts receive similar vectors. The question "How do I connect a database?" and the phrase "Setting up a DB connection" have high cosine similarity despite using different words. This allows finding relevant information regardless of how the query is phrased.

### From Words to Vectors: The Transformation Process

When text enters an embedding model, a multi-stage transformation takes place. First, the text is split into tokens — subwords or word parts that the model recognizes. Then each token passes through transformer layers, where the attention mechanism allows each token to "look at" the surrounding context.

As the text passes through model layers, each word is enriched with information about its surroundings. The word "key" in the sentence "He inserted the key into the keyhole" will receive activations reflecting a mechanical context, while the same word in "The key to success is persistence" will have different activations related to an abstract concept. This contextual understanding is the core advantage of modern embeddings.

At the final stage, the model aggregates representations of all tokens into a single vector. This can be a mean value (mean pooling), a weighted average, or a special [CLS] token. The result is a vector of hundreds or thousands of numbers, each encoding a specific semantic aspect of the text.

### Geometry of the Embedding Space

To understand why embeddings work, one must delve into the geometry of the resulting space.

**Why Cosine Similarity?**

Embeddings are typically compared using cosine similarity rather than Euclidean distance. Cosine similarity computes the angle between vectors via their dot product, normalized by vector lengths. Euclidean distance, by contrast, measures the straight-line geometric distance between points in space.

Reasons for choosing cosine: invariance to magnitude (vector length may vary for technical reasons; cosine ignores it), interpretability (values from -1 to 1, easily understood as an angle between directions), robustness in high dimensions (in high-dimensional spaces, Euclidean distance becomes blurred due to the curse of dimensionality).

Properties of the embedding space include clustering (semantically similar texts group together), analogies (vector arithmetic such as "King - man + woman = Queen"), and hierarchies (general concepts at the center, specific ones at the periphery).

**Metric Learning:**

Embedding models are trained through metric learning — a method where the goal is not classification but learning a distance function. The Triplet Loss function is used, which operates on triplets: anchor (reference text), positive (semantically similar text), and negative (semantically distant text). The model learns to place positive closer to anchor than negative, with a margin. This shapes the structure of the space where proximity equals semantic similarity.

## How Embedding Models Work

### Architectural Foundations

Modern embedding models are based on the transformer architecture, but their purpose differs from generative LLMs. A generative model learns to predict the next token and is optimized for producing coherent text. An embedding model learns to create representations that maximally preserve semantics for search and comparison tasks.

Training embedding models is based on contrastive learning. The model receives triplets: anchor (reference text), positive (semantically similar), and negative (semantically distant). The goal is to learn to place positive closer to anchor than negative. This forms a space where geometric proximity corresponds to semantic similarity.

Embedding dimensionality is an important architectural parameter. Small models (384 dimensions) are compact and fast but may miss subtle semantic distinctions. Large models (1536–3072 dimensions) convey nuances of meaning more accurately but require several times more memory and time for distance computation. For most practical tasks, the sweet spot is 768–1024 dimensions.

### Contextual Understanding and Polysemy

The revolution in embeddings came with the transition from static to contextual representations. Older approaches (Word2Vec, GloVe) created a fixed vector for each word. The word "bank" always received the same vector, regardless of whether it referred to a river bank or a financial bank.

Modern contextual embeddings solve this problem fundamentally. The attention mechanism in the transformer allows each token to analyze surrounding words and form a representation that accounts for context. "Bank" in "He sat on the river bank" will get a vector close to "shore," "water," "nature." "Bank" in "She went to the bank to deposit money" will get a vector close to "finance," "account," "money."

This contextual understanding is especially important for technical texts, where a single term can have different meanings across domains. "Pipeline" in the context of oil extraction and in the context of software development are entirely different concepts, and a good embedding model will distinguish between them.

### Length Limitations and Their Consequences

All embedding models have a maximum input length — the context window. This limitation is dictated by the transformer architecture: the attention mechanism has quadratic complexity relative to sequence length. To control computational costs, models limit the maximum length.

Typical values: compact BERT models — 512 tokens, modern models like text-embedding-3 — 8192 tokens, specialized models for long texts — up to 32768 tokens. Attempting to feed text longer than the limit results in truncation — the model simply discards all tokens beyond the limit.

This makes chunking critically important. If a 10,000-token document is fed whole into a model with a 512-token limit, only the beginning will be processed and the rest will be lost. Proper chunking ensures that every part of the document receives a quality representation.

An interesting nuance: even if a document fits within the context window, embedding quality may degrade for very long texts. Averaging representations of many tokens "dilutes" specificity. Often, smaller, focused segments yield better search results than full-length documents.

## Choosing an Embedding Model

### Selection Criteria

Choosing an embedding model requires balancing several conflicting factors.

Representation quality is the primary criterion. How accurately does the model capture the semantics of your domain? Benchmarks like MTEB (Massive Text Embedding Benchmark) provide a general picture of a model's capabilities, but real-world testing on your data is irreplaceable. A model leading MTEB may underperform compared to a specialized model on medical texts or source code.

Dimensionality affects all other system characteristics. A model with 384 dimensions requires 4 times less memory for vector storage than a model with 1536 dimensions. For one million documents, this is the difference between 1.5 GB and 6 GB of RAM. Search also speeds up proportionally with reduced dimensionality.

Embedding generation speed is critical for real-time systems. Local models can generate hundreds of embeddings per second on an ordinary CPU. API models are limited by network latency and rate limits. When processing terabytes of text, speed becomes the key factor.

Language support is an especially important aspect for non-English applications. Many models were trained predominantly on English and perform poorly with Russian, Chinese, or Arabic. Multilingual models were trained on dozens of languages and deliver acceptable quality across all of them, but may fall short of specialized monolingual models.

Cost and licensing are practical considerations. Cloud APIs charge per thousand tokens. For small volumes this is negligible; for petabytes it becomes significant. Local open-source models require infrastructure but have no variable costs.

### Popular Solutions and Their Niches

OpenAI text-embedding-3-small and text-embedding-3-large are the industry standard for cloud-based solutions. Advantages: excellent quality on diverse tasks, simple integration via API, reliability and scalability of OpenAI. Disadvantages: dependency on an external service, variable cost, network latency.

Models from the Sentence Transformers family (all-MiniLM-L6, all-mpnet-base-v2, e5-large, bge-large) are the choice for self-hosted solutions. Advantages: full control, no API dependency, fixed infrastructure costs. Disadvantages: require administration, GPU for large volumes, may need fine-tuning for specific domains.

Cohere Embed is an alternative to OpenAI with emphasis on multilingual support and enterprise features. Particularly strong in tasks requiring support for dozens of languages simultaneously.

For non-English applications, testing multilingual capabilities is critical. Models trained only on English will produce poor results on other languages. Look for models with explicit multilingual support or specialized variants for your target language.

### Specialization vs. Versatility

General-purpose models were trained on vast, diverse corpora — web pages, books, articles. They perform well on a broad range of texts, from news to blogs.

Specialized models were trained (or fine-tuned) on narrow domains. CodeBERT and StarEncoder understand programming syntax and recognize code patterns. SciBERT and PubMedBERT are optimized for academic publications with their specific terminology and structure.

When should you choose a specialized model? When your domain differs significantly from the general-language corpus. Medical texts are filled with Latin terminology. Legal documents use archaic constructions. Source code is a formal language with strict syntax. In these cases, a specialized model can outperform a general-purpose one by 10–20% on search metrics.

But specialization comes at a cost: a model for code will perform poorly on regular text. If your system needs to handle code, documentation, and user questions, a general-purpose model may be the better trade-off.

## Vector Stores

### Why Specialized Stores Are Needed

Conventional databases are optimized for exact search: find a record with id=123, find all records where name='John'. Vector search is a different task: find records whose vectors are most similar to a given one.

For millions of vectors, brute-force search is unacceptably slow. Specialized vector databases use Approximate Nearest Neighbor (ANN) algorithms, sacrificing minor accuracy for dramatic speedups.

### Index Types

Different ANN algorithms offer various trade-offs between speed, accuracy, and memory consumption.

**HNSW (Hierarchical Navigable Small World)** is a popular graph-based algorithm. It builds a hierarchical structure of multiple layers: upper layers are sparse and contain few nodes for fast navigation, the bottom layer is dense and contains all vectors. Search starts from the top layer and greedily descends, refining results at each level.

Key tuning parameters: M defines the number of neighbors at each level (typically 16–64), efConstruction sets the beam width during index construction, efSearch controls the trade-off between speed and quality during search.

HNSW is best when high accuracy is required (recall > 95%), data fits in memory, search speed matters, and data is rarely updated.

**IVF (Inverted File Index)** partitions the space into clusters using the K-means algorithm, creating from 1024 to 65536 centroids. Each vector is assigned to the nearest centroid and stored in the corresponding inverted list. During search, the system finds the nprobe nearest centroids to the query and scans only their document lists, ignoring the remaining clusters.

Tuning parameters: nlist sets the number of clusters (more clusters = more accurate, but slower index construction), nprobe determines how many clusters to check during search (trade-off between speed and quality).

IVF is best for massive data volumes (billions of vectors), limited memory, situations where some accuracy can be sacrificed, and frequently updated data.

**PQ (Product Quantization)** compresses vectors, significantly reducing memory usage. The idea is to split the vector into subvectors, each quantized separately. A 128-dimensional vector is divided into 8 subvectors of 16 dimensions each. Each subvector is encoded through a lookup table with 256 centroids, requiring only 1 byte. The result is 64x compression. Distance between vectors is computed via precomputed lookup tables.

Combined indexes (IVFPQ, IVFHNSW) merge the advantages of different approaches.

### Popular Solutions and Their Trade-offs

**Pinecone** is a fully managed cloud service. The main advantage is zero operational overhead. You use the API; Pinecone manages all infrastructure, scaling, and replication. An excellent choice for teams without DevOps expertise. The drawback is vendor lock-in and growing costs at scale.

**Weaviate** is a hybrid solution: open-source with an optional managed cloud. It supports hybrid search out of the box and can automatically vectorize data through integrations with embedding services. The GraphQL API is convenient for complex queries. The self-hosted option requires Kubernetes for production.

**Milvus** is an enterprise-grade open-source system for petabyte-scale workloads. Distributed architecture with separation of storage, compute, and coordination. Supports GPU acceleration for indexing and search. A variety of index types (HNSW, IVF, DiskANN) allows fine-tuning of trade-offs. Deployment and administration complexity is a significant barrier.

**Qdrant** is a modern open-source solution written in Rust, optimized for usability and performance. Deploys as a single binary or Docker container. Supports complex payload filtering, which is critical for multi-tenant applications. REST API with good documentation. The sweet spot for most applications.

**Chroma** is an embeddable database for prototyping and small applications. Runs in-process, requires no separate server. Ideal for getting started, local development, and demos. Not suitable for production-scale workloads.

**pgvector** is a PostgreSQL extension for vector search. If your application already uses PostgreSQL, this is the minimal path to RAG without new infrastructure. Store vectors alongside regular data, use transactions, foreign keys, and all familiar Postgres capabilities. The limitation is that performance falls behind specialized solutions at large volumes.

### Storage Selection Criteria

**Scale** is the first and most important factor. Thousands of vectors (a typical corporate knowledge base) — almost any solution will work, including in-memory stores or pgvector. Millions of vectors (e-commerce catalog, large documentation) — optimized indexes like HNSW are needed. Billions of vectors (search engine, social network) — only distributed systems like Milvus will handle this efficiently.

**Metadata filtering** is critical for multi-tenant applications and complex search scenarios. A RAG system for a company with departments requires filtering by department while preserving vector search.

Filtering implementations vary dramatically. Pre-filtering (first filter documents by metadata, then search for nearest neighbors among them) is accurate but slow if the filter leaves few documents. Post-filtering (find K nearest neighbors, then filter by metadata) is fast but may return fewer than K results. Qdrant and Weaviate support efficient integration of filters into vector search.

**Update frequency** affects index choice. Static data (archives, encyclopedias) can be indexed once with a complex index, achieving maximum search speed. Frequently changing data (news sites, forums, real-time monitoring) requires indexes with fast incremental updates.

**Operational requirements** — a realistic assessment of team capabilities. Are there DevOps engineers? Kubernetes experience? Monitoring, alerting, backup strategies? Managed solutions (Pinecone, Weaviate Cloud) remove all this burden at the cost of price and some loss of control.

## Hybrid Search

### Limitations of Purely Vector Search

Semantic search is a powerful tool, but it has blind spots. The main problem is exact matches and specific terms.

A user searches for "error ERR_CONNECTION_REFUSED." Semantic search understands this is about errors and network problems. It will find articles about network errors, troubleshooting connectivity, TCP connection issues. All of this is semantically relevant. But the document containing the exact error code ERR_CONNECTION_REFUSED may not be at the top of the results.

Rare terms and specific nomenclature are another weakness. The embedding model was trained on a general text corpus. A corporate abbreviation or product identifier was unlikely to appear during training. The model does not know their meaning and cannot properly encode them in the semantic space.

People's names, codes, version numbers, exact wording from regulatory documents — all of these are better found by lexical search, which simply looks for text matches.

### Combining Approaches: The Best of Both Worlds

Hybrid search runs two parallel searches on a single query: semantic (vector) search finds conceptually similar documents, lexical (keyword-based) search finds exact text matches.

Lexical search is typically implemented via BM25 — an enhanced algorithm that accounts for term frequency and document length. BM25 assigns a high score to documents containing rare terms from the query, and a low score to documents with common words.

Merging results is a non-trivial task. Semantic search returns cosine similarity from -1 to 1 (typically 0.5–0.95 for relevant documents). BM25 returns arbitrary positive numbers. How do you compare these incomparable scores?

### Reciprocal Rank Fusion: An Elegant Solution

RRF (Reciprocal Rank Fusion) solves the problem of incomparable scores elegantly: it ignores them. Instead of absolute values, only ranking positions are used.

For each document, the sum of reciprocal ranks from all result lists is computed. A smoothing constant (typically 60) is added to the rank for stability. A document at position 1 in both lists receives the highest combined score. A document appearing in only one list receives a lower score.

Advantages of RRF: no calibration of scores between methods is required, it automatically balances contributions from different sources, it is robust to outliers (one method producing an anomalously high score), and it is simple to implement.

This makes RRF the standard for hybrid search in RAG systems.

## Efficiency Optimization

### Choosing Dimensionality

Higher dimensionality means better quality but more memory and slower search. For many tasks, 768 or even 384 dimensions are sufficient. 1536 and above are justified for critical applications or complex domains.

Some models support matryoshka embeddings — a subset of dimensions can be used without losing compatibility. Starting with the first 256 dimensions for coarse search, you can refine with the full 1024.

### Matryoshka Embeddings (2024)

Matryoshka embeddings are an innovative approach to training embeddings that allows using variable dimensionality without retraining the model.

In traditional embeddings, the model generates a vector of fixed dimensionality (e.g., 768 dimensions), and you either use all dimensions or none. Matryoshka embeddings work differently: the same 768-dimensional vector can be used partially. The first 64 dimensions give a coarse representation, the first 128 are better, the first 256 are better still, and all 768 give full quality.

The loss function is computed as a weighted sum of contrastive loss for different vector prefixes: 64, 128, 256, 512, 768 dimensions. Each prefix is trained to be useful independently of the rest, allowing flexible dimensionality selection at inference time.

Practical applications: initial recall with 64–128 dimensions for speed and coarse filtering, final ranking with the full 768 dimensions for quality and precise ordering, mobile/edge with 128–256 for limited memory, production trade-off with 256–512 for balance.

Models supporting Matryoshka: OpenAI text-embedding-3 (dimensions parameter), Nomic Embed, Jina Embeddings v2, E5-mistral-7b-instruct.

### Quantization

Quantization reduces numerical precision to save memory. Float32 takes 4 bytes per dimension, int8 takes 1 byte. For one million vectors of dimensionality 1536, this is the difference between 6 GB and 1.5 GB.

Modern quantization algorithms have minimal impact on search quality. A 1–2% accuracy loss is usually acceptable in exchange for 4x memory savings.

### Filtering and Partitioning

If data naturally divides into categories, partitioning speeds up search. Searching within the "Documentation" partition does not touch the "Blog" partition.

Efficient filtering requires proper indexes. Filtering by a rare value is faster than by a frequent one. A rare-category filter will eliminate most documents before vector search.

### Caching

Frequent queries can be cached. Caching at the query embedding level saves embedding model calls. Caching at the results level saves vector search.

It is important to define an invalidation strategy. When documents are updated, the results cache becomes stale.

## Monitoring and Metrics

### Operational Metrics

**Latency** — time from query to result. P50, P95, P99 show typical and worst-case performance. The target is sub-100ms for interactive applications.

**Throughput** — queries per second. Important for high-load systems.

**Memory usage** — memory consumed by the index. Determines infrastructure cost.

**Recall** — the fraction of truly relevant documents among those retrieved. Measured on a test set with known answers.

### Quality Metrics

**Hit rate** — how often does a relevant document appear in the top-K? If the needed document is in the top-5 for 90% of queries, the system is working well.

**Mean Reciprocal Rank** — the average reciprocal position of the first relevant result. MRR=1 means perfect ranking.

Regular measurement of these metrics on a representative test set allows tracking degradation and the effect of optimizations.

## Key Takeaways

Embeddings convert text into numerical representations where semantic similarity is reflected by geometric proximity of vectors. This is the foundation of semantic search in RAG.

The choice of embedding model is determined by quality, dimensionality, speed, and language support. Benchmarks help, but testing on real data is critical.

Vector stores are optimized for nearest-neighbor search. The choice depends on scale, filtering requirements, and operational capabilities.

Hybrid search combines semantic and lexical approaches, compensating for the limitations of each.

Optimization includes dimensionality selection, quantization, filtering, and caching. Metric monitoring provides observability and quality control.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → RAG
**Previous:** [[02_Chunking_Strategies|Document Chunking Strategies]]
**Next:** [[04_Retrieval_Methods|Information Retrieval Methods]]
