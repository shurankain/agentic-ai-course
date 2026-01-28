# Late Interaction Models

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → RAG
**Previous:** [[05_Advanced_RAG|Advanced RAG Techniques]]
**Next:** [[../07_Frameworks/00_Frameworks_Overview|AI Frameworks Overview]]

---

## Problems with Dense Retrieval

The traditional Dense Retrieval approach uses **a single vector** to represent a document or query. This creates fundamental limitations.

### Single Vector Bottleneck

When an entire document is compressed into a single fixed-dimension vector (typically 768-1536), information is lost:

| Problem | Description | Consequences |
|---------|-------------|--------------|
| Information Loss | Long document → one vector | Details and nuances are lost |
| Semantic Averaging | Different topics get averaged | Precision decreases |
| Query Mismatch | Different parts of the query are ignored | Relevant documents are missed |
| Long Documents | Especially critical for >512 tokens | Late chunking partially addresses this |

### Example of the Problem

A document contains information about Python, JavaScript, and Rust. With single vector embedding:
- The vector will be "averaged" across the three languages
- A query for "Python async/await" may not find the document
- A query about Rust will find the document with a low score

---

## Late Interaction: An Alternative Approach

**Late Interaction** is an architecture where the interaction between query and document occurs at the level of individual tokens rather than entire texts.

### The Multi-Vector Representation Idea

Instead of a single vector, multiple vectors are created — one per token (or group of tokens).

| Approach | Representation | Interaction |
|----------|----------------|-------------|
| Dense Retrieval | 1 vector per document | Cosine similarity of two vectors |
| Late Interaction | N vectors (per token) | MaxSim between sets of vectors |

### Why "Late"?

- **Early Interaction** (Cross-Encoder): query and document are processed together through a transformer — accurate but slow (O(n²))
- **No Interaction** (Bi-Encoder): separate embeddings, only cosine similarity — fast but loses nuances
- **Late Interaction**: separate embeddings but sophisticated interaction via MaxSim — a balance of speed and quality

---

## The MaxSim Operator

**MaxSim (Maximum Similarity)** is the key mechanism of Late Interaction models.

### How It Works

For each query token, the most similar document token is found:

1. Query tokens: q₁, q₂, ..., qₘ (each is a vector)
2. Document tokens: d₁, d₂, ..., dₙ (each is a vector)
3. For each qᵢ, find max(sim(qᵢ, dⱼ)) across all j
4. Score = sum of all maximums

### Formula

The final score between query Q and document D is computed as a sum over all query tokens qᵢ, where for each query token the maximum dot product with document tokens dⱼ is taken. Mathematically: Score equals the sum from i to m (where m is the number of query tokens) of the maximum dot product of qᵢ and dⱼ across all j (where j ranges over all document tokens from 1 to n).

### Intuition

Each word of the query "searches" for the most appropriate word in the document. This allows:
- Finding exact term matches
- Handling synonyms (semantically close tokens)
- Ignoring irrelevant parts of the document

---

## Late Interaction Models

### ColBERT and Its Evolution

| Model | Organization | Features | Application |
|-------|-------------|----------|-------------|
| ColBERT v1 | Stanford | Original architecture | Research |
| ColBERTv2 | Stanford | Improved training, less memory | Production-ready |
| Jina-ColBERT-v2 | Jina AI | 89 languages, 8192 tokens | Multilingual RAG |
| GTE-ModernColBERT | Alibaba | SOTA on BEIR benchmark | Enterprise |

### ColBERTv2 (Stanford, 2021-2024)

Key improvements over v1:
- **Residual Compression**: compression of token vectors
- **Denoised Supervision**: improved training
- **Cross-encoder Distillation**: knowledge from a more accurate model

Results:
- Quality comparable to cross-encoder
- Speed an order of magnitude faster than cross-encoder
- 10-100x more memory-efficient vs v1

### Jina-ColBERT-v2 (Jina AI, 2024)

Features:
- **89 languages** including Russian
- **8192 tokens** context (vs 512 in the original)
- **Flash Attention 2** for efficiency
- Compatible with ColBERT ecosystem

When to use:
- Multilingual systems
- Documents >512 tokens
- International products

### GTE-ModernColBERT (Alibaba, 2024)

Current SOTA on BEIR benchmark:
- Best result among Late Interaction models
- Outperforms many cross-encoder models
- Optimized for production

---

## ColPali: Vision + Late Interaction

**ColPali** and **ColQwen** are groundbreaking models for working with documents as images.

### The Problem with the Traditional Approach

Processing PDF documents typically requires:
1. OCR for text extraction
2. Layout detection for structure
3. Table extraction for tables
4. Chunking of the extracted text
5. Embedding of text chunks

Problems:
- OCR errors accumulate
- Visual information is lost (charts, diagrams)
- Complex pipeline with multiple points of failure
- Tables and formulas are handled poorly

### The ColPali Approach

ColPali processes a PDF page as an image:

1. **Input:** image of a document page
2. **Vision Encoder:** PaliGemma (Vision Language Model)
3. **Output:** grid of patch embeddings (Late Interaction representation)
4. **Retrieval:** MaxSim between text query and visual patches

### Advantages

| Aspect | Traditional Pipeline | ColPali |
|--------|---------------------|---------|
| OCR | Required | Not needed |
| Tables | Problematic | Native |
| Charts | Lost | Preserved |
| Pipeline | 5+ steps | 1 step |
| Latency | High | Low |

### ColQwen (2024)

Improved version based on Qwen2-VL:
- Better quality on documents
- High resolution support
- Multilingual capabilities from Qwen

---

## PyLate: Production-Ready Library

**PyLate** (LightOn, 2024) is a library for working with Late Interaction models in production.

### Capabilities

- Compatibility with Hugging Face models
- Integration with Sentence-Transformers
- Efficient storage of sparse representations
- Optimized inference

### Architecture

PyLate wraps existing ColBERT-style models and provides:
- Unified API for different models
- Batch processing
- Efficient storage format
- Integration with vector stores

---

## Comparison of Retrieval Approaches

| Approach | Quality | Speed | Memory | Application |
|----------|---------|-------|--------|-------------|
| BM25 | Low | Very fast | Minimum | Baseline, keyword |
| Dense (Bi-Encoder) | Medium | Fast | Medium | Production RAG |
| Late Interaction | High | Medium | More | High-precision RAG |
| Cross-Encoder | Very high | Slow | Low | Reranking top-K |

### When to Use Late Interaction

**Recommended:**
- High precision requirements
- Documents with heterogeneous content
- Technical documents (code, formulas)
- Financial and legal documents
- When reranking is needed anyway

**Not recommended:**
- Simple FAQ systems
- Limited memory resources
- Billions of documents (Google scale)
- When BM25 is sufficient

---

## Hybrid Architecture: The Best of Both Worlds

The optimal production architecture often combines approaches:

### Two-Stage Retrieval

**Stage 1: Candidate Generation**
- BM25 or Dense Retrieval
- Fast search for top-1000 candidates
- Priority on recall

**Stage 2: Precise Ranking**
- Late Interaction (ColBERT) or Cross-Encoder
- Reranking top-100
- Priority on precision

### Three-Stage with ColPali

For documents with visual content:

**Stage 1:** Dense retrieval on metadata and descriptions
**Stage 2:** ColPali on page images
**Stage 3:** Cross-encoder for final ranking

---

## Related Topics

- [[04_Retrieval_Methods|Retrieval Methods]] — basic search approaches
- [[05_Advanced_RAG|Advanced RAG]] — taxonomy of RAG architectures
- [[03_Embeddings_and_Vector_Stores|Embeddings and Vector Stores]] — vector storage
- [[../11_Evaluation_Testing/05_RAG_Evaluation|RAG Evaluation]] — retrieval quality metrics

---

## Key Takeaways

1. **Single Vector Bottleneck** — a single vector loses information, especially for long documents

2. **Late Interaction solves this** through multi-vector representation and the MaxSim operator

3. **SOTA models:**
   - ColBERTv2 — balanced production choice
   - Jina-ColBERT-v2 — 89 languages, long context
   - GTE-ModernColBERT — best on BEIR
   - ColPali/ColQwen — PDF as image

4. **When to use Late Interaction:**
   - High-precision requirements
   - Complex documents
   - Resources available for multi-vector storage

5. **Production pattern:** Dense retrieval (recall) → Late Interaction (precision)

---

## Practical Code Examples

### Basic Usage of PyLate with ColBERT

To work with ColBERT through the PyLate library, you need to import the models and rank modules. The process begins with loading a pretrained ColBERT model, for example "colbert-ir/colbertv2.0" from Hugging Face Hub, which is initialized via the models.ColBERT constructor with the model identifier.

A list of documents is prepared for indexing — these can be strings of various content, such as text about the Python and Rust programming languages. Document encoding is performed using the encode method with the mandatory parameter is_query=False, which signals the model to use document processing mode. The batch_size parameter controls the batch size when processing multiple documents; for example, a value of 32 allows processing 32 documents simultaneously for faster throughput.

The result of document encoding is a multi-vector representation where each document is represented not by a single vector but by a set of vectors (one per token). Similarly, the query is encoded using the encode method but with is_query=True, which switches the model to query processing mode. The query is also transformed into a set of token embeddings.

The final step is computing scores via the rank.colbert_score function, which takes query embedding and doc_embeddings and returns a list of relevance scores for each document. Internally, this function implements the MaxSim algorithm: for each query token, it finds the maximum similarity to tokens of each document, sums these maximums, and produces the final document score. For example, for a query about memory safety in programming, a document about Rust will receive a higher score (e.g., 0.89) than a document about Python (0.72), since the tokens "memory" and "safety" will find closer matches in the Rust description.

**Key implementation points:**

**PyLate with multilingual models:** To work with texts in Russian, Chinese, and other languages, the `jinaai/jina-colbert-v2` model is used, which supports 89 languages. When encoding, it is important to specify the `output_value="token_embeddings"` parameter to obtain a multi-vector representation instead of a single vector.

**ColPali for PDF documents:** The model processes PDF pages as images via `ColPali.from_pretrained("vidore/colpali-v1.2")`. The process involves loading a page image, obtaining visual embeddings through `process_images()`, encoding the text query through `process_queries()`, and computing MaxSim between the text query and visual document patches. This completely eliminates the need for OCR and preserves visual information.

**Hybrid Retrieval pattern:** The two-stage architecture starts with fast dense retrieval to obtain 100-1000 candidates from a vector store (using a standard model like `all-MiniLM-L6-v2`). These candidates are then reranked through a ColBERT model to produce precise top-K results. The first stage ensures high recall, the second — high precision.

**Visual Document Store:** For PDF indexing, a dictionary is created where the key is `{pdf_path}:page_{i}` and the value is the multi-vector embeddings of the page. Each page is converted to an image, processed through ColPali, and the embeddings are stored. During search, the text query is encoded, a score is computed with each page via MaxSim, and results are sorted by relevance.

**Efficient storage:** PyLate provides `indexes.ColBERTIndex` with support for residual compression and quantization (`n_bits=2`), which significantly reduces memory requirements. The index is created with a specified path and compression parameters, then documents are added in batches, and search is performed via the `search()` method returning (doc_id, score) pairs.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → RAG
**Previous:** [[05_Advanced_RAG|Advanced RAG Techniques]]
**Next:** [[../07_Frameworks/00_Frameworks_Overview|AI Frameworks Overview]]
