# Retrieval-Augmented Generation (RAG) Basics

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → RAG
**Previous:** [[../05_MCP_Protocol/05_A2A_Protocol|Agent-to-Agent Protocol]]
**Next:** [[02_Chunking_Strategies|Document Chunking Strategies]]

---

## Introduction to RAG

Retrieval-Augmented Generation (RAG) solves a fundamental problem of language models: training data is limited to a specific cutoff date and lacks access to domain-specific information.

RAG gives the model access to up-to-date external information at the time of response generation. The model finds relevant facts in the provided context instead of extracting them from network parameters.

## Problems RAG Solves

### Model Knowledge Limitations

Language models are trained on data collected up to a certain date. Everything that happened after that cutoff date is unknown to the model. For many practical applications — from technical support to legal consulting — information freshness is critically important.

Moreover, even for events before the cutoff date, the model may have incomplete or distorted knowledge. Training data is uneven: popular topics are better represented, niche topics are worse. Internal company documentation, specific procedures, private knowledge bases — all of this is absent from the model by definition.

RAG overcomes these limitations by connecting the model to up-to-date data sources. Instead of retraining the model every time documentation is updated, we simply update the knowledge base that the model queries.

### Hallucinations and Reliability

Language models are prone to hallucinations — generating plausible-sounding but factually incorrect information. The model is optimized for producing coherent text, not factual accuracy. It can confidently assert nonexistent facts, confuse dates, and fabricate quotes.

RAG significantly reduces this problem by grounding responses in specific documents. When the model generates a response based on provided context, it is less likely to fabricate facts. Additionally, verification becomes possible: one can reference the source and verify the information.

This does not mean complete elimination of hallucinations — the model can still misinterpret context or mix information from different sources. However, the problem becomes manageable, and responses become significantly more reliable.

### Specialization Without Retraining

The traditional approach to model specialization is fine-tuning on domain-specific data. This requires significant computational resources, ML expertise, and time. Every time data is updated, the process must be repeated.

RAG offers an alternative: specialization through context. The knowledge base can contain terminology, procedures, examples, and facts for a specific domain. The model uses this information while retaining its general-purpose core capabilities.

This approach is especially valuable for small organizations and specific tasks where building a specialized model is economically impractical.

## RAG System Architecture

### Conceptual Overview

A RAG system consists of two main components: the retriever and the generator. The retriever is responsible for finding relevant documents in the knowledge base. The generator — the language model — uses the retrieved documents to form a response.

The typical query processing flow is as follows. The user asks a question. The retriever converts the question into a search query and finds the most relevant documents or their fragments. The retrieved context is combined with the original question and passed to the generator. The generator forms a response based on the provided context.

This simple scheme has many variations and refinements, but the basic principle remains the same: separation of retrieval and generation tasks.

### Indexing and Data Preparation

Before a RAG system can answer questions, the knowledge base must be prepared. This process is called indexing and includes several stages.

The first stage is document collection. These can be files in various formats: PDF, Word, HTML, Markdown, text files. They can also be data from databases, APIs, and content management systems.

The second stage is text extraction. Textual content is extracted from heterogeneous formats. Preserving structure and metadata is important here: headings, authors, dates, categories.

The third stage is chunking. Documents are split into appropriately sized parts. Chunk size and splitting strategy significantly affect retrieval quality.

The fourth stage is vectorization. Each chunk is converted into a numerical vector representing its semantic meaning.

The fifth stage is index storage. Vectors and their associated texts are stored in a specialized store optimized for fast similar-vector search.

### Information Retrieval Theory

Before discussing semantic search, it is useful to understand classical IR methods that powered search systems for decades.

**TF-IDF** — the idea is simple: a word is important for a document if it appears frequently in it but rarely in the corpus overall. Term Frequency is calculated as the ratio of the term's frequency in the document to the total number of terms in that document. Inverse Document Frequency is calculated as the logarithm of the ratio of the total number of documents to the number of documents containing the term. The final TF-IDF measure is the product of these two values.

The word "the" has a high TF but a low IDF. A specific term like "tokenization" has a low TF in most documents but a high IDF — yielding a high TF-IDF where it appears.

**BM25** — an improvement over TF-IDF that became the de facto standard for lexical search. The algorithm accounts for saturation: after a certain frequency, additional term repetitions contribute less weight. Length normalization is also applied — long documents do not gain an unfair advantage.

The BM25 formula sums the contribution of each query term, where each term's contribution depends on its IDF and its frequency in the document, adjusted by saturation and length normalization parameters. The k1 parameter controls how quickly saturation is reached. The b parameter determines the influence of document length on the final score.

**Lexical vs Semantic Search:** Lexical search is based on exact word matching. Semantic search operates on meaning similarity. BM25 will not find a document containing "Machine Learning" for the query "ML"; semantic search will. BM25 cannot distinguish "bank" (financial institution) from "bank" (river bank) since the keyword is identical; semantic search will distinguish them through context. BM25 works with new terms immediately; semantic search requires the model to be trained on them. BM25 is very fast; semantic search requires a GPU for embedding generation.

Hybrid search combines both approaches: BM25 finds exact matches, semantic search finds conceptually similar documents. Results are merged through ranking methods like Reciprocal Rank Fusion.

### Retrieving Relevant Context

When a user query arrives, the system searches the prepared index. The query is also converted into a vector using the same embedding model. Then a nearest neighbor search is performed — finding chunks whose vectors are most similar to the query vector.

Search results are ranked by relevance. Typically, top-K documents are selected, where K is determined by the balance between context completeness and model input size limitations.

It is important to understand that semantic search differs from traditional full-text keyword search. It finds conceptually similar documents even if they do not contain exact words from the query. The question "How do I connect a payment module?" can retrieve a document about payment gateway integration, even if the word "payment" is absent from it.

### Response Generation

The retrieved context and original question are combined into a prompt for the language model. The typical format includes an instruction, the context itself, and the user's question. A good RAG prompt explicitly instructs the model to acknowledge the absence of information if the answer is not in the context.

Response quality depends on both the quality of the retrieved context and the model's ability to use it correctly. Both components require attention during system design.

### Practical Pipeline

A modern production-ready pipeline uses specialized solutions at each stage.

**Stage 1: Ingest** — Firecrawl, Crawl4AI, or Unstructured.io collect data from web pages, PDF documents, APIs, and databases. Firecrawl is a commercial web scraper with JavaScript support that automatically extracts clean markdown. Crawl4AI is an open-source alternative with an LLM-first approach. Unstructured.io processes various formats. LlamaHub provides a collection of 100+ loaders for various data sources.

**Stage 2: Chunk** — LlamaIndex or LangChain split documents into optimally sized parts. LlamaIndex offers semantic chunking and sentence splitter with a rich selection of strategies. LangChain provides RecursiveCharacterTextSplitter and SemanticChunker with flexible overlap settings. Jina AI Segmenter uses Late Chunking — chunking after embedding the entire document.

**Stage 3: Embed** — converting text into numerical vectors for semantic search. OpenAI text-embedding-3-large remains a strong commercial option. Cohere embed-v3 supports multilingual and compression. bge-m3 is a top open-source choice: multilingual with support for 100+ languages and hybrid retrieval. GTE-Qwen2 and NV-Embed-v2 compete at the top of MTEB benchmarks among open-source models. jina-embeddings-v3 uses late interaction and a multilingual approach.

**Stage 4: Store** — vector databases for storing embeddings and fast search. Qdrant is a specialized database for production with high performance and filtering. Weaviate is a specialized database with hybrid search out of the box. Pinecone is a managed service, serverless with minimal ops overhead. Chroma is an embedded database for prototyping. pgvector is a PostgreSQL extension for when PostgreSQL is already in use. Milvus is a distributed system for scaling to billions of vectors.

**Stage 5: Rerank** — improving result quality through cross-encoder models. A cross-encoder analyzes a "query-document" pair jointly, yielding a more accurate relevance score. Cohere Rerank is the quality leader, multilingual. Pinecone Rerank is integrated into Pinecone. bge-reranker-v2-m3 is open-source, multilingual. Jina Reranker has open weights with good quality.

A typical pattern: first retrieve top-100 documents using fast embedding search, then rerank them with a cross-encoder down to top-10 for passing to the LLM.

**Stage 6: Evaluate** — RAGAS evaluates Context Relevancy, Faithfulness, and Answer Relevancy. LangSmith provides tracing, evaluation datasets, and A/B tests. Phoenix offers LLM observability and tracing. TruLens provides comprehensive RAG evaluation with feedback functions.

A typical production RAG pipeline: Ingest → Chunk → Embed → Store → Retrieve (hybrid search) → Rerank → Generate → Evaluate → Observe.

## Comparison with Alternative Approaches

### RAG vs Fine-tuning

Fine-tuning "imprints" knowledge into model weights through additional training. RAG provides knowledge in the context with each request.

Fine-tuning is preferable when you need to change the model's style, format, or behavior. If the model must always respond in a specific tone or use specialized terminology as a natural part of its language, fine-tuning may be more effective.

RAG is preferable for working with factual knowledge, especially frequently updated knowledge. Keeping documentation current through RAG is trivial — simply update the knowledge base. Keeping it current through fine-tuning requires retraining.

In practice, these approaches are often combined. The model is fine-tuned to understand domain specifics and formatting, while RAG provides access to current data.

### RAG vs Long Context

Modern models support increasingly long contexts — hundreds of thousands of tokens. This raises the question: wouldn't it be simpler to load all documents directly into the context?

In theory, yes. In practice, there are several limitations. First, cost — processing a long context is proportionally more expensive. Second, quality — models perform worse with information in the middle of a very long context (the "lost in the middle" effect). Third, volume — even a million tokens cannot accommodate a serious enterprise knowledge base.

RAG remains relevant as a way to efficiently select the most relevant information from a large corpus. Long context and RAG complement each other: RAG finds relevant documents, and long context allows fitting more of what was found.

### RAG and Knowledge Graphs

Integrating RAG with knowledge graphs combines the advantages of structured and unstructured data.

**Vector RAG:** Documents are split into chunks, converted into embeddings, and search is performed by semantic similarity. It works with any text and captures implicit relationships. However, extracting precise facts is difficult, and there are no explicit relationships between entities.

**Graph RAG:** Text is analyzed to extract entities and relationships, a knowledge graph is built, and navigation proceeds through traversal. It yields precise facts and explicit relationships. However, it requires entity extraction and loses nuances.

**Hybrid approach:** The graph is used for navigation and structural queries, vectors are used for semantic search. For example: "Find companies associated with Elon Musk and documents about their financial results" — the graph finds related companies, and vector search finds relevant financial documents.

A typical Graph RAG pipeline includes: Entity Extraction, Relation Extraction, Graph Construction, Entity Linking with external knowledge bases, and Retrieval through a combination of graph traversal and vector search.

## RAG Quality Metrics

### Retrieval Metrics

Retriever quality is measured using classical IR metrics.

**Precision@K** shows what fraction of the K retrieved documents are actually relevant. **Recall@K** shows what fraction of relevant documents made it into the top-K. **Mean Reciprocal Rank** accounts for the position of the first relevant document. **NDCG** considers both relevance and positions of all documents, giving higher weight to documents ranked higher.

### Generation Metrics

Response quality is evaluated from several perspectives.

**Faithfulness** — does the response correspond to the provided context, are there hallucinations. Low Faithfulness indicates hallucination — the model added information not present in the context.

**Answer Relevance** — does the response address the question asked. Measured through semantic similarity between the question and the answer plus response completeness. Low Relevance means the answer is off-topic or incomplete.

**Context Relevance** — was the retrieved context useful for answering. Evaluates how much the provided context helped the model answer the question.

These metrics are often confused, but they measure different things. An ideal RAG system maximizes all metrics, but sometimes priorities must be chosen.

Automatic evaluation of these metrics is challenging. LLM-as-judge is commonly used: another language model evaluates response quality against defined criteria. Human evaluations are also used, especially during system development and tuning.

### End-to-end Metrics

The ultimate goal of RAG is useful answers for users. End-to-end metrics evaluate exactly this: answer accuracy on test questions, user satisfaction, and task resolution speed.

Creating a high-quality test set is a critical investment. The set should cover typical questions, edge cases, and questions with no answer in the knowledge base.

## Key Takeaways

RAG is a powerful paradigm that extends language model capabilities through access to external knowledge. The technology addresses key LLM problems: time-limited knowledge, hallucinations, and difficulty of specialization.

RAG architecture separates retrieval and generation tasks, allowing each component to be optimized independently. Indexing prepares the knowledge base, the retriever finds relevant context, and the generator forms a response based on what was found.

RAG does not exclude other approaches but complements them. Combining it with fine-tuning allows model behavior to be customized. Long contexts in modern models increase the volume of usable information.

Measuring RAG quality requires attention to both components: retrieval effectiveness and generation quality. Creating test sets and systematic evaluation are essential parts of development.

In the following sections, we will examine each RAG system component in detail: document chunking strategies, working with embeddings, search methods, and advanced techniques for improving quality.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → RAG
**Previous:** [[../05_MCP_Protocol/05_A2A_Protocol|Agent-to-Agent Protocol]]
**Next:** [[02_Chunking_Strategies|Document Chunking Strategies]]
