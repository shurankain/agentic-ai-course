# Project: Production-Ready RAG Chatbot

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Practical Projects
**Previous:** [[../18_AI_Governance/07_Enterprise_AI_Adoption|Enterprise AI Adoption]]
**Next:** [[02_Multi_Agent_System|Multi-Agent System]]

---

## Project Overview

We will build a full-featured RAG chatbot for enterprise documentation — from document ingestion to production deployment. The system processes thousands of documents, supports multiple users, and operates reliably in production.

### Functionality

- Document ingestion (PDF, DOCX, Markdown, HTML)
- Semantic search with hybrid retrieval (dense + sparse)
- Conversational interface with dialog memory
- Source citation
- Streaming responses

### Production Characteristics

- Structured logging and tracing
- Performance metrics
- Graceful error handling
- Rate limiting
- Health checks

## Architecture

The Document Ingestion Pipeline loads and normalizes documents of various formats. The Chunking and Embedding module splits text into semantically meaningful fragments of 512 tokens with an overlap of 64 tokens, generating vector representations via text-embedding-3-small.

The Vector Store (Qdrant) provides fast similarity search. The Retrieval Service performs hybrid search (dense + sparse) with reranking. LLM Generation (Claude) produces answers based on retrieved context. Response Formatting adds source citations. Conversation Memory provides context for multi-turn dialogs.

## Document Ingestion

### Problem Statement

Answer quality depends on indexing quality. Poor chunking loses context. Incorrect embeddings yield irrelevant retrieval. Lost metadata makes citation impossible.

The pipeline must: extract text while preserving structure, split into optimally sized chunks with semantic awareness, enrich metadata for filtering and attribution, generate high-quality embeddings, index into the vector store with error handling.

### Document Loaders

The unified DocumentLoader interface abstracts format differences. PDF requires proper linearization of multi-column layouts. DOCX has rich formatting. Markdown includes code blocks requiring special handling.

A document contains: text content, file source, title, format, creation date, page number (PDF), section (structured documents). Metadata is critical for citation and filtering.

### Chunking Strategy

Recursive character splitter with semantic awareness: target size 512 tokens, overlap 64 tokens (12.5%), split boundaries paragraph → sentence → word, preserve code blocks in their entirety.

Overlap avoids information loss at boundaries — a split sentence is fully present in at least one fragment.

### Embeddings

text-embedding-3-small provides a balance of quality and cost. Best practices: batch processing (grouping chunks), rate limiting (respecting API limits), retry logic (handling transient failures), caching (avoiding redundant generation).

## Vector Store and Retrieval

### Qdrant

Open-source vector database with: hybrid search (dense + sparse), metadata filtering, collection management, snapshot and backup, Docker deployment.

Alternatives: Pinecone (managed service), Weaviate (rich query language), Milvus (high scalability), pgvector (if PostgreSQL is already in use).

### Retrieval Strategy

Multi-strategy retrieval outperforms basic semantic search:

1. Dense retrieval — semantic similarity via embeddings
2. Sparse retrieval — BM25 for exact keyword match
3. Hybrid fusion — Reciprocal Rank Fusion combines results

Enhancements: Query expansion (LLM generates alternative phrasings), HyDE (generating a hypothetical answer for search), Reranking (cross-encoder re-ranks top candidates).

### Conversation Context

Query rewriting reformulates the query using conversation history. The context window includes recent history in the prompt. Conversation summarization compresses long dialogs.

## Generation and Response

### Prompt Engineering

The system prompt defines the role and rules: answer only based on the provided context, acknowledge missing information ("I don't have information about that"), cite sources [Source: document_name], be concise yet thorough.

The context section contains retrieved chunks with metadata. Conversation history includes recent messages for coherence. The user query is the current question, possibly reformulated.

### Streaming

Server-Sent Events or WebSocket for responsive UX. The user sees progress immediately rather than waiting 10 seconds.

### Citation

Inline citations: "OAuth2 tokens expire after 24 hours [Source: auth-guide.md, Section 3.2]". Critical for trust (information verification), compliance (audit trail), debugging (source of incorrect answers).

## Production Hardening

### Error Handling

Graceful handling: LLM API failures (timeout, rate limit, unavailable), vector store issues (connection errors, slow queries), invalid input (malformed queries, injection), resource exhaustion.

Strategies: retry with exponential backoff (transient errors), circuit breaker (cascading failure prevention), fallback responses (when the primary path fails), input validation and sanitization.

### Observability

Logging: structured JSON logs, correlation IDs for tracing, log levels (DEBUG/INFO), sensitive data masking.

Metrics: request latency (P50, P95, P99), token usage (input, output, cost), retrieval quality (chunks returned, relevance scores), error rates by type.

Tracing: end-to-end request traces, span for each stage, integration with Jaeger/Datadog.

### Rate Limiting

Token bucket algorithm with: per-user limits, global limits, burst allowance for legitimate spikes. Protection against abuse, cost control, resource protection.

## Evaluation (RAGAS and Beyond)

RAG systems require specialized evaluation that measures both retrieval and generation quality.

### RAGAS Framework

RAGAS (Retrieval Augmented Generation Assessment) provides metrics specifically designed for RAG:

**Faithfulness** — does the answer use only information from the retrieved context? Measures hallucination risk. Target: >0.9.

**Answer Relevancy** — does the answer address the actual question? Penalizes off-topic responses. Target: >0.85.

**Context Precision** — are the retrieved documents relevant to the question? Measures retrieval quality. Target: >0.8.

**Context Recall** — does the retrieved context contain all the information needed to answer? Measures retrieval completeness. Target: >0.8.

### Evaluation Pipeline

Automated evaluation: create a test set of 50-100 question-answer-context triples. Run RAGAS metrics on every deployment candidate. Set quality gates — block deployment if faithfulness drops below threshold.

LLM-as-Judge: for nuanced quality assessment, use a strong model (GPT-4, Claude) to evaluate response quality on dimensions like helpfulness, accuracy, and completeness.

Human evaluation: sample 50-100 responses weekly. Domain experts rate correctness and usefulness. Track trends over time.

### Continuous Monitoring

In production, track proxy metrics: user satisfaction (thumbs up/down), follow-up question rate (high = poor answer), escalation rate, retrieval score distribution. Alert on quality degradation before users notice.

## Key Takeaways

Production RAG requires attention to every layer. Document processing is underestimated — chunking strategy, metadata extraction, and embedding quality affect results more than LLM selection.

Hybrid retrieval (dense + sparse + reranking) significantly outperforms naive semantic search. Query expansion and HyDE further improve quality.

Evaluation is essential — RAGAS provides RAG-specific metrics (faithfulness, relevancy, precision, recall). Automate evaluation as a quality gate in the deployment pipeline.

Conversation context is critical for multi-turn dialogs. Query rewriting handles anaphora resolution.

Production hardening is not an afterthought. Error handling, observability, and rate limiting are designed from the start.

Streaming responses improve perceived latency. The user sees progress and does not leave.

---

## Navigation
**Previous:** [[../18_AI_Governance/07_Enterprise_AI_Adoption|Enterprise AI Adoption]]
**Next:** [[02_Multi_Agent_System|Multi-Agent System]]

---

## Practical Code Examples

### Minimal RAG Pipeline on Spring AI

```java
import org.springframework.ai.chat.ChatClient;
import org.springframework.ai.chat.ChatResponse;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.reader.TextReader;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Service for performing RAG (Retrieval-Augmented Generation).
 * Loads documents, indexes them, and answers questions using context.
 */
@Service
public class RagChatbotService {

    @Autowired
    private ChatClient chatClient;

    @Autowired
    private EmbeddingModel embeddingModel;

    private VectorStore vectorStore;

    public RagChatbotService(EmbeddingModel embeddingModel) {
        // Initialize in-memory vector store
        this.vectorStore = new SimpleVectorStore(embeddingClient);
    }

    /**
     * Load and index documents.
     *
     * @param documentPaths list of document paths
     */
    public void indexDocuments(List<Path> documentPaths) {
        // Configure text splitting strategy
        // 512 tokens per chunk, overlap of 64 tokens (12.5%)
        TokenTextSplitter textSplitter = new TokenTextSplitter(512, 64);

        for (Path path : documentPaths) {
            try {
                // Load document
                TextReader reader = new TextReader(path.toString());
                List<Document> documents = reader.get();

                // Enrich with metadata for citation
                documents.forEach(doc -> {
                    doc.getMetadata().put("source", path.getFileName().toString());
                    doc.getMetadata().put("file_path", path.toString());
                });

                // Split into chunks while preserving context
                List<Document> chunks = textSplitter.split(documents);

                // Generate embeddings and save to vector store
                vectorStore.add(chunks);

                System.out.println("Indexed: " + path.getFileName() +
                                 " (" + chunks.size() + " chunks)");

            } catch (Exception e) {
                // Graceful error handling - continue with other documents
                System.err.println("Error indexing " + path + ": " + e.getMessage());
            }
        }
    }

    /**
     * Search for relevant documents by query.
     *
     * @param query user query
     * @param topK number of most relevant chunks
     * @return list of found documents
     */
    public List<Document> retrieveRelevantDocuments(String query, int topK) {
        // Semantic search via embeddings
        return vectorStore.similaritySearch(query, topK);
    }

    /**
     * Generate an answer using retrieved context.
     *
     * @param userQuery user question
     * @return answer with source citations
     */
    public String generateAnswer(String userQuery) {
        // Stage 1: Retrieval - search for relevant chunks
        List<Document> relevantDocs = retrieveRelevantDocuments(userQuery, 5);

        if (relevantDocs.isEmpty()) {
            return "Unfortunately, I could not find relevant information in the knowledge base.";
        }

        // Stage 2: Build context with metadata for citation
        String context = buildContextWithSources(relevantDocs);

        // Stage 3: Generation - create prompt for LLM
        String systemPrompt = """
            You are an assistant for working with enterprise documentation.

            RULES:
            1. Answer ONLY based on the provided context
            2. If the information is not in the context, honestly state that
            3. ALWAYS cite sources in the format [Source: file_name]
            4. Be concise yet informative
            5. Use bulleted lists for better readability
            """;

        String userPrompt = String.format("""
            CONTEXT:
            %s

            USER QUESTION:
            %s

            Provide an answer with mandatory source citations.
            """, context, userQuery);

        // Create prompt with system and user messages
        Message systemMessage = new org.springframework.ai.chat.messages.SystemMessage(systemPrompt);
        Message userMessage = new UserMessage(userPrompt);

        Prompt prompt = new Prompt(List.of(systemMessage, userMessage));

        // Call LLM
        ChatResponse response = chatClient.call(prompt);

        return response.getResult().getOutput().getContent();
    }

    /**
     * Build context with source information.
     */
    private String buildContextWithSources(List<Document> documents) {
        return documents.stream()
            .map(doc -> {
                String content = doc.getContent();
                String source = (String) doc.getMetadata().get("source");

                return String.format("--- Source: %s ---\n%s\n", source, content);
            })
            .collect(Collectors.joining("\n"));
    }

    /**
     * Streaming answer for responsive UX.
     *
     * @param userQuery user question
     * @return stream of response tokens
     */
    public void streamAnswer(String userQuery, StreamCallback callback) {
        List<Document> relevantDocs = retrieveRelevantDocuments(userQuery, 5);

        if (relevantDocs.isEmpty()) {
            callback.onToken("Unfortunately, I could not find relevant information in the knowledge base.");
            callback.onComplete();
            return;
        }

        String context = buildContextWithSources(relevantDocs);
        String prompt = buildPrompt(context, userQuery);

        // Streaming response via callback
        chatClient.stream(new Prompt(prompt))
            .subscribe(
                response -> callback.onToken(response.getResult().getOutput().getContent()),
                error -> callback.onError(error),
                () -> callback.onComplete()
            );
    }

    private String buildPrompt(String context, String query) {
        return String.format("""
            Answer only based on the context. Cite sources.

            Context:
            %s

            Question: %s
            """, context, query);
    }

    // Callback interface for streaming
    public interface StreamCallback {
        void onToken(String token);
        void onComplete();
        void onError(Throwable error);
    }
}
```

### Using Qdrant for Production

```java
import io.qdrant.client.QdrantClient;
import io.qdrant.client.QdrantGrpcClient;
import io.qdrant.client.grpc.Collections.*;
import io.qdrant.client.grpc.Points.*;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.UUID;
import java.util.concurrent.ExecutionException;

/**
 * Production-ready Vector Store using Qdrant.
 * Supports hybrid search and metadata filtering.
 */
@Component
public class QdrantVectorStore {

    private final QdrantClient client;
    private final EmbeddingModel embeddingModel;
    private static final String COLLECTION_NAME = "documents";

    public QdrantVectorStore(EmbeddingModel embeddingModel) {
        // Connect to Qdrant (Docker: localhost:6334)
        this.client = new QdrantClient(
            QdrantGrpcClient.newBuilder("localhost", 6334, false).build()
        );
        this.embeddingModel = embeddingClient;

        initializeCollection();
    }

    /**
     * Initialize collection with required parameters.
     */
    private void initializeCollection() {
        try {
            // Create collection with vector size = 1536 (text-embedding-3-small)
            VectorParams vectorParams = VectorParams.newBuilder()
                .setSize(1536)
                .setDistance(Distance.Cosine)
                .build();

            CreateCollection createCollection = CreateCollection.newBuilder()
                .setCollectionName(COLLECTION_NAME)
                .setVectorsConfig(VectorsConfig.newBuilder()
                    .setParams(vectorParams)
                    .build())
                .build();

            client.createCollectionAsync(createCollection).get();
            System.out.println("Qdrant collection created: " + COLLECTION_NAME);

        } catch (ExecutionException | InterruptedException e) {
            System.out.println("Collection already exists or error: " + e.getMessage());
        }
    }

    /**
     * Add documents to the vector store with metadata.
     */
    public void addDocuments(List<Document> documents) {
        documents.forEach(doc -> {
            try {
                // Generate embedding for the document
                List<Double> embedding = embeddingModel.embed(doc.getContent());

                // Prepare point with metadata for filtering
                PointStruct point = PointStruct.newBuilder()
                    .setId(PointId.newBuilder().setUuid(UUID.randomUUID().toString()))
                    .setVectors(Vectors.newBuilder().setVector(
                        io.qdrant.client.grpc.Points.Vector.newBuilder()
                            .addAllData(embedding.stream().map(Double::floatValue).toList())
                    ))
                    .putAllPayload(Map.of(
                        "content", Value.newBuilder().setStringValue(doc.getContent()).build(),
                        "source", Value.newBuilder().setStringValue(
                            (String) doc.getMetadata().get("source")
                        ).build()
                    ))
                    .build();

                // Insert into Qdrant
                client.upsertAsync(COLLECTION_NAME, List.of(point)).get();

            } catch (Exception e) {
                System.err.println("Error adding to Qdrant: " + e.getMessage());
            }
        });
    }

    /**
     * Semantic search with metadata filtering.
     *
     * @param query user query
     * @param limit number of results
     * @param sourceFilter filter by source (optional)
     */
    public List<ScoredPoint> search(String query, int limit, String sourceFilter)
            throws ExecutionException, InterruptedException {

        // Generate embedding for the query
        List<Double> queryEmbedding = embeddingModel.embed(query);

        // Build metadata filter (if specified)
        Filter.Builder filterBuilder = Filter.newBuilder();
        if (sourceFilter != null && !sourceFilter.isEmpty()) {
            filterBuilder.addMust(
                Condition.newBuilder()
                    .setField(FieldCondition.newBuilder()
                        .setKey("source")
                        .setMatch(Match.newBuilder()
                            .setKeyword(sourceFilter)
                        )
                    )
            );
        }

        // Search in Qdrant
        SearchPoints searchPoints = SearchPoints.newBuilder()
            .setCollectionName(COLLECTION_NAME)
            .addAllVector(queryEmbedding.stream().map(Double::floatValue).toList())
            .setLimit(limit)
            .setFilter(filterBuilder.build())
            .setWithPayload(WithPayloadSelector.newBuilder().setEnable(true))
            .build();

        SearchResponse response = client.searchAsync(searchPoints).get();
        return response.getResultList();
    }
}
```
