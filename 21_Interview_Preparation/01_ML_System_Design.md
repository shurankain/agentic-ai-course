# ML System Design for Interviews

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Interview Preparation
**Previous:** [[../20_Architecture_Research/03_Multimodal_Architectures|Multimodal Architectures]]
**Next:** [[02_Coding_Exercises|Coding Exercises]]

---

## Introduction

ML System Design is one of the most challenging stages of interviews for Senior/Staff+ positions. It combines understanding of distributed systems with deep knowledge of ML/AI specifics.

For positions at $300K+, they evaluate the ability to structure an ambiguous problem, make well-reasoned trade-off decisions, think about production concerns from day one, and communicate complex ideas simply.

---

## Framework for ML System Design

### RESHAPE Framework

**R - Requirements:** Clarify functional (data types, features, integrations) and non-functional requirements (latency, throughput, availability, reliability).

**E - Estimation:** Perform calculations to understand scale — data volumes, required storage, network bandwidth, required number of GPUs/CPUs. This justifies architectural decisions.

**S - Service Architecture:** Design a high-level architecture with component separation, define service boundaries, communication protocols, integration points.

**H - How ML Works:** Deep dive into ML specifics — model selection and justification, feature engineering, training pipeline, online versus offline inference.

**A - API Design:** Design clear interfaces between components, define contracts, data formats, API versioning.

**P - Production Concerns:** Address monitoring of quality and performance metrics, A/B testing, rollback strategies, observability.

**E - Edge Cases:** Analyze failure modes, adversarial inputs, degradation upon component failure.

### Timing (45-60 minutes)

Requirements five to seven minutes (scope clarification, constraints), Estimation three to five minutes (back-of-envelope calculations), High-Level Design ten to twelve minutes (components and data flow), ML Deep Dive fifteen to twenty minutes (models, features, training), API & Storage five to seven minutes (interfaces, data schemas), Production five to ten minutes (monitoring, deployment), Q&A five minutes (answering follow-ups).

---

## Case Study 1: Enterprise RAG System

### Problem Statement

Design a RAG system for a large enterprise with 10M documents, supporting 1000 QPS with sub-second latency.

### Requirements Clarification

Functional: what document types (PDF, Word, HTML, Confluence, Slack), is multi-turn conversation needed, is attribution required (links to sources), multi-language support.

Non-functional: latency P50 less than 500 milliseconds, P99 less than 2 seconds, throughput 1000 QPS, availability 99.9%, freshness new documents available within 1 hour.

Constraints: data cannot leave the region (GDPR), some documents are confidential — access control is needed.

### Estimation

With 10 million documents at an average size of 10 kilobytes after text extraction, we get 100 gigabytes of text data. Splitting into chunks of 500 tokens on average, we get approximately 5 chunks per document, totaling 50 million chunks.

Using a model with dimensionality of 1536 (e.g., text-embedding-3-large), each embedding occupies 6 kilobytes when stored in float32 format. The total embedding volume will be 300 gigabytes for all chunks.

At 1000 QPS with an average query of 500 input tokens plus 500 output tokens (response generation), the required LLM throughput is on the order of 1 million tokens per second. This is equivalent to approximately 4-8 GPUs for inference depending on the model.

### High-Level Architecture

The architecture is built on a multi-layer principle:

Input Layer: Load Balancer distributes load across API Gateway instances, which handles authentication, rate limiting, and request routing.

Request Processing Layer: multiple Query Service instances process user requests in parallel, enabling horizontal scaling.

Orchestration Layer: coordinates the entire request processing pipeline — from query understanding through relevant information retrieval to response generation. This layer manages the interaction logic between components.

Specialized Services: Vector Store (e.g., Pinecone) for semantic search over embeddings, Reranking Service with a cross-encoder model for improving result relevance, LLM Gateway (vLLM or API providers) for generating final responses.

### ML Components Deep Dive

#### 1. Document Processing Pipeline

The document processing pipeline consists of sequential stages:

Extraction: Apache Tika or similar tools are used to extract text from various formats (PDF, Word, HTML). Preserving the structural information of the document is important.

Chunking: semantic chunking is applied with awareness of document structure. Optimal chunk size is 500 tokens with an overlap of 50 tokens to preserve context at boundaries. For long documents, hierarchical chunking is used — creating both fine and coarse chunks for multi-level retrieval.

Embedding: each chunk is converted into a vector representation using an embedding model (e.g., text-embedding-3-large with dimensionality of 1536). These vectors represent the semantic meaning of the text.

Indexing: vectors are stored in a vector database with enriched metadata (title, section, date, author, access control labels). This allows filtering results based on metadata and access permissions.

#### 2. Retrieval Pipeline

Multi-stage process for extracting relevant information:

Query Understanding: the first step is query expansion using techniques such as query expansion. HyDE (Hypothetical Document Embeddings) generates a hypothetical answer and uses it for search. Multi-query generates several query variations for broader coverage.

Initial Retrieval: parallel search is performed using two methods — semantic search through the vector store over embeddings (top-50 results) and keyword search through BM25 for exact term matching (also top-50). It is critically important to apply access control filters at this stage, verifying user permissions on documents.

Fusion: reciprocal rank fusion is applied to combine results from semantic and keyword search. This algorithm effectively combines rankings from different sources, giving advantage to documents ranked highly in both lists.

Reranking: the top-20 candidates after fusion pass through a cross-encoder model, which evaluates the relevance of a (query, document) pair more accurately than a bi-encoder. Finally, the top-5 most relevant chunks are returned.

Trade-offs: hybrid search adds 15% to relevance with a 20 millisecond increase in latency compared to pure semantic, cross-encoder reranking improves relevance by 25% but adds 100 milliseconds of latency, HyDE query expansion provides a 10% improvement for vague queries at the cost of 150 milliseconds of additional time.

#### 3. Generation Pipeline

The final response generation process includes:

Prompt Construction: retrieved chunks are formatted into context while preserving metadata (source, date, author). Context window management is critically important — relevant chunks must fit within the model's limited context, prioritizing the most relevant ones and avoiding overflow.

LLM Generation: the model receives a system prompt, context, and the user's question. The prompt instructs the model to answer strictly based on the provided context, avoiding hallucinations, and to include citations in the format [1], [2] for source traceability.

Post-processing: extracting citation references from the model's response and matching them with original documents. Adding source metadata for display to the user. Validating that the response is indeed based on the provided context.

A typical enterprise RAG prompt includes clear instructions: "You are an enterprise assistant. Answer based ONLY on the provided context. If the answer is not in the context, say 'I don't have enough information.'" This is followed by a Context section with chunks_with_metadata, then Question with user_query, and a final directive: "Provide a clear concise answer with citations."

### Production Concerns

Monitoring: retrieval quality (MRR, NDCG on labeled queries), generation quality (faithfulness score via LLM-as-judge), latency (P50, P95, P99 per component), cost (tokens consumed per query).

Caching: semantic cache for similar queries, KV-cache for prefix caching, embedding cache for frequently accessed documents.

Access Control: document-level ACL stored in metadata, query-time filtering in vector store, audit logging of all requests.

### Follow-up Questions

"How do you update the index when documents change?" Incremental indexing with change detection. For critical documents, sync update; for the rest, batch processing every 15 minutes.

"How do you handle very long documents?" Hierarchical retrieval: first find relevant sections (coarse), then chunks within sections (fine).

"How do you combat hallucinations?" Multi-layer approach: chain-of-thought prompting, citation verification, confidence scoring, human review for low-confidence.

---

## Case Study 2: LLM Serving Infrastructure

### Problem Statement

Design a multi-tenant LLM serving infrastructure supporting multiple models, 10K QPS, with cost optimization.

### Requirements

Multi-model: GPT-4, Claude, Llama, custom fine-tuned. Multi-tenant: 100+ enterprise customers. Latency: streaming TTFT less than 500 milliseconds. Cost: minimize while meeting SLOs. Isolation: tenant data isolation.

### High-Level Architecture

Multi-layer architecture for multi-tenant LLM serving:

API Gateway (Kong or similar): the entry point provides tenant authentication, rate limiting by pricing tiers, request routing, and consumption metering for billing.

Router/Orchestrator: an intelligent layer that makes model selection decisions based on task type, tenant SLA, and current availability. Manages fallback chains when providers are unavailable and load balancing across instances.

Backend Provider Layer: External OpenAI Gateway (access to GPT models via API), External Anthropic Gateway (access to Claude models), Self-hosted vLLM Cluster (own GPUs for open-source models and fine-tuned versions), Self-hosted SGLang (alternative inference engine for optimizing specific use cases).

### Intelligent Router

Logic for intelligent request routing:

Tenant Preferences: the first level checks tenant configuration. If a client has a preferred model (e.g., due to compliance requirements or completed validation), the request is routed directly to it.

Task Classification: prompt analysis to determine the task type — simple Q&A, coding, complex reasoning, summarization, and so on. A lightweight classifier or rule-based approach based on keywords and query structure is used.

Cost-Quality Optimization: based on the task type, the optimal model is selected. For simple_qa, the most cost-effective models are used (gpt-4o-mini, claude-haiku-4.5), providing sufficient quality at minimal cost. For coding tasks, specialized models are chosen (claude-sonnet-4, gpt-4o) with high scores on code benchmarks. For complex_reasoning, the most powerful models are applied (claude-opus-4, o3), despite high cost.

Fallback Chain: if the primary model is unavailable or returns an error, the request is automatically rerouted to a fallback model, ensuring high service availability.

### Cost Optimization Strategies

Cascading Models: start with a cheap model; if confidence is low, use a more powerful model — savings of up to 60% while maintaining quality.

Caching: exact match cache with 100% hit at zero cost, semantic cache (embeddings plus similarity threshold), KV-cache sharing for common prefixes.

Batching: continuous batching for self-hosted, request coalescing for API providers.

Spot/Preemptible GPUs: 60-70% savings for self-hosted, graceful degradation upon preemption.

Production Concerns: rate limiting (per-tenant, per-model, global), quota management (token budgets, alerts), fallback (automatic failover between providers), observability (cost attribution, quality metrics per tenant).

---

## Case Study 3: Real-time Content Moderation

### Problem Statement

Design a content moderation system for a social platform with 100M daily posts, sub-100ms latency.

### Architecture

Multi-layer content moderation system:

Pre-filter: the first level uses regex patterns and blocklists for fast rejection of obvious violations. The cheapest stage with latency under 1 millisecond, filtering out approximately 10-15% of content.

ML Pipeline: content passing the pre-filter is processed by a multi-model ensemble — a combination of specialized classifiers for different violation types (hate speech, violence, sexual content, spam).

Policy Engine: combines ML model results with business rules and determines the final action — approve, reject, or escalate to human review.

Action: executing the decision — publishing content, blocking, or sending to a moderator for manual review.

### ML Pipeline

Fast Path (P99 less than 50 milliseconds): lightweight classifier (DistilBERT), 95% of traffic, catches obvious violations.

Slow Path (P99 less than 500 milliseconds): LLM-based analysis for edge cases, multi-aspect evaluation (hate, violence, sexual, spam).

Human Review: appeals, low-confidence decisions, new violation patterns.

Key Considerations: false positive cost (blocking legitimate content leads to user churn), false negative cost (missing violations leads to legal/reputation risk), latency versus quality (cascading with fallback to human review), adversarial robustness (obfuscation, Unicode tricks).

---

## Case Study 4: Recommendation System with LLM Reranking

### Problem Statement

Design a recommendation system for an e-commerce platform that uses LLMs for personalized reranking.

### Two-Tower + LLM Reranking Architecture

Three-stage funnel for recommendations:

Candidate Generation: the first stage uses efficient methods such as Two-Tower architecture or Collaborative Filtering to quickly select 100 relevant items from millions. This is a fast stage with 10-20 millisecond latency, operating on user and item embeddings.

ML Ranking: the second stage applies a gradient boosting model (XGBoost) with numerous features (user demographics, item features, contextual signals, interaction history) for precise ranking of the top-100 candidates. The top-20 with the highest predicted engagement are selected.

LLM Reranking: the final stage uses an LLM for personalized reranking of the top-20, taking into account subtle nuances of the user's context. The LLM receives a User Profile (recent purchases, browsing history, inferred intent), a list of Candidate Items with features, and an instruction to maximize user satisfaction considering relevance, diversity, novelty. The model returns the final top-10 with justification.

Trade-offs: LLM reranking adds 200-500 milliseconds of latency, cost of 0.001-0.01 per rerank, quality plus 5-15% CTR in A/B tests, when to use on high-value pages (homepage, checkout).

---

## Common Mistakes and Red Flags

### What NOT to Do

Jumping straight to a solution without clarifying requirements. Ignoring scale — a solution for 1000 QPS is not the same as a solution for 1 million QPS. Forgetting about costs especially for LLM-heavy systems. Over-engineering — start simple, add complexity as needed. Ignoring failure modes — what if the LLM is unavailable?

### What to Do

Think out loud — the interviewer wants to understand your thought process. Draw diagrams — visualization helps structure ideas. Quantify trade-offs — "this adds 100 milliseconds but improves quality by 20%". Ask clarifying questions — this demonstrates maturity. Consider alternatives — "we could do X but Y is better because..."

---

## Pre-Interview Checklist

- Know the core ML system design patterns
- Can perform back-of-envelope calculations
- Understand trade-offs: latency/throughput/cost/quality
- Know production concerns: monitoring, caching, fallbacks
- Can draw clear architecture diagrams
- Practiced 3-5 case studies end-to-end

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Interview Preparation
**Previous:** [[../20_Architecture_Research/03_Multimodal_Architectures|Multimodal Architectures]]
**Next:** [[02_Coding_Exercises|Coding Exercises]]
