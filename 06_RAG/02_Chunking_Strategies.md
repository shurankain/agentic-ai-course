# Document Chunking Strategies

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → RAG
**Previous:** [[01_RAG_Basics|RAG Basics]]
**Next:** [[03_Embeddings_and_Vector_Stores|Embeddings and Vector Stores]]

---

## Introduction to Chunking

Document chunking critically affects retrieval quality in a RAG system. Chunking is a balance between fragment size and context preservation.

Fragments that are too small lose context and semantic integrity. Fragments that are too large dilute relevance and inefficiently use the model's context window. The optimal size depends on the nature of the documents, the types of questions, and system capabilities.

**Why chunking is critical for RAG:** Chunking defines the unit of relevance. When a user asks "How do I configure authentication?", the system searches not the entire document but specific fragments. If a fragment is too small, it may not contain the complete answer. If too large, most of the information will be irrelevant, making the LLM's job harder.

Chunking quality affects three key aspects: retrieval accuracy (will the system find the exact fragment containing the answer), context completeness (will there be enough information for understanding), and generation efficiency (will the LLM be overwhelmed with irrelevant data).

## Fundamental Considerations

### Fragment Size and Context

Chunk size determines how much information each retrieval unit contains. Typical values range from 100 to 2000 tokens, but no universal optimum exists.

**Small fragments (100-300 tokens)** provide high retrieval accuracy. When a user asks a specific question, the system finds the exact paragraph containing the answer. This is ideal for fact-oriented questions. Advantages: high precision, minimal noise for the LLM, more independent fragments can be passed into context. Disadvantages: loss of context, pronouns and references lose their antecedents, term definitions separated from their usage, low recall for complex questions.

**Large fragments (1000-2000 tokens)** preserve context and logical coherence. They include topic introductions, explanations, examples, and transitions between ideas. Suitable for exploratory questions. Advantages: full context, logical coherence, fewer reference resolution issues, high recall. Disadvantages: low precision, occupies significant space in the LLM context window, harder for the model to extract a specific answer, risk of "drowning" important information in secondary details.

**Medium fragments (500-800 tokens)** are a practical trade-off for most scenarios. They typically correspond to one or two logical units: a documentation section, a paragraph with explanations, a function with comments. This is the optimum for technical documentation, scientific papers, business documents, and educational content.

### Theory of Optimal Chunk Size

From an information theory perspective, the optimal chunk size balances utility and cost. Utility is determined by fragment relevance and topic coverage completeness. Costs include noise and context processing overhead.

**Factors affecting optimal size:** Technical constraints — the embedding model has a maximum context length, the LLM context window determines how many chunks can be passed, computational cost of processing large fragments. Content characteristics — information density in technical documents requires smaller chunks, sparseness of fiction requires more context, document structure facilitates splitting into natural fragments. Usage patterns — question types (factoid vs. exploratory), precision vs. recall priority, update frequency.

**Empirical recommendations by content type:** FAQ and Q&A — 200-500 tokens, each question-answer pair is a self-contained unit. Technical documentation — 500-1000 tokens, a section with examples balancing completeness and focus. Scientific papers — 800-1500 tokens, a paragraph or several related paragraphs with logical coherence. Code — 100-500 tokens, a function or method with documentation and syntactic integrity. Legal documents — 500-1000 tokens, a clause or article in full with accuracy and completeness.

### Fragment Overlap

Overlap is a technique where adjacent fragments partially duplicate each other. If the last 100 tokens of one fragment are repeated at the beginning of the next, context is not lost at boundaries.

**Why overlap is necessary:** The boundary problem is the most common cause of relevant information loss. Imagine an important explanation starting at the end of one fragment and continuing in the next. Without overlap, the first chunk contains an incomplete explanation, the second chunk starts in the middle of a thought, both chunks receive low scores during retrieval, and relevant information is missed. With overlap, both fragments contain the complete explanation — at least one will pass the relevance threshold.

**Optimal overlap size:** The typical value is 10-20% of the fragment size. For a 500-token chunk, this means 50-100 tokens of overlap, roughly 1-2 sentences. Too little overlap fails to solve the boundary problem. Too much overlap creates excessive duplication and increases the index size without proportional benefit.

**Adaptive overlap:** Overlap size can vary depending on content. Technical texts with clear structure require less overlap, narrative texts with smooth transitions require more, lists and enumerations need minimal overlap, complex multi-step explanations need increased overlap.

### Boundaries and Semantic Integrity

The ideal chunk is a logically complete unit of text. A sentence should not be cut in the middle. A paragraph should not be split randomly. A definition should remain together with its illustrative example.

Naive splitting by character count ignores these considerations. More advanced strategies account for text structure: sentences, paragraphs, sections, chapters.

**Principles of semantic integrity:** Preserving atomic units of meaning (a sentence is the minimum complete thought, a paragraph develops a single idea, a section fully covers a subtopic). Linking definitions with examples (a term should be in the same chunk as its definition, an abstract explanation should be accompanied by a concrete example). Contextual dependencies (pronouns should be with their antecedents, references to "the above" should include what is being referenced).

## Chunking Strategies

### 1. Fixed-Size Chunking

The simplest strategy is splitting into fixed-length fragments. The document is divided into parts of N characters or tokens, optionally with overlap. The process is straightforward: take the first N tokens, shift by (N - overlap) positions, take the next N tokens, repeat until the end of the document.

**When to use:** Homogeneous documents without structure (transcripts, logs), baseline for experiments, when speed matters more than quality. **Advantages:** Simple implementation, predictability, fast processing, universality. **Disadvantages:** Breaks logical units, ignores structure, may separate definitions from examples. **Parameters:** 500-1000 tokens, overlap 10-15%.

### 2. Recursive Character Text Splitting

A more intelligent approach is splitting by natural text boundaries. The recursive variant attempts to split text first by large boundaries, then by medium ones, then by small ones.

The algorithm works as follows: attempt to split the document by large delimiters (sections), if fragments are still too large — split by medium delimiters (paragraphs), if still too large — by small ones (periods, commas), as a last resort — forced splitting by characters, after splitting add overlap between fragments.

**When to use:** Text documents with natural structure, books, articles, documentation. The standard choice for most scenarios. **Advantages:** Preserves natural boundaries, logical coherence, balance of simplicity and quality. **Disadvantages:** Variable sizes, does not account for semantics, depends on formatting. **Delimiter hierarchy:** sections → paragraphs → lines → sentences → words.

### 3. Semantic Chunking

Semantic chunking uses embeddings to determine fragment boundaries. The text is analyzed sentence by sentence, and where semantic similarity between adjacent sentences drops sharply, a boundary is drawn.

**Intuition:** If two consecutive sentences discuss different topics, that is a natural place to split. If they continue the same thought, they should stay together.

**Sliding Window Similarity algorithm:** Split the document into sentences, obtain embeddings for each sentence, compute cosine similarity between adjacent sentences, find points of sharp similarity drop (below threshold), draw chunk boundaries at these points, combine sentences between boundaries into chunks.

The system analyzes a sequence of sentences and computes cosine similarity between their embeddings. When several consecutive sentences discuss the same topic, their similarity remains high. As soon as a sentence about a new topic appears, similarity drops — this signals a chunk boundary.

**When to use:** Documents with complex thematic transitions, content without explicit structure, when semantic coherence matters. **Advantages:** Adapts to content, natural topic boundaries, independent of formatting. **Disadvantages:** Computationally expensive, slow for large documents, unpredictable chunk sizes, requires threshold tuning (typically 0.5-0.7).

### 4. Structure-Based Chunking

Structure-based chunking uses the explicit markup of a document. The strategy adapts to the format: an HTML document is split by tags, Markdown by headings, code by functions and classes.

**For Markdown:** Parse headings, each section under a heading is a potential chunk, preserve heading hierarchy as metadata, code blocks and tables remain intact, if a section is too large — split by subsections or paragraphs.

**For HTML:** section and article are large units, h1-h6 define hierarchy, p elements are paragraphs as minimum units, pre, code, table remain indivisible.

**For code:** Parsing via AST, a function/method is one unit, a class may be split into methods, docstrings stay with the code.

**When to use:** Technical documentation (Markdown, HTML), API documentation, codebases, any content with explicit markup. **Advantages:** Preserves logical organization, structural metadata useful for retrieval, syntactic correctness for code. **Disadvantages:** Depends on markup quality, different formats require different parsers, variable sizes.

### 5. Agentic Chunking (LLM-based)

The most advanced approach uses a language model for intelligent chunking. The LLM analyzes the document and determines logical units and their boundaries.

Process: pass the document to the LLM with the prompt "Split this text into logically coherent fragments", the LLM returns boundaries or the chunks themselves, optionally the LLM creates a summary for each chunk.

**When to use:** Small volumes of critically important documents, complex structure, when quality matters more than cost. **Advantages:** Highest quality — contextual understanding, non-trivial decisions (separating digressions, grouping related items), adapts to any content. **Disadvantages:** Expensive, slow, unstable, context window limitation, requires prompt engineering.

## Document Type Specifics

### Technical Documentation

Recommended strategy: Structure-based chunking by level 2-3 headings. Code blocks should remain intact, code without explanatory text is uninformative, tables and lists are logically connected units, usage examples should accompany API descriptions, warnings and important notes should stay with the context they relate to. Metadata to preserve: full heading path, documentation version, programming language, tags. Typical size: 500-1000 tokens (one subsection with examples).

### Legal Documents

Recommended strategy: Structure-based chunking by clauses and articles. A contract clause with all its subclauses is a single indivisible unit, a term definition must accompany the term itself, references to other parts of the document must preserve sufficient context, exceptions and caveats must not be separated from the main rule, numbering is critical for citation. Metadata: article/clause number, hierarchy, document type, effective date. Typical size: 500-1000 tokens. Overlap is less critical — the structure already provides context.

### Scientific Papers

Recommended strategy: Structure-based chunking by standard sections plus semantic chunking within long sections. Abstract is a separate chunk, Introduction can be split by paragraphs, Methods section benefits from larger chunks to preserve sequence, Results — tables and figures with descriptions stay together, Discussion — can be split by discussed aspects. Metadata: section, authors, year, journal, DOI, keywords, figure/table numbers. Typical size: 800-1500 tokens. Metadata application: filtering by sections.

### Correspondence and Chats

Recommended strategy: Grouping by thematic threads or time windows. A single message is rarely self-contained — conversation context is needed, a reply references a question — they should stay together, long threads can be split by topic changes. Grouping options: Thread (question + all replies), sliding window (5-10 messages), semantic clustering, by day for regular correspondence. Metadata: message author, timestamps, Thread ID, channel/topic. Typical size: 300-800 tokens.

### Code

Recommended strategy: AST-based chunking (splitting by syntactic units). A function/method with its docstring is the optimal unit, an entire class if small, class split by methods if large. Docstrings and comments stay with the code, the import section can be included in each chunk's context. The ideal code chunk includes a docstring with description, the function signature with types, the function body, and a usage example. Metadata: programming language, file path, function/class name, parameters and return type, tags from comments. Typical size: 100-500 tokens (function) or 500-1000 tokens (class). Summary chunks can be created — natural language descriptions of what the code does.

## Advanced Optimization Techniques

### Empirical Parameter Tuning

There are no universally optimal chunking parameters. The optimum depends on your specific use case.

**Tuning methodology:** Creating a test set (20-50 representative questions with known answers, diverse question types, coverage of different topics). Defining parameters for experiments (chunk size: 256, 512, 1024, 2048 tokens; overlap: 0%, 10%, 20%, 30%; strategy: fixed, recursive, semantic, structure). Evaluation metrics (Retrieval accuracy, MRR, Answer quality, Context efficiency). Analyzing results (which parameters work better for which question types, where the system makes errors, which documents are chunked poorly).

When testing the question "How to configure SSL in Nginx?", the fixed-size strategy with 512 tokens and 10% overlap returns a configuration code fragment without explanations or context. Structure-based chunking by headings with 800 tokens finds an entire "SSL Configuration" section with a full process description — confirming the advantage of structure-based chunking for technical documentation.

### Hybrid Approaches

Different parts of a knowledge base require different strategies. A single approach is a trade-off, optimal for nothing.

A hybrid approach works as a router that identifies the document type and applies the optimal strategy. Markdown files are processed with structural chunking at 800 tokens, HTML documents are split into 1000-token fragments, code is processed with AST-based chunking at 500 tokens, FAQ is divided into short 300-token chunks, PDFs are processed with recursive splitting at 15% overlap.

Advantages: each content type is processed optimally, code is not split in the middle of a function, question-answer pairs stay together, documentation preserves hierarchy. Implementation: determine type by MIME type or extension, map type to chunking strategy, fall back to a universal strategy for unknown types.

### Multi-Level Indexing (Hierarchical Chunking)

An advanced technique — creating multiple index levels with different granularity.

A document is indexed at several levels of detail simultaneously. The top level (Level 0) contains a brief description of the entire document at roughly 200 tokens. The first level (Level 1) includes large sections of about 1000 tokens. The second level (Level 2) consists of subsections of about 500 tokens. The third level (Level 3) contains paragraphs of about 200 tokens.

Retrieval works as follows: the first query searches Level 0-1 (large fragments), identifies the relevant document section, the second query searches within that section at Level 2-3, and retrieves the precise paragraph with section context.

Advantages: combines broad context and precision, reduces search space, can always move up a level for context, supports drill-down navigation. Disadvantages: implementation complexity, increased index size (3-4x), two retrieval stages are slower, requires parent-child relationship support in the database.

Practical applications: large technical documents (hundreds of pages), legal texts with complex hierarchy, scientific publications with detailed sections, cases where finding a precise answer while preserving context matters.

### Late Chunking (2024)

Late Chunking is a relatively new approach that inverts the traditional pipeline: first embed the entire document, then split.

**Traditional approach (Chunk-then-Embed):** Split the document into chunks, embed each chunk in isolation. Problem: the chunk "It became the largest company" does not know that "It" = "Apple".

**Late Chunking (Embed-then-Chunk):** Embed the entire document at once, obtain embeddings for each token (they have "seen" the entire document through attention), group token embeddings by chunks (pooling). Result: the chunk embedding preserves the context of the entire document.

Consider a document about Apple. With the traditional approach, the sentence "It became the largest technology company" ends up in a separate chunk and is embedded in isolation. The model does not know that the pronoun "It" refers to Apple, so the embedding is not associated with technology. With Late Chunking, the entire document is processed through the transformer at once. The tokens "It became" see the word "Apple" from the first sentence in their attention context, and their embeddings correctly reflect that the topic is a technology company.

**When Late Chunking is critical:** Documents with pronouns and references, technical texts with context-dependent terminology, academic texts with abbreviations, dialogues and correspondence with implicit context. **Limitations:** The document must fit within the embedding model's context window, requires a model with long context (8K+ tokens), computationally more expensive for very short documents. **Practical recommendation:** Late Chunking is most effective for medium-sized documents (2K-8K tokens) with high contextual dependency.

### Metadata Enrichment

Each chunk is not just text but a structured object with metadata that improves retrieval and context.

**Key metadata types:** Structural (heading hierarchy, position in document, content type, nesting level), Contextual (document and section name, creation/modification date, author, version, language), Semantic (automatic tags, chunk summary, named entities, category/topic).

**Metadata applications:** Search filtering ("What does it say about security in the configuration section?" → filter: section="Configuration"). Improving context for the LLM (instead of passing raw text, the model receives information about the section, document, and content type). Result ranking (boosting chunks from current documentation versions, prioritizing chunks from official sources).

**Summary Chunks:** An advanced technique — creating additional summary chunks for large sections. The LLM generates a brief (100-200 token) section description, the summary is indexed alongside regular chunks, useful for overview questions, contains references to detailed chunks for drill-down.

## Key Takeaways and Practical Recommendations

Chunking is a critical RAG system component that directly affects quality. Incorrect chunking can render even the best embedding model and generator useless.

**Core principles:** There is no universal solution, balance precision and context (small chunks for precision, large for context, medium as a trade-off), overlap is mandatory (10-20% overlap prevents information loss at boundaries), semantic integrity (chunk boundaries should align with logical text boundaries), metadata is critical (preserve hierarchy, content type, dates, authors).

**Strategy selection matrix:** Technical documentation — Structure-based (Markdown/HTML), 500-1000 tokens, by level 2-3 headings. FAQ, Q&A — Pair-based splitting, 200-500 tokens, question+answer as a unit. Scientific papers — Structure-based + semantic, 800-1500 tokens, by IMRAD sections. Legal documents — Structure-based by articles, 500-1000 tokens, clause with all subclauses. Code — AST-based, 100-500 tokens, function with docstring. Correspondence/chats — Thread-based, 300-800 tokens, grouping related messages. Arbitrary text — Recursive, 500-1000 tokens, paragraphs with 15% overlap.

**Optimization process:** Start with a baseline (recursive splitting, 512 tokens, 10% overlap), create a test set (20-50 representative questions), experiment (vary size, overlap, strategy), measure (Retrieval accuracy, MRR, answer quality), specialize (different strategies for different document types), iterate (chunking is continuous improvement).

**Advanced techniques for complex cases:** Hierarchical chunking for large documents with level-based navigation, Late chunking for texts with pronouns and contextual references, Summary chunks for overview questions about large sections, hybrid approaches with routing by document type to the optimal strategy.

**Common mistakes:** Using a single strategy for all content types, ignoring overlap, chunks that are too small or too large, breaking logical units, missing metadata.

Investing in quality chunking pays off through improvements across the entire RAG system. It is the foundation on which effective retrieval and generation are built.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → RAG
**Previous:** [[01_RAG_Basics|RAG Basics]]
**Next:** [[03_Embeddings_and_Vector_Stores|Embeddings and Vector Stores]]
