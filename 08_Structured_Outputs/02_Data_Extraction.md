# Data Extraction from Unstructured Text

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Structured Output
**Previous:** [[01_Structured_Output_Techniques|Structured Output Techniques]]
**Next:** [[03_Validation_and_Error_Handling|Validation and Error Handling]]

---

## Introduction: Turning Chaos into Order

Terabytes of unstructured data are created daily — emails, documents, reports, news, reviews, social media. Humans easily understand this text, but for automated processing it is virtually useless.

Before LLMs, data extraction required specialized systems. Named Entity Recognition (NER) took years of training on annotated corpora. Relationship extraction required complex processing pipelines. Each new entity type meant another round of annotation and training.

LLMs changed the game. A model trained on trillions of tokens already "knows" names, dates, companies, products. It understands context, distinguishes homonyms, and grasps nuances. Extraction that used to take months of ML team effort now takes hours.

However, reliable production use requires understanding patterns, limitations, and best practices.

## Types of Extraction Tasks

**Entity Extraction** — finding mentions of objects: people's names, organizations, locations, dates, amounts, products. LLMs understand semantics: "Apple introduced a new iPhone" contains an organization (Apple) and a product (iPhone), while "I ate an apple" does not. The model extracts even unknown categories: ask it to find spaceship names in a science fiction text — it handles it without special training.

**Relationship Extraction** — understanding connections between entities. "John works at Google" — EMPLOYED_BY between PERSON and ORGANIZATION. "Microsoft acquired LinkedIn" — ACQUIRED between ORGANIZATION. Relationships transform a list of entities into a knowledge graph — a coherent picture of the world for analytics and reasoning. LLMs easily handle complex sentences, passive constructions, and nested relationships.

**Attribute Extraction** — obtaining characteristics of an entity. Not just "iPhone mentioned," but "iPhone 15 Pro, price $1199, storage 256GB, color titanium." Attributes can be explicit (stated) or implied (requiring inference): "The latest flagship" implicitly indicates top-tier specifications.

**Document Structuring** — converting a document into a structure. A resume becomes an object with experience, education, and skills. A contract becomes an object with parties, terms, dates, and obligations. LLMs work as intelligent parsers that understand document structure, narrative logic, and connections between sections.

## Extraction Strategies

**Direct Extraction** — send text with a description of what is needed: "Extract all mentions of people and companies, return JSON." Works for simple cases: short text, clear categories, straightforward requirements. The model handles it in a single call. Problems arise with complexity: long text — loses focus, misses mentions; many categories — confuses classifications; complex instructions — fails to follow rules.

**Stepwise Extraction** — break the task into steps: find entities, classify by type, extract relationships, form the structure. Each step is simpler and more reliable, quality is controlled independently. The cost: more API calls, longer processing, more complex pipeline. Justified for critical applications.

**Extraction with Verification** — after initial extraction, the model checks for consistency, completeness, and correctness. Feedback: "You extracted that John is CEO of the company. But the text states he founded the company in 2020. Add the FOUNDED relationship." Catches errors, fills in gaps. Useful for complex documents.

**Ensemble Approach** — run extraction multiple times with different prompts or models, then merge results. A fact found by all runs is correct. A fact from a single run requires verification. Increases recall at the cost of precision; final filtering restores precision. Expensive but maximally reliable.

## Working with Context

Context is the key to high-quality extraction. An isolated sentence is ambiguous; in context, everything is clear.

**Anaphora and Coreference.** "The company introduced a new product. It became a bestseller." — What became a bestseller? "It" refers to "product." LLMs naturally handle coreference thanks to the attention mechanism — they "remember" what is being discussed and correctly resolve pronouns. It is important to provide sufficient context. Cutting out paragraphs and processing them separately is a bad idea — connections will be lost.

**Domain Context.** "The patient takes aspirin" — in medicine, "aspirin" is a MEDICATION; in everyday context, it is a PRODUCT. The domain determines interpretation. For domain-specific extraction, add context to the prompt: document type, relevant categories, terminology specifics.

**Temporal Context.** "Sales grew by 20% last quarter." — When exactly? Depends on the document date. For correct date extraction, provide a reference point: document creation date, event date, current date. The model will resolve relative time references.

## Handling Ambiguity

Real-world texts are full of ambiguities. A good system handles them explicitly rather than ignoring them.

**Multiple Interpretations.** "Apple released new chips" — semiconductor chips? snack chips? The Apple context suggests semiconductors. But if the context is ambiguous, the system returns options with confidence scores: "85% — PRODUCT, 10% — PERSON (model as advertising persona)."

**Incomplete Information.** "The meeting will take place next Tuesday" — the date is not specified. "The price is several million" — the exact amount is unknown. The system returns partial information with incompleteness flags or requests clarification in an interactive scenario.

**Contradictions.** In long documents: "At the beginning of the year, the staff numbered 100 people" and "The company had 150 employees" — which number is correct? The system flags contradictions, prioritizes more recent or reliable sources, and requests human resolution.

## Scaling Extraction

**Batching and Parallelism.** Processing documents one at a time is slow. Grouping into batches and sending them to the API in parallel speeds up the process significantly. Considerations: contexts from different documents must not mix, API rate limits require management, errors in one document must not halt the rest.

**Incremental Processing.** When new documents are added, there is no need to reprocess everything. The system processes only what is new, maintaining and updating the overall knowledge graph.

**Deduplication.** Different documents describe the same entities differently: "Apple Inc.," "Apple," "Apple corporation" — one entity. "Tim Cook," "Timothy D. Cook," "Apple CEO" — one person. Entity resolution is a separate task. LLMs help by comparing descriptions and determining object identity.

## Practical Example: Extraction with Validation

Architecture: an EntityExtractor interface with an extract method takes text and returns a list of entities. Each entity contains text, type (PERSON, ORGANIZATION, LOCATION, DATE, MONEY, PRODUCT), and confidence level.

The ValidatedExtractionService performs extraction in three stages:

1. **Initial Extraction** — the model receives contract text and returns structured data: effective date, expiration date, list of parties.

2. **Validation** — checking logical relationships. The start date cannot be later than the end date. Discrepancies are collected into an error list.

3. **Self-Correction** — the model receives feedback describing the problems along with the document text again. It analyzes the errors and performs re-extraction taking the feedback into account.

The key pattern is the feedback loop. Instead of simple repetition, the model receives a specific description of what went wrong. This significantly increases the likelihood of successful correction. The approach ensures reliability for critical documents where accuracy of every field matters.

## Key Takeaways

Data extraction is turning chaos into order, prose into data structures.

Task types: entities (find objects), relationships (link objects), attributes (object characteristics), document structuring (full transformation).

LLMs radically simplify extraction through semantic understanding. They grasp meaning, resolve ambiguities, and work with new categories without fine-tuning.

Strategies range from direct approaches to complex pipelines with verification and ensembles. The choice depends on task complexity and reliability requirements.

Context is critically important: anaphora, domain, time. Provide the model with enough information for correct interpretation.

Ambiguities are inevitable. A good system handles them explicitly: returns options with confidence scores, flags incompleteness, detects contradictions.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Structured Output
**Previous:** [[01_Structured_Output_Techniques|Structured Output Techniques]]
**Next:** [[03_Validation_and_Error_Handling|Validation and Error Handling]]

---
