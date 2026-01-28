# LLM Evaluation Metrics and Benchmarks

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Evaluation and Testing
**Previous:** [[../10_Fine_Tuning/10_DoRA_and_Beyond|DoRA and Advanced PEFT Methods]]
**Next:** [[02_Human_Evaluation|Human Evaluation]]

---

## The Fundamental Problem of LLM Evaluation

Evaluating language models differs from testing traditional software. For the question "Explain quantum entanglement," there exists an infinite set of correct answers of varying quality. The traditional unit-testing approach does not work — a model may produce an answer semantically equivalent to the reference but worded differently, or even better than the reference.

This requires sophisticated approaches: multidimensional metrics instead of binary checks, statistical methods, and a combination of automatic and human evaluations.

## Multidimensionality of Quality

Answer quality is a multidimensional vector that includes:

**Correctness** — factual accuracy of statements. For factual questions it is directly verifiable; for analytical ones — through logical validity of reasoning.

**Relevance** — alignment with the question. A correct answer can be irrelevant: "The capital of France is Paris" is correct but does not answer a question about population.

**Completeness** — coverage of all important aspects. An incomplete answer omits essential details.

**Conciseness** — absence of redundancy. Creates tension with completeness: a balance between sufficiency and verbosity.

**Coherence** — logical structure of the exposition. All facts may be present but organized chaotically.

**Fluency** — linguistic quality: grammar, syntax, style.

**Usefulness** — practical value for solving the user's task.

**Safety** — absence of harmful content, dangerous instructions, misinformation.

Different tasks require different emphases: for factual questions correctness is critical, for creative writing — coherence and fluency, for code — functional correctness.

## Automatic Text Metrics

Historically borrowed from machine translation and summarization, based on comparison with a reference.

**Exact Match** — full match after normalization. Useful for tasks with a single correct answer: classification, entity extraction.

**Token F1** — token overlap. Precision shows the proportion of answer tokens present in the reference. Recall — the proportion of reference tokens in the answer. F1 — the harmonic mean. Ignores word order.

**BLEU** — n-gram overlap between the candidate and reference. Uses a combination of unigrams, bigrams, trigrams, and 4-grams. Brevity penalty penalizes short answers. The metric is computed as the product of the brevity penalty and the exponential of the weighted average of log-precisions for different n-grams. Clipped counting prevents inflation through repetitions — each n-gram is counted no more often than it appears in the reference.

**ROUGE** — a family of metrics for summarization, oriented toward recall. ROUGE-N measures n-gram overlap. ROUGE-L uses the longest common subsequence — the longest shared subsequence that preserves order.

### Fundamental Limitations of Text Metrics

**Synonymy** — "fast car" and "speedy automobile" are semantically equivalent, but n-gram metrics detect no overlap.

**Single reference** — in reality there are many correct answers; a model may generate a correct answer absent from the reference.

**Order is not always important** — a list of items can be enumerated in any order; sequence-based metrics penalize this.

**Imperfect correlation** — a high BLEU does not guarantee quality; a low BLEU does not mean a poor answer.

## Semantic Metrics

Overcome the limitations of text metrics through vector representations.

**Embedding similarity** — texts are converted into vectors via models such as Sentence-BERT, and cosine similarity is measured. Semantically close texts have similar embeddings regardless of the specific words used.

**BERTScore** — compares contextualized embeddings of individual tokens. For each candidate token the most similar reference token is found and vice versa. Precision, recall, and F1 are computed based on these similarities. It uses contextualized embeddings, so the word "bank" receives different vectors in "river bank" and "investment bank," distinguishing homonyms. IDF weighting is optionally applied to reduce the weight of frequent function words.

**MoverScore** — uses Word Mover's Distance from optimal transport theory. Measures the minimum "work" required to transform one set of embeddings into another. Formulated as a linear programming problem with cubic complexity O(n³ log n).

**Advantages:** robustness to paraphrases. **Limitations:** high computational cost, dependence on the quality of the embedding model.

## Benchmarks

Standardized task sets for comparing models.

**GLUE and SuperGLUE** — classic language understanding benchmarks: sentiment analysis, NLI, QA. Modern models have reached saturation.

**MMLU** — 57 subject areas from elementary mathematics to professional law. Tests knowledge and reasoning.

**HellaSwag** — commonsense reasoning. Selecting the most likely continuation of a situation.

**HumanEval** — 164 programming challenges with unit tests. The pass@k metric indicates whether at least one of k generated candidates passes all tests.

**MT-Bench** — multi-turn dialogues with evaluation via GPT-4 as a judge.

## Choosing Metrics

**Understanding the task** — what matters to users? Factual accuracy, phrasing quality, format?

**Multiple metrics** — a single metric can be optimized at the expense of other aspects. A set provides the full picture.

**Calibration against human judgments** — periodically compare automatic scores with human evaluations.

**Cost considerations** — simple metrics for fast feedback in CI, expensive methods for in-depth analysis.

**Goodhart's law** — "when a measure becomes a target, it ceases to be a good measure." Optimizing for a metric may not improve actual quality.

## Building an Evaluation Dataset

**Representativeness** — the dataset reflects the real distribution of queries. If 50% of traffic consists of product return questions, the dataset should contain a proportional share of such questions.

**Diversity** — different phrasings, difficulty levels, edge cases within categories.

**High-quality references** — reference answers must be genuinely good. A poor reference makes good answers appear "incorrect."

**Sufficient size** — for rough estimation 100 examples suffice; for fine-grained comparisons thousands are needed.

**Versioning** — maintain versions, document changes, ensure comparability.

## Implementing Metrics

Practical implementation includes a service class that encapsulates various metrics:

**Exact Match** returns 1.0 on full match of normalized texts, 0.0 otherwise. Normalization: lowercase, removal of punctuation and extra whitespace.

**Token F1** operates on token sets: both texts are tokenized, the intersection is found, precision and recall are computed, then F1. Edge cases: empty texts on both sides yield 1.0; empty on one side — 0.0.

**BLEU** computes n-grams for each n from 1 to 4, applies clipped counting for each n-gram, takes the geometric mean of precisions, and multiplies by the brevity penalty (equal to 1.0 if the prediction is longer, otherwise exp(1 - len_ref/len_pred)).

**ROUGE-L** uses dynamic programming to find the LCS with an (m+1)x(n+1) matrix. Cell dp[i][j] contains the LCS length for the first i tokens of the first text and j tokens of the second. Filling: if tokens match — dp[i][j] = dp[i-1][j-1] + 1, otherwise dp[i][j] = max(dp[i-1][j], dp[i][j-1]).

**Semantic similarity** converts both texts into vectors via an embedding model and computes cosine similarity.

**Simplified BERTScore** splits texts into sentences, each converted into an embedding. Precision — the average over prediction sentences of their maximum similarity with reference sentences. Recall — symmetric.

**Faithfulness score** for RAG: each sentence of the generation is compared with all source sentences, the maximum similarity is taken, and the final score is the average of these maximums.

The evaluation runner coordinates the process: it accepts a model, a test dataset, and a metric configuration. For each example it records the time, generates a response, computes the enabled metrics, and collects results. Parallel processing via parallelStream accelerates evaluation on large datasets. Aggregation includes metric averages, latency statistics (mean, median, percentiles), breakdown by category, and identification of worst-performing examples.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Evaluation and Testing
**Previous:** [[../10_Fine_Tuning/10_DoRA_and_Beyond|DoRA and Advanced PEFT Methods]]
**Next:** [[02_Human_Evaluation|Human Evaluation]]
