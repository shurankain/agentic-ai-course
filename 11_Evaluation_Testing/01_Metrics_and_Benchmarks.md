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

Standardized task sets for comparing models. The benchmark landscape has evolved significantly — many classic benchmarks have saturated, and new, harder benchmarks have emerged.

### Classic Benchmarks (Saturating)

**GLUE and SuperGLUE** — classic language understanding benchmarks: sentiment analysis, NLI, QA. Modern models have reached saturation — scores are at or above human performance.

**MMLU** — 57 subject areas from elementary mathematics to professional law. Tests knowledge and reasoning. Frontier models now score 88-90%, approaching saturation.

**HellaSwag** — commonsense reasoning. Selecting the most likely continuation of a situation. Modern models exceed 95%.

**HumanEval** — 164 programming challenges with unit tests. The pass@k metric indicates whether at least one of k generated candidates passes all tests. Now largely saturated: frontier models score 93-97% (o3 96.7%, Claude Sonnet 4 93.7%, GPT-4o 90.2%). The industry has shifted to SWE-bench as the more discriminating coding benchmark.

**MT-Bench** — multi-turn dialogues with evaluation via GPT-4 as a judge.

### Modern Benchmarks (2024-2025)

As classic benchmarks saturate, the community has shifted to harder, more discriminating evaluations:

**MMLU-Pro** — a harder successor to MMLU. 12,032 questions across 14 disciplines with 10 answer choices (instead of MMLU's 4), requiring deeper reasoning. Questions are filtered for difficulty — easy questions are removed. Originally much more discriminating than MMLU, but frontier models with reasoning now score 85-90% (top: Gemini 3 Pro Preview 89.8%), approaching saturation. Chain-of-thought reasoning provides larger benefits on MMLU-Pro than on MMLU.

**GPQA Diamond** — Graduate-level Professional Quality Assurance. 198 extremely difficult questions written by domain experts (PhDs) in physics, chemistry, and biology. Questions are validated to be answerable by experts but not by skilled non-experts. Frontier model scores have risen rapidly: latest reasoning models reach 90%+ (Claude Opus 4.6: 91.3%, Gemini 3.1 Pro: 94.1%), up from o3's 87.7% in early 2025. GPT-4o 53.6%. Approaching saturation for the strongest reasoning models.

**SWE-bench Verified** — evaluates autonomous coding agent capability on real GitHub issues from popular open-source projects. Each task is a real bug report or feature request with a test suite. The model must understand the codebase, localize the issue, write a fix, and pass all tests. Scores have improved rapidly: frontier agents approach 80% (top: 79.2% Claude Opus 4.6 Thinking, up from ~72% in mid-2025), nearing estimated human performance of 75-90%. The most practically relevant coding benchmark — measures end-to-end software engineering, not just code generation.

**AIME (American Invitational Mathematics Examination)** — competition-level math problems requiring multi-step reasoning, creative problem-solving, and mathematical insight. Much harder than GSM8K or MATH. Scores on AIME 2024: o3 96.7%, o4-mini 93.4%, o1 83.3%, GPT-4o 13.4%. Note: AIME 2024 scores may be inflated due to data contamination in pretraining; AIME 2025 is considered a cleaner benchmark. Demonstrates the dramatic impact of test-time compute on mathematical reasoning.

**ARC-AGI (Abstraction and Reasoning Corpus)** — visual pattern recognition and abstract reasoning tasks. Each task provides a few input-output grid examples, and the model must infer the transformation rule and apply it to a new input. Designed to test genuine generalization ability, not pattern matching on training data. The original ARC-AGI-1 is largely solved (o3 87.5% with high compute). The harder **ARC-AGI-2** (introduced 2025) is now the frontier benchmark, with top scores reaching 84.6% (Gemini 3 Deep Think) and code-evolution approaches achieving 95.1% (Imbue) as of early 2026, surpassing the human baseline. A key benchmark for measuring progress toward general reasoning.

**Chatbot Arena (LMSYS)** — human preference evaluation at scale. Users submit prompts and compare responses from two anonymous models side-by-side, voting for the better response. Produces Elo ratings from hundreds of thousands of pairwise comparisons. As of early 2026, Claude Opus 4.6 leads the overall rankings, with the top 10 (including Chinese models like GLM-5 and Kimi K2.5) separated by only ~44 Elo points. Open-source models have nearly closed the gap with proprietary ones. The most ecologically valid benchmark — measures what users actually prefer in practice. Categories include coding, math, hard prompts, creative writing, and instruction following.

### Agent-Specific Evaluation Benchmarks

Standard benchmarks measure model capabilities in isolation. Agent benchmarks evaluate the full loop: planning, tool use, multi-step execution, and error recovery.

**GAIA (General AI Assistants)** — a benchmark from Meta requiring agents to answer questions that demand multi-step web browsing, tool use, file manipulation, and reasoning. Questions have unambiguous correct answers but require 5-20 steps to solve. Measures real-world assistant capability rather than isolated skills.

**TAU-bench (Tool-Augmented Understanding)** — evaluates agent performance on tasks requiring correct tool selection and sequencing. Focuses on the tool-use pipeline: understanding when to call tools, interpreting results, and chaining multiple tool calls to reach a goal. Includes retail and airline customer service domains.

**AgentBench** — a comprehensive suite covering 8 environments (OS, database, web browsing, knowledge graph, etc.). Agents must interact with real systems through actions. Measures robustness across diverse environments.

**ToolBench** — evaluates tool use at scale with 16,000+ real-world APIs. Tests API discovery, parameter filling, multi-API orchestration, and error handling. Measures whether agents can navigate large API ecosystems.

**Agent trajectory evaluation** — unlike outcome-only benchmarks, trajectory evaluation assesses the quality of the action sequence: Were the steps efficient? Was backtracking excessive? Were tools used appropriately? This is critical because two agents may reach the same answer through very different paths — one elegant, one wasteful.

**Evaluation frameworks:** UK AISI Inspect AI provides a government-backed framework for agent safety evaluation. Braintrust offers agent evaluation with trace-level scoring. These complement custom eval suites for production use.

### Benchmark Selection for Production

| What You're Evaluating | Recommended Benchmarks |
|------------------------|----------------------|
| General knowledge & reasoning | MMLU-Pro, GPQA Diamond |
| Mathematical reasoning | AIME, MATH |
| Coding agents | SWE-bench Verified, HumanEval |
| Tool-using agents | GAIA, TAU-bench, AgentBench |
| Abstract reasoning | ARC-AGI |
| User preference / overall quality | Chatbot Arena |
| Your specific use case | Custom eval set (always the most important) |

**Key principle:** Public benchmarks are useful for model selection and comparison, but a custom evaluation set on your actual use case is always the most important metric. Models can be optimized for public benchmarks (Goodhart's Law), and benchmark performance may not correlate with your specific workload.

### Benchmark Reliability and Gaming

Benchmarks are treated as objective truth, but several systematic issues undermine this assumption — especially for the most-cited coding benchmark:

**SWE-bench Verified has known reliability problems.** The SWE-Bench+ paper documented: 32.67% of patches contain **solution leakage** (the answer is already present in the issue description — the model can solve the task without understanding the code), 31.08% have suspiciously weak tests (the test suite fails to catch incorrect solutions), and 94% of tasks were created before model training cutoffs (models may have seen the issues and their solutions during pretraining).

**The gap between reported and real-world performance is large.** When the SWE-Bench+ authors filtered out problematic issues, model performance dropped from 3.97% to 0.55%. A practical estimate: **80% on SWE-bench Verified corresponds to approximately 30-40% on genuinely unseen, uncontaminated tasks.** This does not mean the models are weak — it means the benchmark overstates their capability on novel problems.

**Scaffolding matters more than model.** The same Claude Opus 4.5 model scores 17 tasks differently on SWE-bench Verified depending solely on the agent harness wrapping it (Augment, Cursor, Claude Code — same model, different frameworks). This means a "better SWE-bench score" may reflect better scaffolding, not a better model. See [[../../02_Prompt_Engineering/05_Context_Engineering|Context Engineering]] for why context harness determines more than model quality.

**Lesson for practitioners:** Use benchmarks for **relative comparison** (model A vs model B on the same benchmark), not for absolute capability claims ("this model can solve 80% of real bugs"). Always validate benchmark results with a custom evaluation set on your actual workload. And when evaluating coding agents, test with issues from your own repositories — not public benchmarks that may be contaminated.

### Cost per Correct Answer

A practical metric increasingly used in production evaluations: instead of measuring accuracy alone, measure **cost per correct answer.**

| Model | Accuracy | Cost per Query | Cost per Correct Answer |
|-------|----------|---------------|------------------------|
| Model A | 70% | $0.03 | $0.043 |
| Model B | 75% | $2.50 | $3.33 |

Model A is **77x cheaper per correct answer** despite being 5 percentage points less accurate. For production systems processing thousands of queries per day, this metric matters more than raw accuracy — a 5% accuracy improvement rarely justifies a 77x cost increase.

**Evaluation cost itself matters:** LLM-as-Judge evaluation costs approximately $0.06 per sample (15 seconds). Agent-as-Judge (where an agent evaluates another agent's work through multi-step verification) costs approximately $0.96 per sample (15 minutes). Budget evaluation costs into the system design — evaluating every response is often too expensive.

### What Benchmarks Don't Measure

Even reliable benchmarks capture only a fraction of what matters in production:

- **Error recovery** — how the agent handles failures, retries, and unexpected states
- **Cost efficiency** — token consumption, API call count, cache utilization
- **User satisfaction** — whether users find the agent helpful, not just technically correct
- **Time-to-completion** — wall-clock time including tool calls, retries, and thinking
- **Reliability/consistency** — whether the same input produces the same quality output across runs
- **Composability** — whether the agent works well as part of a larger system

A critical principle for agent evaluation: **"Evaluate results, not paths."** Agents often find valid approaches that evaluation designers did not anticipate. A code fix that differs from the gold-standard patch but passes all tests is still correct. Penalizing it because the approach differs from the expected solution is a benchmark design failure, not an agent failure.

---

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
