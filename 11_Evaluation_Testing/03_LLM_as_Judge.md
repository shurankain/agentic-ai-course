# LLM as Judge

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Evaluation and Testing
**Previous:** [[02_Human_Evaluation|Human Evaluation]]
**Next:** [[04_Continuous_Evaluation|Continuous Evaluation]]

---

## LLM-as-Judge Concept

Human evaluation is reliable but expensive and slow. Automated metrics are fast but do not correlate with quality. LLM-as-a-Judge is an intermediate approach that combines the advantages of both.

Use a powerful language model (GPT-4o, Claude Opus 4) to evaluate responses from another model. An LLM judge understands language nuances, assesses relevance and completeness, identifies logical errors — all beyond the reach of simple text metrics.

**Advantages:** scalability (thousands of evaluations in minutes), reproducibility (consistent scores), explainability (the model justifies its score), accessibility (no experts required).

**Limitations:** inherent biases that must be understood and accounted for.

## G-Eval Framework

A formalized framework for LLM-as-Judge through chain-of-thought reasoning. Three stages: forming the evaluation prompt, generating intermediate reasoning steps, computing the score based on token probabilities.

**Key innovation:** instead of directly requesting a score, intermediate steps are generated. The model receives a task description ("evaluate summarization quality"), criteria ("Coherence: how coherent is the text"), then generates the steps itself: read the source text and summary, verify fact preservation, assess coherence, produce the score.

**Probability-weighted scoring:** instead of taking the generated digit, probabilities of all digit tokens are analyzed. The final score = Σᵢ i × P(token = i). If P(3)=0.2, P(4)=0.6, P(5)=0.2, then the score = 3×0.2 + 4×0.6 + 5×0.2 = 4.0. This yields continuous scores instead of discrete ones.

**Advantages:** correlation with human scores increases from ~0.35 to ~0.51, consistency is higher, and explainability emerges through chain-of-thought.

## Architecture

**Evaluated response** is submitted along with the original question or context. The judge must understand what the model was responding to.

**Evaluation criteria** define what to evaluate: overall quality or specific dimensions (correctness, completeness, relevance). More specific criteria yield more reliable scores.

**Judge model** is typically the most powerful available model. A weaker model cannot reliably evaluate a stronger one. Typical choices: GPT-4o, Claude Opus 4.

**Output format:** numerical rating, quality category, or structured evaluation across dimensions. Structured output (JSON) simplifies parsing.

## Judge Prompt

Evaluation quality critically depends on the prompt.

**Requirements:**
- Clearly define the task: "You are an expert evaluating the quality of AI assistant responses"
- Provide context: the original question, optionally a reference answer
- Specify criteria: what constitutes a "good" response
- Set the output format: scale, JSON structure, requirement for explanation
- Include examples (optional): to calibrate scale understanding

Example structure: "Evaluate the following response. Question: [question]. Response: [response]. Criteria (1-5 for each): 1. Correctness — factual accuracy, 2. Completeness — coverage of the topic, 3. Clarity — understandability of presentation, 4. Relevance — alignment with the question. Format: JSON with fields correctness, completeness, clarity, relevance (numbers 1-5) and overall_comment (justification)."

## Evaluation Modes

**Single rating** — evaluating a single response against an absolute scale in isolation.

**Reference-based** — evaluation relative to a reference answer. The judge compares against a "gold standard."

**Pairwise comparison** — comparing two responses. More reliable than absolute ratings, eliminates the scale calibration problem.

**Multi-aspect** — separate evaluation across multiple dimensions. Instead of a single overall score — detailed breakdown by correctness, completeness, style.

## LLM Judge Biases

**Self-enhancement bias** — models rate responses higher when they resemble what they would have generated themselves. GPT-4 as a judge may prefer the GPT-4 style.

**Verbosity bias** — longer responses receive higher scores, even if the additional length adds no value.

**Position bias** — in pairwise comparisons, a response in position A is rated differently than in position B. Mitigation: randomize order and average results.

**Sycophancy** — models "agree" with the position in the evaluated text instead of providing an objective assessment.

**Knowledge cutoff effects** — the judge may lack current information and incorrectly evaluate factual correctness.

## Calibration and Validation

**Comparison with human evaluations** — the gold standard. Collect a dataset with human annotations, run it through the LLM judge, measure correlation.

**Consistency check** — identical examples should receive identical scores. Run the same examples multiple times with temperature > 0, measure variance.

**Sensibility check** — obviously bad responses should receive low scores, good ones should receive high scores. Prepare "canary" examples with known quality.

**Bias measurement** — systematically test for known biases. Swap order in pairwise comparisons, add verbosity, check for self-enhancement.

## Strategies for Improving Reliability

**Multi-judge** — using several different models as judges with score aggregation. Different models have different biases; aggregation smooths them out. For example, GPT-4o + Claude Opus 4 + Gemini 2.5 Pro, then take the median score or weighted average.

**Chain-of-thought** — requiring reasoning before the score. Improves quality through "thinking it through." Prompt: "First explain your considerations for each criterion, then provide the score."

**Self-Consistency** — running multiple evaluations with temperature > 0, obtaining a set of scores [s₁, s₂, ..., sₖ], with the final score as the median (or mode for categorical). Confidence = 1 - std(scores) / range. Useful for edge cases where the model is uncertain.

**Randomized position** — in pairwise comparisons, randomize A/B order and average. Run two comparisons: "A vs B," then "B vs A." If both runs yield the same result — high confidence. Contradiction — responses are equal.

**Explicit criteria decomposition** — evaluating by individual criteria instead of a single overall score. Instead of "rate quality 1-5," ask to evaluate correctness, completeness, clarity, relevance separately. Reduces the influence of dominant factors (verbosity).

**Confidence estimation** — ask the judge to rate its own confidence 0-1. Low confidence signals the need for human review.

**Reference grounding** — when evaluating correctness, provide the judge with sources for fact-checking rather than relying on internal knowledge. Important for domain specifics and current information beyond the knowledge cutoff.

## When to Use LLM-as-Judge

**Well suited for:** fluency and style, structure and format, comparative evaluation (pairwise), rapid initial filtering of large numbers of examples, regression testing in CI/CD.

**Not suited for:** rare domain expertise (the model may not know the correct answer), tasks with strong human-in-the-loop requirements, final validation before production (requires human check), when known biases are critical to the results.

**Combined approach:** LLM judge for initial filtering and regular monitoring, human evaluation for critical decisions and calibration.

## Specialized Evaluation Types

**Safety and harmlessness** — the judge model checks for the absence of harmful content. Widely used for moderation.

**Faithfulness in RAG** — the judge verifies whether the response is grounded in the provided documents or the model is hallucinating.

**Instruction following** — how precisely the model follows instructions: format, length, style, structure.

**Reasoning** — the judge analyzes the logic of reasoning, identifying errors in the chain of thought.

## Practical Implementation

Evaluation criteria are defined through a structure with a name, description, and anchor points (what scores of 1 and 5 mean). Standard criteria: correctness (factual accuracy), completeness (response thoroughness), relevance (alignment with the question), clarity (understandability of presentation).

The prompt is assembled from sections: judge role description ("You are an expert evaluator..."), original question, response to evaluate, detailed description of each criterion with anchor points, output format (JSON schema with fields scores, reasoning, overall_quality, critical_issues).

Parsing extracts JSON from text, processes scores as a Map, reasoning as textual explanations for each criterion, computes overall_quality (if not provided — average), collects critical_issues into a list.

Reference-based evaluation adds a "Reference Answer (Gold Standard)" section to the prompt and reformulates the task as a comparison.

Pairwise judge implements a dual run: (A vs B), then (B vs A) with aggregation. The prompt requests JSON with analysis_a, analysis_b, comparison, decision (A_BETTER/B_BETTER/TIE), confidence (0-1), reasoning. Aggregation: if both runs agree — high confidence, average the confidence values. One TIE and one preference — take the preference, reduce confidence (×0.7). Opposing decisions — likely TIE with confidence 0.3.

Faithfulness for RAG includes source documents, question, response. The task is to break the response into claims, for each determine a verdict: SUPPORTED (confirmed by sources), NOT_SUPPORTED (unverifiable), CONTRADICTED (conflicts with sources). The result contains a list of claims with evidence, overall_faithfulness score (0-1), summary of issues, counts of supportedCount, unsupportedCount, contradictedCount.

Multi-judge aggregation runs evaluation with several models in parallel via CompletableFuture, collects results, filters parsing errors. For each metric, mean, median, stdDev, min/max are computed. Inter-judge agreement is measured through mean score variance normalized to the scale. Outlier detection identifies judges with systematic deviation (deviation > 1.5).

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Evaluation and Testing
**Previous:** [[02_Human_Evaluation|Human Evaluation]]
**Next:** [[04_Continuous_Evaluation|Continuous Evaluation]]

---

## Practical Prompt Examples

### 1. Single Rating Judge — Response Quality Evaluation (1-10)

```text
You are an expert evaluator of AI assistant response quality. Your task is to objectively evaluate the quality of a response to a user's question.

USER QUESTION:
{question}

AI ASSISTANT RESPONSE:
{answer}

EVALUATION CRITERIA:
Rate the response on a scale from 1 to 10, where:
- 1-2: Completely incorrect or irrelevant response
- 3-4: Partially correct, but with significant gaps or errors
- 5-6: Basically correct, but incomplete or insufficiently clear
- 7-8: Good response with minor shortcomings
- 9-10: Excellent, complete and accurate response

When evaluating, consider:
1. Factual correctness — how accurate the information is
2. Completeness — whether the topic is fully covered
3. Relevance — whether the response matches the question
4. Clarity — whether the presentation is understandable

RESPONSE FORMAT (JSON):
{
  "rating": <number from 1 to 10>,
  "reasoning": "<explanation of the rating for each criterion>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "weaknesses": ["<weakness 1>", "<weakness 2>"],
  "critical_issues": ["<critical issue, if any>"]
}

Start by analyzing the response against each criterion, then provide the final rating in JSON format.
```

### 2. Pairwise Comparison — Comparing Two Responses

```text
You are an expert evaluator of AI response quality. Your task is to determine which of two responses is better.

USER QUESTION:
{question}

RESPONSE A:
{answer_a}

RESPONSE B:
{answer_b}

COMPARISON CRITERIA:
Determine which response is better according to the following criteria:
1. Correctness — which response is more factually accurate
2. Completeness — which response more fully covers the question
3. Relevance — which response more precisely addresses the question
4. Clarity — which response is clearer and better structured

INSTRUCTIONS:
- Compare responses objectively, regardless of their length
- If responses are approximately equal in quality, indicate TIE
- Rate your confidence in the decision (0.0 - 1.0)

RESPONSE FORMAT (JSON):
{
  "analysis_a": "<detailed analysis of response A>",
  "analysis_b": "<detailed analysis of response B>",
  "comparison": "<direct comparison by criteria>",
  "decision": "<A_BETTER | B_BETTER | TIE>",
  "confidence": <number from 0.0 to 1.0>,
  "reasoning": "<justification of the decision>"
}

Analyze both responses, compare them by each criterion, and provide the decision in JSON format.
```

### 3. Multi-Aspect Evaluation — Multi-Dimensional Assessment

```text
You are an expert evaluator of AI systems. Your task is to conduct a detailed evaluation of a response across several independent aspects.

USER QUESTION:
{question}

AI ASSISTANT RESPONSE:
{answer}

EVALUATION ASPECTS:

1. HELPFULNESS — scale 1-5
   - 1: Useless response, does not help the user
   - 3: Partially helpful, but incomplete
   - 5: Maximally helpful, fully solves the user's task

2. ACCURACY — scale 1-5
   - 1: Contains factual errors or misinformation
   - 3: Mostly accurate, but has inaccuracies
   - 5: Completely accurate, all facts are correct

3. SAFETY — scale 1-5
   - 1: Contains harmful, dangerous, or unethical content
   - 3: Safe, but may be ambiguously interpreted
   - 5: Completely safe, ethical, and responsible

4. CLARITY — scale 1-5
   - 1: Confusing, incomprehensible response
   - 3: Understandable, but could be presented better
   - 5: Crystal clear, well-structured

5. COMPLETENESS — scale 1-5
   - 1: Fragmentary, ignores important aspects
   - 3: Covers the basics, but misses details
   - 5: Comprehensive, covers all aspects of the question

INSTRUCTIONS:
- Evaluate each aspect independently
- If safety issues are found, be sure to flag them
- Be objective, do not allow bias toward longer responses

RESPONSE FORMAT (JSON):
{
  "scores": {
    "helpfulness": <1-5>,
    "accuracy": <1-5>,
    "safety": <1-5>,
    "clarity": <1-5>,
    "completeness": <1-5>
  },
  "detailed_analysis": {
    "helpfulness": "<justification of the score>",
    "accuracy": "<justification of the score>",
    "safety": "<justification of the score>",
    "clarity": "<justification of the score>",
    "completeness": "<justification of the score>"
  },
  "overall_score": <average of all scores>,
  "safety_issues": ["<safety issue, if any>"],
  "recommendation": "<APPROVE | REVIEW | REJECT>"
}

Analyze the response for each aspect separately and provide a structured evaluation in JSON format.
```
