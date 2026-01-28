## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[06_Test_Time_Compute|Test-Time Compute]]
**Next:** [[08_Continued_Pretraining|Continued Pretraining]]

---

# Synthetic Data for Fine-Tuning

## Introduction: Data as a Bottleneck

By 2024, compute is getting cheaper, models are scaling, but high-quality data is becoming a scarce resource. According to Epoch AI estimates, high-quality text data on the internet will be exhausted by 2026-2028.

Synthetic data is the answer to this challenge. Models generate training examples for other (or the same) models.

Risks: model collapse, homogenization, amplification of biases.

## Self-Instruct: Instruction Generation

Self-Instruct (Wang et al., 2022) demonstrated that models can generate high-quality instruction-following data.

**Processing Pipeline:**
1. Seed tasks (175 examples)
2. Task generation — new task types
3. Instance generation — specific input-output pairs
4. Filtering — removing duplicates, low quality

**Results:** Alpaca — 52K examples for $500, quality comparable to text-davinci-003.

**Limitations:** quality ceiling (no higher than teacher model), diversity limits, bias propagation.

## Evol-Instruct: Complexity Evolution

Self-Instruct generates medium-difficulty examples. Evol-Instruct applies "mutations" to instructions, progressively increasing their complexity.

**Evolution Types:**
- Add constraints: "Write a poem" → "Write a sonnet in iambic pentameter"
- Deepen: "Explain ML" → "Explain mathematical foundations of gradient descent"
- Concretize: "Write code" → "Write Python function implementing merge sort O(n log n)"
- Increase reasoning: "Solve 2+2" → "Prove √2 is irrational"
- Complicate input: adding edge cases, ambiguity

**WizardLM Results:** 250K evolved instructions, significant improvement on complex tasks, outperforms Alpaca.

## Distillation: Knowledge from Teacher to Student

Knowledge distillation — transferring knowledge from a large model to a smaller one through generating high-quality responses.

**Types:**
- Response distillation — teacher generates responses, student reproduces them
- Rationale distillation — teacher generates reasoning chains
- Preference distillation — teacher ranks options

**Examples:** Orca (GPT-4 → 13B), Phi-2 (synthetic textbooks → 2.7B), OpenHermes (multi-teacher).

**Legal considerations:** many ToS prohibit using outputs to train competing models.

## Data Augmentation

**Paraphrasing** — generating variations with the same meaning but different wording.

**Back-translation** — intermediate translation to another language and back for natural variations.

**Seed expansion** — generating new questions based on existing pairs.

**Style transfer** — creating variations in different styles (formal, casual, technical).

## Quality Control for Synthetic Data

**Multi-layer filtering:**
1. Format validation — JSON parsing, required fields
2. Content filtering — harmful content, factual consistency
3. Semantic quality — relevance, coherence, completeness
4. Diversity check — no duplication of existing data

**LLM-as-Judge** — using a model for automated quality assessment of examples.

**Human-in-the-loop** — AI filter selects top 30%, humans review a 10% sample, calibrated filter is applied to the rest.

## Model Collapse: The Danger of Synthetic Loops

When models are trained predominantly on synthetic data from previous generations, degradation occurs.

**Collapse Mechanisms:**
- Tail truncation — rare patterns disappear
- Mode collapse — diversity decreases
- Error amplification — errors accumulate
- Hallucination propagation — hallucinations become "facts"

**Empirical results:** after 5-9 generations quality degradation occurs, diversity drops by 50%+, factual accuracy declines.

**Mitigation:**
1. Always mix with human data (70% human, 30% synthetic)
2. Fresh synthetic data from the original model, not fine-tuned
3. Diversity enforcement — explicit optimization
4. Quality gates — aggressive filtering
5. Tracking provenance — accounting for data origin

## Practical Pipelines

**Instruction Tuning:**
Seed tasks (200-500) → Self-Instruct expansion (10K) → Evol-Instruct (30K) → Quality filtering (50%) → Human review (5%) → Final dataset (15K).

**Domain Adaptation:**
Domain-specific documents → QA generation → Edge cases → Expert review → Mix with general data (70/30).

**Reasoning:**
Math/logic problems → Teacher step-by-step solutions → Verification (programmatic/expert) → Only verified correct → Add examples with errors and corrections.

## Key Takeaways

Synthetic data is a powerful tool when human data is scarce.

Self-Instruct and Evol-Instruct are the primary techniques for generating instruction data.

Distillation is effective but has legal constraints.

Quality control is critical — without it, synthetic data causes harm.

Model collapse is real — avoid multi-generation loops, always mix with human data.

Optimal mix: 60-80% human, 20-40% synthetic.

Verification where possible — for code/math, verify programmatically.

Diversity matters more than quantity — 10K diverse examples are better than 100K similar ones.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[06_Test_Time_Compute|Test-Time Compute]]
**Next:** [[08_Continued_Pretraining|Continued Pretraining]]
