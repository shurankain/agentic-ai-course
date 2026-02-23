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

## Synthetic Reasoning Data (2024-2025)

The training of reasoning models (o1, o3, DeepSeek R1, Qwen QwQ) created massive demand for a new category of synthetic data: multi-step reasoning traces with verifiable correctness.

### Reasoning Trace Generation

**Teacher-generated reasoning chains:** A strong reasoning model (o1, Claude with extended thinking) generates detailed step-by-step solutions. These traces are then used to train smaller models. DeepSeek used this approach: R1 (the full reasoning model) generates reasoning traces → distilled into R1-distill models (1.5B to 70B parameters). The distilled models achieve remarkable performance — R1-distill-Qwen-32B matches o1-mini on many benchmarks.

**Self-play and rejection sampling:** Generate many candidate solutions, keep only those that arrive at the correct answer. For math and code, correctness is verifiable programmatically. This produces a dataset of verified correct reasoning chains without human annotation. Scale: DeepSeek R1 training used millions of such verified examples.

### RL from Programmatic Verifiers

A paradigm shift in 2024-2025: instead of training reward models on human preferences, use **programmatic verifiers** as the reward signal.

**For code:** Execute the generated code against test cases. The reward is binary: all tests pass (1.0) or not (0.0). No reward model bias, no reward hacking — the signal is ground truth.

**For math:** Check the final answer against the known correct answer. Optionally, verify intermediate steps using symbolic math engines (SymPy, Lean proof assistant).

**For logic and reasoning:** Formalize the problem and check the solution programmatically. Constraint satisfaction problems, combinatorial optimization — the solution is verifiable even if generating it is hard.

**GRPO with programmatic rewards:** DeepSeek R1's training pipeline combines Group Relative Policy Optimization with programmatic verification. For each problem: generate K candidate solutions → verify each programmatically → compute group-relative rewards → update the policy. This replaces the entire RLHF pipeline (human preferences → reward model → PPO) with a simpler and more scalable loop (programmatic verification → GRPO).

**Implications for synthetic data pipelines:**

The new pipeline for reasoning data: problems (from curated datasets or generated) → candidate solutions (from the model) → programmatic verification → verified solutions as training data. This is a self-improving loop: better models generate better solutions, which produce better training data, which train even better models. The key constraint is the availability of verifiable problems — expanding the range of programmatically verifiable domains is an active research area.

## Key Takeaways

Synthetic data is a powerful tool when human data is scarce.

Self-Instruct and Evol-Instruct are the primary techniques for generating instruction data.

Distillation is effective but has legal constraints.

Quality control is critical — without it, synthetic data causes harm.

Model collapse is real — avoid multi-generation loops, always mix with human data.

Optimal mix: 60-80% human, 20-40% synthetic.

Verification where possible — for code/math, verify programmatically.

Diversity matters more than quantity — 10K diverse examples are better than 100K similar ones.

Synthetic reasoning data is a new frontier (2024-2025). Programmatic verifiers replace human annotators for code and math. Distillation from reasoning models (R1 → R1-distill) is remarkably effective. GRPO with programmatic rewards simplifies the RLHF pipeline.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[06_Test_Time_Compute|Test-Time Compute]]
**Next:** [[08_Continued_Pretraining|Continued Pretraining]]
