## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[02_Data_Preparation|Data Preparation]]
**Next:** [[04_RLHF_and_Alignment|RLHF and Alignment]]

---

# Fine-Tuning Techniques and Model Evaluation

## Full Fine-Tuning vs Parameter-Efficient Methods

Historically, fine-tuning meant updating all model weights. For a model with billions of parameters, this required enormous computational resources.

**Parameter-Efficient Fine-Tuning (PEFT)** changed the situation. Instead of updating all weights, PEFT methods update only a small fraction of parameters — typically less than 1%. The remaining weights are "frozen." This dramatically reduces memory requirements and speeds up training.

A surprising fact: despite updating only a fraction of a percent of parameters, PEFT achieves quality very close to full fine-tuning on most tasks.

## LoRA: Low-Rank Adaptation

LoRA is the most popular PEFT method. Its idea is mathematically elegant and practically effective.

Instead of directly updating the weight matrix W of size 4096x4096, LoRA adds the product of two small matrices: ΔW = A × B, where A has size 4096×r, and B has size r×4096. The rank parameter r is typically chosen to be small: 8, 16, 32, or 64.

The number of parameters at rank 16 decreases from 16 million to 131 thousand — 125 times fewer.

In practice: fine-tuning a 7B model, which requires 100+ GB VRAM for full training, becomes feasible on a GPU with 16GB of memory.

## Choosing LoRA Hyperparameters

**Rank (r)** is the key parameter. A higher rank means greater capacity for adaptation. For simple tasks, r=8-16 is sufficient. For complex tasks, r=32-64 may be needed.

**Alpha** is a scaling factor. It is typically set equal to the rank or twice the rank.

**Target modules** determine which layers to adapt. Minimum: attention layers (query and value projections). Full adaptation: all attention and feed-forward layers.

**Dropout** helps against overfitting (0.05-0.1).

## QLoRA: Quantized Adaptation

QLoRA goes further in reducing memory requirements. The base model is stored in quantized format (4-bit), while LoRA adapters remain in full precision.

4-bit quantization reduces memory requirements for model storage by approximately 4-8x.

**NF4** is a specialized quantization format optimized for normally distributed neural network weights.

QLoRA enables fine-tuning 7B models on GPUs with 4-8GB of memory, and 70B models on consumer GPUs with 24GB.

## DoRA and GaLore

**DoRA** decomposes weights into magnitude and direction, applying LoRA only to the directional component. Result: +2.6 to +3.7% improvement over standard LoRA at the same rank.

**GaLore** projects gradients into a low-rank subspace instead of weights. Enables training all parameters with memory efficiency comparable to LoRA. Requires more memory than LoRA but less than full fine-tuning.

### PEFT Comparison Table

| Method | Memory | Quality | Training Speed | Best For |
|-------|--------|---------|----------------|----------|
| Full FT | 100% | Baseline | Slow | Maximum quality |
| LoRA | 5-15% | 95-98% | Fast | General PEFT |
| QLoRA | 3-8% | 93-97% | Fast | Memory constrained |
| DoRA | 5-15% | 97-99% | Medium | Low-rank quality boost |
| GaLore | 20-40% | 98-100% | Medium | Full FT alternative |

## OpenAI Fine-Tuning API

For production scenarios, managed services offer the simplest path. The OpenAI API allows fine-tuning GPT-4o-mini and GPT-4o without managing infrastructure.

Process: prepare data in JSONL → upload file via API → create fine-tuning job → monitor → obtain model ID.

**Hyperparameters:** number of epochs (typically 2-4), batch size, learning rate multiplier.

**Cost:** training cost + inference cost.

## Model Quality Evaluation

Without thorough evaluation, it is impossible to determine whether fine-tuning helped.

**Baseline comparison** — comparing against the base model on the same tasks.

**Held-out test set** — data the model has never seen.

**Multiple metrics** — accuracy, format compliance, semantic similarity, latency, cost.

**Regression testing** — verifying that the model has not degraded on other tasks.

## Types of Metrics

**Exact match accuracy** — the proportion of fully correct answers.

**Semantic similarity** — embedding models for comparing the meaning of answers.

**Format compliance** — valid JSON, required sections, structure.

**BLEU, ROUGE** — n-gram metrics for generation.

**Human evaluation** — the gold standard.

**LLM-as-judge** — using a powerful model (GPT-4) for evaluating answers.

## Iterative Improvement Process

Fine-tuning rarely produces a perfect result on the first attempt.

**First iteration** establishes a baseline and reveals obvious problems.

**Error analysis** — where does the model make mistakes? Are there patterns?

**Data improvement** — adding examples for problematic scenarios, removing contradictions, balancing.

**Hyperparameter experiments** — more epochs, different learning rate, different target modules.

**A/B testing** — gradual rollout with monitoring.

## Production Monitoring

**Quality metrics tracking** — regular quality assessment on samples of production traffic.

**Drift detection** — have the input queries changed?

**Cost monitoring** — fine-tuning can change inference costs.

**Feedback collection** — gathering user feedback.

**Rollback capability** — the ability to quickly revert to a previous version.

## Model Versioning and Management

**Model registry** — centralized storage of all versions with metadata.

**Experiment tracking** — recording all experiments with parameters and results.

**Data versioning** — versioning of training data.

**Reproducibility** — the ability to reproduce any model version.

## Key Takeaways

LoRA and QLoRA make fine-tuning accessible. Instead of specialized hardware, a consumer GPU is sufficient.

The choice between managed and self-hosted depends on requirements. The OpenAI API is simple but limited. Self-hosted provides control but requires expertise.

Evaluation is critical. Without metrics and a test set, it is impossible to determine whether fine-tuning works.

Iterations are inevitable. Plan for 3-5 iterations to achieve production quality.

Monitoring does not end at deployment. A model in production requires constant attention.

Versioning and reproducibility are a necessity, not a luxury.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[02_Data_Preparation|Data Preparation]]
**Next:** [[04_RLHF_and_Alignment|RLHF and Alignment]]
