# DoRA and Advanced PEFT Methods

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[09_Preference_Data|Preference Data]]
**Next:** [[../11_Evaluation_Testing/01_Metrics_and_Benchmarks|Metrics and Benchmarks]]

---

## Introduction

LoRA revolutionized fine-tuning, but research continues to advance. New methods offer improvements:

- **DoRA:** up to +3.7% on LLaMA-7B (ICML 2024)
- **GaLore:** full-parameter training with LoRA-level memory efficiency
- **QA-LoRA:** quantization-aware LoRA
- **LongLoRA:** efficient fine-tuning for long context

## DoRA: Weight-Decomposed Low-Rank Adaptation

### The Problem with LoRA

LoRA trains a low-rank update: W' = W + BA. It modifies magnitude and direction of weights simultaneously, which can be suboptimal.

The authors discovered that full fine-tuning changes direction significantly, while magnitude changes less. LoRA does not separate these aspects.

### The Idea Behind DoRA

Decomposes the weight matrix into magnitude (m) and directional component (V / ||V||).

DoRA trains:
1. magnitude update: m' = m + Δm
2. directional update: V' = V + BA

Final weights: W' = m' · (V + BA) / ||V + BA||.

DoRA can better approximate full fine-tuning with the same number of parameters.

### Results

| Model | Method | Commonsense | Arithmetic | Avg |
|-------|--------|-------------|------------|-----|
| LLaMA-7B | LoRA (r=32) | 68.9 | 55.8 | 62.4 |
| LLaMA-7B | DoRA (r=32) | 72.1 | 59.9 | 66.0 |
| LLaMA-7B | Full FT | 73.2 | 61.4 | 67.3 |

+3.7% absolute improvement over LoRA.

**When to use:** there is headroom over LoRA, the task requires precise fine-tuning, you are willing to accept a slight increase in compute.

## GaLore: Gradient Low-Rank Projection

**Different approach:** LoRA constrains the weight update to low-rank, GaLore constrains the gradient to low-rank, but trains all weights.

**Algorithm:**
1. Periodically (every 200 steps) compute SVD decomposition of the gradient
2. Extract top-k singular vectors as the projection matrix
3. At each step, project the gradient onto principal directions
4. Apply the projected low-rank gradient to all weights

### Advantages of GaLore

1. Full-parameter training — all weights are updated
2. Memory efficiency — comparable to LoRA
3. No inference overhead — no additional matrices

| Method | Memory | LLaMA-7B Perplexity |
|--------|--------|---------------------|
| Full Fine-tuning | 58GB | 5.68 |
| LoRA (r=256) | 22GB | 5.82 |
| GaLore | 22GB | 5.71 |

### When to Choose

| Criterion | LoRA | GaLore |
|-----------|------|--------|
| Inference | +Overhead | No overhead |
| Quality | Good | Slightly better |
| Simplicity | Simpler | More complex |
| Multi-task | Easy (swap adapters) | Separate models |
| Maturity | Production-ready | Research stage |

## QA-LoRA: Quantization-Aware LoRA

**Problem:** training in FP16 → merge → quantize loses quality.

**Solution:** train with quantization from the start. The base model is loaded in INT4, LoRA adapters remain in FP16. A trainable scale parameter compensates for the difference between quantized and full-precision computations.

For configuration, BitsAndBytesConfig is used: load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16.

## LongLoRA: Efficient Long Context

**Problem:** fine-tuning for long context (32K+) requires O(L²) memory for attention.

**Shift Short Attention (S²-Attn):**
1. The sequence is split into groups (of 2048 tokens each)
2. Half of the heads are applied to standard groups
3. The other half of the heads are applied to shifted groups (offset by group_size/2)
4. Attention is computed only within each group

Reduces complexity from O(L²) to approximately O(L).

| Context | Method | Memory | Perplexity |
|---------|--------|--------|------------|
| 32K | Full Attention | OOM | - |
| 32K | LongLoRA | 28GB | 4.52 |
| 100K | LongLoRA | 48GB | 4.71 |

LongLoRA enables fine-tuning LLaMA-2-7B on 100K context on a single 48GB GPU.

## Comparison Table

| Method | Memory | Quality | Inference | Complexity | Maturity |
|--------|--------|---------|-----------|------------|----------|
| LoRA | Low | Good | +Overhead | Simple | Production |
| DoRA | Low | Better | +Overhead | Medium | New (2024) |
| GaLore | Low | Best | No overhead | Complex | Research |
| QLoRA | Very Low | Good | +Overhead | Medium | Production |
| LongLoRA | Medium | Good | +Overhead | Medium | Research |

## Decision Framework

**Start with LoRA** — production-proven, broad support, straightforward hyperparameters.

**Upgrade to DoRA** if LoRA does not deliver the required quality — expect +2-4% improvement.

**GaLore** for research/maximum quality — approaches full fine-tuning performance.

**Combine methods:** QLoRA with DoRA-style magnitude, LongLoRA with Flash Attention, Adaptive rank selection.

## What Comes Next

**Current trends:**
- Merging adapters — arithmetic for combining LoRAs
- Sparse adapters — even fewer parameters
- Automatic rank selection — adaptive rank during training
- Cross-model transfer — LoRA adapters between similar models

**Papers to watch:** LoRA-FA (Frozen-A), AdaLoRA (Adaptive rank), LoRA Merging techniques.

## Key Takeaways

DoRA separates weight updates into magnitude and direction — +3.7% over LoRA with the same number of parameters.

GaLore projects gradients into a low-rank subspace — full training of all parameters with memory efficiency comparable to LoRA.

QLoRA combines INT4 quantization of the base with FP16 LoRA adapters — maximum memory savings.

LongLoRA uses Shift Short Attention — efficient fine-tuning on long context (100K+ tokens).

LoRA remains the production-proven default; DoRA for improved quality; GaLore for the closest approximation to full fine-tuning.

Method selection is driven by trade-offs: multi-adapter serving (LoRA/DoRA), maximum quality (GaLore), limited memory (QLoRA), long context (LongLoRA).

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[09_Preference_Data|Preference Data]]
**Next:** [[../11_Evaluation_Testing/01_Metrics_and_Benchmarks|Metrics and Benchmarks]]
