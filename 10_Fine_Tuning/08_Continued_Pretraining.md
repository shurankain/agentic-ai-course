## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[07_Synthetic_Data|Synthetic Data]]
**Next:** [[09_Preference_Data|Preference Data]]

---

# Continued Pretraining: Domain Adaptation

Continued Pretraining (CPT) is the continuation of pretraining on domain-specific data that deeply adapts a model to a new domain while preserving general capabilities.

## What is Continued Pretraining

CPT continues the pretraining process of an already pretrained model on a new domain-specific corpus. It uses the same objective (next token prediction), but the data is specific to the target domain.

| Method | Objective | Data | Depth of Changes |
|--------|-----------|------|------------------|
| Pretraining | Next token | General web corpus | Creation from scratch |
| CPT | Next token | Domain corpus | Deep adaptation |
| SFT | Next token (on examples) | Instructions + responses | Behavioral adaptation |
| RLHF | Reward | Preferences | Alignment |

## When CPT is Needed

**CPT is necessary when:**
- The domain significantly differs from pretraining data
- Domain terminology is specialized
- Large volume of domain data (millions to billions of tokens)
- The task requires deep understanding

**CPT is excessive when:**
- The task can be solved with prompting
- Data is scarce (hundreds of examples)
- Only format/style changes are needed
- The domain is close to the pretraining distribution

## Mechanics of Continued Pretraining

**Training objective:** causal language modeling (next token prediction). Next token prediction forces the model to learn domain language statistics, semantic relationships, reasoning patterns, and factual knowledge.

**Learning rate:** critical hyperparameter. LR = 1e-5 to 5e-5 (lower than pretraining ~1e-4). Warmup: 1-5% of total steps. Decay: cosine or linear to ~10% of peak LR.

**Duration and volume:**
- Minimum viable: ~100M tokens (1 epoch)
- Good CPT: 1-10B tokens
- Serious domain shift: 10-100B+ tokens
- Number of epochs: typically 1-3

## Catastrophic Forgetting: The Main Enemy

**Catastrophic forgetting** is the loss of previously learned knowledge when training on new data. The model "forgets" general capabilities as it immerses into the domain.

**Symptoms:** degradation on general benchmarks, loss of ability to communicate on general topics, forgetting rare knowledge.

**Mitigation methods:**

1. **Data mixing** — training data: α domain + (1-α) general. Typically α = 0.7-0.9.

2. **Replay buffer** — periodic "replay" of examples from pretraining.

3. **Elastic Weight Consolidation (EWC)** — penalty for changing "important" weights via Fisher information.

4. **Progressive learning rate** — different LR for different layers. Early layers (general features): low LR. Deep layers (specific features): higher LR.

5. **Checkpoint averaging** — averaging weights before and after CPT.

## Domain-Adaptive Pretraining (DAPT)

DAPT is CPT specifically optimized for downstream tasks in a domain. Even a relatively small volume of domain data (a few GB) can significantly improve performance.

**DAPT vs TAPT:**
- DAPT: pretraining on all domain data — broad coverage
- TAPT: pretraining on data close to a specific task — narrow but deep

Combining DAPT + TAPT often yields the best results.

**Examples:** BioBERT/PubMedBERT (biomedicine), LegalBERT (legal texts), SciBERT (scientific papers).

## CPT for Low-Resource Languages

For languages with low representation, CPT on a target language corpus significantly improves quality:

1. Vocabulary extension — adding tokens
2. Language modeling — learning language statistics
3. Cultural adaptation — learning culturally specific concepts

Data mixing with English helps preserve multilingual capabilities.

## Practical Recommendations

**Data preparation:** quality over quantity, deduplication, filtering, source balance. Long documents are better than short ones, diversity of styles and sources.

**Base model selection:** 7B models are often optimal for domain-specific tasks. Modern architectures (Llama, Mistral) are preferred.

**Training monitoring:**
- Domain perplexity (should decrease)
- General perplexity (should not increase significantly)
- Downstream task performance

Early stopping if general perplexity starts increasing.

## CPT vs SFT: When to Choose Which

**Decision tree:**

**Step 1:** Do you have >1 billion tokens of domain text?
- No → SFT or few-shot
- Yes → Step 2

**Step 2:** Is deep understanding or format/style needed?
- Format/style only → SFT
- Deep understanding → Step 3

**Step 3:** How much does the domain differ?
- Slightly → Light CPT (100M-1B tokens)
- Significantly → Full CPT (1-10B+ tokens)

## Combining CPT and SFT

Typical pipeline: CPT on domain corpus → SFT on instruction-following data → Optionally DPO/RLHF.

Each stage:
- CPT: domain knowledge and understanding
- SFT: response format and style
- DPO/RLHF: quality and alignment

## Key Takeaways

Continued Pretraining is the continuation of pretraining on domain data for deep adaptation.

When to use: large data volume (>100M tokens), significant domain shift, deep understanding required.

Catastrophic forgetting is the main risk. Mitigation: data mixing, EWC, progressive LR, checkpoint averaging.

DAPT and TAPT are CPT variants for domain-specific and task-specific data.

Learning rate is critical: 1e-5 to 5e-5 with warmup and decay.

CPT + SFT is the typical pipeline: deep understanding through CPT, behavioral adaptation through SFT.

Monitoring is mandatory: domain perplexity (decreasing), general perplexity (not increasing significantly).

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[07_Synthetic_Data|Synthetic Data]]
**Next:** [[09_Preference_Data|Preference Data]]
