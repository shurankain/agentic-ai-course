# Scaling Laws: The Science of Model Scaling

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[07_Modern_Architectures|Modern Architectures: MoE, SSM]]
**Next:** [[09_Interpretability|LLM Interpretability]]

---

## Introduction: When Bigger Really Is Better

In 2020, OpenAI researchers published a paper that changed the industry. They discovered that language model performance follows remarkably predictable mathematical laws: increasing the number of parameters, data, and compute leads to quality improvements along strict power-law relationships.

This discovery transformed LLM development from an empirical art into an engineering discipline. Previously, researchers relied on intuition and experimentation: "let's try a model twice as large and see what happens." Now it became possible to compute optimal configurations, predict results before training begins, and make informed decisions about allocating budgets in the millions of dollars.

Imagine you are planning to train a model with a budget of 10 million dollars. Before scaling laws, you would have proceeded by trial and error. Now you can mathematically compute: how many parameters the model should have, how many tokens it needs to be trained on, and what quality you will get in the end — all within a few percent accuracy.

Understanding scaling laws goes far beyond academic interest. For an AI architect, these laws determine:
- How much it costs to train a model of the required quality (e.g., if you need a GPT-3.5-level model, it will cost approximately 5 million dollars in compute)
- What model size is optimal for a given budget (with 1 million dollars, it is better to train a 7B parameter model for longer than a 70B model briefly)
- When it makes sense to increase data versus parameters (for most tasks, a balanced ratio of 20:1 tokens to parameters)
- What qualitative leaps to expect during scaling (chain-of-thought reasoning appears at approximately 10B parameters)

---

## Original OpenAI Scaling Laws

### Mathematical Formulation

The original scaling laws (Kaplan et al., 2020) established power-law relationships between loss and three factors. A power-law relationship means that improvement follows the formula L = (constant / size) raised to the power alpha. This is fundamentally different from a linear or exponential relationship and has important practical implications.

**Loss as a function of parameters:** model quality improves proportionally to the number of parameters N raised to the power alpha. The formula is L(N) = (N_c / N)^α_N, where N_c and α_N are empirically determined constants. In practice, this means: if you double the number of parameters from 7B to 14B, loss improves by approximately 10-15%.

**Loss as a function of data:** analogous to parameters, quality depends on the number of training tokens D. The formula L(D) = (D_c / D)^α_D shows that doubling data from 500B to 1T tokens yields approximately the same improvement as doubling parameters.

**Loss as a function of compute:** total compute C (measured in FLOPS — floating-point operations) also follows a power-law relationship L(C) = (C_c / C)^α_C. This is the key formula for planning: if you have a specific GPU budget, it allows you to predict the final quality.

**Combined law:** all three factors combine into a single formula that accounts for the interaction of parameters and data. This enables optimization of resource allocation: for example, deciding whether it is better to train a 13B model on 1T tokens or a 7B model on 2T tokens given the same compute budget.

### Key Empirical Findings

**Smoothness of scaling:** The most surprising finding is that the relationship remains strictly power-law across many orders of magnitude. A model with 1 million parameters, 100 million, and 100 billion parameters — all follow the same curve. This means extrapolability: you can train a small model, plot the graph, and predict the quality of a model 1000 times larger.

**Parameter priority:** With a fixed compute budget, the original laws recommended using larger models with fewer data. This led to the creation of "wide" models like GPT-3 (175B parameters on 300B tokens). It later turned out this was a mistake — more on this in the Chinchilla section.

**Diminishing returns:** Each subsequent doubling yields progressively smaller quality gains. If the transition from 1B to 2B parameters improves quality by 15%, then the transition from 100B to 200B yields only 8% improvement. This is a fundamental limitation, meaning it is impossible to achieve perfect quality simply by increasing size — the curve asymptotically approaches a theoretical limit.

**Universality:** The laws work across different architectures (Transformer, LSTM, even CNN for language), different languages (English, Chinese, code), and different tasks (generation, classification). This speaks to the fundamentality of the phenomenon — scaling laws reflect something deep about the nature of learning, not the specifics of a particular architecture.

### Practical Implications of the Original Laws

Based on these laws, OpenAI recommended a specific scaling strategy: with a 10x increase in compute, increase parameters approximately 5x and data only 2x. Large models were recommended to be trained for fewer epochs, using early stopping when loss stops improving. Priority was given to model size over data volume.

This strategy led to the creation of GPT-3 (175B parameters on 300B tokens) and defined the industry's architectural decisions for several years. All major models of 2020-2021 followed this pattern: Gopher (280B), Megatron-Turing NLG (530B), PaLM (540B). All of them were "wide" but relatively "undertrained" by modern standards.

However, everything changed in 2022.

---

## Chinchilla: Rethinking Optimal Scaling

### The 2022 Revolution

In March 2022, DeepMind published the paper "Training Compute-Optimal Large Language Models," known as the "Chinchilla paper." Their conclusions differed radically from OpenAI's recommendations and literally turned the industry upside down within months.

**Key insight:** Most existing models were severely undertrained. GPT-3, Gopher, PaLM — all these giants with hundreds of billions of parameters were trained for too little time. With optimal distribution of compute between parameters and data, training needed to go significantly longer — tens of times longer than was previously practiced.

This was a shocking discovery. The industry had spent hundreds of millions of dollars training enormous models that were suboptimal. It turned out that better results could be achieved with a model 4 times smaller, simply by training it properly.

### Chinchilla Scaling Law: New Mathematics

The new formulation of the optimal ratio reveals symmetry: the optimal number of parameters N and tokens D scale equally with compute C, both proportional to the square root of C. Mathematically, N_opt is proportional to C^0.5, and D_opt is also proportional to C^0.5.

This differs drastically from the original OpenAI scaling laws, which recommended N proportional to C^0.73 and D proportional to C^0.27. The difference is enormous: with a 100x increase in compute, OpenAI recommended increasing parameters 50x and data 2x, while Chinchilla recommended parameters 10x and data 10x.

**Chinchilla rule of thumb:** For compute-optimal training, the number of tokens should approximately equal 20 multiplied by the number of parameters. This simple rule is easy to remember and apply: a 7B parameter model should be trained on 140B tokens, a 70B model on 1.4T tokens.

### Comparing Approaches: The Numbers Speak for Themselves

| Model | Parameters | Training Tokens | Ratio D/N | Status |
|--------|-----------|-----------------|-----------|--------|
| GPT-3 | 175B | 300B | 1.7× | Undertrained |
| Gopher | 280B | 300B | 1.1× | Undertrained |
| Chinchilla | 70B | 1.4T | 20× | Compute-optimal |
| LLaMA | 65B | 1.4T | 21× | Compute-optimal |
| LLaMA 2 | 70B | 2T | 29× | Over-optimal* |

Note: GPT-3 with 175 billion parameters was trained on only 300 billion tokens — a ratio of just 1.7. By Chinchilla standards, it should have been trained on 3.5 trillion tokens — almost 12 times more! Gopher is even worse: 280B parameters on 300B tokens — this is catastrophic undertraining.

### Proof Through Experiment: The Triumph of Chinchilla

DeepMind did not just propose a theory — they proved it experimentally. Chinchilla (70B parameters, 1.4T tokens) outperformed Gopher (280B parameters, 300B tokens) on all 152 benchmarks, using the same compute budget. The result was not marginal — Chinchilla won by 5-10% on most metrics.

This led to revolutionary conclusions:
- **4x fewer parameters to store** — instead of 560GB (Gopher in FP16), only 140GB (Chinchilla)
- **4x faster inference** — proportionally fewer computations per token
- **Better quality** — despite the smaller size
- **Same training cost** — the same compute budget

This is an ideal trade-off: better on all metrics. The only "downside" is the need for more data, but there are trillions of tokens of data available on the internet.

### Practical Consequences: How the Industry Changed

**Industry reassessment:** After the Chinchilla publication in March 2022, all new models began following a compute-optimal or even overtrained approach. LLaMA from Meta (February 2023) — 65B on 1.4T tokens, exactly Chinchilla-optimal. Mistral 7B — overtrained for maximum inference efficiency. Phi models from Microsoft — aggressively overtrained small models.

**Inference vs Training trade-off:** The industry realized that a model can be deliberately overtrained (trained on more tokens than is optimal for training) to produce a smaller model for cheap inference. This makes sense for high-load APIs: overpay once on training, but save millions on inference every day.

**Data becomes the limiting factor:** With compute-optimal training, high-quality data is often insufficient. A 70B model requires 1.4T tokens of quality text. The internet is large, but not infinite. This triggered a race for quality data: books, scientific papers, code, synthetic data from stronger models. Techniques emerged for data filtering, data augmentation, and constitutional AI for generating quality data.

### Why Chinchilla Optimal Is Not Always Optimal

Chinchilla optimal minimizes training cost for a fixed quality level. This is an important metric, but in real production it is not the only one and often not even the primary one.

**Inference cost dominates at scale:** Consider the economics of a real service. Training a 70B model costs approximately 10 million dollars — this is a one-time expense, paid once. Inference for such a model costs approximately $0.01 per request (depends on length, but the order of magnitude is correct).

The break-even point occurs at approximately 1 billion requests: 10 million dollars divided by $0.01 per request = 1 billion requests. Sounds like a lot? For ChatGPT with approximately 100 million active users per day, each making 10 requests, that is 1 billion requests per day. Inference cost exceeds training cost after just one day of operation!

Over a year of operation, inference will cost 3.65 billion dollars, while training was only 10 million — meaning inference is 365 times more expensive. In this scenario, even a 50% reduction in inference cost (through a smaller model) justifies 2-3 times higher training cost.

**Total Cost of Ownership formula:** The total cost of ownership for a model is TCO = Training_cost + Inference_cost × Total_requests. Inference cost is proportional to the number of parameters, context length, and number of tokens in the response. If you expect billions of requests, reducing the model from 70B to 7B (by 10x) yields 10x savings on inference, which quickly recoups any additional training cost.

**When to overtrain a small model:**

| Scenario | Strategy | Example |
|----------|-----------|--------|
| High volume API | Overtrain small | GPT-4o-mini: 8B, 10T+ tokens |
| Edge deployment | Max overtrain | Phi-3-mini: 3.8B, aggressive training |
| Low volume, quality critical | Chinchilla large | Research models |
| One-off analysis | Undertrain large | Quick experiments |

**LLaMA 2 as an anti-Chinchilla example:** Meta deliberately chose an overtraining strategy for LLaMA 2. The 7B model was trained on 2 trillion tokens — a 285x excess over the parameter count. The 70B model on the same 2T tokens — a 28x excess.

By Chinchilla optimal, the 7B model should be trained on 140B tokens, and the 70B on 1.4T tokens. Reality: both models were trained equally long on 2T tokens. The 7B model is "overtrained" 14x relative to optimal.

Why? Because LLaMA 2 7B runs 10 times faster than 70B, requires 10 times less memory, and can run on a single GPU instead of multiple. For production deployment, this is critical. The additional training cost (14x overtraining) pays for itself within weeks of service operation.

**Practical recommendations for choosing a strategy:**
1. Calculate expected total requests over the model's lifetime (typically 1-2 years before retraining)
2. Compute the break-even point where training cost equals inference savings
3. If more than a billion requests are expected, overtrain a smaller model
4. If quality is critical and volume is low (research, specialized tasks), use a Chinchilla-optimal large model
5. Account for latency requirements — smaller models respond faster

---

## Emergence: Qualitative Leaps During Scaling

### The Phenomenon of Emergent Abilities

At a certain scale, models suddenly acquire abilities that cannot be extrapolated from smaller models. This is called emergence — the appearance of new qualities from quantitative growth.

Imagine: you train a 1B parameter model and it cannot solve simple arithmetic problems. You train 3B — still no. 7B — no. Then you train 10B and suddenly it starts adding multi-digit numbers with 70% accuracy. This is not a smooth improvement from 0% to 70% — it is a jump. The 9B model cannot do it, the 10B model can.

**Examples of emergent abilities:**

**Multi-step arithmetic:** The ability to add and multiply multi-digit numbers appears around 10B parameters. Models with 1-7B give virtually random answers to problems like "374 + 892 = ?". Models at 10B+ begin solving such problems with reasonable accuracy. This is a fundamental ability requiring the retention of intermediate results in the model's "working memory."

**Word manipulation:** Anagrams, rhymes, counting letters in a word — these abilities appear at different scales. Models learn to generate meaningful text long before they begin to understand its structure at the character level.

**Theory of mind:** Understanding the mental states of other people — what they know, what they think, what their intentions are. This is a high-level ability requiring modeling of other agents' internal states. It appears only in large models at 70B+.

**Chain-of-thought reasoning:** The ability to reason step-by-step, where the model "thinks aloud" before answering. Models smaller than 10B generate meaningless chains of reasoning that do not lead to the correct answer. Large models use CoT effectively, improving accuracy by 20-50% on complex tasks.

### The Mechanism of Emergence: The Great Debate

The question of the nature of emergence is one of the most hotly debated in modern ML. On one hand, it looks like magic: the model suddenly "realizes" new concepts. On the other hand, it may be an illusion created by our measurement metrics.

**Phase transition hypothesis:**
Some researchers suggest that emergence is a genuine phase transition, analogous to physical phenomena. Water cools smoothly from 10°C to 1°C, but at 0°C a qualitative jump occurs — it turns into ice. Perhaps something analogous happens in neural networks: the model improves smoothly up to a certain threshold, after which the internal structure reorganizes and new abilities appear.

This hypothesis is tempting because it explains why some abilities genuinely appear suddenly. It also has analogies with percolation theory in physics: when enough nodes in a network become connected, the system suddenly acquires the ability to transmit a signal from one end to the other.

**Alternative hypothesis — metric artifact (2023):**
A team of researchers from Stanford and Google in 2023 published a provocative paper "Are Emergent Abilities of Large Language Models a Mirage?" They showed that many "jumps" disappear when the measurement metric is changed.

The key idea: if you measure performance with a binary metric (correct/incorrect, 0 or 1), you create an artificial threshold. The model's actual ability may grow smoothly from 0.1 to 0.9, but until it exceeds 0.5 (or another threshold for a "correct" answer), you see 0% accuracy. Then suddenly — boom, 80% accuracy. A jump!

But if you measure with a continuous metric (e.g., log-probability of the correct answer, or partial correctness), the same process looks like a smooth curve. No jump.

Example: the task "add two three-digit numbers." If an answer is counted as correct only with an exact match (374 + 892 = 1266), small models give 0%, because they err in at least one digit. But if you count partial correctness (how many digits are correct), you see smooth improvement: the model first learns to add units correctly, then tens, then hundreds. This is a gradual process, not a jump.

**Community consensus (2024-2025):**
The debate continues, but a practical consensus is forming:
- Some abilities do appear abruptly — these are real phase transitions
- Many "jumps" are artifacts of measurement, metric choices, and task selection
- Practically important: one cannot reliably extrapolate small model quality to large models for new abilities
- Testing at the target scale is necessary rather than relying on scaling laws for new task types

### Practical Implications for Emergence

**What this means for development:**

**Use continuous metrics where possible:** Instead of binary "correct/incorrect," use metrics that show degree of correctness. For example, for arithmetic: count how many digits are correct, not just the entire answer. For text generation: use BLEU, ROUGE, perplexity, not just exact match. This allows you to see real progress and avoid the illusion of emergence.

**Do not rely on extrapolation from small models:** If a 1B model cannot solve a task, this does not mean that 10B will also fail. Scaling laws predict overall quality (perplexity), but not the appearance of new abilities. Evaluating new capabilities requires testing at the actual scale.

**Test on real tasks, not just benchmarks:** Many standard benchmarks use binary metrics and may create the illusion of emergence. Test on your actual use cases with relevant metrics.

**"Emergence" may indicate a problem with the metric:** If you observe a sharp jump in performance during scaling, ask the question: is this a real new ability or a measurement artifact? Try different metrics and see whether the jump persists.

### Grokking: Delayed Learning

A separate mysterious phenomenon is grokking: an ability appears significantly later than when training loss reaches its minimum. The model may "memorize" the training data (loss drops to zero) but not generalize (test accuracy remains low). Then, after hundreds or thousands of additional training epochs, test accuracy suddenly surges.

The typical grokking pattern: training loss drops quickly in the first epochs and reaches near zero. Test accuracy remains at the level of random guessing. Many more epochs pass without visible progress. And suddenly, for example at epoch 3000, test accuracy begins to rise and reaches 90%+ within a few epochs.

This contradicts the usual intuition: it would seem the model has already converged, loss is not improving, so why continue training? But grokking shows that processes of "restructuring" in a positive sense continue inside the model — the model transitions from memorization to understanding patterns.

**Grokking reveals critically important things:**

**Training loss is a poor predictor of real abilities:** Low training loss does not guarantee generalization. The model can perfectly memorize data but not understand the rule. This is especially important for synthetic tasks (algorithms, mathematics) and structured reasoning.

**Validation on downstream tasks is essential:** Do not rely solely on training/validation loss. Regularly test the model on the real tasks you care about. Grokking may mean that the ability will appear — you just need to train longer.

**Sometimes it is worth training longer than loss suggests:** If loss has converged but performance on downstream tasks is unsatisfactory, try continuing training. Grokking may occur. This is especially relevant for tasks requiring reasoning.

---

## Scaling Laws for Downstream Tasks

### Beyond Perplexity: What Matters in Practice

Perplexity (cross-entropy loss on the next token) is a convenient metric for scaling laws because it is smooth, continuous, and easy to compute. But in practice, we care about specific tasks: summarization, translation, code, question answering. How does scaling affect them?

The good news: scaling laws work for downstream tasks too. The bad news: they work differently for different tasks, and extrapolation is harder.

**Key observations:**

**Linear relationship in log-log scale is preserved:** If you plot "model size" (on the X-axis in logarithmic scale) against "task quality" (on the Y-axis in logarithmic scale), you get an approximately straight line. This indicates a power-law relationship, just as with perplexity.

**Different exponents for different tasks:** Some tasks scale aggressively (doubling the model significantly improves quality), others slowly. For example, simple sentiment classification scales well — a 7B model is already close to the maximum. Complex reasoning scales slowly — even 70B models are far from ideal.

**Task-specific thresholds:** For some tasks, there is a minimum size below which the model cannot cope at all. For example, reasonable-quality code generation requires a minimum of 3-7B parameters. Smaller models generate syntactically valid but logically meaningless code.

### Practical Framework for Evaluating Scaling on Your Task

If you plan to use an LLM for a specific task and want to understand what model size is needed, follow this process:

**Step 1:** Prepare a test set for your task with a relevant metric. Do not rely on public benchmarks — they may not correlate with your tasks.

**Step 2:** Test 3-5 models of different sizes. It is important to cover several orders of magnitude: for example, 1B, 7B, 13B, 70B. You can use public models (LLaMA, Mistral) to avoid training your own.

**Step 3:** Evaluate each model on your metric. Use identical prompts and conditions for all models.

**Step 4:** Plot a log-log graph: log(model size) on the X-axis, log(quality metric) on the Y-axis. If you see an approximately straight line, extrapolation is possible.

**Step 5:** Extrapolate to the target quality and compute the required model size. Account for margin of error — extrapolation beyond 10x is risky.

### When Scaling Does Not Work: Important Limitations

Scaling is a powerful tool, but not a universal one. There are tasks and situations where simply increasing the model is ineffective or useless.

**Factual knowledge cut-off:** The model cannot know facts after its training date, regardless of size. If training ended in January 2024, the model does not know events of February 2024, even if it has a trillion parameters. Solution: RAG (retrieval-augmented generation), regular retraining, or fine-tuning on new data.

**Specific domain knowledge:** Without domain-specific data, size will not help. A model trained on the general internet performs poorly on medical diagnoses or legal documents, regardless of size. Specialized data during training or fine-tuning is required.

**Exact retrieval:** Exact information retrieval (e.g., "what is company X's phone number?") scales poorly. Models learn patterns but are not perfect databases. For exact retrieval, use actual databases or search + LLM.

**Complex multi-step reasoning:** Improvement is slow with scaling. Tasks like "solve a math olympiad problem" or "write a proof of a theorem" require dozens of reasoning steps. Scaling helps but is insufficient. Special techniques are needed: chain-of-thought prompting, tree-of-thoughts, self-consistency, or test-time compute scaling.

### Scaling Laws for Fine-Tuning: Different Mathematics

Fine-tuning has radically different scaling laws from pretraining. Understanding this is critically important because intuition from pretraining can be misleading.

**Key differences from pretraining:**

| Aspect | Pretraining | Fine-tuning |
|--------|-------------|-------------|
| Data requirement | D ∝ N (proportional to parameters) | D << N (100-10K examples sufficient) |
| Compute | C ≈ 6ND (enormous) | C << pretraining (1000x less) |
| Scaling exponent | ~0.3 (slow growth) | ~0.1-0.2 (fast saturation) |
| Bottleneck | Compute/Data quantity | Data quality |

**Formula for fine-tuning scaling:** Performance grows proportionally to the base model quality, multiplied by a saturation function of the fine-tuning data volume. Mathematically: Performance is proportional to base_model_quality multiplied by (1 - exp(-k × D_ft / D_saturation)).

Here D_ft is the fine-tuning data volume, D_saturation is the saturation point (typically 10K-100K examples for most tasks), and k is a task-specific constant determining how quickly learning occurs.

The key difference from pretraining: exponential saturation. In pretraining, more data is almost always better. In fine-tuning, beyond a certain volume, additional data yields minimal gains.

**Practical observations and numbers:**

**Rapid saturation after 10-50K examples:** The typical curve for a classification task looks like this: 100 examples yield 60% accuracy, 1000 examples — 80%, 10000 examples — 90%, 100000 examples — 92%. Note: going from 10K to 100K (a 10x increase in data) yields only 2% improvement. This is classic diminishing returns.

**Base model size matters more than fine-tuning data for complex tasks:** A 70B model with 1000 fine-tuning examples typically outperforms a 7B model with 100000 examples on complex tasks (reasoning, creative writing, code). For simple tasks (sentiment classification, named entity recognition), the difference is smaller — even small models saturate.

**LoRA/PEFT scaling laws:** Parameter-efficient fine-tuning (PEFT) methods like LoRA have their own scaling laws. LoRA rank 8 is sufficient for most tasks and yields 80-90% of full fine-tuning quality. Rank 64 is practically indistinguishable from full fine-tuning. Rank above 64 yields diminishing returns — you spend more memory and compute with almost no quality gain. Performance scales approximately as the logarithm of rank up to the saturation point.

**When to use fine-tuning vs prompting:**

| Fine-tuning Data | Best Approach | Rationale |
|--------------------|---------------|-------------|
| <100 examples | Few-shot prompting | Fine-tuning will overfit, prompting is more effective |
| 100-1K | LoRA rank 8-16 | Sufficient data for adaptation, low rank is adequate |
| 1K-10K | LoRA rank 32-64 | Can learn complex patterns, higher rank is beneficial |
| >10K | Full fine-tuning or QLoRA | Maximum adaptation, use all parameters |

**Critically important insight:** Fine-tuning cannot add knowledge that is absent from the base model. It "activates" and "specializes" existing capabilities. If the base model does not understand medical terminology, no amount of fine-tuning on medical tasks will teach it that terminology from scratch — a base model trained on medical texts is needed, or a massive fine-tuning dataset.

---

## Practical Implications for Architectural Decisions

### Choosing Model Size: A Step-by-Step Approach

**For training a new foundation model from scratch:**

If you plan to train your own model (typically done by large companies or research organizations), follow this process:

**Step 1:** Determine your compute budget C in FLOPS or in dollars. For example, you have 1000 A100 GPUs for 2 months, or a budget of 5 million dollars.

**Step 2:** Compute the optimal number of parameters N_opt using the Chinchilla formula. Simplified: N_opt approximately equals the square root of (C divided by 6). This yields a balanced distribution between model size and training volume.

**Step 3:** Compute the optimal number of tokens D_opt. Formula: D_opt = C divided by (6 multiplied by N_opt). Or more simply: D_opt is approximately 20 multiplied by N_opt per the Chinchilla rule.

**Step 4:** Verify data availability. Do you have D_opt tokens of quality data? If not, you have two options: collect more data or adjust the plan.

**Step 5:** If quality data is less than D_opt, reduce N (model size) and train longer on the available data. This will be overtraining relative to Chinchilla optimal, but better than an undertrained large model.

**For selecting an existing pre-trained model:**

The more common scenario is selecting from existing models for your use case.

**Step 1:** Define the target quality on your task. What is acceptable quality? What metric matters?

**Step 2:** Evaluate several model sizes on your task: for example, 7B, 13B, 70B. Use publicly available models.

**Step 3:** Build a scaling curve and extrapolate if needed. If 7B gives 60%, 13B gives 70%, 70B will give approximately 80-85% (extrapolation).

**Step 4:** Factor in inference cost. A 70B model is 10 times slower and more expensive than 7B. Are the additional 15% in quality worth a 10x increase in cost? That depends on your use case.

### Trade-off: Train-Time vs Inference-Time Compute

One of the most important strategic choices in ML is how to balance training costs against inference costs.

**Overtraining strategy for inference-critical cases:**

If inference cost is critical (high-load API, edge deployment), it makes sense to deliberately overtrain a smaller model. Example: LLaMA 2 7B was trained on 2 trillion tokens — a 285x excess over parameters. By Chinchilla optimal, this is excessive for training. But the result: a 7B model with quality close to a 13B model, but inference 2 times faster.

The economics are straightforward: additional training costs more (14x overtraining means approximately 14x more GPU-hours). But you get a smaller model that runs faster on every inference call. With billions of requests, savings on inference outweigh additional training cost.

**Break-even calculation formula:** Inference cost is proportional to N (number of parameters) multiplied by requests_per_day. Training cost is a fixed one-time cost. The break-even point (when additional training pays off) approximately equals Training_cost divided by ((N_large minus N_small) multiplied by cost_per_request) divided by daily_requests.

Example: you additionally spent 5 million dollars on overtraining 7B instead of using 13B. The difference in inference cost: $0.001 per request. At 10 million requests per day, the savings are $10,000 per day. Break-even in 500 days. Beyond that — pure profit.

### When to Increase Model vs Data vs Compute: Decision Matrix

| Situation | Recommendation | Rationale |
|----------|--------------|-------------|
| Abundant data, limited compute | Small model, overtrain | Maximize data usage, compensate for size |
| Abundant compute, limited data | Large model, undertrain | Large model uses limited data more efficiently |
| Balanced resources | Chinchilla-optimal | Optimal use of both resources |
| Inference-critical (high volume) | Overtrain small model | Inference savings outweigh training cost |
| Quality-critical (low volume) | Large model + overtrain | Maximum quality matters more than efficiency |

---

## Scaling Laws 2.0: Modern Developments

### Test-Time Compute Scaling: An Established Paradigm

Test-time compute (TTC) emerged in 2024 with OpenAI o1 and is now an established scaling dimension, adopted by all major providers. Every frontier lab offers reasoning models that trade inference compute for quality.

**Traditional scaling:** Model quality was determined by three factors — number of parameters N, data volume D, and compute during training C_train. Formula: Quality = f(N, D, C_train). Inference was simply applying the model, with no additional "thinking."

**Test-time scaling:** Quality now also depends on compute during inference C_inference. Formula: Quality = f(N, D, C_train, C_inference). The model can "think longer" on complex tasks, generating intermediate reasoning, testing hypotheses, correcting errors. This is no longer experimental — it is the default for complex tasks.

**All major providers now offer TTC:**
- **OpenAI o3** (January 2025 GA) and **o4-mini** (April 2025): `reasoning_effort` parameter (low/medium/high). o3 achieves 96.7% on AIME 2024, 87.7% on GPQA Diamond
- **Claude extended thinking** (2024-2025): transparent `<thinking>` blocks with `budget_tokens` for fine-grained control. Claude Sonnet 4 with extended thinking competes with o3 on reasoning benchmarks
- **Gemini thinking** (2025): `thinkingBudget` parameter. Gemini 2.5 Pro with thinking leverages 1M+ token context for reasoning
- **DeepSeek-R1** (January 2025): open-weight reasoning model achieving o1-level performance at lower cost, trained via GRPO

### Train-Time vs Test-Time Compute Trade-off

This dimension of optimization is now well-understood and actively exploited in production.

**Formalizing the trade-off:** Total compute now equals C_train (one-time) plus C_inference multiplied by N_requests. The key innovation: C_inference is now a variable that can be chosen for each request.

For a simple request ("Translate 'hello' to French"), you can use C_inference = 1x — minimal thinking, fast response. For a complex request ("Prove Fermat's theorem for n=3"), you can use C_inference = 100x — extended reasoning, deep deliberation.

**Comparing strategies:**

| Strategy | C_train | C_inference | Use Case | Example |
|-----------|---------|-------------|----------|--------|
| Traditional LLM | High | Fixed (low) | General tasks | GPT-4o, Claude Sonnet 4 |
| Reasoning model | Medium | Variable (high) | Reasoning tasks | o3, Claude extended thinking |
| Small + TTC | Low | Very high | Quality-critical | o4-mini, R1-distill-7B |
| Large + minimal | Very high | Low | Latency-critical | Large model, instant responses |

**When test-time compute is economically advantageous:**

Condition: Cost_TTC must be less than Cost_larger_model. This holds with: low request volume (not billions of requests per day), variable task complexity (some requests are simple, some complex — compute can be allocated adaptively), quality prioritized over latency, ability to "think" asynchronously (the user is willing to wait).

**Smaller models + TTC vs larger models:** o4-mini with sufficient reasoning outperforms GPT-4o on math and coding at a fraction of the cost. DeepSeek R1-distill-7B competes with much larger standard models. This democratizes reasoning capabilities — even small models can achieve strong reasoning with sufficient TTC.

**Architectural implications:**

**Adaptive compute allocation:** A system with a router that determines task complexity from the request. Simple requests (factual questions, translation) receive minimal deliberation — fast response. Complex requests (math, code, reasoning) receive extended reasoning — the model "thinks" for minutes. 5-10x savings while maintaining quality on complex tasks and speed on simple ones.

**Latency/quality toggle for the user:** The user can choose a mode: "Fast mode" — 1-2 seconds, reduced quality, for quick questions. "Thinking mode" — 30-60 seconds, maximum quality, for important tasks. "Background mode" — hours, best quality + cheaper (very extended deliberation), for non-urgent analysis.

**New evaluation metrics:** Quality per unit of compute (accounting for both training and inference), quality per dollar (full economic efficiency), latency-quality Pareto frontier (optimal trade-off between speed and quality).

### Scaling Laws for Reasoning: The Specifics of Thinking

Reasoning tasks have distinct scaling laws, different from general tasks:

**Chain-of-thought scaling:** Improvement from CoT grows approximately as the logarithm of model size N. Small models (<7B) show almost no improvement from CoT — they generate meaningless reasoning. Models at 10B+ begin using CoT effectively, with 20-50% improvement on reasoning tasks. The improvement grows logarithmically: from 7B to 70B (10x size) yields approximately 2x improvement in reasoning.

**Self-consistency and sampling:** Generating N_samples responses and selecting the most frequent one improves accuracy. Scaling: improvement is proportional to the square root of N_samples until saturation. 10 samples yield approximately 30% improvement, 100 samples — 50%, beyond that diminishing returns.

**Optimal strategy:** A balance between model size and the number of CoT samples/thinking time. Sometimes a 13B model with 10 samples is better than a 70B model with 1 sample — cheaper and higher quality for certain tasks.

### GRPO: A New Scaling Paradigm for Reasoning

GRPO (Group Relative Policy Optimization), introduced with DeepSeek R1 (January 2025), represents a new scaling paradigm that changes how reasoning capabilities emerge during training.

**How it scales:** For each prompt, GRPO samples K outputs (K=8-64) from the current policy, computes rewards via a verifier, and normalizes advantages within the group. This replaces both the reward model and value model from PPO with simple group statistics. Training scales with the number of samples per prompt and the quality of the verifier.

**R1-Zero: Reasoning emergence from pure RL.** Applying GRPO with only rule-based rewards (correctness, format compliance) to a base model — without any SFT on reasoning traces — led to spontaneous emergence of chain-of-thought reasoning. This is a fundamental scaling result: the capacity for structured reasoning is latent in pretrained models and scales with RL training.

**Distillation scaling:** DeepSeek distilled R1's reasoning into smaller models (1.5B, 7B, 8B, 14B, 32B, 70B variants). Distillation follows its own scaling law: smaller distilled models retain a surprising fraction of the teacher's reasoning capability. R1-distill-32B outperforms many larger standard models on reasoning benchmarks, demonstrating that reasoning can be efficiently compressed.

### Implications for System Architecture

**Reasoning models can be smaller:** It is not necessary to train a gigantic model for reasoning tasks. A small model with effective test-time compute can outperform a large model with minimal thinking. o4-mini competes with much larger models. DeepSeek R1-distill-7B demonstrates strong reasoning at minimal cost.

**Adaptive compute by task type:** Different tasks require radically different inference budgets. The architecture should determine the task type and allocate corresponding compute. Classification — 1x, summarization — 2-5x, reasoning — 10-100x. Budget control APIs (`reasoning_effort`, `budget_tokens`, `thinkingBudget`) enable per-request optimization.

**Cost modeling becomes more complex:** You cannot simply multiply the cost per request by the number of requests. Thinking tokens add variable inference cost. You need to model the distribution of task complexity, adaptive compute allocation, and the latency-quality trade-off for each request type.

---

## Key Takeaways

**Scaling laws are predictable power-law relationships** between compute, parameters, data, and quality. This is not merely an empirical observation but a fundamental property of neural network training, enabling model development planning with mathematical precision. If you know the budget, you can predict quality before training begins.

**Chinchilla optimal: D ≈ 20N** — for compute-optimal training, the number of tokens should be approximately 20 times the number of parameters. This radically changed the industry in 2022, showing that GPT-3 and other models were catastrophically undertrained. Modern models follow this rule or deliberately deviate from it for economic reasons.

**Emergence is real but often overstated** — some abilities genuinely appear abruptly upon reaching a critical scale (chain-of-thought reasoning, multi-step arithmetic). But many "jumps" are artifacts of binary measurement metrics. Use continuous metrics to see the real picture of gradual improvement.

**Overtraining is economically justified for inference efficiency** — if your model will handle billions of requests, it is worth overpaying on training to get a smaller, faster model. Inference cost dominates for high-volume services. LLaMA 2 7B was overtrained 14x relative to Chinchilla optimal precisely for this reason.

**Test-time compute is an established scaling dimension** — all major providers now offer reasoning models (o3, o4-mini, Claude extended thinking, Gemini thinking, DeepSeek R1). GRPO demonstrated that reasoning can emerge from pure RL. Distillation enables efficient transfer of reasoning capabilities to smaller models. Budget control APIs enable per-request cost-quality optimization.

**Scaling is not universal** — there are tasks where simply increasing the model is ineffective: factual knowledge after the cut-off date, exact retrieval from memory, domain-specific knowledge without corresponding data. For these cases, specialized solutions are needed (RAG, fine-tuning, external tools).

**Data is becoming the main bottleneck** — with Chinchilla-optimal scaling, a 70B model requires 1.4 trillion tokens of quality data. High-quality text is not infinite. The industry is moving toward synthetic data, aggressive filtering, and multimodal data to overcome this limitation.

**Fine-tuning has different scaling laws** — saturation occurs quickly (10K-100K examples), base model quality matters more than the volume of fine-tuning data, and knowledge absent from the base model cannot be added. Understanding these differences is critical for efficient resource utilization.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[07_Modern_Architectures|Modern Architectures: MoE, SSM]]
**Next:** [[09_Interpretability|LLM Interpretability]]
