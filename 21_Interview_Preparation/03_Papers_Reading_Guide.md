# Papers Reading Guide

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Interview Preparation
**Previous:** [[02_Coding_Exercises|Coding Exercises]]
**Next:** [[04_Behavioral_Staff_Plus|Behavioral for Staff+]]

---

## Introduction

Senior/Staff+ interviews often involve discussing research papers. This tests depth of understanding of fundamental ideas, the ability to critically analyze approaches, and awareness of the current state of the field.

You do not need to know every paper by heart, but you need to be able to discuss the key ones at this level: what problem it solves, what approach is proposed, what the key results are, and what the limitations are.

---

## How to Effectively Read ML Papers

### Three-Pass Approach

Pass 1 Survey (five to ten minutes): title, abstract, introduction (last paragraph), section headers, conclusions, figures and tables (without details). You understand what the paper is about, what the problem is, what the result is.

Pass 2 Comprehension (thirty to sixty minutes): read fully skipping proofs and details, take notes (key ideas, questions), look at figures and equations carefully. You can explain the paper to a colleague.

Pass 3 Deep Dive (several hours): only for important papers, reproduce derivations, critically analyze assumptions, connect with other works. You can implement and improve.

### What to Record

For each paper create a structured note: metadata (authors, year, link), problem (one sentence on what specific problem it solves), key idea (two to three sentences on the main insight or breakthrough), method (bullet points of the main approach, including key formulas), results (main numbers, benchmarks, comparison with baseline), limitations (what does not work, what assumptions, under what conditions it is not applicable), your questions (what remained unclear, what ideas for improvement).

---

## Foundational Papers

### 1. Attention Is All You Need (2017)

Problem: RNN/LSTM have a sequential bottleneck, do not parallelize.

Key idea: replace recurrence with self-attention. Positions are encoded separately.

Architecture: input data passes through an embedding layer with added positional encoding, then through N sequential blocks (each containing a self-attention mechanism and a feed-forward network), and finally produces output tokens.

Key components: multi-head attention (different heads attend to different aspects), scaled dot-product attention (product of Query by transposed Key divided by the square root of d_k), positional encoding (sin/cos of different frequencies).

Results: BLEU 28.4 on WMT En-De (SOTA at the time), three and a half days of training on eight GPUs.

Discussion points: why specifically sin/cos for positions (extrapolation, relative positions), why the square root of d_k (softmax stabilization), limitations (O(n^2) in memory).

### 2. BERT: Pre-training of Deep Bidirectional Transformers (2018)

Problem: how to use unlabeled text for pre-training?

Key idea: Masked Language Modeling — predict masked tokens using context from both sides.

Approach: mask fifteen percent of tokens, predict the original token, additionally Next Sentence Prediction (NSP).

Results: SOTA on eleven NLP tasks, fine-tuning instead of feature extraction.

Discussion points: why bidirectional matters (GPT was unidirectional), whether NSP is useful (later shown it is not — RoBERTa), mask token mismatch between pre-training and fine-tuning.

### 3. GPT-3: Language Models are Few-Shot Learners (2020)

Problem: fine-tuning requires a lot of labeled data.

Key idea: a sufficiently large model can solve tasks in-context without updating weights.

Scale: 175 billion parameters, 300 billion tokens of training, 45 terabytes of text data.

Few-shot learning: the model is shown several examples of a task directly in the prompt (for example several English-to-French translation pairs), after which it is able to perform an analogous task for new input without additional training.

Emergence: capabilities appear at a certain scale.

Discussion points: why in-context learning works (still not fully understood), scaling laws (loss proportional to N^(-0.076)), limitations (no knowledge update, hallucinations).

### 4. InstructGPT / RLHF (2022)

Problem: GPT-3 is not always helpful and can generate harmful content.

Key idea: align the model with human preferences through Reinforcement Learning.

Pipeline: SFT (fine-tune on demonstrations from humans), Reward Model (train a model to predict human preferences), PPO (optimize the policy to maximize reward).

Results: 1.3 billion InstructGPT outperforms 175 billion GPT-3 by human preference, fewer toxic outputs.

Discussion points: reward hacking (the model optimizes a proxy not the true objective), why the KL penalty from the SFT model (preserve capabilities), alternatives DPO, KTO.

### 5. Constitutional AI (2022)

Problem: RLHF is expensive (requires annotators) and can have biases.

Key idea: replace human feedback with AI feedback guided by a constitution — a set of principles.

Approach: generate responses, AI critiques according to the constitution, AI revises the response, RLAIF instead of RLHF.

Example principle: "Choose the response that is least harmful or offensive."

Discussion points: can AI judge itself (bootstrap problem), what principles to include in the constitution, connection with AI Safety.

---

## Efficiency & Scaling Papers

### 6. LoRA: Low-Rank Adaptation (2021)

Problem: full fine-tuning of a 175 billion model requires 700 GB for gradients.

Key idea: fine-tune only low-rank matrices delta-W = BA, where B in R^(d x r), A in R^(r x d), r much less than d.

Advantages: parameters d times r plus r times d instead of d times d, can quickly swap adapters, quality approximately equal to full fine-tuning.

Discussion points: what rank to choose (4-64 is usually sufficient), which layers to adapt (Q, V are more important than K), why low-rank works (intrinsic dimension hypothesis).

### 7. Flash Attention (2022)

Problem: attention requires O(n^2) memory for the scores matrix.

Key idea: IO-aware algorithm — minimize transfers between HBM and SRAM.

Approach: tiling (process in blocks), online softmax (do not materialize the full matrix), recomputation (do not store attention for backward).

Results: 2-4x speedup, memory O(n) instead of O(n^2), up to 16K context without issues.

Discussion points: trade-off recomputation versus memory, when Flash Attention does not help (short sequences), Flash Attention 2/3 improvements.

### 8. Mixture of Experts — Switch Transformer (2022)

Problem: scaling requires proportional growth of compute.

Key idea: sparse activation — each token passes through only a subset of experts.

Architecture: each input token is processed by a router that directs it to the most relevant K experts from the total pool, after which expert outputs are combined through weighted summation.

Switch Transformer: top-1 routing (simplicity), auxiliary load balancing loss, 1.6 trillion parameters but compute equivalent to 100 billion dense.

Discussion points: why load balancing matters (otherwise collapse to one expert), expert parallelism (how to distribute across GPUs), inference challenges (need to load all experts).

### 9. Mamba: Linear-Time Sequence Modeling (2023)

Problem: Transformer attention O(n^2) limits context length.

Key idea: State Space Models with selective mechanism — linear complexity O(n).

Selective SSM: parameters depend on input (unlike classical SSMs), hardware-aware implementation.

Results: 5x throughput on long sequences, competitive quality with Transformers.

Discussion points: when Mamba is better/worse than Transformer, in-context learning in SSMs, hybrid architectures (Jamba).

---

## Agent & Reasoning Papers

### 10. Chain-of-Thought Prompting (2022)

Problem: LLMs struggle with multi-step reasoning.

Key idea: show examples with intermediate reasoning steps.

Approach: instead of a direct answer the model is shown examples where reasoning is broken into explicit steps. For example when solving an arithmetic problem the initial state is specified first, then intermediate computations, and only then the final answer. This allows the model to think out loud and decompose complex tasks.

Results: GSM8K 17.9% to 58% (PaLM 540B), emergence (does not work on small models).

Discussion points: why this works (decomposition, working memory), zero-shot CoT ("Let's think step by step"), limitations (can generate plausible but incorrect chains).

### 11. ReAct: Reasoning and Acting (2022)

Problem: LLMs cannot interact with the external world.

Key idea: interleave reasoning (thought) and acting (tool use).

Format: the model alternates between three types of steps — Thought (reasoning about what to do next), Action (tool call such as search), and Observation (the result of the action). The cycle continues until a final answer is obtained. This structure allows the model to plan actions, obtain up-to-date information from the external world, and adjust its plan based on the data received.

Results: HotpotQA +6% over Chain-of-Thought, fewer hallucinations through grounding.

Discussion points: how to choose actions (prompted, fine-tuned), error recovery (what if the action fails), connection with AI agents.

### 12. Tree of Thoughts (2023)

Problem: Chain-of-Thought is a linear path, no backtracking.

Key idea: explore multiple reasoning paths, evaluate, backtrack.

Approach: Generate (several variants for the next step), Evaluate (assess each variant), Search (BFS or DFS over the tree).

Results: Game of 24 from 4% to 74%, creative writing significant improvement.

Discussion points: compute cost (many LLM calls), when justified (complex reasoning, high stakes), self-evaluation reliability.

### 13. DPO: Direct Preference Optimization (2023)

Problem: RLHF is complex — requires a separate reward model and PPO training.

Key idea: optimize the policy directly from preference data without an explicit reward model.

Mathematical formulation: the DPO loss function directly maximizes the difference in log-probabilities between the preferred and rejected responses, normalized relative to the reference model, using a sigmoid and a temperature parameter beta to control the strength of preferences.

Advantages: one stage instead of three, more stable than PPO, simpler to implement.

Discussion points: why the reference model pi_ref (regularization), comparison with RLHF on quality, limitations (requires paired preferences).

---

## RAG & Knowledge Papers

### 14. RAG: Retrieval-Augmented Generation (2020)

Problem: LLMs have fixed knowledge, expensive to update.

Key idea: retrieve relevant documents, condition generation on them.

Architecture: the user query is first processed by a retriever component (such as Dense Passage Retrieval), which finds the most relevant K documents from the knowledge base. Then these documents together with the query are fed into a generative model (such as BART), which produces the final answer taking the retrieved information into account.

Two modes: RAG-Sequence (one set of documents for the entire output), RAG-Token (different documents for different tokens).

Discussion points: retriever versus generator training (joint or separate), when RAG versus fine-tuning, faithfulness (how to ensure the answer comes from the documents).

Key insight: parametric knowledge (in weights) plus non-parametric (in retrieval) equals better than either alone.

---

## Safety & Alignment Papers

### 16. Training Language Models to Follow Instructions (InstructGPT)

Foundation for alignment — see section four above.

### 17. Anthropic's Papers on AI Safety

Red Teaming Language Models (2022): methodology for finding vulnerabilities, crowdsourced attacks.

Sleeper Agents (2024): whether a model can be trained to behave well but activate upon a trigger, deceptive alignment concerns.

Discussion points for Anthropic interviews: why alignment matters before creating AGI, the scalable oversight problem, interpretability as a path to safety.

---

## Checklist: Top 10 for Interviews

If time is limited, focus on these: Attention Is All You Need (the foundation of everything), GPT-3 (scaling, in-context learning), InstructGPT (RLHF, alignment), LoRA (efficient fine-tuning), Flash Attention (production optimization), Chain-of-Thought (reasoning), ReAct (agents), RAG (knowledge augmentation), DPO (modern alignment), Constitutional AI (safety).

---

## Typical Interview Questions

On specific papers: "Explain the attention mechanism in transformers", "How does RLHF work? What are its limitations?", "What's the difference between DPO and RLHF?", "How does RAG help with hallucinations?"

Comparative: "When would you use LoRA versus full fine-tuning?", "Compare CoT, ToT, and ReAct", "MoE versus dense models — trade-offs?"

Open-ended: "What's the most important paper from last year?", "What research directions excite you?", "How would you improve paper X?"

---

## Resources

Where to read papers: arXiv (primary source), Papers With Code (with code), Semantic Scholar (connected papers).

Paper discussions: ML Twitter/X, r/MachineLearning, YouTube (Yannic Kilcher, AI Coffee Break).

Reading groups: start your own with colleagues, one paper per week, discussion thirty to sixty minutes.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Interview Preparation
**Previous:** [[02_Coding_Exercises|Coding Exercises]]
**Next:** [[04_Behavioral_Staff_Plus|Behavioral for Staff+]]
