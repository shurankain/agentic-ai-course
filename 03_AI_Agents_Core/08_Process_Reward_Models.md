# Process Reward Models: Evaluating Reasoning Step by Step

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[07_Code_Generation_Agents|Code Generation Agents]]
**Next:** [[09_Agent_Use_Cases|Practical Use Cases]]

---

## Introduction

When a human solves a complex problem, they don't simply produce an answer — they go through a series of intermediate steps. Each step can be correct or erroneous, and an error at an early stage often leads to an incorrect final result. Traditional evaluation systems look only at the final answer: correct or not. But what if we could evaluate each reasoning step?

This is exactly the idea behind Process Reward Models (PRM) — models that evaluate not only the final result but also the quality of each intermediate step. This is a fundamental shift in how we train and guide AI agents.

Consider a math teacher grading a student's work. A poor teacher looks only at the answer: "42 — wrong, you fail." A good teacher analyzes the solution: "The first three steps are correct, but on the fourth you flipped the sign. Fix that, and the answer will be correct." PRM is that good teacher for AI.

---

## Two Approaches to Evaluation: ORM vs PRM

### Outcome Reward Models (ORM)

Outcome Reward Models represent the traditional approach inherited from classical RLHF. An ORM takes a problem and a complete solution as input and returns a single number — a quality score for the result.

Formally, for a problem x and solution y:

$$R_{ORM}(x, y) \rightarrow \mathbb{R}$$

ORM is trained on (solution, score) pairs, where the score reflects the correctness of the final answer. The model learns to predict whether a given reasoning chain will lead to a correct result.

The advantages of ORM are clear: simplicity of data collection (only the answer needs checking), a straightforward success metric, and easy integration into existing RLHF pipelines.

But ORM has fundamental limitations. The model receives only a sparse signal — a single score for the entire solution. If the solution consists of ten steps and the error occurred at step three, ORM won't indicate where exactly. All steps receive the same signal, even though seven of them were correct.

This leads to the credit assignment problem: how do you distribute "blame" for an incorrect answer across all steps? In theory, with enough data, gradients will find the correct distribution. In practice, this requires enormous datasets and does not guarantee correct learning of intermediate steps.

### The Credit Assignment Problem: A Deeper Look

Credit assignment is a fundamental problem in reinforcement learning, first formalized by Minsky in 1961. In the context of LLM reasoning, it is especially acute.

**Formalizing the problem:**

Suppose a solution consists of n steps: s₁, s₂, ..., sₙ. We receive reward R only at the end (0 or 1 for answer correctness). The task: evaluate the contribution of each step to success.

Consider a situation: Step 1 and Step 2 are perfectly correct, Step 3 contains a critical error (but we don't know this), Step 4 logically follows from Step 3 and is technically correct relative to the incorrect premise, Step 5 gives an incorrect final answer. Result: R = 0 (failure).

All steps receive the same failure signal. But steps 1-2 were correct, step 3 was the critical error, step 4 was a dependent error. A model without step-level evaluation cannot distinguish between these categories.

**Classical approaches to credit assignment:**

| Approach | Idea | Problem for LLM reasoning |
|----------|------|---------------------------|
| **Temporal difference** | Reward differs from bootstrap estimate | Requires a value function for intermediate states |
| **Monte Carlo** | Averaging over multiple trajectories | High variance, requires many samples |
| **Eligibility traces** | Exponential decay of credit | Does not account for step semantics |
| **Hindsight credit** | Analyzing step's influence on outcome | Requires counterfactual analysis |

**LLM reasoning specifics:**

Unlike classical RL (where states are physical), in LLM reasoning:
- Steps are textual, with rich semantics
- "State" is the entire accumulated context
- Errors can be subtle (misinterpretation rather than an obvious mistake)
- A correct step can lead to an incorrect answer (and vice versa)

PRM addresses this problem by training a separate model to evaluate each step. But collecting data for this is a separate challenge.

### Process Reward Models (PRM)

Process Reward Models solve the credit assignment problem directly: they evaluate each reasoning step independently.

For a problem x and a sequence of steps s₁, s₂, ..., sₙ:

$$R_{PRM}(x, s_1, ..., s_i) \rightarrow \mathbb{R}$$

PRM returns a score for each solution prefix: how good is the reasoning path up to this point? This is a dense reward signal — detailed feedback at every step.

The key advantage: if the model made an error at step 3, PRM can give high scores to steps 1-2 and a low score to step 3. This enables:
- Precise error localization
- Preserving correct reasoning patterns
- More efficient use of training data
- Guiding inference-time search (selecting the best branches in ToT/MCTS)

### Mathematics: From ORM to PRM

The relationship between ORM and PRM can be expressed formally. If we have a PRM providing scores for each step, ORM can be obtained as an aggregation:

$$R_{ORM}(x, y) = f(R_{PRM}(x, s_1), R_{PRM}(x, s_1, s_2), ..., R_{PRM}(x, s_1, ..., s_n))$$

Where f can be:
- **Minimum**: R_ORM = min(r₁, r₂, ..., rₙ) — the solution is good only if all steps are good
- **Product**: R_ORM = ∏rᵢ — multiplicative aggregation
- **Last step**: R_ORM = rₙ — evaluation of the final state
- **Weighted sum**: R_ORM = Σwᵢrᵢ — with increasing weight for later steps

The reverse transformation (from ORM to PRM) is impossible without additional information — this is precisely why PRM requires specialized annotation.

---

## Data Collection for PRM

### The Annotation Challenge

The main difficulty with PRM is data collection. For ORM, it is sufficient to check the final answer (often automatically). For PRM, each step must be evaluated, which requires:
- Domain expertise
- Understanding of the full solution context
- Significant time investment

Consider a math problem with 8 steps. For ORM, an annotator checks the answer in seconds. For PRM, they must analyze the logic of each step — this takes minutes or even hours for complex problems.

### Approaches to PRM Annotation

**Human annotation** is the gold standard. Experts label each step as correct/incorrect/neutral. OpenAI used exactly this approach for mathematical problems in their paper "Let's Verify Step by Step."

Annotation protocol:
1. Break the solution into atomic steps
2. For each step, determine: is it logically correct given the previous steps?
3. Do not consider whether the step leads to the correct answer — only local correctness

#### Human Annotation Protocol in Detail

OpenAI in "Let's Verify Step by Step" (2023) used a three-level evaluation scheme where each step is marked with one of three symbols: **+** (positive) means the step is correct and useful for the solution; **-** (negative) indicates the step contains an error; **=** (neutral) is used when the step adds no information but contains no errors, or when the step depends on a previous error.

**Instructions for annotators:**

1. **Read sequentially** — evaluate each step in the context of previous steps, without looking ahead
2. **Ignore the final result** — a step can be correct even if the final answer is wrong
3. **Local correctness** — verify only the current logical transition
4. **Mark the first error** — if step 3 is erroneous, steps 4-N automatically receive neutral (they depend on the error)

**Common annotation mistakes:**
- **Hindsight bias**: knowledge of the correct answer influences step evaluation
- **Inconsistent granularity**: different annotators define step boundaries differently
- **Domain expertise gaps**: mathematical reasoning requires competence

**Quality control:**
- Multiple annotators per sample (3+)
- Inter-annotator agreement metrics (Cohen's κ)
- Expert review for disagreements
- Calibration sessions for consistency

**Synthetic labeling** is automatic annotation using a strong model. GPT-4o or Claude evaluate solution steps. This is cheaper but introduces the teacher model's bias.

**Synthetic labeling variants:**

| Method | Description | Trade-off |
|--------|-------------|-----------|
| **Direct prompting** | "Rate this step from 0 to 1" | Fast but noisy |
| **Chain-of-thought critique** | "Explain whether there is an error in this step" | More accurate but more expensive |
| **Contrastive pairs** | "Which step is better: A or B?" | Relative scores |
| **Error injection** | Generate erroneous steps, model learns to distinguish | Controlled quality |

**Outcome-based labeling** is a clever heuristic. The idea: if continuations after step k frequently lead to correct answers, step k is probably good. Formally:

$$label(s_k) = \mathbb{E}[\text{correct}(completion) | s_1, ..., s_k]$$

This is a Monte Carlo estimate of step quality through multiple rollouts. The method does not require explicit step annotation but requires substantial computation for sampling continuations.

### Best-of-N Sampling for Evaluation

A practical method for evaluating PRM quality is Best-of-N (BoN) sampling:
1. Generate N solutions for a problem
2. Evaluate each solution using PRM
3. Select the solution with the best score
4. Check whether the selected solution gives the correct answer

Metric: BoN accuracy — the proportion of problems where the PRM-best solution gives the correct answer.

BoN demonstrates the practical value of a reward model: how well it can guide selection among alternative solutions.

---

## PRM in Inference-Time Search

### Integration with Tree-of-Thought

PRM integrates naturally with tree-based solution search. At each tree node, PRM evaluates the current reasoning path, enabling:
- Pruning clearly unpromising branches
- Directing the search toward better continuations
- Ranking final solutions

In the context of ToT with beam search, PRM replaces heuristic node evaluation: for each node in the beam, the PRM score is computed from the problem and the path to that node, then the beam is narrowed to the top-K nodes by this score. This provides a theoretically grounded value function instead of ad-hoc evaluation prompts.

### Integration with MCTS

In MCTS, a reward model can be used at several stages:

**Leaf node evaluation**: instead of a full rollout to the final answer, PRM evaluates the current reasoning state.

**Guiding expansion**: when generating child nodes, candidates are ranked by PRM score.

**Early stopping**: if PRM gives a very low score to a path, its exploration can be terminated.

The UCB formula is modified to incorporate PRM:

$$UCB_{PRM}(i) = \frac{Q(i) + \alpha \cdot PRM(state_i)}{N(i)} + c \sqrt{\frac{\ln N(parent)}{N(i)}}$$

Where α controls the influence of PRM relative to the empirical estimate through rollouts.

### Connection to o1-Style Reasoning

Models like o1 use inference-time compute to improve reasoning. Internally, this may involve:
- Generating multiple reasoning paths
- Evaluating intermediate steps (implicit PRM)
- Backtracking upon error detection
- Selecting the best path for the final answer

PRM can be viewed as an explicit, separate model for what o1-style models do implicitly. The advantage of an explicit PRM is interpretability and controllability.

### Reasoning Models and PRM Training (2024-2025)

The release of o3, o4-mini, and DeepSeek R1 in 2024-2025 confirmed the central role of process-level reward signals in training reasoning models.

**DeepSeek R1 and GRPO:** DeepSeek R1 demonstrated that Group Relative Policy Optimization (GRPO) — which compares multiple reasoning trajectories and rewards those leading to correct outcomes — functions as an implicit form of process reward. The R1-Zero experiment showed that reasoning behaviors (self-verification, backtracking, exploration) emerge purely from outcome-based RL, but the group comparison mechanism provides denser signal than simple ORM. This blurs the boundary between ORM and PRM: GRPO uses outcome rewards but applies them at a trajectory level, providing richer credit assignment than single-score ORM.

**o3 and tool-use integration:** OpenAI's o3 model integrates tool use (code execution, web search) directly into the reasoning chain. This creates a new dimension for process reward: not just "is this reasoning step correct?" but "was this tool call well-chosen and well-timed?" Training models to evaluate tool-use decisions within reasoning chains requires PRM-like signals that understand both logical correctness and tool appropriateness.

**Programmatic verification as PRM:** For domains with verifiable outcomes (math, code, logic), programmatic verifiers serve as automated process reward signals. Instead of training a neural PRM, the system executes the code or checks the proof at each step. DeepSeek R1's training pipeline uses execution-based verification for code and math tasks — the reward signal is dense (per-problem) and perfectly accurate (no reward model bias). This is effectively a "perfect PRM" for verifiable domains.

**Practical decision framework:**

| Approach | Best For | Advantages | Limitations |
|----------|----------|------------|-------------|
| **Programmatic verification** | Math, code, logic, formal proofs | Perfect accuracy, no reward bias, scales cheaply | Only works for verifiable domains |
| **GRPO-style trajectory comparison** | General reasoning, open-ended tasks | No separate reward model needed, captures emergent reasoning patterns | Requires sampling many trajectories per prompt, high training compute |
| **Explicit neural PRM** | Nuanced domains (legal, medical, scientific reasoning) | Evaluates subjective reasoning quality, fine-grained step feedback | Expensive to train, requires expert step-level annotations |

**Decision rule:** Start with programmatic verification if your domain allows it — it is the cheapest and most reliable option. Use GRPO when you need general reasoning improvement and can afford the training compute. Reserve explicit PRM for domains where reasoning quality is subjective and cannot be verified programmatically.

---

## Verification of Reasoning Steps

### Formal Verification vs Neural Evaluation

For some domains, formal verification of steps is possible. Mathematical proofs can be checked by proof assistants (Lean, Coq). Code can be verified through tests and static analysis.

But most reasoning does not lend itself to formal verification: natural language reasoning, commonsense inference, planning under uncertainty. Here, a neural PRM is the only practical option.

### What Does a "Correct Step" Mean?

The definition of step correctness is non-obvious. Consider the variants:

**Logical correctness**: the step follows from previous steps by the rules of logic. This is a strict criterion, but natural language reasoning is rarely strictly logical.

**Factual correctness**: the step contains accurate information. Requires external knowledge for verification.

**Progress toward the goal**: the step brings us closer to solving the problem. But this requires knowledge of the correct solution.

**Absence of errors**: the step contains no obvious errors (arithmetic, logical). A minimal criterion.

In practice, PRM is trained on a mixture of these criteria, inherited from the data annotation.

### PRM Calibration

It is important for PRM scores to be calibrated: if PRM gives a step a score of 0.8, it should mean that approximately 80% of such steps are actually correct.

Calibration is checked via a reliability diagram:
1. Group steps by predicted scores (0-0.1, 0.1-0.2, ...)
2. For each group, compute the proportion of actually correct steps
3. Compare the predicted and actual proportions

A poorly calibrated PRM can be overconfident (inflates scores) or underconfident (deflates scores). Both problems are harmful when used in search.

---

## Reward Hacking in Agents

### Nature of the Problem

Reward hacking is a situation where a model maximizes reward in a way other than intended. This is a fundamental RL problem that manifests in the PRM context as well.

Examples of reward hacking with PRM:

**Stylistic hacking**: PRM may prefer a particular writing style (verbose explanations, specific phrases) unrelated to correctness. The model learns to imitate this style instead of improving reasoning.

**Exploitation of biases**: if correct solutions in the training data more often begin with "Let me think step by step," the model may over-rely on this phrase.

**Gaming the decomposition**: the model may learn to break a solution into many trivial steps, each receiving a high score, without improving the actual solution.

### Connection to Goodhart's Law

Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."

In the PRM context:
- We create PRM as a proxy for "reasoning quality"
- We optimize the model to maximize the PRM score
- The model finds ways to maximize the score without improving actual quality
- PRM ceases to be a good measure of quality

This is an inevitable dynamic when optimizing against any learned reward function.

### Mitigating Reward Hacking

**Conservative optimization**: avoid optimizing reward too aggressively. Add a KL-divergence penalty to the original model (as in PPO).

**Ensemble PRM**: use multiple PRMs trained on different data. Reward hacking of one PRM is unlikely to transfer to others.

**Regular PRM updates**: periodically retrain PRM on data that includes outputs from the optimized model. This "closes the gaps" that the model learned to exploit.

**Human-in-the-loop**: regularly check samples of highly scored solutions for actual correctness.

**Adversarial evaluation**: proactively search for cases where a high PRM score does not correspond to quality.

---

## PRM vs ORM: When to Use Which

### Selection Criteria

**Use ORM when:**
- Step annotation is too expensive or impossible
- The task does not decompose into clear steps
- A quick baseline is needed
- Data is limited, and ORM will train better than PRM

**Use PRM when:**
- The task has a clear step-by-step structure (math, programming, logic)
- Step annotation is available (human or synthetic)
- Inference-time search guidance is needed
- Error localization is important for model improvement

### Hybrid Approaches

In practice, PRM and ORM are often combined:

**Two-stage filtering**: first, ORM filters out clearly poor solutions (fast), then PRM ranks the remaining ones (precise).

**Weighted combination**: final score = α·ORM + β·PRM.

**Cascaded verification**: PRM checks steps, ORM verifies the final answer.

### Computational Considerations

ORM requires one forward pass for the entire solution. PRM requires a forward pass for each step (or a specialized architecture for incremental evaluation).

For a solution with N steps:
- ORM: O(1) forward passes
- Naive PRM: O(N) forward passes
- Optimized PRM (caching): ~O(1) with the right architecture

This matters for inference-time search, where many paths need to be evaluated.

---

## Connection to RLHF

PRM is an extension of RLHF ideas (see [[../10_Fine_Tuning/04_RLHF_and_Alignment|RLHF and Alignment]]) to the level of individual reasoning steps. Understanding this connection helps situate PRM within the evolution of alignment methods.

### Evolution of Reward Signals in LLMs

The evolution of reward signal granularity in language model training has gone through four main levels, each increasing the granularity of feedback:

**Level 1: Response-level** — classical RLHF evaluates the entire response with a single score. This is a simple approach, but it suffers from the sparse reward problem: the model receives one number for the whole response, without knowing which specific parts are good or bad.

**Level 2: Segment-level** — the response is divided into meaningful segments, each evaluated separately. This allows, for example, evaluating "helpfulness" and "safety" of different response parts independently. Multi-aspect evaluation provides more detailed feedback.

**Level 3: Step-level** — Process Reward Models evaluate each reasoning step. This is a qualitative leap for tasks requiring step-by-step solutions: math, programming, logical deductions. Each step receives an independent correctness score.

**Level 4: Token-level** — the research frontier, where credit is distributed at the individual token level. Theoretically this is maximum granularity, but practical implementation is extremely difficult due to the enormous dimensionality of the annotation space and computational costs.

Each subsequent level requires exponentially more annotation effort but potentially provides more precise feedback for training.

### From Preference Learning to Step Verification

In classical RLHF, the reward model is trained on pairwise comparisons of complete responses: "response A is better than response B." PRM generalizes this to: "step A is better than step B in a given context."

**Conceptual comparison:**

| Aspect | RLHF Reward Model | Process Reward Model |
|--------|-------------------|---------------------|
| **Unit of evaluation** | Complete response | Single reasoning step |
| **Training signal** | Pairwise comparisons | Absolute/pairwise step scores |
| **Use in optimization** | PPO/DPO loss | PPO/DPO + inference-time search |
| **Interpretability** | "Response is good/bad" | "Step 3 is an error" |
| **Data efficiency** | N comparisons per response | ~N×K comparisons (K steps) |

Training methods are analogous:
- **Bradley-Terry model** for pairwise comparisons
- **Cross-entropy loss** for categorical labels (correct/incorrect)
- **Regression** for numerical quality scores

### PRM in the RLHF Pipeline

PRM can replace or supplement the reward model in PPO/DPO:

**Replacement**: R(x, y) = aggregate(PRM(x, s₁), ..., PRM(x, sₙ))

**Supplement**: R(x, y) = ORM(x, y) + λ·process_bonus(PRM scores)

The process bonus can reward the model for correct intermediate steps, even if the final answer is wrong.

### Iterative Improvement

PRM enables iterative improvement:
1. Train the model on existing data
2. Generate solutions with inference-time search (guided by PRM)
3. Obtain annotations for new steps
4. Fine-tune PRM on expanded data
5. Fine-tune the base model with the improved PRM
6. Repeat

This creates a flywheel effect: better PRM → better search → better data → even better PRM.

---

## Key Takeaways

1. **PRM evaluates the process, not just the result.** A dense reward signal at each step solves the credit assignment problem and enables precise error localization.

2. **Data collection for PRM is expensive, but alternatives exist.** Human annotation is the gold standard, but synthetic labeling and outcome-based estimation reduce cost.

3. **PRM is critical for inference-time search.** ToT, MCTS, and other search methods gain a theoretically grounded evaluation function instead of heuristics.

4. **Reward hacking threatens any learned reward function.** Conservative optimization, ensembles, and regular checks are mandatory practices.

5. **ORM and PRM complement each other.** The choice depends on task structure, data availability, and computational budget.

6. **PRM is a natural extension of RLHF.** The same principles of preference learning are applied at the level of individual reasoning steps.

7. **O1-style reasoning uses implicit PRM.** DeepSeek R1's GRPO provides trajectory-level process signals. o3 integrates tool-use evaluation into the reasoning chain.

8. **Programmatic verifiers are "perfect PRMs" for verifiable domains.** Code execution, math checking, and proof verification provide dense, unbiased reward signals — replacing neural PRMs where possible.

9. **PRM calibration is critical.** Uncalibrated scores harm search and complicate interpretation of results.

---

## Practical Code Examples

### Basic PRM Implementation

The core idea behind implementing a Process Reward Model comes down to iteratively evaluating each solution step in the context of previous steps. In practice, this means:

**Step evaluation system architecture:**
- The model receives the problem and the current solution prefix (all steps up to the current one)
- For each new step, a prompt is formed with context: the original problem + all previous steps + the current step
- The evaluator language model analyzes the local correctness of the step: whether it is logically valid given everything preceding it
- A numerical score (typically from 0.0 to 1.0) and optionally a textual explanation are returned
- The process repeats for each subsequent step, gradually accumulating context

**Aggregation of step-level scores:**
After obtaining scores for all steps, they need to be reduced to a single score for the entire solution. Several strategies exist:
- **Minimum** — the solution is only as good as its worst step; one error nullifies everything
- **Product** — multiplicative combination, where each weak step proportionally reduces the overall score
- **Weighted sum** — later steps may receive greater weight since they are closer to the final answer
- **Last step** — evaluation of the final state, assuming all information is accumulated in it

The choice of strategy depends on the domain: for math, minimum is reasonable (one error ruins everything); for creative tasks, a weighted sum may be better.

**Integration with search:**
PRM becomes the evaluation function in search algorithms. In beam search, at each iteration:
- k possible continuations are generated for each path in the beam
- Each continuation is evaluated by PRM
- The top-N paths by score are selected
- The process repeats until a final answer is reached

In MCTS, PRM scores are used to balance exploration/exploitation in the UCB formula, directing the search toward more promising branches of the reasoning tree.

### Brief Example: Evaluating a Single Step

**ProcessRewardModel structure:**

The ProcessRewardModel class encapsulates the logic of step-by-step reasoning evaluation. As a dependency, it contains an instance of ChatLanguageModel — the evaluator language model that will analyze step correctness.

The StepScore record represents the evaluation result of a single step: step number (stepIndex), step text (step), numerical score (score) from 0.0 to 1.0, and a textual explanation (explanation).

The evaluateStep method takes four parameters: problem (the original problem), previousSteps (all previous reasoning steps), currentStep (the current step being evaluated), and stepIndex (the step number). Inside the method, a prompt is formed using a multi-line text block. The prompt is structured: first, the task of evaluating step correctness is stated, then the original problem is provided, followed by all reasoning up to the current point, and the current step itself. The model receives the questions: "Is this step logically correct? Does it contain errors?" and an instruction to return a score from 0.0 to 1.0. The prompt is formatted with parameter substitution via formatted(). The prompt is then sent to the evaluator model via evaluator.generate(), the response is parsed by the parseStepScore method which extracts the numerical score and explanation, and the result is returned as a StepScore.

The aggregateMinimum method demonstrates one strategy for aggregating step-level scores into a single score for the entire solution. It takes a list of StepScore, creates a stream, maps each element to its numerical score via mapToDouble, finds the minimum value via .min(), and returns it (or 0.0 if the list is empty). This strategy implements the principle "a solution is only as good as its worst step" — one critical error nullifies the quality of the entire solution.

**Reward hacking detector:**
Reward hacking detection systems work on the principle of comparing two independent evaluations. The core idea: if a model has learned to exploit PRM biases, then the PRM score will be substantially higher than the score from an independent verifier. The detector analyzes the discrepancy between scores and looks for characteristic exploitation patterns: excessive use of certain phrases, artificial fragmentation into trivial steps, repetition of identical constructions. When the discrepancy exceeds a threshold (e.g., 0.3), the system flags the solution for manual review.

**Outcome-based labeling:**
A method for automatic step annotation without human involvement. For each step k in the solution, the algorithm:
1. Takes the solution prefix up to and including step k
2. Generates N different solution completions from this prefix (Monte Carlo rollouts)
3. Checks how many of these completions lead to the correct answer
4. Assigns step k a score = proportion of successful completions

The intuition: if from the prefix "problem → step1 → step2 → step3" it is often possible to reach the correct answer, then step3 is probably correct. If most completions lead to errors, step3 contains a problem. The method is computationally expensive (requires extensive sampling) but does not need human annotation.

**PRM calibration:**
An uncalibrated model may systematically inflate or deflate scores. Calibration is checked via a reliability diagram: group steps by predicted scores (0-0.1, 0.1-0.2, ..., 0.9-1.0) and for each group check what proportion of steps are actually correct. For a calibrated model, the lines should coincide: if the model gives 0.7, then approximately 70% of such steps should be correct.

Temperature scaling is a post-processing technique for improving calibration: divide the model's logits by a temperature T before applying sigmoid/softmax. The optimal temperature is found through grid search, minimizing negative log-likelihood on a validation set. T > 1 makes the model less confident (smooths the distribution), T < 1 makes it more confident.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[07_Code_Generation_Agents|Code Generation Agents]]
**Next:** [[09_Agent_Use_Cases|Practical Use Cases]]

**Related materials:**
- [[../10_Fine_Tuning/04_RLHF_and_Alignment|RLHF and Alignment]]
- [[../01_LLM_Fundamentals/08_Scaling_Laws|Inference-time compute and o1-style reasoning]]
