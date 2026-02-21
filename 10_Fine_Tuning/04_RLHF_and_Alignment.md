## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[03_Techniques_and_Evaluation|Techniques and Evaluation]]
**Next:** [[05_RLHF_Alternatives|RLHF Alternatives]]

---

# RLHF and Alignment

## Why Classical Fine-Tuning Is Not Enough

Supervised fine-tuning (SFT) teaches the model to imitate examples but not to understand what makes a response truly good. The model copies surface-level patterns, missing deeper quality.

RLHF approaches this differently: instead of showing "correct" answers, we show which answers people prefer. The model learns not to imitate but to optimize — generating responses that best match human preferences.

## RLHF Architecture

### Three Stages of Classical RLHF

**Stage 1: Supervised Fine-Tuning (SFT)**

Classical fine-tuning on high-quality examples. The model is trained to follow instructions and respond in the required format. The goal is to give the model an understanding of the task.

**Stage 2: Reward Model Training**

Training a reward model (RM) that learns to predict human preferences. The RM is a critic that can evaluate response quality with a single number (reward score).

The **Bradley-Terry model** models the probability of preferring A over B through the sigmoid of the difference between their reward scores. The loss function for the RM: the negative expected value of the log-sigmoid of the difference between the scores of the chosen and rejected responses.

**Stage 3: RL Optimization (PPO)**

Training the main model using the Reward Model as the reward function. The model generates responses, the RM evaluates them, and the model weights are adjusted based on the evaluations.

**PPO** uses two stabilization mechanisms:
- **Clipping** clips the probability ratio to the range [1-ε, 1+ε]
- **KL Penalty** penalizes deviation from the reference policy

### Failure Modes of RLHF

**Reward Hacking** — the model finds ways to achieve high reward without improving actual quality. Examples: verbosity bias, sycophancy, format hacking, hedging.

**Goodhart's Law** — the Reward Model is a proxy for human preferences. Optimizing the proxy diverges from the real objective.

**Mode Collapse** — loss of response diversity. The model finds one "success pattern" and uses it everywhere.

**Specification Gaming** — exploiting the gap between what we wanted to optimize and what we actually optimize.

**Mitigation:** KL-penalty, reward ensemble, iterated RLHF, red-teaming.

### Scale and Cost of RLHF

RLHF requires significant resources. Human annotation is expensive and slow. RL training is unstable and requires careful tuning.

Full-scale RLHF is typically performed only by major labs for frontier models (GPT-4, Claude, Gemini).

## DPO: Direct Preference Optimization

### Simplifying RLHF

DPO achieves RLHF results without explicitly training a Reward Model and without RL optimization.

Key insight: one can mathematically show that the optimal policy after RLHF has a specific form. DPO uses this relationship "in reverse" — directly training the model on preference data.

Instead of three stages (SFT → RM → RL), DPO requires only two: SFT followed by direct optimization on preference pairs.

### Advantages of DPO

**Simplicity** — no need to train a Reward Model. No unstable RL required.

**Stability** — supervised learning instead of RL.

**Efficiency** — 2 models instead of 4.

### Limitations of DPO

DPO assumes that preference data is "clean." In practice, preferences are noisy.

DPO is less flexible than RLHF with an explicit Reward Model.

## Beyond DPO: Modern Alternatives

**ORPO** combines SFT and preference alignment into a single training stage. It adds an odds ratio loss to the standard language modeling loss.

**KTO** is based on prospect theory. It works with individual examples labeled as "good" or "bad," without requiring pairwise comparisons.

**IPO** uses a quadratic loss form for stable gradients under strong preferences.

**SimPO** removes the reference model, adding length normalization and margin-based loss.

**GRPO (Group Relative Policy Optimization)** — introduced by DeepSeek in the R1 paper (January 2025), arguably the most important alignment innovation since DPO. GRPO eliminates both the reward model and the value model from PPO. For each prompt, GRPO samples K outputs from the current policy, computes rewards using a verifier (rule-based for math, format checks, or an LLM judge), then normalizes advantages within the group: Advantage_i = (reward_i - mean(rewards)) / std(rewards). The group statistics replace the critic/value function in PPO, dramatically simplifying training.

GRPO's significance extends beyond simplification. DeepSeek's **R1-Zero** experiment demonstrated that applying GRPO with only rule-based rewards (correctness, format compliance) to a base language model — without any SFT on reasoning traces — leads to spontaneous emergence of chain-of-thought reasoning, self-verification, and even "aha moments" of insight. This is a fundamental result: reasoning capabilities can emerge purely from proper RL incentives, without needing to be taught through demonstrations. R1-Zero showed that the capacity for structured reasoning is latent in pretrained language models and can be unlocked through appropriately designed reward signals.

## RLAIF: AI as Annotator

RLAIF replaces human annotators with an AI model. Instead of people, a powerful LLM compares responses.

**Constitutional AI** is a specific type of RLAIF where the AI evaluates responses for compliance with a set of principles — a "constitution."

Advantage: transparency and controllability through explicit formulation of principles.

## Key Takeaways

RLHF is a paradigm shift from imitation to optimization.

Classical RLHF: SFT → Reward Model → RL optimization.

Failure modes are critically important: reward hacking, Goodhart's Law, mode collapse, specification gaming.

DPO simplifies RLHF by eliminating the need for a separate RM and RL.

DPO alternatives expand the toolkit: ORPO, KTO, IPO, SimPO, GRPO.

GRPO (DeepSeek R1) eliminated the reward and value models from RL training, and demonstrated that reasoning can emerge from pure RL without supervised reasoning demonstrations.

RLAIF scales the process by replacing humans with AI.

For applied tasks, full-scale RLHF is excessive. Best-of-N, rejection sampling, DPO on synthetic data are more practical.

Alignment is broader than RLHF and includes architecture, prompts, and guardrails.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[03_Techniques_and_Evaluation|Techniques and Evaluation]]
**Next:** [[05_RLHF_Alternatives|RLHF Alternatives]]
