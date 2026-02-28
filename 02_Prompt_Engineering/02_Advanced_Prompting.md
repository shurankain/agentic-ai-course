# Advanced Prompting Techniques

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Prompt Engineering
**Previous:** [[01_Prompting_Basics|Introduction to Prompting]]
**Next:** [[03_Agent_Prompts|Prompts for AI Agents]]

---

## From Basic Techniques to Mastery

After mastering zero-shot, few-shot, and Chain-of-Thought, the question arises: how do you solve tasks where even CoT fails to produce reliable results?

Advanced techniques represent fundamentally different approaches inspired by cognitive science and decision theory. They enable: improving reliability through consensus, exploring alternative solutions, integrating reasoning with actions, and using the model to improve the prompts themselves.

---

## Self-Consistency: The Power of Consensus

### The Non-Determinism Problem

At non-zero temperature, a model generates different answers to the same question. **Self-Consistency turns this into an advantage**: if the majority of answers converge on the same result, it is likely correct. If they diverge, the task is ambiguous.

### How It Works

Three stages: the model generates several independent answers (with elevated temperature for diversity), the final result is extracted from each, and the most frequent answer is selected (majority voting).

Why does it work? Similarly to ensemble methods in machine learning. Different "reasoning paths" have different errors. If the errors are random and independent, they cancel each other out during voting.

### When to Apply

**Ideal scenarios**: math problems with a specific numerical answer, logic puzzles with a unique solution, classification with a small number of classes, medical or legal diagnosis where errors are costly.

**When NOT to apply**: text generation (diversity is an advantage), creative tasks with many equally valid solutions, brainstorming.

**Economics**: N requests (typically N=5-10) means N-fold cost increase. Apply only where the cost of error is high.

### Self-Consistency Math

**Majority Voting** — the most frequent answer is selected from N responses.

**Empirical confidence**: if all 5 answers match — high confidence; if 3 out of 5 — medium; if all different — low.

**Choosing N:** critical tasks 10-20, standard tasks 5-7, quick check 3. Important: doubling N does not double quality; improvement grows logarithmically.

---

## Tree of Thoughts: Exploring the Solution Space

### Limitations of Linear Reasoning

Chain-of-Thought reasons linearly. If an error is made or a suboptimal approach is chosen, there is no way to go back. **Tree of Thoughts (ToT) solves this** by transforming linear reasoning into exploration of a tree of possibilities.

### The Tree as a Thinking Structure

In ToT, each node is a "thought" or intermediate state. Branches extend from each node as alternative continuations. The model explores the tree, evaluating the promise of branches and pruning dead ends.

**Key components:**
- **Thought generation**: the model proposes several alternatives at each step
- **Evaluation**: each thought is assessed for its promise
- **Search**: BFS for exhaustive exploration, DFS for fast resolution, beam search for balance
- **Backtracking**: return to the parent node at a dead end

### Implementation Details

**Branch factor** determines the number of alternatives: 2-3 for an economical mode, 5-7 for a standard balance, 10+ for exhaustive search. Cost grows exponentially with depth.

**Pruning strategies**: value threshold (prune low-scoring branches), beam pruning (top-k best), depth limiting, early stopping (when a sufficiently good solution is found).

### Connection to Monte Carlo Tree Search

ToT is related to MCTS — an algorithm from game AI (AlphaGo). MCTS operates through four phases: Selection of a promising thought, Expansion — generating new reasoning steps, Simulation — playing out to an answer, Backpropagation — updating ancestor scores.

**UCB (Upper Confidence Bound)** balances exploitation (using what is known) and exploration (searching for new options), preventing premature convergence.

### Applying ToT

**When critically important**: tasks require exploring alternatives, multiple paths to the goal exist, errors at early stages are catastrophic.

**Domains**: strategic planning, puzzles and games, creative tasks with constraints, mathematical proofs, debugging complex problems.

**When excessive**: obvious linear path, quick answer needed, strict budget.

**Limitations**: cost (10-100+ requests), complexity (orchestration, state management), latency (minutes vs seconds).

**Note:** Reasoning models (o3, Claude extended thinking) perform ToT-like internal exploration within a single call — see the section below on "Chain-of-Thought Without Prompting." For most tasks, using a reasoning model eliminates the need for explicit ToT orchestration. Explicit ToT remains valuable when you need custom evaluation functions at each node or domain-specific search strategies.

---

## Chain-of-Thought Without Prompting: The o1 Era

Until 2024, Chain-of-Thought required explicit instructions. OpenAI o1/o3 changed the paradigm: **models are trained to think without being asked**. CoT is built in through specialized training.

### How It Works

**Training through RL**: standard RLHF trains "prompt → answer"; o1-style trains "prompt → reasoning → answer". Reasoning is optimized for final answer accuracy.

**Key differences**: CoT activates automatically, quality is optimized through RL, the model decides on its own when to think deeper. However: numerous internal iterations increase cost, the reasoning process is hidden, latency is higher.

**When to use**: o1 for complex mathematical tasks and reasoning, multi-step problems, high-stakes tasks. Standard models + explicit CoT for simple tasks, when control is needed, or budget is limited.

**Prompting o1**: paradoxically, less "engineering" is needed. CoT instructions are not required. What matters: clear task formulation, context and constraints, output format.

---

## ReAct: Merging Reasoning and Actions

### The Gap Between Thinking and Acting

Traditional LLMs work in a "think then answer" mode, but many tasks require interaction with the external world. Chain-of-Thought enables reasoning but not acting. Tool use enables acting but without deep reasoning.

**ReAct (Reasoning + Acting)** unifies both approaches: the model alternates between reasoning and actions, using the results of actions to adjust its reasoning.

### ReAct Structure

**Interaction cycle**: Thought (reasoning about the situation, planning a step) → Action (calling a tool with parameters) → Observation (the system returns a result) → Repeat (continues until solved).

**Key difference** from simple tool use: explicit "thoughts" between actions. The model explains why it selects a tool, what it expects to receive, and how it advances toward the goal.

### Why ReAct Is Effective

ReAct combines the advantages of both approaches. **From reasoning**: sequence planning, task decomposition, course correction, explainability. **From acting**: up-to-date information via search and databases, fact verification, precise computations, system interaction.

**Synergy**: reasoning guides actions, action results enrich reasoning. This enables solving tasks inaccessible to either pure reasoning or pure tool-use alone.

### ReAct in the Context of Agents

ReAct is the foundational pattern for AI agents. Modern frameworks (LangChain, LlamaIndex, AutoGPT) use variations of this pattern.

Understanding ReAct is critically important because it determines: system prompt structure, tool descriptions, interaction formatting, and error handling.

---

## Meta-Prompting: The Model Improves Prompts

### Prompts Writing Prompts

Meta-prompting means the language model creates and improves prompts. It works because LLMs were trained on a vast amount of text about prompt engineering, have seen thousands of examples, and know what makes a prompt effective.

### Prompt Generation

The simplest case: describe the task and ask the model to create a prompt to solve it. Why it works: LLMs have seen thousands of effective prompts and know best practices (clear instructions, examples, output format, edge cases, constraints).

**Process**: describe the task → model generates a prompt → test → identify problems → ask for improvement → repeat.

**When to apply**: unfamiliar domains (the model knows domain-specific patterns), many similar prompts (faster than manual), learning (analysis teaches best practices).

### Improvement and Self-Analysis

**Iterative improvement**: create a prompt → test → analyze errors → ask the model to improve → repeat. The key: show specific failures, not just "improve it."

**Self-analyzing prompts**: the model analyzes the task before solving it (type? approach? challenges?) → selects a strategy → solves → verifies.

---

## Role-Based Prompting: Expert Roles

### Activating Expert Knowledge

Role-Based Prompting is not simply "imagine you are an expert" but a detailed description of expertise, experience, and approach. Detail significantly improves quality: the model "tunes in," activating relevant knowledge.

**Role anatomy**: Expertise (specific domain, specialization, experience), Context (industry, systems, scale), Methodologies (expert approaches), Philosophy (Security-first vs Move fast), Communication style.

### Multi-Role Analysis

An advanced application — analyzing a task from multiple role-based perspectives. The same code is reviewed through the "eyes" of a security expert, performance engineer, and architect.

**Why multi-role is better**: a single prompt "check the code for everything" produces a superficial analysis. Specialized roles activate focus.

**Process**: define perspectives (security, performance, UX) → create a role for each → run the task separately → aggregate results.

**When to apply**: critical decisions (architecture, production code), important code review, design with competing constraints.

**Trade-off**: N requests, but significantly higher quality.

---

## Structured Output: Guaranteed Formats

### The Problem of Unpredictable Formats

The main problem when integrating LLMs into production is unpredictable output format. The model may add an introductory remark, change field order, or use different key names.

**Structured Output Prompting** is a set of techniques for predictable, parseable responses.

### Structuring Strategies

**JSON Schema** shows the exact structure (types, required fields, constraints). Models understand it well since they were trained on code.

**TypeScript interfaces** — an alternative that is more intuitive for developers.

**Structure examples** — the most reliable method; the model "sees" what is expected. Few-shot (2-3 examples) is even more effective.

**Explicit instructions**: "Return ONLY JSON without explanations", "Start with {", "No markdown". These significantly improve reliability.

**Combining** is the most effective approach: JSON Schema + example + explicit instructions create triple protection.

### Native Structured Outputs

Modern APIs (OpenAI, Anthropic) provide native support for structured outputs — the model is **guaranteed** to return valid JSON conforming to the schema. This is significantly more reliable than prompt-based approaches.

However, understanding prompt techniques remains important: not all models support native mode, sometimes formats other than JSON are needed, and the understanding helps diagnose problems.

---

## Prompt Chaining: Decomposing Complexity

### Limitations of a Single Prompt

Even a carefully constructed prompt has limitations: the context window is finite, model attention diffuses on long inputs, and some tasks are too complex for a single-shot solution.

**Prompt Chaining** breaks a complex task into a chain of simple prompts, where the output of one becomes the input of the next.

**Typical architecture** (document analysis): Extraction (extracting facts) → Analysis (analyzing patterns) → Synthesis (synthesizing conclusions) → Formatting (formatting the result). Each step is a separate prompt optimized for its task.

### Advantages of Chaining

**Modularity**: each step is optimized for its task, developed and tested independently.

**Debugging**: easy to find the problematic step and fix only that step.

**Cost optimization**: different models for different steps (cheaper ones for extraction, powerful ones for synthesis).

**Parallelism**: independent steps execute concurrently.

### Limitations of Chaining

**Error accumulation**: errors propagate and amplify from step to step.

**Latency**: sequential requests increase total time.

**Context loss**: each step sees only its input. Solution: include the original request in every step.

**When justified**: the task naturally decomposes, transparency is important, flexibility is needed.

**When excessive**: simple task, latency is critical, high risk of error accumulation.

---

## Key Takeaways

**Self-Consistency** turns non-determinism into an advantage through majority voting (for tasks with a single correct answer).

**Tree of Thoughts** explores the solution space with backtracking (when linear reasoning is insufficient).

**ReAct** unifies reasoning and acting (the foundation for AI agents).

**Meta-Prompting** uses the model to create and improve prompts (automating iterations).

**Role-Based Prompting** activates expert knowledge through detailed roles.

**Prompt Chaining** decomposes complex tasks into manageable steps (modularity, debugging).

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Prompt Engineering
**Previous:** [[01_Prompting_Basics|Introduction to Prompting]]
**Next:** [[03_Agent_Prompts|Prompts for AI Agents]]

---

## Practical Prompt Examples

### ReAct Prompt: Reasoning and Action Cycle

**Description:** ReAct demonstrates alternation of thoughts, actions, and observations. The model explicitly explains its reasoning before each action.

```text
Task: Find out the current temperature in Kyiv and advise whether a jacket is needed.

Available tools:
- search(query: str): information lookup
- calculate(expression: str): calculations

Format:
Thought: [reasoning about the situation and next step]
Action: [tool(parameters)]
Observation: [result - filled by the system]

Thought: I need to find out the current temperature in Kyiv. I will use search.
Action: search("current temperature Kyiv")
Observation: The temperature in Kyiv is currently +8°C, cloudy, wind 5 m/s.

Thought: At +8°C with wind, a jacket is definitely needed. I can provide a recommendation.
Action: Final answer
Observation: It is +8°C in Kyiv with 5 m/s wind. I recommend taking a jacket, preferably a windproof one.
```

---

### Tree of Thoughts Prompt: Exploring Alternatives

**Description:** ToT allows generating multiple approaches, evaluating them, and selecting the optimal path with the ability to go deeper.

```text
Task: Optimize a slow SQL query on a table with 10M records.

Steps:
1. Generate 3 optimization approaches
2. Rate each on a scale of 1-10
3. Select the best and explain why
4. Expand into a detailed plan

--- Generating alternatives ---

Thought A: Adding indexes
- Composite indexes on WHERE/JOIN columns
- Rating: 8/10 (effective, low risk, simple)

Thought B: Denormalization
- Materialized views with aggregates
- Rating: 7/10 (effective, but complicates synchronization)

Thought C: Partitioning
- Splitting by date/geography
- Rating: 9/10 (maximum impact, requires migration)

--- Selection ---

Selecting C (Partitioning) - 9/10
Reason: Will reduce data scanning by 10-100x

--- Detailed plan ---

Partitioning:
  1. Key: RANGE by date (covers 80% of queries)
  2. Migration: new table → batch transfer → blue-green switch
```

---

### Self-Consistency Prompt: Consensus Through Voting

**Description:** Self-Consistency generates several independent solutions and selects the answer through majority voting.

```text
Task: Maria has 3 more apples than Peter. Peter has 2 times fewer apples than Anna.
There are 27 apples in total. How many does each person have?

Instruction: Solve in 5 different ways, then select the consensus.

--- Solution 1: System of equations ---
M = P + 3, P = A/2, P + M + A = 27
P + (P + 3) + 2P = 27 → 4P = 24 → P = 6
Answer: Peter = 6, Maria = 9, Anna = 12

--- Solution 2: Starting from Anna ---
A + A/2 + A/2 + 3 = 27 → 2A = 24 → A = 12
Answer: Peter = 6, Maria = 9, Anna = 12

--- Solution 3: Starting from Peter ---
P + (P + 3) + 2P = 27 → 4P = 24 → P = 6
Answer: Peter = 6, Maria = 9, Anna = 12

--- Solution 4: Substitution ---
P = 5: total = 23 ✗
P = 6: total = 27 ✓
P = 7: total = 31 ✗
Answer: Peter = 6, Maria = 9, Anna = 12

--- Solution 5: Via base variable ---
x + (x + 3) + 2x = 27 → 4x = 24 → x = 6
Answer: Peter = 6, Maria = 9, Anna = 12

--- Consensus ---
Result "P=6, M=9, A=12": 5/5 solutions
Confidence: HIGH (100% agreement)
```

---

### Practical Application

**ReAct** — for tasks involving external tools where reasoning transparency is needed
**Tree of Thoughts** — when alternatives exist and errors at early stages are critical
**Self-Consistency** — for tasks with a definitive answer and a high cost of error
