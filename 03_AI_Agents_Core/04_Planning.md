# Agent Planning

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[03_Tool_Use|Tool Use]]
**Next:** [[05_Memory_Systems|Agent Memory Systems]]

---

## Introduction

Planning is the capability that separates primitive reactive systems from genuinely intelligent agents. A simple agent reacts to a stimulus, performs an action, and moves on to the next stimulus. A planning agent sees further: it analyzes the task, breaks it into steps, anticipates obstacles, and builds a strategy for reaching the goal.

Consider the difference between a navigator that shows only the next turn and one that builds a complete route accounting for traffic, fuel stops, and arrival time. Both perform navigation, but the second provides a qualitatively different level of utility.

In the world of AI agents, planning enables solving complex multi-step tasks that are beyond the reach of simple ReAct agents. This chapter covers how agents learn to think ahead.

---

## Task Decomposition

### From Complex to Simple

Decomposition is the art of breaking a complex task into manageable subtasks. This is the first and arguably the most important step of planning. A task that seems insurmountable as a whole becomes feasible when broken down into a sequence of simple steps.

Consider the example: "Analyze competitors and prepare a report." This is a vague, high-level task. How should it be executed? A novice agent would attempt everything at once and get confused. An experienced agent first decomposes the task:
1. Identify the list of key competitors
2. Gather information on each competitor
3. Analyze strengths and weaknesses
4. Compare against our product
5. Formulate recommendations
6. Prepare the final report

Each step is now concrete and actionable. Moreover, dependencies become visible: you cannot analyze competitors without first gathering information about them; you cannot formulate recommendations without performing a comparison.

### Principles of Effective Decomposition

Good decomposition follows several principles.

**Atomicity**: each subtask should be simple enough to be completed in a single step or with a minimal number of actions. If a subtask is still too complex, it needs to be broken down further.

**Independence**: where possible, subtasks should be independent of each other. This allows parallel execution and simplifies error recovery. If task A does not depend on task B, they can be executed simultaneously.

**Logical sequence**: steps should follow the order in which they are naturally performed. You cannot put the cart before the horse. Each step must have all necessary input data from previous steps.

**Measurability of results**: for each subtask, it should be clear what constitutes successful completion. "Gather information on competitors" — when is this considered done? When information on five competitors has been collected? Ten? When the information includes specific data points?

### Complexity Assessment

Not all subtasks are equal. Some are trivial, others require significant effort. Complexity assessment helps allocate resources properly and set realistic expectations.

Complexity can be measured on various scales: the number of tools required, the volume of data to process, the probability of errors, the execution time. An agent can use these estimates for prioritization: start with simple tasks for quick wins, or conversely, tackle complex tasks while resources are available.

---

## Hierarchical Planning

### Multi-Level Structure

Hierarchical planning organizes tasks into multiple levels of abstraction. At the top level are global goals. Each goal is broken down into subgoals. Subgoals are broken down into concrete actions.

This is similar to military strategy. The commander-in-chief defines the strategic objective: "Capture the city." Generals develop the operational plan: "Surround the city, block supply lines, conduct the assault." Officers plan tactical actions: "First company takes position X, second company provides cover."

Each level operates at its own scale and in its own language. The commander-in-chief does not think about every soldier's position. A soldier does not think about the campaign's global strategy. But all levels are aligned and work toward a common goal.

### Advantages of Hierarchy

Hierarchical planning provides several advantages.

**Complexity management**: at each level, only a limited number of elements are considered. Instead of a hundred actions — ten subgoals, each containing ten actions.

**Flexibility**: if the plan at a lower level does not work, only that level needs changing without affecting upper levels. If a tactic fails, a different tactic can be tried within the same strategy.

**Reusability**: standard subgoals and plans can be used in different contexts. "Gather information about a company" is a standard subtask applicable to many situations.

**Explainability**: hierarchy makes the plan understandable to humans. The logic can be explained at any level of detail.

### Dynamic Refinement

A hierarchical plan does not have to be built entirely in advance. Upper levels can be defined at the start, while lower levels are refined during execution.

This is particularly useful when complete information is unavailable at the outset. The agent knows it needs to "gather information about competitors," but the specific sources and methods will be determined during the process.

Dynamic refinement saves resources: there is no need to plan in detail steps that may change. However, it requires the ability to revise and augment the plan on the fly.

### Hierarchical Task Networks (HTN): A Formal Perspective

HTN is a formalism for hierarchical planning that divides tasks into two types:

**Primitive tasks** — directly executable actions that correspond to tool calls. They have preconditions (what must be true before execution) and effects (what changes after execution). Examples: send an email, search the web, read a file.

**Compound tasks** — high-level abstractions that are not executed directly but decomposed into subtasks. For each compound task, several decomposition methods are defined — different ways to accomplish it. Examples: research a topic, prepare a report, deploy an application.

Each decomposition method includes preconditions (when this method is applicable) and a sequence of subtasks (how exactly to accomplish the compound task using this method). For example, the task "prepare a report" may have two methods: (1) research and write from scratch, (2) use an existing template. The choice of method depends on context.

**The HTN planning algorithm** works recursively: if all tasks are primitive, the plan is ready; otherwise, the first compound task is selected, an applicable method is chosen for it, the task is replaced with the method's subtasks, and the process repeats.

**Difference from classical planning:** Classical planning (STRIPS) works with a goal state and searches for any way to achieve it, which can be inefficient. HTN encodes domain expert knowledge in decomposition methods, guiding the search but limiting flexibility — only tasks with defined methods can be solved.

**Application to LLM agents:** Language models can act as HTN planners, selecting decomposition methods based on natural language reasoning. The agent analyzes preconditions, selects an appropriate method, then recursively decomposes subtasks until it obtains a sequence of primitive actions.

---

## Classical AI Planning: Theoretical Foundations

### From Formal Logic to Practical Systems

Before moving to modern approaches, it is important to understand classical planning algorithms that formed the foundation of all AI decision-making systems.

**Two approaches to plan search:**

State-space search starts from the initial world state and applies actions until the goal state is reached. This can be done forward (from start to goal) or backward (regression: from goal to start, determining what is needed to achieve it).

Plan-space search works differently: it starts with an "empty" plan containing only the start and goal, then gradually adds actions and ordering constraints, resolving conflicts between actions.

**Partial-Order Planning:**

Unlike a linear plan (strict sequence A → B → C → D), a partial-order plan defines ordering only where dependencies exist. If actions B and C are independent, the plan can represent them as parallel branches after A and before D.

Advantages: greater flexibility (fewer unnecessary ordering constraints), parallelism expressed naturally, fewer backtracks during solution search.

**Causal links and threat detection:**

Partial-Order Planning uses causal links to track dependencies: "Action A creates condition p, which is required by action B." If some action C can delete condition p and execute between A and B, this is called a threat. The planner resolves threats by establishing additional ordering constraints (C must be before A or after B).

Example: if the action "pick up block" creates the condition "block in hand," which is needed for the action "place block on tower," then the action "put down block" between them is a threat that must be eliminated.

### Connection Between Classical and LLM Planning

LLM agents implicitly implement classical planning concepts: state representation (context + memory), action selection (semantic tool matching), goal regression ("What is needed for X?" in CoT), partial-order planning (parallel steps), causal links (explicit dependencies), threat detection (conflict validation).

---

## GOAP Algorithm

### Goals and Actions

GOAP (Goal-Oriented Action Planning) is a formal approach to planning borrowed from the game industry. It represents the world as a set of boolean variables describing the state (e.g., "have key," "door is open," "monster defeated").

Each action is defined by three components:
- **Preconditions** — what must be true for the action to become possible
- **Effects** — which state variables change after the action
- **Cost** — how much it "costs" to perform this action (time, resources, risk)

The goal is formulated as a desired state: a set of variables that must become true. The planner searches for a sequence of actions that transitions the current state to the goal state with minimal total cost.

### Plan Search

GOAP uses pathfinding algorithms (typically A*) to find the optimal plan. The process:
1. Start from the current state
2. Consider all actions whose preconditions are satisfied
3. For each action, compute the new state (by applying effects)
4. Estimate the "distance" from the new state to the goal (heuristic)
5. Select the most promising direction and continue searching
6. Repeat until the goal state is reached

The heuristic is critically important for efficiency. A simple heuristic: the number of mismatched variables between the current and goal states. More sophisticated heuristics account for domain semantics.

### Applicability to AI Agents

GOAP excels for tasks with well-defined states and actions: system configuration, API call sequences, workflows with fixed stages.

However, the world of LLM agents is often more ambiguous. States are hard to formalize as boolean variables. Action effects are nondeterministic (an API may return an unexpected result). Cost varies and depends on context.

Nevertheless, GOAP concepts are valuable. Thinking in terms of "preconditions → action → effects" helps structure planning. GOAP can be used for subtasks with formalizable states, while LLMs handle high-level planning and uncertainty.

---

## Planning with LLMs

### LLM as a Universal Planner

Language models open a radically new approach to planning. Instead of formal algorithms requiring strict specification of states and actions — natural language reasoning. The model receives a task description in human language, a list of available tools, and generates a structured execution plan.

This works surprisingly well for several reasons:

**Implicit knowledge of planning patterns.** Models are trained on massive text corpora: instructions, recipes, business plans, technical specifications, procedures, algorithms. They have learned how humans structure multi-step tasks, which steps typically come first, and which dependencies are typical for different domains.

**Flexibility of formulation.** There is no need to formalize states as boolean variables or describe preconditions in first-order logic. It suffices to say: "Ensure the user is authorized before modifying data." The model understands the semantics and translates this into concrete steps.

**Context adaptation.** The same type of task may require different plans depending on context. An LLM accounts for nuances: available resources, priorities, time constraints, risks.

### Structure of an Effective Planning Prompt

Plan quality depends on prompt quality. An effective prompt includes:

**1. Clear task description:** A specific goal (not "improve the system" but "reduce API response time by 20%"), success criteria, constraints (time, resources, business).

**2. Available tools:** Name, purpose, parameters, return values, limitations (rate limits, side effects).

**3. Output format:** Step structure (action, tool, input data, result), explicit dependencies, optionally — complexity, fallback alternatives on errors.

**4. Quality criteria:** Speed vs. reliability? Risk tolerance or conservatism? Minimize calls or cost?

**5. Examples (few-shot):** Sample plans for similar tasks, examples of good/bad planning.

### Plan Validation and Optimization

Typical problems with LLM-generated plans: missing steps, incorrect dependencies, cycles, inefficient sequencing, nonexistent tools.

**Validation:** LLM critique ("check the plan for correctness, completeness, optimality"), formal checks (dependency graph for cycles, tool existence verification), simulation (dry run without execution).

**Optimization:** Identifying parallelism (graph analysis), removing redundancy (merging duplicates), reordering (minimizing context switches).

---

## Relationship Between Planning and Reinforcement Learning

### Planning as Search

At a fundamental level, planning is search in the space of action sequences. This creates a connection to reinforcement learning (RL), where the agent also searches for an optimal policy.

**Model-Free vs. Model-Based RL:**

Model-Free RL learns from experience without an explicit world model (Q-Learning, Policy Gradient). In agents — analogous to ReAct without planning.

Model-Based RL builds a transition model and plans on top of it (Dyna, MCTS, MuZero). In agents — Plan-and-Execute and LATS.

### LLM as a World Model

The LLM acts as a world model: predicting action outcomes, estimating their value. Why this works: pretraining provides implicit world knowledge, in-context learning adapts the model, Chain-of-Thought functions as a rollout in the world model.

### Monte Carlo Tree Search and LLMs

MCTS is an algorithm from model-based RL, applied in LATS.

**MCTS cycle:** Selection (node selection) → Expansion (generating continuations) → Simulation (playout) → Backpropagation (updating evaluations).

**Adaptation for LLMs:** Selection with UCB on a thought tree, Expansion via LLM, Simulation as LLM-guided rollout, Backprop updates node scores.

**Value Function:** V(state) estimates proximity to the goal. It enables prioritizing promising branches, pruning unpromising ones, and balancing exploration/exploitation.

### When to Use Which Approach

**Deterministic tasks with full information:** Classical planning (exact search is efficient).

**Uncertainty in outcomes:** Model-based RL approaches (accounting for probabilities and expected rewards).

**Feedback only at the end:** MCTS/Tree Search (rollouts for evaluating intermediate states).

**Continuous feedback:** Online replanning (adaptation in progress).

Understanding the connection between planning and RL helps design effective agents, leveraging decades of research.

---

## Adaptive Planning

### Reality vs. Plan

No plan survives contact with reality unchanged. Tools fail. Data differs from expectations. New information emerges that changes priorities. Requirements evolve during execution.

Adaptive planning is the ability to adjust the plan during execution without losing the overall direction toward the goal. This is not just error handling — it is continuous situation monitoring and readiness to revise the approach.

An agent with adaptive planning asks itself questions at every step:
- Is this still the best path to the goal?
- Has new information emerged that changes the optimal strategy?
- Have any steps become unnecessary or impossible?
- Can the goal be reached faster or more reliably by a different method?

### Replanning Strategies on Errors

When a plan step fails, the agent has four primary strategies. The choice depends on the nature of the error, remaining resources, and proximity to the goal.

**Retry:** For transient failures — timeouts, server overload, rate limiting. Use with exponential backoff and an attempt limit. Avoid for fundamental errors (wrong parameters, missing permissions).

**Modify:** When the error is in the parameters but the approach is correct — wrong URL, data format. Use when it is clear what to fix and how. Avoid for unclear problems or multiple failures.

**Alternative:** When the action is impossible but the goal is achievable by another means — fallback API, different tool. Use when a clear alternative with acceptable cost exists. Avoid when the alternative is unknown or too expensive.

**Replan (Full Replanning):** When changes are substantial — build a new plan accounting for what has been completed. Use when multiple errors point to an incorrect approach, or when context has changed fundamentally. Avoid when 1-2 steps remain to the goal or the plan is still workable with minor adjustments.

### Incremental Plan Refinement

Sometimes a plan needs refinement not because of errors but because new information makes subsequent steps more precise.

**Example:** The agent identified competitors and discovered: there are not 5 but 20; some are startups with minimal information, others are corporations with extensive reports. This affects the plan: narrow to top 5, split the analysis (overview of all + deep dive for the important ones), adapt collection methods (social media for startups, financial reports for corporations).

**Principle:** The plan evolves based on accumulated knowledge. Each step provides information for refining the remaining steps. This is not rewriting but organic evolution.

**Balance between planning and execution:** Too much detail upfront — wasted effort on steps that will change. Too little — chaotic wandering. The optimum: a detailed plan for 2-3 steps ahead, a rough plan for the entire task, refinement as progress is made.

### Reasoning Models and Planning

Reasoning models (o3, Claude with extended thinking) shift the planning landscape. These models can perform multi-step planning, contingency analysis, and replanning internally within a single generation pass — tasks that previously required explicit Plan-and-Execute frameworks.

**What changes:** For moderate-complexity tasks (5-10 steps), a reasoning model can generate and adapt a plan within its thinking phase, eliminating the need for a separate planning agent or LangGraph state machine. The model naturally handles the "plan → encounter obstacle → replan" cycle inside its reasoning tokens.

**What stays the same:** For long-running tasks with dozens of steps, external checkpointing, human-in-the-loop approval, or coordination across multiple specialized agents, explicit planning frameworks remain essential. The reasoning model's context window and token budget impose natural limits on how much planning complexity can be internalized.

**Practical rule of thumb:** If the task can be completed in a single model call (even a long one with many tool calls), let the reasoning model plan internally. If the task spans multiple sessions, requires persistent state, or needs human approval at intermediate steps — use an explicit planning framework.

---

## Parallel Execution

### Identifying Parallel Steps

Not all plan steps must be executed sequentially. If two steps do not depend on each other, they can be executed in parallel, speeding up execution and using resources more efficiently.

**Dependency analysis:** Step B depends on step A if:
- B uses the result of A's execution (data dependency)
- B requires state changes produced by A (state dependency)
- B must logically execute after A (ordering constraint)

If none of these conditions hold, the steps are independent and can execute in parallel.

**Building the execution graph:** In practice, identifying parallelism means building a DAG (directed acyclic graph) of dependencies. Nodes are plan steps, edges are dependencies. Then the graph is partitioned into levels:
- Level 1: steps with no input dependencies (can execute immediately)
- Level 2: steps depending only on level 1 (execute after level 1 completes)
- Level 3: steps depending on levels 1 and/or 2
- And so on

Within each level, all steps execute in parallel.

### Error Handling Strategies for Parallelism

What happens if one of the parallel steps fails?

**Fail-fast:** Stop everything on the first error. Saves resources, fast reaction. Use when steps are interrelated and critical. Risk: loss of useful results from other steps.

**Fail-safe:** Continue all steps, collect results, then decide. Maximizes information, partial results are useful. Use when steps are independent. Risk: wasting resources.

**Critical/Non-critical:** Separate into critical (error = stop) and non-critical (error = log). Flexible balance. Use when step criticality is obvious.

### Limitations and Balancing Parallelism

Parallelism is not free. Limitations:

**API:** Rate limits (requests/time), concurrency limits (simultaneous connections), quota limits (volume per period).

**Resources:** Memory, CPU, network bandwidth, token budget for LLM calls.

**Semantics:** Write conflicts, race conditions, deadlocks.

**Balancing:** Dynamic concurrency limiting (start conservatively, adapt). Prioritization (important steps first). Batching (grouping small operations).

**Trade-off:** More parallelism = faster but more complex and riskier. Less = more reliable but slower.

---

## Key Takeaways

**Decomposition is the foundation of planning.** Breaking a complex task into atomic subtasks makes it feasible. Good decomposition accounts for dependencies, complexity, and measurability of results.

**Hierarchy manages complexity.** A multi-level structure (goals → subgoals → actions) enables working with tasks of any scale. HTN encodes expert knowledge in decomposition methods.

**Classical algorithms remain relevant.** GOAP, Partial-Order Planning, the connection with RL — all of this provides a conceptual foundation for designing LLM agents.

**LLMs are flexible planners.** Language models generate plans in natural language, leveraging implicit knowledge of patterns. Plan quality depends on prompt quality and validation.

**Adaptability is critical.** Reality always differs from expectations. Four strategies for responding to errors (Retry, Modify, Alternative, Replan) plus incremental refinement ensure resilience.

**Parallelism accelerates but adds complexity.** Independent steps can execute simultaneously. Resource constraints must be considered, an error handling strategy chosen (fail-fast vs. fail-safe), and speed balanced against reliability.

---

## Practical Example: LLM Planner with Adaptation

This section examines a practical implementation of a planner that combines plan generation, validation, and adaptive execution.

### Prompt Structure for Plan Generation

Effective plan generation using a language model requires a carefully structured prompt. The generatePlan method accepts three key parameters: a task description, a list of available tools, and a set of constraints.

The prompt is built from several clearly separated sections:

**Task section** contains a detailed description of what needs to be accomplished. The more specific the task formulation, the more precise the plan. Instead of a vague "improve performance," it is better to specify "reduce API response time by 20%."

**Available tools section** lists all tools the agent can use. For each tool, the name, description of capabilities, input parameters, return values, and usage limitations are specified. The formatTools method converts the tool list into a readable format the model can analyze.

**Constraints section** defines the boundaries within which the plan must operate. These may be time constraints (complete within 5 minutes), resource constraints (no more than 100 API calls), business rules (do not modify production data), or technical limits (rate limits of external services). The formatConstraints method structures these constraints for the model.

**Instructions section** explicitly tells the model that a detailed execution plan is required.

For each plan step, the model must specify four mandatory elements:

First — a concrete and actionable action. Not an abstract "process data" but a specific "extract email addresses from the CSV file."

Second — the tool needed to perform the action. If the action does not require a tool (e.g., it is a logical reasoning step), "none" is specified.

Third — the expected result, which will allow verification of the step's successful completion. This is critical for adaptive planning and error handling.

Fourth — dependencies on other steps in the format of step numbers separated by commas. If the step is independent and can execute first, "none" is specified. This enables building a dependency graph and identifying opportunities for parallel execution.

**Output format** is structured for easy parsing. Each step begins with a number and action name. Then lines with prefixes Tool, Expected, and Depends follow, containing the corresponding information. Steps are separated by triple dashes for unambiguous boundaries.

The prompt is formatted with parameter substitution via the formatted method, which replaces placeholders with actual values for the task, tools, and constraints.

After receiving the model's response, the parsePlan method parses the structured text into a Plan object containing a list of steps with their metadata and dependency graph.

**Key prompt elements:**
- Clear task and context description
- List of available tools with their capabilities
- Constraints (time, resource, business logic)
- Structured output format for easy parsing
- Explicit dependency specification for identifying parallelism

### Adaptive Execution with Strategy Selection

The adaptive plan executor implements a fault-tolerant execution system with dynamic strategy adaptation. The executePlanAdaptively method manages the entire plan execution lifecycle.

Initialization begins by creating an empty results list for accumulating information about each step's execution. A replanning counter and maximum limit (typically 3) are set to prevent infinite plan rebuilding cycles.

The main execution loop continues until the plan is fully completed and the replanning limit is not exceeded. On each iteration:

The current step is extracted from the plan. The plan itself tracks which step should execute next, accounting for dependencies and already completed operations.

The step is executed via the executeStep method, which calls the corresponding tool with the step's parameters and returns a result with information about success, obtained data, or error details.

The result is added to the list for subsequent analysis and potential use during replanning.

**Handling successful execution:**

If a step completes successfully, it is marked as completed in the plan. This updates the plan's internal state and may unblock subsequent steps that depend on it.

Then incremental refinement of remaining steps is invoked. The current step's result may contain new information affecting subsequent actions. For example, if the step "get competitor list" returned 20 companies instead of the expected 5, the following analysis steps may be adjusted: filtering to the top 5 is added, or the analysis is split into overview and detailed tiers.

**Handling errors:**

When a step execution fails, the system selects the optimal adaptation strategy by analyzing the error type, current plan, problematic step, and original task. The chooseStrategy method applies heuristics to determine the most appropriate approach.

**RETRY strategy:** Used for transient failures such as timeouts, server overload, or rate limiting. The system pauses for a duration calculated using an exponential backoff formula based on the attempt number. First attempt — 1 second, second — 2 seconds, third — 4 seconds, and so on. This gives the overloaded system time to recover.

**MODIFY strategy:** Applied when the error is in the parameters but the overall approach is correct. The fixParameters method analyzes the error message and attempts to correct the step's parameters. For example, if an API returned a "wrong date format" error, the system can convert the date to the correct format and retry the step with corrected parameters.

**ALTERNATIVE strategy:** Selected when the action cannot be performed the current way but the goal is achievable differently. The findAlternative method searches for a different tool or approach to achieve the same goal. For example, if the primary API is unavailable, a fallback API or alternative data source is used.

**REPLAN strategy:** Applied for fundamental problems indicating the entire approach is wrong. The createNewPlan method generates an entirely new plan, accounting for the original task and results of already completed steps. The replanning counter is incremented to prevent infinite revisions.

**ABORT strategy:** Used when the error is critical and unrecoverable, or all adaptation options have been exhausted. A result with a failure indicator and error details is returned.

After processing all steps, the method returns the final execution result containing the list of all step results and a success flag (whether the plan was completed in full).

**Implementation principles:**
- Limit on the number of replannings (preventing infinite loops)
- Incremental refinement after each successful step
- Adaptation strategy selected based on error analysis
- Accumulation of results for use during replanning
- Exponential backoff on retries to avoid overloading

This approach combines the flexibility of LLM planning with the reliability of adaptive execution, ensuring resilience to errors and changes during operation.


---

## Navigation
**Previous:** [[03_Tool_Use|Tool Use]]
**Next:** [[05_Memory_Systems|Agent Memory Systems]]
