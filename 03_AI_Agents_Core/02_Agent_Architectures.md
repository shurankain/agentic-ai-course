# AI Agent Architectures

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[01_What_is_AI_Agent|What is an AI Agent]]
**Next:** [[03_Tool_Use|Tool Use]]

---

## Introduction

If an AI agent is an autonomous system capable of solving tasks, then the agent architecture is its internal design that determines how it thinks, plans, and acts. Over the past several years, researchers have developed numerous architectural patterns, each embodying a distinct philosophy for approaching problem-solving.

Choosing an architecture is more than a technical decision. It determines how the agent will interact with the world: will it impulsively react to every stimulus or carefully plan its actions? Will it learn from its mistakes or forget them immediately? Will it consider alternative paths or follow a single trajectory?

This chapter covers the evolution of AI agent architectures — from simple reactive systems to complex cognitive architectures inspired by human reasoning.

---

## Academic Foundations of Agent Architectures

### Classical Definitions of Agency

Before examining specific architectures, it is important to understand the academic foundations on which modern AI agents are built.

**Russell & Norvig (AI: A Modern Approach, 2021)**

The classical definition views an agent through a cyclical process of perception and action. The agent perceives its environment through sensors (perceive), reasons about the best action (reason), executes it through effectors (act), and learns from the results (learn). This cycle repeats continuously, creating a feedback loop between the agent and the environment. Each action affects the environment, the changed environment is perceived anew, and the process continues. This definition remains the foundation for understanding agent systems.

**Wooldridge (Multi-Agent Systems, 2009)**

Wooldridge identified four key properties of agents:

| Property | Description | Manifestation in LLM Agents |
|----------|-------------|------------------------------|
| **Autonomy** | Acts without direct intervention | Agent independently selects the next step |
| **Social Ability** | Interacts with other agents | Multi-agent systems, A2A/MCP protocols |
| **Reactivity** | Responds to environmental changes | Processing tool outputs, error recovery |
| **Proactivity** | Initiates actions toward goals | Goal-directed planning |

These four properties help assess the "agentness" of a system: the more properties are expressed, the more "agentic" the system is.

### Andrew Ng's 4 Agentic Design Patterns

In 2024, Andrew Ng (founder of DeepLearning.AI, co-founder of Google Brain) proposed a practical taxonomy of four key agentic design patterns:

**1. REFLECTION** — the agent evaluates and improves its own results. This is a self-criticism pattern: after generating a response, the agent analyzes its quality, identifies weaknesses, and iteratively improves it. Includes quality checks, output validation, and gradual solution refinement.

**2. TOOL USE** — the agent extends its capabilities through external tools. Instead of relying solely on its own knowledge, the agent delegates tasks to specialized systems: makes API calls to retrieve data, executes code for computations, queries databases for precise information.

**3. PLANNING** — the agent breaks complex tasks into manageable subtasks. Instead of trying to solve everything at once, the agent formulates a strategy, decomposes the problem into a sequence of steps, and manages progress toward intermediate goals.

**4. MULTI-AGENT** — multiple agents collaborate to solve a task. Each agent has a specialized role, agents can debate to reach consensus, and they divide work according to their competencies.

**Applying the patterns:**

These patterns are not mutually exclusive — production agents typically combine several of them. For example, a RAG agent uses Tool Use for retrieval plus possibly Reflection for relevance assessment. A coding agent combines Planning for decomposition, Tool Use for code execution, and Reflection for test verification. A research agent combines Planning, Tool Use for search, and a Multi-agent approach with different specializations.

### Validated Academic Patterns

The most influential research papers that shaped modern architectures:

| Pattern | Authors (Year) | Key Idea | Citations |
|---------|----------------|----------|-----------|
| **ReAct** | Yao et al. (2022) | Interleaving reasoning and actions | 2000+ |
| **Plan-and-Execute** | Wang et al. | Separating planning from execution | 500+ |
| **Self-Refine** | Madaan et al. (2023) | Iterative improvement through self-feedback | 800+ |
| **Reflexion** | Shinn et al. (2023) | Verbal reinforcement through self-reflection | 1000+ |
| **Tree of Thoughts** | Yao et al. (2023) | Exploration of alternative reasoning paths | 1500+ |

Each of these patterns is examined in detail below.

---

## ReAct: Reasoning and Acting as a Unified Process

### Philosophy

ReAct (Reasoning + Acting) is an architecture that unifies two fundamental cognitive processes: reasoning and acting. The name reflects the key idea: the agent does not simply act — it thinks aloud, explaining its logic before each step.

Consider a detective investigating a case. They do not silently collect evidence — they verbalize their reasoning: "Interesting, a red hair was found at the crime scene. I need to check which suspects have red hair." Then they take action — checking the suspects. Upon receiving a result, they reason again: "Two of them have red hair. Now I need to establish each one's alibi."

A ReAct agent works the same way. Each cycle of its operation consists of three phases: Thought (reasoning), Action (acting), and Observation (observing the result). This structure makes the decision-making process transparent and debuggable.

### Anatomy of the ReAct Cycle

The ReAct cycle begins with reasoning. The agent analyzes the current situation and articulates what it needs to do and why. This is not decoration — reasoning helps the model focus on the correct next step and avoid impulsive decisions.

After reasoning, the agent selects an action from the available set of tools. It specifies the tool name and parameters for the call. The action choice logically follows from the preceding reasoning — it is not a random selection but a justified decision.

After executing the action, the agent receives an observation — the tool's output. This could be a search engine response, a computation result, file contents, or any other information. The observation becomes input for the next reasoning step.

The cycle repeats until the agent reaches a final answer or exhausts the iteration limit.

### Strengths of ReAct

The main advantage of ReAct is **reasoning transparency**. A developer can inspect at any point what the agent was thinking and why it made a particular decision. This is invaluable for debugging and understanding system behavior. In production systems, ReAct agent logs enable rapid problem diagnosis and prompt improvement.

ReAct also stands out for its **simplicity of implementation**. No complex planning or memory mechanisms are needed — a well-formulated prompt and an execution loop are sufficient. A minimalist implementation can fit in 50-100 lines of code.

**Cost efficiency** is another important advantage. ReAct requires a minimal number of LLM calls: typically 5-10 per task. This makes it accessible even on a limited budget.

The architecture works well for a **broad range of tasks**, especially when a solution can be found in a few sequential steps: information search, simple data analysis, question answering with tool use.

### Limitations of the Architecture

However, ReAct has significant limitations. The agent **cannot plan ahead** — each decision is made based on the current state, without considering a global strategy. This is like a chess player thinking only one move ahead. For tasks requiring multi-step strategies, this is critical.

ReAct is **prone to looping**. The agent can endlessly repeat the same actions if it does not get the expected result. It lacks a mechanism for recognizing: "This approach is not working; I need to try something fundamentally different." A practical solution is limiting max_iterations and detecting repeating patterns.

**Absence of learning** means the agent does not improve over time. Each task is solved "from scratch," without leveraging experience from previous attempts. For recurring task types, this is inefficient.

Finally, the **greedy strategy** for action selection can lead to suboptimal solutions. The agent always picks what seems best right now, without exploring alternative paths.

---

## Plan-and-Execute: Think First, Then Act

### Philosophy

Plan-and-Execute represents an entirely different philosophy. Instead of reactively responding to each step, the agent first creates a complete plan for solving the task, then methodically executes it.

This architecture is inspired by how experienced professionals work. An architect does not start building a house by laying bricks one after another on intuition. They create a complete blueprint: foundation, walls, roof, utilities. Only with a holistic vision do they begin implementation.

Plan-and-Execute consists of two independent components: the Planner and the Executor. The Planner analyzes the task and breaks it into a sequence of concrete steps. The Executor takes those steps one by one and carries them out.

### Two-Phase Architecture

The planning phase is a creative process. The Planner receives the task and the list of available tools. Its job is to create a step-by-step plan where each step is concrete, executable, and logically follows from the previous one.

A good plan accounts for dependencies between steps. You cannot analyze data that has not yet been collected. You cannot send a report that has not been written. The Planner must arrange steps in the correct order.

The execution phase is methodical work. The Executor takes the plan and carries it out step by step. For each step, it determines which tool to use and with what parameters. The result of each step is saved and can be used in subsequent steps.

### Adaptive Replanning

A key feature of advanced Plan-and-Execute implementations is the ability to replan. The real world is unpredictable: tools can return errors, data can differ from expectations, new information can emerge.

When the Executor encounters a problem it cannot resolve within the current plan, it returns to the Planner. The Planner analyzes the situation: what has already been completed, what went wrong, what alternatives exist. Then it creates a new plan that accounts for the current reality.

This makes the agent resilient to the unexpected. It does not give up at the first error but searches for workarounds.

### Advantages and Trade-offs

**Advantages of Plan-and-Execute:**

**1. Global optimization** — the Planner sees the entire task and can arrange an optimal sequence of steps. This prevents situations where the agent takes a step that later needs to be undone.

**2. Predictability** — with a plan, it is easier to estimate execution time and required resources. This is critical for production systems where SLA guarantees are needed.

**3. Parallelization** — if the plan contains independent steps, they can be executed in parallel. ReAct, being strictly sequential, does not offer this possibility.

**4. Easier debugging** — when something goes wrong, the plan can be analyzed separately from execution. "Was the plan correct? Or was the problem in execution?"

**Trade-offs and limitations:**

**1. Planning overhead** — creating a plan requires an additional LLM call, increasing latency and cost. For simple tasks, this overhead may be unjustified.

**2. Plan fragility** — if the situation changes rapidly, the plan can become stale. Dynamic tasks require frequent replanning, which negates the advantages.

**3. Implementation complexity** — two components (planner and executor) must be implemented instead of one, their interaction configured, and replanning scenarios handled.

**4. Over-planning risk** — sometimes the agent spends too much effort creating a perfect plan for a simple task where ReAct would have been faster.

---

## Reflexion: Learning Through Self-Analysis

### Philosophy

Reflexion adds a capability to the agent that distinguishes experts from novices — the ability to learn from mistakes. This is not simply retrying — it is deep analysis of what went wrong and how to fix it.

Consider a student preparing for an exam. An inexperienced student who receives a poor grade simply rereads the textbook. An experienced student analyzes their mistakes: "I did not understand topic X, confused concepts Y and Z, did not spend enough time on practice." This analysis guides further preparation.

A Reflexion agent works similarly. After each attempt to solve a task, it evaluates the result, analyzes errors, and formulates lessons. These lessons are stored in memory and used in subsequent attempts.

### Three Components of Reflexion

The Reflexion architecture includes three interacting components: Actor, Evaluator, and Reflector.

The Actor executes the task using available tools. This can be any base architecture — ReAct, a simple agent, or even a program.

The Evaluator assesses the Actor's output. The assessment can be binary (success/failure), numeric (completion percentage), or textual (detailed analysis). The Evaluator answers the question: "How well was the task performed?"

The Reflector analyzes failed attempts. It looks for root causes of errors, formulates lessons, and proposes improvements. The Reflector's output is knowledge that will help avoid the same mistakes in the future.

### Memory as a Key Element

Unlike ReAct and Plan-and-Execute, Reflexion necessarily uses memory. Without memory, reflection is meaningless — lessons must be stored somewhere.

Reflexion memory stores not raw data but high-level insights: "When working with API X, the request rate limit must be considered," "Tasks of type Y are better decomposed into subtasks," "Tool Z often fails on large inputs."

Before each new attempt, the agent retrieves relevant lessons from memory and uses them to improve its approach.

### When Reflexion is Particularly Useful

Reflexion delivers the best results in specific scenarios:

**1. Tasks with clear success criteria** — unit tests for code, correct answers in math, compliance with a specification. The Evaluator can objectively determine success or failure.

**2. Iterative tasks** — where multiple attempts are possible and acceptable. Producing ideal code, writing high-quality text, optimizing a solution. One attempt may be insufficient, but there is an opportunity to improve.

**3. Informative errors** — when failure contains useful information. For example, a traceback on an exception points to a specific line. Or tests reveal which edge cases were not handled. Uninformative errors ("something went wrong") provide no material for reflection.

**4. Patterns of success and failure** — tasks where there are recurring types of errors and successful strategies. Debugging often falls into the same categories of problems. API integration has typical error patterns (rate limits, authentication, malformed requests).

**5. Long-term application** — if the agent solves many similar tasks, accumulated reflections pay off. For one-off tasks, the Reflexion overhead may be unjustified.

**Typical use cases:**
- Code generation with automated tests
- Math problems with verifiable answers
- Text generation per strict specification (legal documents, technical writing)
- API integration where errors are predictable and categorizable
- Optimization tasks where metrics can be iteratively improved

### Reflexion's Connection to Meta-Learning: Learning to Learn

Reflexion is not just an architectural pattern but an implementation of **meta-learning** (learning to learn) in the context of LLM agents.

**What is Meta-Learning:** In traditional machine learning, a model learns to solve a specific task. In meta-learning, the model learns **how to learn** — it acquires the ability to quickly adapt to new tasks.

**Reflexion as Meta-Learning** achieves this through three mechanisms:

**1. Learning from Self-Generated Experience** — the agent generates a training signal from its own experience, without labeled data from experts. After a failure, it analyzes: "Why did I fail? What did I learn?" These insights become meta-knowledge: "For tasks of type X, approach Y works better than Z."

**2. Fast Adaptation (Few-Shot Learning)** — after accumulating reflections, the agent shows rapid improvement on similar tasks. Without reflections, each new task requires the same number of attempts. With Reflexion, the number of attempts drops sharply for familiar task types.

**3. Episodic Memory as Meta-Knowledge** — memory stores not specific solutions ("the answer to task X is 42") but generalized lessons ("recursive tasks benefit from memoization," "on API timeout use exponential backoff").

**Formal connection to MAML:** Model-Agnostic Meta-Learning finds parameter initializations for fast adaptation. Reflexion achieves a similar effect **without modifying model weights**, using prompt conditioning — this is a form of **in-context meta-learning**.

**Practical implications:**
1. **Transfer learning**: reflections from one task type improve performance on similar ones
2. **Curriculum learning**: start with simple tasks, accumulate reflections, transition to complex ones
3. **Memory organization**: organize reflections by task type (coding_lessons, research_strategies, general_patterns) for efficient retrieval

---

## LATS: Exploring the Solution Space

### Philosophy

LATS (Language Agent Tree Search) combines language agents with classical search algorithms. Instead of following a single path, LATS explores many alternative paths and selects the best one.

Consider a chess grandmaster. They do not simply make the first move that looks good. They mentally play through multiple variations: "If I move here, the opponent might respond this way or that way. In the first case I'll have an advantage; in the second, I'll lose."

LATS applies this idea to AI agents. The agent generates several possible actions, evaluates each one, and selects the most promising. But it does not discard alternatives entirely — if the chosen path turns out to be a dead end, it can backtrack and try another.

### Tree-Based Search Structure

LATS represents the solution process as a tree. The root is the initial state (the task). Each node is a state after some sequence of actions. Edges are actions that transition from one state to another.

Search begins at the root. The agent generates several possible actions from the current state — this is called "expanding" a node. Each action creates a new child node.

The agent then evaluates each new node: how close is this state to the goal? This is "simulation" — predicting the success of the path.

Evaluations propagate up the tree — "backpropagation." Nodes leading to good results receive high scores. Nodes leading to dead ends receive low scores.

On the next iteration, the agent selects the node with the best score for expansion. Gradually, a tree is built where more attention is given to promising directions.

### Balancing Exploration and Exploitation

The key problem in search is balancing exploration and exploitation. Exploration means trying new, unknown paths. Exploitation means deepening into already known good paths.

Too much exploration — the agent wastes resources on unpromising directions. Too much exploitation — the agent can get stuck in a local optimum, missing a better solution.

LATS uses the UCB (Upper Confidence Bound) formula for balancing. This formula considers both the node's score and the number of times it has been visited. Rarely visited nodes receive a "curiosity bonus" that encourages the agent to explore them.

### Rollout Strategies: Evaluating the Future

A **rollout** in LATS is a way to assess how promising a tree node is by "looking" into the future. The choice of rollout strategy determines the quality and cost of the search.

**Three main rollout approaches:**

**1. Random Rollout (Monte Carlo)** — the simplest approach where random actions are taken from the current node to a terminal state. Fast, but has high variance. Suitable when the number of actions is small and the state space is limited.

**2. LLM-guided Rollout** — using a language model for intelligent trajectory completion. The model generates a plausible continuation from the current state to the final answer. Produces more realistic estimates, but each rollout requires an expensive LLM call.

**3. Truncated Rollout with Value Function** — a hybrid approach: a short rollout of a few steps plus state evaluation via a value function. Balances accuracy and speed. This is the standard approach in AlphaGo and similar systems.

### Value Function: State Evaluation

The value function V(s) estimates the "value" of a state — the expected reward from that state to the end of the task. In the context of LLM agents, the value function can be implemented in three ways:

**1. LLM as critic** — the model evaluates the current state on a 0-1 scale, estimating the probability of reaching the correct solution. Flexible, but requires an additional LLM call per evaluation.

**2. Reward Model** — if a trained reward model is available (e.g., from RLHF), it is used to directly predict state value. Fast and accurate, but requires prior training.

**3. Heuristic Rules** — a set of heuristics for rapid evaluation: are there progress indicators, are there obvious errors, is the solution approaching a final answer. The fastest approach, but requires domain knowledge to develop good heuristics.

### Computational Cost and Applicability

LATS is the most computationally expensive of the architectures discussed. Each iteration requires multiple model calls: generating candidates, evaluating each one, possibly reflecting.

**Cost estimate:** With typical parameters (3-5 candidates per node, tree depth 5, width 4), a single task can require approximately 100 LLM calls. This is orders of magnitude more expensive than ReAct (5-10 calls) or Plan-and-Execute (10-20 calls).

**LATS is justified when:**
- The cost of error is high (critical decisions, financial transactions)
- There are many possible paths with no obvious best choice
- Sufficient computational budget is available
- Finding the optimal, not just an acceptable, solution matters
- The task has a clear quality metric for evaluating intermediate steps

---

## Advanced Reasoning Techniques

### From Chain-of-Thought to Tree-of-Thought

Chain-of-Thought (CoT) was a breakthrough in LLM reasoning — the model "thinks aloud," sequentially building a logical chain to the answer. But linear chains have a fundamental flaw: a single error in the middle of the reasoning corrupts the entire result. There is no way to go back and try an alternative path.

Tree-of-Thought (ToT) overcomes this limitation by turning the linear chain into a tree. Instead of a single reasoning path, the agent explores multiple branches, evaluates each, and deepens into the most promising ones.

Consider a mathematician solving a complex problem. They do not write a single solution from start to finish. They try approach A, realize it leads to a dead end, go back, and try approach B. ToT formalizes this natural thinking process.

### Tree-of-Thought Structure

ToT consists of four components.

**Decomposition** — breaking the problem into intermediate thoughts. Each thought is a reasoning step small enough to be evaluated and significant enough to make progress toward a solution.

**Generation** — generating possible continuations for the current node. The model proposes several alternative next steps. This can be sampling (generating multiple variants) or propose (explicitly requesting variants).

**Evaluation** — assessing the quality of each step. The model acts as a critic of its own ideas, determining which branches are promising and which lead to dead ends. The evaluation can be numeric (probability of success) or categorical (sure/maybe/impossible).

**Search** — the tree traversal algorithm. Classical strategies apply here: BFS (breadth-first search) or DFS (depth-first search).

### BFS vs DFS: Exploration Strategies

The choice of search strategy critically affects ToT efficiency. Each strategy has its trade-offs.

**BFS (Breadth-First Search)** explores the tree level by level. First all nodes at the first level, then the second, and so on.

**BFS advantages:**
- Guarantees finding the shortest path to a solution
- Finds all solutions at a given depth
- Works well when the solution is shallow (2-4 levels)

**BFS disadvantages:**
- Requires storing all nodes at the current level in memory
- Exponential growth for wide trees (branching factor 5, depth 4 = 625 nodes)
- Many LLM calls to evaluate all nodes

**When to use BFS:** tasks with shallow solutions, when finding the optimal path matters, when there is budget to explore the full breadth.

**DFS (Depth-First Search)** goes deep along one branch to the end, then backtracks and tries alternatives.

**DFS advantages:**
- Memory-efficient (stores only the current path)
- Quickly finds any solution (not necessarily optimal)
- Fewer LLM calls if the first branch succeeds

**DFS disadvantages:**
- Can get stuck in very deep branches
- Does not guarantee an optimal solution
- Requires a depth limit to prevent infinite search

**When to use DFS:** solutions are deep, memory is limited, any valid solution is sufficient.

**Beam Search** — a practical trade-off that retains the k best nodes at each level (typically k=3-5). This limits the breadth of exploration, making the search manageable. Beam search is the standard choice for most production ToT systems.

### Monte Carlo Tree Search and LLMs

MCTS (Monte Carlo Tree Search) — the algorithm that gained fame in AlphaGo — also finds application in LLM agents. MCTS adds statistical sampling and exploration/exploitation balancing to tree search.

The MCTS cycle consists of four phases:

**Selection** — descending the tree from root to leaf using UCB (Upper Confidence Bound) for node selection. UCB balances exploitation (selecting nodes with high scores) and exploration (selecting rarely visited nodes).

**Expansion** — adding new child nodes to the selected leaf. Each node is a potential action or thought.

**Simulation** — a "rollout" from the new node to a terminal state. In the LLM context, this can be a quick completion of reasoning to a final answer.

**Backpropagation** — updating the statistics of all nodes on the path from leaf to root. If the rollout led to success, all nodes along the path receive a bonus.

UCB formula for node selection:

$$UCB(i) = \frac{Q(i)}{N(i)} + c \cdot \sqrt{\frac{\ln N(parent)}{N(i)}}$$

Where Q(i) is the cumulative reward of the node, N(i) is the visit count, c is the exploration coefficient. The first term is exploitation (selecting successful nodes); the second is exploration (giving a chance to under-explored nodes).

MCTS is particularly effective for tasks with large solution spaces and delayed reward — when reasoning quality can only be evaluated at the end.

### O1-style Reasoning: Inference-time Compute

OpenAI's o1 series models demonstrated a paradigm shift: instead of scaling model size, one can scale computation at inference time. The model "thinks longer" on complex questions.

The connection to agent architectures is direct:
- **Extended thinking** — the model generates internal reasoning before the answer, similar to a "scratchpad" in ReAct
- **Search and backtracking** — the internal search resembles ToT/MCTS but is built into the generation process itself
- **Self-correction** — the model corrects errors in its reasoning, like Reflexion

The o1-style approach blurs the boundary between "prompt engineering" and "architecture design." The ability for deep reasoning becomes a property of the model itself, not an external agent framework.

For practitioners, this means: if the model has built-in reasoning (like o1 or Claude with extended thinking), the agent architecture can be simplified. Complex techniques like ToT may be redundant because the model already does something similar internally.

### Self-Consistency: The Power of Multiple Opinions

Self-Consistency is an elegant technique for improving reasoning reliability without a complex tree structure. The idea is simple: generate several independent reasoning chains and take the majority answer.

Why does this work? Different reasoning paths may lead to different intermediate steps, but the correct answer is usually reached by multiple paths. Erroneous reasoning, however, produces random, inconsistent answers.

Self-Consistency algorithm:
1. Generate k independent CoT reasoning chains (with temperature > 0)
2. Extract the final answer from each chain
3. Select the answer with the highest frequency (majority voting)

Self-Consistency is particularly effective for tasks with discrete answers: math, programming, factual QA. For open-ended tasks with many valid answers, more sophisticated aggregation methods are needed.

An important advantage of Self-Consistency is parallelizability. All k reasoning chains are independent and can be generated in parallel, distinguishing this approach from sequential methods like Reflexion.

### Integration with Agent Architectures

Advanced reasoning techniques do not exist in isolation — they combine with base architectures.

**ReAct + Self-Consistency**: execute several independent ReAct trajectories and select the consensus result. Especially useful for tasks with uncertainty.

**Plan-and-Execute + ToT**: use ToT to generate the plan, then execute it. The tree allows exploring alternative plans.

**LATS as integration**: LATS (Language Agent Tree Search) is literally MCTS applied to the agent context. Tree nodes are agent states, actions are tools, rollout is evaluation of the final result.

Practical advice: start simple (ReAct or basic CoT), add Self-Consistency for improved reliability, move to ToT/MCTS only if the task demands deep search.

---

## Cognitive Architectures: Inspired by the Human Mind

### Philosophy

Cognitive architectures attempt to model structures of human thinking in an AI agent. Rather than creating an architecture from scratch, researchers take cognitive psychology models developed over decades of research as their foundation.

### ACT-R: Adaptive Control of Thought

**ACT-R (Adaptive Control of Thought—Rational)** is one of the most influential cognitive architectures, created by John Anderson at Carnegie Mellon. It describes human cognition as an interaction of specialized modules.

The ACT-R architecture consists of a central production rule system (procedural memory) that coordinates module operation. Production rules take the form "IF condition THEN action" — for example, "IF goal = solve math problem AND type = algebra, THEN retrieve algebra strategy." These rules interact with declarative memory (a store of facts as "chunks") and a goal module (managing intentions). All of this is coordinated through working memory buffers — a limited space for current context that imitates the limits of human attention.

**Key ACT-R concepts for LLM agents:**

1. **Activation-based retrieval** — chunks in memory have activation levels that determine retrieval speed. Activation depends on frequency and recency of use: frequently used recent knowledge is retrieved faster. In LLM agents, this is implemented through scoring during retrieval from a vector store.

2. **Production conflict resolution** — when multiple rules are applicable, the one with the highest utility is selected. Utility is computed as: probability of success × goal value − execution cost. This models rational action selection.

3. **Goal stack** — a hierarchical goal structure where subgoals can be "pushed" and "popped" like a function call stack. This allows temporarily diverting to subtasks and then returning to the main goal.

### SOAR: State, Operator, And Result

**SOAR** is a fundamental cognitive architecture emphasizing problem-solving and learning. If ACT-R focuses on memory, SOAR focuses on overcoming impasses.

**Key SOAR cycle:** Input (receiving information) → Elaboration (enriching the state with rules) → Decision (selecting an operator) → Application (applying it) → Output (affecting the environment). When an **impasse** (difficulty) arises, SOAR creates a subgoal and solves it recursively.

**Chunking in SOAR** — the learning mechanism. When a subgoal is successfully resolved, SOAR creates a new rule: "IF similar situation THEN apply the found solution." This prevents the same impasses in the future.

**Connection between SOAR and LLM agents:**
- **Impasse detection** → the agent recognizes insufficient current knowledge
- **Subgoaling** → task decomposition into subtasks (Plan-and-Execute)
- **Chunking** → saving successful solutions for reuse (Reflexion)

### How Cognitive Architectures Inspire LLM Agents

| ACT-R/SOAR Concept | LLM Agent Implementation |
|-------------------|--------------------------|
| Declarative Memory | Vector store with embeddings |
| Procedural Memory | Tool definitions + few-shot examples |
| Working Memory | Context window |
| Activation decay | Recency scoring during retrieval |
| Production rules | In-context learning / prompting |
| Chunking | Saving successful trajectories |
| Goal stack | Hierarchical planning |

AI agents inspired by these architectures emulate their structure. Each module is implemented as a separate component with its own logic.

### Modular Structure

Declarative memory stores facts and knowledge. It is not a simple database — the memory has an activation mechanism. Frequently used facts have high activation and are retrieved quickly. Rarely used facts are "forgotten" — their activation decreases over time.

Procedural memory stores skills in the form of production rules: "IF condition, THEN action." When the current situation matches a rule's condition, the rule can be applied. Rules compete with each other, and the most appropriate one is selected.

Working memory is the agent's "RAM." It stores the current context: what is happening now, what goal is being pursued, what intermediate results have been obtained. Working memory is limited in capacity, imitating the limitations of human attention.

The goal module manages the agent's intentions. Goals can be hierarchical: the main goal is broken into subgoals, subgoals into even smaller tasks. Achieving subgoals moves toward the main goal.

### Information Processing Cycle

A cognitive agent operates cyclically. Each cycle iteration includes: perceiving the current situation, retrieving relevant knowledge from memory, selecting an appropriate rule, executing an action, and updating memory and goals.

This cycle resembles human behavior. We perceive a situation, recall similar experience, apply a familiar skill, observe the result, and adjust our actions.

### Reinforcement Learning

An important feature of cognitive architectures is built-in learning. Successful actions are reinforced: their probability of selection in similar situations increases. Unsuccessful actions are weakened.

This does not require an explicit reflection mechanism as in Reflexion. Learning occurs implicitly through gradual changes in weights and activations.

### Applicability and Limitations

Cognitive architectures are most appropriate for long-lived agents that accumulate experience and improve over time. They are well-suited for tasks where personalization and user adaptation are important.

However, implementing a full cognitive architecture is complex. Careful tuning of numerous parameters is required: forgetting rates, activation thresholds, rule weights.

---

## Agent-Computer Interface (ACI): A New Paradigm from Anthropic

### Rethinking Agent-Computer Interaction

In 2024, Anthropic introduced the concept of **Agent-Computer Interface (ACI)** — a new paradigm for designing tools for AI agents. While the traditional approach to tool use focuses on API calls, ACI takes a broader view: how to make the entire computing environment accessible to an agent.

**Key insight:** Just as a UI (User Interface) is designed for human convenience, ACI should be designed for AI convenience. What is intuitive to a human (visual interfaces, context menus) may be inconvenient for an agent, and vice versa.

### ACI Principles

**1. Affordances for AI** — in UI design, an affordance suggests how to use an object (a button looks pressable). ACI introduces affordances for AI. A poor ACI requires the agent to "find the Submit button in the bottom-right corner" (visual understanding + coordinates). A good ACI provides a direct call: submit_form(form_id="checkout").

**2. Structured over Visual** — when possible, provide structured data instead of visual data. A screenshot of a table requires OCR and parsing. JSON/CSV with the table data gives direct access. Visual interfaces for AI are a last resort when no structured API exists.

**3. Semantic Actions** — actions should be semantically meaningful, not low-level. move_mouse(x, y) and click() are brittle and resolution-dependent. click_element(selector="#submit-button") uses semantic identification and is resilient to layout changes.

### ACI vs Traditional Tool Use

| Aspect | Traditional Tools | ACI Approach |
|--------|-------------------|--------------|
| Focus | Individual APIs | Holistic environment |
| Navigation | Coordinates / OCR | Semantic selectors |
| State | Stateless calls | Persistent context |
| Errors | Return codes | Rich error context |
| Discovery | Documentation | Self-describing tools |

### Practical ACI Implementation

**Computer Use Tools (Anthropic)** — a reference ACI implementation. Anthropic released a set of versioned tools (computer_20241022, text_editor_20241022, bash_20241022) that provide the agent with unified access to the computing environment.

**Key features:**
- **Versioned tool types** — tools are versioned for backward compatibility (when the API changes, old versions continue to work)
- **Composite capabilities** — one tool combines multiple capabilities (the computer tool includes mouse, keyboard, screenshot)
- **Environment awareness** — tools know about their environment (screen size, OS, available applications)

### When to Use the ACI Approach

**ACI is justified when:**
- The agent must work with GUI applications
- No ready-made API exists for the task
- The task requires navigating a complex interface
- Universality is needed (one agent for many applications)

**Traditional tools are better when:**
- A clear API exists
- The task is specific and well-defined
- Speed and reliability are important
- Visual understanding is not required

### ACI's Relationship to Agent Architectures

ACI is not an agent architecture but a paradigm for designing the agent's **interface to the world**. ACI can be used with any architecture:

**ReAct + ACI:** The agent reasons ("I need to open a browser"), then executes an ACI action (computer launch browser), receives an observation (browser opened), and continues the cycle.

**Plan-and-Execute + ACI:** The Planner creates a plan of high-level steps (open browser, find information, extract results). The Executor translates each step into a sequence of ACI calls.

---

## Choosing an Architecture: Practical Recommendations

### Selection Criteria

Choosing an architecture is an engineering decision with clear trade-offs. The key factors are as follows.

**1. Task Complexity**

Simple tasks (3-5 sequential actions): **ReAct** is sufficient. Examples: information search, simple calculations, data retrieval.

Medium complexity (5-15 steps with dependencies): **Plan-and-Execute** is optimal. Examples: report generation, multi-source research, ETL pipelines.

High complexity (many alternative paths): **Tree-of-Thought or LATS**. Examples: mathematical proofs, complex planning.

**2. Reliability Requirements**

Errors are acceptable, speed matters: **ReAct** with a retry mechanism.

Errors are critical, retries are possible: **Reflexion** for iterative improvement.

Optimal solution needed, errors are costly: **LATS** with deep search.

**3. Computational Budget**

Limited budget (startups, prototypes): **ReAct** (5-10 LLM calls).

Medium budget (production apps): **Plan-and-Execute** (10-20 calls) or **Reflexion** (3-5 trials × base architecture).

Large budget (critical systems): **LATS** (50-100+ calls) or **Cognitive architectures**.

**4. Time Constraints**

Fast response needed (< 10 sec): **ReAct** or simple **Plan-and-Execute**.

Waiting is acceptable (30-60 sec): **Reflexion** with several trials or **ToT** with limited depth.

Time is not critical (minutes): **LATS** with deep search.

**5. Duration of Application**

One-off tasks: **ReAct** or **Plan-and-Execute**.

Recurring task types: **Reflexion** (accumulated experience pays off).

Long-lived personalized agent: **Cognitive architecture** (complexity is justified).

### Hybrid Approaches

In practice, pure architectures are rare. Successful production systems combine elements:

**Plan-and-Execute + ReAct**: The Planner creates a high-level plan. Each step is executed by a ReAct agent. This provides global structure with flexible execution.

**Reflexion over any architecture**: Reflexion is a meta-pattern. ReAct, Plan-and-Execute, or ToT can be wrapped in a Reflexion loop to add learning.

**Adaptive architecture**: Architecture is selected based on task characteristics. Simple requests → ReAct. Complex → Plan-and-Execute. Critical → LATS. Requires a task classifier.

**Hierarchical agents**: A master agent (Plan-and-Execute) coordinates specialized sub-agents (ReAct). Each sub-agent is an expert in its domain.

### Practical Selection Workflow

1. **Start with ReAct** — this is the baseline for all tasks. Implement it in a day; verify whether it is sufficient.

2. **Measure metrics** — success rate, latency, cost per task. This provides an objective understanding of the pain points.

3. **Identify the bottleneck** — if failure is due to lack of planning → Plan-and-Execute. Due to uninformative errors → improve error handling. Due to suboptimal solutions → ToT/LATS.

4. **Incrementally add complexity** — do not jump straight to LATS. Try Plan-and-Execute. If insufficient, add Reflexion. Only if it still does not work, consider LATS.

5. **Profile cost** — each added layer of complexity increases cost and latency. Ensure the improvement justifies the overhead.

---

## Architecture Comparison Table

| Architecture | Complexity | Planning | Learning | Best For |
|--------------|------------|----------|----------|----------|
| ReAct | Low | No | No | Simple tool-based tasks |
| Plan-and-Execute | Medium | Yes | No | Multi-step structured tasks |
| Reflexion | Medium | No | Yes | Tasks with a clear feedback loop |
| LATS | High | Yes | Yes | Complex tasks with uncertainty |
| Cognitive | High | Yes | Yes | Long-term personalized agents |
| **Tree-of-Thought** | Medium | Yes | No | Tasks requiring backtracking |
| **Self-Consistency** | Low | No | No | Improving reliability of discrete answers |
| **MCTS + LLM** | High | Yes | Yes | Search in large solution spaces |

---

## Key Takeaways

1. **ReAct is the workhorse** of the agent world. Simplicity, transparency, and sufficient effectiveness for most tasks make it the starting point for any project.

2. **Plan-and-Execute is necessary for complex tasks** where global vision is critical. Separating planning from execution allows optimizing each phase independently.

3. **Reflexion turns mistakes into lessons**. Instead of blind retries, the agent analyzes failures and improves its approach.

4. **LATS explores alternatives** when the best path is unclear. Tree search enables finding solutions where linear approaches stall.

5. **Cognitive architectures** model human thinking. They are complex to implement but provide the most natural and adaptive behavior.

6. **Tree-of-Thought transforms linear reasoning into trees**. The ability to backtrack is critical for tasks where an error mid-reasoning is fatal.

7. **Self-Consistency is a simple way to improve reliability**. Majority voting across several independent reasoning chains often yields a better result than a single perfect chain.

8. **O1-style reasoning blurs the boundary** between model and agent. When the model itself can "think longer," external agent architectures can be simplified.

9. **MCTS + LLM is a powerful combination** of classical search algorithms and neural evaluation. The UCB formula balances exploitation and exploration.

10. **Choosing an architecture is a trade-off** between simplicity, effectiveness, reliability, and computational cost. Start simple and add complexity only when necessary.

---

## Practical Code Examples

### Minimalist ReAct Agent Implementation

A basic ReAct agent implementation consists of a simple loop with three components: a language model, a set of tools, and an iteration limit.

**Main operation loop:**
1. Start with an empty "scratchpad" (a notebook for reasoning)
2. On each iteration, send the model the system prompt + current scratchpad
3. The model generates a Thought (reasoning) and an Action (with parameters)
4. If the response contains "Final Answer," return the result
5. Otherwise, extract the action name and parameters, execute the corresponding tool
6. Add the tool result to the scratchpad as an Observation
7. Repeat the cycle until an answer is obtained or the iteration limit is exceeded

**Prompt format:** The model is provided a structured format with sections for "Thought" (reasoning), "Action" (tool name), "Action Input" (parameters), "Observation" (execution result). The cycle repeats until a "Final Answer" section appears.

**Applicability:** Tasks solvable in 3-7 sequential steps without complex planning. Limited budget for LLM calls. Examples: information search, simple research tasks, data retrieval.

### Minimalist Plan-and-Execute Agent Implementation

Plan-and-Execute separates the process into two independent phases, each with its own logic.

**Planning phase:**
1. The model receives the task and the list of available tools
2. The prompt requests a numbered step-by-step plan
3. The model generates a sequence of concrete steps with descriptions
4. The plan is parsed into a structured format (a list of PlanStep objects)

**Execution phase:**
1. Steps from the plan are taken one at a time in order
2. For each step, context is built: step description + results of previous steps
3. A determination is made whether a tool is needed (analysis of the step description)
4. If needed — the appropriate tool is selected and invoked
5. If not needed — the LLM is used for reasoning
6. The result is saved for use in subsequent steps
7. After all steps are completed, the final answer is synthesized

**Applicability:** Tasks with explicit structure and dependencies between steps. Optimizing the action sequence matters. Decomposition into 5-15 subtasks. Examples: report generation, ETL pipelines, multi-source research.

### Conceptual Structures of Remaining Architectures

The remaining architectures (Reflexion, LATS, Cognitive) are best understood conceptually, as their full implementation requires a substantial amount of code.

#### Reflexion Agent Implementation: Conceptual Structure

Reflexion adds a learning loop on top of any base architecture through four interacting components:

**Actor** executes an attempt to solve the task. This can be a ReAct agent, a simple CoT, or any other base architecture.

**Evaluator** assesses the result. For tasks with clear criteria, binary success/failure evaluation is used (e.g., passing unit tests). For partial solutions, a numeric scale from 0.0 to 1.0 is applied. Open-ended tasks require LLM-based evaluation, and if a reward model is available, it can be used for complex scenarios.

**Reflector** conducts deep analysis upon failure: determines what exactly went wrong (root cause), explains why the chosen approach did not work, proposes alternative strategies, and formulates lessons for the future.

**Episodic Memory** stores structured reflections. Each entry contains: task type (task_signature), failed approach (failed_approach), root cause (root_cause), extracted lesson (lesson_learned), recommended approach (recommended_approach), and applicability context.

**Operation cycle:** On each attempt, the agent retrieves relevant reflections from memory, enriches the context with extracted lessons, the Actor tries to solve the task, the Evaluator assesses the result. On success, the successful pattern is saved and the result is returned. On failure, the Reflector analyzes the error, saves the lesson, and the next attempt begins.

**Practical recommendations:** Start with a simple evaluator (unit tests, assertions). Store reflections in a vector store for semantic search. Limit reflections in context to the top-3 most relevant. Version reflections, as lessons can become outdated when new tools are added.

#### LATS Implementation: Conceptual Structure

LATS (Language Agent Tree Search) combines Monte Carlo Tree Search with LLMs. This is the most complex architecture, requiring an understanding of search algorithms.

**Tree Node Structure:** Each tree node stores the current state (the sequence of actions up to that point), a reference to the parent, a list of child nodes (alternative actions), a visit counter, accumulated reward, and a list of untried actions.

**Selection Phase** uses the UCB (Upper Confidence Bound) formula to descend through the tree. The formula balances exploitation (selecting nodes with high reward) and exploration (selecting rarely visited nodes). It is computed as Q(node)/N(node) plus an exploration bonus c × sqrt(ln(N(parent))/N(node)), where Q is cumulative reward, N is visit count, and c is the exploration constant (typically 1.41).

**Expansion Phase** generates new child nodes. The LLM creates several possible next actions (typically 3-5), each forming a new node. Diversity sampling with a temperature above zero is used to ensure variety.

**Simulation/Rollout** evaluates a node's promise through rapid "completion" from the current node to a terminal state. Can be LLM-based (more accurate but more expensive) or heuristic (faster but less precise). Returns an estimate of how promising the path is.

**Backpropagation** propagates the reward up the tree from leaf to root, updating visit counters and accumulated values. Nodes on a successful path receive a bonus.

**Applicability:** Tasks with a high branching factor (many alternatives at each step). High cost of error requires an optimal solution. Significant computational budget is available. Intermediate steps can be evaluated. Examples: mathematical proofs, strategic planning, code generation with constraints.

#### Cognitive Architecture Implementation: Conceptual Structure

Cognitive architectures model human thinking structure through a system of specialized interacting modules.

**Declarative Memory** stores facts, knowledge, and experience as chunks with activation levels. Frequently used recent knowledge has high activation and is retrieved faster (activation decay). Implemented via a vector store with metadata: last access time, access counter.

**Procedural Memory** contains skills in the form of IF-THEN production rules. When multiple applicable rules conflict, a conflict resolution mechanism selects the rule with the highest utility. Implemented via a rule engine plus prompt templates with examples.

**Working Memory** — limited "RAM" for current context and active goals. Imitates the limits of human attention (limited capacity). Implemented through context window management and a buffer for intermediate results.

**Goal Module** manages a hierarchical goal stack: the main goal is broken into subgoals, those into sub-subgoals. Push/pop goal operations work similarly to a call stack. Implemented via a goal queue with priorities and dependencies.

**Perceptual-Motor Module** provides the interface with the external world. Perception (sensors) receives information from the environment; action (effectors) affects it. Implemented through a tool interface plus observation processing.

**Processing cycle:** The agent perceives the current state (perceive), retrieves relevant chunks from declarative memory (retrieve), finds applicable production rules (match), selects the rule with the highest utility (select), executes the action (execute), updates activations and potentially creates new chunks (learn), checks goal achievement, and updates the goal stack via pop/push operations.

**Applicability:** Long-lived agents that accumulate experience. Personalization and learning user preferences are important. Tasks require contextual adaptation. Resources for complex implementation are available. Examples: personal assistants, tutoring systems, adaptive NPCs.



---

## Navigation
**Previous:** [[01_What_is_AI_Agent|What is an AI Agent]]
**Next:** [[03_Tool_Use|Tool Use]]
