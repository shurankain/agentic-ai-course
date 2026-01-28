# What is an AI Agent

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[../02_Prompt_Engineering/05_Context_Engineering|Context Engineering]]
**Next:** [[02_Agent_Architectures|Agent Architectures]]

---

## From Chatbots to Autonomous Systems

A fundamental shift has occurred in the world of artificial intelligence. While the first LLM applications were essentially "smart chatbots" — systems that answer questions in a "question → answer" format — modern AI agents are something fundamentally different. These are autonomous software systems capable of planning, acting, observing results, and adjusting their behavior.

An **AI agent** is a software system that uses a large language model for autonomous task execution through perceiving the environment, making decisions, and performing actions to achieve specified goals.

### Formal Definition of Agency: The BDI Model

The **BDI (Beliefs-Desires-Intentions)** model is one of the most influential formalizations of agency, developed by philosopher Michael Bratman and formalized in AI by researchers Rao and Georgeff.

**Beliefs** — the agent's representation of the state of the world: information from the system prompt, results of previous observations and actions, knowledge from memory, current understanding of the task and progress.

**Desires** — states of the world the agent wants to achieve: the user's goal, sub-goals from planning, constraints that must be respected.

**Intentions** — the chosen course of action the agent commits to: the current execution plan, the selected next step, commitment to completing the initiated action.

The **BDI agent cycle** is a continuous process: the agent perceives the environment and updates its beliefs, then analyzes possible courses of action based on its desires and beliefs, selects specific intentions through means-ends reasoning, executes the planned actions, observes results, and returns to the beginning of the cycle.

Modern LLM agents implicitly implement the BDI model: beliefs are stored in the context window and memory, desires are set through goals in the prompt, intentions are formed during planning and CoT reasoning. Understanding BDI helps design more effective agents: explicit separation between what the agent "knows," what it wants, and what it plans to do improves debugging and predictability.

The key word here is "autonomously." Unlike a simple chatbot that merely answers questions, an agent is capable of planning a sequence of actions, using tools (APIs, databases, file system), remembering context and learning from experience, reflecting on and adjusting its behavior.

---

## Chatbot vs AI Agent: Fundamental Differences

### Limitations of a Traditional Chatbot

Consider the simplest chatbot: a user asks a question, the model generates an answer. The interaction ends there. This approach has fundamental limitations: no access to external data (current exchange rates, weather, latest news are unavailable), cannot perform actions (will explain how to send an email but will not send it), each request is independent (processed in isolation without special memory mechanisms), no planning (complex multi-step tasks are executed "blindly," without a global strategy).

### Transformation into an Agent

An AI agent removes these limitations. When a user provides a task, the agent does not simply generate a response — it analyzes the task, creates a plan, iteratively executes steps, calls external tools as needed, observes results, and adjusts course.

This is similar to the difference between a calculator and a programmer. A calculator gives an answer to a specific expression. A programmer understands the problem, breaks it down into subtasks, uses various tools, tests intermediate results, and changes approach when necessary.

### Connection to Classical AI Planning

LLM agents did not appear in a vacuum — they inherit ideas from decades of research in classical AI planning.

**STRIPS (Stanford Research Institute Problem Solver, 1971)** — one of the first formalisms for automated planning. STRIPS represents the world as a set of predicates (assertions about state), and actions as operators with preconditions and effects. In LLM agents, state corresponds to the context window, actions correspond to tools with descriptions of their capabilities, preconditions are inferred implicitly by the agent through reasoning, effects are observable results after executing an action. The key difference: an LLM agent does not require a formal description of preconditions and effects — it "understands" them from natural language descriptions.

**PDDL (Planning Domain Definition Language)** — the standard language for describing planning problems. Modern agents use JSON tool descriptions that include name, description, and parameters. The LLM "understands" the textual description and implicitly infers preconditions and effects — what PDDL defines explicitly and formally.

**Hierarchical Task Networks (HTN)** — decompose high-level tasks into subtasks. This is exactly what LLM agents do during planning! When an agent receives the task "deploy the application," it decomposes it into high-level steps (build, test, deploy), then breaks each step into concrete actions, and finally executes primitive tasks through tool calls. The difference is that HTN requires explicit task hierarchy definition, while an LLM agent infers it from context understanding.

**Advantages of LLM agents over classical planning:** no formal domain description required, work with natural language, adapt to unexpected situations, can reason about non-formalizable aspects.

**Advantages of classical planning:** correctness guarantees (if a plan is found, it works), optimality (the shortest plan can be found), predictability (deterministic behavior), formal verification.

---

## Agent Operation Cycle: Observe → Think → Act → Reflect

### Conceptual Model

An AI agent's operation is organized as a cycle resembling human cognitive processes. This cycle consists of four main phases:

**Observe** — the agent perceives the current state of the environment: what the user is saying, what results previous actions returned, what the current task is and what has already been done.

**Think** — based on observations, the agent reasons about what to do next. This is the central phase where the LLM makes decisions: which tool to call, what parameters to pass, or whether the task is already complete.

**Act** — the agent executes the chosen action: calls a tool, generates a response to the user, or modifies internal state.

**Reflect** — the agent analyzes the result of the action. Was it successful? Did it move closer to the goal? Does the strategy need to change?

This cycle repeats until the task is completed, limits are exhausted (steps, tokens, time), or the agent determines the task is infeasible.

### Why Reflection Matters

The reflection phase is what distinguishes advanced agents from simple ReAct systems. Without reflection, the agent blindly executes actions and can reach a dead end without realizing it. With reflection, the agent is able to: recognize that the chosen approach is not working, reassess the strategy and try an alternative, extract lessons for future tasks, avoid repeating the same mistakes.

---

## Levels of Autonomy

AI agents can have different levels of autonomy, depending on task criticality and control requirements.

### Human-in-the-Loop

At this level, a human confirms every significant action by the agent. The agent proposes an action, shows what it intends to do, and waits for approval. The user can approve, modify, or reject the proposed action.

This level is suitable for critical high-risk operations (financial transactions), training users to work with the agent, situations where the cost of error is high. Drawbacks: slow, requires constant human attention, not suitable for large-scale operations.

### Semi-Autonomous

The agent operates independently within pre-defined policies but escalates decisions that fall outside those boundaries. Policies define which actions can be performed automatically, which require notification, which require explicit approval, and which are entirely prohibited.

### Fully Autonomous

The agent works completely independently until the task is completed or limits are exhausted. This is suitable for non-critical low-risk tasks, repetitive operations with well-understood patterns, overnight batch processing, tasks where response time is critical. Full autonomy requires well-tuned guardrails, monitoring, and rollback mechanisms.

### Criteria for Transitioning Between Autonomy Levels

**Autonomy Decision Matrix:** Risk assessment is calculated as the product of three factors: Impact (on a 1-10 scale), Probability of agent error (from 0 to 1), and irreversibility of the action (1 minus Reversibility). The resulting Risk Score helps choose the mode: a value below 2 allows full autonomy, 2-5 suggests semi-autonomous mode with policies, and above 5 requires human oversight at every step.

**Criteria for "upgrading" autonomy:** Transitioning from Human-in-the-Loop to Semi-Autonomous requires: accumulated statistics from 100+ tasks with approval rate above 95%, identified patterns of "safe" operations, established policies for automatic decisions, monitoring with alerts configured. Transitioning from Semi-Autonomous to Fully Autonomous: operating in semi-autonomous mode for more than 1 month without critical incidents, error rate below 1% on automatically executed operations, all edge cases documented and handled, automatic rollback mechanisms in place.

**Criteria for "downgrading" autonomy:** Immediately switch to a more controlled mode if: a critical incident is detected, error rate exceeds the threshold (e.g., above 5%), operating conditions change (new API, new task types), users complain about quality.

---

## Agent Components

### High-Level Architecture

Any AI agent, regardless of specific implementation, consists of several key components:

**Brain** — the large language model that makes decisions. This is the agent's "intelligence," responsible for reasoning, planning, and response synthesis.

**Memory** — the system for storing context and experience. Includes short-term memory (current session), long-term memory (across sessions), and working memory (current task state).

**Tools** — interfaces to the external world. Web search, database operations, sending emails, code execution — all of these are tools that extend the agent's capabilities.

**Orchestrator** — manages the agent's operation cycle. Responsible for transitions between phases, error handling, enforcing limits, and guardrails.

### Component Interaction

When the agent receives a task, the orchestrator launches the processing cycle. The Brain analyzes the task, consulting Memory for relevant context. Based on reasoning, the Brain selects the next action — a Tool call or response generation. The result of the action is saved in Memory and passed back to the Brain for the next iteration. The orchestrator tracks state, handles errors, and ensures completion.

---

## Task Types for Agents

AI agents are not universal. Different task types require different agent configurations:

**Research** — searching and synthesizing information from various sources. The agent needs web search tools, the ability to analyze and compare information, and good memory for accumulating findings.

**Data Analysis** — processing and analyzing structured data. Requires tools for working with databases, statistical calculations, and visualization.

**Automation** — executing sequences of routine operations. Key tools: file handling, API integrations, command execution.

**Content Creation** — generating text, code, and documents. A quality Brain (powerful model) and well-crafted prompts are essential here.

**Decision Support** — assisting in data-driven decision making. Requires a combination of analytics, visualization, and quality synthesis.

---

## Advantages and Limitations of AI Agents

### Advantages

**Autonomy** — the ability to operate without constant human oversight. An agent can process hundreds of tasks overnight while the team sleeps.

**Adaptability** — the ability to handle new situations. Unlike rigidly programmed automation, an agent can improvise and find solutions to previously unseen problems.

**Scalability** — easy scaling to large task volumes. A single agent can analyze thousands of documents or respond to hundreds of tickets.

**Consistency** — uniform quality of work regardless of time and workload. An agent does not get tired, distracted, or make errors due to inattention.

**Integration** — the ability to work with different systems through tools. An agent can simultaneously use ten different APIs, which would be difficult for a human to achieve.

### Limitations and Risks

**Hallucinations** — LLMs can generate incorrect information with full confidence. This is especially dangerous for agents that make decisions based on their "knowledge." Mitigation: result verification, using RAG for ground truth, guardrails for critical decisions.

**Limited Context** — the LLM's context window limits the amount of information available to the agent at any given moment. For complex tasks with large context, this can be a problem. Mitigation: summarization, RAG, hierarchical memory, intelligent context prioritization.

**Cost** — LLM API calls can be expensive at scale. An agent making dozens of calls per task can become prohibitively expensive. Mitigation: caching, prompt optimization, selecting the appropriate model for the task.

**Unpredictability** — LLM behavior is not fully deterministic. The same request can produce different results, complicating testing and debugging. Mitigation: testing, guardrails, human-in-the-loop for critical operations.

**Security** — risks of prompt injection, data leaks, and undesirable actions. An agent with access to the file system or database can cause real harm if attacked. Mitigation: input validation, sandboxing, least privilege, audit logs.

---

## When to Use Agents (and When Not To)

### Good Candidates for Agents

Tasks require multiple steps with external interactions, there is a clear goal but the path is unknown, some behavioral uncertainty is acceptable, the benefits of automation outweigh the risks. Examples: "Analyze the repository and suggest improvements," "Find information and compile a report," "Process a list of tasks in Jira."

### When Agents Are Not Needed

Simple tasks solvable with a single LLM call ("Translate the text"), deterministic operations ("Calculate the sum," "Execute an SQL query"), critically important operations requiring 100% reliability, tasks without a clearly defined goal.

Rule of thumb: if a task can be solved with a single LLM call or simple code, an agent is overkill. If the task requires multiple steps with a path unknown in advance, an agent may be useful.

---

## Iterative Approach to Building Agents

### Principle: "Start simple, add complexity as needed"

One of the key mistakes when building AI agents is attempting to build a complex system from the outset. The Anthropic team emphasized in their December 2024 research: "The most successful implementations weren't using complex frameworks or sophisticated architectures. They were building with simple, composable patterns."

### Step-by-Step Agent Evolution

Agent system evolution happens in stages, from simple to complex:

**Level 1: AUGMENTED LLM** — a base language model with a well-crafted system prompt. This is sufficient for simple tasks: question-answering, text classification, basic content generation.

**Level 2: + TOOLS** — adding the ability to call tools when external actions are needed: API access, database operations, file read/write, command execution.

**Level 3: + MEMORY** — implementing a memory system when interaction history matters: multi-turn dialogues, response personalization, context accumulation across sessions.

**Level 4: + REASONING PATTERNS** — using reasoning patterns (ReAct, Chain-of-Thought) for complex tasks requiring multi-step analysis and planning.

**Level 5: + MULTI-AGENT** — transitioning to a multi-agent architecture only when a single agent is insufficient: different specializations are required, parallel task processing, distributed decision making.

### Continuous Evaluation

Evaluation is not the final stage after all improvements. It is a continuous process that accompanies development from the very beginning. For each level, define: how to measure success (metrics), which edge cases are critical, what constitutes failure. Only after the current level is working and evaluated should you move to the next.

### Practical Recommendation

When starting a new project: define the MVP (what is the minimum functionality needed for initial value?), start with Level 1-2 (Augmented LLM plus necessary tools), measure (what works, what does not?), add complexity as needed (only when the current level is insufficient), document the reasons (why was this component added?).

This approach reduces risk, accelerates time-to-value, and avoids unnecessary complexity.

---

## Key Takeaways

1. An **AI agent** is an autonomous LLM-based system capable of planning, acting, observing results, and reflecting

2. The **agent cycle** (Observe → Think → Act → Reflect) is the architectural foundation of any agent; reflection distinguishes advanced agents from simple ones

3. **Levels of autonomy** range from human-in-the-loop to fully autonomous systems; the choice depends on criticality and risks

4. **Agent components**: brain (LLM), memory, tools, and orchestrator — all are necessary for full-featured operation

5. **Agents excel** at multi-step tasks with external interactions and clear goals

6. **Limitations** (hallucinations, cost, unpredictability, security) require mitigation strategies

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[../02_Prompt_Engineering/05_Context_Engineering|Context Engineering]]
**Next:** [[02_Agent_Architectures|Agent Architectures]]
