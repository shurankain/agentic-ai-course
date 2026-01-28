## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Multi-Agent Systems
**Previous:** [[03_Agent_Orchestration|Agent Orchestration]]
**Next:** [[05_MAS_Consensus_and_Reliability|MAS Consensus and Reliability]]

---

# Frameworks for Multi-Agent Systems

## Why Frameworks Are Needed

Building a multi-agent system from scratch is a large-scale undertaking. You need to implement inter-agent communication mechanisms, lifecycle management, task routing, error handling, state persistence, observability, and much more. Each of these components requires careful design and debugging.

Frameworks for multi-agent systems take on this infrastructure work. They provide ready-made abstractions for creating agents, proven interaction patterns, and tools for debugging and monitoring. Developers can focus on the business logic of their agents without reinventing basic coordination mechanisms.

Choosing a framework is an important architectural decision. Different frameworks make different trade-offs between simplicity and flexibility, between feature completeness and ease of adoption.

## Formal Framework Selection Criteria

### Evaluation System

**Expressive Power:** which MAS patterns can be expressed? Try implementing supervisor, handoff, debate.

**State Management:** how is state managed between agents? Explicit vs implicit, persistence.

**Error Handling:** built-in error handling mechanisms. Retry, fallback, timeout policies.

**Observability:** tracing, logging, metrics. Integration with LangSmith, OpenTelemetry.

**Testability:** how to test agents? Unit tests, mocking, replay.

**Learning Curve:** adoption difficulty. Time to first working system.

**Production Readiness:** readiness for production. Stability, community, documentation.

### Quantitative Evaluation

Framework Score is calculated as the sum of products of criterion weight and framework score per criterion, divided by the sum of weights. Weights depend on the use case. For a prototype, Learning Curve matters more (weight 10); for enterprise, Error Handling, Observability, and Production Readiness are critical (weights 9-10); for research, Expressive Power is key (weight 10).

### Decision Tree

Start with the question: is precise coordination with checkpoints needed? If yes, choose LangGraph. If no, check whether the agents are conversational. If yes, choose AutoGen. If no, check whether the roles are simple. If yes, choose CrewAI. If no, choose Semantic Kernel.

## LangGraph: Graph-Oriented Orchestration

LangGraph is an extension of the LangChain ecosystem, specifically designed for building complex, multi-step agent systems. The central idea is representing the system as a directed graph where nodes are operations (LLM calls, tools, conditions) and edges are transitions between them.

### Graph Philosophy

Why graphs? Traditional chains in LangChain are linear: one step follows another. But real agent systems are rarely linear. An agent may return to a previous step for clarification. Multiple agents may work in parallel. The decision about the next step may depend on the result of the current one.

A graph naturally expresses these nonlinear flows. Cycles in the graph allow an agent to iterate until a result is achieved. Branches allow routing to different handlers depending on conditions. Parallel paths allow multiple agents to work simultaneously.

### State as a First-Class Entity

A key feature of LangGraph is explicit state management. The graph state is a data structure that is passed between nodes and accumulates information as execution progresses. Each node receives the current state, executes its logic, and returns state updates.

This approach makes the system predictable and debuggable. You can save the state at any point and resume execution from the same place. You can inspect the state change history and understand how the system arrived at the current result.

### Internal Architecture

**State Machine Semantics:** LangGraph implements a formal finite automaton model with five components: a set of states (graph nodes), an event alphabet (edge types), a transition function (conditional edges), an initial state (entry point), and final states (END). This mathematical foundation ensures system predictability and verifiability.

**Checkpoint Mechanism:** LangGraph supports a checkpoint mechanism for failure recovery. Each checkpoint contains the current graph state, the active node name, transition history, and a timestamp. Saving occurs automatically after each node execution.

Checkpointing advantages: Resume after crash (the system automatically continues from the last saved point), Time travel debugging (you can return to any checkpoint for analysis), Fork execution (continue execution from one point along several different paths), Human-in-the-loop (pause execution, modify state manually, continue).

**Reducers for state management:** when concurrent state updates occur (when multiple nodes try to modify the same field), LangGraph uses reducer functions. Instead of simply overwriting a value, the reducer defines the logic for merging old and new values. For example, for a message list, the reducer does not replace the entire list but appends new messages to the existing ones.

### When to Use

LangGraph is particularly well-suited for systems where precise coordination between agents and result reproducibility matter. If you need to build an agent with a clearly defined process (for example, step-by-step document analysis), LangGraph provides tools for explicitly describing this process. The framework also integrates well with LangSmith for observability, which is important in production systems.

## AutoGen: Multi-Agent Dialogue from Microsoft

AutoGen is a framework from Microsoft Research that focuses on conversational agents -- agents that interact with each other through natural dialogue.

### Agents as Conversationalists

The central metaphor of AutoGen is group chat. Multiple agents participate in a conversation, each with their own role and expertise. One agent generates code, another reviews it, a third runs tests, a fourth coordinates the discussion. They communicate with each other via text messages, like people in a chat.

This approach is intuitive and flexible. There is no need to design complex interaction schemas -- agents simply "talk" until they reach a solution. New agents can easily be added to the conversation.

### Automatic Interaction

The name AutoGen reflects the automatic nature of interaction. When one agent sends a message, the framework determines who should respond and automatically triggers that agent. The chain continues until a termination condition is met.

This simplifies development -- there is no need to manually program each transition. But it also creates the risk of infinite loops or unpredictable behavior if agents are not configured carefully.

### Human-in-the-Loop

AutoGen provides good support for including a human in the process. You can configure an agent to request human confirmation before critical actions. A human can intervene at any point by adding their message to the dialogue.

### AutoGen 0.4: New Architecture

In late 2024, AutoGen introduced version 0.4 with a completely redesigned architecture. Key changes:

**Event-Driven Architecture:** the old model used direct synchronous interaction between agents. The new model introduces an event-driven architecture with a central event bus: Agent A publishes an event, the Runtime receives it, analyzes it, and routes it to Agent B asynchronously. This makes the system more scalable and allows easier isolated testing of components.

**Typed Messages:** version 0.4 uses strictly typed messages instead of arbitrary strings. Each message type is a data structure with defined fields. Strict typing prevents errors at development time and improves IDE autocompletion.

**Modular Components:** the 0.4 architecture is divided into five key components: Agent (base actor with managed lifecycle), Runtime (manages execution and message routing), Memory (pluggable memory stores), Tools (standardized tool interface), Teams (high-level orchestration of agent groups).

**Improved Testability:** the new architecture significantly simplifies testing. Mock runtimes allow writing isolated unit tests for agents. Deterministic execution mode guarantees reproducible results. Replay capabilities allow recording real interactions and replaying them for debugging.

AutoGen 0.4 is incompatible with version 0.2 -- a full code rewrite is required.

## CrewAI: Role-Based Collaboration

CrewAI offers the metaphor of a crew -- a group of specialists, each with their own role, working toward a common goal.

### Roles and Tasks

In CrewAI, you define agents through their roles: researcher, writer, editor, analyst. Each role has a description, a set of tools, and a work style. Then you define tasks -- specific units of work that need to be completed.

The framework automatically assigns tasks to agents based on role matching. Research tasks go to the researcher, writing tasks go to the writer. This creates an intuitive structure similar to how work is organized in real teams.

### Sequential and Parallel Processes

CrewAI supports two main execution modes. In sequential mode, tasks execute one after another, with each subsequent task receiving the result of the previous one. In hierarchical mode, one agent (the manager) coordinates the work of the others.

The simplicity of the model is both a strength and a limitation of CrewAI. It is easy to get started and create a working system. But for complex scenarios with nonlinear flows, the expressiveness may be insufficient.

### Ideal Scenarios

CrewAI is well-suited for content-generation workflows: topic research, article writing, editing, fact-checking. It also works well for analytical tasks where data needs to be collected from different sources and synthesized into a report.

## Semantic Kernel: Enterprise Orchestration from Microsoft

Semantic Kernel is an SDK from Microsoft for integrating AI into enterprise applications. Unlike more experimental frameworks, Semantic Kernel is oriented toward production-ready use in enterprise environments.

### Plugins and Functions

The main abstraction in Semantic Kernel is plugins containing a set of functions. Functions can be native (regular code in C#, Python, or Java) or semantic (prompts to an LLM). The Kernel orchestrates function calls, manages context, and provides integration with various LLM providers.

This approach allows gradually adding AI capabilities to existing applications. There is no need to rewrite everything from scratch -- it is enough to wrap existing services in plugins and add semantic functions for new functionality.

### Planners

For multi-step tasks, Semantic Kernel offers planners. A planner receives a goal in natural language and automatically composes a plan from available functions. The plan is then executed step by step.

This is a powerful mechanism, but it requires careful tuning. An LLM may compose a suboptimal plan or attempt to call nonexistent functions. A good practice is to validate plans before execution.

### Enterprise Features

Semantic Kernel is developed with enterprise requirements in mind: security, regulatory compliance, Azure integration. For organizations already using the Microsoft stack, it is a natural choice.

## OpenAI Swarm: Experimental Simplicity

OpenAI Swarm is an experimental framework from OpenAI demonstrating a minimalist approach to multi-agent systems.

### Minimalism Philosophy

Swarm is intentionally simple. An agent is just a set of instructions and functions. Interaction between agents is accomplished through handoff -- transferring control from one agent to another.

There are no complex orchestration mechanisms, no state persistence, no built-in monitoring. Swarm is more of a pattern or reference implementation than a full-fledged production framework.

### When to Use

Swarm is suitable for prototyping and experimentation. If you want to quickly validate a multi-agent system idea without diving into complex frameworks, Swarm provides enough structure with minimal overhead.

For production systems, you will need to either significantly extend Swarm or switch to a more full-featured framework.

## Comparison and Selection

Each framework occupies its own niche. The choice depends on the specific requirements of the project.

**LangGraph** -- choose when precise coordination and reproducibility are needed. The graph model gives full control over the execution flow. Integration with the LangChain ecosystem is an additional advantage.

**AutoGen** -- suitable for conversational scenarios where agents naturally interact through dialogue. Particularly strong when human inclusion in the process is needed.

**CrewAI** -- a good choice for content-generation and analytical tasks with a clear division of roles. Low barrier to entry allows getting results quickly.

**Semantic Kernel** -- for enterprise applications, especially within the Microsoft ecosystem. The plugin architecture facilitates integration with existing systems.

**Swarm** -- for rapid prototyping and experimentation, when simplicity matters more than feature completeness.

## Key Takeaways

Frameworks for multi-agent systems significantly accelerate development by providing ready-made solutions for common orchestration, communication, and state management tasks.

LangGraph offers a graph-oriented approach with explicit state management. Well-suited for complex, nonlinear flows requiring reproducibility.

AutoGen focuses on conversational interaction between agents. The intuitive group chat model simplifies development but requires attention to preventing infinite loops.

CrewAI uses the metaphor of a crew with roles and tasks. The simplicity of the model makes it accessible for quick starts but limits flexibility for complex scenarios.

Semantic Kernel is oriented toward enterprise with a plugin architecture and planners. A good choice for gradually introducing AI into existing enterprise systems.

Swarm demonstrates a minimalist pattern. Useful for prototyping but requires further development for production.

When choosing a framework, consider not only current requirements but also the system's growth prospects. Starting with a simple framework and migrating later is a reasonable strategy, provided that migration is technically feasible.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Multi-Agent Systems
**Previous:** [[03_Agent_Orchestration|Agent Orchestration]]
**Next:** [[05_MAS_Consensus_and_Reliability|MAS Consensus and Reliability]]
