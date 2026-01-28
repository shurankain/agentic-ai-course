## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Multi-Agent Systems
**Previous:** [[../03_AI_Agents_Core/10_Resource_Optimization|Resource Optimization]]
**Next:** [[02_MAS_Patterns|Multi-Agent System Patterns]]

---

# Multi-Agent System Fundamentals

## Introduction to Multiple Agents

Multi-agent systems use a specialized team of agents instead of a single general-purpose agent. A research agent finds information, an analyst processes it, a writer converts conclusions into readable text. Each agent focuses on its area of expertise, enabling solutions to problems that are beyond the reach of individual agents.

## Why a Single Agent Is Not Enough

Single agents face fundamental limitations:

**Context window** - finite model memory size. When an agent tries to hold task details, accumulated context, and instructions for dozens of tools, it starts "forgetting" important information. Specialized agents focus only on relevant information.

**Conflicting roles** - a critical analyst and an optimistic idea generator require opposite dispositions. Within a single agent, they conflict and dilute quality. Separate critic and generator agents argue with each other, improving the result.

**Scalability** - a single agent either handles the load or it does not. A multi-agent system scales by adding more workers.

**Expertise** - a general-purpose agent is mediocre at everything. Deep specialization of each agent yields better results.

## What Is a Multi-Agent System

A multi-agent system is a computational system where multiple autonomous agents interact to achieve shared or individual goals.

**Autonomy** - agents make decisions independently, without commands at every step. This distinguishes them from traditional distributed systems, where components simply execute assigned functions.

**Local representation** - no single agent sees the full picture. Each observes only a portion of the environment and other agents' actions. This complicates coordination but makes the system more resilient — a failure of one agent does not paralyze the rest.

**Decentralization** - there is no central controller. Agents self-organize to solve tasks. In practice, hybrid approaches with coordination elements are used.

**Emergent behavior** - complex system-level patterns arise from simple interaction rules, much like ant colony organization emerges from individual ant behavior rules.

## Game Theory for Multi-Agent Systems

Game theory provides the mathematical foundation for analyzing interactions among rational agents with shared or conflicting interests.

### Cooperative and Non-Cooperative Games

**Non-cooperative games** model agents acting independently. The "Prisoner's Dilemma" demonstrates that rational agents reach a Nash equilibrium (both defect), even though cooperation would be more beneficial. Individual optimization produces a suboptimal outcome for all.

**Cooperative games** assume coalitions and binding agreements. The question is not "what will each agent choose" but "how to distribute the coalition's payoff."

### Nash Equilibrium

A state where no agent can improve its outcome by unilaterally changing its strategy. Applied in resource conflicts (who yields the GPU), task bidding, and coalition formation.

### Incentive Mechanisms

The inverse problem: how to design rules so that the equilibrium coincides with the desired outcome. Vickrey auctions incentivize honest information disclosure, token-based incentives promote fair work distribution, and reputation systems encourage honest self-assessment.

## Communication Protocols

Communication is the heart of a multi-agent system. Without it, agents remain isolated programs.

### Message Passing

Direct sending of messages to a specific recipient. A message contains: sender, recipient, type (request, command, notification, response), content, and context.

Synchronous passing (sender waits for a response) is simpler but creates bottlenecks. Asynchronous passing (sender continues working) is more efficient but requires callback handling.

### Shared State

Shared memory (blackboard) where agents write to and read from. Advantages: publish-subscribe pattern, persistent context, simplified debugging. Disadvantages: synchronization under concurrent access, performance bottleneck.

### Formalization

**FIPA-ACL** - a standard for message structure: sender, recipient, communicative act (inform, request, query, propose, accept/reject), content, metadata.

**Blackboard Architecture** - a three-component system: the board (multi-level data structure), Knowledge Sources (specialist agents that activate on relevant data), and Scheduler (selects the agent to activate). Effective for tasks with an unknown processing order.

## Roles and Specialization

### Functional Specialization

Division by function. A document processing system: a loader retrieves documents and extracts text, an analyzer extracts structured information, a synthesizer combines results, and a responder interacts with the user.

### Domain Specialization

Division by expertise. A travel planning system: a flight expert, a hotel expert, an excursion expert, and a planner that combines their proposals into a coherent plan.

### Hierarchies

Top-level managers decompose tasks and distribute them among workers. Advantages: clarity, complexity isolation, scalability. Disadvantages: single points of failure, slower decision-making, horizontal coordination problems.

## Coordination Challenges

### Conflicts

Contradictory opinions (optimist vs. critic) or resource competition. Strategies: voting (majority opinion), priorities (a security expert has veto power), escalation (passing the decision upward), negotiation (mutually beneficial compromises).

### The Common Knowledge Problem

It is not enough for each agent to know fact X. Each must know that all others know X, and so on recursively. Agent A is not sure that B received the message; B is not sure that A knows about the receipt. Solution: explicit acknowledgments and state synchronization.

### Deadlocks and Livelocks

Deadlock — mutual blocking (A waits for B's resource, B waits for A's resource). Livelock — active work without progress, endless reactions to each other's actions. Prevention: careful protocol design, timeouts, and stalemate detection mechanisms.

### CAP Theorem

A system cannot simultaneously guarantee consistency (identical state), availability (response to every request), and partition tolerance. CP systems sacrifice availability for consistency (finance, exclusive resources). AP systems sacrifice consistency for availability (drafts, recommendations).

## Lifecycle

**Creation** - the agent receives a role, tools, rules, connections to other agents, and initial context.

**Active state** - the main loop: perceiving messages, processing, decision-making, actions, communication.

**Suspension** - saving state without consuming resources (waiting for an external operation, conserving resources). Resumption from the point of suspension.

**Termination** - after task completion, a stop command, an error, or a timeout. Releasing resources, notifying connected agents, saving results.

## When Multi-Agent Is Necessary

Excessive complexity turns a simple task into a debugging nightmare.

### Five Key Questions

**1. Does it exceed the context window?** A single agent is sufficient if the information fits within the context window. Multi-agent is needed for 100K+ tokens, unlimited history, or dozens of tools.

**2. Conflicting cognitive modes?** Generator vs. critic, explorer vs. executor conflict within a single agent. Multi-agent separates opposing approaches.

**3. Independent scalability?** A single agent scales vertically (more powerful model), multi-agent scales horizontally (more workers). Horizontal scaling is more efficient when load grows on individual functions.

**4. Auditability requirements?** In finance, medicine, and law, tracking who made a decision is important. Splitting into agents creates accountability boundaries.

**5. Are the costs justified?** Multi-agent is more expensive: tokens x2-10, latency x2-5, debugging x5-20, infrastructure x2-5.

### Anti-Patterns

**Multi-Agent for a summary** - 5 agents for 3 paragraphs. A single agent is 10x cheaper.

**An agent per function** - separate agents for email validation or date formatting. Use regular functions.

**Hierarchy for hierarchy's sake** - Supervisor → Manager → Worker for an atomic task. Redundant levels add latency and errors.

**Swarm for deterministic tasks** - if the step order is known, use a pipeline. Swarm is for exploration, not execution.

## Key Takeaways

Multi-agent systems overcome single-agent limitations through specialization, load distribution, and scaling. Key principles: clear roles and boundaries, appropriate communication, conflict resolution, correct lifecycle management, and protection against deadlocks/livelocks.

Communication: message passing (synchronous/asynchronous) or blackboard. Hierarchy is natural but creates single points of failure. Flat structures are more resilient but harder to coordinate.

Make sure the task actually requires multi-agent architecture. Simplicity often outperforms complexity.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Multi-Agent Systems
**Previous:** [[../03_AI_Agents_Core/10_Resource_Optimization|Resource Optimization]]
**Next:** [[02_MAS_Patterns|Multi-Agent System Patterns]]
