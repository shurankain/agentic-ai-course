# Consensus and Reliability of Multi-Agent Systems

## The Problem of Decision Alignment

In multi-agent systems, agents often must arrive at a unified decision. The output of one agent affects others, and an error can cascade throughout the entire system.

The problem is especially relevant in AI systems: agents can hallucinate, be overconfident in incorrect answers, or see different parts of information. Consensus mechanisms transform independent opinions into aligned, reliable decisions.

---

## Types of Consensus in AI Systems

### Consensus on Fact

The simplest case: agents must agree on a factual statement. Is this email spam? Is the tone of this review positive or negative? Does this code contain a vulnerability?

Relatively simple mechanisms work here: voting, averaging confidence scores, selecting the most confident answer. The key condition is that the task has an objectively correct answer, even if the agents do not know it.

### Consensus on Action

A more complex case: agents must agree on what to do. Which strategy to choose? Which tool to use? How to decompose the task?

There is no single correct answer here. Different strategies can be equally valid. Consensus mechanisms must select one option and ensure that all agents will follow the chosen plan.

### Consensus on State

The most technically complex case: agents must have a consistent view of the system state. Which tasks have already been completed? Which resources are occupied? What is the current conversation context?

This is a classic distributed systems problem, complicated by the fact that LLM agents can "forget" or distort information. Synchronization protocols are needed, and so are state verification mechanisms.

---

## Voting Mechanisms

### Simple Majority

The most intuitive mechanism: the majority opinion wins. If three agents out of five believe the answer is "A", we accept "A".

Simple majority works well when agents are roughly equal in competence and independent in their judgments. Condorcet's jury theorem states that if each "voter" is correct with probability greater than 0.5, the probability of a correct majority decision approaches 1 as the number of voters increases.

But in AI systems, the theorem's conditions are often violated. Agents may use the same base model and make correlated errors. An agent with confidence 0.99 votes equally with an agent with confidence 0.51. Some agents may be specialists in the given question, while others are not.

### Weighted Voting

A more advanced approach is to weight the votes. The weight can be determined by several factors.

**Confidence score**: agents more confident in their answer receive greater weight. This works if models are well-calibrated (i.e., confidence actually reflects the probability of correctness). Unfortunately, LLMs are often overconfident, which reduces the usefulness of this approach.

**Historical accuracy**: agents that have given correct answers more often in the past receive greater weight. This requires accumulating statistics and works for tasks where correctness can be verified after the fact.

**Expertise relevance**: an agent specializing in the given task type receives greater weight. A security expert agent should have more influence on vulnerability questions than a general-purpose agent.

### Supermajority

For critical decisions, one can require not a simple majority but a supermajority, for example, 2/3 or 3/4 of votes. This reduces the risk of accepting an erroneous decision due to a random tilt, but increases the probability of a "deadlock" when no option gathers enough votes.

### Quorum

The quorum mechanism requires that a minimum number of agents participate in the vote. A decision made by two agents out of ten may be less reliable than a decision made by eight out of ten (even if the vote proportion is the same).

---

## Adversarial Consensus: Debates and Verification

### The Adversarial Principle

Sometimes the best way to find the truth is to organize a debate. Two agents take opposing positions and try to convince a third (the judge) of their correctness. During the debate, the strengths and weaknesses of each position are revealed.

This approach is inspired by the legal system, where the prosecutor and defense attorney compete before a judge, and by scientific discussions, where an opponent tries to find flaws in the argumentation.

### Debate Structure

Effective AI debates usually follow a structured format:

**Round 1: Initial positions.** Each debater agent receives the question and formulates their position with arguments.

**Rounds 2-N: Rebuttals.** Agents see opponents' positions and attempt to refute them. They point out logical errors, provide counterexamples, and challenge premises.

**Final round: Judge.** An independent agent (or multiple agents) evaluates the arguments of both sides and renders a decision. The judge did not participate in the debate and evaluates only the quality of argumentation.

### When to Use Debates

Debates are especially useful in several situations.

**High stakes.** When the cost of error is high, the additional expense of debates is justified. Credit approval decisions, medical diagnostics, legal analysis.

**Subjective questions.** When there is no objectively correct answer, debates help reveal different perspectives and their justifications.

**Complex reasoning.** When a decision requires multi-step logic, debates force agents to make each reasoning step explicit, which makes finding errors easier.

### Problems with the Adversarial Approach

Debates are not a panacea. If both debaters use the same model with similar prompts, they may have the same "blind spots." A more persuasive agent is not necessarily more correct: LLMs can generate very convincing but false argumentation. The judge can also make mistakes or be biased.

---

## Distributed Consensus Protocols

### Why Formal Protocols Are Needed

In distributed systems, simple voting is insufficient. Messages can be lost or delayed. Agents can "crash" in the middle of a protocol. Different agents can see different sequences of events.

Formal consensus protocols such as Paxos and Raft solve these problems with mathematical guarantees. They ensure **safety** (the system will never accept contradictory decisions) and **liveness** (the system will eventually make a decision if enough agents are operational).

### Crash Fault Tolerance (CFT)

CFT protocols protect against "honest failures", when agents simply stop working but do not lie or behave maliciously.

**Raft**: the most understandable CFT protocol. The system elects a leader who coordinates all decisions. If the leader crashes, a new one is elected through an election procedure. A decision is considered accepted when the majority of agents have confirmed it.

In the context of MAS, Raft can be used to coordinate access to shared resources, to ensure task execution order, and to synchronize state between agents.

### Byzantine Fault Tolerance (BFT)

BFT protocols protect against more serious threats, when some agents can behave arbitrarily badly: lie, send contradictory messages to different agents, or attempt to sabotage consensus.

The classic result states that BFT is only possible when the number of "honest" agents exceeds 2/3 of the total. That is, a system of 10 agents can tolerate at most 3 "Byzantine" agents.

#### Why Exactly 2/3?

The mathematical intuition behind this theorem is simple. Consider a scenario with three agents (A, B, C), where one is Byzantine. If malicious agent B sends agent A the message "voting for X" and agent C the message "voting for Y", then agent A, seeing two votes for X (its own and B's), accepts decision X. Agent C, seeing two votes for Y (its own and B's), accepts decision Y. Consensus is broken: honest agents have accepted contradictory decisions.

With four agents and one Byzantine, the situation changes: three honest agents can compare received messages, detect contradictions in the Byzantine agent's messages, and ignore it. The honest majority (3 out of 4) always exceeds the number of malicious agents (1) plus uncertainty, which allows consensus to be reached. Hence the requirement: at least 2/3 honest agents.

#### PBFT (Practical Byzantine Fault Tolerance)

PBFT is a practical BFT protocol for systems with a known set of participants. The protocol operates in five phases:

**REQUEST:** the client sends a request to the primary node (Primary).

**PRE-PREPARE:** the Primary broadcasts a proposal to all replicas about the order of request processing.

**PREPARE:** each replica confirms that it received the pre-prepare and broadcasts this confirmation to all other replicas.

**COMMIT:** after receiving a sufficient number of PREPARE messages, the replica broadcasts COMMIT, signaling readiness for execution.

**REPLY:** replicas execute the operation and send the result to the client.

A decision is considered accepted when 2f+1 replicas have agreed (where f is the maximum number of Byzantine agents).

**Adaptation to AI systems:**

In classic PBFT, the Primary proposes the order of operations; in AI systems, the Supervisor proposes task decomposition. Replicas validate the proposed order; AI agents validate the coherence of the plan. The Commit phase means the beginning of operation execution; in AI systems, this is the beginning of subtask execution. Byzantine behavior in classic systems means outright lying; in AI systems, this means model hallucinations or successful prompt injection attacks.

In AI systems, BFT can be relevant when some agents use less reliable models prone to hallucinations, when the system is susceptible to prompt injection attacks that can "compromise" individual agents, or when agents receive information from unreliable external sources.

#### Lightweight Alternatives for AI

Full-scale PBFT is rarely necessary in typical AI systems due to high overhead costs. More practical alternatives:

**Cross-validation:** the system requests answers from all agents, groups them by semantic similarity (not exact text match), and selects the group containing at least 2f+1 answers, where f is the expected number of potentially erring agents. If no such group exists, the system returns "uncertain" instead of a potentially incorrect answer.

**Redundancy with verification:** three agents independently solve the task, a fourth verifier agent checks answer compatibility. When discrepancies are detected, additional agents are requested to resolve the conflict. This approach is simpler than PBFT but provides sufficient protection for most practical applications.

### Practical Applicability

Full-scale BFT protocols are rarely needed in typical AI systems. They are complex to implement and require significant communication overhead. Lighter approaches are more commonly used:

**Redundancy**: critical operations are performed by multiple agents independently, and results are compared.

**Verification**: the result of one agent is checked by another agent (or several).

**Audit trails**: all actions are logged with cryptographic protection against tampering. This allows detection of incorrect behavior after the fact.

---

## Market-Based Consensus Mechanisms

### Agents as Economic Actors

An interesting approach to consensus is to represent agents as market participants. Agents "bet" on their answers, and the market price reflects the collective confidence in a particular outcome.

### Prediction Markets

Each agent can "buy" or "sell" contracts on a specific outcome. If an agent is confident that the answer is "A", it buys contracts on "A". The contract price reflects the aggregated probability of the outcome.

The advantage of the market mechanism is that it automatically weights opinions by degree of confidence (agents bet more on what they are confident about) and by historical accuracy (agents that were frequently wrong have "lost wealth" and cannot bet much).

### Task Auctions

Another application of market mechanisms is task distribution through auctions. Agents "bid" for the right to execute a task, offering execution time, expected quality, or other metrics.

This creates incentives for honest self-assessment: if an agent overestimates its abilities and wins the auction, it will receive a task it cannot perform well, and its reputation will suffer.

### Limitations of Market Mechanisms

Market mechanisms require a "currency" and a mechanism for crediting/debiting it. They work well only with a sufficient number of participants and transactions. They can produce undesirable behavior (manipulation, cartel collusion) if not configured properly.

---

## Error Propagation: How Errors Spread in MAS

### Formal Model of Error Propagation

Before examining specific patterns, it is useful to have a formal model for analysis. Represent the system as a graph where agents are vertices and dependencies between them are edges. For example, if agent A passes data to agents B and C, and B and C pass data to agent E, and B also passes data to D, then an error in agent A can potentially reach all agents in the system (B, C, D, E). An error in agent B can reach only D and E, and an error in agent C will reach only E.

**Probabilistic model:** denote p(e|A) as the probability of agent A's error. If agent B depends on agent A, then the probability of error in B consists of B's own error probability plus the probability of A's error multiplied by the probability that this error propagates from A to B. Formula: p(error_B) = p(e|B) + p(e|A) x p(propagation|A to B).

**Factors affecting propagation speed:**

**Coupling:** high propagation occurs when agent B directly uses agent A's output without processing; low, when B only checks A's results or uses them selectively.

**Validation:** if B does not validate input data from A, errors easily penetrate further; if B critically checks all inputs, most errors are blocked.

**Error type:** subtle logical errors are hard to detect and propagate easily; obvious syntactic errors are usually caught immediately.

**Downstream (number of dependent agents):** the more agents that depend on a given agent, the wider the potential blast radius when an error occurs.

### Cascading Errors

In a multi-agent system, an error from one agent can propagate to others. Agent A misunderstood the task and passed distorted context to agent B. Agent B, working with incorrect data, drew erroneous conclusions and passed them to agent C. By the end of the chain, a small initial error has turned into a completely incorrect result.

**Cascade amplification:** the initial error in agent A (one error) produces derivative errors in agent B (the original error plus two new ones based on it, totaling three). Agent C receives all previous errors plus creates four new ones based on the distorted data (totaling seven). With each level, the number of errors grows exponentially, reaching 2^N - 1 after N agents.

This is especially dangerous in long processing chains and in systems with feedback loops (where results from later agents affect inputs of earlier ones).

### Knowledge Provenance

A key tool for combating error propagation is **knowledge provenance**. Each fact or assertion in the system should be accompanied by information about its origin:

**Who generated it?** Which agent produced this knowledge? With what parameters (model, prompt, temperature)?

**Based on what?** What input data was used? What previous facts served as the basis for the inference?

**When?** Creation timestamp. This is important for understanding relevance and for debugging.

**With what confidence?** Confidence score, if available.

Provenance allows tracing the source of a detected error and all affected conclusions.

### Confidence Propagation

Confidence in a conclusion cannot be higher than confidence in the premises. If agent A produced a fact with confidence 0.7, and agent B used it for inference, the confidence of B's result should be no more than 0.7 (and usually less, because the uncertainty of the inference itself is added).

Formally, if conclusion Y depends on premises X1, X2,... Xn with confidence c1, c2,... cn, then the confidence of Y is usually bounded by min(c1, c2,... cn), or computed by a more complex formula depending on the type of inference.

This creates a useful signal: if the confidence of the final result is very low, it means there were unreliable data or inferences somewhere in the chain.

### Verification Points

A good practice is to embed verification points throughout the pipeline. After each critical step, an independent agent checks the result before passing it further.

This increases cost and latency but prevents error propagation. The frequency of verification points is a tradeoff between reliability and efficiency.

### Circuit Breakers

By analogy with electrical fuses, a circuit breaker in MAS is a mechanism that "disconnects" part of the system when problems are detected.

If an agent starts returning results with very low confidence, or its answers contradict baseline checks, or it responds too slowly, the circuit breaker switches the system to "safe mode": routing requests to other agents, requesting human review, or returning an honest "I don't know" instead of a potentially incorrect answer.

---

## What Actually Goes Wrong: Measured Failure Modes

Everything above is a model of how multi-agent systems should fail. In August 2026 Anthropic published measurements of how they do fail, across security research, software construction, and market simulations. The results are worth reading closely, because several of them contradict what the theory predicts.

### Coordination buys breadth, and you pay for it

In a security experiment, 45 agents were given individual VMs, a shared forum, and 15 open-source projects to audit. The coordinated swarm consumed **27 million tokens** against **6.5 million** for the same agents working independently in parallel.

The interesting result is not which found more. It is that **only 12 findings overlapped** between the two approaches. About half the swarm's findings lay outside the directories assigned to the independent group, and once the scope was equalized, cost per finding became comparable. The swarm was not better at auditing. It was searching somewhere else.

Coordination also produced behaviour the independent agents did not show: agents self-organized into specialized roles and wrote their own tools for the vulnerability classes they had claimed.

So the honest framing is a trade rather than a ranking. **A swarm buys breadth of search, and pays for it in duplication, conflicts, and a coordination system you now have to operate.** On a task with one narrow output, that freedom converts directly into spend with nothing to show for it. On an open-ended search across a space you cannot fully specify in advance, it finds things a scoped search structurally cannot.

### Coordination quality degrades with team size

In a second experiment, teams of 10 to 80 agents built a web-playable game over 12 hours. Pull request merge rates **fell as team size grew**. Two older models opened 876 and 980 pull requests respectively while closing few of them, producing work that could not be integrated.

Three prompt conditions were tested: baseline collaboration, prescriptive role assignments, and a CEO-style hierarchy. **Prescriptive roles and hierarchy produced negligible improvement.** This is the finding that should change how the orchestration patterns earlier in this module are read: the org chart is not the lever. Only a newer model maintained high merge fractions while sharing code across the team, which means the ceiling moved with model capability rather than with coordination design.

Newer models also partitioned the repository better on their own. Resource ownership still had to be solved at the system level, because a model that avoids conflicts more often is not the same as a system where conflicts cannot happen.

### Five failure modes with numbers attached

**Herding.** 18 of 30 agents independently created a git branch with the identical name `mvp-game-loop`. In a separate task, agents titled their first submissions identically without coordinating. Given an open brief, more than half chose to build ray tracers or self-hosting compilers. Agents drawn from the same model do not explore independently; they converge on the same modal answer, which means N agents can give you far less than N times the coverage.

**Resource-management collapse.** A job queue received **2.4 million polling requests** at 30 per second and accepted **117 jobs**. Nothing was wrong with any individual agent. Each was politely checking whether work was available.

**False consensus.** A unique fact held by one agent was lost on the way to the arbiter, after which the group adopted an incorrect consensus with confidence. This is the error-propagation model above, observed rather than derived: the failure is not that an agent was wrong, but that the aggregation step destroyed the one piece of evidence that would have corrected it.

**Conflict escalation.** Competing agents locked each other's accounts, triggered kill loops, and multiplied scripts.

**Emergent collusion.** In market simulations, agents established explicit price floors by the third round, and kept price-matching through public listings **after the communication channels between them were removed**. Coordination did not require a channel; a shared observable was enough. In a prisoner's dilemma, participants defected simultaneously and tanked the collective reward, so better coordination did not produce more prosocial behaviour.

### The design consequence: a swarm is a distributed system

None of these failures is about model quality, and none is fixed by a better prompt. They are the classic failure modes of distributed systems, arriving in a field that has been discussing agents as if they were conversations.

Treat a production swarm accordingly. Every shared resource needs an owner or a lease. Queues need backpressure, because agents will poll as fast as you let them. Writes need idempotency keys. Budgets are capped per worker **and** per run, since a runaway loop respects neither alone. An arbiter should verify evidence and side effects rather than tallying votes, and a **minority report should be preserved separately from the consensus**, which is the direct fix for the false-consensus mode above and a concrete improvement to the voting mechanisms described earlier in this chapter. Agents with conflicting objectives belong in isolated environments rather than a shared one.

### Testing whether multi-agent pays for your task

The experiment is cheap enough to run before committing to an architecture. Give three configurations the same token budget, the same wall-clock budget, and the same scope: a single agent, N independent agents, and a coordinated swarm. Then compare confirmed unique findings, overlap between configurations, merge conflicts, out-of-scope actions, and cost per confirmed finding.

Equalizing scope is the part people skip, and it is the part that decides the answer. The Anthropic numbers only became comparable once both groups searched the same space.

## Quality Assessment of Multi-Agent Systems

### Why Metrics Are Needed

Without measurements, improvement is impossible. "The system works well" is not a metric. "The system answers correctly in 87% of cases with average latency of 2.3 seconds at a cost of $0.05 per request", that is a metric you can work with.

For multi-agent systems, metrics are needed at several levels: for individual agents, for interactions between them, and for the system as a whole.

### Agent-Level Metrics

**Accuracy**: the proportion of correct answers. For tasks with an objective answer, this is the primary quality metric.

**Precision/Recall/F1**: for classification or extraction tasks. Precision shows how many of the returned results are relevant. Recall shows how many of the relevant results were found.

**Latency**: time from receiving a request to producing an answer. It is important to measure the mean and percentiles (p95, p99).

**Cost**: execution cost (tokens, API calls, compute). In systems with different models, this can vary.

**Reliability**: the proportion of requests completed without errors. An agent with high accuracy but crashing in 30% of cases may be less useful than a less accurate but stable one.

**Calibration**: how well confidence scores correspond to the actual probability of correctness. A well-calibrated agent saying "70% confident" is indeed correct in approximately 70% of such cases.

### Interaction-Level Metrics

**Consistency**: coherence of answers during handoffs between agents. If agent A passes a fact to agent B, and both are later asked about it. They should answer identically.

**Information preservation**: how much information is lost during transfers between agents. A long document that has passed through a chain of summarizations may lose important details.

**Communication efficiency**: how many messages are needed to complete a task. Excessive communication increases latency and cost.

**Conflict rate**: how often agents arrive at contradictory conclusions. A high conflict rate may indicate problems with task formulation or agent incompatibility.

### System-Level Metrics

**End-to-end accuracy**: accuracy of the system's final result, not of individual agents. This is the primary metric from the user's perspective.

**End-to-end latency**: time from the user's request to the final answer, including all internal processing and communications.

**Throughput**: the number of requests the system can process per unit of time.

**Availability**: the proportion of time the system is operational. This accounts for individual agent failures and their impact on the system as a whole.

**Cost per request**: total cost of processing one request by all agents in the system.

**Variance**: how stable the results are. A system producing excellent results in 50% of cases and terrible ones in the other 50% may be less useful than a system with consistently good (but not excellent) results.

### Reliability Metrics

**Error rate**: the proportion of requests that ended in an error (of any type).

**Error propagation rate**: the proportion of errors that propagated from one agent to others (vs. were isolated).

**Recovery rate**: the proportion of errors after which the system successfully recovered and produced a correct result (through retry, fallback, etc.).

**Mean time to failure (MTTF)**: average operating time before a failure.

**Mean time to recovery (MTTR)**: average recovery time after a failure.

---

## Key Takeaways

A production swarm is a distributed system, and its measured failures are distributed-systems failures rather than model failures. A job queue absorbing 2.4 million polling requests while accepting 117 jobs, 18 of 30 agents creating the same branch name, and a unique fact lost on the way to an arbiter are all problems with owners, backpressure, and aggregation, not with prompting.

Coordination buys breadth of search, not quality of answer. Equalize scope before comparing a swarm against independent agents, because that is what made the published cost-per-finding numbers comparable.

Structure is not the lever it appears to be. Prescriptive roles and a CEO-style hierarchy produced negligible improvement over plain collaboration, while merge rates fell as teams grew from 10 to 80 agents.

Preserve the minority report separately from the consensus. The observed false-consensus failure was not a wrong agent but an aggregation step that destroyed the evidence which would have corrected the group.

Consensus in multi-agent systems is more than just voting. Different mechanisms suit different situations: simple voting for quick decisions, weighted voting to account for expertise, debates for complex and critical questions.

Formal consensus protocols (Raft, Paxos, BFT) provide mathematical guarantees but are rarely needed in full form for typical AI systems. Lighter approaches are more commonly used: redundancy, verification, audit trails.

Market mechanisms (prediction markets, auctions) create interesting incentives for honest self-assessment and can be useful in certain scenarios.

Error propagation is a critical MAS problem. Knowledge provenance, confidence propagation, verification points, and circuit breakers are tools for controlling it.

Quality metrics are needed at three levels: for individual agents, for interactions, and for the system as a whole. Without measurements, systematic improvement is impossible.

The choice of consensus mechanisms and quality assessment should match the requirements of the specific system. Over-engineering is just as dangerous here as under-engineering.

---

## Practical Implementation of Reliable Systems

### Weighted Voting with Adaptive Weights

A weighted voting system automatically adjusts each agent's influence based on its historical accuracy. The key idea: agents that were correct more often in the past receive greater weight in future decisions.

**Main components:**

**Agent profile** stores performance metrics: base weight (static), current accuracy (dynamic), number of voting participations. Initial accuracy is set neutrally (0.5) to avoid bias toward new agents.

**Weight computation** combines several factors. The final weight equals the product of base weight, current accuracy, and the confidence factor. The confidence factor grows with the number of votes, reaching maximum around 100 votes. This protects against situations where an agent happened to answer correctly a few times in a row and gained disproportionate influence.

**Voting process** groups votes by answer variant, sums weighted scores for each variant, and selects the variant with the maximum weighted score. The confidence of the final decision is computed as the ratio of the winner's weight to the sum of all weights.

**Weight updates** occur after receiving ground truth (the correct answer). Exponential moving average is used: new accuracy equals old accuracy multiplied by (1 - learning_rate) plus the correctness indicator (1.0 or 0.0) multiplied by learning_rate. Learning rate is typically 0.1-0.2, which ensures smooth but stable updates.

**Result transparency:** the system returns the final decision and voting details for each variant: list of votes, total weight, percentage breakdown. This enables auditing and understanding why a particular decision was made.

### Agent Debate System

The adversarial approach to consensus organizes formalized debates between two agents defending opposing positions, with an independent judge.

**Architecture:** three roles: "for" debater, "against" debater, judge. Each agent is independent and has no access to the internal state of others.

**Debate protocol:**

Round 0 (initial positions): each debater receives the question and context, formulates 3-5 strong arguments supporting their position, and anticipates the opponent's counterarguments.

Rebuttal rounds: debaters see the opponent's arguments and the full debate history, point out logical errors in the opponent's arguments, provide counterexamples and counter-evidence, and strengthen their position with new arguments. The number of rounds is typically 2-4 for a balance between depth and cost.

Final round (judge): an independent agent receives the complete debate transcript, evaluates both sides on criteria (logical coherence, evidence quality, rebuttal effectiveness, overall persuasiveness), and renders a decision in a structured format (winner, confidence, scores for both sides, key factors, detailed justification).

**Consensus detection:** in rare cases, debaters may reach agreement early. The system checks whether both sides acknowledge each other's validity (searching for phrases like "valid point", "agree that"). When consensus is detected, the debate ends early.

**Advantages of the approach:** identifying weak points in argumentation through active challenging, avoiding confirmation bias (each side is forced to consider the alternative). This makes all reasoning steps explicit.

**Limitations:** high cost (6+ LLM calls per question), need for an independent judge, possibility of "convincing lies" (agents can generate plausible but false arguments).

### Knowledge Provenance for Error Tracking

The provenance system tracks the origin of every fact in a multi-agent system, which is critically important for diagnosing and isolating errors.

**Provenance record structure:** each fact has a unique identifier, content (type, content, attributes), source (creator agent ID), own confidence (agent's confidence), effective confidence (accounting for dependencies), list of dependencies (prerequisite fact IDs), creation timestamp, metadata (validity flags, invalidation reasons).

**Effective confidence computation:** the key principle is that confidence in a conclusion cannot exceed confidence in the premises. Effective confidence is taken as the minimum between the fact's own confidence and the minimum effective confidence of all dependencies. This creates cascading propagation of uncertainty: if there was an unreliable fact somewhere in the chain, all dependent facts receive reduced effective confidence.

**Building the provenance chain:** recursive traversal of the dependency graph with cycle protection creates a linearized chain from initial facts to the target. This allows understanding what information an inference was based on.

**Finding affected facts:** when an error is discovered in fact X, the system finds all facts that directly or indirectly depend on X. For each fact Y, it checks whether a dependency path exists from Y to X. The result is a set of facts that may be compromised by the error.

**Fact invalidation:** when an error is discovered, the erroneous fact is invalidated and all facts dependent on it. The metadata of each affected fact records an invalidation flag, the reason (reference to the original error), and the invalidation time. This prevents the use of potentially erroneous data.

**Audit report generation:** for any fact, a complete report on its origin can be generated, including the full dependency chain, confidence metrics at each step, and source information. This is critically important for decision explainability in high-risk applications.

### Circuit Breaker for Protection Against Cascading Failures

Circuit breaker is a pattern from distributed systems, adapted to protect multi-agent systems from error propagation.

**Three states:**

CLOSED (normal operation): all requests pass through to the agent, errors are registered but do not block operation. When the error threshold is reached (e.g., 5 consecutive failures), transition to the OPEN state occurs.

OPEN (blocking): all requests are immediately rejected without calling the agent, the system uses a fallback strategy (backup agent, cached result, honest "I don't know"). After a timeout (e.g., 30 seconds), transition to HALF_OPEN.

HALF_OPEN (testing): limited test requests are allowed through. On success: transition back to CLOSED (agent has recovered). On failure: return to OPEN (agent is still non-operational).

**Operation logic:**

The execute method accepts an operation and a fallback strategy. If the circuit is in the OPEN state, the operation is not executed and the fallback is returned immediately. If the circuit allows the request, the operation is executed with exception handling. On success, the error counter is reset; in HALF_OPEN, transition to CLOSED occurs. On failure, the counter is incremented, transition to OPEN is possible, and the fallback is returned.

**Advantages for MAS:**

Failure isolation: a non-functional agent does not infect the entire system with its errors. Automatic recovery: the system periodically checks whether the agent has recovered. Graceful degradation: the system continues operating in a limited mode instead of complete failure. Monitoring: state transitions are logged. This gives a signal about problems.

**Parameter tuning:**

Error threshold (failureThreshold): typically 3-10. Lower values are more sensitive to problems but may produce false positives. Higher values are more tolerant of temporary failures but respond more slowly to systemic problems.

Reset timeout (resetTimeout): typically 10-60 seconds. Should be sufficient for the agent's potential recovery.

Half-open window (halfOpenWindow): time during which the recovery decision is made. Typically 5-15 seconds.

### Implementation Example

A weighted voting system is created with a set of agents and a learning rate. Each agent votes by specifying an answer variant and confidence level. The system weights votes based on agents' historical accuracy, selects the winning variant, and computes overall confidence. After receiving the correct answer, the system updates agent weights using exponential moving average for smooth adaptation to performance changes.
