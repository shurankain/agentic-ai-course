## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[04_MCP_Client_Integration|MCP Client Integration]]
**Next:** [[../06_RAG/01_RAG_Basics|RAG Basics]]

---

# Agent-to-Agent Protocol (A2A)

## Introduction: Inter-Agent Communication

MCP enables agents to connect to tools and data. Agent-to-Agent Protocol (A2A), introduced by Google in 2025, addresses the problem of inter-agent interaction.

Example: a travel planning agent coordinates with hotel booking, flight booking, and restaurant recommendation agents. A2A defines a standard protocol for such interactions.

## Why A2A When MCP Exists?

### Different Levels of Abstraction

MCP and A2A operate at different levels. MCP defines how an agent accesses resources: reads files, executes database queries, calls APIs. This is vertical integration — the agent reaches "down" to data.

A2A defines how agents interact with each other. This is horizontal integration — an agent reaches "sideways" to other agents. The hotel booking agent does not need to know the implementation details of the flight booking agent. It simply sends a request "find a flight from London to Paris on January 15" and receives a structured response.

### Complementary, Not Competing

MCP and A2A do not compete — they complement each other. In a multi-agent system, agents use the A2A protocol for horizontal communication with each other, delegating tasks and coordinating actions. Simultaneously, each agent uses MCP for vertical integration with tools and data. For example, a travel planning agent coordinates with hotel and flight agents via A2A, while each of these agents accesses their specialized APIs and databases through MCP. This creates a clean separation: A2A for inter-agent coordination, MCP for resource access.

### When to Use Which

**MCP — for resource access:**
- Reading/writing files
- Database queries
- REST/GraphQL API calls
- Managing local tools

**A2A — for agent coordination:**
- Delegating tasks to specialized agents
- Parallel execution of independent subtasks
- Complex workflows involving multiple agents
- Integrating agents from different providers

## A2A Architecture

### Core Concepts

**Agent** — an autonomous unit with defined capabilities. An agent can be implemented on any platform (LangChain, AutoGen, CrewAI, custom) and use any LLM. A2A abstracts away these differences.

**Agent Card** — agent metadata in JSON-LD format. Contains:
- Capabilities (what the agent can do)
- Endpoints (how to reach it)
- Authentication requirements
- Supported input/output schemas

**Task** — a unit of work delegated to an agent. Tasks can be:
- Synchronous (immediate response)
- Asynchronous (long-running, result delivered later)
- Streaming (partial results as execution progresses)

**Message** — a unit of communication within a task. Can contain text, files, structured data.

### Agent Identity

Before agents can interact, they must be able to identify each other. A2A defines a formal identity model:

**Identification levels:**

| Level | Example | Purpose |
|-------|---------|---------|
| **URN** | `urn:agent:hotel-booking:v2` | Globally unique identifier |
| **DID** | `did:web:hotel-agent.example` | Decentralized identity |
| **Instance ID** | `agent-001-eu-west` | Specific instance |
| **Session ID** | `sess_abc123` | Current interaction session |

**Identity components:** Each agent has a multi-layered identity that includes a URN (unique identifier such as `urn:agent:travel-planner:v1`), Provider (verified publisher, e.g. Acme Corp), Instance ID (specific instance like `travel-planner-prod-001`), Location (deployment region), Trust level (enterprise verified, community, etc.), and an X.509 certificate from a trusted CA.

**Identity verification:** When establishing a connection, agents exchange certificates and verify the certificate's authenticity through the chain up to the CA, the identity match in the Agent Card, validity through a revocation check, and rights to the claimed capabilities. This is similar to PKI in TLS but adapted for the agent ecosystem.

### Agent Cards: Discovery and Capabilities

Agent Card solves the discovery problem — how does one agent find another and understand what it can do? It is a JSON-LD document that describes the agent's capabilities, its endpoints, authentication requirements, and constraints.

**Agent Card structure includes:**

**Metadata:** The agent's name, description, and version allow identifying it within the ecosystem. Versioning is critical for ensuring compatibility when capabilities are updated.

**Capabilities:** Each capability describes a specific agent function. For each one, a name, description, JSON Schema for input parameters and output data are specified. For example, the "searchHotels" capability can accept location, checkIn, checkOut, and guests, and return an array of HotelResult. A capability can also specify async and streaming flags so the client knows how to properly invoke the function.

**Endpoints:** Define how to reach the agent. The primary API endpoint for task execution, a well-known URL for discovery (typically `/.well-known/agent.json`), and optionally a WebSocket endpoint for streaming communication.

**Authentication:** Describes authentication requirements. A2A supports OAuth2 (client credentials or authorization code flow), API keys, and mutual TLS. For each method, the necessary parameters are specified — token URL, scopes, headers.

**Rate Limits:** Specify constraints — requests per minute, maximum number of concurrent tasks, maximum task execution duration. This allows clients to properly manage load.

Agent Cards are published at a well-known URL (`/.well-known/agent.json`), enabling automatic discovery. An orchestrator agent can scan known domains and build a catalog of available agents. This is similar to an OpenAPI specification but specifically adapted for agent systems with support for async operations and streaming.

### Long-Running Tasks

One of A2A's key features is native support for long-running tasks. Unlike synchronous API calls, agent tasks can take minutes, hours, or even days.

**Task lifecycle:** A task goes through several states. SUBMITTED — the task is accepted but not started. WORKING — the agent is actively working. INPUT_REQUIRED — the agent awaits additional input from the caller (e.g., payment confirmation). BLOCKED — the task is blocked by external factors (external service unavailable). Final states: COMPLETED (success), FAILED (error), CANCELLED (cancelled by the caller).

**Asynchronous workflow:** The client sends POST /tasks with a capability and receives 202 Accepted with a taskId and SUBMITTED status. It periodically polls GET /tasks/{taskId} to check the status. If the status is INPUT_REQUIRED, the client sends POST /tasks/{taskId}/messages with the required data, and the task transitions back to WORKING. When the task is complete, the agent can send a webhook callback or the client receives COMPLETED status on the next poll.

### Streaming and Progressive Results

For tasks with intermediate results, A2A supports streaming via Server-Sent Events (SSE). The client establishes an SSE connection when creating a task by specifying the stream=true flag. The agent sends events of several types: progress events with completion percentage and current status, artifact events with partial results (e.g., the first documents found in a research task), and a final complete event with the overall result.

This is critical for user experience in agent systems — the user can see what is happening, receive intermediate results, and can cancel the task if the direction is wrong. For example, a research agent can stream discovered articles as the search progresses rather than waiting for the entire analysis to complete.

## Multi-Agent Orchestration Patterns

### Hub and Spoke

A central orchestrator agent coordinates specialized agents (Flight Agent, Hotel Agent, Car Agent). The orchestrator receives a task from the user, breaks it into subtasks, delegates them to the corresponding agents, collects results, and forms the final response.

**Advantages:** Simple coordination, single point of control, easy to add new agents.

**Disadvantages:** Single point of failure, bottleneck when scaling, the orchestrator must understand all domains.

### Mesh / Peer-to-Peer

Agents communicate directly with each other without a central coordinator. Flight Agent can directly query Hotel Agent, which in turn coordinates with Car Agent. Each agent decides on its own whom to contact to fulfill its task.

**Advantages:** No single point of failure, parallel communication, better scalability.

**Disadvantages:** Harder to track state, potential cycles and deadlocks, requires a discovery mechanism.

### Hierarchical

A multi-level structure with delegation. Travel Planner Agent coordinates domain managers (Transport Manager, Lodging Manager, Activity Manager), each of which manages specialized agents (Flight + Train, Hotel + Airbnb, Tour Guide). Tasks are delegated down the hierarchy, results are aggregated upward.

**Advantages:** Scalability, separation of responsibility, specialization at each level.

**Disadvantages:** Latency increases with depth, harder to debug, requires clear domain separation.

## Adoption and Ecosystem

### A2A Compared to Alternatives

A2A is not the only approach to inter-agent communication. Understanding the alternatives helps evaluate its positioning:

| Aspect | A2A (Google) | Agent Protocol (AI21) | FIPA-ACL (classic) | Custom gRPC |
|--------|--------------|----------------------|---------------------|-------------|
| **Transport** | HTTP/WebSocket/gRPC | HTTP only | IIOP/HTTP | gRPC |
| **Discovery** | Well-known URLs | Registry API | DF (Directory Facilitator) | Service mesh |
| **Task model** | Async + streaming | Sync only | Performatives | Depends |
| **Schema** | JSON Schema | OpenAPI | FIPA-SL | Protobuf |
| **Adoption** | 150+ announced partners | Emerging | Academic | Custom |
| **LLM focus** | Native | Yes | No | No |

**Why Google chose a new protocol:**
1. **Existing standards (FIPA) are too academic** — too complex for practical use
2. **Agent Protocol from AI21 is too simple** — lacks async, streaming, and rich capabilities support
3. **Direct use of gRPC/REST requires ad-hoc coordination** — no standard patterns

**Actual timeline:** A2A was announced by Google in April 2025 with initial partner support. It was donated to the Linux Foundation in June 2025. In December 2025, MCP was donated to the Agentic AI Foundation (AAIF) co-founded by Anthropic, OpenAI, and Block — establishing AAIF as the primary governance body for agent protocol standardization. While A2A attracted partner announcements, MCP has seen broader real-world adoption with 10,000+ servers and 97M+ monthly SDK downloads.

**Open ecosystem questions:** Centralized vs. federated agent registry, approaches to Agent Card versioning, standardization path (W3C, IETF, or industry consortium).

### A2A Partners

At its April 2025 announcement, A2A listed 150+ partners (primarily announcements of intent rather than production implementations):

**Cloud Providers:**
- AWS
- Microsoft Azure
- Google Cloud (protocol author)

**Agent Frameworks:**
- LangChain
- AutoGen (Microsoft)
- CrewAI
- Semantic Kernel

**Enterprise:**
- Salesforce (Agentforce)
- SAP
- ServiceNow
- Atlassian

### Integration with MCP

Google and Anthropic coordinate protocol development for interoperability. In a typical agent ecosystem, Agent A (on Claude) and Agent B (on GPT-4) communicate with each other via the A2A Protocol. Each uses MCP to access their tools — Agent A to GitHub Server, Agent B to Jira Server.

A typical scenario: Agent A receives a task via A2A from Agent B, uses MCP to access GitHub (e.g., reads code or creates a PR), and returns the result via A2A back to Agent B. This demonstrates a clean separation: A2A for coordination between agents, MCP for accessing data and tools.

## Security and Trust

### Authentication Chain

A2A defines a trust chain through a central Identity Provider that issues tokens to agents (Agent A Token, Agent B Token, Agent C Token). During interaction, agents perform mutual verification — they exchange tokens and verify each other's rights. Each agent has an identity verifiable through OAuth2/OIDC.

### Capability-Based Security

Instead of role-based access, A2A uses a capability-based security model. When delegating a task to an agent, the following are explicitly specified: requiredCapabilities (what is needed for execution), delegatedCapabilities (what the agent can do on its own), and restrictedCapabilities (what the caller must handle).

For example, a "bookFlight" task may require flight:search, flight:book, and payment:process. But the agent receives delegation only for flight:search and flight:book, while payment:process remains with the caller — the agent must request payment confirmation through the INPUT_REQUIRED state. The principle of least privilege is applied automatically at the protocol level.

### Audit Trail

All interactions between agents are logged with detailed information for auditing. Each event includes a timestamp, taskId, source and target agents with their identities, the action performed (TASK_SUBMITTED, TASK_COMPLETED, etc.), the capability used, and metadata with a correlationId for tracing the entire call chain.

This is critical for enterprise deployments with compliance requirements — it is possible to trace which agent did what, at whose request, and with what data. The audit trail also helps in debugging complex multi-agent workflows where understanding the sequence of events and the causes of failures is important.

## MCP vs. A2A Comparison

| Aspect | MCP | A2A |
|--------|-----|-----|
| **Purpose** | Agent → Tools/Data | Agent → Agent |
| **Pattern** | Client-Server | Peer-to-Peer / Orchestration |
| **Tasks** | Synchronous (primarily) | Sync + Async + Streaming |
| **Discovery** | Client configuration | Agent Cards + Well-Known URLs |
| **State** | Stateless | Stateful (task lifecycle) |
| **Transport** | stdio, HTTP+SSE | HTTP, WebSocket, gRPC |
| **Author** | Anthropic | Google |
| **Focus** | Tools | Coordination |

**When both are needed:**
- Multi-agent systems with access to external data
- Enterprise workflows with specialized agents
- Integrations between different AI platforms

## Key Takeaways

A2A Protocol solves the problem of coordination between autonomous agents, complementing MCP. If MCP is "how an agent gets data," then A2A is "how agents work together."

Key features of A2A include Agent Cards for automatic discovery and capability description, full support for long-running tasks with a detailed lifecycle, streaming for receiving progressive results as execution proceeds, and multi-provider support enabling agents on different LLMs to interact with each other. The protocol is designed as an enterprise-ready solution with built-in support for security, audit trail, and compliance requirements.

Orchestration architectural patterns — Hub-and-Spoke, Mesh, and Hierarchical — address different challenges. Hub-and-Spoke provides centralized control, Mesh delivers fault tolerance through peer-to-peer communication, and Hierarchical enables scaling complex systems through delegation.

Security in A2A is implemented through a capability-based model with mutual agent authentication, an audit trail for all interactions, and the principle of least privilege at the protocol level.

Understanding the complementary nature of MCP and A2A is critically important: MCP provides vertical integration (agent → data and tools), while A2A implements horizontal integration (agent → agent). Together they form a complete ecosystem for building complex multi-agent systems.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Model Context Protocol
**Previous:** [[04_MCP_Client_Integration|MCP Client Integration]]
**Next:** [[../06_RAG/01_RAG_Basics|RAG Basics]]
