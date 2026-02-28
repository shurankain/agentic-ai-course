# AI Frameworks Overview

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Frameworks
**Previous:** [[../06_RAG/06_Late_Interaction_Retrieval|Late Interaction Retrieval]]
**Next:** [[01_LangChain4j|LangChain4j for Java]]

---

## Introduction

The ecosystem of frameworks for developing AI agents and LLM applications is evolving rapidly. New tools appear monthly, existing ones undergo radical changes, and some merge or become discontinued. This chapter provides an overview of the current state of the industry and helps select the right tool for a specific task.

It is important to understand that choosing a framework is not about picking the "best" one but about selecting the one that fits your requirements. A simple RAG chatbot and a complex multi-agent system require different tools.

---

## Framework Categories

### By Abstraction Level

Frameworks are organized by abstraction level from high to low:

**HIGH-LEVEL (No-code / Low-code)** — visual tools for building AI workflows without programming. This level includes n8n, Make, and Zapier AI. These platforms provide graphical editors for creating scenarios with minimal code, making them accessible to non-developers.

**MID-LEVEL (Agent Frameworks)** — frameworks for orchestrating multi-agent systems with ready-made patterns. CrewAI, LangGraph, and AutoGen provide high-level abstractions for building complex agent systems while hiding implementation details.

**LOW-LEVEL (Building Blocks)** — modular components for building LLM applications. LangChain, LlamaIndex, and Haystack offer building blocks such as memory, tools, and chains that developers compose into the desired architecture.

**PROVIDER SDKs (Direct API)** — direct provider SDKs such as OpenAI, Anthropic, and Google. They provide maximum control with minimal abstractions, requiring more code but offering full flexibility.

---

## Key Market Players

### Leaders in Agent-Oriented Development

**LangChain and LangGraph** remain the de facto standard with over 80,000 GitHub stars and 4.2 million downloads per month. A vast integration ecosystem and LangSmith for monitoring make them the first choice for Python development. Complexity and a steep learning curve are the main barriers for newcomers.

**CrewAI** is growing rapidly, reaching 30,000 stars and attracting $18 million in investment (Series A). The role-based approach to agents and easy onboarding attract developers, but less flexibility compared to LangGraph limits its use in complex scenarios.

**Microsoft AutoGen** with 25,000 stars focused on multi-agent conversations and code execution. Now in maintenance mode — Microsoft has consolidated its agent strategy into **Microsoft Agent Framework**, which unifies AutoGen and Semantic Kernel concepts. New projects should use Microsoft Agent Framework instead.

**LlamaIndex** with 35,000 stars is the leader specifically for RAG scenarios. A vast collection of data connectors makes it indispensable for retrieval tasks, but its focus on information retrieval means fewer capabilities for agent workflows.

### Cloud Provider Solutions

**AWS Bedrock Agents** are used by over 100,000 organizations, providing enterprise SLA and full integration with the AWS ecosystem. Vendor lock-in remains the main drawback for companies planning a multi-cloud strategy.

**Google Vertex AI Agent Builder** shows growth with over 7 million Agent Development Kit downloads. Integration with Gemini models and the ability to ground through Google Search make it attractive for the GCP ecosystem.

**Azure AI Agent Service** is in preview status, offering integration with the enterprise Microsoft stack. Limited availability currently hinders broad adoption.

### Low-Code Platforms

**n8n** leads in low-code AI automation with over 200,000 business users. Native MCP (Model Context Protocol) support and the ability for self-hosted deployment are critical for enterprise scenarios with data security requirements.

**Make** offers a stable platform for visual scenarios with limited MCP support.

**Zapier AI** is developing AI capabilities leveraging integrations with over 6,000 applications but does not yet support MCP.

---

## LangChain and LangGraph: Detailed Overview

### LangChain: Building Blocks

LangChain remains the most widely used framework for LLM applications. It provides modular components for all typical tasks.

**LLM Integration** provides a unified interface for working with OpenAI, Anthropic, local models via Ollama, and dozens of other providers. Developers write code once and can then switch models by changing configuration.

**Chains** implement sequences of operations where the output of one step is passed to the next. A typical chain: a prompt is formed from a template, passed to an LLM, and the response is parsed into the required format. Chains compose into more complex workflows.

**Memory** supports both short-term memory for the current conversation and long-term memory through vector stores. This allows agents to maintain context in multi-turn conversations and remember information between sessions.

**Tools** provide integration with external APIs and services. The model can invoke tools to perform actions: fetching data, sending notifications, managing systems.

**RAG Components** include retrievers for finding relevant information, vector stores for storing embeddings, and document loaders for importing data from various sources.

**When to use LangChain**: rapid prototyping of LLM applications, building RAG systems, simple call chains, scenarios where a rich ecosystem of ready-made integrations is critical.

### LangGraph: Stateful Agents

LangGraph extends LangChain for creating complex agent systems with explicit state management. An agent is represented as a graph where nodes are actions and edges are conditional transitions.

**State** represents explicit state passed between graph nodes. This can include message history, intermediate results, and custom metadata.

**Nodes** are functions that read the current state, perform an action (model call, tool execution), and return updated state.

**Edges** define transitions between nodes. Conditional edges enable branching: depending on a node's output, the graph can follow different paths.

**Persistence** through checkpointing allows saving the state of long-running agents. If an agent performs a task requiring hours or days, its state is periodically saved. On failure, the agent resumes from the last checkpoint.

**When to use LangGraph**: complex multi-step agents with numerous conditional transitions, cases requiring explicit control over execution flow, human-in-the-loop workflows requiring a pause for user confirmation, scenarios demanding granular control over each step.

---

## CrewAI: Role-Based Multi-Agent

CrewAI offers a declarative approach: agents as a "team" with roles. Instead of an explicit graph, the developer describes agents and their interactions declaratively.

**Agent** represents an entity with three key attributes: role (role in the team), goal (what the agent should achieve), and backstory (context that makes behavior more natural).

**Task** is a unit of work assigned to a specific agent. A task describes what needs to be done, what tools are available, and what output is expected.

**Crew** combines a group of agents working together toward a common goal. A Crew defines how agents coordinate with each other.

**Process** defines the execution mode: sequential (agents work one after another, passing results to the next) or hierarchical (a manager agent distributes tasks among worker agents).

**When to use CrewAI**: multi-agent systems with clearly defined roles (e.g., a research team with researcher, writer, reviewer), when simplicity matters more than flexibility, for a quick start with multi-agent without deep framework study.

---

## Provider SDKs: Working Directly with Models

### OpenAI Agents SDK

OpenAI has transitioned from the Assistants API to a new architecture. The Assistants API was deprecated in mid-2025, replaced by the **Responses API** (stateless, single-turn successor to Chat Completions with built-in tool use) and the **Agents SDK** (production multi-agent framework).

**Responses API**: the new foundational API replacing both Chat Completions and Assistants API. Supports web search, file search, code interpreter, and computer use as built-in tools. Stateless by design — state management is handled by the application or the Agents SDK.

**Agents SDK** (released March 2025, now mature): a production-grade Python framework for multi-agent systems. Key capabilities: **Handoffs** for routing between specialized agents, **Guardrails** for input/output validation running in parallel with the agent, **Tracing** for built-in observability of every agent run. The SDK evolved from the experimental Swarm framework, preserving its simplicity while adding production features.

### Claude SDK and Claude Code

Anthropic provides the Claude API with tool use capabilities and the Claude Code CLI as its agent platform. Rather than a separate agent framework, Anthropic's approach emphasizes the model's native tool use ability combined with MCP for extensibility.

**Key capabilities**: powerful tool use via the Claude API allows models to call functions with high accuracy. Computer Use enables browser and desktop automation where the model controls the computer. Extended thinking provides transparent chain-of-thought reasoning. Extended context of 200,000 tokens (1M in beta) allows processing vast amounts of information. **Claude Code** is a CLI-based coding agent that demonstrates the agentic pattern using Claude's native capabilities — it reads files, writes code, runs commands, and uses MCP servers for extensibility.

### Google Vertex AI and ADK

Google Agent Development Kit is a unified toolkit for building agents in the Google Cloud ecosystem.

**Key capabilities**: deep integration with Gemini models provides access to multimodal capabilities. Grounding through Google Search allows models to verify facts via search. Support for the A2A protocol (Agent-to-Agent) simplifies building multi-agent systems.

---

## AutoGen and Microsoft Agent Framework

Microsoft AutoGen has entered maintenance mode as Microsoft consolidates its agent strategy.

**Current state**: AutoGen 0.4 was the last major release. The project continues to receive security updates but no new features. Microsoft's strategic investment has shifted to **Microsoft Agent Framework**, which unifies concepts from AutoGen (event-driven multi-agent patterns) and Semantic Kernel (plugin/function model, enterprise integrations).

**For new projects**: use Microsoft Agent Framework for enterprise Microsoft ecosystem projects. For non-Microsoft scenarios, evaluate LangGraph, OpenAI Agents SDK, or CrewAI. Existing AutoGen deployments can continue operating but should plan migration to Microsoft Agent Framework.

---

## n8n: Low-Code AI Leader

n8n deserves special attention as a bridge between no-code and code approaches. Native MCP support makes it a powerful tool for AI workflows.

**Why n8n for AI**: the self-hosted option is critical for enterprises with data sovereignty requirements. Out-of-the-box MCP support provides access to the growing MCP server ecosystem. Over 100 ready-made integrations cover typical scenarios. The visual workflow builder reduces the learning curve. Code nodes allow adding custom logic in JavaScript/Python where visual components are insufficient.

---

## Framework Selection Criteria

### Practical Recommendations

**Quick RAG prototype**: LlamaIndex if the focus is exclusively on retrieval and document indexing. LangChain if greater flexibility and integrations with other components are needed.

**Multi-agent with roles**: CrewAI is ideal when agent roles and their interactions can be clearly defined declaratively.

**Complex stateful agent**: LangGraph provides full control over flow through an explicit state graph and transitions.

**Enterprise cloud solution**: AWS Bedrock for the AWS ecosystem, Vertex AI for GCP, Azure AI Agent Service for the Microsoft stack. The choice is determined by existing infrastructure.

**No-code with AI**: n8n for self-hosted with MCP, Make for managed cloud solutions.

**Java/Spring stack**: LangChain4j for maximum compatibility with the LangChain ecosystem, Spring AI for deep Spring integration.

**Maximum control**: direct provider SDKs (OpenAI, Anthropic, Google) when framework abstractions are limiting or when maximum performance is needed.

### Red Flags When Choosing

**"Let's use the most popular one"** — popularity by GitHub stars does not guarantee suitability for your task. Analyze requirements, not metrics.

**Unnecessary complexity** — using CrewAI or LangGraph for a simple single-agent chatbot adds unwarranted complexity.

**Insufficient abstraction** — attempting to build a complex multi-step agent directly through an API will result in enormous amounts of boilerplate code and errors.

**Ignoring operations** — a framework without observability, tracing, and monitoring will create problems in production. Verify the availability of metrics, logging, and tracing.

---

## Key Takeaways

1. **LangChain/LangGraph** — the de facto standard for Python LLM development with a massive ecosystem and community support.

2. **CrewAI** — the best choice for role-based multi-agent systems when declarativeness matters more than granular control.

3. **AutoGen** — now in maintenance mode; new projects should use **Microsoft Agent Framework** (unified successor combining AutoGen + Semantic Kernel).

4. **Cloud providers** (AWS, GCP, Azure) — production-ready solutions with enterprise SLA, but vendor lock-in limits flexibility.

5. **n8n** — the leader in low-code AI automation with a unique combination of a visual interface and MCP support.

6. **Anthropic's approach to agents** — use the Claude API with tool calling and MCP for extensibility; Claude Code demonstrates the pattern of a production coding agent built on native capabilities.

7. **Choose by task, not by hype** — a simple RAG does not require complex agent frameworks, and a complex multi-agent system cannot be built on basic chains.

---

## Related Materials

- [[01_LangChain4j|LangChain4j for Java]]
- [[02_Spring_AI|Spring AI]]
- [[03_Semantic_Kernel|Semantic Kernel]]
- [[../05_MCP_Protocol/01_MCP_Basics|MCP Protocol]]
- [[../03_AI_Agents_Core/09_Agent_Use_Cases|AI Agent Use Cases]]

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Frameworks
**Previous:** [[../06_RAG/06_Late_Interaction_Retrieval|Late Interaction Retrieval]]
**Next:** [[01_LangChain4j|LangChain4j for Java]]
