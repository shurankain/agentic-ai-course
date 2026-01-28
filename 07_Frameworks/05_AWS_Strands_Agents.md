# AWS Strands Agents

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Frameworks
**Previous:** [[04_JAX_Ecosystem|JAX Ecosystem]]
**Next:** [[../08_Structured_Outputs/01_Structured_Output_Techniques|Structured Output Techniques]]

---

## Introduction

AWS Strands Agents SDK is an open-source framework from AWS for building AI agents. Strands implements a model-driven approach, contrasting with the graph-driven architecture of LangGraph.

Key characteristics: the model-driven approach means the LLM makes decisions about which actions to take. Open-source under the Apache 2.0 license. Code size for a basic agent is only 5-10 lines. Native MCP support. For enterprise use, integration with Bedrock AgentCore is available.

## Model-Driven vs Graph-Driven

Two fundamentally different approaches to building agents:

### Graph-Driven (LangGraph, Traditional)

The developer explicitly defines a graph of states and transitions. Nodes represent agent actions. Edges represent transition conditions. State is an explicit workflow state.

**Advantages**: predictable behavior, easier to debug, explicit control over flow.

**Disadvantages**: significant boilerplate code, rigid structure, difficult to adapt to unexpected situations.

### Model-Driven (Strands)

The LLM independently decides which tools to use and in what order. Agent Loop is an iterative decision-making cycle. Tools are available instruments. Model makes all decisions about next steps.

**Advantages**: minimal code, runtime flexibility, adaptability to new situations.

**Disadvantages**: less predictable, harder to control, depends on model quality.

**Comparison of approaches**: graph-driven gives the developer control with high predictability but low flexibility, requires significant code, is easier to debug, and has low adaptability. Model-driven transfers control to the LLM with medium predictability but high flexibility, requires little code, is harder to debug, and has high adaptability.

## Strands Architecture

### Three Core Components

**Agent** is the central object combining Model (LLM for decision-making), System Prompt (instructions and context), and Tools (available instruments).

**Tool** is a function that the agent can invoke. It has strictly typed inputs/outputs, a description for the LLM, and automatic serialization of results.

**Agentic Loop** is the execution cycle consisting of the following steps: receive a task from the user, the model decides whether to use a tool or respond directly, if a tool is needed — execute it and return to step two, if a response is ready — return it to the user. This cycle repeats until a final answer is reached.

## Multi-Agent Patterns in Strands

Strands supports several patterns for multi-agent systems:

### Agents-as-Tools

One agent can use another agent as a tool. Specialized agents are exposed as tools for an orchestrator agent.

**When to use**: clear agent hierarchy, task specialization, centralized control needed.

### Agent Handoffs

An agent can transfer control to another agent. The agent determines that a task is better suited for another agent and "passes the baton."

**When to use**: different domains of expertise, context switching in conversation, escalation of complex cases.

### Swarm Pattern

Multiple peer agents work together. Agents can freely interact with each other without a central orchestrator.

**When to use**: complex collaborative tasks, no clear hierarchy, brainstorming and debate scenarios.

### Graph Orchestration

For cases requiring explicit control, Strands supports integration with graph-based orchestration when needed.

**When to use**: regulatory requirements, critical workflows, audit and compliance.

**Pattern comparison**: Agents-as-Tools provides high control with medium flexibility and low complexity for hierarchical systems. Agent Handoffs offers medium control with high flexibility and medium complexity for customer support. Swarm provides low control with very high flexibility and high complexity for research and brainstorming. Graph Orchestration delivers very high control with low flexibility and medium complexity for compliance workflows.

## MCP Support

Strands has native MCP (Model Context Protocol) support. Agents can use MCP servers as tool sources. Automatic discovery of available tools simplifies integration. Compatibility with the existing MCP ecosystem provides access to a growing library of MCP servers.

**Advantages**: access to the growing MCP server ecosystem, a unified protocol for tools, and interoperability — working with tools from different sources.

## Bedrock AgentCore

For enterprise deployment, AWS offers Bedrock AgentCore — a managed service built on top of Strands.

**Capabilities**: managed infrastructure with auto-scaling and monitoring. Security through IAM integration and encryption. Observability with built-in tracing and metrics. Guardrails for content filtering and safety checks. Multi-Model support for different LLM providers.

### When to Use Strands SDK vs Bedrock AgentCore

**Strands SDK**: full control over infrastructure, development and testing, on-premise deployment, custom requirements.

**Bedrock AgentCore**: rapid production deployment, enterprise compliance, managed operations, AWS-native stack.

## Comparison with LangGraph

**Approach**: Strands uses model-driven where the LLM decides; LangGraph uses graph-driven where the developer defines.

**Minimal agent**: Strands — 5-10 lines, LangGraph — 50-100 lines.

**Flow control**: Strands — the LLM decides, LangGraph — the developer defines.

**Debugging**: Strands is harder, LangGraph is easier.

**Flexibility**: Strands is high, LangGraph is medium.

**Enterprise**: Strands — Bedrock AgentCore, LangGraph — LangSmith.

**MCP Support**: Strands is native, LangGraph uses adapters.

**Vendor**: Strands — AWS, LangGraph — LangChain.

### When to Choose Strands

Prototyping and MVP. Flexible, adaptive agents. AWS-native infrastructure. Minimal boilerplate code.

### When to Choose LangGraph

Strict flow requirements. Compliance and audit. Complex state machines. Full control needed.

## Practical Architecture

### Working with Strands Basics

**Creating the simplest agent**: a basic agent in Strands is created in just a few lines. You import the Agent class and the @tool decorator, define a tool function using the decorator, then create an agent instance passing a list of available tools. After that, you can call the agent like a regular function, passing a text query. The agent automatically decides whether to use a tool to answer.

**Agent configuration with system prompt**: for more complex scenarios, an agent can be assigned a specific role via the system_prompt parameter, a model can be specified (e.g., anthropic.claude-3-sonnet), and multiple specialized tools can be provided. For example, for a research assistant, you can create tools for database search and document summarization, and specify in the system prompt that the agent should always cite sources.

### Implementing Multi-Agent Patterns

**Agents-as-Tools pattern**: specialized agents are created for specific tasks (e.g., data analyst and technical writer), each with its own set of tools and system prompt. These agents are then wrapped in tool functions using the @tool decorator. The main orchestrator agent gains access to these wrapped agents as its own tools, allowing it to delegate specialized tasks to the appropriate experts. When receiving a complex request, the orchestrator automatically decides which specialized agent to engage.

**Agent Handoffs pattern**: the handoffs mechanism allows agents to transfer control to each other. This uses the @handoff decorator specifying the target agent. For example, in a customer support system, three agents are created: general support, technical support, and billing. For the general support agent, handoff functions are defined for escalation to technical support or billing. When a user describes a payment issue, the general support agent automatically recognizes the context and transfers the conversation to the billing agent, preserving the entire conversation history.

**Swarm pattern**: for collaborative tasks, the Swarm class is used to combine multiple peer agents. Agents are created with different roles (researcher, critic, synthesizer), each with its own system prompt. Swarm can operate in round_robin mode (agents speak in turn) or free_form mode (free interaction). When a task is launched, agents iteratively exchange opinions, producing deeper analysis than a single agent could provide.

### Ecosystem Integration

**Connecting MCP servers**: Model Context Protocol is natively supported through the MCPToolProvider class. You specify the command to launch an MCP server (e.g., a filesystem server via npx) with the necessary arguments. This provider is then passed to the agent via the tool_providers parameter. The agent automatically discovers all tools provided by the MCP server and can use them without additional configuration.

**Deployment via Bedrock AgentCore**: for production, the BedrockAgent class is used. During creation, a unique agent_id, model, guardrails settings (content filtering, personal data detection), and tracing parameters for monitoring are specified. The agent is invoked through the invoke() method with a session_id for tracking user sessions. All interactions are automatically logged, and traces are available for analysis and debugging.

## Key Takeaways

Model-Driven approach means the LLM makes workflow decisions with minimal code. Three components — Agent, Tools, and Agentic Loop — form the architectural foundation.

Multi-Agent patterns include Agents-as-Tools for hierarchy, Handoffs for control transfer, Swarm for peer collaboration, and Graph when control is needed.

Native MCP provides integration with the MCP server ecosystem. For Enterprise, Bedrock AgentCore offers a managed service for production.

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Frameworks
**Previous:** [[04_JAX_Ecosystem|JAX Ecosystem]]
**Next:** [[../08_Structured_Outputs/01_Structured_Output_Techniques|Structured Output Techniques]]
