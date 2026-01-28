# Agentic AI Course (2025-2026)

> Comprehensive course on building production AI agents.
> From Transformer fundamentals to multi-agent orchestration, RLHF, and production deployment.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Who This Is For

- Engineers transitioning to AI/ML agent development
- Backend developers building AI-powered features
- Architects designing enterprise agent systems
- Anyone preparing for AI/ML engineering interviews

## Why This Course

- **Practical** — Code examples in Java, Python, and Rust. Not toy demos.
- **Deep** — From attention mechanism internals to RLHF pipelines and GPU optimization
- **Current** — MCP protocol, Context Engineering (Karpathy 2025), DeepSeek-R1, A2A protocol
- **Production-focused** — Deployment, observability, security, cost optimization, and governance
- **Enterprise-grade** — Multi-tenant patterns, compliance frameworks, and real-world case studies

## Curriculum

| # | Module | Lessons | Key Topics |
|---|--------|---------|------------|
| 01 | [LLM Fundamentals](./01_LLM_Fundamentals/) | 10 | Transformer, Tokenization, MoE, Mamba, Scaling Laws, Interpretability |
| 02 | [Prompt Engineering](./02_Prompt_Engineering/) | 5 | CoT, ToT, ReAct prompts, Context Engineering |
| 03 | [AI Agents Core](./03_AI_Agents_Core/) | 10 | Architectures, Memory, Planning, Tool Use, Computer Use |
| 04 | [Multi-Agent Systems](./04_Multi_Agent_Systems/) | 5 | Patterns, Orchestration, Consensus, Reliability |
| 05 | [MCP Protocol](./05_MCP_Protocol/) | 5 | Server Development, Client Integration, A2A Protocol |
| 06 | [RAG](./06_RAG/) | 6 | Chunking, Embeddings, GraphRAG, CAG, Late Interaction |
| 07 | [Frameworks](./07_Frameworks/) | 6 | LangChain4j, Spring AI, Semantic Kernel, JAX, AWS Strands |
| 08 | [Structured Outputs](./08_Structured_Outputs/) | 3 | Techniques, Data Extraction, Validation |
| 09 | [Conversational AI](./09_Conversational_AI/) | 3 | Dialog Management, Voice, Multimodality |
| 10 | [Fine-Tuning](./10_Fine_Tuning/) | 10 | LoRA, QLoRA, RLHF, DPO, Synthetic Data, DoRA |
| 11 | [Evaluation & Testing](./11_Evaluation_Testing/) | 5 | Benchmarks, LLM-as-Judge, RAG Evaluation |
| 12 | [Observability](./12_Observability/) | 4 | Tracing, Metrics, LLM Debugging, AgentOps |
| 13 | [Deployment](./13_Deployment/) | 3 | Kubernetes, CI/CD, Self-Hosted Models |
| 14 | [Security & Safety](./14_Security_Safety/) | 5 | Prompt Injection, Guardrails, Compliance |
| 15 | [GPU Architecture](./15_GPU_Architecture/) | 5 | Flash Attention, Triton, Profiling, Quantization |
| 16 | [Distributed Training](./16_Distributed_Training/) | 5 | FSDP2, DeepSpeed ZeRO, 3D Parallelism |
| 17 | [Production Inference](./17_Production_Inference/) | 7 | vLLM, Speculative Decoding, Cost Optimization |
| 18 | [AI Governance](./18_AI_Governance/) | 7 | Regulation, Risk, Red Teaming, Alignment Research |
| 19 | [Practical Projects](./19_Practical_Projects/) | 5 | RAG Chatbot, Multi-Agent System, MCP Server |
| 20 | [Architecture Research](./20_Architecture_Research/) | 3 | MoE, State Space Models, Multimodal |
| 21 | [Interview Preparation](./21_Interview_Preparation/) | 5 | System Design, Coding, Papers, Staff+ Behavioral |
| | **Total** | **117** | |

## Quick Start

Start with [01. LLM Basics](./01_LLM_Fundamentals/01_LLM_Basics.md) and follow the recommended learning path below.

### Recommended Learning Path

**Level 1 — Fundamentals**
1. LLM APIs (OpenAI, Anthropic)
2. Prompt Engineering
3. Java frameworks (LangChain4j, Spring AI)
4. Basic RAG with a vector database
5. Tool Use / Function Calling
6. Simple ReAct agent implementation

**Level 2 — Core Agent Development**
1. Agent architectures (ReAct, Plan-and-Execute)
2. LangGraph for stateful agents
3. Multi-agent patterns
4. Advanced RAG techniques
5. Structured outputs
6. Agent evaluation

**Level 3 — Production & Scaling**
1. MCP protocol
2. Observability (LangSmith / Langfuse)
3. Security and guardrails
4. Deployment patterns
5. Multi-agent orchestration

## How to Use

This course works best with [Obsidian](https://obsidian.md/) or any Markdown wiki viewer. Each lesson includes Previous/Next navigation links and wiki-style cross-references.

You can also browse it directly on GitHub — start from [00_Home.md](./00_Home.md) or use the [Table of Contents](./Table_of_Contents.md).

## Author

**Oleksandr Husiev** — AI Solutions Architect with 14+ years in enterprise engineering.

Building AI agent systems for fintech, enterprise, and Fortune 500 clients. Creator of [Skreaver](https://github.com/shurankain/skreaver) — an open-source AI agent orchestration framework in Rust.

- [LinkedIn](https://linkedin.com/in/oleksandr-husiev-90497082)
- [GitHub](https://github.com/shurankain)

## License

MIT — use freely, attribution appreciated. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Feel free to open issues for corrections, suggestions, or submit pull requests with improvements.

If you find this course useful, consider giving it a star — it helps others discover the material.
