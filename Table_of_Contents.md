# AI Agents Course — Full Contents

## Navigation

**Module:** Course Contents
**Home:** [[00_Home|Course Home Page]]

---

> Detailed table of contents for all topics covered in the course.
> The structure maps exactly to the course files.

---

## 01. LLM Fundamentals

### 01.1 LLM Basics (`01_LLM_Basics.md`)
- What LLMs are and why they are "large"
- Transformer architecture: a revolution in language processing
  - The world before Transformers (RNN, LSTM, GRU)
  - Attention is all you need
  - How the attention mechanism works (Query, Key, Value)
  - Multi-Head Attention: looking from multiple perspectives
  - Causal attention: no peeking into the future
- Feed-Forward Networks: where knowledge is stored
- Transformer layers: depth of understanding
- Positional encoding: understanding word order (RoPE, ALiBi)
- Scale of modern models
- Emergent abilities: when quantity transforms into quality
- In-Context Learning: learning without training
- Connection to AI agents

### 01.2 Tokenization (`02_Tokenization.md`)
- Why you can't just use characters or words
  - Character-level approach
  - Word-level approach
  - The sweet spot: subword tokenization
- BPE: the algorithm that conquered the world
  - How BPE trains
  - Determinism and consistency
- Other tokenization algorithms
  - WordPiece (Google)
  - SentencePiece
- Special tokens: beyond just text
- Practical aspects: counting and saving tokens
- Code tokenization specifics
- Tokenization problems and their consequences
- Tokenization in the context of AI agents

### 01.3 Context Windows (`03_Context_Windows.md`)
- What goes into the context window
- Evolution of context sizes
- Why context is limited
  - KV-cache and its optimization
- The "lost in the middle" problem
- Context extension techniques
  - Sliding Window Attention
  - Sparse Attention
  - RoPE, ALiBi
- Context management in AI agents
- Working with long documents
- Optimizing context usage

### 01.4 Generation Parameters (`04_Generation_Parameters.md`)
- How LLMs generate text
- Temperature: degree of randomness
- Top-p (Nucleus Sampling): adaptive selection
- Top-k: hard limit
- Combining parameters
- Other important parameters
  - Max tokens
  - Stop sequences
  - Frequency/Presence penalty
- Generation parameters in AI agents
- Impact on cost and performance

### 01.5 LLM Providers and APIs (`05_LLM_Providers_and_API.md`)
- Major providers and their models
  - OpenAI (GPT-5, GPT-4o, o3, GPT-4.1)
  - Anthropic (Claude Opus 4.6, Sonnet 4.6, Haiku 3.5)
  - Google (Gemini)
  - Meta (Llama)
  - Mistral
- API structure: common principles
- Pricing comparison
- Choosing a model for different tasks
- Tool Use / Function Calling
- Reliability strategies
- Integration via LangChain4j
- Local models
  - vLLM, Ollama, LocalAI
  - PagedAttention, Speculative Decoding
  - Quantization (INT8, INT4, GPTQ, AWQ)

### 01.6 Integration Patterns (`06_Integration_Patterns.md`)
- Synchronous vs Asynchronous requests
  - The synchronous approach problem
  - Asynchronous approach (CompletableFuture)
  - Reactive approach (Project Reactor)
- Streaming responses
  - Why streaming matters
  - Server-Sent Events (SSE)
  - WebSocket
- Rate Limiting and Retry
  - Understanding rate limits (RPM, TPM)
  - Exponential backoff
- Caching
  - Exact match caching
  - Semantic caching
- Cost optimization
- Error handling
- Monitoring and metrics

### 01.7 Modern Architectures (`07_Modern_Architectures.md`)
- Mixture of Experts (MoE)
  - The philosophy of sparse computation
  - MoE architecture: router, experts
  - Mixtral 8x7B: MoE in production
  - DeepSeekMoE: fine-grained experts
  - Load balancing and Expert Parallelism
- State Space Models (SSM)
  - The quadratic complexity problem of attention
  - Mamba: selective state spaces
  - Mamba-2: Structured State Space Duality
  - When SSMs outperform Transformers
- Hybrid architectures
  - Jamba: SSM + Attention + MoE
  - NVIDIA Bamba, IBM Granite 4.0
  - Architectural patterns of hybrid models
- Choosing an architecture: decision framework

### 01.8 Scaling Laws (`08_Scaling_Laws.md`)
- Original OpenAI Scaling Laws
  - Power-law dependencies of loss on N, D, C
  - Practical implications
- Chinchilla: rethinking optimal scaling
  - D_optimal ≈ 20 × N
  - Compute-optimal training
- Emergence: qualitative leaps at scale
  - The phenomenon of emergent abilities
  - Grokking: delayed learning
- Scaling Laws for downstream tasks
- Practical implications
  - Choosing model size
  - Trade-off: Train-time vs Inference-time compute
- Test-Time Compute Scaling
  - o1/o3 models
  - A new dimension of scaling

### 01.9 Interpretability (`09_Interpretability.md`)
- Why understanding what's inside a model matters
- Circuits and Features
  - Attention heads as pattern matchers
  - MLP layers as memory
  - Composition of simple components
- Superposition Hypothesis
  - More features than dimensions
  - Sparse Autoencoders (SAE) for decomposition
- Probing and Activation Analysis
  - Linear probes for specific features
  - Activation patching and causal tracing
- Logit Lens and Tuned Lens
  - Reading intermediate representations
  - Evolution of predictions through layers
- Connection to alignment and safety
- Current limitations of interpretability

### 01.10 Implementation from Scratch (`10_Implementation_from_Scratch.md`)
- Why understanding the internals matters
- Self-Attention mechanism
  - Query, Key, Value intuition
  - Scaling and softmax
  - Multi-Head Attention
- Positional Encoding
  - Sinusoidal positions
  - Learned vs Fixed positions
- Tokenization
  - BPE algorithm
  - Building the vocabulary
- Architectural components
  - Layer Normalization
  - Feed-Forward Network
  - Residual Connections
- Assembling the Transformer Block
- Text generation
  - Temperature and Top-K/Top-P sampling
- Bridging theory and practice

---

## 02. Prompt Engineering

### 02.1 Introduction to Prompting (`01_Prompting_Basics.md`)
- The prompt as a natural language program
- Why this is critical for AI agents
- Anatomy of an effective prompt
  - Roles (system, user, assistant)
  - Structure and components
- Basic prompting techniques
  - Zero-shot prompting
  - Few-shot prompting
  - Chain-of-Thought (CoT)
- Structuring prompts
- Common mistakes and how to avoid them
- Iterative development process

### 02.2 Advanced Prompting (`02_Advanced_Prompting.md`)
- From basic techniques to mastery
- Self-Consistency: the power of consensus
- Tree of Thoughts: exploring the solution space
- ReAct: merging reasoning and action
- Meta-Prompting: the model improves prompts
- Role-Based Prompting: expert roles
- Structured Output: guaranteed formats
- Prompt Chaining: decomposing complexity

### 02.3 Agent Prompts (`03_Agent_Prompts.md`)
- The special nature of agent prompts
- System prompt: the agent's constitution
- Role-based design
- Tool descriptions: the art of precision
- Planning prompts
- Prompts for reflection and self-correction
- Multi-agent prompts
- Agent prompt anti-patterns

### 02.4 Prompt Optimization (`04_Prompt_Optimization.md`)
- From a working prompt to production-ready
- Prompt testing: the foundation of optimization
- A/B testing prompts
- Token optimization
- Prompt versioning
- Production monitoring
- Automated optimization
- Optimization checklist

### 02.5 Context Engineering (`05_Context_Engineering.md`)
- Evolution: from Prompt Engineering to Context Engineering (Karpathy 2025)
- Context Engineering philosophy: everything that shapes the context
- WSCI context management strategies
  - Write: generating context (scratchpads, tool outputs)
  - Select: choosing what's relevant (RAG, search, filtering)
  - Compress: compressing context (summarization, distillation)
  - Isolate: separating streams (multi-agent delegation)
- Memory Management: MemGPT, LangMem
- Context budgets and token economics
- Practical patterns: Router, Memory Manager, Summarizer

---

## 03. AI Agents Core

### 03.1 What is an AI Agent (`01_What_is_AI_Agent.md`)
- From chatbots to autonomous systems
- Chatbot vs AI agent: fundamental differences
- Agent work cycle: Observe → Think → Act → Reflect
- Autonomy levels (L1-L5)
- Agent components
- Task types for agents
- Advantages of AI agents
- Limitations and risks
- When to use agents (and when not to)
- **Iterative approach to building agents**
  - "Start simple, add complexity as needed"
  - Step-by-step evolution: Augmented LLM → Tools → Memory → Reasoning → Multi-agent
  - Anthropic's recommendation (December 2024)

### 03.2 Agent Architectures (`02_Agent_Architectures.md`)
- Introduction
- **Academic foundations of agent architectures**
  - Russell & Norvig: Perceive → Reason → Act → Learn
  - Wooldridge: Autonomy, Social Ability, Reactivity, Proactivity
  - Andrew Ng's 4 Agentic Design Patterns (Reflection, Tool Use, Planning, Multi-agent)
  - Validated academic patterns (ReAct, Plan-and-Execute, Self-Refine, Reflexion, ToT)
- ReAct: Reasoning and action as a unified process
- Plan-and-Execute: Think first, then act
- Reflexion: Learning through self-analysis
- LATS: Exploring the solution space
- Cognitive architectures: Inspired by the human mind
- Choosing an architecture: Practical recommendations
- Architecture comparison table

### 03.3 Tool Use (`03_Tool_Use.md`)
- Introduction
- The Function Calling concept
- Anatomy of a tool
  - JSON Schema description
  - Input/Output
- Types of tools
- Tool selection strategies
- Execution and error handling
- Tool security
- Integration with LangChain4j

### 03.4 Planning (`04_Planning.md`)
- Introduction
- Task decomposition
- Hierarchical planning
- GOAP algorithm (Goal-Oriented Action Planning)
- Planning with LLMs
- Adaptive planning
- Parallel execution

### 03.5 Memory Systems (`05_Memory_Systems.md`)
- Introduction
- Types of memory
  - Short-term memory
  - Long-term memory
  - Episodic memory
  - Semantic memory
  - Working memory
- Integrated memory architecture
- Retrieval-Augmented Memory
- Memory prioritization
- Forgetting mechanisms
- Persistence and storage

### 03.6 Computer Use Agents (`06_Computer_Use_Agents.md`)
- Introduction: AI masters the human interface
- Claude Computer Use: architecture and capabilities
  - Three tools: computer, text_editor, bash
  - Coordinate system and screenshot analysis
- Technical challenges
  - Latency: 1-3 seconds per action
  - Coordinate accuracy
  - Context via screenshots
- Computer Use security
  - Sandboxing and isolation
  - Allowlist/Denylist approaches
  - Human-in-the-loop for critical operations
- OSWorld Benchmark: 14.9% success rate
- Use Cases: QA automation, Legacy integration, Data entry
- The future of Computer Use agents

### 03.7 Code Generation Agents (`07_Code_Generation_Agents.md`)
- From autocomplete to autonomous developers
- IDE-integrated tools
  - GitHub Copilot: 55% accepted suggestions
  - Cursor: AI-first IDE
  - Codeium, Tabnine
- Autonomous coding agents
  - Devin: the first AI Software Engineer
  - OpenHands (ex-OpenDevin): open-source alternative
  - Claude Code: Anthropic's terminal agent
- SWE-bench: the evaluation standard
  - Evolution: 1.96% → 72.0% (Claude Opus 4)
  - SWE-bench Verified subset
- Architectural patterns
  - Plan → Edit → Test → Debug loop
  - Codebase understanding via RAG
  - Multi-file editing
- Code agent security
  - Sandboxed execution
  - Git-based changes
  - Review before commit

### 03.8 Process Reward Models (`08_Process_Reward_Models.md`)
- Why separate reward models for agents
- Process vs Outcome Reward Models
  - PRM: reward for each step
  - ORM: reward only for the result
  - When to use which
- PRM Architecture
  - Step-level scoring
  - Aggregation strategies
- Reasoning verification
  - Math reasoning validation
  - Code generation verification
- Reward hacking in agents
  - Exploitation problems
  - Mitigation strategies
- Connection to RLHF and Test-Time Compute

### 03.9 Practical AI Agent Use Cases (`09_Agent_Use_Cases.md`)
- Taxonomy of AI agent applications
- **Coding Agents** — the dominant category
  - Cursor, GitHub Copilot, Devin ($4B+)
  - Architectural patterns
- **Agentic RAG** — intelligent search
  - Perplexity ($9B, 22M+ users)
  - Router and Multi-Agent patterns
- **Workflow Automation** — the no-code revolution
  - n8n (200K+ businesses, MCP support)
- **Voice Agents** — latency breakthrough (75ms ElevenLabs)
- **Computer Use Agents** — Anthropic, OpenAI Operator (Beta)
- **Browser Automation** — Browser Use (27K+ stars), Amazon Nova Act
- **Deep Research** — Gemini Deep Research, OpenAI Deep Research
- Criteria for choosing a use case

### 03.10 Resource Optimization (`10_Resource_Optimization.md`)
- AI agent economics: Cost-aware approach
- Cost-Aware Model Routing
  - Tiered architecture: Simple → Standard → Complex → Reasoning
  - Confidence-based routing
  - Token economics
- Token Budgeting
  - Static vs Dynamic budgets
  - Per-step allocation
  - Graceful degradation
- Semantic Caching
  - Exact match vs Semantic similarity
  - Cache invalidation strategies
  - Cost/benefit analysis
- Circuit Breaker Pattern for LLMs
  - States: Closed → Open → Half-Open
  - Fallback strategies
- Rate Limiting and Priority Queues
- Monitoring and Cost Alerts

---

## 04. Multi-Agent Systems

### 04.1 Multi-Agent System Fundamentals (`01_MAS_Basics.md`)
- Introduction to the world of multiple agents
- Why a single agent is not enough
- What a multi-agent system is
- Inter-agent communication protocols
- Agent roles and specialization
- Coordination challenges
- Agent lifecycle

### 04.2 Multi-Agent System Patterns (`02_MAS_Patterns.md`)
- Architectural patterns as a design language
- Supervisor pattern
- Hierarchical agents
- Peer-to-Peer collaboration
- Debate pattern (Adversarial/Debate)
- Swarm Intelligence
- Mixture of Experts for agents
- Handoff pattern

### 04.3 Agent Orchestration (`03_Agent_Orchestration.md`)
- The conductor of an invisible orchestra
- Routing and dispatching
- Load balancing
- Conflict resolution
- Consensus mechanisms
- Agent lifecycle management
- Scaling

### 04.4 MAS Frameworks (`04_MAS_Frameworks.md`)
- Why frameworks are needed
- LangGraph: graph-oriented orchestration
- AutoGen: multi-agent dialog from Microsoft
- CrewAI: role-based collaboration
- Semantic Kernel: enterprise orchestration from Microsoft
- OpenAI Swarm: experimental simplicity
- Comparison and selection

### 04.5 Consensus and Reliability of Multi-Agent Systems (`05_MAS_Consensus_and_Reliability.md`)
- When agents must agree
- Types of consensus in AI systems
- Voting mechanisms
  - Simple Voting, Weighted Voting, Quorum
- Adversarial consensus: debates and verification
- Distributed consensus protocols (CFT, BFT)
- Market-based consensus mechanisms
- Error propagation: how errors spread in MAS
- Evaluating multi-agent system quality

---

## 05. MCP Protocol (Model Context Protocol)

### 05.1 MCP Basics (`01_MCP_Basics.md`)
- Introduction: a new standard for AI interaction
- The problem MCP solves
- What MCP is
- MCP Architecture (Host, Client, Server)
- MCP advantages
- MCP in the context of agent systems

### 05.2 MCP Components (`02_MCP_Components.md`)
- Introduction to the MCP component model
- Resources: access to data and content
  - URI schemes
  - MIME types
  - Resource templates
- Tools: executing actions
  - Input schemas
  - Progress reporting
  - Cancellation
- Prompts: interaction templates
- Sampling: reverse model queries
- Component interaction
- Lifecycle and discovery

### 05.3 MCP Server Development (`03_MCP_Server_Development.md`)
- Introduction to server development
- Architectural principles
- Server lifecycle
- Transport mechanisms (stdio, HTTP/SSE)
- Error handling
- Security
- **Modern MCP ecosystem**
  - FastMCP — de facto standard Python SDK
  - MCP Inspector (CVE warning for versions <0.14.1)
  - Python SDK mcp v1.25.0+, TypeScript @modelcontextprotocol/sdk
  - Support: Claude Desktop, VS Code, Cursor
  - MCP transferred to Linux Foundation (December 2024)
- Testing MCP servers
- Monitoring and observability

### 05.4 MCP Client Integration (`04_MCP_Client_Integration.md`)
- Introduction to client integration
- Client application architecture
- Connecting to servers
- Aggregating multiple servers
- Integration with language models
- User interface
- Client-side security

### 05.5 Agent-to-Agent Protocol (A2A) (`05_A2A_Protocol.md`)
- Introduction: inter-agent communication
- MCP vs A2A: different levels of abstraction
  - Vertical vs horizontal integration
  - Complementary protocols
- A2A Architecture
  - Agent Cards: Discovery and Capabilities
  - Tasks: synchronous, asynchronous, streaming
  - Messages and Artifacts
- Task lifecycle
- Security model
  - Authentication and Authorization
  - Capability-based access
- Comparison with alternatives
- MCP + A2A integration

---

## 06. RAG (Retrieval-Augmented Generation)

### 06.1 RAG Basics (`01_RAG_Basics.md`)
- Introduction to RAG
- Problems RAG solves
  - Information freshness
  - Hallucinations
  - Domain-specific knowledge
- RAG system architecture
  - Indexing pipeline
  - Retrieval pipeline
  - Generation pipeline
- **Practical Pipeline**
  - Ingest: Firecrawl, Crawl4AI, Unstructured.io
  - Chunk: LlamaIndex, LangChain, Jina Late Chunking
  - Embed: OpenAI, Cohere, bge-m3 (multilingual)
  - Store: Qdrant, Weaviate, Pinecone, pgvector
  - Rerank: Cohere Rerank, bge-reranker-v2-m3
  - Evaluate: RAGAS, LangSmith, Phoenix
- Comparison with alternative approaches
- RAG quality metrics

### 06.2 Chunking Strategies (`02_Chunking_Strategies.md`)
- Introduction to chunking
- Fundamental considerations
  - Chunk size
  - Overlap
- Splitting strategies
  - Fixed-size chunking
  - Sentence-based
  - Semantic chunking
  - Recursive chunking
- Document type specifics
- Chunking optimization

### 06.3 Embeddings and Vector Stores (`03_Embeddings_and_Vector_Stores.md`)
- The nature of embeddings
- How embedding models work
- Choosing an embedding model
  - OpenAI text-embedding-3
  - Sentence Transformers
  - Multilingual models
- Vector stores
  - Pinecone, Weaviate, Milvus
  - Chroma, Qdrant
  - pgvector
- Index types (HNSW, IVF, PQ)
- Hybrid search
- Efficiency optimization
- Monitoring and metrics

### 06.4 Retrieval Methods (`04_Retrieval_Methods.md`)
- The role of retrieval in RAG
- Basic search methods
  - Semantic search
  - Keyword search (BM25)
  - Hybrid search
- Advanced retrieval techniques
  - Query expansion
  - HyDE
  - Multi-query retrieval
- Reranking: refining results
  - Cross-encoder reranking
  - Cohere Rerank
- Contextual retrieval
- Filtering and metadata
- Iterative and multi-hop retrieval

### 06.5 Advanced RAG (`05_Advanced_RAG.md`)
- Beyond basic RAG
- **RAG architecture taxonomy**
  - Naive RAG (Retrieve-Read)
  - Retrieve-and-Rerank (two-stage)
  - Hybrid RAG (dense + sparse: BM25/TF-IDF)
  - Graph RAG (Microsoft GraphRAG v1.0)
  - Multimodal RAG (ColPali, GPT-4V)
  - Agentic RAG Router (LLM selects the source)
  - Agentic RAG Multi-Agent (agent coordination)
- Corrective RAG (CRAG)
- Self-RAG
- **Latest techniques**
  - CAG (Cache-Augmented Generation) — loading the database into extended context
  - Late Chunking (Jina AI) — chunking after full embedding
  - Contextual Retrieval (Anthropic) — LLM generates context for chunks
  - ColPali — vision-language for multimodal retrieval
- Evaluation and quality improvement
- Performance optimization

### 06.6 Late Interaction Retrieval (`06_Late_Interaction_Retrieval.md`)
- Evolution of Retrieval architectures: Sparse → Dense → Late Interaction
- Late Interaction: principles and advantages
  - Multi-vector document representation
  - MaxSim operator for scoring
  - Token-level matching
- ColBERT: architecture and mechanism
  - Query encoder vs Document encoder
  - Precomputed document embeddings
  - Efficient ANN search
- Modern Late Interaction models
  - Jina-ColBERT-v2: multilingual, 8K context
  - GTE-ModernColBERT: Alibaba, production-ready
  - ColPali: Vision-Language Late Interaction
- PyLate: Python library for Late Interaction
  - Inference and Training
  - Hugging Face integration
- Hybrid Architecture: Dense + Late Interaction
- Trade-offs and when to use

---

## 07. Frameworks

### 07.0 AI Frameworks Overview (`00_Frameworks_Overview.md`)
- Framework categories by abstraction level
  - High-level (No-code): n8n, Make, Zapier AI
  - Mid-level (Agent Frameworks): CrewAI, LangGraph, AutoGen
  - Low-level (Building Blocks): LangChain, LlamaIndex
  - Provider SDKs: OpenAI, Anthropic, Google
- **Key player comparison table**
  - LangChain/LangGraph: 80K+ stars, 4.2M downloads/month
  - CrewAI: 30K+ stars, $18M Series A
  - AutoGen → Microsoft Agent Framework (merge Q1 2026)
  - AWS Bedrock Agents: 100K+ organizations
  - n8n: 200K+ businesses, native MCP support
- LangChain vs LangGraph: when to use which
- Provider SDKs: OpenAI Agents SDK (March 2025), Claude SDK
- Framework selection criteria

### 07.1 LangChain4j (`01_LangChain4j.md`)
- Introduction to LangChain4j
- Architecture and core components
  - ChatLanguageModel
  - AIServices
  - Memory
- Declarative AI Services
- Spring integration
- Working with tools

### 07.2 Spring AI (`02_Spring_AI.md`)
- Introduction to Spring AI
- Project philosophy
- Spring AI architecture
  - ChatClient
  - Prompt templates
  - Advisors
- Configuration and auto-configuration
- Function Calling
- Retrieval Augmented Generation
- Observability

### 07.3 Semantic Kernel (`03_Semantic_Kernel.md`)
- Introduction to Semantic Kernel
- Architectural concepts
  - Kernel
  - Plugins
  - Functions
- Plugin development
- Provider integration
- Connectors and extensions
- Observability and debugging

### 07.4 JAX Ecosystem (`04_JAX_Ecosystem.md`)
- JAX philosophy: NumPy + Autodiff + XLA
- Key transformations (jit, grad, vmap, pmap)
- Ecosystem: Flax, Optax, Orbax
- JAX vs PyTorch: when to choose which
- Practical patterns
- Migrating from PyTorch

### 07.5 AWS Strands Agents (`05_AWS_Strands_Agents.md`)
- AWS Strands: Open-Source SDK for agents (May 2025, GA July 2025)
- Model-Driven vs Graph-Driven approaches
  - Model-Driven: LLM as orchestrator (Strands, Claude)
  - Graph-Driven: Explicit workflows (LangGraph, Prefect)
  - Trade-off comparison table
- AWS Strands architecture
  - Agent → Model → Tools lifecycle
  - Automatic tool discovery
  - Built-in conversation memory
- Multi-Agent patterns
  - Agents-as-Tools: agents as tools of other agents
  - Handoffs: transferring control
  - Swarm pattern: dynamic coalitions
- MCP Support: native integration
- AWS Bedrock AgentCore (Preview)
  - Enterprise guardrails
  - Identity propagation
  - Unified observability
- When to use Model-Driven vs Graph-Driven

---

## 08. Structured Outputs

### 08.1 Structured Output Techniques (`01_Structured_Output_Techniques.md`)
- Introduction: when text is not enough
- The unstructured text problem
- Structured output guarantee levels
  - Prompt-based
  - JSON mode
  - Function calling
  - Grammar-constrained
- Choosing the method for your task
- Frameworks and libraries

### 08.2 Data Extraction (`02_Data_Extraction.md`)
- Introduction: turning chaos into order
- Types of extraction tasks
  - Named Entity Recognition
  - Relation extraction
  - Event extraction
- Extraction strategies
- Working with context
- Handling ambiguity
- Scaling extraction

### 08.3 Validation and Error Handling (`03_Validation_and_Error_Handling.md`)
- Introduction: why the model isn't always right
- Validation levels
  - Syntactic validation
  - Schema validation (JSON Schema)
  - Semantic validation
  - Business logic validation
- Error handling strategies
- Feedback loops (retry with correction)
- Handling edge cases
- Logging and monitoring

---

## 09. Conversational AI

### 09.1 Dialog Management (`01_Dialog_Management.md`)
- Introduction: conversation as a dance
- Anatomy of a dialog
- Dialog state
- Intent recognition
- Slot filling
- Multi-turn dialogs

### 09.2 Conversation Design (`02_Conversation_Design.md`)
- Introduction: the art of dialog
- Conversation flows
- Handling misunderstandings
- Fallback strategies
- Personality and tone
- Greetings and farewells
- Handling problematic situations
- Personalization

### 09.3 Voice and Multimodality (`03_Voice_and_Multimodality.md`)
- Introduction to multimodal interfaces
- Voice interface: from sound to meaning
  - Speech-to-Text
  - Text-to-Speech
- Voice agent architecture
- Computer vision in conversational systems
- Multimodal agents
- Voice interface design specifics
- Security and ethics of multimodal systems

---

## 10. Fine-Tuning

### 10.1 Fine-Tuning Basics (`01_Fine_Tuning_Basics.md`)
- What fine-tuning is
- Where fine-tuning fits in the methods ecosystem
- When fine-tuning is justified
- When fine-tuning is not appropriate
- The decision-making process for choosing an approach
- Data requirements for fine-tuning
- Cost and resources
- Risks and limitations

### 10.2 Data Preparation (`02_Data_Preparation.md`)
- The importance of data quality
- Data sources
- Data formats (JSONL, conversations)
- Structure of a quality example
- Data quality criteria
- Cleaning and filtering
- Data augmentation
- Train/validation/test split
- Handling imbalance
- Dataset documentation

### 10.3 Techniques and Evaluation (`03_Techniques_and_Evaluation.md`)
- Full fine-tuning vs Parameter-Efficient methods
- LoRA: Low-Rank Adaptation
  - Rank selection
  - Target modules
  - Alpha parameter
- LoRA hyperparameter selection
- QLoRA: quantized adaptation
- OpenAI Fine-Tuning API
- Alternative platforms
- Fine-tuned model quality evaluation
- Metric types
- Iterative improvement process
- Production monitoring
- Model versioning and management

### 10.4 RLHF and Alignment (`04_RLHF_and_Alignment.md`)
- Why classical fine-tuning is not enough
- RLHF Architecture
  - SFT (Supervised Fine-Tuning)
  - Reward Modeling
  - PPO (Proximal Policy Optimization)
- Reward Modeling: the heart of RLHF
  - Preference data collection
  - Bradley-Terry model
  - Reward hacking
- DPO: Direct Preference Optimization
  - Implicit reward model
  - Reference model
- RLAIF: AI as annotator
  - Constitutional AI
  - Self-improvement
- Practical alternatives for applied tasks
  - Rejection sampling
  - Best-of-N
- Alignment: broader than RLHF
  - Safety alignment
  - Value alignment

### 10.5 RLHF Alternatives (`05_RLHF_Alternatives.md`)
- Problems with classical RLHF
  - PPO instability
  - Computational cost
  - Reward model bottleneck
- DPO: Direct Preference Optimization
  - Closed-form reward optimization
  - Reference model requirement
  - Practical implementation
- KTO: Kahneman-Tversky Optimization
  - Binary signal (no paired comparisons)
  - Loss aversion principle
  - Working with imbalanced data
- ORPO: Odds Ratio Preference Optimization
  - No reference model needed
  - SFT + alignment in one step
- SimPO: Simple Preference Optimization
  - Length normalization
  - Maximum simplicity
- Method comparison table
- Hugging Face TRL Integration

### 10.6 Test-Time Compute (`06_Test_Time_Compute.md`)
- OpenAI o1/o3: a reasoning breakthrough
  - AIME 2024: 13% → 83%
  - Extended thinking and self-correction
- Test-Time Compute mechanisms
  - Process Reward Models (PRMs)
  - Best-of-N Sampling
  - Self-Consistency
  - Tree Search with Value Function
  - Budget Forcing
- DeepSeek-R1: Open-Source Reasoning
  - Pure RL approach
  - Distillation into small models
- Compute-Optimal Inference Strategies
  - 4× efficiency improvement
  - Smaller models + TTC vs larger models

### 10.7 Synthetic Data (`07_Synthetic_Data.md`)
- Self-Instruct: instruction generation
  - Self-Instruct pipeline
  - Limitations and risks
- Evol-Instruct: complexity evolution
  - Evolution types (constraints, deepen, concretize)
  - WizardLM results
- Distillation: teacher → student
  - Response, Rationale, Preference distillation
  - Legal and ethical considerations
- Data Augmentation for Fine-Tuning
  - Paraphrasing, Back-translation
  - Style transfer
- Quality Control for Synthetic Data
  - Multi-layer filtering
  - LLM-as-Judge
- Model Collapse: the danger of the synthetic loop
  - Collapse mechanism
  - Mitigation strategies
- Practical Pipelines

### 10.8 Continued Pretraining (`08_Continued_Pretraining.md`)
- Why continued pretraining matters
  - Domain adaptation without losing general knowledge
  - When CPT vs SFT vs few-shot
- Domain-Adaptive Pretraining
  - Preparing the domain corpus
  - Mixing strategies with general data
- Catastrophic Forgetting
  - Why models "forget"
  - Mitigation: replay, regularization, architecture
- Data Mixing Strategies
  - Domain vs general proportions
  - Curriculum strategies
- Practical aspects
  - Learning rate scheduling
  - Evaluation on domain and general tasks

### 10.9 Preference Data (`09_Preference_Data.md`)
- The importance of quality preference data for alignment
- Bradley-Terry Model
  - Mathematical formulation
  - Connection to Elo ratings
- Collecting human preferences
  - Annotation guideline design
  - Inter-annotator agreement
  - Active learning for efficient collection
- Synthetic preferences (RLAIF)
  - Constitutional AI for generation
  - Synthetic data quality control
- Data Quality and Alignment Quality
  - Noise in labels
  - Bias in annotators
- Practical pipelines for preference collection

### 10.10 DoRA and Advanced PEFT Methods (`10_DoRA_and_Beyond.md`)
- Beyond LoRA: the evolution of PEFT methods
- DoRA (Weight-Decomposed Low-Rank Adaptation)
  - Decomposition: magnitude + direction
  - ICML 2024 results: +3.7% on LLaMA-7B
- GaLore (Gradient Low-Rank Projection)
  - Full-parameter training with low memory
- QA-LoRA: Quantization-aware LoRA
- LongLoRA: Efficient long context fine-tuning
- Method comparison table
- Decision framework: when to choose what

---

## 11. Evaluation & Testing

### 11.1 Metrics and Benchmarks (`01_Metrics_and_Benchmarks.md`)
- Why LLM evaluation is a hard problem
- Multi-dimensionality of quality
  - Correctness, Relevance, Completeness
  - Brevity, Coherence, Fluency
  - Usefulness, Safety
- Automatic text-matching metrics
  - Exact Match
  - Token F1
  - BLEU
  - ROUGE
- Limitations of text metrics
- Semantic metrics
  - Embedding similarity
  - BERTScore
  - MoverScore
- Benchmarks and standard datasets
  - GLUE, SuperGLUE
  - MMLU
  - HellaSwag
  - HumanEval
  - MT-Bench
- Principles of metric selection
- Building an evaluation dataset

### 11.2 Human Evaluation (`02_Human_Evaluation.md`)
- Why human evaluation remains the gold standard
- Task design for evaluators
- Rating scales (Likert, ordinal)
- Pairwise comparison
- Evaluator selection
- Number of evaluators and statistical power
- Bias in human evaluation
- Organizing the evaluation process
- When to use human evaluation

### 11.3 LLM as Judge (`03_LLM_as_Judge.md`)
- The idea of using LLMs for evaluation
- LLM-as-Judge architecture
- The judge prompt
- Evaluation modes
  - Single rating
  - Reference-based
  - Pairwise comparison
  - Multi-aspect
- LLM judge biases
  - Self-enhancement bias
  - Verbosity bias
  - Position bias
  - Sycophancy
- Calibration and validation
- Reliability improvement strategies
  - Multi-judge
  - Debate evaluation
- When to use LLM-as-Judge
- Specialized evaluation types
- Scaling and automation

### 11.4 Continuous Evaluation (`04_Continuous_Evaluation.md`)
- From one-time evaluation to continuous monitoring
- Continuous evaluation system architecture
- Sampling strategies
  - Random sampling
  - Stratified sampling
  - Importance sampling
  - Error-triggered sampling
  - Coverage sampling
- A/B testing
- A/B test metrics
  - Guardrail metrics
  - Primary metrics
  - Secondary metrics
- Statistical significance
- Regression testing in CI/CD
- Monitoring and alerting
- Feedback loops
- Organizational aspects

### 11.5 RAG Evaluation (`05_RAG_Evaluation.md`)
- The specifics of RAG system evaluation: Retrieval + Generation
- BEIR Benchmark
  - 18 datasets, 9 tasks
  - Zero-shot evaluation protocol
  - Metrics: nDCG@10, Recall@100
- FreshStack (2024)
  - Real-world enterprise data
  - Realistic queries
  - Multi-domain evaluation
- 4 dimensions of RAG quality
  - Context Precision: relevance of retrieved results
  - Context Recall: coverage completeness
  - Faithfulness: fidelity to sources
  - Answer Relevance: correspondence to the question
- RAGAS Framework
  - Automatic metrics
  - LLM-as-Judge approach
  - Synthetic test generation
- 8 scenarios before production
  - Failure mode analysis
  - Edge cases
  - Adversarial queries

---

## 12. Observability

### 12.1 Tracing and Logging (`01_Tracing_and_Logging.md`)
- Unique challenges of LLM system observability
- Three pillars of observability (Logs, Metrics, Traces)
- Anatomy of an LLM application trace
- What to include in a trace
- Structured logging
- OpenTelemetry for LLMs
- Specialized LLM Observability platforms
  - LangSmith, LangFuse
  - Weights & Biases
- Storage and retention

### 12.2 Metrics and Dashboards (`02_Metrics_and_Dashboards.md`)
- Metrics philosophy in the world of language models
- LLM system metrics hierarchy
- Tokens as the currency of the LLM economy
- Latency: from first token to completion
  - Time to First Token (TTFT)
  - Percentiles (P50, P95, P99)
- Quality metrics: measuring the immeasurable
  - Faithfulness, Relevance
  - Hallucination rate
- Business metrics and ROI
  - Cost per conversation
  - Resolution rate
  - User satisfaction
- Metrics collection architecture
- Dashboards: from information to understanding
- Alerting: balancing sensitivity and noise
- Cost monitoring and budgeting
- Production quality monitoring

### 12.3 LLM Debugging (`03_LLM_Debugging.md`)
- The unique nature of bugs in LLM systems
- Taxonomy of LLM system problems
- Debugging toolkit
- Reproducibility — the holy grail of LLM debugging
- Debugging response quality issues
- Debugging RAG systems
- Debugging call chains
- Working with edge cases and adversarial inputs
- Performance debugging
- Production debugging
- Tools for systematic debugging

### 12.4 AgentOps (`04_AgentOps.md`)
- AgentOps: DevOps + MLOps for AI agents
- Evolution: DevOps → MLOps → LLMOps → AgentOps
- AgentOps Lifecycle
  - Plan: architecture and requirements
  - Build: development with observability
  - Evaluate: offline and online testing
  - Deploy: staging → canary → production
  - Monitor: real-time observability
- Key AgentOps metrics
  - Session Success Rate
  - Token Efficiency
  - Tool Execution Metrics
  - Guardrail Trigger Rate
  - Human Escalation Rate
- Ecosystem tools
  - AgentOps.ai: specialized platform
  - LangSmith: LangChain ecosystem
  - LangFuse: open-source alternative
  - Phoenix (Arize): enterprise observability
- Session-Based Tracing vs Request-Based
- Cost Attribution and Budgeting
- Incident Response for agents

---

## 13. Deployment

### 13.1 Deployment Strategies (`01_Deployment_Strategies.md`)
- Deployment specifics in the language model world
- LLM application architecture
  - API Gateway
  - Application Layer
  - LLM Orchestration Layer
- Synchronous deployment pattern
- Asynchronous pattern with polling
- Server-Sent Events and streaming
- WebSocket for interactive agents
- Circuit Breaker and Fallback strategies
- Timeout management
- LLM response caching
  - Exact match caching
  - Semantic caching

### 13.2 Containerization and Kubernetes (`02_Containerization_and_Kubernetes.md`)
- Containerization as the foundation of modern deployment
- Docker image optimization
  - Multi-stage builds
- Configuring JVM for containers
- Health checks and graceful shutdown
- Docker Compose for local development
- Kubernetes: basics for LLM applications
  - Deployments and ReplicaSets
- Service and Ingress
- Horizontal Pod Autoscaler
- Resource Management
- ConfigMaps and Secrets

### 13.3 CI/CD and Self-Hosted (`03_CI_CD_and_Self_Hosted.md`)
- Continuous integration for LLM applications
- CI Pipeline structure
- Continuous Deployment and release strategies
  - Canary deployments
  - Blue-green deployments
- Smoke Tests and Quality Gates
- Self-hosted models: why and when
- vLLM: high-performance inference
- Infrastructure for GPU workloads
- Inference optimization
- Hybrid approach: self-hosted + cloud
- Inference server monitoring

---

## 14. Security & Safety

### 14.1 Prompt Injection (`01_Prompt_Injection.md`)
- A new class of vulnerabilities
- Anatomy of direct injection attacks
- Indirect Prompt Injection: the hidden threat
- Jailbreaking: bypassing restrictions
- Multi-layered defense
- Sandwich Defense and context isolation
- LLM-as-Guard: using the model for protection
- System prompt protection

### 14.2 Data Protection (`02_Data_Protection.md`)
- New dimensions of the privacy problem
- PII Detection: the first line of defense
- Masking and anonymization
- System prompt protection
- Data Minimization and Retention
- Encryption and key practices
- Audit Logging for Compliance
- Integration with external providers

### 14.3 Agent Security (`03_Agent_Security.md`)
- Autonomy as a source of risk
- Principle of least privilege
- Permission System for tools
- Code execution sandboxing
- Static code analysis
- Human-in-the-Loop
- Rate Limiting and anomalies
- Action logging and auditing
- Graceful degradation
- Agent security testing

### 14.4 Moderation and Compliance (`04_Moderation_and_Compliance.md`)
- Content responsibility
- Moderation architecture
- Harmful content categories
- Input content moderation
- Output content moderation
- Factual Claims and disinformation
- Compliance frameworks
  - GDPR, HIPAA, SOC 2
- Data retention and deletion
- Security monitoring and incident response
- Transparency and accountability

### 14.5 NeMo Guardrails (`05_NeMo_Guardrails.md`)
- Open-source toolkit from NVIDIA
- Colang: the Rails definition language
  - User Messages and Intents
  - Bot Messages
  - Flows
- Types of Rails
  - Input Rails
  - Output Rails
  - Dialog Rails
  - Retrieval Rails
- Practical configuration
- Production Deployment
- Comparison with alternatives (Guardrails AI, Custom)
- Best Practices: Layered Defense, Graceful Degradation

---

## 15. GPU Architecture

### 15.1 GPU Architecture (`01_GPU_Architecture.md`)
- From CPU to GPU: the evolution of ML computation
- GPU memory hierarchy
  - HBM (High Bandwidth Memory) — 2-3 TB/s
  - SRAM — up to 19 TB/s
  - Registers — thread memory
- CUDA programming model
  - Threads, Blocks, Grids
  - Occupancy and its optimization
- Tensor Cores: specialized ML cores
  - Hardware acceleration of matrix operations
  - Mixed precision computation
- Memory coalescing and bandwidth optimization
- GPU operation profiling

### 15.2 Flash Attention (`02_Flash_Attention.md`)
- The quadratic complexity problem of attention
- IO-awareness as the key idea
- The Flash Attention algorithm
  - Tiling: block decomposition
  - Online softmax: processing without materialization
  - Recomputation in the backward pass
- Memory savings: up to 20× at 4K sequence length
- Flash Attention 2 and 3: evolution of optimizations
- Integration with PyTorch and Hugging Face
- Practical benchmarks

### 15.3 Triton Programming (`03_Triton_Programming.md`)
- Triton vs CUDA: the complexity-flexibility tradeoff
- Block-level programming model
- Writing custom kernels
  - Fused Softmax example
  - Hyperparameter autotuning
- When Triton vs when CUDA
- PyTorch integration
- Debugging and profiling Triton kernels

### 15.4 GPU Profiling (`04_GPU_Profiling.md`)
- Profiling tools
  - torch.profiler
  - Nsight Systems
  - Nsight Compute
- Roofline Model: theoretical limits
- Identifying the bottleneck: compute vs memory bound
- Optimizing kernel launch overhead
- Practical approach to profiling

### 15.5 Quantization Deep Dive (`05_Quantization_Deep_Dive.md`)
- Why quantization matters (not just for inference)
- Quantization types
  - Weight quantization vs Activation quantization
  - Symmetric vs Asymmetric
  - Per-tensor vs Per-channel
- GPTQ: Post-Training Quantization
  - Optimal Brain Quantization foundation
  - Hessian-based weight adjustment
  - Calibration dataset requirements
- AWQ: Activation-aware Weight Quantization
  - Salient weight identification
  - Per-channel scaling
  - Comparison with GPTQ
- GGML/GGUF formats
  - CPU-optimized quantization
  - Support for various types (Q4_0, Q4_K_M, Q8_0)
- Quantization-Aware Training (QAT)
  - Fake quantization during training
  - When QAT is better than PTQ
- Trade-offs: Quality vs Speed vs Memory
- Practical application with vLLM and llama.cpp

---

## 16. Distributed Training

### 16.1 Distributed Training Basics (`01_Distributed_Training_Basics.md`)
- Why distributed training is needed
- Types of parallelism
  - Data Parallelism (DP)
  - Tensor Parallelism (TP)
  - Pipeline Parallelism (PP)
  - Expert Parallelism (EP)
- Communication primitives
  - AllReduce, AllGather, ReduceScatter
  - Ring AllReduce
- Collective communication libraries (NCCL)
- Calculating required resources

### 16.2 PyTorch FSDP2 (`02_PyTorch_FSDP2.md`)
- Evolution from DDP to FSDP
- FSDP2: architecture redesign
  - fully_shard API
  - Sharding granularity
- Sharding strategies
  - FULL_SHARD: all parameters
  - SHARD_GRAD_OP: gradients only
  - NO_SHARD: no sharding
- Mixed Precision Training
- Activation Checkpointing
- Checkpointing large models
- Metrics: 28.54% memory savings, 68.67% throughput increase

### 16.3 DeepSpeed ZeRO (`03_DeepSpeed_ZeRO.md`)
- The ZeRO optimization philosophy
- ZeRO Stage 1: Optimizer State Partitioning
- ZeRO Stage 2: Gradient Partitioning
- ZeRO Stage 3: Parameter Partitioning
- ZeRO-Offload: CPU offloading
- ZeRO-Infinity: NVMe offloading
- Configuration and setup
- HuggingFace Trainer integration

### 16.4 3D Parallelism (`04_3D_Parallelism.md`)
- Combining types of parallelism
- Megatron-DeepSpeed architecture
- Process group organization
  - TP groups, DP groups, PP groups
- Calculating optimal configuration
- Communication patterns
- Fault tolerance and checkpoint recovery
- Communication profiling

### 16.5 Distributed Training Practice (`05_Distributed_Training_Practice.md`)
- End-to-end distributed training example
- Setting up the environment and dependencies
- Launch scripts (torchrun, deepspeed)
- Checkpointing and recovery
- Fault tolerance strategies
- Debugging distributed training
- Best practices and common mistakes

---

## 17. Production Inference

### 17.1 Inference Architecture (`01_Inference_Architecture.md`)
- Autoregressive generation: token by token
- Inference phases
  - Prefill: prompt processing
  - Decode: token generation
- KV-cache: mechanics and size calculation
- Latency vs Throughput trade-offs
- Batching strategies
- GPU characteristics for inference

### 17.2 vLLM Internals (`02_vLLM_internals.md`)
- PagedAttention: a revolution in memory management
  - The fragmentation problem (60-80% waste)
  - Paging as a solution
  - Block tables and physical blocks
- Continuous Batching
  - Iteration-level scheduling
  - Preemption strategies
- Prefix Caching: KV-cache reuse
- Tensor Parallelism in vLLM
- Practical vLLM deployment

### 17.3 Speculative Decoding (`03_Speculative_Decoding.md`)
- The idea of speculative generation
- Draft models and verification
- EAGLE: draft-free speculative decoding
  - Autoregression head architecture
  - EAGLE-3: up to 2.5× speedup
- Medusa: parallel predictions
  - Multiple heads architecture
  - Tree attention
- Acceptance rate and its significance
- vLLM integration
- Choosing the method for the task

### 17.4 Model Quantization (`04_Model_Quantization.md`)
- Why quantization is needed
- Data types
  - FP32, FP16, BF16
  - INT8, INT4
  - FP8 (Hopper/Blackwell)
- Post-Training Quantization (PTQ)
- Quantization-Aware Training (QAT)
- GPTQ: precise 4-bit quantization
- AWQ: Activation-aware Weight Quantization
- Accuracy vs Latency trade-offs
- Practical application

### 17.5 Inference Cost Optimization (`05_Inference_Cost_Optimization.md`)
- LLM inference economics
- Spot instances: 60-90% savings
  - Resilience strategies
  - Multi-AZ deployment
- Model cascading
  - Router-based selection
  - Confidence thresholds
- Autoscaling strategies
  - Queue-based scaling
  - Predictive scaling
- Reserved vs On-demand vs Spot
- Cost monitoring and budgeting
- TCO calculation

### 17.6 Long Context Inference (`06_Long_Context_Inference.md`)
- The long context problem
  - KV-cache memory explosion
  - Quadratic attention cost
- RoPE Scaling Techniques
  - Position Interpolation (PI)
  - NTK-aware Scaling
  - YaRN (Yet another RoPE extensioN)
  - Dynamic NTK scaling
- Architectural Solutions
  - Sliding Window Attention (Mistral)
  - Landmark Attention
  - StreamingLLM approach
- Memory Requirements Scaling
  - KV-cache size calculations
  - Batch size vs context length trade-off
- Long Context vs RAG
  - When to use which
  - Cost/latency analysis
  - "Lost in the Middle" problem
  - Hybrid approaches

### 17.7 SGLang and Alternatives (`07_SGLang_and_Alternatives.md`)
- SGLang: the next step after vLLM
  - Frontend Language for LLM Programs
  - RadixAttention: KV-cache as a radix tree
  - Up to 3× throughput for chatbots
- Inference engine comparison
  - vLLM: PagedAttention reference
  - SGLang: structured generation + RadixAttention
  - TensorRT-LLM: maximum NVIDIA performance
  - Text Generation Inference (TGI): HuggingFace production
- Decision Framework: when to choose what
- SGLang Language Features
  - Primitives: gen, select, fork, join
  - Constrained decoding
- Practical deployment

---

## 18. AI Governance

### 18.1 Regulatory Landscape (`01_Regulatory_Landscape.md`)
- Introduction: AI at the intersection of law and technology
- EU AI Act: the first comprehensive law
  - Risk-based framework (4 risk levels)
  - Unacceptable risk: prohibited practices
  - High-risk AI: conformity assessment requirements
  - GPAI (General Purpose AI) obligations
  - Timeline: February 2025 — August 2027
- NIST AI Risk Management Framework
  - GOVERN, MAP, MEASURE, MANAGE
  - AI RMF Playbook
- Sector-specific regulations
  - FDA: AI/ML in medical devices
  - SEC: AI in financial services
  - FedRAMP: AI in the public sector
- Global regulatory map
- Practical compliance strategies

### 18.2 AI Risk Management (`02_AI_Risk_Management.md`)
- Why AI systems require a special approach to risk
- AI risk taxonomy
  - Technical risks (hallucinations, adversarial, drift)
  - Ethical risks (bias, privacy, fairness tradeoffs)
  - Operational risks (vendor lock-in, cost unpredictability)
  - Strategic risks (competitive displacement)
- NIST AI RMF: GOVERN, MAP, MEASURE, MANAGE
- ISO/IEC 23894: AI Risk Management
- Risk assessment methodology
  - AI System Inventory
  - Impact Assessment
  - FMEA for AI
- Mitigations: technical, organizational, contractual
- Continuous Monitoring

### 18.3 Red Teaming (`03_Red_Teaming.md`)
- From penetration testing to AI red teaming
- Taxonomy of attacks on AI systems
  - Prompt Injection (direct, indirect)
  - Jailbreaking (DAN, persona, encoding)
  - Data Poisoning (backdoors)
  - Model Extraction and Inversion
  - Adversarial Examples
- AI Red Teaming methodology
  - Preparation: scope, threat modeling
  - Execution: automated + manual testing
  - Reporting: structured findings
- Tools and Frameworks
  - Garak (NVIDIA)
  - Microsoft Counterfit
  - Adversarial Robustness Toolkit (IBM)
  - PyRIT (Microsoft)
- Organizing a Red Team Program
  - Team composition
  - Engagement models (internal, external, bug bounty)
- Multi-Turn Attack Scenarios

### 18.4 Enterprise RAG (`04_Enterprise_RAG.md`)
- Why enterprise RAG differs from a prototype
- Multi-Tenancy architectures
  - Shared vs Isolated Infrastructure
  - Namespace isolation
  - Encryption-based isolation
  - Cross-Tenant Risks
- Access Control for RAG
  - Document-Level ACL
  - Chunk-Level Considerations
  - Query-Time Enforcement
  - Integration with Enterprise Identity (SSO, LDAP, RBAC)
- Data Governance
  - Data Classification
  - Data Lineage
  - Retention Policies
  - PII Handling
- Audit and Compliance
  - Comprehensive Audit Logging
  - GDPR, HIPAA, SOC 2 reporting
  - Anomaly Detection
- Security Architecture (Defense in Depth)

### 18.5 Governance Frameworks (`05_Governance_Frameworks.md`)
- From technology to organizational maturity
- Organizational models of AI Governance
  - Centralized model (AI CoE)
  - Federated model
  - Hybrid (Hub and Spoke)
- Key Framework components
  - AI Ethics Board: composition, mandate, operating model
  - AI Policy Framework (Use, Development, Operations, Risk)
  - Model Cards and Documentation
  - Model Lifecycle Management
- Approval Workflows
  - Risk-Based Tiering (Tier 1-4)
  - Stage-Gate Process
  - Exception Handling
- Model Versioning and Change Management
  - Semantic versioning for ML
  - Change Request/Approval/Implementation
  - A/B Testing Governance
- Metrics and Reporting
  - Process, Compliance, Risk, Culture metrics
  - Executive Dashboard

### 18.6 Alignment Research (`06_Alignment_Research.md`)
- Introduction: why alignment is the central AI problem
- Connection between Interpretability and Safety
  - Circuits and Features in transformers
  - Superposition Hypothesis
  - Sparse Autoencoders (SAE) for interpretation
  - Activation Patching and Causal Tracing
  - Logit Lens and Tuned Lens
- Scalable Oversight Problem
  - Why human oversight doesn't scale
  - Current approaches and their limitations
- Debate as an Alignment Approach
  - Adversarial verification
  - Honest vs Deceptive agents
  - Practical experiments
- Recursive Reward Modeling (RRM)
  - Bootstrapping trust
  - Iterated Distillation and Amplification
  - Connection to Process Reward Models
- Constitutional AI in depth
  - SL-CAI: Self-supervised Constitutional Critique
  - RLAIF: RL from AI Feedback
  - Constitution structure and principles
- Current Open Problems
  - Deceptive alignment
  - Mesa-optimization
  - Reward hacking and Goodhart's Law
  - Corrigibility
  - Value learning

### 18.7 Enterprise AI Adoption (`07_Enterprise_AI_Adoption.md`)
- **The reality of AI transformation**
  - 80%+ of companies use AI
  - Only 25% of AI initiatives achieve expected ROI
  - 40% of agentic AI projects will fail by 2027 (Gartner)
- **Key Enterprise AI challenges**
  - Governance & Security (75% of tech leaders, only 6% have an AI security strategy)
  - Compliance: EU AI Act, HIPAA, risk categories
  - ROI Measurement: evaluation framework (Cost Savings, Productivity, Quality, Strategic Value)
  - Data Quality: 62% — the main obstacle
  - Talent Gap: 73% use AI, only 29% have advanced literacy
- **Governance-first approach**
  - AI Governance Maturity Model (Ad-hoc → Aware → Managed → Optimized)
  - Choosing a pilot project
  - Scaling: from pilot to production
- **Organizational patterns**
  - Center of Excellence vs Distributed
  - AI Product Manager as a new role
- **CTO's Checklist**

---

## 19. Practical Projects

### 19.1 RAG Chatbot (`01_RAG_Chatbot.md`)
- Introduction: from theory to a production-ready system
- Phase 1: Document Ingestion Pipeline
  - Multi-format parsing (PDF, DOCX, HTML)
  - Recursive chunking with overlap
  - Metadata extraction
- Phase 2: Embedding and indexing
  - Batch embedding generation
  - Choosing a vector store
  - Indexing with metadata
- Phase 3: Retrieval system
  - Hybrid retrieval (dense + sparse)
  - Reranking (cross-encoder)
  - Query expansion
- Phase 4: Generation Pipeline
  - Prompt engineering for RAG
  - Streaming responses
  - Citation extraction
- Phase 5: Observability
  - Tracing all components
  - Quality metrics
  - Cost tracking

### 19.2 Multi-Agent System (`02_Multi_Agent_System.md`)
- Introduction: collective intelligence in action
- Code Review system architecture
- Specialized agents
  - Security Reviewer: vulnerabilities, injection, secrets
  - Performance Reviewer: complexity, memory, patterns
  - Style Reviewer: conventions, formatting, naming
  - Architecture Reviewer: SOLID, patterns, coupling
- Orchestrator
  - Parallel execution
  - Result aggregation
  - Conflict resolution
- GitHub integration
  - Webhook processing
  - PR comments
  - Check runs
- Monitoring and metrics

### 19.3 MCP Server (`03_MCP_Server.md`)
- Introduction: standardized data access
- Designing a Knowledge Base MCP Server
- MCP Components
  - Resources: documents, articles, categories
  - Tools: search, create, update, delete
  - Prompts: templates for common tasks
- Transport layer
  - stdio for local integration
  - Streamable HTTP for remote access
- Security
  - Authentication
  - Authorization
  - Input validation
- Testing MCP servers
- Claude Desktop integration

### 19.4 Fine-Tuning Pipeline (`04_Fine_Tuning_Pipeline.md`)
- Introduction: from data to a deployed model
- Phase 1: Data Collection
  - Data sources (support tickets, knowledge base)
  - Synthetic data generation
  - Quality filtering
- Phase 2: Data Processing
  - PII detection and masking
  - Format conversion (JSONL)
  - Train/validation split
- Phase 3: Fine-Tuning
  - LoRA configuration (rank, alpha, target modules)
  - Hyperparameter selection
  - Training monitoring
- Phase 4: Evaluation
  - Automatic metrics
  - LLM-as-Judge
  - Human evaluation
  - A/B testing
- Phase 5: Deployment
  - Model versioning
  - Gradual rollout
  - Monitoring in production

### 19.5 Production Agent (`05_Production_Agent.md`)
- Introduction: from prototype to production
- Phase 1: Architecture and infrastructure
  - Stateless design
  - Container orchestration (Kubernetes)
  - Configuration and secrets management
- Phase 2: API and request handling
  - RESTful API design
  - Rate limiting
  - Async processing with queues
- Phase 3: Reliability
  - Circuit breaker pattern
  - Retry with exponential backoff
  - Idempotency
  - Health checks
- Phase 4: Observability
  - Three pillars: metrics, logs, traces
  - AI-specific metrics
  - Alerting based on SLOs
- Phase 5: Security
  - Authentication/Authorization
  - Prompt injection protection
  - Audit logging
- Phase 6: Continuous Improvement
  - A/B testing
  - Feedback loops
  - Canary deployments

---

## 20. Architecture Research

### 20.1 Mixture of Experts (`01_Mixture_of_Experts.md`)
- Why sparse models
  - Conditional computation
  - Scaling without linear compute growth
- MoE history: from Switch Transformer to Mixtral
  - GShard, Switch Transformer
  - GLaM, ST-MoE
  - Mixtral 8x7B, DeepSeek-MoE
- Router Architecture
  - Linear router (Top-K)
  - Expert Choice routing
  - Soft MoE (fully differentiable)
- Load Balancing
  - Auxiliary losses
  - Capacity factor
  - Z-loss for stability
- Expert Parallelism
  - Distributing experts across GPUs
  - All-to-All communication
- Sparse vs Dense Trade-offs
  - Training efficiency
  - Inference challenges
  - Expert specialization analysis

### 20.2 State Space Models (`02_State_Space_Models.md`)
- Attention limitations
  - Quadratic complexity O(L²)
  - Memory requirements scaling
- State Space Models intuition
  - Continuous-time dynamical systems
  - Discretization for sequences
- S4: Structured State Spaces
  - HiPPO initialization
  - Diagonal structure
- Mamba: Selective State Spaces
  - Input-dependent selection
  - Hardware-aware algorithm
  - Linear time complexity
- Linear Attention Alternatives
  - Approximating softmax attention
  - RetNet: Retentive Networks
- Hybrid Architectures
  - Jamba: SSM + Attention + MoE
  - When SSM vs Transformer
- Current SSM Limitations
  - In-context learning challenges
  - Retrieval tasks

### 20.3 Multimodal Architectures (`03_Multimodal_Architectures.md`)
- Vision Encoders
  - ViT: Vision Transformer
  - CLIP: Contrastive Language-Image Pretraining
  - SigLIP: Sigmoid loss alternative
- Contrastive Learning
  - InfoNCE loss
  - Temperature scaling
  - Batch size requirements
- Cross-Modal Fusion
  - Early vs Late fusion
  - Cross-attention mechanisms
  - Perceiver Resampler (Flamingo)
- Visual Instruction Tuning
  - LLaVA architecture
  - Projection layers
  - Two-stage training
- Multimodal Tokenization
  - Discrete visual tokens
  - Unified vocabularies
- Video Understanding
  - Temporal modeling challenges
  - Frame sampling strategies
  - Efficiency considerations

---

## 21. Interview Preparation

### 21.1 ML System Design (`01_ML_System_Design.md`)
- RESHAPE Framework for system design
  - Requirements, Estimation, System, High-level, API, Productionization, Evaluation
- Case Study 1: Enterprise RAG System
  - Requirements gathering
  - Capacity estimation
  - Architecture components
- Case Study 2: LLM Serving Platform
  - Multi-tenant architecture
  - Model routing
- Case Study 3: Content Moderation System
- Case Study 4: Real-Time Recommendations
- Typical questions and answers
- Red Flags in interviews

### 21.2 Coding Exercises (`02_Coding_Exercises.md`)
- Category 1: Implementations from scratch
  - Self-Attention (NumPy)
  - Multi-Head Attention
  - LSTM Cell
  - K-Means Clustering
  - Gradient Descent
  - BPE Tokenizer
- Category 2: Debugging and optimization
  - Distributed training issues
  - Memory optimization
- Category 3: Practical tasks
  - RAG pipeline
  - Prompt engineering
- Tips for success

### 21.3 Papers Reading Guide (`03_Papers_Reading_Guide.md`)
- Three-Pass Approach to reading papers
- Essential papers (with discussion points)
  - Attention Is All You Need
  - BERT / GPT-3
  - InstructGPT / RLHF
  - LoRA
  - Flash Attention
  - Chain-of-Thought / ReAct
  - RAG
- Cutting-Edge Papers
  - DPO, o1/o3, Mamba, DeepSeek V3
- How to discuss papers in interviews
- Paper note template

### 21.4 Behavioral for Staff+ (`04_Behavioral_Staff_Plus.md`)
- Staff+ level expectations
  - Technical Leadership
  - Cross-team Impact
  - Mentorship
  - Strategic Thinking
- STAR Framework for ML
  - Situation, Task, Action, Result
- Amazon Leadership Principles for AI/ML
- Typical questions and example answers
- Red Flags and Green Flags
- How to structure your answers

### 21.5 AI Safety Interview (`05_AI_Safety_Interview.md`)
- Anthropic/OpenAI specifics
- Alignment and its complexity
  - Inner vs Outer Alignment
  - Mesa-optimization
- RLHF Deep Dive
  - Reward hacking
  - Distributional shift
- Constitutional AI
  - SL-CAI and RLAIF
- Interpretability for interviews
  - Sparse Autoencoders
  - Activation patching
- Red Teaming
  - Methodology
  - Types of attacks
- Governance questions

---

## Appendices

### Code Examples
Each chapter contains a "Practical Code Examples" section with Java implementations:
- LangChain4j
- Spring AI
- Semantic Kernel

---

## Course Statistics

| Section | Files |
|---------|-------|
| 01. LLM Fundamentals | 10 |
| 02. Prompt Engineering | 5 |
| 03. AI Agents Core | 10 |
| 04. Multi-Agent Systems | 5 |
| 05. MCP Protocol | 5 |
| 06. RAG | 6 |
| 07. Frameworks | 6 |
| 08. Structured Outputs | 3 |
| 09. Conversational AI | 3 |
| 10. Fine-Tuning | 10 |
| 11. Evaluation & Testing | 5 |
| 12. Observability | 4 |
| 13. Deployment | 3 |
| 14. Security & Safety | 5 |
| 15. GPU Architecture | 5 |
| 16. Distributed Training | 5 |
| 17. Production Inference | 7 |
| 18. AI Governance | 7 |
| 19. Practical Projects | 5 |
| 20. Architecture Research | 3 |
| 21. Interview Preparation | 5 |
| **Total** | **117 files** |

---

*Generated based on actual course file contents.*

---

## Navigation

**Module:** Course Contents
**Home:** [[00_Home|Course Home Page]]
