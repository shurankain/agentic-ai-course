# Code Generation Agents: From Autocomplete to Autonomous Development

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[06_Computer_Use_Agents|Computer Use Agents]]
**Next:** [[08_Process_Reward_Models|Process Reward Models]]

---

## Introduction: Evolution of AI in Software Development

The history of AI assistants in programming began with simple autocomplete — suggesting the next few characters based on code statistics. Copilot raised the bar by generating entire functions. But the real revolution happened when AI systems started not just writing code, but fully solving tasks: understanding requirements, exploring the codebase, making changes across multiple files, running tests, and iterating until success.

This chapter examines the spectrum from interactive assistants (Cursor, Copilot) to fully autonomous agents (Devin, OpenHands). Understanding the capabilities and limitations of each approach is critical for an AI architect making tooling decisions for a development team.

---

## Theoretical Foundations: Program Synthesis

Before examining specific tools, it is important to understand the theoretical foundation. Code generation agents solve the problem of **program synthesis** — automatically constructing a program from a specification.

### From Specification to Program

Program synthesis has existed since the 1960s, when researchers began formalizing the idea of "program from description." The classical formulation:

**Given:** Specification S (what the program should do)
**Find:** Program P such that P ⊨ S (P satisfies S)

The specification can be expressed in different ways:

| Specification Type | Example | Verification Complexity |
|-----------------|--------|----------------------|
| **Formal logic** | ∀x. sorted(P(x)) ∧ permutation(x, P(x)) | Automatic (proof assistants) |
| **Input-output examples** | f([3,1,2]) = [1,2,3], f([5,4]) = [4,5] | Automatic (tests) |
| **Natural language** | "Sort the list in ascending order" | Requires interpretation |
| **Types + contracts** | sort: List[T] -> List[T] with pre/post | Partially automatic |

LLM-based code generation is program synthesis with natural language specification. This is the most complex form because:
1. Natural language is ambiguous
2. There is no formal way to verify correctness
3. The specification is often incomplete

### Inductive Program Synthesis

When the specification is given as examples (input-output pairs), the task is called **inductive program synthesis** or **programming by example** (PBE). For instance, from the examples "hello world" → "Hello World", "foo bar" → "Foo Bar", the system must derive a capitalize_words function.

Classical approaches to PBE:
- **Enumerative search**: iterating through programs in order of increasing complexity
- **Constraint-based**: formulating as a SAT/SMT problem
- **Version space algebra**: maintaining a set of consistent programs

LLMs revolutionized this process by replacing combinatorial search with neural generation. But fundamental problems remain:
- **Underfitting**: the program works on examples but does not generalize
- **Overfitting**: the program is hardcoded for specific examples
- **Ambiguity**: examples admit multiple interpretations

### Connection to Compiler Theory

LLM code generation has an interesting connection to classical compilers. A traditional compiler works through formal phases: Source code → AST → IR → Optimized IR → Machine code. LLM code generation works differently: Natural language → Neural "understanding" → Code tokens → AST.

The key difference: a compiler operates on formal languages with precise semantics; an LLM operates on fuzzy representations. This explains why LLMs can generate syntactically incorrect code — they lack a built-in "parser" for the target language.

Modern approaches (Structured Outputs, constrained decoding) attempt to add formal guarantees on top of neural generation.

---

## IDE-Integrated Tools: Augmenting the Developer

### GitHub Copilot: The Pioneer

Copilot (launched in 2021) became the first mass-market AI coding assistant:

**Capabilities:**
- Line-level and function-level autocomplete
- Code generation from comments
- Multi-language support
- Integration with VS Code, JetBrains, Neovim

**Architecture:** Works through a simple pipeline — the user writes code/comments, the system gathers context (current file, open tabs), sends it to a cloud-based model (Codex/GPT), receives suggestions, and displays them as ghost text in the editor.

**Statistics (2025):**
- 55% of code is written with Copilot assistance (per GitHub data)
- 1.8M+ paying subscribers
- Available in Individual, Business, and Enterprise editions

**Copilot Agent Mode (2025):**
GitHub Copilot added an agent mode in VS Code that goes beyond inline suggestions. In agent mode, Copilot can autonomously plan changes across multiple files, run terminal commands, execute tests, and iterate on failures. This brings Copilot closer to Cursor and Claude Code in capability, though it remains more tightly integrated with the VS Code/GitHub ecosystem.

**Limitations (standard mode):**
- Does not understand full project context (agent mode partially addresses this)
- Hallucinates non-existent APIs
- A 2022 Stanford study found earlier Copilot versions generated insecure code in ~40% of security-relevant scenarios; later versions have improved through better filters and model upgrades

### Cursor: The Next Generation

Cursor went further by building an IDE around AI:

**Key innovations:**
1. **Codebase-aware RAG** — indexes the entire project
2. **Multi-file editing** — Cmd+K edits multiple files
3. **Composer** — natural language → code with full context
4. **Terminal integration** — AI can see command output

**Cursor Architecture:** Built as a multi-layered system where the base Monaco editor works together with the AI Sidebar (providing Chat and Cmd+K functionality), relying on a local index with embeddings of the entire project. The system can use different LLM APIs (Claude, GPT-4, Custom models) depending on the task and user settings.

**Cmd+K workflow:** Select code, press Cmd+K, describe the change in natural language, AI generates a diff, you accept/reject/edit.

**Composer mode:** For complex multi-file tasks — for example, "Add authentication to all API endpoints". Cursor analyzes the codebase, creates a plan ("I'll modify these 5 files..."), generates changes for each file, and the user reviews and applies.

### Windsurf (formerly Codeium), Tabnine, Amazon Q

**Windsurf** (rebranded from Codeium in late 2024):
- Full agentic IDE with Cascade engine
- Free tier available
- Multi-model support (Claude, GPT, custom)
- See the Autonomous Coding Agents section for details

**Tabnine:**
- Privacy-first (option for fully local operation)
- Personalization to team coding style
- Enterprise self-hosted

**Amazon Q Developer:**
- AWS integration
- Security scanning
- Transformations (Java upgrades, etc.)

### IDE Tools Comparison

| Feature | Copilot | Cursor | Windsurf | Tabnine |
|---------|---------|--------|----------|---------|
| Multi-file edit | Agent mode | ✓ | ✓ (Cascade) | ✗ |
| Codebase RAG | Limited | ✓ | ✓ | ✓ |
| Agent mode | ✓ (2025) | ✓ (Composer) | ✓ (Cascade) | ✗ |
| Terminal integration | Agent mode | ✓ | ✓ | ✗ |
| Custom models | Limited | ✓ | ✓ | Limited |
| Price/month | $10-39 | $20 | Free-$15 | $12 |

---

## Autonomous Coding Agents

### Devin (Cognition): The First AI Software Engineer

Cognition introduced Devin (March 2024) as the "first AI software engineer," marking the beginning of the autonomous coding agent category:

**Capabilities:**
- Understands tasks from descriptions
- Explores the codebase independently
- Writes and edits code
- Runs tests, debugging
- Deploys changes
- Communicates for clarifications

**Devin Architecture:** A hierarchical system with a central Task Manager that decomposes goals and tracks progress. Below it operate three specialized agents — Planner (subtask planning), Coder (code writing), and Tester (correctness verification). All agents interact through a shared Execution Environment providing Browser, Terminal, Editor, and Git tools for executing real actions.

**SWE-bench Results:**
- Devin at launch (March 2024): 13.86% — a breakthrough at announcement time
- By late 2025, Claude-based agents surpassed 72% on SWE-bench Verified

**Context:** Devin pioneered the category but faced scrutiny over marketing claims. The initial 13.86% SWE-bench score, impressive at launch, was quickly surpassed by open-source alternatives and model-native coding agents. Devin's value proposition shifted toward being an "AI teammate" that works asynchronously on tasks via Slack integration, rather than competing purely on benchmark scores.

**Pricing:**
- $500/month for teams
- Metered usage for compute

### OpenHands (formerly OpenDevin)

Open-source alternative to Devin:

**Features:**
- Fully open source
- Extensible architecture
- Support for different LLMs (Claude, GPT, Llama)
- Docker-based sandboxing

**Architecture:** Consists of three main components — AgentController (planning and decision-making), DockerRuntime (isolated execution), AgentMemory (tracking action history for replanning). The agent works in an iterative cycle: creates a plan, executes actions in a Docker sandbox, saves results to memory, and replans based on accumulated experience when necessary.

**Self-hosting:** Available via Docker image `ghcr.io/all-hands-ai/openhands:latest`, runs on port 3000. Fully user-controlled, data does not leave the infrastructure.

### Claude Code (Anthropic)

Anthropic's CLI-based coding agent, one of the most capable autonomous coding tools available:

**Capabilities:**
- Full agentic workflow in the terminal with autonomous multi-step task execution
- Deep codebase understanding through file reading, grep, glob, and semantic search
- Multi-file editing with surgical precision (edit specific lines, not whole files)
- Runs commands, tests, and build tools with output analysis
- Git integration (commits, PRs, branch management)
- MCP server support for extensibility (connect to databases, APIs, custom tools)
- Sub-agent architecture (spawns specialized agents for parallel subtasks)
- Extended thinking for complex reasoning before acting
- Hooks system for custom automation (pre/post tool execution)
- Project memory (CLAUDE.md files for persistent project context)

**Architecture:** Claude Code operates as an agentic loop: the user provides a task, the agent reads relevant files, forms a plan, executes edits and commands, verifies results (runs tests, checks compilation), and iterates until success. Unlike IDE-integrated tools, it has full system access — it can install dependencies, run Docker containers, interact with APIs, and manage infrastructure.

**Key differentiators:**
- **No IDE required** — works in any terminal, making it accessible for server-side development, CI/CD integration, and headless environments
- **MCP extensibility** — connects to any MCP server, giving it access to databases, APIs, and custom tools beyond file/terminal operations
- **Spec-driven development** — excels at implementing features from detailed specifications, making it effective for the "write spec → agent implements" workflow pattern

**SWE-bench performance:** Claude Code achieved 72.7% on SWE-bench Verified with Claude Sonnet 4 (mid-2025). With the latest Claude models, scores have continued to improve, with frontier agents now exceeding 80%.

### OpenAI Codex (2025)

OpenAI's cloud-based autonomous coding agent, released mid-2025:

**Capabilities:**
- Runs in a cloud sandbox environment (not local)
- Works on GitHub repositories — reads code, writes implementations, creates PRs
- Parallel task execution — can work on multiple tasks simultaneously
- Powered by codex-1 (specialized model derived from o3)
- Integrated into ChatGPT Pro and Team plans

**Architecture:** Codex operates asynchronously in cloud sandboxes. The user assigns a task (e.g., "Implement the login endpoint based on this spec"), Codex clones the repo, works autonomously, and produces a PR with changes. Multiple tasks can run in parallel. The sandbox provides isolation but limits interaction — no real-time collaboration like Claude Code.

**Trade-offs:** Excellent for batch-style development (assign multiple tasks, review PRs later), but less suitable for interactive, iterative development where real-time feedback is important.

### Windsurf (formerly Codeium)

Codeium rebranded to Windsurf in late 2024, pivoting from autocomplete to an agentic IDE:

**Key features:**
- **Cascade** — an agentic coding engine that can reason across multiple files, run terminal commands, and iterate on solutions
- **Flows** — persistent context that tracks what you're working on across sessions
- Multi-model support (Claude, GPT, custom models)
- Free tier with generous limits

**Positioning:** Competes directly with Cursor as an AI-native IDE, with a focus on the agentic "flow" experience rather than individual completions.

---

## Benchmarks: Measuring Progress

### SWE-bench: Real GitHub Issues

**What it is:**
- 2,294 real issues from popular Python repos
- Requires: understanding the issue, finding the code, fixing it, passing tests
- Gold standard for coding agents

#### SWE-bench Evaluation Methodology

SWE-bench uses a rigorous methodology that makes it a valuable benchmark:

**1. Data Collection:** From popular Python projects (Django, Flask, Requests, Scikit-learn, Matplotlib, Pandas), issues are selected that have associated PRs (ground truth fix), pass tests (fix is correct), and have a failing test (reproducibility). Final dataset: 2,294 tasks.

**2. Task Structure:** Each task includes an issue description (text from GitHub issue), repository snapshot (code before the fix), failing test (fails before fix, passes after), and ground truth patch (actual PR from maintainers).

**3. Evaluation Process:** The agent receives issue + repo snapshot, generates a patch (code changes), the patch is applied to the snapshot, repository tests are run, evaluation: do the failing tests pass?

The key point: evaluation is **functional**, not textual. The agent does not need to reproduce the exact human patch — it is sufficient for the tests to pass. This is fairer than comparison with a reference solution.

**4. Factors that make SWE-bench challenging:** Multi-file changes (often need to modify several files), understanding the codebase (finding relevant code among thousands of files), domain knowledge (understanding library specifics), test isolation (must not break other tests).

**SWE-bench Verified (2024):**
Curated subset with verified solutions (500 tasks, manually verified):

| Model/Agent | Score |
|-------------|-------|
| Human (estimated) | ~75-90% |
| Frontier agents (early 2026) | ~80%+ |
| Claude Opus 4 (2025) | 72.0% |
| Claude Sonnet 4 (2025) | 72.7% |
| OpenAI o1 | 48.9% |
| GPT-4o | 33.2% |
| Devin (launch, 2024) | 13.86% |

*Note:* Scores continue to improve rapidly. By early 2026, the best agents exceed 80%, approaching human performance. SWE-bench Pro (a harder variant) was introduced where top agents score ~23%.

### SWE-bench Pro (2025)

A more realistic version:
- Harder than Verified
- Less "data contamination"
- Multi-file changes required
- Integration testing

### HumanEval: Basic Coding

**What it is:**
- 164 Python programming problems
- Function-level generation
- Automated test verification

**Results (2025):**
- GPT-4o: 90.2%
- Claude Sonnet 4: 93.7%
- o3: 96.7%

**Limitations:**
- Tasks are too simple — frontier models have nearly saturated this benchmark
- Does not reflect real-world work
- SWE-bench has largely replaced HumanEval as the meaningful coding benchmark

### MBPP (Mostly Basic Python Problems)

- 974 programming problems
- Simpler than HumanEval
- Greater coverage

---

## Coding Agent Architecture

### Agentic Coding Loop: The Autonomous Development Cycle

Before examining the components, it is important to understand the fundamental cycle of a coding agent. This is not simply "generate code" — it is an iterative process with feedback, consisting of four sequential phases: **PLAN → EDIT → TEST → DEBUG**, where the last phase creates a feedback loop to the beginning of the cycle, allowing the agent to replan and fix errors until success.

**PLAN Phase:**
1. Task analysis (issue, feature request)
2. Codebase exploration (finding relevant files)
3. Context understanding (how existing code works)
4. Decomposition into subtasks
5. Identifying files to modify

**EDIT Phase:**
1. Generating changes (diffs or full files)
2. Syntax checking
3. Applying changes
4. Optionally: refactoring or formatting

**TEST Phase:**
1. Running existing tests (regression check)
2. Running specific tests for modified functionality
3. Optionally: generating new tests
4. Collecting results (pass/fail, coverage, errors)

**DEBUG Phase:**
1. Analyzing failing tests
2. Interpreting error messages and stack traces
3. Localizing the error (which file, which line)
4. Forming a hypothesis about the cause
5. Transitioning to a new PLAN or EDIT iteration

### Critical Decisions in the Loop

**When to stop?**
- All tests pass → success
- Iteration limit reached → partial success or failure
- Stuck in a loop (repeating errors) → human assistance needed

**How to avoid infinite loops?**
- Tracking attempts and error patterns
- Detecting stalls (same errors for 3+ iterations)
- Exponential backoff with different strategies

**When to request human assistance?**
- Unclear requirements
- Multiple interpretations
- Errors beyond the scope of code (infrastructure, permissions)

### Core Components

A Coding Agent consists of three main top-level modules: **Context Manager** (managing the codebase and extracting relevant context), **Planning Module** (decomposing tasks into executable steps), and **Execution Engine** (performing actions). All modules operate through a unified **Tool Execution Layer** providing five categories of tools: File IO (reading/writing files), Search (code search), Terminal (command execution), Git (version control), and Test Runner (running tests).

### Context Management

**Goal:** Managing the codebase and extracting relevant context for the task.

**Key functions:**
- **Indexing:** Building the repository structure, creating embeddings for semantic search, building a dependency graph to understand relationships between modules.
- **Search:** Semantic search for relevant files based on the task description, using vector embeddings or keyword-based methods.
- **Expansion:** Automatically adding dependent files (imports, called functions, base classes) for context completeness.
- **Truncation:** Managing the token limit — selecting the most relevant code sections when the context window is exceeded.

**Indexing strategies:**
- **Full-file indexing:** The entire file as a unit (simple but wasteful on tokens)
- **AST-based chunking:** Splitting by functions/classes through parsing (more precise, requires a parser for each language)
- **Sliding window:** Fixed windows of N lines with overlap (universal but may cut logical blocks)
- **Semantic chunking:** Using embeddings to determine boundaries of semantic blocks (complex but most intelligent)

### Planning Strategy

**Goal:** Decomposing a high-level task into specific executable steps.

**Planning process:**
1. **Task analysis:** The LLM parses requirements, identifies key constraints, and determines change scope.
2. **Codebase exploration:** Finding relevant files, understanding existing architecture, identifying usage patterns.
3. **Decomposition:** Breaking down into steps specifying the action (read/write/test), target file, and specific changes.
4. **Verification:** For each step, success criteria are defined (tests pass, code compiles, etc.).

**Plan types:**
- **Sequential:** Linear sequence of steps (simple but inflexible)
- **Conditional:** Branching based on results (if tests fail → debug, else → continue)
- **Hierarchical:** Nested sub-tasks with recursive decomposition
- **Iterative:** Loops for repeating operations (apply fix to each file)

**Replanning triggers:**
- Compilation/test errors (approach adjustment needed)
- Discovery of unexpected dependencies (scope expansion)
- Looping (repeating errors require a strategy change)

### Tool Integration

**Goal:** Providing the agent with capabilities for interacting with the codebase and environment.

**Core tools:**

**File Operations:**
- `read_file(path)` — reading with encoding handling, large files
- `write_file(path, content)` — writing with atomic operations for safety
- `patch_file(path, diff)` — applying a diff instead of full rewrite (more efficient for large files)

**Code Search:**
- `search_code(query, pattern)` — regex or semantic search
- `find_definitions(symbol)` — finding function/class definitions
- `find_references(symbol)` — finding all usages

**Execution:**
- `run_command(cmd, timeout)` — executing shell commands in a sandbox
- `run_tests(path, args)` — running the test suite with result parsing
- `compile_code(files)` — syntax and type checking

**Version Control:**
- `git_diff()` — viewing changes
- `git_status()` — working directory state
- `git_log(file)` — file change history for context

**Sandboxing is critical:** Tools must execute in an isolated environment (Docker, VM, containerized environment) to prevent malicious actions.

---

## Integration into the Development Workflow

### Human-AI Collaboration Patterns

**Pattern 1: AI-First Draft** — The human describes the task, AI generates an initial implementation, the human reviews and provides feedback, AI refines based on feedback, the human gives final approval and merges. Suitable for new features.

**Pattern 2: AI as Pair Programmer** — The human writes code, AI suggests improvements in real-time, the human accepts/rejects suggestions, AI explains on request. Increases productivity of experienced developers.

**Pattern 3: AI for Tedious Tasks** — The human defines a repetitive task (bulk tests, migrations, boilerplate), AI generates code en masse, the human spot-checks quality, AI fixes identified issues. Saves time on routine work.

**Pattern 4: Autonomous with Checkpoints** — AI works autonomously on a defined task, reaches a checkpoint → requests review, the human checks progress and provides direction, AI continues to the next checkpoint, the human gives final approval. Maximum autonomy with quality control.


### Security Considerations

**1. Code Injection Risks:** The agent may generate unsafe code such as `os.system(user_input)` — generated code must be validated.

**2. Secret Exposure:** The agent may accidentally commit secrets — a pre-commit hook with secret scanning is needed.

**3. Dependency Confusion:** The agent may add incorrect dependencies — package origin verification is needed.

---

## Connection to Formal Verification

An interesting question: can we not just test generated code, but **prove** its correctness? This is the intersection of LLM code generation and formal methods.

### Spectrum of Correctness Guarantees

A spectrum of code correctness guarantees exists — from minimal to maximal:
1. **Syntax check** — the code parses
2. **Tests pass** — works on examples
3. **Types check** — types are consistent
4. **Contracts verify** — pre/post conditions are satisfied
5. **Proof check** — correctness is mathematically proven

Most coding agents stop at the "tests pass" level. But research shows prospects for stronger guarantees.

### LLM + Proof Assistants

**AlphaProof** (DeepMind, 2024) demonstrated that LLMs can generate mathematical proofs in Lean. A similar approach is applicable to code: the LLM generates code + proof sketch, the proof assistant (Lean, Coq) verifies, if the proof fails — feedback to the LLM, and the LLM iterates until success.

### Practical Approximations

Full formal verification is expensive. Practical intermediate approaches:

**1. Design by Contract (DbC):** The agent generates not only code but also contracts (@require, @ensure) that check pre/post conditions.

**2. Property-Based Testing:** Instead of specific tests, the agent generates properties that must hold for all inputs.

**3. Gradual Verification:** A combination of runtime checks with static analysis. What can be checked statically is verified at compile time; the rest is checked at runtime.

### Future: Verified Code Generation

A research direction — training models to generate **inherently verifiable** code:
- Code in a language subset with decidable verification
- Automatic generation of loop invariants
- Integration with SMT solvers (Z3) for constraint checking

This is still in the future, but understanding the connection between neural generation and formal methods is critical for an AI architect.

---

## Spec-Driven Development: A New Workflow Pattern

The rise of capable coding agents has enabled a new development workflow: **spec-driven development**, where the human writes a detailed specification and the agent implements it.

### The Pattern

1. **Human writes spec** — a detailed description of what to build: requirements, API contracts, data models, edge cases, acceptance criteria
2. **Agent implements** — the coding agent reads the spec and autonomously writes the code, tests, and documentation
3. **Human reviews** — reviews the implementation against the spec, provides feedback
4. **Agent iterates** — refines based on feedback until the spec is satisfied

### Why This Works Now

Coding agents like Claude Code and Codex excel when given clear, detailed specifications. The spec provides unambiguous context that eliminates the interpretation problems inherent in natural language. The agent's ability to read the entire codebase, run tests, and iterate means it can handle implementation details autonomously.

### Practical Considerations

**Good specs include:** API contracts (request/response schemas), data models with field descriptions, error handling requirements, edge cases enumerated explicitly, acceptance criteria that map to testable conditions.

**The spec is the new bottleneck.** Writing a good spec takes significant thought — arguably the hardest part of software engineering. But the spec is also the most valuable artifact: it serves as documentation, test criteria, and implementation guide simultaneously.

**Tools supporting this pattern:** Claude Code (reads spec files, implements across codebase), Codex (takes specs as task descriptions, produces PRs), Cursor Composer (reads spec in context, generates multi-file changes).

---

## Key Takeaways

1. **IDE tools (Copilot, Cursor, Windsurf) augment the developer** — agent modes now enable multi-file autonomous editing alongside traditional autocomplete

2. **Autonomous agents (Claude Code, Codex, OpenHands) solve tasks end-to-end** — with 80%+ on SWE-bench Verified, they are at or near human-level on real GitHub issues

3. **SWE-bench is the gold standard** for measurement; top agents exceed 80% (early 2026), approaching human-level (~75-90%)

4. **Context management is critical** — the agent must understand the codebase, not just the current file

5. **Spec-driven development** is emerging as a powerful workflow: human writes spec, agent implements

6. **Security concerns remain:** code injection, secrets, dependencies

7. **Trend: convergence toward agent-native tools** — Cursor, Windsurf, and Copilot agent mode all moving toward autonomous multi-file editing

8. **CLI agents (Claude Code) and cloud agents (Codex)** represent two complementary paradigms: interactive vs. asynchronous

---

## Practical Code Examples

This section demonstrates key coding agent concepts through one complete example and textual descriptions of additional approaches.

### Conceptual Example: Basic Coding Agent with an Iterative Loop

A minimalist coding agent demonstrates the core PLAN-EDIT-TEST-DEBUG loop through the following architecture:

**CodeTask Data Structure:** Encapsulates a task for the agent, including description (textual description of requirements), files_to_modify (list of files to change), tests_to_run (list of tests for verification), and success_criteria (criteria for successful completion).

**SimpleCodingAgent Class:** The main agent initialized with llm_client (client for the LLM API), repo_path (path to the repository), and max_iterations (maximum number of iterations to prevent infinite loops).

**Main solve_task() Method:** Implements the full task-solving cycle through four phases:
1. **PLAN phase** — gathers context via _gather_context() (reading all relevant files) and creates a plan via _create_plan() (LLM generates JSON with a sequence of steps)
2. **Iterative execution** — loop up to max_iterations, where _execute_step() performs the current plan step (reading/writing files, running tests)
3. **TEST phase** — _run_tests() runs pytest and returns results (passed/failed with error messages)
4. **DEBUG phase** — on failure, _replan() calls the LLM with error context to create a new plan

**Helper methods:**
- _gather_context() — reads all files from files_to_modify, forms a unified context for the LLM
- _create_plan() — constructs a prompt with task description + context, receives a JSON plan with an array of steps from the LLM
- _execute_step() — handles "write" actions (code generation and writing) and "test" actions (running tests)
- _generate_code() — delegates code generation to the LLM with specific instructions for the step
- _write_file() — atomic file writing with parent directory creation
- _replan() — on failure, creates a new plan accounting for accumulated errors
- _verify_solution() — final verification of all tests after changes are complete

**Key implementation concepts:**
- Iterative execution with bounded loops (max_iterations) to prevent infinite cycles
- Separation of concerns: planning, execution, testing, and verification are separated
- Error-driven replanning: errors become input for creating an improved plan
- Context awareness: the agent works with the full context of relevant files
- Status model: each step returns a status (continue/complete/error) for flow control

### Additional Code Generation Strategies (Textual Description)

Beyond the basic iterative agent, several specialized approaches to code generation exist, each with its own advantages:

#### 1. Codebase-Aware RAG Approach

**Concept:** Instead of working with limited context, the agent indexes the entire repository through embeddings and uses semantic search to find relevant code fragments.

**Architectural components:**
- **Indexing:** The repository is split into semantic chunks (functions, classes, modules) using an AST parser (e.g., tree-sitter). Each chunk receives an embedding via a model (sentence-transformers, code-specific models like CodeBERT).
- **Storage:** Embeddings are stored in a vector database (FAISS, ChromaDB, Pinecone) along with metadata (file path, code type, dependencies).
- **Search:** When a task is received, query encoding finds top-K relevant chunks via cosine similarity.
- **Context expansion:** Found chunks are enriched with dependencies (imports, called functions) for a complete picture.

**Advantages:**
- Scales to large codebases (10K+ files)
- Finds relevant code even without explicit file specification
- Discovers patterns and best practices from existing code

**Limitations:**
- Requires pre-processing the entire repository
- Quality depends on the embedding model
- Additional computational resources for indexing

**Application:** Ideal for tasks like "add a new function in the style of existing code" or "find all places where authentication is handled."

#### 2. Test-Driven Development (TDD) Agent

**Concept:** The agent follows classic TDD methodology — first generates tests, then writes code to pass them.

**Workflow (Red-Green-Refactor):**
1. **RED phase:** The agent analyzes the feature description and generates a comprehensive test suite covering the happy path, edge cases, and error handling. Tests are run and should fail (the code is not yet written).

2. **GREEN phase:** The agent generates a minimal implementation sufficient to pass the tests. If tests fail, the agent analyzes error messages and iteratively fixes the code (up to 5-10 iterations).

3. **REFACTOR phase:** After tests pass, the agent improves code quality — removes duplicates, improves naming, optimizes structure. Then re-runs tests for a regression check.

**Advantages:**
- Guarantees testability of generated code
- Creates documentation through tests
- Reduces bug risk (tests are written before code, not after)
- Natural feedback loop for debugging

**Limitations:**
- Slower than direct code generation
- Requires good understanding of testing frameworks
- Difficulty testing certain code types (UI, async, I/O)

**Application:** Excellent for libraries, API endpoints, business logic — any code where tests are critical.

#### 3. Multi-Agent Code Review Approach

**Concept:** Multiple specialized agents work in a pipeline — one generates code, another checks for security, a third optimizes performance.

**Agents in the pipeline:**
- **Generator Agent:** Focuses solely on functionality — generates code that solves the task.
- **Security Reviewer Agent:** Scans for vulnerabilities (SQL injection, XSS, hardcoded secrets, insecure dependencies).
- **Performance Reviewer Agent:** Analyzes algorithmic complexity, identifies N+1 queries, suggests optimizations.
- **Style Reviewer Agent:** Checks compliance with the code style guide, naming conventions, and documentation standards.
- **Integration Agent:** Coordinates feedback between agents and makes the final code version decision.

**Workflow:** The task goes to the Generator Agent to create Code v1, which sequentially passes through Security Review (security check with feedback), then Performance Review (performance analysis with feedback), and finally Style Review to produce the Final Code. Each reviewer can return code to the generator for corrections.

**Advantages:**
- Agent specialization improves quality in each dimension
- Natural imitation of the human code review process
- Modularity — reviewer agents can be added or removed

**Limitations:**
- High latency (multiple LLM calls)
- Risk of conflicting recommendations
- Feedback integration complexity

**Application:** Production-ready code for critical applications where quality matters more than speed.

#### 4. Incremental Refinement Strategy

**Concept:** Instead of generating a complete solution in one pass, the agent creates a series of incremental improvements.

**Process:**
1. **Skeleton Generation:** The agent creates the basic structure — classes, functions, interfaces without implementation.
2. **Core Logic Implementation:** Fills in the main logic in critical functions.
3. **Edge Case Handling:** Adds handling for edge cases and errors.
4. **Optimization Pass:** Improves performance and readability.
5. **Documentation Pass:** Adds docstrings, comments, type hints.

**Advantages:**
- Allows human-in-the-loop after each stage
- Easier debugging (fewer changes at a time)
- Reduces cognitive load on the LLM

**Limitations:**
- More LLM calls
- Requires clear separation into stages

**Application:** Complex features requiring architectural decisions.

### Choosing a Strategy

The choice of approach depends on context:

| Strategy | Best For | Worst For |
|-----------|-----------|----------|
| **Basic iterative** | Quick fixes, simple features | Large codebases, complex logic |
| **RAG-based** | Large repositories, code in existing style | New projects, non-standard code |
| **TDD Agent** | APIs, libraries, critical logic | UI code, prototypes |
| **Multi-Agent Review** | Production code, security-critical | Quick experiments |
| **Incremental Refinement** | Complex features, new architectures | Minor bugfixes |

In practice, hybrid approaches are often used — for example, RAG for context gathering + TDD for implementation + Multi-Agent for final review.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[06_Computer_Use_Agents|Computer Use Agents]]
**Next:** [[08_Process_Reward_Models|Process Reward Models]]
