# Introduction to Prompt Engineering

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Prompt Engineering
**Previous:** [[../01_LLM_Fundamentals/10_Implementation_from_Scratch|Implementing a Transformer from Scratch]]
**Next:** [[02_Advanced_Prompting|Advanced Prompting Techniques]]

---

## The Prompt as a Natural Language Program

When large language models began demonstrating remarkable capabilities in the 2020s, researchers faced an unusual problem: how exactly should one "talk" to these systems to obtain desired results? The answer was a new discipline — **Prompt Engineering**, the art and science of formulating queries for language models.

The key insight: **a prompt is not merely a question or command — it is a program for a language model, written in natural language**. As in programming, there are patterns, anti-patterns, best practices, and debugging methods.

---

## How LLMs Learn to Follow Instructions

The ability of modern LLMs to understand and execute instructions results from specialized training through two stages:

1. **SFT (Supervised Fine-Tuning)** — the model is trained on (instruction, response) pairs
2. **RLHF (Reinforcement Learning from Human Feedback)** — the model learns to prefer responses that humans rate higher

A base model simply predicts the next token. A model after SFT takes the user's instruction into account. After RLHF, the model maximizes human preference scores.

### What RLHF Shapes

RLHF creates several key behavioral patterns:

**Format adherence** — humans prefer structured responses, so the model becomes better at responding in the requested format.

**Helpfulness** — reward for useful responses makes the model strive to help rather than simply continue text.

**Acknowledging uncertainty** — penalties for hallucinations teach the model that "I don't know" is an acceptable answer.

**Safety alignment** — penalties for harmful content create a mechanism for refusing dangerous requests.

### Practical Implications

1. The model "wants" to help — no need to persuade it, just clearly explain what is needed
2. Explicit instructions work better than hints
3. Format in the prompt → format in the response
4. Conflict between safety and helpfulness — sometimes the model refuses when it should not

---

## Why This Is Critical for AI Agents

In the context of AI agents, prompt engineering becomes especially significant:

**System prompts as the agent's constitution** — they define all agent behavior throughout the entire interaction. A poor system prompt produces unpredictable behavior.

**Tool prompts as API contracts** — the quality of tool descriptions determines how effectively the agent uses them.

**Reasoning prompts as cognitive architecture** — how you formulate instructions for reasoning directly affects the quality of the agent's decisions.

**Output formatting as an integration contract** — if the format is not precisely specified, parsing will break, and the agent will degrade.

---

## Anatomy of an Effective Prompt

Effective prompts typically contain five key components.

### Role and Context

Defining the model's role activates specific knowledge patterns. When you tell the model "You are an experienced Java architect with 15 years of experience," you activate expert reasoning patterns the model encountered in training data.

Context acts as a filter: "You are working on an enterprise banking system that must comply with PCI DSS requirements" — this steers all model decisions toward security and compliance.

### Instructions

The heart of the prompt — what you want to receive. The more specific the instructions, the better the result.

Compare: "Help with the code" vs. "Analyze the code for: (1) performance issues, (2) SOLID violations, (3) security vulnerabilities. For each issue, describe it, explain the consequences, and propose a fix."

Specific instructions create a clear structure for the response and establish quality criteria.

### Input Data

When the prompt includes data for processing, it is important to clearly separate it from the instructions. Recommended approaches: using delimiters (triple quotes, XML tags), explicit "Input data:" labels, placing data after instructions.

### Output Format

Explicitly specifying the expected format is one of the most underestimated aspects. For agentic systems this is especially critical: if you expect JSON but the model returns Markdown, your parser will break.

Effective strategies: show an example of the desired output, provide a JSON Schema, use a TypeScript-style interface description.

### Examples (Few-shot)

Examples are arguably the most powerful tool. Even a few high-quality examples can dramatically improve response quality. Instead of explaining with words, you demonstrate — the model extrapolates the pattern.

Key principles: representativeness (examples should cover typical cases), diversity, quality, ordering (start with simple ones).

---

## Basic Prompting Techniques

### Zero-shot: Power Without Examples

Zero-shot prompting is an approach where you give the model a task without examples. The model relies solely on its knowledge from training.

Works well for familiar tasks: translation, summarization, classification, information extraction.

Advantages: token savings, simplicity, speed. Limitations: less predictable format, worse performance on non-standard tasks.

### Few-shot: Learning Through Examples

Few-shot prompting is a game changer for complex tasks. You provide several input-output example pairs, and the model learns the pattern on the fly through in-context learning.

Recommendations: the optimal number of examples is 3-5, choose examples semantically close to the expected query, include edge cases, pay attention to ordering (models often better "remember" recent examples).

### Chain-of-Thought: Thinking Out Loud

Chain-of-Thought is one of the most significant discoveries. Instead of asking for an immediate answer, we ask the model to "think out loud," showing intermediate reasoning steps.

The simplest form is adding the phrase "Let's think step by step." This modification can dramatically improve results on tasks that require reasoning.

Why does it work? When the model "thinks out loud," it creates intermediate representations that serve as "working memory" for subsequent steps. Without CoT, the model must compress all reasoning into a single "jump" to the answer.

Effective for: mathematical problems, logic puzzles, multi-step tasks, planning, analysis. Less useful for simple factual questions.

---

## Structuring Prompts

### Markdown as a Structural Language

Markdown has become the standard for structuring complex prompts. Models understand headings, lists, code blocks, and tables well. This helps the model correctly parse the prompt and organize the response.

### XML Tags for Clear Boundaries

For particularly complex prompts, XML tags provide even clearer section separation. Tags such as `<system>`, `<context>`, `<question>`, `<instructions>` create explicit boundaries between prompt parts. Especially important for programmatic use.

### Format Selection: Recommendations

**Plain text** — for simple requests and creative tasks.
**Markdown** — for code review, reports, and structured output.
**XML tags** — for clear section separation, ideal for RAG systems.

Practical recommendations:
- For Claude: XML tags work especially well
- For GPT-4: Markdown is preferred
- For code tasks: Markdown with triple backticks
- For RAG/context: XML tags for clear separation

Important: high-quality prompt content matters more than format.

---

## Common Mistakes and How to Avoid Them

### Overly General Instructions

"Make it good" or "Improve the code" — these are not instructions, they are wishes. The model is forced to guess what you mean.

Solution: define specific criteria. Instead of "Make it good," specify: "Readability: methods under 20 lines, camelCase for methods, JavaDoc for public methods."

### Missing Examples for Non-obvious Tasks

If the task is non-standard or the output format is specific, the absence of examples almost guarantees disappointment. The model cannot guess your specific format.

### Conflicting Instructions

Surprisingly often, prompts contain mutually exclusive requirements: "Be concise. Explain each step in detail. The answer must be one sentence. Provide examples for each point."

The model is forced to ignore part of the instructions. Always check the prompt for consistency.

### Ignoring Information Order

Research shows primacy and recency effects — models better "remember" information at the beginning and end of the prompt. Long context in the middle can partially "get lost."

Practical advice: place the most important information at the beginning, key instructions at the end, and less critical context in the middle.

### Example Order in Few-shot

The order of examples affects results more than many expect.

**Recency bias** — the model tends to "copy" the format of the last example. The final example exerts a disproportionately large influence.

**Label bias** — if most examples have a particular label, the model will be biased toward that label.

Optimal order: start with a simple example (sets the baseline) → add 1-2 medium-complexity examples → end with the example closest to the target query.

Research (Min et al., 2022): the format of examples matters more than their "correctness," but correct answers provide +10-15% accuracy.

---

## Iterative Development Process

Prompt development is an iterative process, similar to code development. Nobody writes a perfect prompt on the first attempt.

Effective cycle:
1. Start simple — a minimal prompt expressing the core idea
2. Test on diverse inputs — not on a single example
3. Analyze errors — understand why the result is unsatisfactory
4. Add refinements — address specific problems
5. Add examples — for cases that are hard to explain with words
6. Simplify — remove everything that does not improve the result
7. Document — record which changes led to which improvements

---

## Key Takeaways

1. **A prompt is a program** in natural language that demands serious treatment

2. **Five components of an effective prompt**: role/context, instructions, input data, output format, examples

3. **Three basic techniques** — zero-shot, few-shot, and Chain-of-Thought — solve different classes of tasks

4. **Structuring via Markdown and XML** helps the model correctly parse complex prompts

5. **Specificity is critical**: vague instructions produce vague results

6. **Iterative development** is the only path to high-quality prompts

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Prompt Engineering
**Previous:** [[../01_LLM_Fundamentals/10_Implementation_from_Scratch|Implementing a Transformer from Scratch]]
**Next:** [[02_Advanced_Prompting|Advanced Prompting Techniques]]

---

## Practical Prompt Examples

### Example 1: Zero-shot Prompt

A zero-shot prompt relies on the model's knowledge without providing examples. Suitable for standard tasks.

```text
You are an experienced Python developer. Analyze the code for performance and security issues.

Code:
def process_user_data(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    results = []
    for item in database.execute(query):
        results.append(item)
    return results

For each issue, provide: description, consequences, solution.
```

### Example 2: Few-shot Prompt

A few-shot prompt shows several examples of desired behavior to establish a clear pattern.

```text
Classify customer reviews: TECHNICAL_ISSUE, FEATURE_REQUEST, SERVICE_COMPLAINT.

Examples:

Review: "The app crashes when uploading a photo. iPhone 12."
Category: TECHNICAL_ISSUE
Priority: High

Review: "I want to export data to Excel."
Category: FEATURE_REQUEST
Priority: Medium

Review: "Support takes three days to respond — unacceptable for a paid subscription!"
Category: SERVICE_COMPLAINT
Priority: High

Classify: "I can't log in after the update, it says 'authentication error'."
```

### Example 3: Chain-of-Thought Prompt

Chain-of-Thought helps the model "think out loud," breaking a complex task into steps.

```text
You are a software architect. Design a system for an online store.

Requirements:
- 10,000 orders/day, peak load: 500 concurrent users
- Integration with 3 payment systems
- 3-year order history, 99.9% availability

Let's think step by step:

1. Analyze the load and determine the RPS
2. Identify critical components and scalability needs
3. Choose a database with justification
4. Design the payment architecture with reliability in mind
5. Define a strategy for ensuring 99.9% availability
6. Describe the data storage scheme for 3 years

For each step, explain the reasoning, then formulate the final architecture.
```

### Example 4: System Prompt for an AI Agent

A combined prompt for creating an agent system prompt using multiple techniques:

```markdown
# Role
AI agent for code review of Python projects. Goal: improve code quality.

# Context
Project: a FastAPI web application for a fintech startup
Requirements: PEP 8, type hints, 80%+ test coverage, security

# Analysis Process
1. Structure: module organization and dependencies
2. Style: PEP 8, type hints
3. Security: SQL injection, XSS, secrets
4. Performance: N+1 queries, algorithms
5. Testability: dependencies, coverage

# Output Format
## [File]
### Critical Issues
- [Description] → [Solution]

### Improvements
- [Description] → [Recommendation]

### Positive Highlights
- [What is done well]

# Constraints
- Specific improvements, not a full rewrite
- Priority: security > style
- If the code is good, say so
```
