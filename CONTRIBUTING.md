# Contributing

Thanks for your interest in improving the AI Agents Course. Contributions of all kinds are welcome — from fixing typos to adding new content.

## Ways to Contribute

### Report Issues
- Factual errors or outdated information
- Broken links or navigation issues
- Unclear explanations
- Missing topics you'd like to see covered

Open an [issue](https://github.com/shurankain/agentic-ai-course/issues) with a clear description of the problem and which lesson it affects.

### Fix Errors
- Typos, grammar, formatting
- Incorrect code examples
- Cross-references that name the wrong module or a lesson that no longer exists

### Improve Existing Content
- Clarify complex explanations
- Add practical code examples (Java, Rust, or Python)
- Add diagrams or visual aids
- Update references to newer papers or tools

### Add New Content
- New lessons within existing modules
- Hands-on exercises or assignments
- Additional practical projects
- Translations

## How to Submit Changes

1. Fork the repository
2. Create a branch: `git checkout -b fix/description` or `git checkout -b add/description`
3. Make your changes
4. Submit a pull request with a clear description of what you changed and why

## Content Guidelines

- **Keep it practical.** Prefer concrete examples over abstract theory.
- **Stay current.** Reference the latest versions of tools, frameworks, and papers.
- **Be concise.** Respect the reader's time. Say it once, say it clearly.
- **Follow existing structure.** A lesson opens on its H1 and closes with Key Takeaways. Do not add navigation blocks, Previous and Next lines, or `[[wiki-links]]`: GitHub does not render them and they break on every rename. Point at another lesson in plain prose, for example "see Module 14, Prompt Injection". Only `00_Home.md` and `Table_of_Contents.md` carry Markdown links.
- **Code examples** should be in Java (primary), Rust, or Python. Use realistic scenarios, not toy examples.
- **No placeholder API keys.** Always use environment variables in examples (e.g., `OPENAI_API_KEY`).

## File Naming

- Modules: `XX_Module_Name/` (e.g., `03_AI_Agents_Core/`)
- Lessons: `XX_Lesson_Name.md` (e.g., `02_Agent_Architectures.md`)
- Use underscores, not spaces or hyphens

## Questions?

Open a [discussion](https://github.com/shurankain/agentic-ai-course/discussions) or an issue. Happy to help.
