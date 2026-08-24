# LLM Providers and Working with APIs

Consider the cloud computing market in the early 2010s: AWS dominated, Azure and Google Cloud were emerging, each offering their services with different APIs, pricing, and capabilities. Today we observe a similar, but far more dynamic, landscape in the world of large language models. OpenAI, Anthropic, Google, Meta, DeepSeek, xAI, and dozens of other providers offer their models through APIs, and choosing the right provider becomes a strategic decision for any project involving AI agents.

Understanding the LLM provider landscape is not simply a matter of picking the "best model." It is a question of balancing quality, cost, speed, reliability, and compliance with security requirements. The market has also split into two paradigms: **standard models** (optimized for speed and general tasks) and **reasoning models** (optimized for complex problem-solving via extended thinking). For agent developers, this knowledge shapes system architecture, fallback strategy, and project budget.

---

## Major Providers and Their Models

### OpenAI

OpenAI popularized LLMs through ChatGPT and continues to set the pace of the industry. Their API remains the de facto standard that other providers emulate.

**Frontier models:**
- **GPT-5.6**: current flagship family (public release July 9, 2026), shipped in three variants. **Sol** is the frontier model for coding, science, and cybersecurity, at $5/$30 per 1M tokens. **Terra** matches GPT-5.5 quality at roughly half the cost. **Luna** is the fast, high-volume tier. Sol is about 54% more token-efficient on coding tasks than GPT-5.5.
- **GPT-5.6 rollout under government review**: OpenAI previewed the family on June 26 to roughly 20 government-vetted organizations, with access approved customer by customer, then released it publicly on July 9 once the review cleared. The trigger was Sol's cyber capability. This is the first US frontier release gated on government sign-off, and together with the Fable 5 export ban it makes national-security policy a first-order constraint on model availability. Module 18 covers the framework behind it.
- **July 30 price cuts**: Luna dropped 80% to $0.20/$1.20 per 1M tokens and Terra dropped 20%, while Sol was unchanged. OpenAI attributed the cuts to roughly 20% lower serving costs from inference and GPU-kernel optimization. Three weeks from general availability to an 80% price cut is the clearest single measure of how fast inference economics move.
- **GPT-5.6-Cyber**: released August 10, 2026 for exploit development and security research, alongside an expanded Daybreak program. Access depends on user verification. See Module 14 for what its reported capability numbers do and do not measure.
- **GPT-Live-1 and GPT-Live-1 mini**: full-duplex voice models (July 2026) that listen and speak at the same time instead of waiting for a turn to end. Now the default voice models for Go, Plus, and Pro. Module 09 covers the architecture.
- **GPT-5.5 Instant**: previous default ChatGPT model (May 5, 2026). 88.7% SWE-bench Verified. 52.5% fewer hallucinations on high-stakes prompts (medicine, law, finance). ~$2.50/$10.00 per 1M tokens. Full GPT-5.5 API: $5/$30 per 1M tokens; Pro configuration $30/$180.
- **GPT-5.4**: earlier flagship (March 2026). $2.50/$15.00 per 1M tokens.
- **GPT-5.4-Cyber** and **GPT-5.5-Cyber**: specialized cybersecurity models for vulnerability triage, malware analysis, red teaming, and patch validation. Access via Trusted Access for Cyber (TAC), a tiered identity verification system.
- **GPT-Rosalind**: first vertical frontier model, purpose-built for life sciences and drug discovery (June 3, 2026). Combines GPT-5.5's agentic capabilities with stronger performance in medicinal chemistry, genomics, and lab workflows. Uses 31% fewer tokens than GPT-5.5 on research tasks. Named after Rosalind Franklin. Available via trusted-access deployment structure.
- **GPT-5.2**: **retired from ChatGPT (June 12, 2026)**: GPT-5.2, GPT-5.2 Instant, GPT-5.2 Thinking, and GPT-5.2 Pro all removed without advance notice. Existing conversations migrated to GPT-5.5 automatically. API deprecation to follow. Previously: natively multimodal, 128K-1M context.
- **GPT-5**: earlier generation. 128K-1M context.
- **GPT-5.2-Codex / GPT-5.3-Codex**: specialized agentic coding models optimized for autonomous multi-step code generation and tool use.
- **GPT-5.3-Codex-Spark**: research preview (April 2026). First model designed for real-time coding, optimized for near-instant feel: delivers 1000+ tokens per second.
- **GPT-4o**: deprecated from ChatGPT (February 13, 2026). API still available but being phased out. 128K context.
- **GPT-4.1**: deprecated from ChatGPT (February 13, 2026) and API (February 16, 2026). Previously optimized for coding and instruction following.

**Reasoning models:**
- **o3**: frontier reasoning model. Generates internal "thinking tokens" before the final answer. Excels on math, science, and complex multi-step problems.
- **o4-mini**: cost-efficient reasoning model. Strong on STEM tasks at a fraction of o3's cost.

**Budget models:**
- **GPT-5-mini**: next-generation budget model ($0.25/$2.00 per 1M tokens).
- **GPT-5-nano**: ultra-light model for high-volume simple tasks ($0.05/$0.40 per 1M tokens).
- **GPT-4o-mini**: previous-gen budget model, still widely available.
- GPT-3.5 Turbo is fully deprecated.

**OpenAI highlights:**
- Most mature ecosystem and documentation
- Broad support across SDKs and frameworks
- Native function calling / tool use with parallel calls and strict mode
- Structured outputs (guaranteed JSON via constrained decoding)
- Batching API for bulk processing at a 50% discount
- Prompt caching for savings on repeated prefixes
- Responses API (successor to Assistants API, which was deprecated in August 2025 with sunset scheduled for August 2026, so verify its status before building on it) and Agents SDK (v0.10.2, provider-agnostic, supports 100+ LLMs) for multi-agent orchestration
- OpenAI filed a confidential S-1 (May 22, 2026). Listing timing is unsettled as of August 2026: reports variously describe a public prospectus targeting September, a Q4 2026 listing contingent on completing the for-profit restructuring, and a slip into 2027
- **Astra** is the model to watch, not to plan around. OpenAI previewed it to US policymakers in late July as a family built to coordinate multiple agents on tasks running hours or days, and reported that it solved ten long-open problems in mathematics and theoretical computer science. Whether it ships as GPT-6 or a point release is undecided, and it is slated to be the first model routed through the government pre-release review
- **MS-OpenAI exclusivity ended** (April 27, 2026). OpenAI models now available on AWS Bedrock, Google Cloud Vertex, Oracle, and sovereign cloud providers. Microsoft retains non-exclusive license through 2032 and ~27% of OpenAI Group PBC

### Anthropic

Anthropic, founded by former OpenAI researchers, positions itself as a company focused on AI safety. Their Claude models are distinguished by excellent instruction following, strong code generation, and long context capabilities.

**Claude model family (latest generation):**
- **Claude Fable 5**: the Mythos-class flagship (launched June 9, 2026). 95.0% SWE-bench Verified (independently confirmed), 80.3% SWE-bench Pro (vendor-reported). $10/$50 per 1M tokens. 1M context, 128K max output. Multi-day agentic coding sessions. Genomics: outperformed Science paper results at 100x smaller scale. In high-risk domains (cybersecurity, biology, chemistry, distillation), Fable 5 blocks responses and falls back to a lower tier. Knowledge cutoff January 2026. Access was suspended June 12 under an export-control directive and restored globally on July 1, 2026 after the Department of Commerce lifted it. The outage ran 19 days; see the export controls section below. From July 20, Max and Team Premium include Fable 5 up to 50% of weekly limits, while Pro and Team Standard reach it through usage credits.
- **Claude Opus 5**: current Opus tier (July 24, 2026), at unchanged Opus pricing of $5/$25 per 1M tokens. Anthropic positions it as approaching Fable 5's intelligence at half the price, and it scores about 96% on SWE-bench Verified, at the top of a cluster where the leaderboard has stopped separating models. 1M context, 128K max output, adaptive thinking by default, and a five-level effort dial. Fast mode runs about 2.5x faster at twice the base price. Default model for Claude Max. Two features matter for agent architecture rather than for a spec sheet: automatic fallback routing, and tool swapping mid-conversation without breaking the prompt cache.
- **Claude Sonnet 5**: current Sonnet tier (June 30, 2026) and the most agentic Sonnet so far. 1M-token context, 63.2% SWE-bench Pro, 81.2% OSWorld on computer use, and it edges Opus 4.8 on GDPval knowledge work. Anthropic reports lower rates of hallucination, sycophancy, and susceptibility to prompt injection than Sonnet 4.6. Became the default for Free and Pro users on July 1. Introductory pricing of $2/$10 per 1M tokens runs through August 31, 2026, after which the standard $3/$15 applies.
- **Claude Haiku 4.5**: fast and cost-effective model for high-volume, simpler tasks.
- **Claude Opus 4.8**: superseded by Opus 5 (May 28, 2026). 88.6% SWE-bench Verified, 69.2% SWE-bench Pro, 74.6% Terminal-Bench 2.1. Sharper judgement: the model is more likely to flag uncertainties and less likely to make unsupported claims. $5/$25 per 1M tokens. Messages API accepts `system` records inside `messages[]`. This allows mid-task system prompt changes without breaking prefix cache. Released alongside Anthropic's $65B funding round at $965B valuation.
- **Claude Opus 4.7**: earlier flagship (April 16, 2026). 87.6% SWE-bench Verified (13% lift over 4.6). High-resolution vision (2576px / 3.75MP, first Claude model with this capability). Task budgets give the model a running countdown to prioritize work and finish agentic loops gracefully. Effort levels enable adaptive reasoning: the model decides how much to think based on task complexity.
- **Claude Opus 4.6** and **Claude Sonnet 4.6**: earlier generation, still deployed in production code that has not migrated.

*Note:* Claude Opus 4 (`claude-opus-4-20250514`) and Sonnet 4 (`claude-sonnet-4-20250514`) were **retired from the API on June 15, 2026**, hard cutoff, no grace period. Existing code using these model IDs must migrate to the current Opus and Sonnet tiers, now Opus 5 and Sonnet 5.

**Anthropic operates a 4-tier model hierarchy (as of August 2026):** Mythos ($10/$50), then Opus ($5/$25), then Sonnet ($3/$15), then Haiku ($0.25/$1.25). Fable 5 is the public-facing Mythos-class model; full Mythos 5 remains restricted. The tiers are no longer a simple quality ladder: Opus 5 leads SWE-bench Verified while sitting a tier below Fable 5, which is what happens when a lab optimizes a cheaper tier against the benchmark the expensive tier already saturated.

**Claude Mythos 5**: Anthropic's most capable model, the unrestricted version of the Mythos architecture. Fable 5 is a safety-filtered public release of the same underlying model. Full Mythos 5 is offered in limited availability to approved customers in Project Glasswing. Previously available as "Mythos Preview" to ~52 organizations achieving 93.9% SWE-bench Verified (April 2026). The model succeeds 73% of the time on expert-level cyber-attack simulations that no model could complete before April 2025.

**Export controls (June 12, 2026):** The Trump administration issued the first-ever model-level export ban, targeting Fable 5 and Mythos 5, not hardware. All foreign national access was suspended, including foreign nationals inside the US and foreign employees of Anthropic. Anthropic received the directive on a Friday evening and disabled both models for all customers while working to comply, because it could not separate foreign nationals from domestic users in real time. Officials cited a technique to bypass Fable 5's safeguards as the trigger. The Department of Commerce lifted the directive on June 30 and Anthropic restored global access on July 1, across Claude.ai, the Platform API, Claude Code, and Cowork.

The episode is a landmark precedent worth remembering even though the models are back: the first time export controls targeted an AI model rather than the chips used to build it. For anyone designing on a single provider, it is also the clearest available argument for a fallback path. A capability you depend on can disappear on a Friday evening for reasons that have nothing to do with your architecture, your contract, or the vendor's intentions.

Anthropic reached $47B ARR (as of late May 2026) against OpenAI's $25B. Read that pair with its accounting caveat: Anthropic reports gross, counting end-customer spend that flows through cloud resellers, while OpenAI reports closer to net, so the gap is smaller than the headline by an amount neither company publishes and the ordering is unresolved. Growth trajectory: $87M (Jan 2024), then $1B (Dec 2024), then $9B (late 2025), then $30B (April 2026), then $47B (May 2026): 80x in three years. $65B Series H at $965B post-money, the largest private round in the sector to that point. Module 18, Enterprise AI Adoption carries the full comparison. Claude Code reached $1B ARR within 6 months of launch. Andrej Karpathy joined Anthropic's pre-training team (May 19, 2026). Anthropic acquired Stainless (May 18, 2026) for SDK/API-lifecycle tooling. At the Code with Claude conference (May 6, 2026), Anthropic announced **Dreaming**, background optimization where agents analyze past sessions and rewrite their memory between runs, effectively "sleeping" to improve, the first implementation of continual learning for production agents without model retraining.

**Extended thinking:** Claude models support an "extended thinking" mode: Anthropic's approach to reasoning. The model produces a chain-of-thought in a dedicated thinking block before generating the final response. Controlled via the `budget_tokens` parameter.

**Anthropic highlights:**
- Context window: 200K tokens standard, 1M available on Opus 4.6 / 4.7 / 4.8
- "Constitutional AI": models trained to follow ethical principles
- Excellent performance in long document analysis and code generation
- **MCP (Model Context Protocol)**: Anthropic-created open standard for tool integration, now an industry standard governed by AAIF
- Tool use API compatible with the OpenAI format
- Computer use capability: Claude can interact with desktop applications via screenshots

### Google

Google offers models through Vertex AI (enterprise) and Google AI Studio (for developers).

**Gemini family:**
- **Gemini 3.7 Flash**: current Flash tier (August 13, 2026). Context 1,048,576 tokens, max output 65,536, thinking configurable at `low`, `medium`, or `high`. Accepts text, image, video, audio, and PDF. Function calling, code execution, file search, and structured output are generally available; computer use remains in preview. Pricing is $0.75/$3.75 per 1M tokens through December 31, 2026, rising to $1.50/$7.50 on January 1, 2027. Google reports 43.6 on FrontierCode against 34.4 for 3.6 Flash, and 30.4 on AutomationBench against 17.
- **Gemini 3.6 Flash, 3.5 Flash-Lite, 3.5 Flash Cyber**: released July 21, 2026. The Cyber variant is restricted to governments and trusted partners.
- **Gemini 3.5 Flash**: frontier-quality model at Flash speed (May 19, Google I/O 2026). $1.50/$9 per 1M tokens, 1.05M context. 280+ output tok/sec. Terminal-Bench 2.1 76.2%, MCP Atlas 83.6%. `thinking_level` enum replaces `thinking_budget` int.
- **Gemini 3.5 Pro**: announced at I/O on May 19, 2026 and **still unshipped as of August 2026**. Targeted for June, slipped to July, then delayed by several months after internal testing showed shortfalls on coding and long-horizon reasoning. It remains in limited enterprise preview, with no model card, pricing, or official benchmarks. Numbers circulating for it come from leaks, so treat them as rumor.
- **Gemini 3.1 Pro Preview**: Google's most capable published reasoning model (February 2026).
- **Gemini 3 Flash**: default Gemini model (December 2025). Outperforms 2.5 Pro at 3x the speed. Excellent cost/performance ratio.
- **Gemini Omni**: video output with physics understanding, SynthID watermark on all outputs (Google I/O 2026).
- **Gemini 2.5 Pro**: previous flagship with 1M token context and built-in thinking capabilities. Still widely available.
- **Gemini 2.5 Flash**: optimized for speed and cost with configurable "thinking budgets," ideal for real-time applications.
- **Gemini 2.0 Flash**: budget model for high-volume workloads. Retired June 1, 2026.

The "Gemini Ultra" branding has been abandoned, and the Pro tier is nominally the flagship. In practice Google spent the summer of 2026 shipping Flash models while its Pro flagship stayed in preview, and it has said pretraining has begun for Gemini 4. Two lessons for anyone planning against a vendor roadmap. An announced model is not a shipped model, and the tier a lab actually iterates on tells you where it thinks the volume is.

The Gemini 3.7 Flash AutomationBench score is worth sitting with: 30.4 means the model fails roughly 70% of that benchmark's agentic tasks. Coding benchmarks near saturation make current models look finished. Agentic benchmarks show how much room is left.

**Google advantages:**
- Record context sizes, 1M tokens standard (Gemini 3 Flash/Pro), 2M in preview
- Native multimodal understanding and generation (text, images, audio, video)
- Grounding with Google Search for up-to-date information
- Native integration with Google Cloud services
- Competitive pricing, especially for Flash models
- **Antigravity 2.0** (May 19, 2026): replaced Gemini CLI as the desktop orchestration hub with CLI, SDK, plugins, skills, hooks, and subagent workflows
- Moving from daily prompt limits to compute-used pricing model. AI Ultra $100/month with 5x higher limits. Gemini Spark personal agent for AI Ultra subscribers
- **Siri AI deal (WWDC, June 9, 2026):** Google Gemini powers the rebuilt Siri AI under a ~$1B/year multi-year agreement. Distribution across 2+ billion Apple devices: the largest AI distribution deal
- **Gemini 3.5 Flash in Google Search** (June 17): folded into Google Search as the global standard, with agents monitoring information continuously and auto-booking capability. Marks the transition of agentic AI from demo to default in the world's most-used search engine

### xAI

Founded by Elon Musk, xAI formally ceased as a separate company (May 6, 2026): now operates as the SpaceXAI division. SpaceX-xAI filed an S-1 for IPO, ticker SPCX, targeting $1.75-2T valuation, roadshow June 4-8.

**Models:**
- **Grok 4.6**: current flagship (August 12, 2026), tuned for long-running agents, coding, and multi-step tasks. 500K context at $2/$6 per 1M tokens below 200K, and the price doubles above 200K. That cliff is a useful thing to internalize: on this model, a context-stuffing strategy changes cost class partway through a long session rather than scaling smoothly. Module 17 covers the general case.
- **Grok 4.5**: previous flagship (July 8, 2026), released with an Opus-class claim. Independent evaluation from Artificial Analysis placed it fourth behind Claude Fable 5, GPT-5.5, and Claude Opus 4.8, which is a useful gap between vendor positioning and third-party measurement.
- **Grok 4.3**: earlier frontier model (May 4, 2026). $1.25/M input tokens, 1M-token context, native video input, built-in reasoning.
- **Grok Build 0.1**: dedicated coding model (May 14, 2026). 256K context, $1/$2 per 1M tokens. Optimized for code generation and agent workflows.
- **Grok 4.20 Beta**: previous flagship (February 2026). Routes queries to 4 parallel specialist agents.
- **Grok 3**: previous flagship, competitive on major benchmarks (MMLU 92.7%, GPQA Diamond 84.6%)
- **Grok 3 Mini**: efficient model for standard tasks

**xAI highlights:**
- Competitive pricing aimed at undercutting incumbents
- Real-time access to X (Twitter) data
- OpenAI-compatible API format
- **Connectors**: integrations with SharePoint, Outlook, OneDrive, Google Workspace, Notion, GitHub, Linear, and custom MCP servers
- **Computer use**: can write code, run programs, install dependencies, create files (competitor to Claude Computer Use)

### Mistral AI

A French company offering efficient open-weight models. Attractive for European companies due to GDPR compliance.

**Models:**
- **Mistral Large 3**: flagship model (675B total, 41B active MoE, Apache 2.0), competitive with frontier proprietary models
- **Mistral Medium 3.5**: new flagship (May 1, 2026). Dense 128B model, 256K context, open weights (modified MIT license). 77.6% SWE-bench Verified. $1.50/$7.50 per 1M tokens. Notable: one model for chat, reasoning, and code (no separate reasoning models). Remote agents via Vibe: async cloud agents for routine tasks with GitHub/Slack integration.
- **Mistral Medium 3**: mid-tier model ($0.40/$2.00 per 1M tokens), strong cost/quality balance
- **Mistral Small 3**: compact and efficient for simpler tasks
- **Magistral**: reasoning-focused models
- **Devstral 2**: specialized for code generation
- **Ministral**: edge-friendly compact models

**Mistral highlights:**
- Models available with open weights (can be self-hosted)
- Competitive pricing with good quality
- GDPR compliance out of the box (EU company)
- Strong multilingual support for European languages

### Other Notable Providers

**Amazon Bedrock**: access to models from multiple providers (Anthropic, Meta, Mistral, Cohere, and now OpenAI) through a unified AWS API with enterprise guarantees. OpenAI models became available on Bedrock in April 2026, ending nearly 7 years of Azure exclusivity for OpenAI models. AWS shops no longer need Azure for GPT-5.5 access.

**Azure OpenAI**: OpenAI models with enterprise SLA, compliance, and integration with the Azure ecosystem. No longer the exclusive hyperscaler for OpenAI models (as of April 27, 2026), but retains deep integration with Microsoft 365 and Azure AI services.

**Microsoft MAI models** (Build 2026, June 2), Microsoft's first in-house model family, developed without OpenAI. Seven models announced: **MAI-Thinking-1** (35B parameters, 256K context, first Microsoft reasoning model), **MAI-Code-1-Flash** (optimized for code), **MAI-Image-2.5**, **MAI-Transcribe-1.5**, **MAI-Voice-2**, and others. MAI-Thinking-1 in private preview on Microsoft Foundry. This represents a self-sufficiency pivot: Microsoft building its own models to reduce OpenAI dependency, while simultaneously ending OpenAI's exclusivity.

**Cohere**: specializes in enterprise search and RAG. Embed models for embeddings, Command for generation, Rerank for re-ranking search results.

---

## Reasoning Models: A New API Paradigm

### What Are Reasoning Models?

Starting with OpenAI's o1 (September 2024) and rapidly expanding through o3, o4-mini, DeepSeek-R1, Claude extended thinking, and Gemini thinking modes, reasoning models represent a fundamentally new API paradigm.

Unlike standard models that generate responses token by token, reasoning models produce **thinking tokens**: an internal chain of reasoning: before generating the visible response. This dramatically improves performance on complex tasks: math, science, multi-step planning, and code.

### How the API Differs

**Thinking tokens:**
- Invisible to the user (in most implementations) but billed as output tokens
- Can represent 80-90% of total output for complex queries
- Controlled via parameters like `reasoning_effort` (OpenAI) or `budget_tokens` (Anthropic)

**Parameter restrictions:**
- Temperature is typically fixed at 1.0 (OpenAI reasoning models) or has limited ranges
- System prompts may be handled differently (some models use "developer" messages)
- Streaming behavior differs, thinking tokens may not stream

**Cost structure:**
- Thinking tokens are billed at the output token rate
- A simple question might cost 10x more through a reasoning model due to thinking overhead
- The `reasoning_effort` parameter lets you trade quality for cost (low/medium/high)

### When to Use Reasoning Models

**Use reasoning models for:** complex math and logic, multi-step planning, scientific analysis, hard coding problems, tasks where accuracy matters more than speed.

**Use standard models for:** simple Q&A, classification, extraction, creative writing, real-time chat, high-volume tasks where cost and latency matter.

**The hybrid approach is optimal:** route simple tasks to standard models, escalate complex ones to reasoning models. This can cut costs by 70%+ while maintaining quality where it matters.

---

## Open-Source Models: In-Depth Comparison

The open-weight landscape stopped trailing during 2026. On SWE-bench Verified, DeepSeek V4 Pro 0813 scores about 96.4% on the vals.ai leaderboard, within a point of the closed leader, and Kimi K3 about 93.4%, both above Claude Opus 4.8 at 88.6%. Kimi K3 took first place on Arena.ai's Frontend Code Arena with a 76% pairwise win rate, ahead of Fable 5 and GPT-5.6 Sol on that benchmark. Any mental model that treats open weights as the budget option several points behind the frontier is now wrong on the numbers.

Two qualifications keep this honest. Licenses have diverged: some of the strongest releases carry custom terms rather than MIT or Apache 2.0, so "open weights" no longer implies "usable on your terms". And serving capacity bounds real availability. Moonshot halted new subscriptions for Kimi K3 when demand outran its compute, which is a constraint a benchmark table cannot show you.

A third qualification sits outside the technology. Whether these weights stay downloadable is now a policy question: an open letter in July split the industry over restricting open-weight releases, the US administration accused Moonshot of distilling Anthropic's Fable model to train Kimi K3, and Treasury threatened sanctions against Chinese AI firms. Picking an open-weight model on benchmark scores alone therefore leaves out an availability risk that a hosted API does not carry. Module 18, Enterprise AI Adoption sets out who took which side and what it means for a procurement decision.

### DeepSeek

A Chinese lab that has repeatedly disrupted the industry, first with V3's training efficiency, then with R1's reasoning breakthrough, and now with V4's unified architecture.

**DeepSeek V4** (April 24, 2026): V4-Pro at 1.6T parameters / 49B active, V4-Flash at 284B / 13B active. 1M context. Hybrid CSA+HCA attention achieving 27% inference FLOPs vs V3.2 on 1M context. Open weights under MIT license. GPQA Diamond 90.1%. The R1 reasoning line has been discontinued, reasoning is now folded into V4 as a mode. Cached input at $0.003625/MTok is the lowest frontier cache-hit price. DeepSeek raised $7.4B at $52-59B valuation.

**DeepSeek V4-Flash-0731** (July 31, 2026) is the more instructive release. Same architecture, same size, retrained weights, and the DeepSWE coding-agent score went from 7.3 to 54.4 through post-training alone, at $0.14/$0.28 per 1M tokens. A jump that large without an architecture change is a reminder that post-training, not parameter count, is where much of 2026's capability gain came from.

**DeepSeek V4 Pro** reached general availability on August 13, 2026 and scores about 96.4% on SWE-bench Verified per vals.ai. Note that its peak pricing rose at general availability. The shorthand that DeepSeek is always the cheapest option no longer substitutes for computing the cost of your actual task mix.

**DeepSeek V3**: 671B parameters, 37B active (Mixture of Experts). Cut training cost by activating 37B of 671B parameters per token. Superseded by V4.

**Key innovations:**
- **Multi-head Latent Attention (MLA)**: KV-cache compression via low-rank projections: 93% memory savings
- **Multi-Token Prediction (MTP)**: Predicting 2 tokens per step: 1.8x inference speedup
- **FP8 Training**: Full training in 8-bit floating point: the first successful example at this scale
- **Training cost**: $5.5M: an order of magnitude cheaper than competitors

**DeepSeek R1**: the reasoning breakthrough. Achieved frontier-level reasoning through pure reinforcement learning (GRPO, Group Relative Policy Optimization), without supervised fine-tuning on reasoning traces. R1-Zero showed that reasoning can emerge from RL alone.

- Open-weight with MIT-like license
- Distilled versions available from 1.5B to 70B parameters
- Competitive with o3 on math and science benchmarks
- R1-0528: improved version with better instruction following

**DeepSeek V3.2**: integrates thinking and tool-use into a unified model, eliminating the need for separate standard and reasoning models.

**DeepSeek R2**: next-generation reasoning model, building on R1's GRPO approach with further improvements.

**When to choose DeepSeek:**
- Mathematical and algorithmic tasks (leader)
- Self-hosting with a limited GPU budget (MoE efficiency)
- Reasoning tasks where cost matters (R1 distilled variants)
- Research on MoE architectures and RL-based reasoning

### Meta Llama 4

Meta's latest generation represents a fundamental architectural shift from dense models to Mixture of Experts with native multimodality.

**Models:**
- **Llama 4 Scout**: 16 experts, 10M token context window (the largest of any production model). Designed for long-context tasks.
- **Llama 4 Maverick**: 128 experts, ~400B total parameters with 17B active. The performance flagship.
- **Llama 4 Behemoth**: ~2T parameters, originally planned as the largest open model. Cancelled/delayed (May 2026). No Llama 5 confirmed.

**Key changes from Llama 3:**
- **MoE architecture** (previously dense): dramatically better inference efficiency
- **Native multimodality**: trained from scratch on text, images, and video (not bolt-on adapters)
- **Massive context**: 10M tokens in Scout, far beyond any predecessor
- **Interleaved attention**: architectural innovation for efficiency

*Note:* Meta announced ~8,000 layoffs (~10% of workforce, May 20, 2026) and restructured as Meta Superintelligence Labs with Alexandr Wang as CAO. Llama 4 Behemoth was cancelled and no Llama 5 has been confirmed.

**When to choose Llama 4:**
- Production self-hosting (strongest ecosystem, mature tooling)
- Extremely long context requirements (Scout 10M)
- Multimodal applications
- When a permissive license matters (Llama license allows broad commercial use)

### Qwen 3

Alibaba's latest series, leading many benchmarks and offering the broadest size range.

**Models:**
- **Qwen3.8-Max**: Max-tier multimodal MoE (August 3, 2026), roughly 2.4T total parameters with 95B active per inference and a 1M-token context. Alibaba positioned it second only to Claude Fable 5 at announcement. Confirm whether weights actually shipped before treating it as open: reporting on this has been inconsistent, and the Max tier has historically been API-only.
- **Qwen 3.7 Max/Plus**: API-only closed-weights models (May 20, 2026). $2.50/$7.50 per 1M tokens, 1M context. Intelligence Index 56.6 (#1 Chinese model). Hallucination rate halved from 44.2% to 22.9% (as of late May 2026).
- **Qwen 3 235B**: MoE flagship (22B active parameters), 119 languages
- **Qwen 3 72B**: dense flagship
- **Qwen 3 32B / 14B / 8B**: range of dense models for different GPU budgets
- **Qwen 3.5 Medium**: mid-tier model (February 2026) with improved reasoning
- **Qwen 3.5 / 3.6**: open-weight models (Apache 2.0), standard for on-device deployment
- **Qwen 3 0.6B / 1.7B / 4B**: edge and mobile-friendly

**Key innovations:**
- **Hybrid reasoning**: can switch between "thinking mode" (extended reasoning) and "non-thinking mode" (fast) within the same model, controlled via system prompt
- **119 language support**: the most multilingual open model
- **Apache 2.0 license**: fully permissive

**When to choose Qwen 3:**
- Multilingual applications (broadest language coverage)
- When you need a single model with both reasoning and fast modes
- Asian language markets
- Edge deployment (strong small models down to 0.6B)

### Google Gemma 4

Google's newest open-weight family (April 2, 2026), distinct from the proprietary Gemini models. Designed for edge AI and personal AI development.

**Models:** Four sizes, Gemma 4 E2B (Effective 2B), E4B (Effective 4B), 26B MoE, and 31B Dense. All released under Apache 2.0 license. Natively multimodal (text + images).

**Key differentiator:** The smallest open multimodal models that run on consumer hardware. E2B runs on mobile devices, E4B on laptops, 26B MoE on a single GPU. Positioned for on-device AI, privacy-preserving applications, and development environments where sending data to cloud APIs is unacceptable.

**When to choose Gemma 4:** Edge/mobile deployment, on-device AI assistants, privacy-sensitive applications where no data leaves the device, development and testing of multimodal capabilities without GPU servers.

### Cohere Command A+

**Cohere Command A+** (May 20, 2026): the first fully open-source frontier model. 218B Sparse MoE / 25B active parameters, Apache 2.0 license. W4A4 quantization runs on 1xB200 or 2xH100: changes sovereign deployment economics from ~$240K to ~$60K per rack (as of late May 2026). Designed for enterprise search, RAG, and document processing.

**When to choose Command A+:** Sovereign deployment where Apache 2.0 matters, enterprise RAG at frontier quality without proprietary dependencies, GPU-constrained environments where quantized MoE is essential.

### Kimi K3 and the K2 line

**Kimi K3** (Moonshot AI, listed July 16, 2026, weights released around July 27): roughly 2.8T parameters, natively multimodal MoE, 1M-token context. It took first place on Arena.ai's Frontend Code Arena with a 76% pairwise win rate, ahead of Claude Fable 5 and GPT-5.6 Sol on that benchmark, and scores about 93.4% on SWE-bench Verified per vals.ai. Released under a custom "Kimi K3 License" rather than a permissive one, so check the terms before assuming commercial use.

Two things about K3 matter more than its scores. Moonshot **paused new subscriptions** days after release when demand outran available compute. That is what serving capacity looks like as a binding constraint. And the US administration publicly accused Moonshot of distilling Anthropic's Fable model to train it. That accusation put K3 at the centre of the open-weight policy fight described above.

**Kimi K2.6** (April 20, 2026), 1T MoE / 32B active parameters, modified MIT license. Agent Swarm: up to 300 sub-agents and 4000 steps in a single run. Designed for agentic workflows requiring massive parallelism.

**Kimi K2.7-Code** (June 12, 2026), coding-focused successor. Same 1T MoE / 32B active architecture, 256K context. +21.8% on Kimi Code Bench v2, +11.0% on Program Bench, +31.5% on MLS Bench Lite vs K2.6. 30% lower reasoning-token usage. $0.95/$4.00 per 1M tokens. **Caveat:** all published benchmarks are Moonshot proprietary: no independent third-party results on standard public suites at release. Moonshot is also seeking $2B at $30B valuation (its third financing round in six months).

### NVIDIA Nemotron 3 Ultra

**Nemotron 3 Ultra** (June 4, 2026): 550B MoE (55B active), hybrid Mamba-2/Transformer architecture. Combines Mamba-2 state-space layers for efficient long-context processing with Transformer attention blocks for tasks where attention excels. 1M token context with low memory usage. Artificial Analysis Intelligence Index 48, second among US open-weight models. ~6x higher inference throughput than comparable open LLMs. Includes Multi-Token Prediction (MTP) layers and Latent MoE routing. Available on Hugging Face, NVIDIA NIM, and OpenRouter under the permissive Linux Foundation license: shipped with training data and recipes.

**Nemotron 3 Nano Omni** (June 10, 2026), 30B-parameter omni-modal model unifying vision, audio, and language in a single open-weight architecture. Designed for edge and embedded deployments where running separate models per modality is impractical.

### MiniMax 3

**MiniMax 3** (China, June 12, 2026): aggressively priced at approximately $0.53 per 1M tokens, 15-30x cheaper than Western frontier models. 1M-token multimodal context. The pricing makes it relevant for high-volume production workloads where cost dominates model selection, particularly in Asian markets where Chinese models have strong language support.

### Zyphra ZAYA1-8B

**Zyphra ZAYA1-8B**: first notable fully AMD-trained release (Apache 2.0). Demonstrates viability of AMD MI300X for model training, breaking NVIDIA's training monopoly.

### Meta Muse Glimmer

**Muse Glimmer** (August 10, 2026, Apache 2.0): about 29.6B parameters including a 1.8B ViT-G/14 perception encoder, accepting text and images and generating text. Context above 131K. The architecture is worth studying rather than just listing, and Module 20 covers it: 52 layers in a repeating `local-local-local-global` pattern with a 2,048-token sliding window, grouped-query attention with 32 query heads and 2 KV heads, and RoPE position encoding. Three local layers cut full-attention compute and KV traffic, then a global layer exchanges information across the whole context.

Be careful with the deployment claim. 4-bit text weights occupy roughly 16.8 to 19.7 GB, which sounds like a 24 GB card, but the vision encoder, KV cache, activations, and runtime buffers also need memory, and at 131K context that gap is large. Any statement that a model "runs on 24 GB" is incomplete without the quantization, the context length, and the serving stack.

### Other notable open-weight releases

- **Thinking Machines Inkling** (around July 15, 2026): the first release from Mira Murati's lab, reported at 975B parameters.
- **GLM-5.3** (Z.ai, August 14, 2026): vendor claims strong coding and high cyber capability, with sensitive functions behind trusted access. A full open release had not happened as of mid-August, so the numbers remain vendor claims.
- **GLM-5.2** (Zhipu, June 13, 2026): released under MIT with no regional limits, explicitly courting developers locked out of US models by export-control directives. A clean illustration that restrictions redirect demand rather than removing it.

### Open-Source Comparison Table

The table below provides a side-by-side comparison of the three leading open-source model families across architecture, licensing, and capability dimensions. Use it to quickly identify which model best fits a given deployment scenario -- whether the priority is reasoning strength, multilingual coverage, or licensing flexibility.

| Criterion | DeepSeek R1 | Llama 4 Maverick | Qwen 3 235B |
|----------|-------------|------------------|-------------|
| **Architecture** | MoE (671B/37B) | MoE (400B/17B) | MoE (235B/22B) |
| **Context** | 128K | 1M+ | 128K |
| **Licensing** | MIT-like | Llama License | Apache 2.0 |
| **Commercial use** | Yes | Yes, with restrictions | Yes |
| **Reasoning** | 5 of 5 | 3 of 5 | 4 of 5, hybrid |
| **Coding** | 4 of 5 | 3 of 5 | 4 of 5 |
| **Math** | 5 of 5 | 3 of 5 | 4 of 5 |
| **Multilingual** | 2 of 5, Chinese focus | 3 of 5 | 5 of 5, 119 languages |
| **Multimodal** | Text only (V3) | Native | Yes |

### Practical Recommendations

**For a startup with a limited budget:** start with Qwen 3 in a 14B or 32B configuration: good balance between quality and resources. Use hybrid reasoning mode to avoid needing separate models. For pure reasoning tasks, DeepSeek V4-Flash (284B/13B active, MIT license) or R1 distilled variants (7B-70B) deliver outstanding performance.

**For enterprise self-hosting:** Llama 4 Maverick or Scout are the optimal choices thanks to the mature ecosystem, strong tooling (vLLM, TGI), and clear licensing. Scout's 10M context window eliminates many RAG complexity needs. For reasoning-heavy workloads, DeepSeek V4 folds reasoning into a unified model (no need for separate R1).

**For research:** DeepSeek R1 remains of interest for studying GRPO and RL-based reasoning emergence (the R1 line was discontinued, but V4 inherits its innovations). Qwen 3's hybrid reasoning mode offers insights into mode-switching architectures. All three families provide full model weights for analysis.

### Interview Questions: Open-Source Models

1. **Compare MoE architectures across DeepSeek V3, Llama 4, and Qwen 3. What are the trade-offs?**
   - All three use MoE, but with different expert counts and routing strategies
   - DeepSeek: 256 experts, auxiliary-loss-free balancing, MLA for KV-cache efficiency
   - Llama 4: 16-128 experts, interleaved attention, native multimodality
   - Qwen 3: traditional MoE with hybrid reasoning mode
   - More experts = finer specialization but more complex routing

2. **When is an open-source model preferable to a proprietary one?**
   - Data privacy requirements (healthcare, finance)
   - Predictable costs at high volume
   - Need for fine-tuning (especially RL-based, like GRPO)
   - Offline/edge deployment
   - Vendor independence

3. **What made DeepSeek R1 significant for the field?**
   - Demonstrated reasoning can emerge from pure RL (GRPO) without supervised reasoning traces
   - R1-Zero showed reasoning emergence without any SFT
   - Open-weight release enabled distillation into smaller models
   - Training cost transparency challenged industry assumptions

---

## API Structure: Common Principles

Despite differences between providers, most LLM APIs follow similar principles. Understanding this common structure allows you to switch between providers easily and write provider-agnostic code.

### Message Format

Virtually all modern APIs use a "messages" format, an array of messages with roles:

- **system** (or **developer** in some reasoning model APIs), instructions for the model defining its behavior, style, and constraints
- **user**: user messages
- **assistant**: previous model responses
- **tool**: results of tool invocations

This format reflects the conversation structure and allows the model to understand dialog context. The system prompt is typically placed first and defines the assistant's "persona."

### Core Request Parameters

A typical API request includes:
- **model**: model identifier (e.g., "gpt-5.6", "claude-opus-5", "gemini-3.7-flash"). Provider IDs change with every release, so read this one from configuration rather than hard-coding it
- **messages**: array of messages
- **temperature**: degree of randomness in responses (0-2; note: some reasoning models fix this)
- **max_tokens**: maximum response length
- **top_p**: nucleus sampling
- **stop**: character sequences that trigger generation stop
- **tools**: descriptions of available tools (for agents)
- **reasoning_effort**: for reasoning models: low/medium/high (OpenAI) or budget_tokens (Anthropic)

### Response Format

The response typically contains:
- **content**: generated text
- **usage**: token usage statistics (input, output, total; reasoning models also report thinking tokens)
- **finish_reason**: reason for stopping (stop, length, tool_calls, content_filter)
- **tool_calls**: requested tool invocations (if the model decided to use a tool)

---

## Pricing Comparison

Cost is a critical factor when choosing a provider and model. All providers charge by token count, with input and output tokens priced differently. Prices are quoted **per 1M tokens** (the industry convention since 2025).

### Pricing Tiers (approximate, August 2026)

**Reasoning models:**

Reasoning models carry a premium because thinking tokens are billed at output rates and can represent 80-90% of total output for complex queries. The table below lists current pricing for the major reasoning-capable models. This enables direct cost comparison when selecting between providers for reasoning-heavy workloads.

| Model | Input | Output | Notes |
|--------|-------|--------|-------|
| Claude Fable 5 | $10.00 | $50.00 | Mythos-class, 95.0% SWE-bench Verified, 80.3% Pro |
| Claude Opus 5 | $5.00 | $25.00 | About 96% SWE-bench Verified, top of a saturated cluster |
| GPT-5.6 Sol | $5.00 | $30.00 | Frontier tier for coding, science, cyber |
| GPT-5.5 (full API) | $5.00 | $30.00 | Pro config: $30/$180 |
| o3 | $2.00 | $8.00 | 80% price cut since launch (June 2025) |
| o4-mini | $1.10 | $4.40 | Cost-efficient reasoning |
| Claude Opus 4.8 (extended thinking) | $5.00 | $25.00 | Superseded by Opus 5, 88.6% Verified |
| Gemini 3.1 Pro Preview (thinking) | $2.00 | $12.00 | Google's most capable published reasoning model |
| Gemini 2.5 Pro (thinking) | $1.25-2.50 | $10.00-15.00 | Tiered by context length (<=/>200K) |

The ranking moved twice in two months. Claude Fable 5 took SWE-bench Verified at 95.0% in June, then Claude Opus 5 passed it at about 96% in July at half the price, with GPT-5.6 Sol at roughly 96.2% and the open-weight DeepSeek V4 Pro at about 96.4% on vals.ai. Six models now sit inside two points of each other, which means the order depends on which harness you read. Treat Verified as saturated and read Module 11 before quoting a SWE-bench Pro number, because that benchmark reports three different "best" scores depending on the harness.

**Standard models (as of August 2026):**

Standard models cover the broadest range of use cases, from general-purpose chat to code generation and document analysis. Prices vary by an order of magnitude, reflecting differences in model size, capability, and provider strategy.

| Model | Input | Output |
|--------|-------|--------|
| Claude Fable 5 | $10.00 | $50.00 |
| Claude Opus 5 | $5.00 | $25.00 |
| GPT-5.6 Sol | $5.00 | $30.00 |
| Claude Sonnet 5 | $2.00 promo / $3.00 from Sept 1 | $10.00 promo / $15.00 from Sept 1 |
| GPT-5.6 Terra | $2.50 | $15.00 |
| GPT-5.5 Instant | $2.50 | $10.00 |
| GPT-5 | $1.25 | $10.00 |
| Gemini 3.7 Flash | $0.75 to Dec 31, then $1.50 | $3.75 to Dec 31, then $7.50 |
| Gemini 3.5 Flash | $1.50 | $9.00 |
| Gemini 3.1 Pro Preview | $2.00 | $12.00 |
| Gemini 3 Flash | $0.50 | $3.00 |
| Grok 4.6 | $2.00 below 200K, doubles above | $6.00 below 200K, doubles above |
| Mistral Large 3 | $2.00 | $6.00 |
| Mistral Medium 3.5 | $1.50 | $7.50 |
| Qwen 3.7 Max/Plus | $2.50 | $7.50 |
| Grok Build 0.1 | $1.00 | $2.00 |
| DeepSeek V4-Pro | $0.435 | $0.87 |
| DeepSeek V4-Flash | $0.14 | $0.28 |
| GPT-5.6 Luna | $0.20 | $1.20 |

Four entries in this table carry an expiry date rather than a price: Sonnet 5 on September 1, 2026, Gemini 3.7 Flash on January 1, 2027, Grok 4.6 above 200K tokens, and GPT-5.6 Luna after an 80% cut three weeks into general availability. Any cost model built on a single quoted rate has a shelf life measured in weeks. Build the rate into configuration, not into a spreadsheet.

**Budget models:**

Budget models are designed for high-volume, lower-complexity tasks where cost and latency matter more than peak reasoning quality. They are typically 10-50x cheaper than frontier models. This makes them the default choice for classification, extraction, simple Q&A, and routing layers within agentic systems.

| Model | Input | Output |
|--------|-------|--------|
| GPT-5-mini | $0.25 | $2.00 |
| GPT-5-nano | $0.05 | $0.40 |
| GPT-4o-mini | $0.15 | $0.60 |
| Claude Haiku 4.5 | $0.25 | $1.25 |
| Gemini 2.5 Flash | $0.15 | $0.60 |
| Gemini 2.0 Flash *(retired)* | $0.10 | $0.40 |
| Mistral Small 3 | $0.10 | $0.30 |

**Note:** Reasoning model costs can be deceptive. A query that costs $0.01 with a standard model might cost $0.10-0.50 with a reasoning model due to thinking tokens. Always account for thinking token overhead in cost estimates.

### Factors Affecting Actual Cost

**Input/output ratio:** Output tokens are typically 4-6x more expensive than input tokens. For tasks with long prompts and short responses (classification, data extraction), this works in your favor. For generating long texts, output becomes the primary expense.

**Thinking token overhead:** Reasoning models may generate 5-50x more thinking tokens than visible output tokens. A 100-token answer might require 5,000 thinking tokens. The `reasoning_effort` parameter helps control this.

**Prompt caching:** OpenAI, Anthropic, and Google offer discounts when the same prefixes are reused. An agent's system prompt repeats in every request, caching can save 50-90% on its cost.

### Prompt Caching: How It Works

Prompt caching is a mechanism where the provider saves the processing results (KV-cache) of the initial portion of the prompt and reuses them in subsequent requests.

**How it works:**
1. On the first request, the model computes the KV-cache for the entire prompt
2. The provider saves the KV-cache for the repeating **prefix** (beginning of the prompt)
3. On the next request with the same prefix, cache hit, only the new portion is computed
4. TTL is typically 5-10 minutes, after that the cache is invalidated

**Conditions for a cache hit:**
- The prefix must be **byte-identical**
- System prompt + beginning of messages must match
- Minimum prefix length: 1024-2048 tokens (depends on the provider)

**Savings by provider:**

The economics of prompt caching vary across providers. Anthropic offers the steepest discount (90%) but with a shorter TTL, while OpenAI provides a more moderate 50% discount. The minimum prefix length determines how long a system prompt must be before caching provides any benefit.

| Provider | Cache hit discount | Min prefix length | TTL |
|-----------|-------------------|-------------------|-----|
| OpenAI | 50% | 1024 tokens | 5-10 min |
| Anthropic | 90% | 1024 tokens | 5 min |
| Google | Varies | 32K tokens | - |

**Optimal prompt structure for caching:**

The ideal prompt organization for maximum caching efficiency consists of four layers. At the top is the system prompt, a long, unchanging block of instructions that is fully cached. Next come few-shot examples, multiple request-response examples that teach the model the desired format, also fully cached. Then comes the context: documents or RAG results that may be partially cached if they repeat across requests. Finally, the user message: unique per request and never cached.

**When prompt caching is effective:**
- Long system prompt (>2000 tokens) with detailed instructions
- Many few-shot examples in each request
- Shared context reused across requests
- High frequency of similar requests (>1 req/min)

**When it is NOT effective:**
- Short prompts (<1024 tokens)
- Highly varying context (RAG with different documents)
- Infrequent requests (cache expires)

### Batching API: Architecture and Economics

**Batching:** Deferred processing of requests in batches (not real-time) often yields a 50% discount. Suitable for analytical tasks, document processing, and content generation.

**Batching API architecture:**

The batch processing architecture follows an asynchronous pattern: the client submits requests to a Batch Queue, from which they enter the Batch Processor, which executes them at low priority. Results are saved in Results Storage, and the client is notified either via periodic Status Polling or through webhook/callback notifications upon completion.

**How it works (using OpenAI as an example):**
1. **Submit batch**: you send a JSONL file with requests (up to 50,000 requests)
2. **Queue**: requests enter a low-priority queue
3. **Processing**: execution within 24 hours (SLA)
4. **Results**: you retrieve results via polling or webhook

**Batching economics:**
- 50% discount on all tokens (input and output)
- Trade-off: latency (up to 24 hours) for cost reduction
- Minimum volume: typically none, but overhead makes sense at >100 requests

**Use cases for batching:**
- Daily processing of accumulated documents
- Generating embeddings for a large collection
- A/B testing prompts on historical data
- Evaluating model quality on a test set
- Content moderation backlog

**Client-side code architecture:**

A proper client-side batching implementation follows a specific pattern. First, requests accumulate in a buffer until a threshold is reached. This can be either a certain number of requests or a time interval. When the threshold is met, a batch is formed and submitted to the API, which returns a unique batch identifier (batch_id). The client then either periodically polls the processing status every few minutes or awaits a webhook notification. When processing completes, results are retrieved from storage and used to update downstream systems.

---

## Choosing a Model for Different Tasks

Different tasks require different models. Using a reasoning model for simple classification wastes money and adds latency. Using a budget model for complex reasoning produces unreliable results.

### Complex Reasoning and Analysis

**Recommended models:**
- Claude Opus 5 (about 96% SWE-bench Verified, at half the Fable 5 price)
- Claude Fable 5 (Mythos-class, strongest on multi-day agentic sessions)
- GPT-5.6 Sol (frontier tier for science and long-horizon work)
- o3 / o4-mini (OpenAI reasoning)
- Gemini 3.1 Pro Preview or 2.5 Pro with thinking (Google)

**Typical tasks:** multi-step planning, complex document analysis, mathematical proofs, scientific reasoning, architectural decisions, research tasks.

### Code Generation

**Recommended models:**
- Claude Opus 5 (about 96% SWE-bench Verified)
- GPT-5.6 Sol (about 96.2% Verified, roughly 54% more token-efficient on coding than GPT-5.5)
- Claude Fable 5 (95.0% Verified, 80.3% Pro)
- Claude Sonnet 5 (63.2% SWE-bench Pro at a fraction of flagship cost)
- DeepSeek V4 Pro (about 96.4% Verified per vals.ai, open weights)
- Kimi K3 (first on Arena.ai Frontend Code Arena, about 93.4% Verified, custom license)
- Gemini 3.7 Flash (43.6 FrontierCode, cheapest of the current frontier-tier options)
- GPT-5.3-Codex (specialized agentic coding models)

**Typical tasks:** writing new code, refactoring, debugging, writing tests, code review, documentation.

### Simple Tasks and Classification

**Recommended models:**
- GPT-5-mini / GPT-5-nano
- GPT-4o-mini
- Claude Haiku 4.5
- Gemini 2.5 Flash

**Typical tasks:** text categorization, entity extraction, sentiment analysis, simple format conversions, filtering.

### Long Documents

**Recommended models:**
- Gemini 3.7 Flash (1,048,576 tokens, 65,536 max output)
- Claude Opus 5 and Sonnet 5 (1M tokens each)
- Grok 4.6 (500K, but note the price doubles above 200K)
- GPT-5.5 / GPT-5.6 (128K-1M context)

**Typical tasks:** analyzing books, lengthy reports, codebases, legal documents, scientific papers.

### Real-Time Applications

**Recommended models:**
- Gemini 3 Flash / 2.5 Flash
- GPT-5-mini
- Claude Haiku 4.5

**Typical tasks:** chatbots with instant responses, autocomplete, interactive assistants.

---

## Tool Use / Function Calling

The ability to invoke external functions is a key capability for AI agents. It transforms an LLM from a text generator into an interactive assistant capable of interacting with the real world.

### How It Works

1. **Tool descriptions:** You provide the model with a JSON schema of available tools: their names, descriptions, and parameters
2. **Request analysis:** The model analyzes the user's request and decides whether a tool is needed
3. **Call generation:** If a tool is needed, the model generates structured JSON with the function name and arguments
4. **Execution:** Your code executes the function and obtains the result
5. **Continuation:** The result is returned to the model, which uses it to form the final response

### Comparing Provider Implementations

**OpenAI:** The most mature implementation. Support for parallel calls (multiple tools at once), strict mode with guaranteed JSON schema, excellent documentation.

**Anthropic:** Compatible format, high reliability in selecting the correct tool. Claude is particularly strong in complex scenarios with many tools. MCP (Model Context Protocol) provides a standardized way to expose tools.

**Google:** Full-featured function calling support, integration with Google Cloud Functions for serverless execution.

**Mistral:** Tool support exists, but the implementation is less mature compared to the leaders.

### Structured Outputs

An important capability: guaranteed conformance of the response to a JSON schema. OpenAI implemented this through constrained decoding: at the token generation level, the model is physically unable to produce invalid JSON. Anthropic and Google offer similar capabilities.

This is critical for agents, where parsing reliability must be 100%. Previously, you had to handle parsing errors and retry, now that is unnecessary.

---

## Reliability Strategies

Production systems cannot depend on a single provider. APIs go down, rate limits are hit, and models become temporarily unavailable. Reliability requires a well-designed architecture.

### Multi-Provider Architecture

A good practice is to have a fallback to an alternative provider:

1. **Primary request** to the preferred provider (e.g., Anthropic)
2. **On failure**: automatic switch to the fallback (OpenAI)
3. **Monitoring** of success rates for each provider
4. **Automatic rotation** when the primary degrades

Important: models from different providers are not identical. The same prompt may behave differently. Test your fallback scenarios.

### Handling Rate Limits

All providers have limits on requests per minute and tokens per minute. Working strategies:

- **Exponential backoff** on receiving 429 errors: wait and retry with increasing intervals
- **Request queue** with rate control: do not send more than N requests per second
- **Load distribution** across multiple API keys
- **Respect the Retry-After header**: if the API says wait 30 seconds, wait 30 seconds

### Timeouts and Retries

LLM requests can take from one second to a minute (reasoning models can take several minutes). Rules:

- **Reasonable timeouts:** 30-60 seconds for standard generation, 10-20 seconds for classification, 2-5 minutes for reasoning models
- **Retry with exponential backoff:** start at 1 second, double up to a maximum
- **Jitter:** add randomness to the delay to avoid thundering herd
- **Circuit breaker:** after N consecutive errors, stop trying and return an error immediately

### SLA and Reliability Engineering for LLM APIs

When designing production systems, understanding the actual provider guarantees and planning accordingly is critically important.

**Typical SLA metrics:**

When designing production systems, it is important to understand what providers actually guarantee versus what they typically deliver. The table below lists the core SLA dimensions along with representative values observed across major providers. Note that reasoning models have higher latency percentiles than standard models.

| Metric | Description | Typical Values |
|---------|----------|-------------------|
| **Availability** | Service uptime | 99.9% - 99.99% |
| **Latency P50** | Median response time | 1-3 sec (standard), 5-30 sec (reasoning) |
| **Latency P95** | 95th percentile | 5-15 sec (standard), 30-120 sec (reasoning) |
| **Latency P99** | 99th percentile | 15-60 sec |
| **Error rate** | Percentage of 5xx errors | <0.1% |

**Latency percentiles: why they matter:**

Average latency is deceptive. If P50 = 2 sec but P99 = 30 sec, every hundredth user waits half a minute. For interactive applications, this is unacceptable.

A typical latency distribution looks like this: P50 is approximately 2.1 seconds, meaning half of all requests complete faster. P75 increases to 3.5 seconds, P90 to 6.2 seconds, P95 to 9.8 seconds. Critical values are P99 at 24.3 seconds (every hundredth request) and P99.9 at 58 seconds (every thousandth request). These distribution "tails" determine the worst-case user experience.

**Capacity planning with latency in mind:**

System throughput is calculated by the formula: Throughput = Concurrency / Latency. For example, if you need to handle 100 requests per second with a median latency of 2 seconds, you need 200 concurrent connections. However, to handle the distribution tail at P99 latency of 20 seconds, a buffer of 2000 connections is needed to prevent system degradation under peak loads.

**SLA monitoring:**

What to track:
1. **Latency distribution**: not just average, but P50/P95/P99
2. **Error rate by type**: 429 (rate limit), 500 (server), 503 (overload)
3. **Token throughput**: tokens/second for capacity planning
4. **Cost per request**: for budget alerting
5. **Thinking token ratio**: for reasoning models, monitor thinking-to-output ratio

**Degradation strategies:**

A well-designed system defines explicit strategies for each load level, progressing from normal operation through graceful degradation. The table below outlines a four-tier approach that balances user experience against system stability as load increases.

| Load Level | Strategy |
|------------------|-----------|
| Normal | All requests to primary provider |
| High | Shedding low-priority requests |
| Overload | Fallback to faster/cheaper model |
| Emergency | Graceful degradation (cached responses, simplified prompts) |

**Practical recommendations:**

1. **Never rely on a single provider**: even with 99.9% SLA, that is 8.7 hours of downtime per year
2. **Measure your own metrics**: provider SLA ≠ your user experience
3. **Plan for burst capacity**: LLM load is often spiky (start of business day)
4. **Budget for latency**: if P99 = 30s, your timeout should be >30s

---

## Integration via LangChain4j

LangChain4j is a popular Java framework that abstracts working with different LLM providers. It provides a unified interface, simplifying switching between models.

### Benefits of a Unified Interface

Instead of learning each provider's API separately, you work with a single `ChatLanguageModel` interface. Switching between providers means changing a single configuration line. The code remains the same.

### Supported Providers

LangChain4j supports out of the box:
- OpenAI and Azure OpenAI
- Anthropic
- Google Vertex AI and Google AI
- Mistral
- Amazon Bedrock
- Ollama (local models)
- HuggingFace
- Cohere
- And many others

### Additional Framework Capabilities

- **Automatic retry** and rate limiting
- **Moderators** for content filtering
- **Response caching** out of the box
- **Integration with vector stores** for RAG
- **AI Services**: a declarative way to create agents through interfaces

---

## Local Models

Cloud providers are not always the right choice. In some cases, data cannot be sent to external APIs, full control over infrastructure is needed, or predictable costs are required.

### When to Consider Local Models

- **Strict data privacy:** medical data, financial documents, government secrets
- **Predictable cost:** fixed infrastructure expenses instead of per-token billing
- **Offline scenarios:** edge computing, working without internet
- **Customization:** full fine-tuning (including RL-based methods like GRPO) on proprietary data
- **Independence:** avoiding vendor lock-in and dependency on external services

### Popular Solutions for Self-Hosting

**Ollama**: the simplest way to run models locally. One-command installation, support for Llama, Mistral, Qwen, DeepSeek, and other open-weight models. Ideal for development and testing.

**vLLM**: a high-performance inference server for production. Continuous batching, speculative decoding, PagedAttention for efficient GPU memory utilization. OpenAI-compatible API.

**SGLang**: optimized for multi-turn conversations via RadixAttention. Excellent for chatbots and agents. Strong structured output support.

**Text Generation Inference (TGI)**: a production-ready solution from Hugging Face. Good documentation, enterprise features, native HF ecosystem integration.

### Hardware Requirements

Modern LLMs require significant resources. The table below maps model size to GPU memory requirements in both full precision (FP16) and quantized (INT4) formats, along with representative models in each category. Quantization reduces requirements by 2-4x. This makes larger models accessible on consumer hardware at a small quality cost.

| Model Size | GPU RAM (FP16) | GPU RAM (INT4) | Examples |
|---------------|---------|---------|---------|
| 7-8B | 16 GB | 4-6 GB | Llama 4 8B, Qwen 3 8B, Mistral 7B |
| 14B | 28 GB | 8-10 GB | Qwen 3 14B |
| 32B | 64 GB | 16-20 GB | Qwen 3 32B |
| 70B | 140 GB | 35-40 GB | Llama 3.3 70B, Qwen 3 72B, DeepSeek R1 70B distill |
| MoE 100B+ active | 200+ GB | 80+ GB | Llama 4 Maverick, Qwen 3 235B |

**Quantization** reduces requirements by 2-4x with a small quality loss. 4-bit quantization (GPTQ, AWQ, GGUF) allows running a 70B model on a single 24GB GPU.

---

## Key Takeaways

1. **The LLM provider market is diverse and fast-moving.** OpenAI, Anthropic, Google, xAI, Mistral, DeepSeek, Meta. Each has its own strengths. There is no universally "best" provider.

2. **Reasoning models are a new paradigm.** o3, Claude extended thinking, Gemini thinking: they trade latency and cost for dramatically better performance on complex tasks. Use them selectively.

3. **Model choice depends on the task.** Complex reasoning requires reasoning models; standard tasks require standard models; simple tasks require budget models. The right routing strategy saves 70%+ on costs.

4. **Cost is determined by tokens.** Output tokens are 4-6x more expensive than input. Thinking tokens in reasoning models can dominate costs. Caching, batching, and routing can reduce costs by several times.

5. **Open weights lead on some benchmarks, not just match.** DeepSeek V4 Pro scores about 96.4% on SWE-bench Verified and Kimi K3 took first place on Arena.ai's Frontend Code Arena, both ahead of Claude Opus 4.8. Two caveats travel with that: licenses have diverged from MIT and Apache 2.0 toward custom terms, and serving capacity limits real availability, as Moonshot showed by pausing Kimi K3 signups when demand outran compute.

6. **A model's tier no longer predicts its ranking.** Claude Opus 5 outscores Claude Fable 5 on SWE-bench Verified from a tier below it, at half the price, and two open-weight models sit in the same cluster. Read the current numbers rather than the price list, and check which harness produced them.

7. **Quoted prices expire.** GPT-5.6 Luna dropped 80% three weeks after general availability, Sonnet 5 promotional pricing ends September 1, 2026, Gemini 3.7 Flash rises January 1, 2027, and Grok 4.6 doubles above 200K tokens. Put rates in configuration, not in a spreadsheet.

8. **Model availability is now a policy variable.** Claude Fable 5 was disabled worldwide for 19 days by an export-control directive, and GPT-5.6 shipped only after a government review cleared it. A single-provider design has a failure mode that has nothing to do with uptime.

9. **Tool use / function calling** is the standard for agents. All major providers support this capability with compatible formats.

10. **Reliability requires a multi-provider approach.** Fallback, retry with backoff, circuit breakers, mandatory elements of a production system.

11. **Local models** are suitable for specific requirements: privacy, predictable cost, offline operation, full fine-tuning control. They require infrastructure investment.

---

## Brief Code Example

### Basic Integration via LangChain4j

Working with different providers through LangChain4j is unified thanks to the shared `ChatLanguageModel` interface. To create an OpenAI client, use `OpenAiChatModel.builder()` specifying the API key, model name (e.g., "gpt-5.6"), temperature, and maximum token count. Anthropic clients are created similarly via `AnthropicChatModel.builder()` with model "claude-sonnet-4-6-20250923", Google via `GoogleAiGeminiChatModel.builder()` with model "gemini-2.5-pro", and even local models via `OllamaChatModel.builder()` specifying the URL and a model such as "llama4:maverick".

**Multi-provider architecture with fallback:** To ensure reliability, a client is created that maintains a list of providers in priority order. When generating a response, it sequentially attempts each provider starting with the first. If a provider returns an error (e.g., rate limit or server error), the client automatically switches to the next one. To handle transient failures, a retry mechanism with exponential backoff is implemented: the first attempt after 1 second, the second after 2 seconds, the third after 4 seconds, with random jitter added to avoid synchronized retry storms. Only after exhausting all attempts across all providers is a final error thrown.

**Routing by task type:** An efficient architecture routes requests to optimal models depending on task complexity. For simple tasks like classification or data extraction, economical models such as Claude Haiku 4.5 or GPT-5.6 Luna are used. For complex reasoning and analysis, reasoning models like Claude Opus 5 or GPT-5.6 Sol with extended thinking are applied. Code generation works best with Claude Sonnet 4.6 or GPT-5.3-Codex, which lead coding benchmarks. For real-time applications where speed is critical, the fastest models such as Claude Haiku 4.5 or Gemini 2.5 Flash are selected.

**Tool Use via AI Services:** LangChain4j provides a declarative way to create agents with tools. Tools are defined as regular Java methods annotated with `@Tool`, which contains the function description. Method parameters are annotated with `@P` including a description for the model. Then an agent interface is created with a `chat(String message)` method, and `AiServices.builder()` is used to automatically generate the implementation. The model independently analyzes the user's request, decides which tools are needed, invokes them with the correct parameters, and forms the final response based on the results.

**Structured Output:** To guarantee structured data output, the `responseFormat("json_schema")` mode is used with OpenAI models. A Java record is defined with the required fields (e.g., Product with fields name, category, price, features), an interface is created with an extraction method annotated with `@UserMessage`, and AI Services automatically handles parsing the response into a typed object. Thanks to constrained decoding, the model is physically unable to generate invalid JSON. This gives 100% parsing reliability.

**Cost calculation:** Monitoring LLM API expenses is critically important. Cost is calculated by the formula: (inputTokens / 1,000,000) x inputPrice + (outputTokens / 1,000,000) x outputPrice. For reasoning models, add thinking tokens: (thinkingTokens / 1,000,000) x outputPrice. For a typical workload, monthly expenses can be projected by multiplying the cost per request by the number of requests per day and by 30 days. For example, at 10,000 requests per day with an average of 500 input / 200 output tokens, GPT-5.6 would cost approximately $1,275/month (input: 300K requests x 500 tokens / 1M x $2.50 = $375; output: 300K x 200 / 1M x $15 = $900), whereas GPT-5-mini would be only about $158/month (input: $37.50, output: $120): an 8x difference with comparable quality for simple tasks. A single complex reasoning query with o3 might use 5,000 thinking tokens, costing $0.20 per request. This makes selective routing essential.
