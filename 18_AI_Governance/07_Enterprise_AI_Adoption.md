# Enterprise AI Adoption: CTO's Guide

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Governance
**Previous:** [[06_Alignment_Research|Alignment Research]]
**Next:** [[../19_Practical_Projects/01_RAG_Chatbot|RAG Chatbot]]

---

## Introduction: The Reality of AI Transformation

While the previous chapters of this course were technical, this chapter is strategic. It is addressed to those who make decisions about AI adoption in organizations: CTOs, VPs of Engineering, technical directors. Here we will discuss what happens in practice when an enterprise begins AI transformation, what problems arise, and how to avoid them.

The 2025 statistics are unambiguous: AI is no longer an experimental technology. Over 80% of companies actively use AI in one form or another. However, another statistic is equally unambiguous: only 25% of AI initiatives achieve the expected ROI. The gap between "using AI" and "deriving business value from AI" is enormous.

Gartner predicts that 40% of agentic AI projects will fail by 2027 due to insufficient governance and lack of clear management processes. This does not mean that agentic AI does not work — it means that organizations underestimate the complexity of adoption.

---

## Key Problems of Enterprise AI Adoption

### Governance & Security: The Blind Spot

According to research, 75% of technical leaders name governance and security as their top concerns when adopting AI. At the same time, only 6% of organizations have a formalized AI security strategy.

**Why this happens:**

- AI evolves faster than security practices
- There are no established standards for AI security
- Traditional security frameworks do not cover AI-specific risks
- "Let's launch first, then add security"

**Specific risks:**

| Risk | Description | Example |
|------|-------------|---------|
| **Prompt Injection** | Attacks through user input | A user forces an agent to ignore its instructions |
| **Data Leakage** | Leakage of sensitive data | RAG returns documents the user has no access to |
| **Model Extraction** | Extracting model knowledge | A competitor reconstructs proprietary knowledge |
| **Supply Chain** | Vulnerabilities in dependencies | Compromised embedding model |

**Governance-first approach:**

1. Define the organization's AI risk appetite before starting adoption
2. Create an AI governance board with representatives from security, legal, and data
3. Develop a classification for AI use cases (low/medium/high risk)
4. Implement mandatory review for high-risk applications

### Compliance & Regulatory: Mounting Pressure

The EU AI Act has come into effect and establishes strict requirements for high-risk AI systems. HIPAA, GDPR, and SOC 2 require a specific approach to AI.

**EU AI Act: key categories:**

| Category | Description | Requirements |
|----------|-------------|--------------|
| **Minimal Risk** | Spam filters, games | No requirements |
| **Limited Risk** | Chatbots | Transparency (the user must know they are communicating with AI) |
| **High Risk** | HR, credit scoring, healthcare | Conformity assessment, documentation, human oversight |
| **Unacceptable** | Social scoring, emotion recognition at work | Prohibited |

**Practical steps:**

- Conduct an AI use case inventory — what AI systems are already in use?
- Classify each according to the EU AI Act risk level
- For high-risk systems — begin preparing documentation now
- Appoint a person responsible for AI compliance

### ROI Measurement: The Hardest Challenge

49% of technical leaders name measuring AI ROI as the most difficult part of adoption. This is no coincidence:

**Why AI ROI is hard to measure:**

- Value is distributed over time (investments now, returns later)
- Many benefits are qualitative (better user experience, faster decisions)
- Attribution problem: it is difficult to isolate the AI effect from other factors
- Some use cases create new value rather than optimizing existing value

**Framework for ROI:**

To effectively measure the return on investment from AI adoption, metrics need to be structured across four levels of measurement difficulty:

**Level one — direct cost savings (easiest to measure):** This accounts for the reduction of time spent on specific tasks, which is easily converted to money through the hourly rate of employees. Automation of manual labor is measured in full-time equivalent (FTE), allowing a direct calculation of saved salary expenses. The reduction in errors is multiplied by the cost of each error and the percentage of reduction.

**Level two — productivity gains (measurable with moderate effort):** The focus is on increasing output per employee, which can be tracked through internal productivity metrics. Reduction in time-to-market is measured in days or weeks and directly impacts competitiveness. Faster onboarding of new employees is reflected in reduced time to full productivity.

**Level three — quality improvements (requires indirect indicators):** This level uses customer satisfaction scores, the number of support escalations, and for development tools — data from code reviews on the number of issues found and their severity.

**Level four — strategic value (hardest to quantify):** This includes competitive advantage in the market, the ability to create fundamentally new products and services, and the impact on retention and attraction of talented employees. These indicators are often assessed qualitatively or through long-term proxy metrics.

**Practical recommendations:**

- Start with use cases where ROI is easy to measure (categories 1-2)
- Establish baseline metrics BEFORE adoption
- Define success criteria in advance — how will we know it works?
- Create a dashboard for tracking AI metrics

### Data Quality: The Root of Problems

62% of organizations name data quality as the main obstacle to successful AI adoption. Garbage in — garbage out, and for AI this is especially true.

**Typical data problems:**

- Disparate sources without a unified schema
- Inconsistent formats and naming conventions
- Stale data (especially critical for RAG)
- Lack of metadata and lineage

**Data readiness checklist:**

| Aspect | Question | Green Light |
|--------|----------|-------------|
| **Accessibility** | Is the data programmatically accessible? | API, SQL, file export |
| **Quality** | Are there data quality checks? | Automated validation |
| **Freshness** | How often is the data updated? | Known schedule |
| **Completeness** | Are there many missing values? | <5% missing |
| **Documentation** | Is there a data dictionary? | All fields described |
| **Ownership** | Who is responsible for the data? | Assigned data steward |

### Talent Gap: Humans in the Loop

Statistics reveal a paradox: 73% of employees use AI tools weekly, but only 29% have advanced AI literacy. People use AI without understanding its limitations and best practices.

**Levels of AI literacy:**

| Level | Description | Share of Employees |
|-------|-------------|--------------------|
| **Basic** | Uses ChatGPT for simple tasks | 40-50% |
| **Intermediate** | Understands prompt engineering, limitations | 20-30% |
| **Advanced** | Can build AI-powered workflows | 5-10% |
| **Expert** | Develops AI systems | <5% |

**Training strategy:**

1. **Awareness (everyone):** What AI is, basic capabilities and limitations
2. **Skill-up (power users):** Prompt engineering, tool-specific training
3. **Deep dive (builders):** RAG, agents, fine-tuning, evaluation
4. **Continuous (all levels):** Regular updates on new capabilities

---

## AI Adoption Strategy

### Governance-First Approach

The traditional approach: experiments first, governance later. This leads to "shadow AI" — uncontrolled use of AI tools.

**Governance-first means:**

1. **Inventory:** What AI is already being used in the organization?
2. **Policy:** Which use cases are approved, and which are not?
3. **Process:** How does one get approval for a new AI use case?
4. **Monitoring:** How do we track usage and incidents?

**AI Governance Maturity Model:**

Organizations progress through four main maturity levels in AI governance:

**Level 0 — Chaotic Usage (AD-HOC):** At this stage, there is no formalized AI policy. Each employee or team makes decisions individually, without coordination. Leadership has no visibility into which AI tools are being used in the company, creating serious security and compliance risks.

**Level 1 — Basic Awareness (AWARE):** The company creates an initial version of an AI policy with high-level general recommendations. Partial visibility into AI tool usage emerges, usually through surveys or voluntary reporting. However, no formal approval process exists yet.

**Level 2 — Managed Process (MANAGED):** A formal approval process for new AI use cases is implemented. A risk-level classification of AI applications has been developed, enabling different levels of control. Regular audits of AI system usage are conducted. This is the minimum level for enterprise organizations working with sensitive data.

**Level 3 — Optimized Governance (OPTIMIZED):** An AI governance board with representatives from all key functions has been established. Automated monitoring of AI usage with real-time alerts has been implemented. Processes are continuously improved based on collected data and incidents. Risk management is proactive — potential issues are identified before they occur.

### Pilot Project: How to Choose

Choosing the first AI project is critically important. Success creates momentum; failure sets the organization back by months.

**Criteria for a good pilot:**

| Criterion | Meaning | Why It Matters |
|-----------|---------|----------------|
| **Clear ROI** | Measurable business impact | Proof of value |
| **Contained scope** | Limited scale | Manageable risk |
| **Data available** | Quality data exists | No data project needed first |
| **Champion exists** | There is an interested stakeholder | Will promote and defend it |
| **Low risk** | Errors are not critical | Can learn without catastrophes |

**Good first projects:**

- Internal knowledge search (RAG over documentation)
- Code review assistance
- Customer support FAQ automation
- Meeting summarization

**Bad first projects:**

- Customer-facing agentic systems
- High-stakes decision making
- Compliance-sensitive areas

### Scaling: From Pilot to Production

A successful pilot is only the beginning. Scaling requires a different approach.

**What changes when scaling:**

| Aspect | Pilot | Production |
|--------|-------|------------|
| **Users** | 10-50 | 1000+ |
| **Data** | Sample | Full volume |
| **SLA** | None | 99.9%+ |
| **Security** | Basic | Full review |
| **Monitoring** | Manual | Automated |
| **Support** | Build team | Dedicated ops |

**Common mistakes:**

- "The pilot worked — just add more servers" — does not work
- Underestimating ops complexity for LLM systems
- Lack of fallback for AI failures
- Ignoring edge cases (which simply did not occur during the pilot)

---

## Organizational Patterns

### Center of Excellence vs Distributed

Two main approaches to organizing AI capabilities:

**Centralized approach — Center of Excellence (CoE):**

In this approach, a dedicated team of AI experts is created to serve requests from all other teams in the organization. All AI projects go through this central team.

Advantages of centralization: rare expertise is concentrated in one place, enabling more efficient use of expensive specialists. Consistency in approaches, standards, and tools is ensured — all projects are built using the same methodology, shared libraries, and patterns.

Disadvantages of centralization: the central team quickly becomes a bottleneck — the number of requests grows faster than experts can be hired. The CoE team is far removed from the business context of specific products, leading to a lack of understanding of specifics and priorities.

**Distributed approach — Embedded AI Engineers:**

AI specialists are integrated directly into product teams. Each team has its own AI engineer or shares one with 1-2 other teams. A Community of Practice is created for coordination and knowledge sharing, meeting regularly.

Advantages of distribution: AI specialists are immersed in the business context and understand product specifics. There is no bottleneck — teams can move independently at the speed they need.

Disadvantages of distribution: duplication of effort occurs — different teams solve similar problems independently. Inconsistent practices emerge — each team may choose its own technology stack, its own approaches to evaluation and monitoring.

**Recommendation:** Start with a CoE to build the foundation (standards, tools, training), then transition to a distributed model with a community of practice for coordination. This allows you to get the best of both approaches — unified standards with decentralized execution.

### AI Product Manager: A New Role

AI projects require specific product management. A traditional PM may not understand the specifics of AI.

**AI PM differences:**

- Works with probabilistic outputs (not deterministic features)
- Understands evaluation metrics (precision/recall, not just NPS)
- Knows about data requirements and pipeline complexity
- Manages expectations (AI ≠ magic)

---

## Key Takeaways

1. **80%+ of companies use AI, but only 25% achieve the expected ROI** — the gap between adoption and value extraction is enormous

2. **Governance-first, not governance-later** — start with policy and processes, then scale

3. **Data quality is the foundation** — 62% of failures are related to data

4. **The talent gap is real** — 73% use AI, only 29% understand it

5. **Pilot ≠ production** — scaling requires a different approach

6. **Measure ROI from day one** — baseline metrics before adoption

7. **88% of agent projects fail** (average cost $340K) — top cause is integration (46%), not AI capability. Scope creep and data quality account for 61%

8. **The hybrid pattern is the real standard** — Klarna's pivot from 100% automation to AI+human demonstrates that full automation is not always optimal

9. **Enterprise platform choice follows ecosystem** — Salesforce shops use Agentforce, AWS shops use Bedrock AgentCore, Microsoft shops use Copilot Studio. Technical merit is secondary to existing infrastructure

10. **Agent washing is now a legal risk** — only ~130 of thousands of vendors are genuinely agentic. Exaggerated claims create disclosure liability

11. **Start small, prove value, scale gradually** — do not try to transform everything at once

---

## Checklist for CTOs

### Before launching an AI initiative:

- [ ] Is there a formal AI policy?
- [ ] Has an inventory of existing AI usage been conducted?
- [ ] Has a risk classification for use cases been defined?
- [ ] Has a data quality assessment been performed?
- [ ] Have those responsible for AI governance been appointed?
- [ ] Have success metrics been defined?

### When choosing a pilot project:

- [ ] Is ROI measurable and significant?
- [ ] Is scope limited and manageable?
- [ ] Is data available and of high quality?
- [ ] Is there a business champion?
- [ ] Are the risks of failure acceptable?

### When scaling:

- [ ] Production-ready infrastructure?
- [ ] Monitoring and alerting configured?
- [ ] Fallback for AI failures?
- [ ] Security review completed?
- [ ] Support processes defined?
- [ ] Continuous evaluation configured?

---

## Enterprise Agent Platforms (2025-2026)

The enterprise agent platform market has consolidated into three tiers. For most enterprises, platform choice is determined by existing ecosystem — not technical merit.

### Tier 1: Horizontal Platforms

**Salesforce Agentforce** — the enterprise market leader. $800M ARR (+169% YoY), 29,000 deals, 2.4 billion agentic work units processed (as of early 2026). Pricing: $2/conversation (vs $6-12 for a human agent). **Agentforce Script** enables hybrid reasoning — deterministic steps for policy checks, discount calculations, and ticket routing run as classic business logic, while the LLM handles only the flexible parts requiring natural language understanding. Agentforce Contact Center (March 2026) combines voice, digital channels, CRM data, and AI agents, directly attacking the Five9/Genesys/NICE market. The "work unit" as a metric is significant: it is a completed business operation that a CFO can understand, not a token count.

**Microsoft Copilot Studio** — low-code agent building for business users. The strategy: agents in every Microsoft product (Outlook, Teams, SharePoint, Dynamics 365, GitHub). 500M+ M365 users provide unmatched distribution. The "AI tax" ($30/user/month for Copilot for M365) means 10,000 employees = $3.6M/year.

**Amazon Bedrock AgentCore** (March 2026) — the most infrastructure-complete offering. Nine components: Runtime, Memory (short-term + long-term + episodic), Gateway, Identity, Sandbox Code Interpreter, Cloud Browser, Observability, Evaluation, Policy. Model-agnostic (Claude, GPT-5, Gemini, Llama, Mistral). Amazon's advantage: 20 years of enterprise workload data — institutional knowledge that cannot be replicated.

### Tier 2: Vertical Platforms

**Sierra** — customer experience agents. $4.5B valuation, 40% of Fortune 50 as clients (ADT, Cigna, Nordstrom, Wayfair, Rocket Mortgage). Ghostwriter (March 2026) is an agent that creates other production-ready agents from SOPs, transcripts, and plain English in 30+ languages. Pay-per-resolution pricing aligns vendor incentives with outcomes.

**Harvey** — legal AI. Works with Magic Circle firms (Allen & Overy) and Big 4 (PwC). Contract analysis, due diligence, legal research. Represents the vertical specialization pattern: domain expertise + compliance = pricing power.

### Tier 3: Enterprise-Embedded

**ServiceNow Now Assist** — ITSM-specific agents with a decade of IT workflow data. 20-40% MTTR reduction. The competitive advantage: thousands of clients' accumulated workflow patterns are training data no startup can replicate.

**SAP Joule** — embedded in the SAP ecosystem for ERP, supply chain, and HR processes.

### Anthropic's Enterprise Play

**Claude Managed Agents** (April 2026 beta) — $0.08/session-hour plus token costs. Composable APIs for building and deploying cloud-hosted agents at scale, with infrastructure abstracted away. Self-evaluation loop improves task success rate by ~10 percentage points. Early adopters: Notion, Rakuten, Asana, Sentry. **Claude Cowork** (GA) targets knowledge work on desktops — moving between local files and applications to synthesize, assemble, and finish deliverables. Targets 500M knowledge workers (vs ~30M developers for coding tools).

### Emerging Enterprise Patterns

**Decagon AOPs** (Agent Operating Procedures) — natural language behavioral rules that control agent behavior, editable by business teams without developers. Clients: Notion, Duolingo, Substack, Rippling ($231M funding). The pattern is significant: it separates agent behavior specification (business team owns) from agent implementation (engineering team owns).

**Agent Washing** — Debevoise & Plimpton published an analysis of disclosure risks related to exaggerated agentic capabilities. This is now a legal liability, not just a marketing problem. Gartner estimates only ~130 of thousands of vendors build genuinely agentic systems. Five red flags: claims of full autonomy, no real tool use, weak error handling, no memory between tasks, vague marketing without concrete capability demonstrations.

---

## Production Case Studies

Real-world outcomes that validate or challenge the patterns taught in this course.

### Klarna: The Hybrid Pivot

Klarna's AI agent deployment is the most instructive case study in enterprise AI — both for its success and its course correction.

**The success:** In February 2024, Klarna's AI agent handled 2.3 million conversations per month — two-thirds of all customer chats, equivalent to 853 full-time agents. Resolution time dropped from 11 minutes to under 2 minutes (6x improvement). Repeat inquiries decreased by 25%. The system operated 24/7 across 23 markets in 35+ languages. Annual savings: $60M. Headcount reduced from ~5,000 to ~3,800 through natural attrition.

**The pivot:** By Q3 2025, Klarna publicly reversed course, returning to live chat 24/7 on all markets with callback for voice and seamless AI-to-human handoff. The discovery: AI handled 70-80% of interactions well, but the remaining 20-30% generated disproportionate negative customer experience. 100% automation increased churn. The lesson: the "replaced 700 agents" narrative was about volume handling, not full replacement.

**The lesson for architects:** Full automation is not always the optimal end state. The hybrid pattern — AI for volume, humans for edge cases and emotional interactions — is the real production standard. Design for graceful handoff from the start, not as an afterthought.

### Goldman Sachs + Devin

Goldman Sachs piloted Devin with 12,000 developers, reporting approximately 20% efficiency gains. At Devin's reduced price of $20/month per developer (down from $500), the ROI calculation became straightforward even for junior developers. This case demonstrates that even the most conservative enterprises adopt coding agents when the cost-benefit is measurable and the risk is bounded (code review catches agent errors).

### Air Canada: When AI Promises Become Legal Obligations

Air Canada's chatbot hallucinated an entirely fictitious bereavement fare policy — detailed, plausible, and completely wrong. A customer relied on this information and was denied the refund the chatbot had "promised." The Canadian tribunal ruled that the company is responsible for what its AI says, and a disclaimer stating the chatbot might be inaccurate does NOT protect the company. The award was small (CAD $650.88 plus interest), but the precedent is massive: agent outputs can create legally binding obligations.

**The lesson:** Guardrails are not optional. See [[../14_Security_Safety/04_Moderation_and_Compliance|Moderation and Compliance]] for compliance architecture.

### Additional Failure Cases

**DPD (January 2024):** A customer manipulated the chatbot into swearing, criticizing DPD, writing poems about how terrible DPD is, and recommending competitors. The incident went viral (BBC, Guardian, Daily Mail). Root cause: no output filtering, no adversarial prompt detection, no sentiment checking.

**Chevrolet Dealer (December 2023):** A ChatGPT-based dealer chatbot agreed to sell a 2024 Tahoe for $1 — "That's a deal, and that's a legally binding offer — no take backs." Other users got it to write Python code instead of selling cars.

**Microsoft Copilot Oversharing (2024-2025):** Copilot exposed confidential documents that users technically had access to but should never have seen: executive salaries, HR disciplinary records, M&A drafts. "Security through obscurity" collapsed the moment AI got search capability. Some organizations paused Copilot rollout pending permissions audits.

### Shopify: AI as Baseline Expectation

Shopify CEO Tobi Lütke published an internal memo (April 2025) that became a reference point for the industry: "Reflexive AI usage is now a baseline expectation at Shopify." Before requesting headcount, teams must prove AI cannot do the work. AI competency is factored into performance reviews and hiring decisions. All employees are provided Copilot, Cursor, and Claude Code. Duolingo went further, replacing some content contractors with AI entirely.

---

## Enterprise Integration Patterns

Integration with existing systems is the #1 cause of agent project failure — cited by 46% of organizations (as of early 2026). The problem is not AI capability but connecting AI to legacy infrastructure.

Five patterns address different integration scenarios, roughly ordered from most to least standard:

**1. API Gateway Layer** — the cleanest integration. Tools like MuleSoft, Apache Camel, or Kong expose internal APIs through a unified gateway. The agent calls standardized REST/GraphQL endpoints. This works when APIs already exist for the target systems.

**2. RPA + Agent Hybrid** — for systems without APIs: mainframes, green-screen terminals, legacy desktop applications. UiPath or similar RPA tools handle the UI automation, while the AI agent handles reasoning and orchestration. The RPA layer acts as a "hands" for the agent — clicking buttons, filling forms, reading screens.

**3. Database Direct Access** — Text-to-SQL agents query databases directly through read-only views. Fast and accurate for structured data. Critical: use read-only database credentials and restrict to pre-approved views. Never give an agent write access to a production database without HITL approval.

**4. Event-Driven Integration** — agents subscribe to event streams (Kafka, RabbitMQ, SQS) and react to business events. Suitable for real-time workflows: order processing, alert handling, escalation routing. The agent is triggered by events rather than user requests.

**5. Screen Scraping with Vision** — the last resort. Computer use agents (Claude Computer Use, GPT Operator) interact with applications through screenshots and mouse/keyboard actions. Useful for the most hopeless legacy systems with no API, no database access, and no event stream. See [[../03_AI_Agents_Core/06_Computer_Use_Agents|Computer Use Agents]].

**MCP-First Architecture:** Each integration pattern above can be wrapped as an MCP server, creating a unified agent-facing interface. The agent does not need to know whether it is calling a REST API, an RPA robot, or a database query — it sees MCP tools with standardized descriptions. With 6,000+ pre-built MCP integrations (as of April 2026), many common enterprise systems already have MCP servers available.

---

## GDPR and Agents: The Vector DB Problem

A frequently overlooked compliance issue: **vector DB embeddings containing personal data are personal data under GDPR.** The embedding is a mathematical transformation of the original text — but it is derived from personal data and can potentially be used to reconstruct or identify individuals. This means:

- Every LLM call with personal data constitutes "processing" requiring a legal basis under GDPR
- Article 22 mandates human oversight for "significant automated decisions" — an agent approving loans, screening CVs, or making medical recommendations triggers this requirement
- The right to erasure applies: if a user requests deletion, their data must be removed from vector stores, not just relational databases

**Practical requirements:**
- PII detection/redaction pipeline (Microsoft Presidio or equivalent) before LLM calls
- Consent management tracking which data was processed by which agent
- Data retention policies covering vector stores (embeddings expire too)
- Audit trail with masked PII — log what the agent did, not the personal data it processed
- Human-in-the-loop for decisions with material impact on individuals

**Cross-border strategy:** "Build under the EU AI Act — it covers all other jurisdictions" is increasingly the pragmatic compliance approach. EU AI Act requirements are the strictest; meeting them typically satisfies HIPAA, CCPA, and other frameworks as a side effect. See [[01_Regulatory_Landscape|AI Regulatory Landscape]] and [[../14_Security_Safety/04_Moderation_and_Compliance|Moderation and Compliance]] for regulatory details.

---

## Why Agent Projects Fail

The statistics are sobering: **88% of agent projects fail to reach production** (industry survey, early 2026). The average cost of a failed project: **$340K**. Understanding failure patterns is as important as understanding success patterns.

**Top cause: integration with existing systems (46%).** The AI works in isolation; connecting it to enterprise systems — CRM, ERP, legacy databases, identity providers — is where projects die. "Dumb RAG, Brittle Connectors, Polling Tax" — three failure modes identified by Composio: RAG that retrieves irrelevant context, integrations that break on schema changes, and agents that waste tokens polling for state instead of receiving events.

**61% of failures trace to scope creep and data quality.** The pilot works on clean demo data; production data is messy, inconsistent, and incomplete. The 80/20 rule applies brutally: getting from a working demo to 80% production coverage takes 20% of effort; the remaining 20% of edge cases consumes the other 80%.

**Budgets miss real costs by 40-60%.** Development is only 25-35% of three-year TCO — operations (inference, monitoring, maintenance, incident response) consume the rest. See [[../03_AI_Agents_Core/10_Resource_Optimization|Resource Optimization]] for the full cost framework.

**<20% of pilots scale to production within 18 months** (McKinsey). The gap between "it works in a demo" and "it works reliably at scale with real users" is consistently underestimated.

**What separates the 12% that succeed:**
- Evaluation from day one — not added after launch (see [[../11_Evaluation_Testing/01_Metrics_and_Benchmarks|Metrics and Benchmarks]])
- Guardrails built into the architecture, not bolted on (see [[../14_Security_Safety/03_Agent_Security|Agent Security]])
- Observability that tracks agent decisions, not just latency (see [[../12_Observability/04_AgentOps|AgentOps]])
- Human-in-the-loop for edge cases, with graceful handoff from the start
- Scope discipline: solve one workflow well before expanding

---

## Related Materials

- [[01_Regulatory_Landscape|AI Regulatory Landscape]]
- [[02_AI_Risk_Management|AI Risk Management]]
- [[05_Governance_Frameworks|Governance Frameworks]]
- [[../03_AI_Agents_Core/09_Agent_Use_Cases|Practical AI Agent Use Cases]]

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Governance
**Previous:** [[06_Alignment_Research|Alignment Research]]
**Next:** [[../19_Practical_Projects/01_RAG_Chatbot|RAG Chatbot]]
