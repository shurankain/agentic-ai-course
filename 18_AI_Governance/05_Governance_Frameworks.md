# AI Governance Frameworks: Organizational Structures

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Governance
**Previous:** [[04_Enterprise_RAG|Enterprise RAG]]
**Next:** [[06_Alignment_Research|Alignment Research]]

---

## Introduction: From Technology to Organizational Maturity

AI technical capabilities outpace organizational readiness to use them. A company can deploy GPT-4 in a day, but building a governance framework for responsible AI use takes years. The gap between technical capability and governance maturity creates risks that materialize in headlines about AI failures, bias scandals, and privacy breaches.

Governance is not bureaucratic overhead but an enabler. Organizations with mature AI governance ship AI products faster because they know what guardrails are needed. They avoid costly post-hoc remediations because they built safety into the process. They attract talented AI specialists who want to work in an ethical environment.

History shows the cost of absent governance. Amazon abandoned its AI recruiting tool after discovering gender bias — millions spent on development, reputational damage, missed opportunities. Microsoft's Tay chatbot became racist within 24 hours — no red teaming or content moderation. Clearview AI faced lawsuits worldwide for using facial recognition without consent.

## Organizational Models of AI Governance

### Centralized Model

**AI Center of Excellence (CoE)** — a centralized team responsible for all aspects of AI governance. The CoE defines standards, approves use cases, conducts reviews, and provides training.

Advantages of centralization:
- Consistent standards across the organization
- Deep expertise concentration
- Clear accountability
- Economies of scale in tooling and processes

Disadvantages:
- Bottleneck effect — everyone waits for CoE approval
- Distance from business context
- May be perceived as "AI police"
- Difficult to scale as AI adoption grows

The centralized model works well for organizations at early stages of AI adoption or in heavily regulated industries.

### Federated Model

**Distributed governance** — each business unit has its own AI governance capability, coordinated by a central team. The center establishes policies and standards; business units implement them.

Advantages:
- Faster decision-making at the local level
- Better business context understanding
- Scalable as AI adoption grows
- Ownership at the business unit level

Disadvantages:
- Risk of inconsistency
- Duplication of effort
- Coordination complexity
- Varying maturity across units

The federated model suits large, diverse organizations with a mature governance culture.

### Hybrid Model

**Hub and spoke** — a central CoE (hub) plus AI champions in business units (spokes). The CoE owns strategy, standards, and high-risk approvals. Champions handle day-to-day governance, escalating complex cases.

This model balances central control with local autonomy. The CoE ensures consistency; spokes provide speed and context. Champions serve as a bridge between central policies and local implementation.

## Key Components of a Governance Framework

### AI Ethics Board

**Composition** — a diverse group including:
- Senior executives (accountability)
- Technical AI experts (feasibility)
- Legal/compliance (regulatory requirements)
- HR/Ethics specialists (human impact)
- External advisors (independent perspective)
- Employee representatives (front-line insight)

**Mandate**:
- Review high-risk AI use cases
- Set ethical guidelines and principles
- Adjudicate controversial decisions
- Advise on strategic AI direction
- Engage with external stakeholders

**Operating model**:
- Regular meetings (monthly or quarterly)
- Ad-hoc sessions for urgent issues
- Clear escalation criteria
- Documented decisions with rationale
- Transparency reports (internal or public)

### AI Policy Framework

Comprehensive policies covering:

**AI Use Policy** — when and how AI may be used:
- Approved use cases and prohibited uses
- Required approvals for different risk levels
- Data requirements and restrictions
- Human oversight requirements
- Vendor selection criteria

**AI Development Standards** — how AI should be developed:
- Model documentation requirements
- Testing and validation standards
- Bias and fairness testing
- Security requirements
- Code review and approval processes

**AI Operations Policy** — how AI should operate in production:
- Monitoring requirements
- Incident response procedures
- Model update and retirement processes
- Performance thresholds and SLAs
- Audit and compliance requirements

**AI Risk Policy** — how AI risks are managed:
- Risk assessment methodology
- Risk acceptance criteria
- Mitigation requirements
- Reporting and escalation
- Continuous monitoring

### Model Cards and Documentation

**Model Card** — standardized documentation format:

- **Model details**: name, version, type, training date
- **Intended use**: primary use cases, users, out-of-scope uses
- **Training data**: sources, size, date range, known biases
- **Performance metrics**: accuracy, fairness metrics, latency
- **Limitations**: known failure modes, edge cases
- **Ethical considerations**: potential harms, mitigations
- **Maintenance**: owner, update schedule, deprecation plan

Model cards ensure transparency and accountability. They help users understand a model's capabilities and limitations. They facilitate audits and regulatory compliance.

### Model Lifecycle Management

AI models require lifecycle governance from ideation to retirement:

**Ideation & Approval**:
- Business case documentation
- Risk assessment
- Ethics review for high-risk cases
- Resource allocation approval

**Development**:
- Adherence to development standards
- Regular checkpoints and reviews
- Testing and validation gates
- Documentation requirements

**Deployment**:
- Pre-deployment checklist
- Staged rollout procedures
- Monitoring setup
- Rollback plan

**Operation**:
- Continuous monitoring
- Periodic reviews
- Update procedures
- Incident handling

**Retirement**:
- Deprecation notice
- Data handling (retention, deletion)
- Replacement planning
- Historical documentation

## Approval Workflows

### Risk-Based Tiering

Not all AI use cases require the same level of oversight. Risk-based tiering directs appropriate scrutiny to appropriate cases:

**Tier 1 (Low Risk)**:
- Internal productivity tools
- Non-customer-facing
- Reversible decisions
- No sensitive data
- *Approval*: Team lead approval, self-service with guardrails

**Tier 2 (Medium Risk)**:
- Customer-facing but advisory
- Some sensitive data
- Human oversight in the loop
- *Approval*: Department head + AI CoE review

**Tier 3 (High Risk)**:
- Autonomous decisions affecting people
- Regulatory implications
- Significant sensitive data
- *Approval*: AI Ethics Board review required

**Tier 4 (Critical)**:
- Life/safety implications
- Large-scale impact
- Novel/unprecedented use
- *Approval*: Full Ethics Board + Executive sponsor + External review

### Stage-Gate Process

Each AI initiative passes through defined gates:

**Gate 0: Ideation**
- Is this aligned with AI principles?
- Is the data available and appropriate?
- Initial risk assessment
- → Proceed to exploration?

**Gate 1: Feasibility**
- Technical feasibility confirmed
- Data quality sufficient
- Preliminary fairness analysis
- → Proceed to development?

**Gate 2: Development Complete**
- All testing passed
- Documentation complete
- Security review passed
- → Proceed to pilot?

**Gate 3: Pilot Complete**
- Pilot success criteria met
- User feedback incorporated
- Monitoring working
- → Proceed to production?

**Gate 4: Production**
- All operational requirements met
- Training complete
- Support procedures in place
- → Approved for general availability

### Exception Handling

Rigid processes need flexibility for legitimate exceptions:

**Emergency deployments** — for urgent business needs with expedited review and a post-hoc full review requirement.

**Innovation sandboxes** — controlled environments for experimentation with relaxed production requirements.

**Pilot programs** — limited scope deployments with enhanced monitoring and clear success/failure criteria.

All exceptions are documented with clear justification and compensating controls.

## Model Versioning and Change Management

### Version Control Strategy

AI models require disciplined versioning:

**Semantic versioning adapted for ML**:
- MAJOR: Fundamental architecture change, new training data epoch
- MINOR: Significant performance improvement, new capabilities
- PATCH: Bug fixes, minor tuning

**Immutable artifacts** — each model version is preserved and reproducible. No "fixes in place."

**Lineage tracking** — the link between versions, training data, code, and configurations.

### Change Management

Changes to production AI require a controlled process:

**Change Request**:
- Description of change
- Reason/justification
- Risk assessment
- Testing evidence
- Rollback plan

**Change Approval**:
- Technical review
- Business impact assessment
- Appropriate approvals based on change scope

**Change Implementation**:
- Staged rollout
- Monitoring during rollout
- Clear success criteria
- Documented rollback procedure

**Post-Change Review**:
- Verification of expected outcomes
- Performance comparison
- Lessons learned

### A/B Testing Governance

A/B testing AI models requires ethical guardrails:

**Consent and transparency** — users know they might see different experiences

**Fairness in assignment** — random assignment shouldn't create disadvantaged groups

**Guardrails** — even experimental variants meet minimum safety standards

**Duration limits** — experiments don't run indefinitely

**Exit criteria** — clear rules for stopping experiments

## Metrics and Reporting

### Governance Metrics

Track governance effectiveness:

**Process metrics**:
- Time to approval by tier
- Approval rate
- Exception frequency
- Escalation patterns

**Compliance metrics**:
- Policy violations detected
- Audit findings
- Remediation time
- Training completion

**Risk metrics**:
- Open high-risk items
- Incident frequency and severity
- Near-miss reporting

**Culture metrics**:
- Governance satisfaction surveys
- Voluntary consultation requests
- Self-reported concerns

### Executive Dashboard

C-suite needs visibility into AI governance:

**Portfolio view**:
- Total AI initiatives by stage
- Risk distribution
- Business value delivery

**Risk posture**:
- Current risk level vs appetite
- Trending directions
- Key risks and mitigations

**Compliance status**:
- Regulatory compliance status
- Upcoming requirements
- Audit readiness

**Investment efficiency**:
- Governance overhead as % of AI investment
- Cost of non-compliance avoided

## Key Takeaways

AI Governance is not bureaucratic overhead but a competitive advantage. Organizations with mature governance deploy AI faster and more safely, avoiding costly failures.

Organizational models range from centralized (AI CoE) to federated (distributed with coordination) to hybrid (hub and spoke). The choice depends on organization size, regulatory environment, and AI maturity.

Key framework components include an AI Ethics Board, a comprehensive policy framework, model cards for documentation, and lifecycle management.

Risk-based tiering directs appropriate oversight to appropriate cases. Low-risk use cases receive streamlined approval; high-risk ones get full Ethics Board review.

A stage-gate process with clear gates from ideation to production ensures quality and accountability. Exception handling mechanisms provide flexibility without undermining governance.

Model versioning and change management are critical for production AI. Semantic versioning, immutable artifacts, and a controlled change process ensure stability and traceability.

Governance metrics — process, compliance, risk, culture — allow tracking effectiveness and continuous improvement. Executive dashboards provide visibility for the C-suite.

## Practical Code Examples

### AI Governance Platform

A centralized platform for managing AI governance that unifies all aspects of the AI initiative lifecycle: from submission to production deployment and retirement.

**Base enums and classification:**

RiskTier defines four risk levels for AI initiatives: LOW (low risk — internal productivity tools, no sensitive data, reversible decisions), MEDIUM (medium — customer-facing advisory systems, some sensitive data, human oversight), HIGH (high — autonomous decisions affecting people, regulatory implications, significant sensitive data), CRITICAL (critical — life/safety implications, large-scale impact, novel unprecedented use).

Each tier has a required_approvers property that returns a list of roles for approval: LOW requires only team_lead, MEDIUM requires department_head and ai_coe_reviewer, HIGH requires ethics_board_member, legal, and security, CRITICAL requires ethics_board_chair, executive_sponsor, and external_advisor.

InitiativeStatus tracks lifecycle stages: DRAFT (draft), SUBMITTED (submitted for review), UNDER_REVIEW (under review), APPROVED (approved), REJECTED (rejected), IN_DEVELOPMENT (in development), IN_PILOT (pilot testing), IN_PRODUCTION (in production), DEPRECATED (deprecated), RETIRED (retired).

ApprovalStatus for individual approval requests: PENDING (awaiting decision), APPROVED (approved), REJECTED (rejected), DEFERRED (deferred for additional information).

**Data structures:**

Stakeholder represents a governance participant: user_id, name, email, roles list, department, can_approve_tiers list of tiers the person can approve.

ApprovalRequest — an approval request: request_id, initiative_id, approver_role required role, approver_id who actually approved (populated upon processing), status, requested_at and responded_at timestamps, comments from the approver, conditions list of approval conditions.

ModelCard — standardized model documentation in Google Model Cards format: model_id, name, version, model_type, description, intended use (primary_use_cases, target_users, out_of_scope_uses), training data (sources, size, date_range, known_data_biases), performance (metrics dictionary, fairness_metrics by group, latency_p50_ms and p99_ms), limitations (known_limitations, failure_modes, edge_cases), ethical considerations (potential_harms, mitigation_strategies, ethical_review_date), maintenance info (owner_id, update_schedule, deprecation_date). The to_markdown method exports to readable markdown for sharing and audits.

AIInitiative — the central entity representing an AI project: initiative_id, name, description, business_owner, technical_owner, department, risk classification (risk_tier, use_case_category, data_classification), status and current_stage, timeline (created_at, submitted_at, approved_at, launched_at), approval_requests list, optional model_card and risk_assessment, tags and notes for metadata.

**GovernanceWorkflow — central orchestrator:**

The class manages the full lifecycle of initiatives. It stores initiatives and stakeholders dictionaries and a notifications list.

The register_stakeholder method adds a participant to the system.

The create_initiative method creates a new initiative with a unique ID (first 8 characters of a UUID in uppercase), sets the status to DRAFT and stage to "ideation", and records the created_at timestamp.

The submit_for_approval method transitions the initiative from DRAFT to SUBMITTED. It validates the current status (only DRAFT can be submitted). Based on the risk_tier, it retrieves the list of required_approvers and creates an ApprovalRequest for each role with PENDING status. It calls _notify_role to send notifications to all stakeholders with the corresponding role. It sets submitted_at.

The process_approval method handles an approver's decision. It verifies the approver has the required role. It finds the matching PENDING request and updates: approver_id, status (APPROVED or REJECTED), responded_at, comments, conditions. It calls _update_initiative_status to check whether all approvals are complete.

The private _update_initiative_status method analyzes all approval requests: if any is REJECTED, the initiative is REJECTED; if all are APPROVED with none PENDING, the initiative is APPROVED with an approved_at timestamp; otherwise it remains UNDER_REVIEW.

The get_pending_approvals method returns the list of approvals awaiting decision for a user, filtering by their roles.

The advance_stage method transitions the initiative to the next stage, logging the transition in notes. Certain stages automatically update the status (development → IN_DEVELOPMENT, pilot → IN_PILOT, production → IN_PRODUCTION with launched_at, deprecated → DEPRECATED, retired → RETIRED).

The generate_dashboard method creates an executive dashboard: total_initiatives, by_status breakdown, by_risk_tier distribution, pending_approvals count, at_risk list of initiatives submitted more than 7 days ago and still pending (age in days for each).

**ModelRegistry — model version management:**

The class stores all versions of all models: a models dictionary (model_id to list of ModelCard versions), a production_versions dictionary (model_id to currently deployed version string).

The register_model method adds a new model version, checking version uniqueness.

The get_model method returns a ModelCard by ID and optional version. Without a version, it returns the latest (by created_at).

The set_production_version method marks a version as deployed.

The get_production_version method returns the current production version of a model.

The deprecate_model method sets a deprecation_date on a version.

The get_deprecated_models method finds versions with a deprecation_date in the past that are still in production — requiring replacement.

**The usage example demonstrates the full workflow:** Platform initialization, registering stakeholders (Alice — team lead and AI CoE reviewer, Bob — department head and ethics board member, Carol — legal and security). Creating an initiative "Customer Churn Prediction" with MEDIUM risk tier. Submission for approval, generating requests to department_head and ai_coe_reviewer. Checking pending approvals for Alice. Processing approvals from both approvers (Alice approves with "Technical approach looks sound", Bob with condition "Monthly fairness audit required"). After all approvals, the status is APPROVED. Advancing to the development stage, updating the status to IN_DEVELOPMENT. The dashboard shows aggregate metrics. The model registry demo creates a ModelCard for a trained model with full metrics, sets it to production, and exports markdown.

### Approval Workflow Automation

An automated governance workflow with external system integration (Slack, Email, Jira) for notifications, escalations, and SLA tracking.

**Notification system:**

The NotificationChannel enum defines channels: EMAIL, SLACK, TEAMS, JIRA.

NotificationConfig contains channel settings: channel, enabled flag, webhook_url for Slack/Teams, api_key for JIRA, default_recipients list.

NotificationService — a centralized sending service. The configure method accepts a NotificationConfig. The send method dispatches a notification through the specified channel with recipients, subject, message, and optional metadata. The private methods _send_slack, _send_email, _create_jira_ticket encapsulate integrations (mocks in the example, API calls in production). All sent notifications are logged to sent_notifications for audit.

EscalationRule defines an escalation rule: name, condition function that takes a context and returns True if escalation is needed, escalate_to list of recipients, notification_template string with placeholders, wait_time_hours how many hours to wait before escalation.

**AutomatedWorkflow — main orchestrator:**

The class accepts a NotificationService in the constructor. It stores an escalation_rules list and a pending_items dictionary.

The add_escalation_rule method registers a rule.

The request_approval method creates an approval request: saves it to pending_items with item_id, item_type, title, description, required_approvers, an empty received_approvals list, priority, due_date, created_at, status "pending". It sends notifications via Slack and Email to all required_approvers with details. If the priority is "high", it creates a tracking ticket in Jira.

The check_escalations method periodically checks pending items. For each pending item, it computes age_hours. It iterates through escalation_rules, calling each condition function with a context (item, age_hours, priority, pending_approvers). If the condition returns True and age >= wait_time_hours, it triggers escalation: sends a notification to escalate_to recipients with a formatted message from the template (substituting title, age_hours, item_id) and logs it to the escalations list.

The record_approval method processes an approval decision. It verifies the approver is in required_approvers. It adds a record to received_approvals with approver, approved boolean, comments, timestamp. It checks completion: if anyone has rejected, status becomes "rejected" and it calls _notify_outcome. If all required approvers are in approved_by, status becomes "approved" and it calls _notify_outcome. Otherwise it remains pending.

The private _notify_outcome method sends a Slack notification to stakeholders: a green checkmark and "approved and can proceed" or a red cross and "rejected by [name]".

The send_reminder method sends a reminder to approvers who have not yet responded (pending_approvers — the difference between required and received).

The get_sla_status method returns SLA status for all pending items. For each, it computes age, determines SLA hours by priority (critical: 4h, high: 24h, normal: 72h, low: 168h), computes sla_deadline and a breached flag (now > deadline). It returns a list sorted by breached (breached first) and age_hours (oldest first).

**Usage example:** Setting up a notification service with configs for Slack (webhook), Email, and Jira (api_key). Creating a workflow. Adding escalation rules: "high_priority_24h" (if priority is high and age >= 24h, escalate to department head and AI governance lead with a warning message) and "any_priority_72h" (if age >= 72h regardless of priority, escalate to governance manager with an overdue notice). Requesting approval for a "Customer Sentiment Analysis" initiative at high priority with 2 approvers and a 3-day due date. Recording approvals from both approvers. Checking SLA status showing age, SLA hours, deadline, breached flag, pending approver count. Checking escalations (normally a scheduled job) triggering rules if conditions are met.


---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Governance
**Previous:** [[04_Enterprise_RAG|Enterprise RAG]]
**Next:** [[06_Alignment_Research|Alignment Research]]
