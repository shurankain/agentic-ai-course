# Resource-Aware Optimization

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → AI Agents: Core Concepts
**Previous:** [[09_Agent_Use_Cases|Practical Agent Use Cases]]
**Next:** [[../04_Multi_Agent_Systems/01_MAS_Basics|Multi-Agent Systems Fundamentals]]

---

## Introduction

AI agents consume significant resources: tokens, compute, time. In a production environment, uncontrolled consumption leads to:
- Unpredictable costs
- Quality degradation under load
- Poor user experience (latency)
- Rate limit exhaustion

Resource-aware optimization is a set of techniques for efficient resource utilization without sacrificing quality.

---

## Cost-Aware Routing

### Principle

Not all requests require the most powerful model. Smart routing directs requests to the optimal model based on the cost/quality ratio.

### Model Hierarchy

| Tier | Models | Cost | Use Case |
|------|--------|------|----------|
| Tier 1 (Economy) | GPT-4o-mini, Claude Haiku 4.5 | $0.15-1.00/1M | Simple tasks, classification |
| Tier 2 (Standard) | GPT-4o, Claude Sonnet 4 | $2.50-10/1M | Most tasks |
| Tier 3 (Premium) | o3, Claude Opus 4 | $10-75/1M | Complex reasoning |

### Routing Strategies

**Rule-based Routing:**
| Condition | Route |
|-----------|-------|
| Query < 50 tokens | Tier 1 |
| Simple Q&A detected | Tier 1 |
| Code generation | Tier 2 |
| Multi-step reasoning | Tier 3 |
| User = Premium | Tier 2-3 |

**Classifier-based Routing:**
A trained classifier determines query complexity:
- Input: query text, user context
- Output: recommended tier
- Trained on historical data (quality vs model used)

**Cascade Routing:**
1. Try Tier 1
2. If confidence < threshold, retry with Tier 2
3. Escalate to Tier 3 if necessary

---

## Token Budgeting

### Why a Token Budget Is Needed

Without controls, an agent can spend an unlimited number of tokens:
- Infinite reasoning loops
- Excessive context
- Repeated failed attempts

### Budget Allocation Strategy

**Per-Session Budget:**
| Category | Budget | Allocation |
|-----------|--------|------------|
| System prompt | Fixed | 500-1000 tokens |
| Retrieved context | Dynamic | 2000-4000 tokens |
| History | Sliding | Last N turns, max 2000 tokens |
| Generation | Capped | Max 1000 tokens |
| Tool responses | Limited | 500 tokens per tool |

**Adaptive Budgeting:**
- Start with a minimal budget
- Increase as needed
- Hard cap to protect against runaway

### Token Budget Enforcement

**Pre-generation:**
- Check: is there enough budget for the request?
- Truncate context if it exceeds the limit
- Warn or block if critical

**Post-generation:**
- Track actual consumption
- Update remaining budget
- Alert when approaching the limit

---

## Caching Strategies

### Semantic Caching

Traditional exact-match caching is inefficient for LLMs — identical queries are rare. Semantic caching uses embeddings to find similar queries.

**How it works:**
1. Receive query
2. Compute embedding
3. Search cache for close embeddings (cosine > threshold)
4. If found — return cached response
5. Otherwise — call the LLM, store in cache

**When effective:**
- FAQ-style queries
- Repeated patterns
- Stable knowledge base

**When NOT effective:**
- Highly personalized responses
- Time-sensitive data
- Creative tasks

### Response Caching Tiers

| Tier | TTL | Invalidation | Use Case |
|------|-----|--------------|----------|
| Hot | 1 hour | On update | Frequent queries |
| Warm | 24 hours | Daily | Stable data |
| Cold | 7 days | Manual | Reference data |

### Prompt Caching (Provider Feature)

Anthropic and OpenAI offer prompt caching:
- Caching of system prompt and context
- Up to 90% discount on cached tokens
- Automatic for long prompts

---

## Exception Handling

### Graceful Degradation

When errors occur, the system should degrade gracefully rather than fail completely.

**Degradation Levels:**

| Level | Trigger | Behaviour |
|-------|---------|-----------|
| Normal | Everything operational | Full functionality |
| Degraded | Model timeout | Fallback model |
| Limited | API unavailable | Cached responses only |
| Minimal | Critical failure | Static responses |

### Retry Strategies

| Error Type | Strategy | Max Retries |
|------------|----------|-------------|
| Rate limit | Exponential backoff | 5 |
| Timeout | Immediate retry | 2 |
| Server error | Backoff + fallback | 3 |
| Invalid response | Re-prompt | 2 |

**Exponential Backoff:**
- Retry 1: wait 1s
- Retry 2: wait 2s
- Retry 3: wait 4s
- Retry 4: wait 8s
- Give up or fallback

### Circuit Breaker

A pattern for protection against cascading failures:

| State | Behaviour |
|-------|-----------|
| Closed | Normal operation |
| Open | Fail fast, no calls |
| Half-Open | Test calls to check recovery |

**Transition rules:**
- Closed → Open: N failures in M seconds
- Open → Half-Open: After cooldown period
- Half-Open → Closed: Success in test call
- Half-Open → Open: Failure in test call

---

## Prioritization in Multi-Agent Systems

### Resource Contention

When multiple agents compete for resources:
- API rate limits
- GPU compute
- Memory
- Token budget

### Priority Queue

| Priority | Criteria | Resources |
|----------|----------|-----------|
| Critical | User-facing, real-time | Guaranteed allocation |
| High | Important workflows | Preemptive access |
| Normal | Background tasks | Fair share |
| Low | Batch processing | Best effort |

### Fair Scheduling

**Round Robin:**
- Each agent receives an equal slice
- Simple, but does not account for importance

**Weighted Fair Queuing:**
- Agents have weights
- Distribution proportional to weights

**Priority + Fair Share:**
- Guaranteed minimum for all
- Excess capacity allocated by priority

---

## Rate Limiting

### Token-based Rate Limiting

| Resource | Limit | Window |
|----------|-------|--------|
| API calls | 1000/min | Rolling |
| Tokens | 100K/min | Rolling |
| Cost | $100/day | Daily reset |

### User-level Limits

| User Tier | Requests/min | Tokens/day |
|-----------|--------------|------------|
| Free | 10 | 10K |
| Basic | 60 | 100K |
| Pro | 300 | 1M |
| Enterprise | Custom | Custom |

### Graceful Rate Limit Handling

When approaching the limit:
1. Warning at 80%
2. Throttling at 90%
3. Queue at 95%
4. Reject at 100%

---

## Monitoring Cost

### Key Cost Metrics

| Metric | Formula | Alert Threshold |
|--------|---------|-----------------|
| Cost per query | Total cost / Queries | >$0.10 |
| Cost per user | Total cost / Users | >$1/day |
| Token efficiency | Output value / Tokens | <0.5 |
| Cache hit rate | Cached / Total | <30% |

### Cost Attribution

Tracking expenses by:
- User / Account
- Feature / Use case
- Agent / Model
- Time period

### Budget Alerts

| Level | Threshold | Action |
|-------|-----------|--------|
| Info | 50% daily budget | Log |
| Warning | 80% daily budget | Alert team |
| Critical | 95% daily budget | Throttle |
| Emergency | 100% budget | Block new requests |

---

## Related Topics

- [[04_Planning|Planning]] — efficient planning conserves resources
- [[05_Memory_Systems|Memory Systems]] — memory vs re-computation trade-off
- [[../04_Multi_Agent_Systems/01_MAS_Basics|Multi-Agent Fundamentals]] — resource sharing between agents
- [[../12_Observability/02_Metrics_and_Dashboards|Metrics and Dashboards]] — resource monitoring

---

## Key Takeaways

1. **Cost-Aware Routing** — route simple tasks to cheaper models

2. **Token Budgeting** — set limits for each component:
   - System prompt: fixed
   - Context: dynamic with cap
   - Generation: capped
   - History: sliding window

3. **Caching:**
   - Semantic cache for similar queries
   - Prompt caching from providers
   - TTL tiers based on data stability

4. **Exception Handling:**
   - Graceful degradation levels
   - Exponential backoff for retries
   - Circuit breaker for protection

5. **Prioritization:**
   - Priority queues for multi-agent
   - Fair scheduling + priority
   - Rate limiting by user tiers

---

## Code Examples

### Cost-Aware Router

Implementing a Cost-Aware Router begins with defining the model hierarchy through an enumeration of tiers: economy, standard, and premium. Each tier corresponds to different models with varying costs and capabilities.

A ModelConfig data structure is created to describe a model configuration. It contains the model name, its tier, cost per thousand input tokens, cost per thousand output tokens, and maximum context size. These parameters are critical for routing decisions.

The MODEL_CONFIGS dictionary maps each tier to a specific configuration:

The ECONOMY tier uses the gpt-4o-mini model at a cost of $0.00015 per thousand input tokens and $0.0006 per thousand output tokens. The maximum context is 128,000 tokens. This model is optimal for simple tasks such as classification and short question-answer exchanges.

The STANDARD tier uses the gpt-4o model at a cost of $0.0025 per thousand input tokens and $0.01 per thousand output tokens. The maximum context is also 128,000 tokens. This is the primary workhorse for the majority of tasks.

The PREMIUM tier uses a reasoning model such as o3 at a cost of $0.01 per thousand input tokens and $0.04 per thousand output tokens. These models support large contexts (200K tokens) and deliver the highest reasoning quality. They are used for complex analytical tasks.

The CostAwareRouter class implements request routing logic. During initialization, it accepts an optional complexity classifier for more advanced strategies.

The main route method takes query text and an optional context, returning the optimal model configuration. The routing logic operates as follows:

First, it checks whether the query is simple. If so, the ECONOMY tier model is returned to minimize costs.

Next, it checks whether the query requires complex reasoning. If so, the PREMIUM tier model is returned for maximum quality.

For all other cases, the STANDARD tier model is returned as the optimal trade-off between cost and quality.

The simple query detection method uses a set of patterns: "what is", "who is", "when was", "how many", "yes or no". A query is considered simple if it contains at least one of these patterns AND is shorter than 100 characters. The length constraint is important because even a simple question with extensive context can be complex.

The complex reasoning detection method looks for indicators: "analyze", "compare and contrast", "step by step", "explain why", "implications". The presence of any of these phrases signals the need for deep analysis.

This approach enables saving 70-80% of API costs by routing simple queries to cheaper models while maintaining high quality for complex tasks.

### Token Budget Manager

The token budget management system consists of three interconnected components: budget definition, state tracking, and the management controller.

**TokenBudget structure** defines token limits for each usage category:
- System prompt (system_prompt): a fixed limit of 1,000 tokens for agent instructions
- Context (context): 4,000 tokens for retrieved data and documents
- History (history): 2,000 tokens for storing previous interactions
- Generation (generation): 1,000 tokens for model responses
- Tools (tools): 1,000 tokens for tool call results

The total property computes the overall budget by summing all categories, yielding the maximum allowable token count for the entire session.

**BudgetState structure** tracks current token usage. It holds a reference to the original budget and usage counters for each category, all initialized to zero.

The remaining_total property computes the token remainder as the difference between the total budget and the sum of used tokens.

The total_used property sums used tokens across all categories to obtain total consumption.

The can_allocate method checks whether the specified number of tokens can be allocated for a category. It retrieves the category limit from the budget, the current usage from the state, and verifies that the new allocation would not exceed the limit.

The allocate method performs the actual token allocation. It first verifies feasibility via can_allocate. If the check fails, it returns false. Otherwise, the current usage counter is incremented by the requested amount and true is returned.

**TokenBudgetManager class** provides a high-level API for budget management. During initialization, it stores the original budget and creates a new tracking state.

The request_tokens method requests tokens for a category and returns a tuple: operation success and the actual allocated amount. This allows flexible handling of situations where more is requested than available. The algorithm:
1. Compute the available token count as the difference between the limit and the amount used
2. If zero or fewer are available, return failure
3. Allocate the minimum of the requested and available amounts (partial allocation on shortage)
4. Reserve tokens via the allocate method
5. Return success and the actual allocated amount

The truncate_to_budget method trims text to fit the category budget. This is critical for controlling context and history size. The process:
1. Count the number of tokens in the text
2. Compute the available budget remainder
3. If the text fits, return it unchanged and allocate the tokens
4. If it exceeds the limit, compute a truncation ratio (available / required)
5. Truncate the text with a 10% safety margin (factor of 0.9) to compensate for counting inaccuracies
6. Append a truncation indicator (ellipsis)
7. Recount the tokens in the truncated text
8. Allocate the tokens and return the truncated text

The get_status method returns a detailed budget status report as a dictionary with the following fields:
- total_budget: the original total budget
- total_used: total tokens used
- remaining: tokens remaining
- utilization: usage percentage (0 to 1)
- by_category: per-category breakdown in "used/limit" format

The count_tokens method performs a simplified token count estimate based on text length. It uses the heuristic of "4 characters per token", which yields an approximate estimate. For production systems, precise tokenizers such as tiktoken are recommended.

This system enables precise control over token expenditure, prevents runaway consumption, and ensures predictable request costs.




---

## Navigation
**Previous:** [[09_Agent_Use_Cases|Practical Agent Use Cases]]
**Next:** [[../04_Multi_Agent_Systems/01_MAS_Basics|Multi-Agent Systems Fundamentals]]
