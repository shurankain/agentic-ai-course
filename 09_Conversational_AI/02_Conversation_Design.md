# Conversation Design

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Conversational AI
**Previous:** [[01_Dialog_Management|Dialog Management]]
**Next:** [[03_Voice_and_Multimodality|Voice and Multimodality]]

---

## Introduction: The Art of Dialogue

Technology is only half the equation. You can build a flawless infrastructure, but if the conversations are poorly designed, users will be frustrated. Conversation design is the art of creating pleasant, efficient, and human-like interactions. It is not programming but scriptwriting. Not algorithms but empathy.

## Conversation Flows

A good conversation has structure. It does not wander chaotically, yet it does not feel mechanical. It strikes a balance between direction and flexibility.

### Linear Flows

The simplest structure is a linear step-by-step flow: greeting → intent identification → information gathering → confirmation → action → closure. These are easy to implement and intuitive for users but inflexible. Users cannot skip ahead or go back without special handling.

### Branching Flows

A decision tree: different paths depending on the response. "Would you like to make a booking or modify an existing one?" leads to different branches. Branching personalizes the experience but complicates design. With N branching points and M options, there are M^N possible paths. Testing becomes combinatorially complex.

### State Graph

The most flexible model is a graph with navigation between states in any direction. Started booking a flight, switched to a hotel, returned to the flight — all valid. Graphs are harder to design and test but provide the most natural interaction. Modern LLM-based dialog systems use this model.

## Handling Misunderstanding

No system understands everything. How should ambiguity be handled?

### Clarifying Questions

The first line of defense is clarification. Do not guess — ask. "I didn't quite understand. Would you like to book a flight or check on an existing reservation?" Clarification shows respect. It is better to ask than to act incorrectly.

### Paraphrasing

Paraphrase what was heard: "If I understood correctly, you're looking for a flight to Paris next week?" This gives the user a chance to confirm or correct. Useful for complex requests with many details.

### Offering Options

When there is complete ambiguity, offer options: "I can help with booking, finding information, or modifying an order. What do you need?" A structured choice is easier than an open-ended question.

### Graceful Degradation

If nothing works, acknowledge limitations honestly: "Unfortunately, I didn't understand your request. Could you rephrase it?" The worst approach is pretending to understand when you do not. This leads to incorrect actions and loss of trust.

## Fallback Strategies

When standard mechanisms fail, fallback options are needed.

### Progressive Escalation

Do not give up immediately — gradually expand assistance:
- First attempt: "I didn't understand, could you rephrase?"
- Second attempt: "I'm having difficulty. Here's what I can do: list of capabilities"
- Third attempt: "It seems I need help. Would you like to speak with a human?"

Each level provides more context and alternatives.

### Handoff to a Human

Sometimes the best response is to transfer the conversation to a live agent. Complex complaints, unusual situations, emotional users — these are better handled by a person. During handoff, it is critical to preserve context. The agent should see the history, what the user was trying to do, and where the problem occurred. Forcing users to repeat everything from scratch is a path to frustration.

### Alternative Channels

If the voice channel is not working, offer text. If chat cannot handle it, provide a phone number. If online is unavailable, offer email. Users should know that even if this method does not work, other options exist.

## Persona Design: The System Prompt as Personality

An AI assistant is not merely a function but a character. The system prompt defines this character — not just what the agent knows, but how it communicates, what it refuses to discuss, and how it handles edge cases. Persona design is the conversational equivalent of UX design.

### Building a Persona Through the System Prompt

A well-designed persona prompt defines four layers. **Identity:** who the agent is ("You are a financial advisor at Acme Bank"). **Capabilities:** what it can help with ("You can answer questions about accounts, transactions, and investment products"). **Boundaries:** what it will not do ("You do not provide specific investment advice, tax guidance, or access to other customers' accounts. If asked, redirect to a human advisor"). **Communication style:** how it speaks ("Professional but warm. Use simple language. Avoid jargon unless the customer uses it first. Never use exclamation marks in financial contexts").

The boundaries layer is often the most important and the most neglected. Without explicit boundaries, the agent will attempt to help with anything — including topics where it should decline. A customer service agent answering medical questions, a coding assistant giving legal advice, a banking bot making investment recommendations — each is a liability risk caused by missing persona boundaries.

### Consistency Across Turns

The chosen tone must survive edge cases. If the assistant is friendly, it does not become cold when an error occurs — it becomes empathetically apologetic. If it is formal, it does not suddenly use slang. Consistency creates predictability. Users develop an implicit model of how the agent communicates, and violations of this model feel jarring. Test persona consistency by running the agent through scenarios: happy path, error path, frustrated user, ambiguous request, out-of-scope request. The tone should flex but not break.

### Adapting to the User

Tone adapts to the specific user. If users write formally, respond formally. If they use emoji and slang, soften the style accordingly. Mirroring creates a sense of mutual understanding. LLMs are naturally good at this — include in the system prompt: "Adapt your communication style to match the user's tone and level of formality."

### Emotional Intelligence

The assistant recognizes emotions and responds accordingly. Is the user upset? Start with empathy: "I understand this is a frustrating situation," then address the problem. Is the user happy? Share in the joy: "Glad everything worked out." A lack of emotional awareness is the cardinal sin of automated systems. Voice agents (see [[03_Voice_and_Multimodality|Voice and Multimodality]]) have an advantage here — Retell AI and similar platforms detect frustration, satisfaction, and confusion from voice signals in real time, enabling automated escalation based on emotional state.

## Greetings and Farewells

The beginning and end of a conversation shape the overall impression.

### Greeting

The greeting sets the tone. It should be brief (not overwhelming), helpful (conveying what the system can do), and open (inviting interaction). Do not repeat the same greeting. For returning users: "Welcome back! Last time you were looking at flights to Rome. Shall we continue?"

### Farewell

A good farewell reinforces a positive experience. It includes a summary (what was accomplished), next steps (what comes next), and an open door (an invitation to return). Even when the conversation ends unsuccessfully, be gracious: "Sorry I couldn't help today. You can reach the support hotline at..."

## Handling Difficult Situations

### Frustrated Users

The key is to avoid making things worse. Do not argue, do not make excuses, do not assign blame. Acknowledge the problem, express understanding, and focus on a solution: "I understand this is a frustrating situation, and I apologize. Let's see what can be done."

### Abusive Behavior

The assistant should have boundaries. A gentle reminder: "I'm here to help and would be happy to continue in a respectful tone." If it continues: "Unfortunately, this style of communication makes it difficult for me to assist effectively. We can continue later, or you can reach out to a live agent."

### Impossible Requests

Honesty is more important than trying to please: "Unfortunately, this is beyond my capabilities. But I can suggest an alternative..." Always offer an alternative. A flat refusal leaves users at a dead end.

### System Errors

When the system makes a mistake, acknowledge it. People forgive errors when they see sincerity: "I apologize, an error occurred. I'll try to fix it. Could you repeat...?"

## Personalization

Conversations with each user can and should differ.

### Using Names

Addressing users by name creates a personal connection. But do not overdo it — using the name in every sentence is unnatural and irritating.

### Remembering Preferences

Returning users should not have to repeat basic information every time: "I see you usually fly business class. Should I search for business class only?"

### Adapting to Patterns

If users always select certain options, offer those first: "As usual, a hotel near the city center?"

### Proactive Suggestions

Knowing the history, you can offer relevant suggestions: "You often fly to Berlin on Thursdays. This week there are some good deals."

## Error Recovery in Depth

Beyond fallback strategies, a well-designed dialog system has patterns for recovering from specific error types without restarting the conversation.

**Misinterpretation recovery.** The agent acted on a wrong interpretation. The user says "No, that's not what I meant." The agent should: acknowledge the error ("I apologize for the misunderstanding"), discard the incorrect action's results (rollback), re-prompt with explicit options ("Let me try again — did you mean X or Y?"), and not repeat the same interpretation. Implementation: track failed interpretations in session state and exclude them from future classification candidates.

**Partial information recovery.** The agent collected 4 out of 5 required slots, then the user said something confusing. Do not restart from scratch — confirm what is known ("I have your departure from London to Paris on Tuesday, business class. I just need to confirm — how many passengers?") and collect only the missing piece. Users abandon conversations when forced to repeat information.

**Graceful "I don't know" patterns.** When the agent genuinely cannot help, the quality of the rejection matters. Bad: "I cannot help with that." Better: "I don't have information about that, but I can help you with X, Y, or Z — or connect you with someone who can." Best: "That's outside my expertise. Based on your question, I think our billing team would be the best fit — shall I connect you?" The agent should always leave the user with a next step, never at a dead end.

## Proactive Engagement

Most conversational agents are reactive — they wait for the user to speak. Proactive engagement means the agent initiates or steers the conversation based on context, timing, or inferred needs.

**When to initiate.** The agent should proactively engage when: it detects the user is stuck (long pause, repeated similar queries), it has relevant information the user might not know to ask for ("I noticed your flight has a 4-hour layover — would you like lounge access options?"), a deadline or trigger event approaches ("Your subscription renews in 3 days — would you like to review your plan?"), or the current task naturally leads to a related need (booked a flight → "Would you like hotel recommendations in Paris?").

**When NOT to initiate.** Do not interrupt focused work, do not push repeatedly after a dismissal, and do not suggest when the user has explicitly said they are done. The line between helpful and pushy is thin — err on the side of less. A good rule: one proactive suggestion per session unless the user engages with it.

**Recommendation engine in conversation.** Based on the user's history and current context, the agent suggests relevant options without being asked. This requires: access to user history (past bookings, preferences, frequently asked topics), current session context (what the user is doing now), and a relevance threshold (only suggest when confidence is high). The Salesforce Agentforce pattern: surface the top recommendation inline in the conversation, with a brief rationale — "Based on your past trips, you might prefer the direct flight at 10:15 AM."

## Conversation Metrics

How do you know if your conversational AI is working well? Subjective impressions are unreliable. Metrics provide objective measurement.

**Resolution rate** — the percentage of conversations where the user's issue was resolved without escalation to a human. Target: 70-85% for customer service (higher means the agent handles only trivial cases; lower means it is not useful enough). Track qualified resolution rate — exclude conversations where the user abandoned before stating a problem.

**Average turns to resolution** — fewer turns means a more efficient agent. But very low turn counts may indicate the agent is cutting conversations short rather than solving problems. Benchmark: 3-5 turns for simple tasks (FAQ, status check), 8-15 turns for complex tasks (troubleshooting, multi-step booking).

**User satisfaction correlation.** CSAT scores correlate with specific conversation patterns. Research consistently shows: fast first response improves CSAT more than fast resolution. Empathetic language in error scenarios improves CSAT by 15-20%. Forcing users to repeat information is the strongest negative CSAT predictor. Proactive context use ("I see your order #12345") is the strongest positive CSAT predictor. Monitor these patterns and optimize the dialog design accordingly.

**CSAT prediction from dialog features.** Instead of surveying every user (low response rate, biased toward extremes), train a classifier to predict CSAT from dialog features: number of turns, number of clarification requests, presence of frustration keywords, time to resolution, whether escalation occurred, and whether the agent acknowledged errors. This gives a CSAT estimate for every conversation, enabling continuous quality monitoring without surveys.

**Abandonment rate** — conversations where the user stops responding before resolution. High abandonment (>30%) signals: the agent is not understanding the user, too many turns before progress, or the user found an alternative. Analyze abandoned conversations to identify common drop-off points — these are the highest-priority conversation design improvements.

## Key Takeaways

Conversation design is the art of creating pleasant and efficient interactions. Technology matters, but user experience matters more.

Conversation flows can be linear, branching, or graph-based. More flexible structures are harder to implement but yield more natural interactions.

Handling misunderstanding is critical. It is better to ask than to guess. Offer options. Acknowledge limitations honestly.

Fallback strategies include progressive escalation, handoff to a human, and alternative channels. There should always be a path to resolution.

Personality and tone define the assistant's character. Consistency, user adaptation, and emotional intelligence make communication feel human.

Difficult situations require patience and empathy. Do not argue with frustrated users, set boundaries with abusive ones, and offer alternatives for impossible requests.

Personalization makes interactions unique. Remember names, preferences, and history. Offer suggestions proactively — but respect the line between helpful and pushy.

Error recovery should preserve collected information — never force users to repeat themselves. Always leave the user with a next step, never at a dead end.

Measure conversations quantitatively: resolution rate, turns to resolution, CSAT correlation with dialog patterns, abandonment rate. CSAT prediction from dialog features enables continuous quality monitoring without surveys.

## Practical Implementation

Handling misunderstanding is built on progressive escalation. On the first failure, the system asks for rephrasing. On the second, it offers a list of capabilities. On the third, it recommends alternative channels or a live agent. It is critical not to keep guessing when confidence is low.

Generating clarifying questions uses an LLM to create natural follow-up questions based on possible interpretations. The model receives the user's message and a list of possible intents, then formulates a question to resolve the ambiguity.

Tone adaptation analyzes the user's communication style from their history (formality, level of detail, technical level) and applies the corresponding style to system responses. This creates a sense of mutual understanding and improves interaction comfort.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Conversational AI
**Previous:** [[01_Dialog_Management|Dialog Management]]
**Next:** [[03_Voice_and_Multimodality|Voice and Multimodality]]

---
