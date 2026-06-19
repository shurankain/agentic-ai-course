# Dialog Management

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Conversational AI
**Previous:** [[../08_Structured_Outputs/03_Validation_and_Error_Handling|Validation and Error Handling]]
**Next:** [[02_Conversation_Design|Conversation Design]]

---

## Introduction: Conversation as a Dance

Dialog is not a series of isolated utterances but a coherent dance. Topics flow into one another, the unsaid is implied, and references to prior context are understood without explanation. Building an AI system capable of participating in this dance is an extremely challenging task. A simple chatbot answers questions in isolation. A true dialog system remembers context, understands intents, and steers the conversation toward a goal.

## Anatomy of a Dialog

### Turns and Moves

The basic unit of dialog is a turn. The user speaks — one turn; the system responds — another. Not all turns are equal: "Hello" is an opening move, "Goodbye" is a closing move, "Yes" is a confirmation, "What do you mean?" is a clarification request.

### Speech Act Theory

Linguists John Austin and John Searle developed the theory that utterances are actions, not merely information transfer.

**Three levels of a speech act:**
- **Locutionary** (what is said): "It's cold in here" → a proposition about temperature
- **Illocutionary** (what is meant): a request to close the window, a statement of fact, or a complaint
- **Perlocutionary** (what effect it has): the listener closes the window, apologizes, or ignores it

**Dialog Act Categories:**
- Assert: informing ("The flight is at 10:00")
- Request: requesting an action ("Find a flight")
- Question: requesting information ("What time is the departure?")
- Confirm/Reject: confirmation or rejection
- Greet/Thank: social acts

Understanding illocutionary force helps produce correct responses. "Can you show me the menu?" is not a question about ability but a request for action.

### Sessions and Context

Dialog occurs within a session that has a beginning and an end. Context accumulates within it: what has been discussed, what decisions have been made, what information is known. "What is its price?" is meaningless without context. But if the previous message was about a laptop, "its" clearly refers to it.

### Topics and Focus

Topics switch, branch, and return. A user asks about a flight, switches to a hotel, then returns to the flight. The system must track the current topic (focus) and be able to switch without losing context. Like a stack: the current topic sits on top, previous ones beneath it.

## Dialog State

State is everything the system "remembers" about the conversation.

### State Components

**Message history** — the sequence of turns. Useful for understanding context and generating responses.

**Current intent** — what the user is trying to accomplish. Book a flight? Get information? File a complaint?

**Slots** — filled and unfilled parameters. For flight booking: origin, destination, date, number of passengers. Which are known? Which need to be asked?

**Dialog phase** — where in the process the conversation is: greeting, information gathering, confirmation, execution, completion.

**Context flags** — additional information: error, escalation, user dissatisfaction.

### State Transitions

Each user message triggers a transition. From "greeting," upon detecting an intent → to "information gathering." When all slots are filled → to "confirmation." Upon agreement → to "execution." For complex dialogs, state graphs or trees work better than finite state machines.

### State Machine Approaches

Finite state machines (FSMs) remain valuable for structured dialog flows where the conversation path is predictable. A flight booking FSM has states: IDLE → COLLECTING_ORIGIN → COLLECTING_DESTINATION → COLLECTING_DATE → CONFIRMING → EXECUTING → DONE. Transitions are triggered by slot-filling events. The advantage of FSMs: deterministic behavior, easy to test, predictable cost (fixed number of LLM calls), and clear audit trail (every transition is logged).

**When state machines beat free-form LLM dialog.** Regulated domains (healthcare, finance) where the conversation must follow a prescribed sequence — a medical triage bot cannot skip the symptom collection phase. High-volume transactional flows (order placement, appointment booking) where predictability matters more than flexibility. Cases where cost control is critical — a state machine makes exactly the LLM calls defined in the flow, while a free-form agent may loop unpredictably.

**Hierarchical state machines** handle complexity better than flat FSMs. A top-level machine manages the overall flow (booking → payment → confirmation), while nested machines handle sub-flows (booking: origin → destination → date → passengers). The nested machine runs independently and returns control to the parent when complete. This is the "nested dialog" pattern formalized.

**The hybrid approach (2026 standard).** Deterministic state machine for flow control (which phase are we in?), LLM for natural language understanding within each phase (extract slots from free text, generate natural responses, handle ambiguity). This gives the predictability of a state machine with the naturalness of LLM interaction — the best of both worlds. Anthropic's "Level 2 — Workflows" pattern from the agent progression model follows this principle.

### State Persistence

State is stored in memory (fast but lost on restart) or in a database (reliable but slower). A hybrid approach is recommended: active state in a cache such as Redis, with periodic synchronization to persistent storage. When the user returns, the system loads the previous context: "Last time you were looking for a flight to Paris. Continue?"

## Intent Recognition

Intent detection determines what the user wants. It is a key capability of a dialog system.

### Intent Classification

The traditional approach uses a fixed set of categories: BOOK_FLIGHT, CHECK_STATUS, ASK_PRICE, CANCEL_BOOKING, COMPLAIN, CHIT_CHAT. Each intent has examples and slots. LLMs simplify classification: describe the intents in a prompt and ask the model to identify the appropriate one.

### Intent Hierarchy

Intents form a hierarchy: BOOKING → BOOK_FLIGHT, BOOK_HOTEL, BOOK_CAR → ONE_WAY, ROUND_TRIP, MULTI_CITY. The hierarchy allows operating at different levels of granularity.

### Multiple Intents

A single message can express multiple intents: "Book a flight to Paris and find a hotel there" — BOOK_FLIGHT + BOOK_HOTEL. The system must recognize and handle multiple intents by splitting them into subtasks.

### Confidence and Ambiguity

The intent is not always obvious. "So how's it going?" — what does that mean? The system measures confidence for each candidate. When confidence is low, it requests clarification instead of guessing.

## Slot Filling

Slot filling extracts the parameters needed to fulfill an intent.

### Slot Definition

Each intent has slots — parameters to collect. BOOK_FLIGHT: origin, destination, date, passengers, class. Slots are required or optional. Without origin and destination a flight cannot be booked, but class can default to "economy."

### Extraction from Context

Slots are filled from the current and previous messages. "I want to go to Paris" → destination=Paris. "From London" → origin=London. "Next week" → date (normalization required). LLMs excel at extracting slots from natural language, including implicit references.

### Proactive Slot Prompting

When slots are unfilled, the system asks. Bad: "Specify departure point" (robotic). Good: "Where would you like to fly from?" Even better: "Got it, you're flying to Paris next week. Which city would be most convenient to depart from?" — demonstrates contextual awareness.

### Progressive Disclosure in Slot Filling

Asking for all required information at once feels like an interrogation. Progressive disclosure asks for information naturally, in the order that makes conversational sense. Instead of: "I need your origin, destination, date, number of passengers, and class preference" — start with: "Where would you like to go?" then build on each answer: "Paris, great! And where would you be departing from?" Each question incorporates the previously collected information, creating a natural conversational flow.

The principle: ask one thing at a time, confirm implicitly by echoing back, and infer what you can from context. "I want to fly to Paris next Tuesday" fills destination and date in one utterance — do not ask for the date again. "Business class as usual" — if the user has a preference history, pre-fill slots and only confirm: "Business class from London to Paris on Tuesday — shall I search?"

### Slot Validation

Values are validated. "A flight on February 30" — that date does not exist. "From Paris to Paris" — meaningless. For invalid values, the system provides an explanation and re-prompts: "February 30 does not exist. Did you mean the end of February or the beginning of March?"

## Multi-Turn Dialogs

### Context Management

Each message is interpreted in the context of previous ones. Too little context — the system "forgets." Too much — confusion from old conversation. The sweet spot: a sliding context window plus long-term memory for important information.

### Topic Switching

The user may abruptly change the topic. In the middle of booking a flight: "What's the weather like in Paris?" — unrelated, but it cannot be ignored. The system detects the switch, saves the current context, handles the new topic, and offers to return.

### Nested Dialogs

A topic nests like a subroutine call: "I want to book a flight" → "I need your details. What is your name?" → data collection → return to booking. The current state goes onto the stack, the subtask executes, then control returns.

### Rollbacks and Cancellations

The user may change their mind. "Cancel" — abort the action. "Back" — return to the previous step. "Start over" — clear the state. These commands work from anywhere as global handlers.

## Multi-Turn Context Management at Scale

Short conversations fit comfortably in a context window. But production dialog systems handle sessions that span hours, days, or weeks — customer support threads, ongoing project discussions, iterative document reviews. The context eventually exceeds any model's window.

**Summarization strategies.** When the conversation exceeds a threshold (e.g., 60% of the context window), summarize older turns into a compressed representation. The model continues with: system prompt + summary of earlier turns + recent N turns in full. The summary preserves: key decisions made, information collected (slots filled), commitments made by the system, and the current conversation state. What it discards: the exact wording of old messages, redundant back-and-forth, and social pleasantries. Claude Code's compaction pattern implements this automatically.

**Sliding window with anchors.** Keep the last N turns in full context, plus "anchor" messages that are always retained: the initial user request (defines the overall goal), any messages containing confirmed slot values, and messages where the system made commitments ("I'll send the confirmation email"). Anchors prevent the system from contradicting earlier commitments when old context is summarized away.

**Hierarchical memory for long conversations.** For multi-day sessions (ongoing projects, iterative work): Level 1 (in-context): last 10-20 turns, full fidelity. Level 2 (summary): summarized earlier turns, refreshed periodically. Level 3 (persistent): key facts, decisions, and preferences stored in a database (see [[../03_AI_Agents_Core/05_Memory_Systems|Memory Systems]]). The dialog manager queries Level 3 for relevant facts before each response, ensuring consistency even weeks into a conversation.

## Handoff Patterns

Conversations cross boundaries — from bot to human, from one agent to another, from one channel to another. Each handoff risks losing context and frustrating the user.

**Bot-to-human handoff.** Triggered by: repeated failures (3+ failed understanding attempts), explicit user request ("talk to a human"), detected frustration (sentiment analysis or escalation keywords), high-stakes decisions (the bot cannot authorize a refund above $500). The handoff must include: full conversation transcript, extracted intent and slot values, the specific point of failure, and any system-side context (user account details, order history). The human agent should never ask the user to repeat information.

**Agent-to-agent handoff.** In multi-agent systems, one agent may need to transfer the conversation to a specialist. The triage agent determines that this is a billing issue and hands off to the billing agent. The handoff contract: the receiving agent gets the conversation history, the current state (intent, filled slots), and a handoff reason. The receiving agent should acknowledge the handoff: "I see you have a billing question about order #12345. Let me help with that." Not: "Hi! How can I help you?" — which signals context loss and erodes trust. See [[../04_Multi_Agent_Systems/02_MAS_Patterns|MAS Patterns]] for the handoff pattern in multi-agent systems.

**Cross-channel handoff.** User starts on web chat, continues on the phone. The context must follow. Implementation: store the session state in a channel-agnostic format (not tied to WebSocket session or HTTP request), index by user identity, and load it when the user is authenticated on any channel. The user experience: "I see you were chatting with us about your delivery. Let me pick up where we left off."

## Key Takeaways

Dialog is a coherent conversation with context, topics, and goals — not isolated question-and-answer pairs.

Dialog state includes history, intent, slots, and phase. Everything is tracked and updated with each message.

Intent detection identifies the user's intent. The traditional approach classifies into categories. LLMs do this more flexibly and naturally.

Slot filling extracts parameters. Slots are required or optional. Unfilled slots are proactively prompted.

Multi-turn dialogs require context management, topic switching, nested dialogs, cancellations, and rollbacks. For long conversations: summarization, sliding window with anchors, and hierarchical memory.

State machines provide predictability for structured flows; LLMs provide naturalness for open-ended interaction. The hybrid approach (state machine for flow control + LLM for understanding) is the 2026 production standard.

Handoff patterns (bot-to-human, agent-to-agent, cross-channel) must preserve full context. Users should never have to repeat information after a handoff.

## Practical Implementation

The dialog manager controls the conversation state through phases: intent detection, slot filling, confirmation, execution. State includes intent, slots, history, and the current phase. The system transitions between phases based on information completeness. Upon detecting an intent, slots are extracted from context. If all slots are filled, the system moves to confirmation. Upon positive confirmation, the action is executed.

LLMs are used for intent recognition and slot extraction from natural language utterances. The model analyzes the message in the context of the history and determines the intent and slots simultaneously. For production systems, state persistence, error handling, more sophisticated transitions, and multiple intent handling are added.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Conversational AI
**Previous:** [[../08_Structured_Outputs/03_Validation_and_Error_Handling|Validation and Error Handling]]
**Next:** [[02_Conversation_Design|Conversation Design]]

---
