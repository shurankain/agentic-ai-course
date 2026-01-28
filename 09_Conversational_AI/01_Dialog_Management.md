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

## Key Takeaways

Dialog is a coherent conversation with context, topics, and goals — not isolated question-and-answer pairs.

Dialog state includes history, intent, slots, and phase. Everything is tracked and updated with each message.

Intent detection identifies the user's intent. The traditional approach classifies into categories. LLMs do this more flexibly and naturally.

Slot filling extracts parameters. Slots are required or optional. Unfilled slots are proactively prompted.

Multi-turn dialogs require context management, topic switching, nested dialogs, cancellations, and rollbacks.

## Practical Implementation

The dialog manager controls the conversation state through phases: intent detection, slot filling, confirmation, execution. State includes intent, slots, history, and the current phase. The system transitions between phases based on information completeness. Upon detecting an intent, slots are extracted from context. If all slots are filled, the system moves to confirmation. Upon positive confirmation, the action is executed.

LLMs are used for intent recognition and slot extraction from natural language utterances. The model analyzes the message in the context of the history and determines the intent and slots simultaneously. For production systems, state persistence, error handling, more sophisticated transitions, and multiple intent handling are added.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Conversational AI
**Previous:** [[../08_Structured_Outputs/03_Validation_and_Error_Handling|Validation and Error Handling]]
**Next:** [[02_Conversation_Design|Conversation Design]]

---
