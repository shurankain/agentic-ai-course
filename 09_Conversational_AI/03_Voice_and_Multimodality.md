# Voice and Multimodality

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Conversational AI
**Previous:** [[02_Conversation_Design|Conversation Design]]
**Next:** [[../10_Fine_Tuning/01_Fine_Tuning_Basics|Fine-Tuning Basics]]

---

## Introduction to Multimodal Interfaces

Human communication is multimodal: we use voice with intonation, gestures, facial expressions, and show images. Conversational AI systems limited to text miss a vast layer of information. Multimodality refers to a system's ability to perceive and generate information through various channels: text, speech, images, and video. This represents a qualitative leap in interaction naturalness.

## Voice Interface: From Sound to Meaning

### Philosophy of Voice Interaction

Voice is the most natural mode of communication. Children learn to speak long before they learn to write. Voice interaction frees the hands and eyes, allows communication during other activities, and makes technology accessible to people with disabilities.

Speech is linear and ephemeral — a spoken word cannot be "re-read." Voice conveys emotional context that text does not express. Speech perception speed is lower than reading speed, but voice does not require visual attention. Responses should be concise and structured. Important information is repeated. The system provides the ability to interrupt a long response. Pauses and intonation aid comprehension.

### Speech-to-Text

Speech recognition (ASR) has evolved from early systems with limited vocabularies to modern models with human-comparable accuracy. Modern systems work end-to-end: they accept an audio signal and generate text directly, without intermediate stages. Whisper by OpenAI is trained on hundreds of thousands of hours of speech and operates under conditions of noise, accents, and non-standard pronunciation.

Key characteristics: low Word Error Rate (WER) — for modern systems, 2-5% on clean speech, comparable to human performance. In real-world conditions (noise, multiple speakers, specialized terminology), quality degrades.

Streaming ASR delivers results in real time as words are spoken. This is critically important for interactive systems — latency affects naturalness. Users expect a response immediately after finishing a phrase.

End-of-utterance detection is a non-trivial task. A pause may indicate completion of a thought or a moment of deliberation. Triggering too early interrupts the user. Triggering too late causes unnatural delays. A combination of acoustic features and language models is used.

### Text-to-Speech

Speech synthesis (TTS) is the inverse task: converting text into natural-sounding speech. Modern neural network systems have reached a level where synthesized speech is difficult to distinguish from a human recording.

High-quality synthesis includes:
- **Naturalness** — sounds human, without mechanical artifacts
- **Intelligibility** — words are pronounced clearly and understandably
- **Expressiveness** — conveys emotions and intonation
- **Controllability** — ability to control speed, tone, and style

Voice cloning from a short sample opens opportunities for personalization — the agent speaks in a voice chosen by the user or matching the brand. However, it creates risks of misuse: fraud and deepfakes.

Emotional synthesis conveys coloring: sad news sounds empathetic, happy news sounds upbeat, important warnings sound serious and insistent. This requires emotion markup in the text or automatic detection from context.

Prosody — rhythm, stress, and intonation — is critically important. Incorrect stress changes the meaning of a word; incorrect intonation turns a statement into a question. Modern systems learn from large corpora, but complex cases require manual annotation.

## The Native Speech-to-Speech Paradigm Shift (2024-2025)

### From Pipeline to Native Processing

The traditional voice agent architecture — ASR → LLM → TTS — is being disrupted by **native speech-to-speech models** that process and generate audio directly, without text as an intermediary.

**OpenAI Realtime API (October 2024):** The first production API for native speech-to-speech interaction. GPT-4o processes audio tokens directly — speech is not transcribed to text before the model processes it. This enables:
- **Sub-200ms latency** — no ASR/TTS overhead, the model reasons on audio directly
- **Natural prosody** — the model perceives and generates intonation, emphasis, emotion, and pauses
- **Voice style control** — choose from preset voices or define tone characteristics
- **Interruption handling** — native barge-in detection at the model level
- **WebSocket-based streaming** — persistent bidirectional connection for real-time conversation

The API uses a **session-based model**: create a session with configuration (voice, instructions, tools), stream audio in, receive audio back. The model can use tools (function calling) mid-conversation, enabling voice-driven agents that book appointments, look up data, or control systems.

**Google Gemini Live (December 2024):** Google's real-time conversational API, leveraging Gemini's native multimodal capabilities. Processes audio, video, and text simultaneously — a user can talk while sharing their screen, and the model understands both streams. Supports long audio input (hours of audio via Gemini's 1M+ token context).

**Anthropic Claude voice:** Claude processes audio input through transcription but excels at understanding the *content* of conversations, meeting recordings, and spoken instructions. While not natively speech-to-speech, Claude's strength is in deep reasoning about audio-derived content.

### Impact on Voice Agent Architecture

The native speech-to-speech approach changes the architecture fundamentally:

**Traditional pipeline (still valid for many use cases):**
Audio → ASR (Whisper) → Text → LLM → Text → TTS → Audio
- Latency: 500-2000ms total
- Each stage adds error and latency
- But: transparent, debuggable, each component can be swapped

**Native speech-to-speech (2024+):**
Audio → Multimodal LLM → Audio
- Latency: 100-300ms
- Preserves prosody, emotion, speaker characteristics
- But: less transparent, harder to debug, higher API cost

**Hybrid approach (emerging best practice):**
Use native speech-to-speech for real-time conversation. Fall back to the pipeline for complex tasks requiring tool use or structured output. Use transcription in parallel for logging, analytics, and compliance.

## Voice Agent Architecture

### Voice Processing Pipeline (Traditional)

The traditional voice agent remains a pipeline of components, each contributing to latency and quality:

1. **Audio capture** — a microphone converts sound waves into a signal that is digitized. Microphone quality, acoustics, echo cancellation, and noise suppression affect the input signal.

2. **Speech recognition** — a model converts audio to text. Locally (fast, offline, resource-limited) or on a server (powerful models, network latency).

3. **Understanding and response generation** — text is passed to a language model that determines intent, extracts entities, and formulates a response. This is typically the most time-consuming part.

4. **Speech synthesis** — converting the text response to audio. Locally or on a server.

5. **Playback** — audio output to a speaker or headphones.

### Queue Management and Interruptions

In natural conversation, people do not wait for a turn to end before they start listening. The ability to interrupt the system during a response (barge-in) is critically important. This requires simultaneously listening to the microphone and playing the response, with an algorithm to determine whether the signal is user speech or echo. Upon detecting speech, the system stops playback and processes the new input.

Full-duplex communication: both participants speak simultaneously. This requires sophisticated processing to separate voices and understand how utterances relate to each other.

### Latency Optimization

Latency is the primary enemy of natural interaction. Latency exceeding 300-400 ms is perceived as unnatural and irritating. Total latency accumulates: audio capture (10-50 ms), server transmission (20-100 ms), speech recognition (100-500 ms), language model (200-2000 ms), speech synthesis (100-500 ms), transmission and playback (20-100 ms). The language model is often the bottleneck.

Minimization strategies:
- Streaming processing at all stages — begin synthesis without waiting for the complete response
- Caching frequent responses
- Smaller models for simple queries
- Prediction and pre-generation of probable responses
- Edge computing for local stages

## Computer Vision in Conversational Systems

### From Text to Multimodal Understanding

Vision-language models (VLMs) are models that perceive and analyze images and text simultaneously. GPT-4o, Claude Sonnet 4, and Gemini 2.5 open new possibilities. Users can not only describe a problem but also show a screenshot of an error, a photo of a broken device, or a diagram. An agent capable of seeing analyzes visual information and provides accurate and relevant answers.

Typical scenarios: screenshot analysis for technical support, document recognition and processing, product identification from photos, chart and diagram analysis, design assistance.

### Understanding Visual Context

VLMs do not merely describe an image — they understand it in the context of the conversation. "What is this error?" — the model understands that the screenshot shows an application window with an error message, reads the error text, understands the cause, and suggests a solution.

This requires multiple levels of understanding: low-level object and text recognition, mid-level interpretation of depicted content, and high-level comprehension of meaning in the context of the question. Modern VLMs handle all levels.

Understanding spatial relationships: "the button to the right of the input field," "the chart at the bottom of the screen," "the text above the image" — these are natural for humans and are increasingly well understood by VLMs.

### Visual Content Generation

Modern AI systems can generate images. DALL-E, Midjourney, and Stable Diffusion create images from text descriptions. Integration into agents opens new possibilities: an agent not only describes a solution but also draws a diagram; not only recommends a design but also creates a prototype.

Image generation requires caution: models can create unwanted content, violate copyrights, or generate images of real people without consent. Filters and policies are necessary.

## Multimodal Agents

### Modality Integration

True multimodality is not merely having channels but integrating them. A user explains a problem with words, then shows a screenshot, then asks by voice "do you see this button on the right?" The agent connects information from different modalities into a unified understanding.

Cross-modal reference resolution is the understanding of references between modalities. "This" refers to an object in the image. "As I already said" refers to text from a previous message. "Right here" refers to a location indicated by a gesture.

Modalities complement and reinforce each other: voice conveys urgency and emotion, an image shows what is difficult to describe in words, text precisely conveys code or a formula. A multimodal agent selects the most appropriate modality for each part of the response.

### Response Modality Selection

An intelligent agent selects the most appropriate modality for the situation. A voice question on the go — the response is voice-based and brief. Working at a computer with code — a text response with formatting.

Context determines the preferred modality: in a car — voice; when working with documents — text with copy capability; when discussing design — visual examples; when explaining a process — step-by-step images.

User preferences play a role. Some prefer reading, others prefer listening. Some perceive information better visually, others verbally. An adaptive agent learns the preferences of a specific user.

### Multimodal Memory and Context

Multimodal context is more complex to store and process. Images consume more memory. Audio requires transcription for search. Video is a combination of images and sound with temporal alignment.

Efficient compression: images are replaced with descriptions, retaining only key visual elements; audio is replaced with transcription annotated with emotions and intonation; video is reduced to key frames and summarization.

Preserving connections between modalities is important. "That photo I showed yesterday" — the agent must understand which image is being referenced, even if it has been removed from the immediate context.

## Voice Interface Design Specifics

### Conversational Design for Voice

Voice interfaces require a distinct approach. What reads naturally may be difficult to perceive by ear.

**Brevity** is the key principle. Voice responses are significantly shorter than text responses. Where text contains five points, voice limits to three. Long lists are broken into parts with confirmation: "Would you like to hear the remaining options?"

**Structure** aids comprehension: "I found three options. First... Second... Third..." Transitional phrases ("moving on," "on another note," "in conclusion") help orient the listener within the speech flow.

**Repeating important information** compensates for the ephemeral nature of speech. A phone number, address, or meeting time should be repeated or offered via text: "Ready to write it down? Your code is: 4-5-2-8. Again: 4-5-2-8."

### Error Handling in Voice Interfaces

Recognition errors are inevitable: unclear speech, background noise, and specialized terms absent from the vocabulary. The system handles uncertainty gracefully.

The re-prompting strategy is unobtrusive: "Did you say 'Moscow'?" for confirming critically important data. "Could you repeat that?" for unintelligible phrases. "If I understood correctly, you want to..." for paraphrasing complex requests.

Graceful degradation: if voice input fails, offer text input; if speech synthesis is unavailable, display text on screen; if the network is unstable, use offline models.

Explicit confirmations for critical actions: "You want to cancel an order totaling $150. Say 'yes' to confirm." Important and irreversible actions require explicit consent, possibly using a specific confirmation word.

### Voice Agent Persona

Voice creates a stronger impression of personality than text. Timbre, speech rate, intonation patterns, and vocabulary shape the perception.

Voice selection should match the tasks and brand: a medical assistant — calm, inspiring trust; a youth-oriented service — energetic and informal; a financial advisor — confident and professional.

Persona consistency matters. An agent should not suddenly change speech style or formality. If it introduced itself with humor, it should not become dry and official. The persona is deliberate and consistent.

Cultural adaptation goes beyond translation: speech rate, acceptable pauses, directness or indirectness of expression, and use of formal address vary between cultures and must be considered during localization.

## Safety and Ethics of Multimodal Systems

### Privacy Protection

Voice and images are biometric data. Voice identifies a person, reveals emotional state and medical conditions. Images contain faces, document numbers, and personal information in the background.

Minimizing data collection is the first principle. Collect only what is necessary, store for the minimum time, delete as early as possible. If a transcription suffices — do not store the audio. If extracted information suffices — do not store the image.

Transparency about data usage. Users must understand whether their voice is being recorded, whether photos are stored, and whether data is used for model training. Consent must be explicit and informed.

On-device processing where possible. Processing voice and images on the device minimizes transmission of sensitive data. Edge AI is becoming more powerful and enables complex tasks to be performed locally.

### Countering Misuse

Voice cloning opens opportunities for fraud: social engineering, spoofing voice authentication, and creating compromising content.

Watermarking and detection of synthesized speech are actively developing fields. Invisible markers in synthesized audio allow origin determination. Deepfake detectors analyze audio for signs of artificial origin.

Visual content moderation prevents generation and processing of unwanted material. Input filters block problematic content before processing. Output filters verify generated images.

### Accessibility and Inclusivity

Multimodal interfaces must account for user diversity. A voice interface is useless for deaf users; a visual interface is useless for blind users. A well-designed system offers alternatives.

Alternative text for images, subtitles for audio, and the ability to switch between modalities are elements of accessible design. The agent asks about preferences and adapts to the user's capabilities.

Accounting for speech diversity is especially important. Accents, dialects, speech impairments, and atypical speech rates — recognition systems must work for all users, not only for those who speak "standard" language. Bias in training data leads to bias in the system.

## Key Takeaways

Multimodality is the natural direction of AI agent evolution. Human communication is inherently multimodal; restricting it to text impoverishes the interaction.

Native speech-to-speech is a paradigm shift. OpenAI Realtime API and Gemini Live process audio directly — no ASR/TTS pipeline needed. Sub-200ms latency enables truly natural conversation. The traditional pipeline remains valuable for debugging, compliance, and complex workflows, but the direction is clear.

Voice interfaces require a fundamental rethinking of design. Brevity, structure, error tolerance, and natural interruptions distinguish voice interaction from text interaction. Latency is a critical factor: hundreds of milliseconds determine the difference between a natural conversation and an irritating wait.

Vision-language models open new scenarios. Showing instead of describing, seeing context that cannot be conveyed in words, and combining visual and textual information expand agent capabilities.

Modality integration matters more than modality availability. A multimodal agent connects information from different channels into a unified understanding, selects the appropriate modality for responses, and maintains cross-modal context.

Safety and ethics gain a new dimension. Voice and images are biometric data requiring special protection. Voice cloning and deepfake technologies create risks of misuse. Accessibility and inclusivity require support for alternative modalities.

Technical challenges: minimizing latency, ensuring operation in noisy conditions, adapting to various accents and speech styles, and efficiently storing and retrieving multimodal context. Solving these challenges determines the quality of user experience.

## Practical Implementation

A multimodal agent combines speech recognition, a language model with image support, and speech synthesis. The processing pipeline includes: audio transcription via a recognition API (Whisper), multimodal query formation combining text and images, response generation via a vision model, and speech synthesis from the text response.

Speech recognition integration requires audio preprocessing (volume normalization, noise suppression, format conversion), asynchronous processing with timeouts, and retries. The result contains not only text but also metadata: language, confidence level, and temporal segments.

Speech synthesis converts the text response into natural speech. APIs provide multiple voices with different characteristics. Optimization includes caching frequently used phrases (greetings, confirmations) and streaming synthesis for long texts (splitting into sentences, playback as segments become ready).

Computer vision integration uses a multimodal language model (GPT-4o, Claude Sonnet 4) that accepts combinations of text and images. Query formation includes text parts and images in the correct order. Images are passed as base64-encoded data or URLs.

Multimodal memory stores conversation history including images and their context. Direct storage of all images requires significant memory. Summarization strategy: a VLM creates a text description of the image that is stored instead of the full image, and thumbnails are stored for quick access.

Adapting responses for voice differs from text: long responses are shortened, Markdown formatting is removed, natural pauses are added through punctuation, and the agent's persona style is applied (formal, friendly, empathetic).

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Conversational AI
**Previous:** [[02_Conversation_Design|Conversation Design]]
**Next:** [[../10_Fine_Tuning/01_Fine_Tuning_Basics|Fine-Tuning Basics]]

---
