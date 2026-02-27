# Tokenization: How Text Becomes Numbers

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[01_LLM_Basics|Large Language Model (LLM) Fundamentals]]
**Next:** [[03_Context_Windows|Context Windows and Their Management]]

---

Imagine you are trying to teach a computer to read a book. Computers do not understand letters and words — they only work with numbers. How do you bridge this gap? The answer lies in tokenization — converting human text into a sequence of numbers that a neural network can process.

Tokenization is the first and arguably one of the most underappreciated stages in any language model's operation. How text is split into tokens affects virtually everything: API request costs, context window limits, prompt efficiency, and even some "strange" errors that LLMs produce.

For AI agent developers, understanding tokenization is not merely academic knowledge. It is a practical skill that helps save money on APIs, write more efficient prompts, and understand why a model sometimes behaves unexpectedly.

---

## Why Not Simply Use Characters or Words?

### Character-Level Approach: Too Much, Too Little

The most obvious method would seem to be splitting text into individual characters. The English alphabet contains only 26 letters (plus digits and punctuation), so the vocabulary would be compact. But this approach has a critical flaw.

Consider the word "artificial." In a character-level representation, this is 10 separate tokens: a-r-t-i-f-i-c-i-a-l. Each token is a single letter carrying minimal semantic information. To understand that these 10 characters together mean "artificial," the model must learn complex patterns over very long sequences.

When training on a book with a million words, the sequence stretches to millions of characters. The Transformer attention mechanism has quadratic complexity — doubling the sequence length increases computational cost by a factor of four. The character-level approach simply does not scale.

### Word-Level Approach: Too Many Unique Entries

The other extreme is splitting text by words. Each word becomes a separate token. The approach is intuitive but creates its own problems.

How many words exist in the English language? Estimates range from 170,000 to over a million when counting technical terms, slang, and neologisms. And if you add Russian, Chinese, Arabic, and hundreds of other languages? The vocabulary inflates to unimaginable sizes.

But even a massive vocabulary does not solve the main problem — new words. "ChatGPT" did not exist in any vocabulary before 2022. "Covidiot" is a 2020 neologism. What do you do with such words? Mark them as "unknown" and lose all information? That is unacceptable.

Furthermore, word-level tokenization ignores morphology. Words like "run," "runs," "running," "ran" are semantically related concepts but four entirely different tokens. It becomes harder for the model to capture these relationships.

### The Middle Ground: Subword Tokenization

Modern LLMs use a compromise approach — subword tokenization. Text is split not into characters or words but into fragments — morphemes, syllables, frequent character combinations.

Common words like "the," "and," "to" remain whole tokens — they appear millions of times in training data, and having dedicated representations for them is efficient. Rare words, however, are broken into smaller parts.

The word "tokenization" may become the sequence ["token", "ization"]. "Unbelievable" — ["Un", "believ", "able"]. Even an entirely new word absent from the vocabulary will be split into familiar parts, and the model can infer its meaning by analogy.

The vocabulary size remains reasonable — typically from 32,000 to 100,000 tokens. This is large enough for efficient representation of any text but not so large as to create memory or training issues.

---

## BPE: The Algorithm That Conquered the World

Byte Pair Encoding (BPE) is the most widely used tokenization algorithm in modern LLMs. It is used in OpenAI's GPT family and many other models. The algorithm's idea is elegant in its simplicity.

### How BPE Is Trained

Imagine you have a text corpus and want to create a token vocabulary. BPE starts with a base set — all individual characters (or bytes). The algorithm then iteratively improves the vocabulary:

1. Count which pairs of adjacent tokens appear most frequently
2. Merge the most frequent pair into a new token and add it to the vocabulary
3. Replace all occurrences of that pair in the corpus with the new token
4. Repeat until the desired vocabulary size is reached

Suppose the sequence "th" appears frequently in the text. After several iterations, BPE will merge "t" and "h" into the token "th." Then, perhaps "th" and "e" will become "the." Popular words gradually become whole tokens, while rare ones remain split into parts.

### Determinism and Consistency

An important property of BPE is determinism. The same text is always tokenized identically. This is critical for experiment reproducibility and predictable model behavior.

Moreover, merge rules are preserved in the order they were created. When tokenizing new text, the rules are applied in the same order, guaranteeing consistency between model training and inference.

### Byte-Level BPE: Why GPT Works with Bytes

The original BPE operates at the Unicode character level. However, GPT-2 and subsequent OpenAI models use **Byte-level BPE** — a variant that operates on raw UTF-8 bytes.

Why does this matter? Unicode contains over 140,000 characters — from Latin script to Egyptian hieroglyphs. If BPE starts from characters, the base vocabulary is enormous. When working with bytes, the base vocabulary is always exactly **256 elements** (all possible byte values from 0 to 255).

**Byte-level advantages:**
- **Universality**: Any text in any language is guaranteed to be tokenized. There is no concept of an "unknown character."
- **Compact base vocabulary**: 256 vs. 100,000+ Unicode characters.
- **Resilience to emoji and special characters**: 🎉 is simply broken into UTF-8 bytes.

**Disadvantages:**
- **Less intuitive**: A single Unicode character can become multiple tokens (e.g., a Cyrillic letter = 2 UTF-8 bytes).
- **Efficiency for non-ASCII**: Languages with multi-byte characters require more tokens.

Character-level BPE (as in early models) is easier to understand, but byte-level won out due to its universality. GPT-3, GPT-4, Claude — all use the byte-level approach.

---

## Other Tokenization Algorithms

### WordPiece: Google's Alternative

WordPiece is used in BERT and other Google models. The main difference from BPE is the criterion for selecting pairs to merge. Instead of simple frequency, WordPiece selects the pair that maximizes the likelihood of the training data under the language model.

This is a "smarter" selection that accounts not only for frequency but also for impact on model quality. In practice, BPE and WordPiece results are often similar, but WordPiece can be more effective for certain tasks.

A distinctive feature of WordPiece is the "##" prefix for word continuations. The word "embedding" may be tokenized as ["em", "##bed", "##ding"]. The prefix indicates that the token is a continuation of the previous one rather than the start of a new word.

### SentencePiece: A Universal Approach

Google's SentencePiece addresses an important problem — language dependence. Traditional tokenizers assume that words are separated by spaces, which holds for English or Russian but not for Chinese or Japanese.

SentencePiece works directly with "raw" Unicode text without prior segmentation. Spaces are treated as regular characters, represented by the special symbol "▁". This makes the tokenizer truly language-independent.

The text "Hello world" may become ["▁Hello", "▁world"]. The "▁" symbol marks the beginning of a word (or the space before it), enabling exact reconstruction of the original text during detokenization.

### Tiktoken: OpenAI's Modern Choice

In 2022, OpenAI introduced **tiktoken** — a new tokenization library written in Rust. It is not a new algorithm but a highly optimized implementation of Byte-level BPE.

**Why did OpenAI move away from Python-based solutions?**

1. **Speed**: Tiktoken is 3-6x faster than SentencePiece on typical tasks. For production systems processing millions of requests, this is critical.

2. **Parallelism**: Rust enables efficient multi-threaded processing. Batches of texts can be tokenized in parallel.

3. **Memory predictability**: No Python garbage collector, fewer memory usage spikes.

4. **Accuracy**: Identical results with OpenAI models — guaranteed compatibility.

| Characteristic | SentencePiece | Tiktoken |
|----------------|---------------|----------|
| Language | C++ | Rust |
| Speed | Baseline | 3-6x faster |
| Models | Llama, T5 | GPT-4o, GPT-5, o3 |
| Algorithms | BPE, Unigram | Byte-level BPE |

For working with OpenAI models, use tiktoken. For Llama and open-source models, use SentencePiece or their specific tokenizers via HuggingFace Transformers.

---

## Special Tokens: Beyond Text

Every modern tokenizer includes a set of special tokens — service markers that do not correspond to actual text but carry important information for the model.

**Structural tokens:**
- `<BOS>` or `<s>` — Beginning of Sequence
- `<EOS>` or `</s>` — End of Sequence
- `<PAD>` — Padding for length alignment
- `<UNK>` — Unknown token

**Chat model tokens:**
- `<|im_start|>`, `<|im_end|>` — message start and end
- `<|system|>`, `<|user|>`, `<|assistant|>` — conversation roles
- `<|endoftext|>` — end of text (GPT)

These tokens are particularly important for chat models and agents. Proper role markup allows the model to understand conversation structure and generate responses in the correct context.

When you use the OpenAI or Anthropic API, the library automatically adds these special tokens. But understanding their existence is important for debugging and prompt optimization.

---

## Practical Aspects: Counting and Saving Tokens

### How Much Does a Token Cost?

Understanding token costs is critically important for budgeting AI projects. LLM providers charge precisely by token count — both input and output.

For English text, the empirical rule applies: **1 token ≈ 4 characters** or **1 token ≈ 0.75 words**. A page of text (~500 words) occupies approximately 650-750 tokens.

For Russian text, the ratio is worse due to Cyrillic. Cyrillic characters appear less frequently in training data (which is predominantly English), so they are split into smaller fragments. In practice: **1 token ≈ 2-3 characters** for Russian text. The same text in Russian requires more tokens than in English.

This has direct financial implications. At a price of $10 per million tokens, Russian text costs roughly 1.5-2x more than its English equivalent.

### Optimizing Token Usage

Understanding tokenization opens opportunities for prompt optimization:

**Removing redundancy.** Extra spaces, empty lines, repeated instructions — all consume tokens. The phrase "Please kindly consider" takes more tokens than simply describing the task.

**Language choice.** For some tasks, it may be more cost-effective to formulate prompts in English even if the target language is Russian. English instructions consume fewer tokens.

**Contractions and abbreviations.** "do not" and "don't" produce different token counts. In long prompts, such details accumulate.

**Structured formats.** JSON is often more compact than equivalent verbal descriptions. Instead of "The user's name is John and their age is 25," you can use `{"name": "John", "age": 25}`.

---

## Code Tokenization Specifics

Program code is tokenized differently from natural language, and this is a source of interesting observations.

**CamelCase gets split.** The word `getUserName` becomes multiple tokens: ["get", "User", "Name"]. The model understands the constituent parts, which helps generate meaningful function names.

**Snake_case is less efficient.** The same meaning in the form `get_user_name` takes more tokens: ["get", "_", "user", "_", "name"]. Underscores are separate tokens.

**Indentation counts as tokens.** Four spaces for indentation constitute one or more tokens. In deeply nested code, indentation can consume a significant portion of the context. This is one reason why models sometimes "forget" to close brackets in long functions.

**Numbers are unpredictable.** "123" may be a single token, while "1234" becomes two: ["12", "34"]. "12345" might be ["123", "45"]. This instability is one reason LLMs perform poorly at arithmetic. The model "sees" different patterns for similar numbers.

---

## Tokenization Problems and Their Consequences

### Language Inequality and Tokenizer Fertility

Tokenizers are trained predominantly on English-language data. The result: English is represented maximally efficiently, while other languages suffer.

To quantify this inequality, the **Tokenizer Fertility** metric is used — the average number of tokens per word or character for a given language:

**Fertility = token_count / word_count**

| Language | Fertility (GPT-4) | Relative Cost |
|----------|-------------------|---------------|
| English | 1.3 | 1.0x (baseline) |
| Spanish | 1.8 | 1.4x |
| Russian | 2.4 | 1.8x |
| Chinese | 2.1 | 1.6x |
| Hindi | 3.2 | 2.5x |
| Thai | 4.1 | 3.2x |

The same meaning in different languages:
- "Hello, how are you?" — approximately 6 tokens
- "Привет, как дела?" (Russian) — approximately 10-12 tokens
- "你好，你好吗？" (Chinese) — approximately 8-12 tokens

**Why fertility matters:**

1. **Economics**: High fertility = more tokens = higher API costs.

2. **Effective context**: With the same token limit (e.g., 128K), less semantic content fits in Russian.

3. **Model quality**: Research shows a correlation between fertility and generation quality — languages with high fertility produce worse benchmark results.

4. **AI fairness**: This creates inequality — users of non-English languages pay more for lower quality.

The solution is training tokenizers on more balanced multilingual corpora. Llama 3 and Qwen 2.5 models show improved fertility for non-English languages thanks to targeted work on this aspect.

### Rare Words and Names

Rare words are split into many small tokens. The name "Schwarzenegger" might become five tokens: ["Sch", "war", "zen", "eg", "ger"]. A long Ukrainian place name could be split into a dozen parts.

This creates an interesting effect: models are worse at "remembering" rare names and terms. By the end of a long text, the model may "forget" the beginning of a complex name because it is represented by disjointed tokens with no explicit connection.

### Arithmetic and Numbers: Why LLMs Are Bad at Math

The instability of number tokenization is one of the **primary reasons** for LLM weakness in mathematics. This is not a flaw in architecture or training — it is a fundamental limitation of the representation.

**Problem 1: Inconsistent splitting**

When "380 + 427" is tokenized as ["380", " +", " ", "42", "7"], the model does not "see" the number 427 as a whole. Different numbers are split differently:
- "12" → ["12"] (1 token)
- "123" → ["123"] (1 token)
- "1234" → ["12", "34"] (2 tokens)
- "12345" → ["123", "45"] (2 tokens)

The model has learned patterns for "12" and "34" separately, but their combination "1234" is an entirely different representation.

**Problem 2: Loss of positional information**

In the decimal system, digit position is critical: the 5 in 500 means 5×100, while in 50 it means 5×10. But tokenization destroys this structure. The token "500" is an atomic unit — the model does not "know" that it contains three positions.

**Problem 3: Rarity of large numbers**

Small numbers (1-100) appear millions of times in training data. Large numbers (e.g., 847293) appear extremely rarely. The model simply has not seen enough examples of arithmetic with large numbers.

**Experiments confirm:**
- LLMs perform better with numbers that are whole tokens
- Errors occur more often at token boundaries
- Adding spaces between digits (5 + 3 = 8 vs. 5+3=8) can change the result

**Solutions:**
1. **Chain-of-thought**: Step-by-step computation reduces errors
2. **Calculators as tools**: The agent calls an external calculator
3. **Specialized tokenizers**: Some research models use character-level tokenization for numbers

---

## Tokenization in the Context of AI Agents

For agent developers, tokenization affects several critical aspects:

### System Prompt Design

The system prompt is a baseline token expenditure on every request. Verbose instructions increase cost and reduce space for context.

Compare:
- "You are a helpful assistant that is designed to help users with their questions. You should always try to provide accurate and helpful responses." — verbose, many tokens
- "You are a helpful, accurate assistant." — same essence, far more compact

### Tool Descriptions

In agents that use tools, each tool's description consumes tokens. With dozens of tools, their descriptions can occupy thousands of context tokens.

The effective practice is minimalism in descriptions. Instead of verbose explanations, use brief one-line summaries of the tool's function, a list of required parameters with types, and a concise description of the return value. For example, for a web search tool, it suffices to state that it takes a string query as a required parameter and returns a list of objects with title, URL, and text snippet.

Avoid duplicating information the model can infer from context. Do not describe obvious things like "this tool is used to..." — get straight to the point.

### Conversation History Management

Conversation history accumulates tokens. At each step, user messages and agent responses are added. Without management, the context quickly overflows.

Management strategies:
- Trimming old messages (FIFO)
- History summarization
- Relevance-based selection

All these strategies rely on accurate token counting.

---

## Key Takeaways

1. **Tokenization is not a trivial process.** Text is converted to numbers through complex algorithms, and understanding this process helps write effective prompts.

2. **Subword tokenization (BPE, WordPiece, SentencePiece)** is the standard in modern LLMs. It balances vocabulary compactness with the flexibility to represent any text.

3. **Token count determines everything:** API request costs, processing speed, available context size. Saving tokens means saving money and resources.

4. **Different languages tokenize differently.** English is the most efficient; Russian and other languages require more tokens for the same meaning.

5. **Numbers and rare words are tokenization's weak spot.** This explains some "strange" LLM errors in arithmetic and name handling.

6. **For AI agents,** tokenization affects prompt design, tool descriptions, and context management strategies.

---

## Practical Implementation Aspects

### Token Counting in Production Systems

For effective context management, accurate token count estimation is critically important. In production systems, this is achieved through specialized tokenization libraries matching the specific model.

**The basic approach** involves creating a token counting utility that initializes the tokenizer for the chosen model (e.g., OpenAI or Claude) and provides methods for estimating both individual texts and entire message chains. This enables predictable context budget management.

**Prompt efficiency analysis** is based on measuring the character-to-token ratio. If a prompt shows 3.5+ characters per token, this indicates predominantly English text. A ratio of 2.5-3.5 suggests mixed content, and below 2.5 indicates a large amount of Cyrillic or other multi-byte characters. This metric helps optimize language choice for prompts.

### Cost Estimation Without API Calls

Real-world systems often need to estimate request cost before execution. For this, heuristics based on text language are used. English text is estimated as character length divided by 4. Russian — by 2.5. Chinese and Japanese — by 1.5. Program code typically falls in between with a coefficient of about 3.

**Cost calculation** accounts for both input and output tokens, since providers price them differently. For example, GPT-4o input tokens cost $2.50 per 1M, output $10 per 1M. For Claude Sonnet 4 — $3 and $15 per 1M respectively. Newer models like GPT-5 and Claude Opus 4.6 are priced higher; economy models like GPT-4o-mini ($0.15/$0.60 per 1M) and Claude Haiku 4.5 ($1/$5 per 1M) are dramatically cheaper. Multiplying token counts by prices and summing yields a realistic cost estimate.

Such preliminary estimation helps avoid unexpected expenses in production, especially when processing large volumes of text or lengthy conversations.

### Context Management Through History Trimming

A key task in agent systems is keeping conversation history within the model's context window. The naive approach of "add all messages" quickly leads to context overflow.

**FIFO (First-In-First-Out) strategy** retains recent messages while removing old ones. The algorithm works in reverse order: starting from the last message, messages are added to the trimmed history until the token sum reaches the available limit. The limit is calculated as the maximum context size minus system prompt tokens minus a reserve for the model's response.

This technique ensures the model always has access to the most recent messages, which is critical for maintaining dialogue coherence. Old messages, while lost, are typically less relevant to the current conversation context.

**Alternative approaches** include summarizing old messages (compressing long history into a brief summary) or semantic search over history (extracting only fragments relevant to the current query). The choice of strategy depends on the agent type and task characteristics.

### Formatting with Special Tokens

When working with chat models, it is important to understand how messages of different roles are formatted. Provider APIs typically abstract away this complexity, but for low-level model interaction, role markers must be explicitly added.

**Message structure** uses special tokens to mark the beginning and end of each block. A system message is wrapped in `<|system|>` ... `<|end|>`. A user message is wrapped in `<|user|>` ... `<|end|>`. An assistant response is wrapped in `<|assistant|>` ... `<|end|>` accordingly.

**For agents with tools,** additional special tokens are added to denote function calls. For example, `<|tool_call|>` wraps a JSON object with the tool name and arguments, and `<|end_tool_call|>` closes that block. This allows the model to unambiguously distinguish regular text from tool execution commands.

Correct formatting is critical for proper model behavior — incorrectly placed tokens can lead to misinterpreted roles or loss of dialogue structure.

**A basic token counting utility** is created by initializing the tokenizer for a specific model (e.g., OpenAiTokenizer) and providing methods for counting tokens in text. The counting method accepts a string and returns the token count after tokenization. An additional efficiency analysis method computes the character-to-token ratio, which helps optimize prompts.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[01_LLM_Basics|Large Language Model (LLM) Fundamentals]]
**Next:** [[03_Context_Windows|Context Windows and Their Management]]
