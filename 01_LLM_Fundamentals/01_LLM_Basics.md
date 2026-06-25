# Large Language Model (LLM) Fundamentals

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[../00_Home|Home Page]]
**Next:** [[02_Tokenization|Tokenization: How Text Becomes Numbers]]

---

Imagine a library that doesn't just store texts but has "absorbed" their content, internalized language patterns, authorial styles, and reasoning logic. That is essentially what a large language model is — a system that has learned to understand and generate human language.

Large Language Models (LLMs) have become a revolution in artificial intelligence. They form the foundation of modern AI agents and determine their ability to understand user queries, reason, generate code, and interact with the external world.

---

## What Is an LLM and Why Are They "Large"

A language model is a statistical model that predicts the probability of the next word (token) in a sequence based on the preceding words. Behind this simple formulation lies tremendous power.

The "large" prefix refers to the scale of these models. Modern LLMs contain billions of parameters — numbers that the model has "learned" during training on massive volumes of text data. GPT-4, for example, is estimated to contain over a trillion parameters.

But it is not just about the number of parameters. These models are trained on unprecedented volumes of data — trillions of text tokens, including books, scientific papers, code, and web pages. This allows them to learn not only grammar and syntax but also factual knowledge, logical relationships, and elements of common sense.

---

## The Transformer Architecture: A Revolution in Language Processing

### The World Before Transformer

Before 2017, the primary tools for processing text sequences were recurrent neural networks (RNNs) and their improved variants — LSTMs and GRUs. These architectures processed text sequentially, word by word, passing a "memory" of what had been read from one step to the next.

Imagine reading a book through a small window that shows only one word at a time. You must remember everything you have read in a single "compressed" mental image. The further you progress, the harder it becomes to retain important details from the beginning of the book. This is exactly the problem RNNs faced — information "faded" over long sequences.

Furthermore, sequential processing made training slow. It was impossible to begin processing the tenth word until the first nine had been processed. In the era of powerful GPUs, this was a serious limitation.

### Attention Is All You Need

In 2017, researchers at Google published the paper "Attention Is All You Need." They introduced the Transformer architecture, which solved both problems simultaneously — memory decay and sequential processing.

The key idea behind the Transformer: instead of passing information sequentially, allow each word to directly "look at" all other words in the text and decide which ones to attend to. This is the Self-Attention mechanism.

Returning to the book analogy: instead of a small window, you can now see the entire page at once. When reading the word "he," you can instantly look back and understand that "he" refers to "Dr. Watson," mentioned three sentences earlier. Moreover, you do this for all words simultaneously, in parallel.

### How the Attention Mechanism Works

The attention mechanism can be thought of as a query system over a database. Self-Attention implements a soft (differentiable) lookup over a key-value store. Each token simultaneously serves as both a record in the "database" and a query to it.

For each word (token) in the text, three vectors are created:

**Query** — "what am I looking for?" A vector describing what information the given token wants to obtain from other tokens. For example, if a pronoun "he" is being processed, its Query might implicitly encode the question "who is the subject?"

**Key** — "what can I offer?" Each token has a key describing what information it can provide to other tokens. The word "Watson" might have a key associated with the concept "person, male, subject."

**Value** — "what exact information do I transmit?" The actual content that a token passes to those attending to it.

The process: the Query of each token is compared with all Keys in the sequence. The more similar the Query and Key, the higher the "attention score." These scores are normalized through softmax, turning them into weights. The final result for each token is a weighted sum of all Values, where the weights are determined by the attention scores. The result is divided by the square root of the dimension (scaled dot-product attention) to prevent excessively large or small values that could destabilize training.

### Multi-Head Attention: Looking from Different Perspectives

A single attention mechanism is useful, but what if a word needs to be analyzed from different perspectives? The word "bank" can simultaneously relate to "finance" (a financial institution), a "river" (the edge of a waterway), or a "pool" (a reserve of resources) depending on context.

Multi-Head Attention solves this problem by running multiple attention mechanisms in parallel. Each "head" can learn to focus on different aspects: one on syntactic relationships, another on semantic ones, a third on proximity relationships in the text. The results from all heads are concatenated and projected back into a unified representation.

Modern models use dozens or even hundreds of attention heads. GPT-4 is estimated to use 96 heads in each of its 96 layers (based on community analysis — OpenAI has not officially disclosed GPT-4's architecture).

### Multi-Head Attention Optimizations: MQA and GQA

Standard Multi-Head Attention has a significant drawback: each head requires separate Key and Value matrices, creating enormous memory requirements during inference.

**Multi-Query Attention (MQA)**, proposed by Google in 2019, solves this problem radically: all heads share a single Key-Value pair while maintaining separate Queries. This reduces the KV-cache size by a factor of num_heads (e.g., 32x). However, quality suffers slightly — a single KV pair cannot capture the full diversity of attention patterns.

**Grouped Query Attention (GQA)** is the middle ground, introduced in 2023. Heads are organized into groups, and each group shares a single KV pair. For example, with 32 heads and 8 groups, every 4 heads share one Key-Value pair.

| Method | Query Heads | KV Heads | KV-cache | Quality |
|--------|-------------|----------|----------|---------|
| MHA    | 32          | 32       | 100%     | Baseline |
| GQA-8  | 32          | 8        | 25%      | ~Baseline |
| MQA    | 32          | 1        | 3%       | Slightly lower |

GQA is the standard attention mechanism for production models. Llama 2 70B, Llama 3, Mistral, Mixtral — all use GQA. Note that frontier models in 2025-2026 increasingly use Mixture of Experts (MoE) architectures (DeepSeek V3, Llama 4, Qwen 3), but GQA remains the standard attention mechanism within each expert.

### Causal Attention: No Peeking into the Future

During text generation, the model must predict the next word based only on the preceding ones. It would be "cheating" to look at words that have not yet been generated. For this purpose, a causal mask — an attention mask — is used.

Imagine a triangular matrix: a token at position 5 can "see" tokens at positions 1, 2, 3, 4, and 5, but cannot see tokens 6, 7, and beyond. This is achieved by setting attention scores for "future" tokens to negative infinity before softmax — after normalization, their weights become zero.

---

## Feed-Forward Networks: Where Knowledge Is Stored

After the attention layer, each token passes through a fully connected neural network (feed-forward network, FFN). This consists of two linear transformations with a nonlinear activation between them.

The FFN can be thought of as the model's "memory." If the attention mechanism is responsible for determining what information to gather from the context, the FFN is responsible for interpreting that information using knowledge accumulated during training.

Research shows that factual knowledge is stored in the FFN weights. When the model "knows" that Paris is the capital of France, that knowledge is encoded in the FFN parameters. Some researchers have even learned to locate and edit individual facts by modifying specific FFN weights.

The FFN size is typically 4 times the model dimension. For a model with dimension 4096, the FFN will have an intermediate layer of size 16384. This is a massive number of parameters, and they constitute the bulk of the model's "weight."

---

## Transformer Layers: Depth of Understanding

A single Transformer layer combines Multi-Head Attention, FFN, normalization, and residual connections. Modern models stack dozens of such layers on top of one another.

### Residual Connections: Why Deep Networks Work at All

Residual connections (skip connections) are one of the key innovations that made training very deep networks possible. The idea is simple: instead of having a layer compute the desired output directly, it computes only the difference (residual) between the input and the desired output: `y = x + F(x)` instead of `y = F(x)`.

Why is this critical? During backpropagation, the gradient must pass through all layers of the network. In deep networks without skip connections, gradients either vanish or explode.

Residual connections create "shortcuts" for the gradient. Even if the gradient through F(x) has decayed to zero, it will still flow through the direct connection (+x). Mathematically: the gradient with respect to x equals 1 + dF/dx, not just dF/dx. The constant 1 guarantees that at least a portion of the gradient reaches the earlier layers.

### Normalization: Layer Norm vs RMSNorm

Layer normalization stabilizes activations, preventing them from exploding or vanishing. The original Transformer used Layer Normalization (LN), which normalizes across all features for each token: `LN(x) = γ · (x - μ) / σ + β`.

However, modern models (Llama, Mistral, Qwen) have switched to **RMSNorm** — a simplified version: `RMSNorm(x) = γ · x / RMS(x)`, where `RMS(x) = √(mean(x²))`.

Why has RMSNorm become the standard?
1. **Efficiency**: it does not require computing the mean or subtraction — saving ~10-15% compute in the normalization layer.
2. **Stability**: empirically yields comparable or better training quality.
3. **Simplicity**: fewer trainable parameters (no bias β).

Each successive Transformer layer builds increasingly abstract representations. Early layers may focus on basic syntactic structures, middle layers on semantic relationships, and deep layers on complex logical reasoning and inference.

---

## Positional Encoding: Understanding Word Order

The attention mechanism in its basic form does not account for token positions — for it, "the cat ate the mouse" and "the mouse ate the cat" would look identical. To enable the model to understand word order, positional encoding is used.

The original Transformer paper used sinusoidal positional encodings — a mathematically elegant approach where position is encoded using a combination of sines and cosines at different frequencies. This allows the model to easily compute relative positions and even extrapolate to sequences longer than those seen during training.

Modern models often use more advanced techniques. **RoPE (Rotary Position Embedding)**, used in Llama, encodes positions through vector rotation in complex space. **ALiBi (Attention with Linear Biases)** adds a linear distance penalty between tokens directly to the attention scores. These techniques show better results when working with long contexts.

---

## Scale of Modern Models

Modern LLMs are staggering in their scale. The table below compares key architectural parameters across generations of frontier models, illustrating the shift from fully disclosed dense architectures (GPT-3) to opaque Mixture-of-Experts designs where only total and active parameter counts are public. Understanding these numbers helps calibrate expectations about memory requirements, inference cost, and the practical feasibility of self-hosting.

| Model | Parameters | Layers | Dimension | Attention Heads |
|-------|-----------|--------|-----------|-----------------|
| GPT-3 | 175B | 96 | 12288 | 96 |
| GPT-4 | ~1.7T (estimated MoE) | - | - | - |
| GPT-5 | undisclosed (MoE) | - | - | - |
| Claude Opus 4.7 | undisclosed | - | - | - |
| Gemini 2.5 Pro | undisclosed (MoE) | - | - | - |
| Llama 4 Maverick | 400B (17B active, 128 experts) | - | - | - |
| DeepSeek V3 | 671B (37B active) | 61 | 7168 | 128 |

Simply increasing model size does not guarantee improved quality. There are "scaling laws" that determine the optimal ratio between model size, data volume, and compute budget. A model with fewer parameters trained on more data can outperform a much larger model that was insufficiently trained.

---

## Scaling Laws and Emergent Abilities

Model quality follows predictable mathematical laws when scaling — larger models trained on more data improve in regular, measurable ways. The Chinchilla law (Google DeepMind, 2022) established that the optimal ratio is approximately 20 tokens per parameter. Modern practice deliberately over-trains smaller models (Llama 3 8B used 2,000 tokens per parameter — 100x the Chinchilla optimum) to reduce inference cost while maintaining quality.

Beyond predictable scaling, models exhibit **emergent abilities** — skills that appear abruptly once a certain scale is reached. Multi-step reasoning, code generation, and complex instruction following do not improve gradually but emerge as qualitative jumps. Whether this reflects genuine phase transitions or measurement artifacts is actively debated.

A new scaling dimension emerged in 2024-2025: **inference-time compute** (test-time compute). Models like o3 generate chains of reasoning tokens before the final answer, dramatically improving performance on complex tasks. This means quality can be improved by spending more compute at inference, not just at training.

These topics — scaling laws, emergence, over-training, and inference-time compute — are covered in depth in [[08_Scaling_Laws|Scaling Laws]].

---

## In-Context Learning: Learning Without Training

Another remarkable ability of modern LLMs is in-context learning. The model can "learn" a new task during query processing, without modifying its weights.

Show the model a few examples of a new response format — and it starts following that format. Give it a few examples of translation into a rare language — and it continues translating in the same style. This is called few-shot learning.

In-context learning is the foundation of prompting techniques. Instead of fine-tuning the model for each task (which is expensive and complex), the prompt can simply be formulated with the appropriate examples. This democratized AI usage — one no longer needs to be an ML engineer to adapt a model to specific needs.

---

## Connection to AI Agents

Understanding the Transformer architecture is critically important for developing effective AI agents:

**Context window** is the maximum number of tokens the model can process at once. For an agent, this determines how much conversation history, instructions, and context it can take into account.

**The attention mechanism** allows the model to focus on relevant parts of a long context. When an agent selects a tool from a list, attention helps it match the user's request to the appropriate tool.

**Emergent abilities** make complex planning, task decomposition, and tool use possible. An agent can break a complex task into steps and execute them sequentially.

**In-context learning** enables rapid adaptation of an agent to new domains and formats through prompting, without expensive fine-tuning.

---

## Key Takeaways

1. **LLMs are neural networks with billions of parameters**, trained to predict the next token in text. This simple task, at sufficient scale, gives rise to complex abilities.

2. **The Transformer architecture** revolutionized language processing. The Self-Attention mechanism allows each token to "see" the entire context, and parallel processing makes training efficient.

3. **Multi-Head Attention** allows the model to analyze text from multiple perspectives simultaneously — syntactic, semantic, and logical.

4. **Scaling laws** determine the optimal ratio of model size to data. Chinchilla demonstrated the importance of balancing N and D. Modern practice involves over-training smaller models for inference efficiency.

5. **Scaling produces emergent abilities**. Upon reaching a certain size, models begin demonstrating skills they were never explicitly taught.

6. **In-context learning** allows adapting the model to new tasks directly in the prompt, without modifying weights.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[../00_Home|Home Page]]
**Next:** [[02_Tokenization|Tokenization: How Text Becomes Numbers]]
