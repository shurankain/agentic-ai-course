# Generation Parameters: Controlling Model Behavior

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[03_Context_Windows|Context Windows and Managing Them]]
**Next:** [[05_LLM_Providers_and_API|LLM Providers and Working with APIs]]

---

Imagine a writer standing before the choice of the next word in a novel. Sometimes precision is required — when drafting a legal document or a technical manual. Sometimes creativity matters — in poetry or a film script. And sometimes balance is needed — vivid language without drifting into absurdity.

Generation parameters in language models are exactly these "tuning knobs" that control the degree of creativity and predictability in responses. Understanding these parameters is critically important for an AI agent developer: tool selection demands precision, while responding to a user demands lively language.

---

## How LLMs Generate Text

Before diving into parameters, it is important to understand the generation process itself. An LLM operates autoregressively — generating text token by token, where each subsequent token depends on all preceding ones.

At each step, the model does the following:

1. **Receives context** — all tokens generated up to this point (plus the original prompt)

2. **Computes logits** — for each token in the vocabulary, the model produces a number (logit) reflecting its "confidence" that this particular token should come next

3. **Converts logits to probabilities** — through the softmax function, logits are transformed into a probability distribution where all values are positive and sum to 1

4. **Selects the next token** — this is where generation parameters come into play

The process repeats until a special end-of-sequence token is generated or the token limit is reached.

---

## Temperature: Degree of Randomness

### What Is Temperature

Temperature is the primary parameter controlling the "randomness" of token selection. Technically, temperature divides the logits before applying softmax: each logit is divided by the temperature value, then softmax is applied to the result to obtain probabilities.

What happens at different temperature values?

**Temperature = 0 (or close to zero)**: The distribution becomes "sharp" — virtually all probability concentrates on the token with the maximum logit. The model always selects the most probable token. This is deterministic mode — the same prompt always produces the same response.

**Temperature = 1**: The original distribution as the model "sees" it after training. A balance between probable and less probable tokens.

**Temperature > 1**: The distribution "flattens" — even unlikely tokens get a chance of being selected. The model becomes unpredictable, sometimes creative, sometimes nonsensical.

### Visualizing the Effect

Suppose the model predicted the following logits for four tokens:
- "the": logit 2.0
- "a": logit 1.5
- "an": logit 1.0
- "this": logit 0.5

At different temperatures, the probabilities distribute as follows:

| Temperature | "the" | "a" | "an" | "this" |
|-------------|-------|-----|------|--------|
| 0.1 | 99.3% | 0.6% | 0.1% | 0.0% |
| 0.5 | 76.4% | 16.8% | 4.9% | 1.4% |
| 1.0 | 43.7% | 26.5% | 16.1% | 9.8% |
| 1.5 | 33.6% | 24.7% | 18.1% | 13.3% |
| 2.0 | 29.0% | 23.4% | 18.9% | 15.3% |

At temperature 0.1, the choice almost always falls on "the". At temperature 2.0, the distribution is nearly uniform — any token has a realistic chance.

### When to Use Which Temperature

**Temperature 0 (deterministic mode)**:
- Code generation where syntax must be exact
- Mathematical computations
- Data extraction from text
- Classification
- Tool selection in an agent

**Temperature 0.3-0.5 (low variability)**:
- Document summarization
- Translation
- Answering factual questions
- Generating tool arguments

**Temperature 0.7-0.8 (moderate variability)**:
- User-facing conversation
- Writing emails
- Explaining concepts
- Most assistant tasks

**Temperature 0.9-1.2 (high creativity)**:
- Creative writing
- Brainstorming ideas
- Generating alternatives
- Poetry and literary texts

### Relationship Between Temperature and Information Theory

Temperature has a deep connection to the concepts of entropy and perplexity:

**Entropy** of a probability distribution measures the uncertainty of selection. The entropy formula sums the product of each token's probability and the logarithm of that probability (with a negative sign). At temperature=0, all probability is on a single token, so entropy equals zero (no uncertainty). At high temperature, the distribution approaches uniform, and entropy is maximal.

**Perplexity** is the exponential of entropy (2 raised to the power H). Intuitively, perplexity indicates how many equally probable options the model is "choosing" from on average. A perplexity of 10 means the choice is equivalent to randomly selecting from 10 equally probable tokens.

Temperature directly controls the perplexity of generation: low temperature → low perplexity (confident selection), high temperature → high perplexity (many alternatives).

### Sampling vs Greedy: Exploration vs Exploitation

Why use sampling (random selection) at all, rather than always picking the most probable token (greedy)?

This is the classic **exploration vs exploitation** dilemma:

**Greedy (exploitation)**: Always take the best option according to the current estimate. Pros: stable, reproducible. Cons: gets stuck in local optima, generates monotonous text.

**Sampling (exploration)**: Sometimes select less probable options. Pros: discovers unexpectedly good solutions, produces diverse text. Cons: may select a poor token.

In practice:
- For **deterministic tasks** (code, data) — greedy: we need the single correct answer
- For **creative tasks** — sampling: diversity matters more than stability
- For **dialogue** — moderate sampling: lively language without absurdity

---

## Top-p (Nucleus Sampling): Adaptive Selection

### The Problem with Fixed Temperature

Temperature has a drawback: it affects all situations equally. But sometimes the model is highly confident about the next token (99% probability), and sometimes it is not (the distribution is uniform).

Consider the sentence: "Paris is the capital of ___". The model is nearly certain the next word is "France". Here, a temperature of 0.7 can already cause problems — why give less probable tokens a chance?

But in the sentence "My favorite ___ is..." there are many reasonable continuations: "dish", "activity", "place", "season". Here, a temperature of 0.7 may be too conservative.

### How Top-p Works

Top-p (also known as nucleus sampling) solves this problem adaptively. Instead of temperature, we specify a cumulative probability — and consider only the tokens whose combined probability reaches this threshold.

Algorithm:
1. Sort tokens by descending probability
2. Accumulate probabilities until the sum reaches top_p
3. Sample randomly from the resulting set (nucleus)

With top_p = 0.9, we consider the minimal set of tokens covering 90% of the probability mass.

### Adaptiveness of Top-p

Here is why top-p is "smarter" than temperature:

**Model is confident** (one token with probability 0.95):
- Top-p = 0.9 → only 1 token is considered
- Randomness is minimal, as it should be

**Model is uncertain** (five tokens at 0.18-0.20 each):
- Top-p = 0.9 → all 5 tokens are considered
- Randomness is high, reflecting the actual uncertainty

Top-p automatically adapts to the model's confidence, whereas temperature operates the same way in all situations.

### Recommended Top-p Values

- **Top-p = 0.5**: Very conservative, little variability
- **Top-p = 0.9**: Standard value, good balance
- **Top-p = 0.95**: More diversity
- **Top-p = 1.0**: All tokens are considered (effectively disabled)

---

## Top-k: Hard Limit

### How Top-k Works

Top-k is the simplest of the three parameters. It simply restricts the selection to the k tokens with the highest probability.

With top_k = 5, the model selects the next token only from the five most probable, ignoring all others.

### Drawback: Inflexibility

Top-k has a significant drawback — it does not adapt to the situation:

**Problem 1: top_k is too small**

When the model is uncertain, many tokens have similar probabilities. With top_k = 5, we may cut off good options.

Example: 10 tokens with probabilities of 10-12% each. Top_k = 5 cuts off half the reasonable options.

**Problem 2: top_k is too large**

When the model is confident, considering many tokens is pointless.

Example: one token with 95% probability, the rest at 0.1% each. Top_k = 50 includes a bunch of unlikely tokens, but they will rarely be selected anyway.

### When to Use Top-k

Top-k is useful as an additional "safety valve" in combination with other parameters. A value of top_k = 40-50 prevents selection of extremely unlikely tokens without overly restricting the choice.

---

## Min-P: Adaptive Threshold (2024)

### The Problem with Top-p and Top-k

Both classic methods have a drawback: they do not account for the **absolute** probability of tokens, only their rank or cumulative share.

Consider the following situation:
- Token A: 40% probability
- Token B: 30%
- Token C: 15%
- Tokens D-Z: 0.5% each

With top_p = 0.95, we include tokens A, B, C, and several from D-Z. But why include a token with 0.5% probability when the best one has 40%?

### How Min-P Works

**Min-P** sets a **relative probability threshold**: a token is included for consideration only if its probability is greater than or equal to min_p multiplied by the probability of the best token.

With min_p = 0.1 and the best token at 40% probability:
- Threshold = 0.1 × 0.4 = 4%
- Tokens with probability < 4% are discarded

This is **adaptive**: when the model is confident (best token at 90%), the threshold is high (9%). When uncertain (best at 20%), the threshold is low (2%).

### Advantages of Min-P

| Method | Adapts to confidence? | Accounts for absolute probabilities? |
|--------|----------------------|--------------------------------------|
| Temperature | No | Yes (via scaling) |
| Top-k | No | No |
| Top-p | Partially | No |
| **Min-p** | **Yes** | **Yes** |

Min-P is popular in the community for local models (llama.cpp, Ollama) and demonstrates better results on creative tasks than top-p when properly tuned.

**Recommended values:** min_p = 0.05-0.1

---

## Combining Parameters

In practice, parameters are often used together. The typical order of application:

1. **Temperature** is applied to logits
2. **Softmax** converts to probabilities
3. **Top-k** removes all tokens except the top k
4. **Top-p** removes tokens outside the nucleus
5. **Selection** from the remaining set

### Recommended Combinations

**For precise tasks (code, data)**: temperature = 0, top_p = 1.0 (irrelevant at temp=0), top_k = 1 (greedy decoding).

**For factual responses**: temperature = 0.3, top_p = 0.9, top_k = 40.

**For a chat assistant**: temperature = 0.7, top_p = 0.95, top_k = 50.

**For creative tasks**: temperature = 1.0, top_p = 0.98, top_k = 100.

---

## Other Important Parameters

### max_tokens

The maximum number of tokens in a response. A critically important parameter:

- Too small a value — the response gets cut off
- Too large — you pay for unused tokens (in some APIs)

Recommendations by task:
- Classification: 5-20 tokens
- Short answer: 50-100 tokens
- Explanation: 200-500 tokens
- Function code: 500-1000 tokens
- Article: 1000-2000 tokens

### stop sequences

Character sequences that cause generation to stop when encountered. Useful for structured responses:

- `"\n"` — stop after the first line
- `"```"` — stop after a code block
- `"\nObservation:"` — stop before the next step in an agent
- `"</answer>"` — stop after the closing tag

### presence_penalty and frequency_penalty

Penalties for repetition:

**presence_penalty**: A penalty for using a token that has already appeared in the text. The penalty is the same regardless of the number of repetitions. Helps avoid fixating on the same words.

**frequency_penalty**: A penalty proportional to the number of occurrences. The more often a token has been used, the lower its probability of being selected again.

### Mathematics of Repetition Penalties

Both penalties work by modifying logits **before** applying softmax. Each logit has two penalty terms subtracted: presence_penalty multiplied by a token appearance indicator (1 if the token has appeared, 0 otherwise), plus frequency_penalty multiplied by the number of token occurrences.

**Presence penalty** subtracts a fixed value from the logit if the token has appeared at least once. This is a "one-time" penalty — it does not matter whether the token appeared 1 time or 100 times, the penalty is the same.

**Frequency penalty** subtracts a value proportional to the number of occurrences. Each repetition increases the penalty.

| Scenario | Presence | Frequency | Effect |
|----------|----------|-----------|--------|
| Token appeared 1 time | -0.5 | -0.5 | -1.0 to logit |
| Token appeared 5 times | -0.5 | -2.5 | -3.0 to logit |
| Token appeared 10 times | -0.5 | -5.0 | -5.5 to logit |

### Repetition penalty (alternative)

Some frameworks (especially Hugging Face) use **repetition_penalty** — a multiplicative penalty instead of an additive one. If a token has already appeared, its logit is modified depending on its sign: positive logits are divided by repetition_penalty, negative logits are multiplied by it.

With repetition_penalty = 1.2, positive logits decrease (divided by 1.2), while negative logits become even more negative (multiplied by 1.2). This prevents repetition regardless of the logit's sign.

**Key difference:** Additive penalties (presence/frequency) operate in logit space, while the multiplicative penalty (repetition) scales them. At high values, the multiplicative penalty can be more aggressive.

Recommendations:
- To avoid looping: presence = 0.5, frequency = 0.5
- For technical text (term repetition is OK): presence = 0, frequency = 0.2
- For creative text: presence = 1.0, frequency = 0.8

---

## Generation Parameters in AI Agents

In agentic systems, different components require different generation settings.

### Tool Selection

When an agent decides which tool to use, this is a critical juncture where errors are costly. Calling the wrong API or querying the wrong database has real consequences.

Optimal settings: temperature = 0, max_tokens = 50-100, stop_sequences = ["\n"]. Determinism here matters more than diversity.

### Argument Generation

Tool arguments must be precise — correct JSON syntax, valid parameter values. Use temperature = 0, max_tokens = 200-500, stop_sequences = ["```"].

### Reasoning (Chain of Thought)

When an agent "thinks" about a problem, a small amount of variability can help find unconventional solutions. Recommended parameters: temperature = 0.5, top_p = 0.9, max_tokens = 500-1000, stop_sequences = ["Action:", "Final Answer:"].

### Response to User

The final response should be lively and natural, but not overly creative. Optimal configuration: temperature = 0.7, top_p = 0.95, max_tokens = 500-2000, presence_penalty = 0.5, frequency_penalty = 0.3.

---

## Impact on Cost and Performance

Generation parameters affect not only quality but also economics:

**max_tokens**: Directly impacts the cost of output tokens. Setting a reasonable limit is a direct cost saving.

**stop_sequences**: Proper stop sequences prevent generation of unnecessary text. If the model tends to append "Have questions? Feel free to ask!" — a stop sequence can cut that off.

**temperature = 0**: Some providers use caching for identical requests with temp=0. Potential savings on repeated calls.

---

## Key Takeaways

1. **Temperature** is the primary parameter for controlling randomness. 0 for precision, 0.7 for balance, 1+ for creativity.

2. **Top-p (nucleus sampling)** adaptively selects the number of tokens under consideration based on the model's confidence. Generally preferable to top-k.

3. **Top-k** strictly limits the number of candidates. Useful as an additional filter but less flexible.

4. **Combining parameters** provides better control. A typical combination: temperature + top_p + a reasonable top_k.

5. **Different stages of agent operation require different settings**. Tool selection — temperature=0, user response — temperature=0.7.

6. **Experimentation is essential**. Optimal parameters depend on the specific model, task, and user expectations.

---

## Practical Implementation

### How Parameters Are Applied in a Real Sampler

In production systems, all parameters are applied sequentially in a strict order. First, temperature scales the logits, making the distribution more or less "sharp". Then softmax converts logits into probabilities. After that, top-k removes all tokens except the k most probable. Next, top-p finds the minimal set of tokens covering the specified probability mass. Finally, a random selection is made from the remaining tokens according to their probabilities.

The key point: at temperature = 0, all other parameters are ignored, and the system simply selects the token with the maximum logit (greedy decoding). This guarantees determinism and reproducibility of results.

Top-k works by finding the k-th largest element: all tokens with probability below this threshold are zeroed out, and the remaining ones are renormalized so their sum equals one again.

Top-p requires sorting tokens by descending probability. Then a cumulative sum is accumulated until it reaches the topP threshold. All tokens beyond this "nucleus" are discarded, and the remaining ones are renormalized.

Final sampling uses the "roulette wheel" method: a random number between 0 and 1 is generated, then tokens are traversed while accumulating their probabilities until the sum exceeds the random number. That token is selected.

### Configuration for Different Agent Modes

In agentic systems, it is critically important to use different generation parameters at different stages of operation. This is not merely a recommendation but a necessity for reliable operation.

**Tool selection mode**: Temperature must be strictly 0 to guarantee deterministic selection. Max_tokens is limited to 100 tokens since tool names are short. The stop sequence is set to "\n" to prevent the model from generating explanations after the tool name. Repetition penalties are disabled since they are unnecessary for such short generation.

**Argument generation mode**: Also requires temperature = 0 for JSON accuracy. Max_tokens is expanded to 500, as arguments can be complex objects. Stop sequences include "```" (end of code block) and a double newline. This prevents generation of extraneous text after the arguments.

**Reasoning mode (Chain of Thought)**: A small degree of creativity is acceptable here — temperature = 0.5. Top-p is set to 0.9 for diversity of thought. Max_tokens is increased to 1000, as reasoning can be extensive. Stop sequences include "Action:" and "Final Answer:" to prevent the model from advancing to the next stage on its own. Small repetition penalties (presence = 0.3, frequency = 0.1) prevent the model from fixating on the same phrases.

**User response mode**: Temperature is raised to 0.7 for natural language. Top-p = 0.95 provides more freedom in word choice. Max_tokens up to 2000 for detailed explanations. Stop sequences are empty — the model decides when to finish. Repetition penalties are more significant (presence = 0.5, frequency = 0.3) to keep the text from becoming monotonous.

This differentiation of parameters allows the agent to be precise where precision is needed and natural where fluent communication matters.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[03_Context_Windows|Context Windows and Managing Them]]
**Next:** [[05_LLM_Providers_and_API|LLM Providers and Working with APIs]]
