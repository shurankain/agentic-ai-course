# Implementing Key Components from Scratch

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[09_Interpretability|LLM Interpretability]]
**Next:** [[../02_Prompt_Engineering/01_Prompting_Basics|Prompt Engineering]]

---

In interviews at OpenAI, Anthropic, and DeepMind, you may be asked to implement key components from scratch — attention, tokenizer, embeddings. This tests not only conceptual understanding but also the ability to translate theory into working code.

---

## Self-Attention: Concept and Math

### Scaled Dot-Product Attention Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Each token in the sequence receives three vectors:

**Query (Q)** — "what am I looking for?" Describes what information the given token wants to obtain from others.

**Key (K)** — "what can I offer?" Describes what information the token can provide to others.

**Value (V)** — "what information do I transmit?" The actual content to be transferred.

### Why Divide by √d_k?

Without scaling, at large dimensionality d_k, the dot products Q·K become very large. This causes softmax to "saturate" — nearly all probability concentrates on a single token, and gradients approach zero. Dividing by √d_k normalizes the variance.

### Multi-Head Attention

Instead of a single large attention, we run multiple parallel "heads." Each head operates in its own subspace and can focus on different aspects: syntactic relationships, semantics, positional relations. The results are concatenated and projected back.

Formula: MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)Wᴼ

### Causal Mask for Decoder

Autoregressive models (GPT) must not "peek" at future tokens during generation. The causal mask is a lower-triangular matrix where position i can only see positions ≤ i. It is implemented by masking attention scores with -∞ before softmax.

---

## Positional Encoding

A Transformer without positional encoding cannot distinguish token order — "the cat ate the mouse" and "the mouse ate the cat" look identical.

### Sinusoidal Encoding (Original Transformer)

PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Advantages: fixed, requires no training, can theoretically extrapolate to lengths beyond those seen during training.

### Rotary Position Embedding (RoPE)

A modern approach used in LLaMA and GPT-NeoX. Instead of adding position to embeddings, RoPE "rotates" Q and K vectors in complex space. This encodes relative positions directly into attention scores and works well with KV-cache.

---

## Tokenizer: Byte Pair Encoding

BPE is the foundation of tokenizers in GPT, LLaMA, and Claude.

### Training Algorithm

1. Start with a vocabulary of individual characters
2. Count the frequency of all adjacent token pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until the desired vocabulary size is reached

### Tokenization Algorithm

For new text, apply the learned merges in the order they were learned. This guarantees determinism — the same text is always tokenized identically.

### Details

- The end-of-word marker (</w>) distinguishes "the" in the middle versus end of a word
- Rare words are split into subwords
- Vocab size is a hyperparameter (typically 32K–100K)

---

## Layer Normalization

### LayerNorm vs BatchNorm

BatchNorm normalizes along the batch dimension — it requires batches during inference and is unstable with small batches. LayerNorm normalizes along the feature dimension — it works with batch_size=1 and is more stable for sequence data.

### RMSNorm

A simplified version of LayerNorm, used in LLaMA. It removes mean subtraction, retaining only division by RMS (root mean square). Approximately 10–15% faster with comparable quality.

---

## Feed-Forward Network

The FFN processes each position independently. It serves as the model's "memory" — factual knowledge is stored in the FFN weights.

### Architecture

Standard: FFN(x) = GELU(xW₁)W₂

The intermediate layer size is typically 4× d_model. For a model with d_model=4096, that is 16384 neurons.

### SwiGLU

A modern alternative used in LLaMA and PaLM: SwiGLU(x) = (xW₁ ⊙ σ(xW₂))W₃

Uses three matrices instead of two, but yields better quality.

---

## Transformer Block

The full block combines all components with residual connections.

### Pre-LN vs Post-LN

**Post-LN (original)**: x → Attention → Add → Norm → FFN → Add → Norm

**Pre-LN (GPT-2+)**: x → Norm → Attention → Add → Norm → FFN → Add

Pre-LN is more stable when training deep networks and does not require learning rate warmup.

### Residual Connections

Allow gradients to flow directly through layers; critical for training deep networks. Each layer adds to the input rather than replacing it.

---

## Gradient Checkpointing

When training deep networks, activations of each layer are stored in memory for the backward pass. This requires O(n × d × L) memory, where L is the number of layers. For a 70B model, this amounts to terabytes.

### Checkpointing Concept

Instead of storing all activations, a selective saving strategy is used. Only every k-th layer's activations are saved (checkpoints), and intermediate activations are recomputed on the fly during the backward pass. This is a classic trade-off: less memory in exchange for more computation.

### Trade-off Math

Without checkpointing, O(L) memory is used for activations, and 1× forward + 1× backward pass is performed (equivalent to 2× forward in computation).

With checkpointing every √L layers, memory is reduced to O(√L) for checkpoints plus O(√L) for recomputation. Compute increases: 1× forward + √L × (forward between checkpoints) + 1× backward, which amounts to approximately 2× additional forward (roughly 3× forward total instead of 2×).

### Practical Strategies

| Strategy | Memory | Compute overhead | When to use |
|-----------|--------|------------------|-------------------|
| No checkpoint | O(L) | 0% | Memory is sufficient |
| Every layer | O(1) | +100% (2× compute) | Minimum memory |
| Every √L layers | O(√L) | +33% | Balance |
| Selective | Variable | Variable | Checkpoint only attention |

In PyTorch, this is implemented via torch.utils.checkpoint.checkpoint(module, x), which automatically recomputes activations as needed.

---

## Mixed Precision Training

Using lower-precision formats (FP16/BF16) for speedup and memory savings.

### Number Formats

Several formats exist with different bit distributions among sign, exponent, and mantissa:

**FP32 (float32)** — standard precision: 1 sign bit, 8 exponent bits, 23 mantissa bits. Range ±3.4×10³⁸, precision 7 significant digits.

**FP16 (float16)** — half precision: 1 sign bit, 5 exponent bits, 10 mantissa bits. Limited range ±65504, precision 3–4 digits. Supported by Tensor Cores.

**BF16 (bfloat16)** — brain float: 1 sign bit, 8 exponent bits (same as FP32!), 7 mantissa bits. Range same as FP32 (±3.4×10³⁸), but lower precision at 2–3 digits. Available on A100+ and TPU.

**TF32 (tensor float)** — 1 sign bit, 8 exponent bits, 10 mantissa bits. Automatically used on A100+ to accelerate matmul.

### Why BF16 > FP16 for LLM

FP16 has limited dynamic range (maximum around 65K). During LLM training, gradients can be very small (underflow) while activations can be large (overflow). BF16 solves this: same range as FP32, just less precision. This is ideal for deep learning, where a wide range of values matters more than absolute precision.

### Loss Scaling for FP16

When working with FP16, small gradients can underflow. The solution is to scale the loss before the backward pass by multiplying it by a coefficient (e.g., 1024 or 2048), then dividing the gradients back before the optimizer step. PyTorch provides torch.cuda.amp.GradScaler, which automatically manages this process: it scales the loss, checks gradients for inf/nan, and dynamically adjusts the scaling coefficient.

### Where to Keep Which Precision

The optimal strategy: model weights are stored in FP32 (master copy) and converted to BF16 for computation. Activations are kept in BF16 for the forward and backward pass. Gradients are computed in BF16 but accumulated in FP32 for accuracy. Optimizer states (AdamW momentum and variance) are always in FP32, as they need high precision for stable convergence.

---

## Flash Attention and IO-Awareness

Flash Attention (Dao et al., 2022) is an optimization that changes the attention implementation to minimize memory bandwidth.

### The Problem with Standard Implementation

The naive attention implementation makes three accesses to slow HBM (High Bandwidth Memory): first writing the attention score matrix S = Q @ K.T of size O(n²), then reading S and writing probabilities P = softmax(S), and finally reading P to compute the output O = P @ V. This totals 3 × O(n²) memory transfers, which becomes the bottleneck.

### Flash Attention: Tiling Approach

The key idea is to avoid materializing the entire n×n attention score matrix in slow memory. Instead, we work with blocks (tiles) that fit in fast on-chip GPU SRAM. Q, K, V data is stored in slow HBM, but for computation we load only small blocks Q_block, K_block, V_block into SRAM, compute attention locally, and write the result back. This drastically reduces memory transfers.

### Online Softmax Math

The key mathematical trick is computing softmax incrementally without the full matrix. The algorithm maintains running statistics: m (current maximum) and l (sum of exponents). When processing each new block, m is updated, previous sums are corrected via exp(old_m - new_m), and new contributions are accumulated. This allows processing the matrix in parts while producing exactly the same result as full softmax.

The algorithm splits Q, K, V into blocks (typically 64–128 tokens). For each query block, it iterates over all key/value blocks, computes local attention scores, updates the running max and sum, and accumulates the weighted output. At the end, it normalizes by the accumulated sum.

### Practical Impact

| Metric | Standard | Flash Attention |
|---------|----------|-----------------|
| Memory | O(n²) | O(n) |
| Speed (A100) | baseline | 2-4× faster |
| Training 4K context | OOM | Works |
| Training 16K context | Impossible | Possible |

Flash Attention reduces memory usage from quadratic to linear, providing 2–4× speedup on A100. This makes training on long contexts feasible: 4K works instead of OOM, and 16K becomes a reality.

### Flash Attention 2 and 3

**FA2** — further optimization of thread parallelism and register usage, yielding an additional 2× speedup over FA1.

**FA3** — adds asynchronous operations, FP8 support, and improved work distribution across warps. 1.5–2× speedup over FA2.

---

## Text Generation

### Autoregressive Generation

The model generates one token per step. At each step:
1. Feed all previous tokens
2. Obtain logits for the next token
3. Sample from the distribution
4. Append to the sequence

### Temperature

Divides logits before softmax. T=1 — original distribution. T<1 — more deterministic. T>1 — more random.

### Top-p (Nucleus) Sampling

Select the smallest set of tokens whose cumulative probability is ≥ p, then sample only from them. This adaptively cuts off unlikely tokens.

---

## Interview Tips

### Typical Questions

1. **"Implement attention from scratch"**
   - Start with the formula QK^T/√d_k
   - Explain each step (scaling, softmax, weighted sum)
   - Mention numerical stability (subtracting max before softmax)

2. **"What is the complexity of attention?"**
   - O(n² · d) in both memory and time
   - Where n is the sequence length, d is the dimensionality
   - This is the bottleneck for long contexts

3. **"How does multi-head attention work?"**
   - Parallel attention heads in different subspaces
   - Allows the model to focus on different aspects
   - Concat + linear projection at the end

4. **"Explain positional encoding"**
   - LLMs have no notion of position by default
   - Sinusoidal: fixed, extrapolates
   - RoPE: relative positions, works with KV-cache

### Red Flags in Interviews

- Not understanding why √d_k scaling is needed
- Ignoring numerical stability
- Confusing encoder and decoder attention
- Inability to explain complexity

---

## Key Takeaways

1. **Attention is weighted aggregation** via Q, K, V projections. Scaling by √d_k prevents softmax saturation.

2. **Multi-head** allows the model to examine data from multiple perspectives in parallel.

3. **Causal mask** ensures the autoregressive property — each token sees only preceding ones.

4. **Positional encoding** adds order information. RoPE is the modern standard.

5. **BPE tokenizer** balances between character-level and word-level approaches and is trained on data.

6. **RMSNorm** is simpler and faster than LayerNorm, used in modern models.

7. **Pre-LN architecture** is more stable than Post-LN for deep networks.

8. **Residual connections** are critical — without them, deep networks fail to train.

---

## Concepts in Action

**Scaled Dot-Product Attention** is implemented by computing scores as the matrix product of Q and transposed K, divided by the square root of the dimension d_k. Before softmax, a numerically stable trick is applied — subtracting the maximum to prevent overflow. If a mask is provided (for causal attention), scores are masked with -1e9 at the appropriate positions. Softmax is then applied along the last axis to obtain attention weights, which are multiplied by V to produce the output values.

Multi-head attention runs multiple such attention mechanisms in parallel with different projection matrices W_q, W_k, W_v for each head, then concatenates the results and applies a final projection through W_o.

**Text Generation with Top-p Sampling** works autoregressively: at each generation step, all previous tokens are fed into the model to obtain logits for the next token, which are divided by temperature to control randomness. Probabilities are computed via softmax, then sorted in descending order to find the cutoff — the smallest set of tokens with cumulative probability ≥ p. The next token is sampled from this nucleus with renormalized probabilities. The token is appended to the sequence, and the process repeats until max_tokens is reached or an EOS token is generated.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[09_Interpretability|LLM Interpretability]]
**Next:** [[../02_Prompt_Engineering/01_Prompting_Basics|Prompt Engineering]]
