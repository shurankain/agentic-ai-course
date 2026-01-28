# State Space Models: Beyond Quadratic Complexity

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Architecture Research
**Previous:** [[01_Mixture_of_Experts|Mixture of Experts]]
**Next:** [[03_Multimodal_Architectures|Multimodal Architectures]]

---

## The Fundamental Problem of the Transformer

Attention is a brilliant mechanism, but with a costly secret: each token must "look" at all previous tokens. This creates quadratic complexity O(n²) with respect to sequence length.

For short contexts this is not a problem. But for long documents, video, and genomic data, quadratic scaling becomes a killer. Doubling the context increases compute by 4x. A context of 1M tokens requires an attention matrix of 10¹² elements — terabytes of memory.

State Space Models offer a radically different approach: linear complexity O(n) through recurrent hidden state updates. Mamba, a modern SSM, achieves quality comparable to Transformer on many tasks at linear cost.

But this is not a free lunch. The SSM trade-off: efficiency at the cost of certain capabilities. Understanding when to use SSM versus Transformer is critical knowledge for architectural design.

## State Space Models Intuition

SSMs originated in control theory of the 1960s, long before deep learning. The idea is simple: a system is described by a hidden state h(t) that evolves over time under the influence of input x(t).

A continuous-time SSM is described by two equations: the time derivative of the hidden state equals Ah(t) plus Bx(t), the output y(t) equals Ch(t) plus Dx(t). Here h(t) is the hidden state of dimension N, x(t) is the input signal, y(t) is the output, and A, B, C, D are system parameters.

To work with sequences we discretize over time: h_k equals A_bar times h_{k-1} plus B_bar times x_k, y_k equals C times h_k plus D times x_k, where A_bar, B_bar are discretized versions (methods: Zero-Order Hold, Bilinear transform).

### Why is this faster than attention?

Recurrent computation: To obtain y_n you only need h_{n-1} and x_n — O(N) operations regardless of history length.

Convolutional computation: An SSM can be equivalently represented as a convolution where the output is the convolution of the input with a kernel K defined by the sequence CB, CAB, CA²B, and so on. The convolution is computed via FFT in O(n log n) — subquadratic.

Dual nature: An SSM can be used recurrently for inference (linear time, constant memory) or convolutionally for training (parallel across sequence). This is the "best of both worlds": parallel training like Transformer, efficient inference like RNN.

### What is lost compared to attention?

In-context retrieval: Attention can directly "copy" information from any position in the context. An SSM must "memorize" it in the compressed state h.

Random access: Attention accesses any position in O(1). An SSM requires traversing the entire sequence.

Position-specific patterns: Attention easily learns "if X is at position 3, then...". An SSM processes everything through uniform dynamics.

## SSM Evolution: From S4 to Mamba

### S4: Structured State Spaces

S4 (2022) from Stanford was the first SSM competing with Transformer on long sequences.

The HiPPO problem: How to initialize the matrix A so the system "remembers" history? Random initialization does not work — gradients vanish or explode.

HiPPO (High-order Polynomial Projection Operator): A special structure for matrix A based on Legendre polynomials. Elements are defined by a formula depending on indices n and k: a negative product of square roots for n greater than k, n+1 for the diagonal, zero for n less than k. This special structure forces h(t) to approximate the Legendre polynomial coefficients of the history — optimal compression.

Diagonalization: S4 works with diagonalized A equal to V times Lambda times V inverse. The recurrent update simplifies: h_k equals (Lambda plus Delta B) times h_{k-1} plus Delta B times x_k. The diagonal structure enables efficient computation of the convolutional kernel K via FFT.

S4 results: State-of-the-art on Long Range Arena (sequences up to 16K), comparable to Transformer on language tasks at linear complexity.

### Mamba: Selective State Spaces

Mamba (2023) was a breakthrough in SSM, the first to compete with Transformer on language modeling.

The key innovation is the Selection Mechanism: Unlike standard SSMs where parameters A, B, C are fixed for all tokens, Mamba makes parameters B, C, and Delta (discretization step size) input-dependent.

For each input token the model dynamically computes these parameters via linear projections: Delta(x) equals softplus of a linear projection of the input (determines the discretization step size), B(x) is a linear projection of the input (controls projection of the input into state space), C(x) is a linear projection of the input (controls the output projection).

This allows the model to selectively decide what to remember: a large Delta value forces the system to forget the previous state more aggressively and pay more attention to the current token, while a small Delta preserves history and reduces the influence of the current input.

Mamba Block architecture: The input token passes through two parallel processing branches. The first branch expands dimensionality via a linear projection, then applies a 1D convolution for local context, after which data is fed into the selective SSM module. The second branch also expands dimensionality via a separate linear projection and serves as a gating mechanism. Outputs from both branches are multiplied element-wise, allowing the model to selectively pass information. A final linear projection returns the dimensionality to the original size, and the result is added to the input via a residual connection.

Complexity comparison: Transformer has O(n²) for training and inference, O(n × d) memory for KV cache. Mamba has O(n) for training, O(1) per token for inference, O(d_state) memory where d_state is approximately 16.

Why this matters: Consider the task "The capital of France is [answer]". The model must "notice" that France is being discussed, "recall" that the capital is Paris, and "ignore" irrelevant information in between. Fixed parameters cannot adapt what to memorize. Input-dependent parameters allow selectively increasing Delta for important tokens (stronger influence on state) and decreasing Delta for background tokens (almost no state change).

The Selective Scan algorithm: At each step it computes input-dependent parameters via linear projections and softplus, then discretizes the continuous parameters (A_bar as the exponential of the product of delta and diagonal A, B_bar as the product of delta and B), and then performs a recurrent hidden state update (h is multiplied by A_bar plus B_bar times the current input). The final output is the dot product of C and the updated state h.

Performance optimization: A naive implementation of selective scan does not parallelize due to sequential step dependencies. Mamba addresses this through kernel fusion (combining all operations into a single CUDA kernel to minimize memory access overhead), recomputation (recomputing intermediate activations in the backward pass instead of storing them to save memory), and hardware-aware memory layout (placing data in an optimal order for efficient GPU access).

Result: Mamba-3B is comparable to Transformer-7B on many tasks at 5x lower inference compute.

### H3: Hungry Hungry Hippos

H3 (2023) simplified S4 and added two mechanisms to improve language modeling.

Shift SSM: Instead of complex HiPPO, a simple shift operator where h_k equals shift of h_{k-1} plus x_k. This creates a "sliding window" memory.

Input-dependent gating: The output is modulated by the input — y_k equals h_k times gate of x_k.

H3 limitation: Parameters are fixed and do not depend on the input. This limits expressiveness.

## Linear Attention: An Alternative Path

Quadratic complexity of attention is due to softmax: the standard formula Attention(Q, K, V) equals softmax of QK transposed divided by the square root of d times V. You cannot compute softmax of QK transposed times V without the full QK transposed. But you can remove softmax.

Linear Attention: Replace the exponential of the product of q and k with the product of phi(q) transposed and phi(k) for some feature map phi. Linear Attention(Q, K, V) equals phi(Q) times phi(K) transposed times V.

Computational trick: Normally the product QK transposed times V has complexity O(n²d). The linear product Q times K transposed times V has complexity O(nd²) if you first compute S equals K transposed times V. When d is much less than n (typically d=64, n=4096), linear attention is significantly faster.

Feature Maps: ReLU where phi(x) equals ReLU(x) (problem: loss of negative values), ELU + 1 where phi(x) equals ELU(x) plus 1 (guarantees positivity), Random Fourier Features where phi(x) equals exponential of x transposed omega divided by square root of m (approximates the softmax kernel), Polynomial where phi(x) equals (1, x, x², ...) Taylor expansion of softmax.

Problems with Linear Attention: No normalization (Softmax normalizes weights to sum to 1, Linear attention can produce very large or small values), worse quality (on language modeling linear attention lags behind softmax by several perplexity points), limited rank (phi(Q) times phi(K) transposed times V has rank less than or equal to the dimension of phi, while softmax attention is full-rank).

### RetNet: Retention Networks

RetNet (2023) from Microsoft combines ideas. Retention(Q, K, V) equals the product of Q times K transposed element-wise multiplied by D times V, where D is a causal decay matrix with elements D_{nm} equal to gamma raised to the power of n minus m times the indicator that n is greater than or equal to m.

Three modes: Parallel for training like attention, Recurrent for inference like RNN, Chunkwise hybrid for long sequences.

RetNet achieves quality close to Transformer at linear inference complexity.

## Hybrid Architectures

Research shows that SSMs underperform Transformer on tasks requiring precise retrieval from context, complex reasoning with many variables, and in-context learning of new patterns. Hybrid architectures combine SSM and attention, getting the best of both.

### Jamba: SSM + Attention

Jamba (2024) from AI21 is the first production-ready hybrid model.

Architectural features: Jamba uses a 4:1 ratio between Mamba and Attention layers — for every four Mamba layers there is one Attention layer. The model additionally integrates Expert mixture (MoE) for capacity scaling: the total parameter count is 52B, but only 12B are active at any given time. The repeating block structure: four sequential Mamba layers, then one Attention layer, and the block concludes with an MoE module.

Component synergy: Mamba layers efficiently process "background" information and long sequences with linear complexity, Attention layers focus on critical data points (entities, relationships) where precise retrieval is required, MoE modules add model capacity without proportional increase in computational cost since only a subset of experts is activated.

Results: 256K context with quality comparable to Mixtral at lower compute.

### StripedHyena

A hybrid variation from Together AI: Alternating Hyena (SSM-like) and attention layers, Smaller attention windows (local context), SSM for global context.

### Practical Considerations

When to add attention: Early layers (low-level patterns), every N layers (regularly), late layers (high-level reasoning).

Attention window size: Full attention (maximum quality, maximum cost), Sliding window (trade-off), Sparse patterns (task-specific).

## When SSM vs Transformer

### SSM Excels

Very long context (over 100K): Genomics (millions of base pairs), audio/video (hours of material), documents (entire books).

Streaming inference: Constant memory independent of history, real-time processing.

Compute-constrained deployment: Edge devices, high-throughput serving.

### Transformer Excels

In-context learning: Few-shot tasks, novel pattern recognition.

Precise retrieval: Question answering with explicit copying, code completion with exact references.

Complex reasoning: Math word problems, multi-hop reasoning.

### Empirical guidelines

Long document summarization — SSM/Hybrid. Code generation — Transformer. Conversational AI — Hybrid. DNA sequence modeling — SSM. Mathematical reasoning — Transformer. Audio transcription — SSM. In-context learning — Transformer. Real-time inference — SSM.

## Current Limitations of SSM

Limitation 1 Fixed state size: The state h has a fixed size. The entire history is compressed into N numbers. For a sufficiently long history, any compression loses information. The Attention alternative: KV cache grows with context but preserves everything.

Limitation 2 Training stability: SSMs are more sensitive to learning rate (require smaller LR), initialization (HiPPO or special initialization is mandatory), and precision (BF16 is sometimes problematic).

Limitation 3 Hardware efficiency: Despite theoretical linearity, in practice selective scan poorly utilizes tensor cores, is memory bandwidth limited (not compute), and requires custom CUDA kernels.

Limitation 4 Ecosystem: Transformer has optimized libraries (FlashAttention and others), proven training recipes, and extensive community experience. SSM still lags behind in tooling and established practices.

## Connection to Other Course Topics

Flash Attention (Section 15): IO-aware memory access optimization. Mamba uses analogous principles for selective scan.

Long Context Inference (Section 17): SSM is an alternative to RoPE scaling and sliding window for long context. Trade-off: SSMs are linear but lose precise retrieval.

Distributed Training (Section 16): SSMs are easier to shard — no attention matrices between GPUs. Sequence parallelism is simpler.

Scaling Laws (Section 01): SSMs have different scaling properties. Compute scales linearly with context — enabling exploration of regimes inaccessible to Transformer.

## Key Takeaways

Quadratic complexity of attention is a fundamental limitation. For long sequences, alternatives are needed.

SSMs offer linear complexity through recurrent state updates. Training is parallel (convolution), inference is sequential (RNN).

Mamba made SSMs competitive thanks to input-dependent parameters (selectivity). The model "chooses" what to remember.

Linear attention is another path. It removes softmax, replacing it with feature maps. Simpler, but often lower quality.

Hybrids are a practical trade-off. Jamba, StripedHyena combine SSM and attention, gaining SSM efficiency and attention capability.

SSM is not a universal replacement. In-context learning, precise retrieval, complex reasoning — Transformer still performs better.

Hardware efficiency still lags. SSMs are theoretically more efficient, but optimizations are not as mature as for Transformer.

This is an active area. Mamba2, Linear Attention variants, new hybrids — progress is rapid.

The choice depends on the task. Long context streaming — SSM. Few-shot reasoning — Transformer. General purpose — Hybrid.

State size is the fundamental trade-off. Fixed memory vs growing KV cache. Compression vs exact storage.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Architecture Research
**Previous:** [[01_Mixture_of_Experts|Mixture of Experts]]
**Next:** [[03_Multimodal_Architectures|Multimodal Architectures]]
