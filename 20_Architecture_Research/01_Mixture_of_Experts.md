# Mixture of Experts: Scaling Through Sparsity

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Architecture Research
**Previous:** [[../19_Practical_Projects/05_Production_Agent|Production Agent]]
**Next:** [[02_State_Space_Models|State Space Models]]

---

## The Problem of Scaling Dense Models

Scaling laws demonstrated a direct relationship: more parameters equals better quality. However, dense models, where every parameter participates in every computation, hit a fundamental limitation — compute grows linearly with the number of parameters.

GPT-3 with 175B parameters requires approximately 350 PFLOP per forward pass. GPT-4 with an estimated 1.8T parameters has correspondingly astronomical inference and training costs.

Mixture of Experts offers a revolutionary solution: increase the number of parameters without a proportional increase in compute. Instead of using all parameters for every token, MoE activates only a small subset — specialized "experts."

Mixtral 8x7B demonstrates the power of this approach: the model has 47B parameters but uses only approximately 13B per token — equivalent to a 7B dense model in compute. Meanwhile, quality is comparable to LLaMA 2 70B. A 5x compute savings while maintaining quality.

## MoE Architecture

### Core Concept

MoE replaces a single large FFN block in a transformer with N specialized FFN blocks (experts) plus a router that decides which experts to use for each token.

A standard Transformer uses the sequence: Input → Attention → FFN → Output. An MoE Transformer transforms this into: Input → Attention → Router → selected Experts → Combine → Output.

**Router** (gating network) — a small neural network, typically a single linear layer, that takes an input token and produces a probability distribution over experts. The gating function G(x) computes the softmax of the product of learnable weights and the input vector, creating a probability distribution over all experts.

**Top-K routing:** Instead of a weighted combination of all experts, only the K experts with the highest probabilities are used. The final output is a weighted sum of the selected experts' outputs, where weights are determined by routing probabilities.

### Switch Transformer: Minimalism Works

In 2021, Google introduced the Switch Transformer with a radical simplification: K=1, meaning each token is routed to only one expert.

Why this works: maximum sparsity at K=1 means compute does not grow with the number of experts, routing simplicity eliminates the need to combine outputs, and experts are forced to become narrow specialists.

Switch Transformer achieved a 7x speedup compared to T5-Base at equivalent quality, scaling to 1.6T parameters. Hard routing creates a discrete choice — addressed through Gumbel-Softmax or a straight-through estimator for gradient flow.

### GShard: Load Balancing

Google's GShard (2020) introduced a critical optimization — auxiliary load balancing loss. The balancing loss formula: the number of experts N is multiplied by the sum of products of two quantities for each expert — the fraction of tokens routed to the expert and the average routing probability for that expert, all multiplied by coefficient alpha (typically 0.01).

Without balancing, the router tends to "collapse" — routing all tokens to a few experts while ignoring the rest. This wastes MoE advantages (unused experts), creates a bottleneck (overloaded experts), and degrades quality (less specialization).

**MoE Routing Health Metrics:**

Expert Utilization Rate shows the fraction of tokens routed to each expert. The ideal value is approximately 1/N for all experts, where N is the number of experts.

Routing Entropy measures distribution uniformity as the negative sum of products of probabilities and their logarithms. Maximum entropy log(N) is achieved with uniform distribution. A ratio H/log(N) close to 1 indicates good balancing.

Expert Collapse Detection identifies collapse when a single expert receives more than 50% of tokens or when active experts number fewer than half the total.

Router Confidence — the average value of the maximum probability after softmax. High confidence means clear selection, low confidence may indicate underfitting.

**Token Dropping Under Overload:**

With Top-K routing and a capacity factor, the maximum tokens per expert is defined: capacity equals c multiplied by batch_size multiplied by seq_len divided by N, where c is the capacity factor. If the number of tokens exceeds capacity, excess tokens are dropped. At c = 1.0 drops happen frequently, the standard c = 1.25 provides a 25% buffer, and at c = 2.0 drops are rare but require twice the memory.

### Mixtral: The Modern Standard

Mixtral 8x7B (December 2023) from Mistral AI became the benchmark for open-source MoE:

Architecture: 8 experts per MoE layer, Top-2 routing (each token is routed to 2 experts), 32 transformer layers each with an MoE FFN, 47B total params but only approximately 13B active params.

Technical decisions: Top-2 instead of Top-1 provides more capacity and less risk of information loss, Expert parallelism places each expert on a separate GPU or group of GPUs, Sliding window attention with a window of 4096 for efficiency.

Results: Mixtral 8x7B outperforms LLaMA 2 70B on most benchmarks with 5x less inference compute.

### DeepSeek V3: State-of-the-Art (December 2024)

DeepSeek V3 is the defining release of the year, redefining the economics of training large models. The model achieves GPT-4o quality at a radically lower training cost.

Specifications: 671B total parameters, 37B activated per token, training cost of only $5.5M (95% cheaper than competitors), 14.8T training tokens, 128K context length, MIT license (fully open).

**Architectural Innovations:**

**Multi-Head Latent Attention (MLA):** Standard Multi-Head Attention requires storing K and V for each head — KV cache for 128K context x 70B model amounts to hundreds of GB of memory. MLA compresses KV representation through a learned projection into a latent space of much smaller dimensionality. Result: KV cache is reduced by 5-10x with minimal quality loss, enabling long context without proportional memory growth, making inference on 128K context practical.

**Multi-Token Prediction (MTP):** Instead of predicting a single next token, the model predicts several simultaneously — probability distributions for tokens at positions t+1, t+2, and beyond given the context. During training, additional prediction heads are trained in parallel, predicting future tokens at different horizons. During inference, a speculative-like approach is used where the model predicts multiple tokens, then verifies them and accepts or rejects. Result: up to 60 tokens/sec while maintaining quality — a 1.8x speedup compared to the standard autoregressive approach.

**FP8 Mixed Precision Training:** DeepSeek V3 became the first model of this scale trained with FP8 precision across all parts of the pipeline. The forward pass uses FP8 for weights and activations, the backward pass applies FP8 for gradients, and master weights are stored in FP32 for stability. Challenges addressed through adaptive loss scaling for the limited FP8 range, gradient overflow handling, and quantization noise compensation. Result: approximately 2x compute efficiency compared to FP16.

**Auxiliary-loss-free Load Balancing:** The traditional approach uses an auxiliary loss added to the main loss. The problem is that the balance loss can conflict with the main loss, degrading quality. DeepSeek V3 uses an expert bias term instead of auxiliary loss: each expert has a learnable bias adjusted through running load statistics. Overloaded experts receive negative bias, underloaded experts receive positive bias. Advantage: balance is achieved without affecting the main optimization objective.

**Comparison with Competitors:**

GPT-4o: approximately 1.8T total/unknown active params, $100M+ training cost, 128K context, 88.7 MMLU.
Claude 3.5: unknown params, unknown cost, 200K context, 88.7 MMLU.
LLaMA 3.1 405B: 405B total/405B active, approximately $50M cost, 128K context, 88.6 MMLU.
DeepSeek V3: 671B total/37B active, $5.5M cost, 128K context, 88.5 MMLU.
Mixtral 8x22B: 141B total/39B active, unknown cost, 64K context, 77.8 MMLU.

**Practical Implications:**

Inference efficiency: 37B active params is comparable to LLaMA-70B in compute, but quality is at the 405B level.

Memory requirements: Full precision approximately 1.3TB, INT8 approximately 670GB, INT4 approximately 340GB. Requires multi-GPU setup, but MLA reduces KV cache overhead.

Democratization: MIT license plus low training cost means companies can fine-tune on their own data, the research community gets a SOTA baseline, and competition with closed-source models becomes realistic.

Training at scale: $5.5M training cost proves that the right architecture matters more than brute-force compute, FP8 training is practical for production, and MoE plus architectural innovations deliver massive efficiency gains.

## Router Architectures

### Linear Router

The simplest router is a single linear layer applying TopK to the Softmax of a linear projection of the input. Advantages: minimal overhead, simplicity. Disadvantages: limited expressiveness.

### Expert Choice Routing

Instead of "the token chooses the expert," Expert Choice inverts this: "the expert chooses tokens." Algorithm: The router computes scores for all token-expert pairs, each expert selects the top-K tokens with the highest scores, guaranteeing uniform load.

Formalization: a score matrix is computed as the product of tokens and transposed router weights; expert i processes the tokens with the highest scores in column i.

Advantages: perfect load balancing by design, no need for auxiliary loss, each expert processes exactly a fixed number of tokens.

Disadvantages: a token may not reach any expert (addressed via residual connection), requires global coordination (harder to parallelize).

### Soft MoE

Soft MoE (2023) from Google proposes a fully differentiable approach: the output is the sum over all experts of the product of Softmax slot scores and the expert output from a weighted combination of input tokens.

Idea: Instead of hard routing, each expert receives a weighted combination of input tokens. This allows gradients to flow to all experts.

Trade-off: Compute efficiency is lost (all experts are active), but training stability improves.

### Learned vs Fixed Routing

Hash routing: Deterministic routing based on a hash function of the token. No learnable parameters, perfect balance, but no adaptivity.

Random routing: Random expert selection. Surprisingly performs acceptably — experts specialize on "random slices" of data.

Cluster routing: Tokens are grouped (k-means on embeddings), each cluster is routed to its own expert. Precomputed, not trained end-to-end.

## Capacity Factor and Overflow Handling

With a fixed batch size and Top-K routing, the number of tokens for each expert varies. One expert may receive 100 tokens, another — 10.

The capacity factor defines the maximum tokens per expert: tokens in batch multiplied by K divided by the number of experts multiplied by C, where C is the capacity factor (typically 1.0-2.0). At C = 1.0, there is perfect balance but overflow is inevitable. At C = 2.0, there is double buffer with less overflow but twice the memory.

**Handling Overflow:**

Drop tokens: Discard "excess" tokens. Simple but loses information.

Residual connection: Add the original input directly to the output. Dropped tokens pass through the residual.

Auxiliary experts: A "catch-all" expert for overflow tokens.

Dynamic batching: Redistribute tokens across micro-batches for balance.

Capacity constraints create non-determinism: the same input can produce different output depending on batch composition. This complicates reproducibility, debugging, and evaluation (batch composition must be controlled).

## Expert Parallelism

### Tensor vs Expert Parallelism

Tensor parallelism: A single expert is split across GPUs. AllReduce after each expert.

Expert parallelism: Each expert resides entirely on its own GPU. AllToAll for routing.

For MoE with a large number of experts, EP is generally more efficient: less communication (AllToAll vs numerous AllReduce operations), linear scaling with the number of experts, and simpler implementation.

### AllToAll Communication

AllToAll is a collective operation where each process sends data to every other process. Before the operation, each GPU holds tokens routed to all experts (routed locally). After the operation, each GPU receives all tokens for its assigned expert from all other GPUs.

Communication cost: AllToAll time is defined as (N-1) multiplied by data volume divided by bandwidth, where N is the number of GPUs.

### Hierarchical MoE

For very large numbers of experts (over 64), a hierarchical structure with two levels is used: the Group Router selects a group of experts, the Expert Router selects a specific expert within the group.

Advantages: fewer parameters in the router (logarithmic growth instead of linear), the possibility of locality-aware routing where groups are placed on the same node, and hierarchical specialization where a group specializes in a broad area (e.g., "math") while experts within it specialize in narrow sub-areas (e.g., "algebra").

## Sparse vs Dense: Trade-offs

### When MoE Wins

Large inference budget: If you can afford a large model but compute is limited, MoE provides more parameters for the same FLOP.

Diverse tasks: MoE naturally specializes experts on different tasks and domains.

Scaling: MoE scales to trillions of parameters more efficiently than dense.

### When Dense Is Better

Small models: Routing overhead and potential balance issues are not justified for models under 10B.

Homogeneous data: If all data is similar, expert specialization does not help.

Strict latency: Additional AllToAll communication increases latency.

Memory-constrained inference: MoE requires loading all experts, even if only a fraction is active.

### Qualitative Differences

Dense models: Smoother representations, better transfer between tasks, easier to interpret.

MoE models: Modular representations, potentially better on out-of-distribution data, harder to debug (which expert is responsible?).

## Analysis of Expert Specialization

Research reveals interesting specialization patterns:

Syntactic specialization: Some experts specialize in nouns, verbs, punctuation, and numbers.

Semantic specialization: Scientific text, code, casual conversation, and formal language.

Positional specialization: Sentence beginnings, sentence endings, and the middle of long contexts.

### Analysis Methods

Expert activation patterns: For each token, activated experts are recorded, then the data is grouped by token types to identify patterns — which token types are routed to which experts.

Expert probing: Linear classifiers are trained on hidden states within each expert to verify what information they store, for example whether an expert can determine part of speech or sentiment.

Ablation studies: An expert is completely disabled, then the model's quality drop is measured on various tasks to understand what that expert is responsible for.

### Practical Observations

Not all experts are equally useful: Often 20% of experts process 80% of tokens even with balance loss.

Specialization emerges: Experts specialize even without explicit supervision.

Redundancy: Several experts may learn similar functions (potential for pruning).

## Connections to Other Course Topics

Distributed Training (section 16): Expert parallelism is a specific type of model parallelism. AllToAll communication requires optimization just like AllReduce.

Scaling Laws (section 01): MoE changes scaling laws — parameters vs compute become decoupled. "Chinchilla optimal" for MoE is different.

Inference (section 17): MoE inference requires loading all experts (memory), AllToAll for distributed inference, and specific KV cache management.

Quantization (section 15): Unused experts can be quantized more aggressively, active ones more carefully.

## Key Takeaways

MoE decouples parameters and compute. More parameters does not mean proportionally more computation — only active experts consume compute.

The router is a critical component. Routing quality determines MoE efficiency. Poor routing equals wasted parameters.

Load balancing is essential. Without auxiliary losses, experts collapse. Balance loss is a mandatory part of training.

Capacity factor is a trade-off. More capacity equals less overflow but more memory. Typical choice: 1.25-2.0.

Expert parallelism scales efficiently. Each expert on its own GPU, AllToAll for routing. Linear scaling with the number of experts.

Specialization is emergent. Experts automatically specialize on different aspects of language and tasks without explicit supervision.

Top-K selects precision vs efficiency. K=1 is maximally efficient, K=2 provides a safety margin, K greater than 2 is rarely justified.

MoE is not for every scenario. Small models, homogeneous data, strict latency — dense may be better.

Memory footprint remains high. All experts must reside in memory, even if only a small fraction is active. This limits inference on edge devices.

This is an active research area. Soft MoE, Expert Choice, hierarchical routing — new approaches emerge regularly.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Architecture Research
**Previous:** [[../19_Practical_Projects/05_Production_Agent|Production Agent]]
**Next:** [[02_State_Space_Models|State Space Models]]
