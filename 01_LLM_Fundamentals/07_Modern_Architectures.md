# Modern LLM Architectures

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[06_Integration_Patterns|LLM Integration Patterns]]
**Next:** [[08_Scaling_Laws|Scaling Laws]]

---

## Introduction: Beyond the Classic Transformer

The Transformer architecture, introduced in 2017, revolutionized natural language processing and became the foundation for all large language models. However, over seven years of its dominance, researchers have identified a number of fundamental limitations that become increasingly critical as model scale grows.

### Fundamental Limitations of the Classic Approach

**Quadratic complexity of the attention mechanism** is not merely a technical detail. When you process a sequence of N tokens, attention computes an N×N matrix of relationships. For 1K tokens, that is one million elements. For 100K tokens, ten billion elements. The growth in computational requirements is quadratic: doubling the context length increases computation fourfold.

This limitation defines the fundamental economics of modern LLM systems. Why did most models before 2024 have a context window of 4K-8K tokens? Not because training on long contexts was impossible, but because of the prohibitively high cost of inference. Each request to a model with a 100K token context costs roughly 100 times more than one with a 4K token context.

**The problem of dense computation** is less obvious but no less critical. In a classic Transformer, every model parameter is activated for every token. Consider a model with 70B parameters: generating a single token requires 70 billion multiplications and additions. But think about the nature of queries — a question about Python programming activates the same parameters as a discussion of quantum physics.

This is fundamentally inefficient. The human brain does not work this way: when you solve a math problem, the neurons responsible for face recognition are not activated. Evolution optimized the brain for sparse activation — only relevant regions are activated. Dense Transformers go against this principle.

**Economic consequences in production systems** turn these limitations into a business problem. A company processing one billion requests per day to a 70B model spends on the order of $100K-500K on inference daily. If architectural changes can reduce active parameters from 70B to 15B without quality loss, that represents savings of hundreds of thousands of dollars daily.

### Evolution of Architectural Thinking

In response to these limitations, the research community pursued two fundamentally different paths:

**The first path — sparse computation (MoE):** If the entire model is not needed for every request, create multiple specialized "experts" and dynamically select the relevant ones. This idea is not new — the first MoE models appeared in 1991. But only in 2023-2024, with Mixtral and GPT-4, did MoE prove its viability. By 2025, MoE became the dominant architecture for frontier models: DeepSeek V3/R1 (671B total, 37B active), Llama 4 Scout/Maverick, Qwen 3 235B, and GPT-5 all use MoE.

**The second path — alternative sequence processing (SSM):** If attention with quadratic complexity is the bottleneck, replace it with a mechanism that has linear complexity. State Space Models borrow ideas from control theory: instead of storing the entire history, compress it into a fixed-size state that updates with each new token. Mamba (December 2023) demonstrated that this approach can compete with Transformers.

**Hybrid architectures** represent a third path — an acknowledgment that different parts of the model have different requirements. Perhaps early layers should use SSM for efficient long-sequence processing, while later layers use attention for precise information retrieval? Jamba (March 2024) was the first to demonstrate a viable hybrid design for production use.

### Why This Is Critical for Senior Engineers

Architecture selection is not an academic question. It determines:
- **Cost** — direct inference expenses
- **System capabilities** — what the system can and cannot do
- **Latency profile** — system responsiveness
- **Infrastructure complexity** — difficulty of deployment and scaling

A senior engineer or AI lead must understand not only "how attention works" but also "why MoE could reduce costs by 60% for our use case, but would require reworking the infrastructure." This knowledge distinguishes "we use the GPT-4 API" from "we built a cost-effective system with the right architectural trade-offs."

---

## Mixture of Experts: Sparse Computation

### MoE Philosophy: Not All Knowledge Is Needed All the Time

Imagine a university where a student visits every department for every question — from mathematics to linguistics. Absurdly inefficient. It is far more sensible to direct a math question to the mathematics department and literary analysis to the philologists.

Mixture of Experts implements exactly this idea. Instead of a single monolithic feed-forward network in each Transformer layer, a set of specialized "experts" is used — separate neural networks. A routing network (router) decides which experts should process each token.

**Key insight:** when properly implemented, MoE allows increasing the number of model parameters (and, correspondingly, its capacity for storing knowledge) without a proportional increase in computational cost for inference. This breaks the traditional link of "more parameters = more compute."

### Historical Evolution: From Idea to Production

MoE is not a new idea. The concept appeared in 1991, long before the deep learning revolution. But only in recent years have three factors converged to make MoE practical:

**Hardware advances:** Modern GPUs with hundreds of gigabytes of memory can host all experts simultaneously. In 2015, a model with 8 experts of 7B parameters each was impossible on available hardware.

**Training techniques:** Training MoE models requires sophisticated load balancing between experts. Without auxiliary loss functions and capacity constraints, the model "collapses" — all tokens are routed to one or two experts, negating the advantages of the architecture. These techniques were developed only in 2020-2023.

**Inference infrastructure:** Efficient MoE deployment requires expert parallelism — placing different experts on different GPUs. This differs from standard tensor parallelism and requires all-to-all communication patterns. Frameworks like DeepSpeed and Megatron-LM learned to do this efficiently only recently.

**Production validation:** Before Mixtral (December 2023), there was no widely available open-source MoE model demonstrating production-ready quality. Mixtral opened the door, and by 2025, MoE is the default for frontier models. DeepSeek V3 (December 2024) proved that MoE + architectural innovations can match GPT-4o quality at $5.5M training cost. Llama 4 (April 2025) demonstrated MoE with native multimodal early fusion and 10M token context (Scout variant). Every major frontier model now uses some form of sparse computation.

### MoE Architecture: How It Works

A classic MoE layer replaces the feed-forward network in the Transformer:

**Data flow:** Hidden states → Router Gate → Top-k Expert Selection → Experts (E1...E8) → Weighted Sum → Output

The router computes scores for each expert, then selects the top-k (typically k=1 or k=2) with the highest values. Only the outputs of the activated experts are computed and combined via weighted summation.

**Key point:** The remaining (8-k) experts are completely skipped — their parameters do not participate in computation. This is what enables sparse computation.

### Mixtral 8x7B: MoE in Production

Mixtral, released by Mistral AI in December 2023, became the first widely available MoE model to demonstrate the practical applicability of the architecture.

**Architectural characteristics:**

Mixtral contains eight experts with 7 billion parameters each in FFN layers (56B total), plus approximately 11B shared parameters in the attention and embedding layers. The total parameter count reaches approximately 47 billion. Thanks to the Top-2 routing mechanism, only 2 of 8 experts are activated for each token, meaning approximately 13B parameters (~28% of the total) are used per token.

**Results:** Mixtral achieves quality comparable to Llama 2 70B while using only ~25% of the computation at inference. This is not marketing exaggeration — the results are confirmed on standard benchmarks.

**Critical trade-off — memory vs. computation:** Although inference is faster, all 47B parameters must be stored in GPU memory. This means deploying Mixtral requires more GPU memory than a 13B parameter model, despite comparable inference speed. In practice: ~94GB in bf16 or ~24GB with 4-bit quantization.

### DeepSeekMoE: Fine-Grained Experts

DeepSeek went further with a more granular approach: 64 small experts instead of 8 large ones + several shared experts that are always active for base capabilities. DeepSeek-V2: 236B parameters, only 21B active — aggressive sparsity delivers GPT-4-Turbo quality at dramatically lower cost.

### The Load Balancing Problem

The key challenge in MoE is ensuring uniform expert utilization. Without special measures, a "winner-take-all" effect emerges: the router begins directing all tokens to a few "popular" experts, which:
- Negates the advantages of sparsity
- Leads to underutilization of the remaining experts
- Reduces overall model quality

**Solutions:**

1. **Auxiliary loss** — an additional loss component that penalizes uneven load distribution among experts. The average usage probability of each expert is computed, then the deviation from an ideal uniform distribution (1/num_experts) is measured. The loss increases with imbalance, forcing the model during training to distribute tokens more evenly.

2. **Capacity constraints** — hard limits on the number of tokens a single expert can process within one batch. If an expert reaches its capacity limit, additional tokens are rerouted to other experts, even if the router originally selected the overloaded expert.

3. **Random routing** — adding stochasticity to routing decisions. Instead of always selecting the top-k experts deterministically, sampling based on router probabilities is used. This prevents complete dominance of some experts over others.

### Expert Parallelism: Deployment Challenge

MoE deployment differs from dense models: instead of tensor parallelism (each GPU holds a portion of each layer), expert parallelism is used (each GPU holds entire experts). This requires an all-to-all communication pattern — each GPU potentially sends data to every other GPU.

**Practical implications:** High-speed interconnect (NVLink/InfiniBand) is required, otherwise communication overhead will consume the benefits of sparse compute. Latency grows with the number of GPUs. In practice: MoE is efficient on single-server multi-GPU setups, but challenging on distributed clusters.

---

## State Space Models: Linear Complexity

### The Problem of Quadratic Attention Complexity

The self-attention mechanism computes relationships between all pairs of tokens: it multiplies the Query matrix by the transposed Key matrix, normalizes through softmax, and applies the result to the Value matrix. The resulting attention scores matrix has size (seq_len × seq_len). For a 100K token sequence, that is 10 billion elements — for just one attention head of one layer.

This is not merely a memory problem — even with Flash Attention, which optimizes memory access patterns, the computational complexity remains O(n²).

**Why is this critical?** Because the context window has become a competitive advantage. Gemini 2.5 Pro with 1M+ token context, Claude Sonnet 4 with 200K, Llama 4 Scout with 10M tokens — long context has shifted from a desirable feature to a necessity for enterprise use cases. But quadratic complexity makes this prohibitively expensive with classic attention.

**Attempts to solve through attention modifications** proceeded in parallel: Sparse Attention (process not all token pairs), Linear Attention (approximate softmax), Flash Attention (optimize memory access). All of these improved constants but did not change the fundamental O(n²) complexity.

**SSM offers a radical solution:** abandon attention entirely. Instead of computing relationships between all token pairs, compress history into a fixed-size state and update it sequentially. This yields O(n) complexity — linear, not quadratic.

### SSM: Inspiration from Control Theory

State Space Models come from an entirely different field — control theory and signal processing.

**Classical SSM formulation:** The system has a hidden state x(t) that updates with each new input u(t) according to learnable parameters A, B, C, D. The output y(t) is computed from the current state.

**Key property:** The state x(t) compresses the entire input history into a fixed size, regardless of sequence length. Whether you have processed 100 tokens or 100K tokens, the state has the same size (typically 16-256 dimensions).

**Fundamental difference from Attention:** In the classic attention mechanism, memory grows linearly with sequence length — keys and values must be stored for all previous tokens. Processing 10K tokens requires storing 10K key-value vector pairs. In SSM, by contrast, memory is constant: regardless of whether 100 or 100K tokens have been processed, the state vector has a fixed size (e.g., 64 dimensions). When a new token is added, the state is updated in-place, compressing new information into the existing representation. This yields O(1) memory footprint instead of O(n).

### Mamba: Structured State Space Models

Mamba (December 2023) introduced a key innovation — selective state spaces:

**The vanilla SSM problem:** Parameters A, B, C, D are fixed for all inputs. This means the model cannot adaptively choose which information to retain in the state.

**Mamba's solution:** Make B, C, and Δ (discretization step) input-dependent:
- B(x), C(x) — input-dependent projection
- Δ(x) — controls how strongly the state is updated

### Why Selective State Spaces Are Critical

Understanding **why** input-dependent selection is so important requires a deeper look at the problem:

**The vanilla SSM (S4) problem:** In a classic SSM, parameters A, B, C are fixed. This means every token is "remembered" with equal strength. The model cannot "choose" what is important and what to ignore. Information accumulates linearly, without prioritization.

**Analogy:** Imagine reading a book but being forced to remember every word with equal intensity — both the articles "the" and the key terms. Building understanding is impossible. The human brain works through selective attention — we automatically filter out irrelevant information.

**How selection solves the problem:**

Mamba makes parameters B, C, and Δ (discretization step) input-dependent:
- **Δ(x)** controls "memory strength" — a small value means ignoring the token, a large value means a strong state update
- **B(x)** determines "what to write" — input-dependent projection highlights relevant features
- **C(x)** determines "what to read" — adaptive retrieval from the compressed state

**Key insight:** A selective SSM can emulate attention-like behavior — "paying attention" to important tokens by retaining them in the state with higher weight, and ignoring noise. All while maintaining O(n) complexity instead of O(n²).

**Practical result:** Comparable quality with Transformers on many tasks with significantly better performance on long sequences. This is not a theoretical possibility — Mamba models demonstrate this in practice.

### Mamba-2: Structured State Space Duality

Mamba-2 (May 2024) went further, discovering a deep connection between SSM and attention:

**Key theoretical result — State Space Duality (SSD):** The authors showed that selective SSM can be reformulated as a form of **structured masked attention**. SSM computations are mathematically equivalent to a specific form of attention with a structured mask.

**Why this matters:**

First, it provides **theoretical justification** — SSMs are not merely "similar" to attention; they are mathematically equivalent to a specific subclass. This explains why SSMs can achieve comparable quality.

Second, it enables **optimization** — the reformulation allows using GPU tensor cores optimized for matrix operations instead of sequential scans.

Third, it creates a **unified framework** for SSM and attention layers, simplifying hybrid architectures.

**Practical improvements in Mamba-2:**
- 2-8× speedup compared to Mamba-1 due to better GPU utilization
- Simplified architecture — fewer hyperparameters to tune
- State capacity increased from 16 to 64-256 dimensions — more "memory" for history
- Better scaling on multi-GPU setups

### RWKV-6: A Linear Attention Alternative

RWKV (Receptance Weighted Key Value) is another approach to linear complexity, developed by an active open-source community.

**RWKV philosophy:** Instead of a state space formulation, RWKV modifies the attention mechanism itself for linear complexity through **time-mixing** (processing temporal dependencies) and **channel-mixing** (processing feature dimensions).

**Key difference from Mamba:** RWKV is closer to an RNN with attention-like components, while Mamba stems from state space theory. Both achieve O(n) complexity and O(1) inference per token, but via different paths.

**RWKV-6 evolution:** Version 6 added data-dependent decay (adaptive "forgetting") and improved normalization. The Eagle and Finch variants expanded state capacity and added multi-scale processing.

**When to choose RWKV:**
- Open-source project with MIT license — full transparency and control
- Active community development — rapid iterations
- Strong multilingual support — especially for Asian languages
- Focus on edge devices — well optimized for resource-constrained environments

**Trade-offs compared to Mamba:** RWKV has a more mature ecosystem due to its earlier start, but Mamba has stronger theoretical foundations and support from research institutions.

### When SSMs Outperform Transformers

**SSMs win:**
- Long sequences (>32K) — constant memory vs. quadratic growth
- Streaming applications — O(1) latency per token vs. O(n) recomputation
- Edge deployment — smaller memory footprint
- Document summarization — efficient processing without precise retrieval

**Transformers win:**
- Precise in-context retrieval ("quote line 47")
- Complex multi-hop reasoning between distant parts of the context
- Few-shot learning — dynamic adaptation through examples in the prompt
- Mature tooling ecosystem — debugging, optimization, deployment

**Empirical reality:** On standard benchmarks, Mamba is roughly on par with a Transformer of the same size. On long context (>32K), Mamba significantly outperforms Transformers in efficiency. On precise information retrieval, Transformers significantly outperform Mamba. The choice depends on the distribution of your specific use cases.

---

## Hybrid Architectures: The Best of Both Worlds

### Why Hybrid Models?

Neither Transformers nor SSMs are universally superior. Hybrid architectures attempt to combine:
- SSM efficiency for long-sequence processing
- Attention precision for retrieval tasks

**Fundamental insight:** Different layers of a model serve different functions. Early layers process syntax and low-level patterns — the full attention matrix is not needed here. Later layers perform reasoning and retrieval — attention is critical here.

This leads to the idea of **layer-wise specialization:** use SSM for most layers (efficiency), but periodically insert attention layers for "global information synchronization."

**The design space is vast:** How many SSM layers per attention layer? What positions are optimal for attention? Static or dynamic routing? Each decision creates a unique trade-off profile.

**Why hybrids only now?** Because only in 2023-2024 did sufficiently mature SSMs (Mamba) and understanding of how to combine them with attention emerge. Jamba became the first production-ready hybrid model, proving the viability of the approach.

### Jamba: SSM + Attention + MoE

AI21 Labs introduced Jamba (March 2024) — the first production-ready hybrid model combining all three approaches:

**Key characteristics:**
- Alternating Mamba and Attention layers (~7:1 ratio) + MoE in FFN
- 52B total parameters, only 12B active per token
- 256K token context with a smaller memory footprint than a pure Transformer
- Quality comparable to Mixtral 8x7B

**Why this matters:** Jamba proved that hybrid architectures work in production. It paved the way for NVIDIA Bamba (optimized for TensorRT-LLM) and IBM Granite 4.0 (enterprise focus with compliance features).

### Architectural Patterns of Hybrid Models

**Pattern 1: Layer alternation** — SSM layers for efficient processing, periodic attention layers for "global synchronization." For example: 7 SSM layers → 1 attention layer → 7 SSM → 1 attention. This yields 87.5% efficiency of pure SSM while preserving attention capabilities.

**Pattern 2: Parallel branches** — SSM and Attention process the input in parallel, results are combined (weighted sum or gating). Delivers better quality but requires more compute — both paths are active simultaneously.

**Pattern 3: Conditional routing** — dynamic selection of the SSM or Attention path based on input characteristics. The most complex approach, but potentially the most efficient — attention is used only when genuinely needed.

### Architecture Selection: A Practical Approach

**Simple starting rule:**
- Long context (>32K) dominates? → SSM or Hybrid
- Precise in-context retrieval is critical? → Transformer
- Inference cost is the primary concern? → MoE
- Need a balance of everything? → Hybrid (if engineering capacity is available)

**Going deeper: task characteristics determine the choice**

Precise retrieval ("find and quote line 47 from the contract") requires attention — SSMs compress history and lose exact positions. But semantic summarization of a long document works well on SSMs at significantly lower cost.

Streaming applications (real-time chat, live transcription) are a natural fit for SSMs — O(1) per token means constant latency regardless of history length. Transformers require recomputation of the entire history for each new token.

High-throughput scenarios with short queries benefit from Transformers — batching is efficient, and quadratic complexity is not critical on short sequences.

**Infrastructure realities**

GPU memory < 24GB: Dense 7B with quantization — deployment simplicity matters more than cutting-edge architecture.

GPU memory 80GB+: MoE (best quality/compute ratio) or large dense models become feasible.

Latency budget < 100ms TTFT: SSM — the absence of quadratic complexity provides predictably low latency.

**Team competency — an underestimated factor**

Team without ML experience? Dense Transformer via cloud API or standard inference servers. Massive ecosystem, minimal surprises.

ML engineers on the team? MoE or SSM can deliver significant gains in cost and performance with acceptable engineering overhead.

Research team? Experiment with hybrids — the potential gains are enormous, but deep understanding is required.

**Risk tolerance and production readiness**

Conservative organization, mission-critical application? Dense Transformer — proven technology, mature tooling, predictable behavior.

Startup optimizing costs, willing to experiment? SSM or MoE can provide a competitive advantage through efficiency.

Enterprise with diverse workloads, long-term investments? Hybrid architectures — more complex now, but potentially the best long-term choice.

---

## Practical Implications for the AI Lead

### Selecting a Model for a Project: A Systematic Approach

When choosing an architecture for a production system, consider not individual characteristics but their interactions:

**1. Use case characteristics determine the optimal architecture**

If your median context length is < 8K tokens, the quadratic complexity of attention is not critical. Dense Transformer provides the best quality and mature tooling. But if 20% of requests use >32K context, those 20% may account for 80% of inference costs.

**Real-world case:** A company performs legal document analysis. The median document is 3K tokens (summaries, contracts). But 15% of documents are court cases with >50K tokens. On a Dense Transformer, those 15% create 70% of costs. Switching to a hybrid SSM/Attention architecture reduced total costs by 55%.

**2. Infrastructure constraints are often underestimated**

MoE models require not just "more memory" but specific infrastructure. If you have a fleet of GPU servers across different generations (A100 + V100), expert parallelism may be impractical — different memory capacities and interconnect bandwidths will create bottlenecks.

**Real-world trap:** "We have 4xA100 80GB, let's run Mixtral." But efficient MoE inference requires high-speed interconnect. On servers without NVLink, you will get only 30% of theoretical throughput due to communication overhead.

**3. Team expertise — an often-ignored factor**

Debugging MoE routing issues or SSM state corruption requires deep understanding of internals. If you do not have ML researchers on the team, experiments with cutting-edge architectures can turn into months of debugging.

**Rule of thumb:** If your team is comfortable working with huggingface/transformers but has no experience modifying training loops, start with Dense or a well-supported MoE (Mixtral). Leave SSM and hybrids for teams with strong research backgrounds.

### Cost-Benefit Analysis: Real Numbers

**MoE (Mixtral-style) in production:**
- Compute savings: 60-70% compared to a dense model of equivalent quality
- Memory overhead: 3.5-4× compared to active parameters
- Infrastructure complexity: +40% compared to dense model deployment
- **Optimal scenario:** High-throughput systems where compute is more expensive than memory

**When MoE pays off:** If you process >100M requests per month, the compute savings of hundreds of thousands of dollars offset the infrastructure investment.

**SSM (Mamba-style) in production:**
- Long-context efficiency: 5-10× faster than Transformer on contexts >32K
- Quality gap: 5-15% worse on standard benchmarks, but comparable on long documents
- Tooling maturity: ~60% of the Transformer ecosystem
- **Optimal scenario:** Document processing, streaming applications, edge deployment

**When SSM pays off:** If >50% of your requests use >16K context OR latency is critical (streaming chat, real-time processing).

**Hybrid architectures:**
- Balance: 70-80% of SSM efficiency + 85-90% of Transformer quality
- Development overhead: 2-3× more compared to a pure architecture
- Production track record: limited (Jamba is the only widely known example)
- **Optimal scenario:** When neither Dense nor SSM fully satisfies requirements

**When hybrids are justified:** Enterprise scenarios with diverse workloads where neither quality nor efficiency can be sacrificed, and engineering resources exist for custom optimization.

### Architecture Landscape in 2025: Confirmed Trends

**MoE is the dominant frontier architecture:** What was a prediction is now reality. DeepSeek V3/R1 (671B total, 37B active), Llama 4 Scout and Maverick, Qwen 3 235B, Gemini 2.5 Pro, and GPT-5 all use MoE. At scales >100B parameters, dense inference is economically impractical. MoE is no longer experimental — it is the default.

**Llama 4 architecture (April 2025)** deserves special attention. Meta's Llama 4 introduced MoE with native multimodal early fusion — images, text, and video are tokenized into a shared space from the start, not bolted on via adapters. **Scout** (109B total, 17B active, 16 experts) achieves a 10M token context window through an interleaved attention pattern. **Maverick** (400B total, 17B active, 128 experts) targets quality-competitive performance with GPT-4o and Claude Sonnet 4. The MoE routing uses a shared expert (always active) plus routed experts, similar to DeepSeek's approach.

**SSMs found their niche but did not replace Transformers:** Mamba-2 (May 2024) proved SSM-attention duality and improved GPU utilization 2-8x. RWKV-6 added adaptive forgetting. But pure SSMs have not surpassed Transformers on broad benchmarks. The practical outcome: SSMs excel at long context and edge deployment, but the main event is hybrid architectures.

**Hybrid architectures are the practical path:** Jamba proved the concept in early 2024. By 2025, hybrid designs are mainstream: NVIDIA Bamba (optimized for TensorRT-LLM), IBM Granite 4.0, Google's Griffin and Hawk models. The typical pattern is SSM layers for efficiency with periodic attention layers for retrieval precision.

**Hardware co-evolution is real:** NVIDIA Blackwell GPUs (2024) and the H200 are optimized for sparse computation patterns. FP8 training (pioneered by DeepSeek V3) is now standard practice, delivering ~2x compute efficiency. The hardware-software co-design cycle is accelerating.

---

## Key Takeaways

**Architectural evolution:** We are living through a fundamental shift. Dense Transformers dominated in 2017-2023, but their limitations (quadratic complexity, inefficient parameter utilization) have become a bottleneck. MoE and SSM attack these problems from different angles.

**Trade-offs are inevitable:** MoE provides more capacity with less compute but requires more memory. SSMs deliver linear complexity but handle precise information retrieval less well. Hybrids balance these factors but are harder to deploy. There is no universally best choice.

**Business impact:** Architectural decisions determine not only technical characteristics but the economics of the entire system. The right choice can reduce costs by 50-70% or enable previously impossible use cases (256K+ token context).

**The senior engineer's responsibility:** Understanding these trade-offs distinguishes experienced engineering from simply "using an API." You need to know not only "how Mamba works" but also "why Mamba could save $200K per month for our use case, but would require 3 months of infrastructure work."

**Tooling gap:** New architectures have less mature tooling. Be prepared for additional engineering investment — or accept the conservative choice (Dense Transformer) with full awareness of the missed opportunities.

---

## Practical Examples

### Working with Modern Architectures

**Loading and using MoE models (Mixtral):**

MoE models require more GPU memory (all 47B parameters must be loaded), but use only ~13B active parameters during inference. For deployment on consumer hardware, apply 4-bit quantization via BitsAndBytes. Mixtral uses the ChatML format for prompts — use the tokenizer's `apply_chat_template()` for proper formatting.

**Key considerations:**
- MoE requires device_map="auto" for automatic expert distribution across GPUs
- Quantization reduces memory footprint from ~94GB (bf16) to ~24GB (4-bit)
- Router decisions are not visible through the standard API — custom hooks are needed for analysis

**Working with SSM models (Mamba):**

SSM models demonstrate linear scaling with context length. When benchmarking on sequences of 1K, 4K, 16K, 32K tokens, you will observe:
- Time per token remains constant (O(1))
- Memory grows linearly, not quadratically
- At contexts >16K tokens, the advantage over Transformers becomes dramatic

**Performance testing:**

When comparing architectures, measure three key metrics:
1. Time to First Token (TTFT) — generation start latency
2. Tokens per second — throughput after generation begins
3. Peak memory usage — maximum memory consumption

For correct benchmarking:
- Perform a warm-up run to initialize CUDA kernels
- Use torch.cuda.synchronize() before and after measurements
- Test at realistic context lengths for your use case

**Architectural trade-offs in real numbers:**

Dense Transformer (e.g., Mistral 7B) has 7B active parameters out of 7B total, requires approximately 14GB of memory in bf16 format, generates roughly 150 tokens per second at 4K context, with a Time to First Token of approximately 50ms.

MoE architecture (Mixtral 8x7B) activates 13B parameters out of 47B total, requiring significantly more memory — approximately 94GB in bf16 or ~24GB with 4-bit quantization. Generation speed is approximately 100 tokens per second at 4K context with a TTFT of approximately 40ms. Key point: more total parameters yield better quality, but active parameters determine inference speed.

SSM models (Mamba 2.8B) demonstrate the best efficiency for their size: 2.8B parameters, only ~6GB of memory, speeds up to 200 tokens per second on short context with a TTFT of just ~20ms. The critical advantage emerges on long context: at 32K tokens, Mamba maintains ~180 tok/s, while a comparable Transformer drops to ~30 tok/s due to the quadratic complexity of attention.

### Practical Decision-Making Framework

**Choosing an architecture by workload profile:**

If your median context length is < 8K tokens AND precise retrieval is required — use a Dense Transformer. Tooling is mature, quality is predictable, deployment is straightforward.

If you need long context (>32K) OR streaming processing — consider SSM (Mamba). Dramatic efficiency advantages on long sequences compensate for less mature tooling.

If inference cost is critical but GPU memory is available — MoE delivers large-model quality at small-model computational cost. Trade-off: more complex deployment, more memory required.

For enterprise use cases with diverse requirements — hybrid models (Jamba-style) balance characteristics but require greater investment in infrastructure and R&D.

**Hardware considerations:**

On consumer GPU (24GB): Dense 7B quantized or Mamba up to 8B
On single A100 (80GB): Dense up to 34B, MoE like Mixtral (quantized), Mamba up to 13B
On multi-GPU setup: any architecture, but MoE requires high-speed interconnect

**Team competency assessment:**

If you lack a dedicated ML infrastructure team — start with Dense Transformer. Massive ecosystem, ready-made solutions, minimal surprises.

If you have ML engineers and willingness to optimize — SSM or MoE can deliver significant gains in performance and cost.

Only for teams with strong research competencies — experiment with hybrid architectures. Cutting-edge technology, but limited production experience.

## Brief Code Example

### Basic Inference with Different Architectures

All modern architectures use a unified API through the Transformers library. The basic loading and generation pattern is identical: use `AutoTokenizer` and `AutoModelForCausalLM`, specify `torch_dtype=torch.bfloat16` for memory efficiency, and `device_map="auto"` for automatic GPU distribution. For Dense models, load a model such as "mistralai/Mistral-7B-v0.1"; for MoE, "mistralai/Mixtral-8x7B-v0.1"; for SSM, "state-spaces/mamba-2.8b-hf". After tokenizing the input, call `model.generate()` with specified maximum length and temperature to control diversity.

**Quantizing MoE for consumer GPUs:** Running Mixtral on consumer hardware requires 4-bit quantization via BitsAndBytesConfig. It reduces memory from ~94GB (bf16) to ~24GB, allowing the model to fit on a single consumer GPU. Use the parameters `load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.bfloat16` for computation precision, and `bnb_4bit_quant_type="nf4"` for the optimal balance of quality and compression.

**Long-context benchmarking:** For a correct comparison of SSM and Transformer on long sequences, test across various lengths (1K, 4K, 16K, 32K tokens). Always perform a warm-up run to initialize CUDA kernels, then use `torch.cuda.synchronize()` before and after timing for accuracy. SSM will show constant or near-linear processing time, while Transformers demonstrate quadratic growth with increasing context length.

**Memory footprint analysis:** For MoE models, it is important to distinguish total from active parameters. Compute the total parameter count via `sum(p.numel() for p in model.parameters())`, then for MoE check for the `num_experts_per_tok` attribute in config. Active parameters are a fraction of total: shared parameters (attention, embeddings) plus only selected experts. This explains why Mixtral with 47B parameters uses only ~13B during inference, delivering the efficiency of a smaller model with the capacity of a larger one.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[06_Integration_Patterns|LLM Integration Patterns]]
**Next:** [[08_Scaling_Laws|Scaling Laws]]
