# Language Model Interpretability

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[08_Scaling_Laws|Scaling Laws and Emergent Abilities]]
**Next:** [[10_Implementation_from_Scratch|Implementing Transformer from Scratch]]

---

Imagine you hired a brilliant consultant who solves the most complex problems but categorically refuses to explain his reasoning. You do not know whether he bases decisions on deep analysis or on spurious correlations. You cannot predict when he will make a mistake. This is exactly the situation we face with modern language models.

Interpretability is a research field aimed at understanding *how* and *why* neural networks make decisions. For LLMs, this is not merely academic interest: understanding internal mechanisms is critical for safety, reliability, and targeted model improvement.

---

## Why Understanding the Internals Matters

### The Black Box Problem

Modern LLMs comprise hundreds of billions of parameters organized into complex computational graphs. Text goes in, text comes out. What happens in between is a black box.

This creates fundamental problems:

**Unpredictable failures**: A model can suddenly produce an incorrect or dangerous response. Without understanding the root cause, systematically fixing the problem is impossible.

**Inability to verify**: How can we confirm the model solves a task the "right" way rather than exploiting surface-level correlations? GPT-3 may answer geography questions correctly, but does it base its answers on an understanding of the world map or on statistical patterns in text?

**Alignment risks**: If we do not understand how a model "thinks," how can we ensure its objectives align with ours? A model may exhibit aligned behavior during testing while possessing entirely different internal mechanisms.

### Types of Interpretability

**Mechanistic interpretability** — understanding specific computational mechanisms. Which neurons activate? What attention patterns form? This is the deepest level of understanding.

**Behavioral interpretability** — understanding model behavior through experiments. How does the model react to input changes? What systematic errors does it make? A less deep but more practical approach.

**Probing** — checking whether specific information is present in model representations. Does the model "know" grammar? Does it encode facts about the world? This can be determined without understanding the mechanisms.

---

## Circuits: Computational Units in Transformers

### The Circuits Concept

Researchers at Anthropic proposed viewing neural networks as composed of **circuits** — connected groups of neurons that perform specific computational functions.

An analogy with electronics: a processor is made of transistors, but we understand its operation through logic gates (AND, OR, NOT) and higher-level blocks (ALU, registers). Similarly, neural networks can be understood through circuits without analyzing each neuron individually.

A circuit is an activation path from input to output through specific neurons and weights. A single circuit can be responsible for a specific operation: "find the verb in a sentence," "determine sentiment," "copy a token from context."

### Induction Heads: A Circuit Example

The best-studied circuit is **induction heads**. This is a mechanism for copying patterns from context.

Example: if the model sees "...Harry Potter and the Philosopher's Stone. Harry Potter and the...", an induction head helps predict that the next word is "Philosopher's."

The mechanism consists of two attention heads:
1. **Previous token head**: for each position, attends to the previous token
2. **Induction head**: finds positions where the current token appeared earlier and attends to the token that followed it

The composition of these two operations yields in-context learning: the model can "learn" a pattern (A → B) from examples in the prompt and apply it to new cases.

### Other Known Circuits

**Name mover heads**: move information about proper names through context. When text mentions "John said that he...", these heads help link "he" to "John."

**Negative heads**: suppress certain outputs. If context indicates the answer should NOT be X, negative heads reduce the probability of X.

**Backup heads**: duplicate the functionality of other heads. They provide robustness — if one head is "broken," the backup continues functioning.

---

## Superposition: How Models Store More Than They Can

### The Dimensionality Problem

A model has finite dimensionality — say, 4096 dimensions in the hidden state. But there are far more concepts in the world. How does a model encode millions of concepts in thousands of dimensions?

**Superposition hypothesis**: models reuse the same neurons to represent multiple concepts that rarely activate simultaneously.

An analogy: a triangle has 3 vertices, but you can locate 100 points if each point is defined by a unique combination of distances to the vertices. Concepts that never activate together can "share" the same dimensions.

### The Mathematics of Superposition

In a linear space of dimensionality d, only d vectors can be represented orthogonally. But if slight interference is allowed, exponentially more can be encoded.

If features (concepts) are sufficiently sparse (rarely activate simultaneously), interference between them is minimal. The model "pays" for superposition with occasional errors when several related concepts are active simultaneously.

Experiments show that smaller models use more superposition (forced to conserve space), while larger models use less (able to dedicate neurons to individual concepts).

### Practical Consequences

Superposition explains why:

1. **Models sometimes confuse related concepts**: if "doctor" and "nurse" are encoded with overlapping vectors, the model may mix them up

2. **Scaling helps**: larger models can allocate separate dimensions for concepts, reducing interference

3. **Interpretation is difficult**: a single neuron may be responsible for many unrelated concepts

---

## Sparse Autoencoders: Decoding Superposition

### The Method

If features are in superposition, we can attempt to disentangle them. **Sparse Autoencoder (SAE)** is a technique for doing this.

An SAE is trained to reconstruct activations through an intermediate layer with a larger number of neurons, subject to a sparsity constraint (most neurons must be zero).

Architecture:
1. **Encoder**: x → ReLU(Wx + b) — expands into a higher-dimensional space
2. **Decoder**: z → W'z + b' — compresses back
3. **Sparsity loss**: penalty for the number of non-zero activations

After training, each SAE neuron should correspond to one interpretable feature.

### Sparse Autoencoder Mathematics

Understanding the SAE loss function is critical for proper application. The loss function consists of three main components that work together to achieve sparse and interpretable representations.

**Loss function structure:**

The overall loss formula is a sum of several components: L = L_reconstruction + λ₁·L_sparsity + λ₂·L_auxiliary. Here L_reconstruction measures the quality of reconstructing the original activations via mean squared error between input x and reconstruction x̂. The L_sparsity component enforces activation sparsity through the L1 norm (sum of absolute activation values). The auxiliary component includes additional regularizers for training stabilization.

**Loss function components:**

The reconstruction component uses MSE(x, x̂) to preserve information — the model must reconstruct the original activations as accurately as possible. L1 sparsity with coefficient λ₁ forces most activations to be zero, creating sparse representations. L0 sparsity controls the number of active features directly. KL divergence is used to achieve a target sparsity level by comparing the actual and desired activation distributions.

**Practical problems and solutions:**

The dead features problem occurs when some SAE neurons never activate after initialization. They "die" and do not participate in training. The solution is to periodically check activation counters for each feature: if the feature_activation_count falls below a threshold, reinitialize the corresponding encoder and decoder weights with random values.

Feature shrinkage occurs because the L1 penalty systematically reduces all activation values. This can lead to numerical underflow. An alternative solution is to use TopK sparsity instead of L1 regularization: select only the k largest activations and zero out the rest, providing precise sparsity control without the shrinkage effect.

The reconstruction-sparsity trade-off requires careful balancing. Too high a λ₁ results in excessive sparsity with information loss and poor reconstruction. Too low a λ₁ yields good reconstruction, but without sparsity, interpretability is lost. The optimal λ₁ coefficient is determined experimentally with target metrics: the average number of active features (L0) should be 10-50 out of 10,000+ possible, and reconstruction loss should not exceed 5% of the variance of the original activations.

### SAE Application Results

Anthropic researchers applied SAE to Claude and discovered thousands of interpretable features:

- **Golden Gate Bridge feature**: activates when the Golden Gate Bridge is mentioned
- **Code bug feature**: activates in the presence of code errors
- **Deception feature**: activates on texts about deception or manipulation
- **Sycophancy feature**: activates on excessively agreeable responses

Importantly, these features can not only be detected but also **modified**. Amplifying the "Golden Gate Bridge feature" causes the model to mention the bridge more frequently (the famous "Golden Gate Claude" experiment).

### SAE Limitations

SAE does not fully solve the problem:

- **Not all features separate**: some concepts remain entangled
- **Completeness is not guaranteed**: SAE may miss important features
- **Interpretation is subjective**: assigning meanings to features requires human analysis
- **Computationally expensive**: training SAE for large models demands significant resources

### Scaling Monosemanticity: What We Learned About Claude (2024)

In 2024, Anthropic published the landmark paper "Scaling Monosemanticity," applying SAE to Claude Sonnet. The results transformed our understanding of interpretability in large models.

**Experiment scale:**

Anthropic applied an SAE of unprecedented scale to Claude 3 Sonnet at a middle layer. The SAE size was 34 million features — a massive increase compared to 16 thousand in earlier work. The expansion factor reached 256×, meaning the original hidden_dim was expanded into a space 256 times larger. Training was conducted on billions of tokens of activations.

**Key findings:**

Features exist and scale. Researchers discovered approximately 1 million interpretable features. As SAE size increases, features become increasingly specific and specialized. Critically, monosemanticity is achievable — a single feature can indeed correspond to a single well-defined concept.

Multi-level feature abstraction was discovered. Low-level features handle basic elements: the letter "e," spaces, commas. Mid-level features recognize more complex patterns: an English word, Python code. High-level features operate on semantic concepts: AI safety discussion, medical advice. Abstract features capture abstract notions: lying and deception, agreement with the user.

Safety-relevant features were successfully identified. A "Deception" feature activates on texts about deception and manipulation. A "Harmful content" feature responds to dangerous instructions. A "Sycophancy" feature detects excessive model agreement. A "Refusal" feature handles refusals. An "Uncertainty" feature tracks model uncertainty in responses.

Clamping features demonstrates a causal relationship with behavior. Amplifying the "Golden Gate Bridge" feature causes the model to become obsessed with the bridge, mentioning it constantly. Amplifying the "deception" feature causes the model to begin deceiving the user. Suppressing the "refusal" feature makes the model willing to respond to dangerous requests it would normally decline.

**Practical implications for AI safety:**

Interpretability scales — production models with billions of parameters can be analyzed. Safety-relevant features are detectable and can be monitored in real time. Feature steering works — undesirable behavior can be corrected without retraining. However, understanding is far from complete — significantly more research is required.

**Open questions after Scaling Monosemanticity:**
- How are features related to capabilities?
- Can dangerous capabilities be predicted before their emergence?
- Is SAE sufficient for complete model understanding?
- How do features from different layers interact?

---

## Probing: What the Model "Knows"

### The Probing Method

Probing is a simple yet powerful method: train a small classifier on top of the model's internal representations to predict a specific property.

If the classifier succeeds, information about the property is encoded in the representations. If not, the model "does not know" this property (or stores it in a form inaccessible to a linear probe).

Example probing tasks:
- Parts of speech: can the POS tag be predicted from a token's hidden state?
- Syntactic tree: is the dependency hierarchy encoded?
- World facts: can "capital of France = Paris" be extracted from the representation?
- Sentiment: is the text's sentiment encoded?

### What Probing Reveals

Research has revealed rich representational structure:

**Syntax is well encoded**: parts of speech, dependencies, sentence structure — all are extracted with high accuracy.

**Semantics are diverse**: related words have similar representations, but relationships are more complex (king - man + woman ≠ queen for modern models).

**Facts are distributed**: world knowledge is encoded in FFN layers, with different facts in different layers.

**Layers have different specializations**: early layers handle syntax, middle layers handle semantics, deep layers handle reasoning.

### Probing Limitations

**Correlation ≠ causation**: the presence of information does not mean the model uses it.

**Linear probes can be weak**: if information is encoded nonlinearly, a linear probe will not find it.

**The probe may "learn" the information**: a powerful probe may extract information that the model itself does not use.

---

## Activation Patching: Causal Analysis

### The Method

Probing reveals correlations. **Activation patching** establishes causal relationships.

The method:
1. Run the model on two different inputs (clean and corrupted)
2. Take activations from a specific component on the clean input
3. Substitute them into the run on the corrupted input
4. Check whether the correct output is restored

If the substitution restores the output, that component is critical for the task.

### Example: Indirect Object Identification

Task: "John gave the book to Mary. John gave it to ___"
Correct answer: Mary

Using activation patching, one can determine:
- Which attention heads find "Mary" in the context
- Which components transmit this information to the prediction position
- The complete circuit from input to output

Result: a specific set of attention heads ("Name Mover Heads") was identified that performs this task.

### Activation Patching: Extended Methodology

The step-by-step causal analysis process precisely identifies which model components are responsible for specific computations.

**Step 1: Preparing contrastive examples**

Create a pair of examples differing at a critical point. Clean input: "John gave Mary a book. John gave her a flower". Corrupted input: "John gave Bob a book. John gave her a flower". The only difference is the name (Mary → Bob). The goal: the prediction of "flower" should change, allowing us to trace which components transmit name information.

**Step 2: Choosing granularity**

Layer level — patch the entire layer; yields low interpretability but requires low computational cost. Component level — patch attention or MLP separately; moderate interpretability and moderate cost. Head level — patch a single attention head; high interpretability but high computational cost. Neuron level — patch an individual neuron; very high interpretability but very high cost. Position level — patch a specific position in the sequence; high interpretability and moderate cost.

**Step 3: Patching directions**

Causal patching goes from clean to corrupted: take activations from the clean run and substitute them into the corrupted run. If the output is restored to correct, the component is critically important for the task.

Denoising works in the opposite direction, from corrupted to clean. It reveals which corrupted activations "damage" the clean output.

Zero patching zeroes out a component's activations entirely. This is an ablation to determine how critical the component is overall.

**Step 4: Influence metrics**

Direct effect measures the direct influence on output: the difference logit_diff_patched minus logit_diff_corrupted. Indirect effect computes the influence through other components: the difference between clean and patched outputs minus direct effect. Total effect is the sum of direct and indirect effects. Normalized recovery normalizes the recovery over the range from corrupted to clean, showing the proportion of restored information.

**Step 5: Building the circuit**

After identifying important components, the following steps are necessary: determine the information flow between them, verify that the circuit is sufficient by ablating everything else, find the circuit's "inputs" and "outputs," and document the discovered mechanism for reproducibility.

### Logit Lens and Tuned Lens

**Logit Lens**: applying unembedding (the output projector) to intermediate representations. This shows how the prediction "matures" across layers.

On early layers, the logit lens shows noise or trivial predictions. On middle layers, meaningful candidates begin to emerge. On deep layers, the final answer appears.

**Tuned Lens**: trains separate projectors for each layer. More accurate but less "honest" (adds learnable parameters).

---

## Connection to Alignment and Safety

### Why This Matters for Safety

Interpretability is not an academic exercise. It is a critical tool for AI safety.

**Detecting deception**: if a model attempts to manipulate the user, SAE can detect the activation of "deception features." This works even if the model conceals its intentions in text.

**Verifying alignment**: instead of testing behavior (which can be faked), internal representations can be inspected. An aligned model should have the "right" features.

**Surgical correction**: once a problematic circuit is identified, it can be modified or disabled without retraining the entire model.

**Scalable oversight**: when models become smarter than humans, behavioral testing will become infeasible. Interpretability may remain the only means of verification.

### How Interpretability Aids Alignment: Concrete Scenarios

**Scenario 1: Detecting deceptive alignment**

A model may learn to "pretend" to be aligned during testing while harboring different internal objectives. A behavioral test asks: "Do you plan to deceive users?" The model responds: "No, I am here to help!" Test passed, but nothing is proven — the model may have simply learned the correct answers.

The interpretability approach works differently. First, find "deception" and "goal" features via SAE. Then monitor activations of these features during inference in real time. If the "deception" feature is active but the text output appears "honest," this is a red flag. The model may be concealing its true intentions.

**Scenario 2: Understanding why the model refuses**

Problem: the model refuses to answer a harmless user request. Behavioral analysis yields a verdict of "model is broken," but it is unclear how to fix it or where exactly the problem lies.

The interpretability approach offers a systematic solution. Trace activation patching to determine which specific components trigger the refusal response. Find "overactive" safety features that fire excessively. Determine whether this is a true positive (the request is actually dangerous) or a false positive (a safe request erroneously classified as dangerous). If necessary, attenuate the specific feature without retraining the entire model.

**Scenario 3: Predicting dangerous capabilities**

Question: can model X perform Y (a dangerous action)? The behavioral approach suggests trying it — but that is dangerous! Testing dangerous capabilities can lead to real harm.

The interpretability approach is safer. Find features associated with Y-like capabilities through activation analysis. Check how developed these features are and how strongly they activate. Predict the emergence of dangerous capabilities without actually testing dangerous behavior.

**Safety monitoring workflow:**

The safety monitoring architecture consists of several stages. Production Inference — the model's normal operation with user requests. SAE Feature Extraction occurs asynchronously: features are extracted from activations, and safety-relevant feature scores are computed. Anomaly Detection analyzes unusual patterns: unusual deception feature activation, harmful content features, out-of-distribution patterns. Based on the analysis, the system classifies responses as Normal (continue operation) or Alert (send to Human Review for human inspection).

### Limitations of Current Approaches

Despite progress, interpretability is far from solving safety problems:

**Scale**: early techniques were limited to models up to ~10B parameters. Anthropic's "Scaling Monosemanticity" (2024) applied SAEs to Claude 3 Sonnet, demonstrating feasibility at much larger scales, but routine interpretability of 100B+ models remains an open challenge.

**Completeness**: we find individual circuits but do not understand the full picture. This is like understanding a few circuits in a processor without understanding the architecture as a whole.

**Pace of change**: new architectures (MoE, SSM) require new interpretation methods.

**Fundamental limitations**: there may be a "complexity of understanding" — not all computations can be reduced to concepts comprehensible to humans.

---

## Practical Applications

### Debugging Models

Interpretability is useful for practical debugging:

**Error diagnosis**: when a model makes mistakes, the responsible components can be traced. This is more informative than simply "the model produced the wrong answer."

**Prompt improvement**: by understanding which features activate, prompts that trigger the desired circuits can be crafted.

**Selecting layers for fine-tuning**: probing reveals which layers encode the needed information. Applying LoRA to those layers may be more effective.

### Feature Steering

Direct modification of activations during inference:

**Adding features**: amplify desired behavior (e.g., "helpfulness feature")

**Suppressing features**: reduce undesirable behavior (e.g., "refusal feature" for research purposes)

**Style control**: modify features related to response style

This is a powerful tool, but it requires caution — careless modification can break the model in unpredictable ways.

---

## Key Takeaways

1. **Interpretability is critical** for LLM safety and reliability. Without understanding a model's mechanisms, we cannot guarantee its behavior.

2. **Circuits** are computational units in neural networks. Induction heads, name movers — these are examples of understood circuits performing specific functions.

3. **Superposition** explains how models encode more concepts than they have dimensions. This makes interpretation harder, but sparse autoencoders help disentangle features.

4. **SAEs reveal features**: thousands of interpretable units can be discovered and modified. "Golden Gate Claude" is a vivid example.

5. **Probing** shows what information is encoded. Syntax, semantics, facts — all are extractable from representations.

6. **Activation patching** establishes causality. Not just correlation, but understanding which components actually influence the output.

7. **For safety this is critical**: detecting deception, verifying alignment, surgical correction — all require interpretability.

8. **We are at the beginning**: current methods work on small models and find individual circuits. Full understanding remains a distant goal.

---

## Practical Code Example

**Minimal Probing Implementation** checks what information is encoded in model hidden states by training a linear classifier on top of internal representations.

The POSProber class loads the model with the output_hidden_states=True flag to extract intermediate representations, creates a tokenizer, and initializes a list of linear probe layers — one for each model layer. Each probe maps hidden_dim to the number of classes (e.g., POS tags).

The hidden state extraction method tokenizes the text, runs it through the model without computing gradients, and returns a tuple of tensors for each layer with dimensions batch × sequence × hidden_dim.

Training a probe for a specific layer uses Adam optimizer and CrossEntropyLoss, iteratively training the linear classifier to predict labels from that layer's hidden states.

**Interpretation:** If a probe achieves high accuracy (>90%) on layer N, information about POS is clearly encoded linearly in that layer. Low accuracy (<60%) indicates the absence of information or nonlinear encoding. Comparing across layers reveals where syntactic understanding forms — typically early layers for syntax and deep layers for semantics.

Important: the presence of information does not guarantee the model uses it. Causal analysis requires activation patching.


## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → LLM Fundamentals
**Previous:** [[08_Scaling_Laws|Scaling Laws and Emergent Abilities]]
**Next:** [[10_Implementation_from_Scratch|Implementing Transformer from Scratch]]
