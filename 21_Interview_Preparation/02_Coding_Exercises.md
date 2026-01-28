# Coding Exercises for Interviews

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Interview Preparation
**Previous:** [[01_ML_System_Design|ML System Design]]
**Next:** [[03_Papers_Reading_Guide|Papers Reading Guide]]

---

## Introduction

ML/AI interviews often require implementing basic algorithms from scratch — without using PyTorch, TensorFlow, or sklearn. This tests depth of understanding, not the ability to call APIs.

Typical requirements: NumPy only (sometimes pure Python), thirty to forty-five minutes for implementation, explanation of each step, discussion of complexity and trade-offs.

---

## 1. Self-Attention (Single Head)

### Problem

Implement the self-attention mechanism for a sequence.

### Conceptual Solution

The self-attention mechanism consists of four key steps:

Step 1 Linear projections: the input sequence X (of dimension sequence times d_model) is projected through three weight matrices W_q, W_k, and W_v to create Query, Key, and Value matrices. Each of these matrices has dimension sequence times d_k. This allows the model to learn different representations of the input data for computing attention.

Step 2 Computing attention scores: attention scores are computed via the matrix product Q times transposed K. It is critically important to scale the result by dividing by the square root of d_k. This scaling prevents the growth of dot product values with increasing dimensionality, which could make softmax too sharp and lead to vanishing gradients.

Step 3 Applying softmax: the softmax function is applied to the scores along the last axis, normalizing the attention scores so that the sum of weights for each token equals one. For numerical stability, the trick of subtracting the maximum value before exponentiation is used: exp(x minus max(x)) instead of exp(x).

Step 4 Weighted summation: the final output is obtained by multiplying the attention weight matrix by the value matrix V. This creates a contextualized representation where each token is enriched with information from other tokens according to the attention weights.

### Key Discussion Points

Why divide by the square root of d_k: without this, dot products grow with dimensionality proportionally to d_k, softmax concentrates at extreme values (zero or one), gradients in flat regions of softmax become very small, scaling preserves the variance of scores regardless of dimensionality.

Complexity: time O(n squared times d) where n equals seq_len dominated by the Q times transposed K operation, space O(n squared) for the attention matrix — the main bottleneck for long sequences. It is precisely the quadratic complexity that motivates alternatives such as Linear Attention.

Numerical stability of softmax: subtract max to prevent overflow when exponentiating large numbers, mathematically equivalent since softmax is invariant to shifting by a constant, also prevents underflow when computing exp of very negative numbers.

---

## 2. Multi-Head Attention

### Problem

Extend single-head attention to multi-head.

### Conceptual Solution

Multi-head attention extends the single-head mechanism, allowing the model to simultaneously attend to different aspects of the input data.

Architecture: the input sequence is first projected through matrices W_q, W_k, W_v (each of dimension d_model times d_model). Then the resulting Q, K, V matrices are split into num_heads parts. If d_model equals 512 and num_heads equals 8, then each head operates with dimension d_k equal to 64.

Processing pipeline: projection through W_q, W_k, W_v creates matrices of size sequence times d_model. Reshape transforms them into sequence times num_heads times d_k, then transpose into num_heads times sequence times d_k. For each head, attention is computed independently: scores equal (Q times transposed K) divided by the square root of d_k, then softmax and multiplication by V. Outputs of all heads are concatenated: num_heads times sequence times d_k is transformed into sequence times d_model. A final projection through W_o transforms the concatenated result.

Intuition behind multiple heads: different heads specialize in different patterns. One head may focus on syntactic dependencies, another on semantic relationships, a third on positional patterns. This is analogous to using multiple filters in CNN.

### Key Points

Why multiple heads: each head learns to attend to different aspects of the sequence, increases model expressiveness without proportional growth in computational complexity, provides an ensemble effect within a single layer, allows the model to process different types of dependencies in parallel.

Parameters: typically d_k equals d_model divided by num_heads to preserve overall complexity, total number of parameters is four times d_model squared (W_q, W_k, W_v, W_o), computational complexity is the same as single-head with d_k equal to d_model, typical values are num_heads equals 8 or 16 for large models.

---

## 3. LSTM Cell

### Problem

Implement a single step of an LSTM cell.

### Conceptual Solution

LSTM (Long Short-Term Memory) solves the vanishing gradient problem in standard RNNs through a gating mechanism and a separate cell state.

Architecture and components: LSTM has two internal states — cell state (c_t) and hidden state (h_t). Cell state is a "pipeline" of information that flows through the entire sequence with minimal changes.

Four key elements:

Forget gate (f_t): decides what information from the previous cell state to forget. Computed as sigmoid(W_f times concatenation of x_t and h_prev plus b_f). Output is in the range zero to one, where zero means completely forget, one means completely retain. Important: the bias is initialized to one so that the model remembers by default rather than forgets.

Input gate (i_t): determines what new information will be added to the cell state. Also a sigmoid activation controlling how much the candidate values influence the new state.

Candidate cell state (c_candidate): new potential values for adding to the cell state, computed through tanh activation. Uses tanh (range minus one to plus one) instead of sigmoid to allow both positive and negative updates.

Output gate (o_t): controls what part of the cell state enters the hidden state. Sigmoid activation filters tanh(c_t) before output.

State updates: new cell state c_t equals f_t times c_prev plus i_t times c_candidate (additive update!). New hidden state h_t equals o_t times tanh(c_t).

Input concatenation: all gates receive as input the concatenation of the current input x_t and the previous hidden state h_prev with dimensionality input_size plus hidden_size. This allows gates to make decisions based on both the current context and accumulated information.

### Key Points

Forget gate bias equals one: initializing b_f equal to one shifts sigmoid toward larger values (closer to one), the model remembers by default rather than forgets — this is critical for training, without this trick training stalls for a long time with vanishing gradients, empirically proven to improve convergence in early stages.

Why LSTM solves vanishing gradient: cell state c_t is updated additively (new equals old times f plus candidate times i), the gradient can flow directly through the addition operation without repeated multiplications, in standard RNNs the gradient passes through a chain of tanh multiplications causing exponential decay, LSTM creates a highway for the gradient through cell state.

Comparison with GRU: GRU has two gates (reset, update) instead of three, no separate cell state. LSTM has four parameter matrices (W_f, W_i, W_c, W_o) versus three for GRU, more parameters. GRU often trains faster and requires less memory. LSTM typically shows better results on very long sequences (100+ steps). In practice, the quality difference is often negligible, and GRU is preferred for its simplicity.

---

## 4. K-Means Clustering

### Problem

Implement K-Means from scratch.

### Conceptual Solution

K-Means is an iterative clustering algorithm that partitions n data points into k clusters by minimizing within-cluster variance.

The algorithm consists of two alternating steps:

Step 1 Assignment: for each data point, distances to all k centroids are computed (typically Euclidean distance). The point is assigned to the cluster with the nearest centroid. An efficient implementation uses NumPy broadcasting: distances equals norm(X with added axis minus centroids along axis two), creating an n_samples times k matrix in a single operation. Then labels equals argmin(distances along axis one) finds the index of the minimum distance for each point.

Step 2 Update: centroids are updated as the mean of all points assigned to a given cluster. For cluster i: new_centroid_i equals mean(X[labels equals i]). An important edge case: if a cluster remains empty (no assigned points), the old centroid is preserved to avoid NaN.

Initialization (K-means++): instead of randomly selecting k points, K-means++ uses a smart strategy. The first centroid is chosen randomly from the data. For each subsequent centroid: compute for each point the distance to the nearest already-chosen centroid (squared). Select the next centroid with probability proportional to this squared distance. Points far from existing centroids have a higher chance of being selected. This ensures a good initial distribution of centroids.

Convergence criterion: the algorithm stops when centroids stop changing significantly (if norm(new_centroids minus old_centroids) is less than tolerance) or the maximum number of iterations is reached.

### Key Points

K-means++ initialization: selection proportional to squared distance (D squared) avoids local minima, theoretically guarantees O(log k)-approximation of the optimal solution, practically yields significantly faster convergence and more stable results, without K-means++ the result strongly depends on random initialization.

Complexity: time O(n times k times d times iterations) where n equals points, k equals clusters, d equals dimensionality. Typically iterations is much less than 100, in practice often 10-20. Space O(n times k) for the distance matrix (main memory). Can be optimized to O(n plus k) by computing distances on the fly.

Limitations: sensitive to outliers (a single outlier can shift an entire centroid), requires knowing k in advance (Elbow method or Silhouette score is used for selection), finds only convex/spherical clusters (does not work for complex shapes), result depends on initialization (often run multiple times), assumes clusters of approximately equal size.

---

## 5. Gradient Descent with Backpropagation

### Problem

Implement a training loop for a simple neural network.

### Conceptual Solution

Training a neural network consists of three key phases: forward pass (computing predictions), backward pass (computing gradients), and parameter update (updating weights).

Forward Pass: for a two-layer network with ReLU, the input data X is multiplied by weights W1 and bias b1 is added to obtain z1. Then ReLU activation is applied: a1 equals max(zero, z1). The hidden representation a1 is projected through W2 and b2 to obtain logits z2. Finally, softmax is applied to obtain class probabilities. Critical: all intermediate values (X, z1, a1, z2) are saved for use in the backward pass.

Backward Pass (Backpropagation): gradient computation proceeds in reverse order through the chain rule. For the output layer with the softmax plus cross-entropy combination, the gradient simplifies to an elegant form: dz2 equals y_pred minus y_true. This is not a coincidence but a mathematical property of this combination of functions. Then the weight gradients are computed: dW2 equals transposed a1 times dz2 and db2 equals mean(dz2). For the hidden layer: the gradient is first passed through the transposed weights (da1 equals dz2 times transposed W2), then multiplied by the ReLU derivative (dz1 equals da1 times (z1 greater than zero)), which equals one for positive z1 and zero for negative. Finally dW1 equals transposed X times dz1 and db1 equals mean(dz1).

Weight Initialization (Xavier/He): weights are initialized as W equals randn(shape) times sqrt(2.0 divided by input_size). The multiplier sqrt(2.0 divided by input_size) is critical for training stability. Without it, the variance of activations grows or shrinks exponentially with network depth, causing vanishing or exploding gradients. Xavier uses sqrt(2 divided by (input_size plus output_size)), He uses sqrt(2 divided by input_size) and works better with ReLU.

Training Loop: mini-batch gradient descent shuffles the data each epoch and processes it in batches. For each batch: a forward pass is performed, the loss is computed, a backward pass is performed to obtain gradients, and weights are updated as W equals W minus learning_rate times dW. Batches provide a balance between the stochasticity of SGD (better generalization) and the stability of full-batch GD.

### Key Points

Why Xavier/He initialization: preserves the variance of activations approximately constant across layers, without this each layer either compresses (vanishing) or expands (exploding) the signal, Xavier for tanh/sigmoid, He for ReLU (accounts for ReLU killing half of neurons), enables training deep networks without batch normalization.

ReLU gradient: derivative is one if x greater than zero, else zero (technically undefined at zero, zero is used). Dead ReLU problem: if a neuron always receives negative inputs, the gradient is always zero. Solutions: Leaky ReLU (0.01 times x instead of zero), proper initialization, not too large a learning rate. Advantages: computationally efficient, does not saturate for positive values.

Softmax plus Cross-entropy gradient: mathematically d(CrossEntropy)/dz equals softmax(z) minus y_true — an incredibly simple formula. Why: the derivatives of the exponentials in softmax and the logarithms in cross-entropy cancel each other out. This property makes training classifiers efficient and numerically stable. It is important to use stabilized softmax (subtracting max) to avoid numerical issues.

---

## 6. BPE Tokenizer

### Problem

Implement BPE tokenizer training.

### Conceptual Solution

Byte Pair Encoding (BPE) is a data compression algorithm adapted for tokenization. It iteratively merges the most frequent pairs of characters/subwords into new tokens.

Dictionary initialization: the text corpus is first split into words, each word is represented as a sequence of individual characters with an end-of-word marker appended. For example, "low" becomes "l o w end_of_word". The dictionary contains the frequency of each word in the corpus. The end_of_word marker is critically important as it allows distinguishing "low" as a standalone word from "low" as the beginning of the word "lower".

Iterative training process: at each iteration the algorithm counts the frequency of all adjacent character pairs across all words weighted by word frequency, finds the most frequent pair (for example ("l", "o") occurs ten times), merges that pair in all occurrences ("l o w" becomes "lo w"), and saves this merge operation to an ordered list. The process is repeated a specified number of times (typically ten to fifty thousand merge operations).

Tokenization with the trained model: to tokenize a new word, the saved merges are applied in the same order in which they were learned. The word is split into characters, then merge operations are applied iteratively. The order is critical for determinism: different orders can produce different tokenizations.

Example on the corpus "low lower lowest": Merge 1 ("l", "o") produces "lo" (occurs in low, lower, lowest), Merge 2 ("lo", "w") produces "low" (occurs in all three), Merge 3 ("low", "e") produces "lowe" (only in lower, lowest). Result: the frequent word "low" becomes a single token, the rare word "lowest" is split into "low" plus "est" or a similar combination.

### Key Points

Why BPE works: frequent words and subwords are compressed into single tokens (efficiency), rare words are split into known subtokens (coverage without a huge dictionary), balance between vocabulary size (memory, embedding matrix) and sequence length (compute), can represent any text through sufficient decomposition into characters (no UNK tokens), morphologically similar words receive similar tokenizations ("play", "playing", "player").

Order of merges matters: merges are applied in the exact training order, not by frequency during tokenization, this guarantees determinism (the same word is always tokenized identically), earlier merges take priority which reflects global frequency in the corpus, greedy algorithm (locally optimal decisions at each step).

End-of-word marker: distinguishes "low" as a standalone word versus "low" as the beginning of "lower", without the marker the words "low " and "low" (as a prefix) are indistinguishable, allows the model to learn different representations for prefixes versus complete words, alternative is byte-level BPE (GPT-2+) without explicit markers (spaces are part of tokens).

---

## 7. Cosine Similarity Search

### Problem

Implement nearest neighbor search by cosine similarity.

### Conceptual Solution

Cosine similarity measures the angle between two vectors, ignoring their magnitude. This makes the metric ideal for comparing semantic embeddings where direction matters more than length.

Mathematics: cosine similarity is defined as sim(a, b) equals (a dot b) divided by (norm a times norm b), where a dot b is the dot product, norm a is the Euclidean norm (length) of the vector. The value lies in the range minus one to plus one, where one means identical direction, zero means orthogonality, minus one means opposite direction. For embedding vectors, all values are typically positive (zero to one).

Efficient batch implementation: the naive approach computes similarity for each query-document pair in a loop, O(n times d) for n documents of dimensionality d. The efficient approach uses matrix operations: normalize the query (query_norm equals query divided by norm of query), normalize all documents in a single operation (docs_norm equals documents divided by norm of documents along axis one), compute all similarities with a single matrix multiplication (similarities equals docs_norm times query_norm). This yields the same result O(n times d) but uses optimized BLAS operations and vectorization.

Top-K search: after obtaining the similarities array, np.argsort is used to obtain indices in descending order of similarity. argsort returns indices in ascending order, so reversal is applied, and the first k are selected for the top-k. Complexity is O(n log n) for sorting, but for small k, np.argpartition with O(n) can be used.

Numerical stability: a small epsilon (1e-8) is added to the denominator during normalization to avoid division by zero for zero vectors. This is especially important if embeddings may contain very small values or be zero in some dimensions.

### Key Points

Normalize once, dot product many times: pre-normalize documents once at indexing time, store normalized versions, at search time normalize only the query and compute the dot product, cosine similarity of normalized vectors is mathematically equivalent to the dot product, saves computation when one set of documents is used for many queries.

Numerical stability: add eps (1e-8) to the denominator to prevent division by zero, important for vectors with very small norm (close to zero), alternative is to check the norm before division and handle the edge case separately, for large systems use float32 instead of float64 to save memory without losing precision.

Scalability: for millions of documents, brute-force search is inefficient at O(n times d) per query, solutions include FAISS, Annoy, HNSW approximate nearest neighbor (ANN) algorithms, ANN provides approximately 99% accuracy with 10-100x speedup through indexing, trade-off of accuracy versus speed versus memory.

---

## Interview Tips

### What Interviewers Expect

Clean code — clear variable names, comments. Explain as you go — think out loud. Handle edge cases — empty input, division by zero. Know complexity — Time and Space. Trade-offs — can be made faster but requires more memory.

### Common Mistakes

Forgetting numerical stability — softmax overflow, log(zero). Incorrect dimensions — especially in matrix operations. Off-by-one errors — in loops, indices. Not testing — at least one simple example.

### Preparation

Practice on paper — interviews are often on a whiteboard. Know formulas by heart — attention, LSTM gates. Understand why — not just how but why it works. Prepare questions — what batch size to use.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Interview Preparation
**Previous:** [[01_ML_System_Design|ML System Design]]
**Next:** [[03_Papers_Reading_Guide|Papers Reading Guide]]

---

## Practical Code Examples

### 1. Self-Attention Mechanism

```python
import numpy as np

def self_attention(X, d_k):
    """
    Single-head self-attention implementation

    Args:
        X: input sequence (seq_len, d_model)
        d_k: dimensionality for Q, K, V projections
    Returns:
        output: contextualized representation (seq_len, d_k)
    """
    seq_len, d_model = X.shape

    # Initialize weight matrices (learnable parameters in practice)
    W_q = np.random.randn(d_model, d_k) / np.sqrt(d_model)
    W_k = np.random.randn(d_model, d_k) / np.sqrt(d_model)
    W_v = np.random.randn(d_model, d_k) / np.sqrt(d_model)

    # Step 1: Linear projections into Q, K, V
    Q = X @ W_q  # (seq_len, d_k)
    K = X @ W_k  # (seq_len, d_k)
    V = X @ W_v  # (seq_len, d_k)

    # Step 2: Compute attention scores with scaling
    scores = Q @ K.T / np.sqrt(d_k)  # (seq_len, seq_len)

    # Step 3: Apply softmax to obtain attention weights
    # Subtract max for numerical stability
    scores_shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_shifted)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Step 4: Weighted summation of values
    output = attention_weights @ V  # (seq_len, d_k)

    return output, attention_weights

# Usage example
X = np.random.randn(5, 8)  # 5 tokens, dimensionality 8
output, weights = self_attention(X, d_k=4)
print(f"Output shape: {output.shape}")  # (5, 4)
print(f"Attention weights:\n{weights}")
```

### 2. BPE Tokenizer Basic Implementation

```python
from collections import defaultdict, Counter

def get_pairs(word):
    """Get all adjacent character pairs in a word"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def train_bpe(corpus, num_merges):
    """
    Train a BPE tokenizer

    Args:
        corpus: list of words for training
        num_merges: number of merge operations
    Returns:
        merges: list of pairs to merge in execution order
    """
    # Initialization: each word as a sequence of characters
    vocab = defaultdict(int)
    for word in corpus:
        # Add end-of-word marker
        vocab[' '.join(list(word) + ['</w>'])] += 1

    merges = []

    for i in range(num_merges):
        # Count frequency of all character pairs
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for pair in get_pairs(symbols):
                pairs[pair] += freq

        if not pairs:
            break

        # Find the most frequent pair
        best_pair = max(pairs, key=pairs.get)
        merges.append(best_pair)

        # Merge this pair in all words
        vocab = merge_vocab(best_pair, vocab)

    return merges

def merge_vocab(pair, vocab):
    """Merge a character pair in all words"""
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)

    for word in vocab:
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = vocab[word]

    return new_vocab

# Usage example
corpus = ['low', 'low', 'low', 'lower', 'lower', 'lowest']
merges = train_bpe(corpus, num_merges=5)
print("Learned merges:", merges[:3])
```

### 3. Simple Gradient Descent

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, epochs=100):
    """
    Gradient descent for linear regression

    Args:
        X: features (n_samples, n_features)
        y: target values (n_samples,)
        learning_rate: learning rate
        epochs: number of epochs
    Returns:
        weights: learned weights (n_features,)
        bias: learned bias (scalar)
        losses: loss history
    """
    n_samples, n_features = X.shape

    # Initialize parameters
    weights = np.zeros(n_features)
    bias = 0
    losses = []

    for epoch in range(epochs):
        # Forward pass: compute predictions
        y_pred = X @ weights + bias

        # Compute loss (MSE)
        loss = np.mean((y_pred - y) ** 2)
        losses.append(loss)

        # Backward pass: compute gradients
        error = y_pred - y
        dw = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * np.sum(error)

        # Update parameters
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias, losses

# Usage example
np.random.seed(42)
X = np.random.randn(100, 3)  # 100 samples, 3 features
true_weights = np.array([2.5, -1.3, 0.8])
y = X @ true_weights + np.random.randn(100) * 0.1  # small noise

weights, bias, losses = gradient_descent(X, y, learning_rate=0.1, epochs=200)
print(f"True weights: {true_weights}")
print(f"Found weights: {weights}")
print(f"Final loss: {losses[-1]:.4f}")
```
