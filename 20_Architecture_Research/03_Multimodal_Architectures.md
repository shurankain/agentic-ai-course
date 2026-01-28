# Multimodal Architectures: When LLM Sees the World

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Architecture Research
**Previous:** [[02_State_Space_Models|State Space Models]]
**Next:** [[../21_Interview_Preparation/01_ML_System_Design|ML System Design]]

---

## Why Text Is Not Enough

Human understanding of the world is multimodal. We read text, see images, hear sounds — and all of this integrates into a unified picture. LLMs trained only on text exist in a "blind" world: they can reason about images but have never seen them.

GPT-4V, Claude 3, Gemini Pro — modern frontier models can "see." This opens revolutionary possibilities: understanding documents with charts, analyzing medical scans, robot navigation, creative design assistance.

But how do you teach a language model to see? Simply combining a vision encoder and an LLM does not work — a proper alignment architecture between modalities is needed. This chapter covers the architectural solutions that make multimodal AI possible.

## Vision Encoders: How to Extract Meaning from Pixels

### Vision Transformer (ViT)

ViT (2020) showed that Transformers work for images no worse than CNNs.

Patchification: An image H × W × C is split into patches of size P × P. The number of patches equals H times W divided by P². For a 224×224 image with 16×16 patches, this yields 196 patches.

Linear projection: Each patch (flattened into a vector P² × C) is projected into an embedding of dimension D. The embedding z_i equals matrix E times flatten of patch i plus positional embedding E_pos for index i.

Transformer: Standard transformer layers process the sequence of patches.

CLS token: A special [CLS] token at the beginning aggregates information for classification.

Advantages of ViT: Scales better than CNNs given sufficient data, global receptive field from the first layer, unified architecture with NLP.

### CLIP: Contrastive Language-Image Pretraining

CLIP (2021) from OpenAI is the foundation for multimodal AI. Key idea: train vision and text encoders on (image, text) pairs through contrastive learning.

Architecture: CLIP consists of two parallel encoders — a vision encoder (typically ViT) and a text encoder (typically Transformer). An image passes through the vision encoder and becomes an image embedding, text passes through the text encoder into a text embedding. Both embeddings have the same dimensionality (typically 512 or 768) and are normalized to unit length.

Contrastive loss: For each image in the batch, its "correct" text description should be closer than all other descriptions (negatives). Loss formula: the negative average over all i of the logarithm of the ratio of the exponential of similarity between I_i and T_i divided by temperature to the sum of exponentials of similarity between I_i and all T_j divided by temperature.

CLIP results: Zero-shot classification on ImageNet without fine-tuning, robust to distribution shift, universal representations for downstream tasks.

### SigLIP: Sigmoid Loss

SigLIP (2023) from Google simplifies CLIP by replacing softmax with sigmoid. Loss is computed as the negative average over all pairs i,j of the sum y_ij times log sigmoid of z_ij plus (1 minus y_ij) times log (1 minus sigmoid of z_ij), where y_ij equals 1 if the pair matches, otherwise 0.

Advantages: Does not require global batch normalization (simpler distributed training), scales better to very large batch sizes, slightly better quality all else being equal.

### EVA-CLIP and DINOv2

EVA-CLIP: Scaling CLIP to 18B parameters with an improved training recipe.

DINOv2: Self-supervised vision encoder without text pairs. Uses self-distillation: Teacher (momentum average) and student on different augmentations of the same image, contrastive loss between them. DINOv2 yields richer representations than CLIP for certain tasks (depth estimation, segmentation) without explicit language supervision.

## Contrastive Learning: Deeper into the Mechanics

Contrastive learning creates representations where semantically similar inputs are close in embedding space, semantically different inputs are far apart.

For vision-language: "A dog on the beach" should be close to an image of a dog on the beach and far from "A cat in the garden." This implicitly teaches object recognition (what is in the image), attribute binding (what color, what shape), scene understanding (where it takes place), action recognition (what it is doing).

### Temperature parameter

Temperature controls the "sharpness" of the distribution. Low temperature (0.01): the model is confident, small differences matter. High temperature (1.0): soft distribution, tolerance to inaccuracies.

Trade-off: Low temperature — better fine-grained distinctions, but harder optimization. High temperature — easier training, but less discriminative representations. Typical values: 0.07 for CLIP, 0.1 for SigLIP.

### Batch size importance

Contrastive learning requires large batch sizes. More negatives equals better gradient signal. CLIP was trained with batch 32768. Small batch equals few informative negatives.

Memory-efficient alternatives: MoCo (momentum queue for storing negatives), SwAV (clustering instead of explicit negatives), BYOL (only positives, but works worse for multimodal).

### Hard negatives

Not all negatives are equally useful. "A dog on the beach" vs "A cat in space" — easy to distinguish. "A dog on the beach" vs "A dog in the park" — harder, more informative.

Hard negative mining: A technique for selecting the most difficult negative examples for training. Instead of random negative pairs, those with the highest similarity to positive examples are selected. Pairwise similarity scores are computed between all images and texts in the batch, then top-k negative pairs with the highest similarity are selected for each positive example.

Advantage: the model learns to distinguish finer semantic differences, which improves representation quality. The model does not waste gradients on "obvious" negative examples, focusing instead on difficult cases.

Caution: overly hard negatives may be false negatives (genuinely similar pairs that happen to be unmatched in the data). This can harm training. Typically a combination is used: part random negatives plus part hard negatives.

## Cross-Modal Attention: How Modalities "Communicate"

The vision encoder and the LLM "speak different languages": Vision: 576 patch tokens × 1024 dimensions (ViT-L), LLM: 32000 vocabulary × 4096 dimensions (LLaMA). A "translation" mechanism between them is needed.

### Projection layers

The simplest approach is linear projection: z_LLM equals W times z_vision plus b. Advantages: minimal overhead, easy to train. Disadvantages: limited expressiveness, does not account for context.

### Cross-Attention

Cross-attention allows the LLM to "query" the needed visual information. CrossAttn of Q_text, K_vision, V_vision equals softmax of Q_text times K_vision transposed times V_vision. Q (Query) from text hidden states, K, V (Key, Value) from visual embeddings.

Advantages: Dynamic extraction of relevant visual information, text "asks" — the image "answers," attention weights show what LLM looks at.

Variants: Before every LLM layer, every N layers, at the beginning (resampling).

### Perceiver Resampler

Flamingo (2022) introduced the Perceiver Resampler for compressing visual information. It is an architectural block that takes a large number of visual tokens as input (e.g., 576 patches from high resolution) and compresses them to a fixed smaller number (e.g., 64 learned queries), while preserving the most important information.

Mechanics: Learnable query tokens (fixed count, e.g., 64), Cross-attention (queries attend to visual tokens), Self-attention (queries refine representations), Output (compressed representation).

Advantages: Fixed number of tokens regardless of resolution, less compute for the LLM, queries learn to extract what is relevant.

## Early vs Late Fusion Strategies

### Early Fusion

Modalities are combined at the input: visual and text tokens are concatenated together and fed into a single Transformer, which processes the entire sequence as a unified whole.

LLaVA approach: Visual tokens concatenated with text tokens, single LLM processes everything, visual tokens as "virtual prefix."

Advantages: Deep integration of modalities, all layers see both modalities, simple architecture. Disadvantages: Long sequences (many visual tokens), compute grows with image resolution.

### Late Fusion

Modalities are processed separately, combined at the end: the image passes through the vision encoder independently, text through the text encoder independently, and only at the final stage are their features combined through a special fusion layer to produce the output.

Advantages: Pretrained encoders can be used as-is, less compute (each encoder is independent), modular — components can be swapped. Disadvantages: Less deep integration, may miss cross-modal patterns.

### Middle Fusion

A hybrid: partial processing separately, then combination. The image passes through several early vision layers, text through several early LLM layers, then their representations are combined and processed by joint layers to produce the final output. BLIP-2 uses "frozen" vision and language models with a trainable Q-Former between them, which serves as an adapter-bridge between modalities.

## Visual Instruction Tuning

CLIP provides good representations but cannot "converse" about images. Visual Instruction Tuning teaches the model to answer questions, follow instructions, and conduct dialogue.

Data for instruction tuning: (image, Q&A) pairs — questions about the image and detailed answers, Conversations about images — multi-turn dialogues where the model answers follow-up questions, Image captioning with detailed descriptions — not just "a dog on the beach," but a detailed description of all elements of the scene.

Dialogue format: Human asks a question about the image (with an image position marker), Assistant answers in detail. Follow-up questions then follow, creating a multi-turn dialogue. For example: "What is in the image?" — a description of a golden retriever on the beach — "What color is the frisbee?" — "bright red."

### LLaVA: Large Language-and-Vision Assistant

LLaVA (2023) is a simple but effective architecture.

Components: CLIP ViT-L/14 (frozen or fine-tuned), Linear projection layer, Vicuna LLM (fine-tuned).

Visual tokens as text: H_v equals W times Z_v, X equals concatenation [<image>, H_v, </image>, X_text].

Training: Stage 1 (Pre-training) — only the projection layer on image-caption pairs. Stage 2 (Fine-tuning) — full model on instruction data.

LLaVA 1.5 improvements: Higher resolution (336×336 or multiple scales), more training data, better base LLM.

### InstructBLIP

InstructBLIP (2023) extends BLIP-2. Q-Former: Learnable queries with instruction-aware extraction. Q_instr equals Concat of Q_learnable and Embed of instruction. Queries "know" what is being asked and extract what is relevant. Advantage: More efficient extraction, fewer visual tokens.

## Multimodal Tokenization

LLMs work with discrete tokens. Images are continuous. How to reconcile?

Approach 1: Treat visual embeddings as continuous "soft tokens." The projection layer makes visual embeddings compatible in dimension, the LLM processes them as regular tokens.

Approach 2: Discrete visual tokenization. VQ-VAE/VQGAN quantize images into discrete tokens, the image is converted into a sequence of discrete codebook indices, unified vocabulary (text plus visual codes).

### VQ-VAE for Visual Tokenization

VQ-VAE (Vector Quantized VAE) is an architecture for discrete image tokenization. Process: the image passes through an encoder (creates a continuous latent representation), then quantization converts each spatial location to the nearest discrete code from the codebook, and these discrete codes can be decoded back into an image through a decoder.

Codebook: K learned embedding vectors (e.g., K=8192), each spatial position mapped to the nearest codebook entry. For multimodal: the image is tokenized into a sequence of codebook indices, these indices are added to the LLM vocabulary, the LLM can generate images as well. DALL-E, Parti use this approach for image generation.

### Trade-offs

Continuous (soft tokens): Better preserves visual information, simpler training, does not allow the LLM to generate images.

Discrete (VQ tokens): Unified generation (text plus image), lossy compression (loss of detail), harder training (codebook collapse).

## Video Understanding Challenges

Naive approach: process each frame separately. Problems: Enormous number of tokens (30 fps × 60 sec = 1800 frames), temporal redundancy (adjacent frames are similar), long-range dependencies (an action starts here, ends there).

### Temporal Modeling

Frame sampling: The video is sampled into N frames (e.g., 8 or 16), each frame is processed by the vision encoder independently, then features are aggregated through pooling or attention to produce a single video representation. A simple approach, but it loses fine-grained temporal information about motion between frames.

3D convolutions: Extending 2D conv to 3D (H × W × T). Captures local motion patterns.

Temporal attention: Attention between frames — TemporalAttn of Q_t, K_t', V_t'. Global temporal context.

Space-time factorized attention: Spatial and temporal attention are separated into distinct operations. First, spatial attention is applied within each frame (patches within the frame attend to each other), then temporal attention between frames (corresponding positions of different frames attend to each other). This requires less compute than full spatio-temporal attention but preserves modeling of both dimensions.

### Video LLMs

VideoChat, Video-LLaVA, LLaMA-VID:

Common pattern: Extract features for sampled frames, temporal aggregation (pooling, attention), feed to LLM as video tokens.

Challenges: Memory for long videos, temporal grounding (where in the video is the answer to the question), long-form video understanding (movies, lectures).

### Temporal grounding

"At what moment does the dog catch the ball?" — requires understanding the question, localizing in time, answering with a timestamp. This is harder than image Q&A — reasoning about time is needed.

## Connections to Other Course Topics

Scaling (section 01): Multimodal models show emergence at large scale. GPT-4V capabilities appeared at sufficient size.

RLHF (section 10): Multimodal RLHF is harder — visual understanding needs to be evaluated, not just text quality. Human feedback for image descriptions is more subjective.

Inference (section 17): Visual tokens increase context length. KV cache for 576+ visual tokens is substantial. Strategies: visual token compression, caching.

Distributed Training (section 16): Multimodal training requires larger batch sizes (contrastive learning), cross-modal synchronization, different parallelism for vision and language.

## Key Takeaways

Vision encoders are the foundation. ViT, CLIP, SigLIP provide visual understanding. Contrastive pretraining creates an aligned vision-language space.

Contrastive learning requires scale. Large batch sizes, many negatives, and the right temperature are critical for quality.

Cross-modal alignment is non-trivial. Projection layers, cross-attention, Perceiver — different trade-offs between complexity and expressiveness.

Early fusion is deeper, late fusion is simpler. LLaVA-style concatenation provides deep integration. BLIP-2 style frozen encoders plus adapter is easier to train.

Instruction tuning transforms capabilities. Going from CLIP embeddings to a full-fledged vision-language assistant requires instruction fine-tuning.

Discrete vs continuous tokens. Continuous is simpler for understanding, discrete is necessary for unified generation.

Video is exponentially harder. Temporal dimension, memory requirements, long-range reasoning — open problems.

Resolution matters. High resolution equals more detail but more tokens. Trade-off quality vs compute.

Modular architectures dominate. Frozen pretrained encoders plus trainable adapters are more efficient than end-to-end training.

This is a rapidly evolving field. GPT-4o, Gemini 1.5, Claude 3 — new capabilities emerge every few months.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Architecture Research
**Previous:** [[02_State_Space_Models|State Space Models]]
**Next:** [[../21_Interview_Preparation/01_ML_System_Design|ML System Design]]
