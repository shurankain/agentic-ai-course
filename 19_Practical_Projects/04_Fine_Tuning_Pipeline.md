# Project: End-to-End Fine-Tuning Pipeline

## Project Overview

A complete pipeline for fine-tuning a model: from data collection to deployment. A specialized model for customer support that understands product specifics and responds in the appropriate tone.

### Goals

Business: understanding product terminology, responding in brand voice, knowledge of typical problems and solutions, escalating complex cases to humans.

Technical: reproducible data pipeline, quality-focused curation, LoRA fine-tuning, evaluation suite, production deployment.

## Pipeline Architecture

Five sequential phases:

**DATA COLLECTION**: Support Tickets (real dialogues), Knowledge Base (documentation, FAQ), Synthetic Data (LLM-generated examples).

**DATA PROCESSING**: Formatting & Cleaning (normalization, noise removal), Quality Filtering (scoring and filtering by quality score), Dataset Splitting (train/val/test with stratification).

**TRAINING**: Base Model (Mistral/Llama/Qwen), LoRA Config (adaptation parameters), Training Loop with monitoring.

**EVALUATION**: Metrics (perplexity, ROUGE, BERTScore), Human Eval (expert assessment), A/B Testing (comparison with baseline).

**DEPLOYMENT**: Model Registry (versioning), Serving Infra (vLLM/TGI), Monitoring & Alerts (latency, throughput, quality drift).

## Data Collection

### Sources

Historical Support Tickets, authentic language, real problems. Challenges: noise, PII, inconsistent quality.

Knowledge Base Articles: accurate information, structured. Challenges: formal tone, may not match conversation style.

Synthetic Data: controllable, scalable. Challenges: potential model collapse, lack of authenticity.

### Quality Framework

Level 1: Basic Filtering: minimum length >50 chars, language detection, deduplication (exact + near-duplicate), PII removal.

Level 2: Quality Scoring: resolution success, customer satisfaction (CSAT), response coherence (LLM-based), factual accuracy (verified against knowledge base).

Level 3: Diversity Sampling: topic distribution (all product areas), difficulty distribution (easy, then complex), sentiment distribution (positive, then negative).

### Data Format

Standard conversation format with roles: system (instructions), user (customer), assistant (support). Each example is a complete dialogue in JSON with a message array. Mandatory inclusion of system prompt in every example.

## Data Processing

### Formatting

Convert to unified conversation format, normalize whitespace and special characters, standardize entity mentions (product names, URLs), remove internal notes.

### PII Handling

Detection: regex patterns + NER model for names, emails, phone numbers, addresses.

Strategy: Replacement with placeholders ([EMAIL], [NAME]) to balance privacy and utility.

### Quality Filtering

100% raw data, then basic filters, then 70%, then quality scoring, then 40%, then diversity sampling, then 25% final dataset. Aggressive filtering produces better models: fewer high-quality examples are better than many noisy ones.

### Split

Training 80% (gradient updates), Validation 10% (hyperparameter tuning, early stopping), Test 10% (final evaluation, never seen during training). Stratified split by topics.

## Fine-Tuning Configuration

### Base Model

Pick on four criteria rather than on a name, because the specific models turn over every few months while the criteria do not: parameter count you can afford to train and then serve, licence terms that permit your use, quality on your language and domain, and whether the ecosystem has tested LoRA against that architecture. The last one is underrated and costs the most time when it fails.

**Around 7B to 8B** is the workhorse tier for a first fine-tune. It trains on a single 24 GB card with QLoRA, serves cheaply, and is where the tooling is most reliable. Llama 3.1 8B Instruct and Qwen 7B Instruct are the long-standing reference points here, and Mistral's 7B line is the usual choice when European languages matter.

**Around 70B** buys quality at the cost of multi-GPU training or aggressive quantization, and it changes the serving bill more than the training bill. Worth it when the 8B tier demonstrably fails your evaluation rather than as a starting point.

**Around 2B to 3B** is for edge and mobile, where the model has to fit on the device. It fine-tunes fast and gives up real capability, so validate against your task before committing to the form factor.

Check the current open-weight field in Module 01, LLM Providers and APIs before choosing, since that lesson tracks which releases lead and, more importantly, which of them carry custom licences rather than MIT or Apache 2.0. A model that cannot be used commercially is not a candidate no matter how it benchmarks.

### LoRA Configuration

Rank r: 8 (minimal adaptation, fast), 16 (optimal default), 32+ (maximum adaptation, risk of overfitting). lora_alpha typically 32, dropout 0.05 (regularization). Target modules: q_proj, k_proj, v_proj, o_proj (attention), gate_proj, up_proj, down_proj (feed-forward).

### Training Hyperparameters

Batch size 4 per device, gradient accumulation 4 (effective batch 16). Learning rate 2e-4 (optimal for LoRA). Epochs 3 (sufficient without overfitting). Warmup ratio 0.03 (gradual LR increase over the first 3% of steps). Cosine scheduler (gradual LR decay). FP16/BF16 (speedup, memory savings). Checkpoint every 100 steps, evaluation every 100 steps.

### Monitoring

Track: Loss (steady decrease), Validation loss (increase = overfitting), Learning rate (schedule visualization), Gradient norm (spikes = instability). Early stopping if validation loss does not improve for N steps.

### Alignment: DPO and ORPO (2024-2025)

Beyond supervised fine-tuning, preference alignment improves response quality:

**DPO (Direct Preference Optimization):** Trains directly on preference pairs (chosen vs rejected responses) without a separate reward model. Simpler than RLHF, stable training, strong results. Pipeline: generate response pairs, then collect preferences (human or LLM-as-Judge), then train with DPO loss. Typical setup: SFT first, then DPO on 5K-20K preference pairs.

**ORPO (Odds Ratio Preference Optimization):** Combines SFT and preference alignment into a single training stage. No need for a separate SFT step: ORPO simultaneously teaches the model to generate good responses and avoid bad ones. Simpler pipeline, fewer hyperparameters, competitive results.

**Recommendation:** Start with SFT (LoRA) for the initial model. If quality needs improvement: add DPO on preference data. ORPO when you want a simpler pipeline with fewer training stages.

### Unsloth for Efficient Training

**Unsloth** (2024-2025) provides 2-5x faster LoRA fine-tuning with 60-80% less memory:

Custom CUDA kernels for attention, MLP, and LoRA operations. Automatic mixed-precision optimization. Memory-efficient gradient checkpointing. Supports Llama, Mistral, Qwen, Gemma model families.

**Practical impact:** Fine-tune Llama 3.1 8B on a single RTX 4090 (24GB) instead of requiring an A100. Training that took 4 hours now takes 1-2 hours. Drop-in replacement for HuggingFace Trainer: change the import, keep the same code.

**4-bit QLoRA via Unsloth:** Quantize the base model to 4-bit, train LoRA adapters in FP16. 70B model fine-tuning on a single 48GB GPU. Quality within 1-2% of full-precision LoRA.

## Evaluation Framework

### Automatic Metrics

Perplexity: model confidence in predictions. Lower is better; a baseline is needed for comparison.

ROUGE/BLEU: overlap with reference. Useful for consistency but does not capture semantic quality.

BERTScore: semantic similarity with reference. Better than ROUGE for open-ended generation.

### LLM-as-Judge

A strong model (GPT-5.6, Claude) evaluates on a 1-5 scale: helpfulness (does it solve the problem), accuracy (correctness), tone (friendly and professional), completeness (thoroughness of the solution). Evaluation with a brief justification. Enables assessment of semantic quality and brand voice.

### A/B Testing

Route 10% traffic to the new model. Compare: resolution rate, customer satisfaction, escalation rate, response time. Statistical significance testing. Gradual rollout if positive.

### Human Evaluation

Sample 100 responses, expert annotators, inter-annotator agreement, qualitative feedback.

## Deployment

### Model Registry

Version control: each version (customer-support-v1, v2) contains adapter_config.json (LoRA parameters), adapter_model.safetensors (weights), training_args.json (hyperparameters), evaluation_results.json (metrics), metadata.json (timestamp, git commit, description). Tracked for reproducibility and rollback.

### Serving Infrastructure

vLLM (high-performance, PagedAttention), TGI (HuggingFace solution), OpenAI-compatible API (drop-in replacement). For LoRA: serve base + adapter or merge before serving.

### Monitoring

Latency (P50, P95, P99), Throughput (requests/second), Error rate (failed generations), Quality drift (periodic LLM-as-Judge scoring), Cost (tokens per request, total spend). Alerts on anomalies.

## Key Takeaways

An end-to-end pipeline requires attention to every stage. Data quality matters more than quantity.

Modern base models (Llama 3.1, Qwen 2.5) provide stronger starting points: fine-tuning yields better results with less data than older models.

Multi-level filtering (basic, then quality, then diversity) produces better datasets than naive collection.

LoRA efficiently adapts large models on consumer hardware. Rank 16 is a good default. Unsloth provides 2-5x speedup and 60-80% memory reduction for LoRA training.

Alignment after SFT: DPO for preference-based improvement, ORPO for a simpler single-stage pipeline. Both improve response quality beyond SFT alone.

Evaluation is multi-faceted: automatic metrics (fast iteration), LLM-as-Judge (semantic quality), A/B testing (production validation), human evaluation (ground truth).

Iterative improvement: fine-tuning is not one-shot. Error analysis, then data improvement, then retrain, then evaluate, then repeat.

Production deployment requires a proper registry, serving infrastructure, and continuous monitoring to catch degradation.
