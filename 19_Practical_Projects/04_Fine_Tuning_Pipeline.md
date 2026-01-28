# Project: End-to-End Fine-Tuning Pipeline

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Practical Projects
**Previous:** [[03_MCP_Server|MCP Server]]
**Next:** [[05_Production_Agent|Production Agent]]

---

## Project Overview

A complete pipeline for fine-tuning a model — from data collection to deployment. A specialized model for customer support that understands product specifics and responds in the appropriate tone.

### Goals

Business: understanding product terminology, responding in brand voice, knowledge of typical problems and solutions, escalating complex cases to humans.

Technical: reproducible data pipeline, quality-focused curation, LoRA fine-tuning, comprehensive evaluation suite, production deployment.

## Pipeline Architecture

Five sequential phases:

**DATA COLLECTION** — Support Tickets (real dialogues), Knowledge Base (documentation, FAQ), Synthetic Data (LLM-generated examples).

**DATA PROCESSING** — Formatting & Cleaning (normalization, noise removal), Quality Filtering (scoring and filtering by quality score), Dataset Splitting (train/val/test with stratification).

**TRAINING** — Base Model (Mistral/Llama/Qwen), LoRA Config (adaptation parameters), Training Loop with monitoring.

**EVALUATION** — Metrics (perplexity, ROUGE, BERTScore), Human Eval (expert assessment), A/B Testing (comparison with baseline).

**DEPLOYMENT** — Model Registry (versioning), Serving Infra (vLLM/TGI), Monitoring & Alerts (latency, throughput, quality drift).

## Data Collection

### Sources

Historical Support Tickets — authentic language, real problems. Challenges: noise, PII, inconsistent quality.

Knowledge Base Articles — accurate information, structured. Challenges: formal tone, may not match conversation style.

Synthetic Data — controllable, scalable. Challenges: potential model collapse, lack of authenticity.

### Quality Framework

Level 1 — Basic Filtering: minimum length >50 chars, language detection, deduplication (exact + near-duplicate), PII removal.

Level 2 — Quality Scoring: resolution success, customer satisfaction (CSAT), response coherence (LLM-based), factual accuracy (verified against knowledge base).

Level 3 — Diversity Sampling: topic distribution (all product areas), difficulty distribution (easy → complex), sentiment distribution (positive → negative).

### Data Format

Standard conversation format with roles: system (instructions), user (customer), assistant (support). Each example is a complete dialogue in JSON with a message array. Mandatory inclusion of system prompt in every example.

## Data Processing

### Formatting

Convert to unified conversation format, normalize whitespace and special characters, standardize entity mentions (product names, URLs), remove internal notes.

### PII Handling

Detection: regex patterns + NER model for names, emails, phone numbers, addresses.

Strategy: Replacement with placeholders ([EMAIL], [NAME]) to balance privacy and utility.

### Quality Filtering

100% raw data → basic filters → 70% → quality scoring → 40% → diversity sampling → 25% final dataset. Aggressive filtering produces better models — fewer high-quality examples are better than many noisy ones.

### Split

Training 80% (gradient updates), Validation 10% (hyperparameter tuning, early stopping), Test 10% (final evaluation, never seen during training). Stratified split by topics.

## Fine-Tuning Configuration

### Base Model

Mistral 7B Instruct — balance of size/quality, strong instruction following. Alternatives: Llama 3 8B (newer, better reasoning), Qwen 2.5 7B (excellent multilingual).

### LoRA Configuration

Rank r: 8 (minimal adaptation, fast), 16 (optimal default), 32+ (maximum adaptation, risk of overfitting). lora_alpha typically 32, dropout 0.05 (regularization). Target modules: q_proj, k_proj, v_proj, o_proj (attention), gate_proj, up_proj, down_proj (feed-forward).

### Training Hyperparameters

Batch size 4 per device, gradient accumulation 4 (effective batch 16). Learning rate 2e-4 (optimal for LoRA). Epochs 3 (sufficient without overfitting). Warmup ratio 0.03 (gradual LR increase over the first 3% of steps). Cosine scheduler (gradual LR decay). FP16/BF16 (speedup, memory savings). Checkpoint every 100 steps, evaluation every 100 steps.

### Monitoring

Track: Loss (steady decrease), Validation loss (increase = overfitting), Learning rate (schedule visualization), Gradient norm (spikes = instability). Early stopping if validation loss does not improve for N steps.

## Evaluation Framework

### Automatic Metrics

Perplexity — model confidence in predictions. Lower is better; a baseline is needed for comparison.

ROUGE/BLEU — overlap with reference. Useful for consistency but does not capture semantic quality.

BERTScore — semantic similarity with reference. Better than ROUGE for open-ended generation.

### LLM-as-Judge

A strong model (GPT-4, Claude) evaluates on a 1-5 scale: helpfulness (does it solve the problem), accuracy (correctness), tone (friendly and professional), completeness (thoroughness of the solution). Evaluation with a brief justification. Enables assessment of semantic quality and brand voice.

### A/B Testing

Route 10% traffic to the new model. Compare: resolution rate, customer satisfaction, escalation rate, response time. Statistical significance testing. Gradual rollout if positive.

### Human Evaluation

Sample 100 responses, expert annotators, inter-annotator agreement, qualitative feedback.

## Deployment

### Model Registry

Version control: each version (customer-support-v1, v2) contains adapter_config.json (LoRA parameters), adapter_model.safetensors (weights), training_args.json (hyperparameters), evaluation_results.json (metrics), metadata.json (timestamp, git commit, description). Tracked for reproducibility and rollback.

### Serving Infrastructure

vLLM (high-performance, PagedAttention), TGI (HuggingFace solution), OpenAI-compatible API (drop-in replacement). For LoRA — serve base + adapter or merge before serving.

### Monitoring

Latency (P50, P95, P99), Throughput (requests/second), Error rate (failed generations), Quality drift (periodic LLM-as-Judge scoring), Cost (tokens per request, total spend). Alerts on anomalies.

## Key Takeaways

An end-to-end pipeline requires attention to every stage. Data quality matters more than quantity.

Multi-level filtering (basic → quality → diversity) produces better datasets than naive collection.

LoRA efficiently adapts large models on consumer hardware. Rank 16 is a good default.

Evaluation is multi-faceted: automatic metrics (fast iteration), LLM-as-Judge (semantic quality), A/B testing (production validation), human evaluation (ground truth).

Iterative improvement — fine-tuning is not one-shot. Error analysis → data improvement → retrain → evaluate → repeat.

Production deployment requires a proper registry, serving infrastructure, and continuous monitoring to catch degradation.

---

## Navigation
**Previous:** [[03_MCP_Server|MCP Server]]
**Next:** [[05_Production_Agent|Production Agent]]
