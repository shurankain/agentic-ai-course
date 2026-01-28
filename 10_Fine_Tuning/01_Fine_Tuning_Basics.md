# Fine-Tuning Basics

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[../09_Conversational_AI/03_Voice_and_Multimodality|Voice and Multimodality]]
**Next:** [[02_Data_Preparation|Data Preparation]]

---

## What Is Fine-Tuning

Fine-tuning is the process of additionally training a pre-trained language model on a specific dataset. Instead of training from scratch, it uses a model that already possesses general knowledge of language and the ability to follow instructions.

### Transfer Learning Theory

Low-level features (tokenization, syntax) are universal and transfer with almost no changes. Middle layers (semantics, concepts) partially adapt. Upper layers change significantly to match task specifics.

**Catastrophic Forgetting** is the main risk during fine-tuning. The model may "forget" general knowledge while concentrating on the new task. Mitigation: add replay data, early stopping, low learning rate with warmup.

During fine-tuning, model weights are adjusted based on examples of desired behavior. The model does not learn to "think" but rather channels its expertise in the required direction.

## Fine-Tuning's Place in the Methods Ecosystem

**Prompt engineering** is the simplest approach. Formulating instructions and context directly in the query, without modifying the model.

**Few-shot learning** extends the prompt with examples. The model picks up the pattern and applies it to the new query. Limited by context window size.

**RAG** loads relevant context from an external knowledge base. The model remains unchanged but receives up-to-date information.

**Fine-tuning** modifies the model itself. Appropriate when you need to change not what the model knows, but how it behaves: style, format, reasoning sequence, domain terminology.

**RAG + fine-tuning** provides maximum flexibility for complex enterprise applications.

## When Fine-Tuning Is Justified

**Specific output format** — a particular JSON structure, specific markdown template, formalized report that is difficult to describe via prompt.

**Corporate voice and style** — tone of voice: formal/informal, technical/accessible, restrained/emotional.

**Domain terminology** — legal language, medical terminology, internal industry jargon.

**Prompt reduction** — a long system prompt is "embedded" into the model, saving tokens.

**Improved reliability** — increased accuracy or consistency on specific tasks.

## When Fine-Tuning Is Not Suitable

**Adding new knowledge** — use RAG. Fine-tuning handles factual knowledge injection poorly.

**Frequently changing information** — data is updated regularly, fine-tuning cannot keep up.

**Rarely used features** — training costs are not justified by the scale of usage.

**Tasks without clear examples** — impossible to create a dataset with correct answers.

**Complex reasoning** — fine-tuning may degrade chain-of-thought reasoning.

## Data Requirements

**Quantity:** minimum 50-100 examples for an effect, optimally 500-1000, more than 10K rarely yields proportional improvement.

**Quality matters more than quantity:** 200 carefully prepared examples are better than 2000 of dubious quality.

**Diversity:** examples should cover the full spectrum of expected scenarios.

**Format consistency:** uniform structuring of all examples.

**Absence of contradictions:** internal consistency of the dataset is critical.

## Cost and Resources

**Direct training costs:** OpenAI charges approximately $0.008 per 1000 tokens for GPT-4o mini. Training on 1000 examples can cost $50-200.

**Indirect costs:** collecting, labeling, and validating the dataset can take weeks.

**Infrastructure costs:** LoRA on a 7B model requires a GPU with 16GB VRAM.

## Risks and Limitations

**Catastrophic forgetting** — loss of general capabilities when training on specific examples.

**Overfitting** — memorizing examples instead of extracting general patterns.

**Drift from the base model** — base model updates are not applied automatically.

**Amplification of biases** — biases in the data are amplified.

**Lock-in** — vendor lock-in.

## Key Takeaways

Fine-tuning is a tool for changing model behavior, not its knowledge.

The correct order: prompt engineering → few-shot → RAG → fine-tuning.

Data quality determines result quality.

A combination of methods is often optimal: RAG for knowledge, fine-tuning for style.

Evaluate ROI honestly: cost includes not only compute but also data preparation time.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[../09_Conversational_AI/03_Voice_and_Multimodality|Voice and Multimodality]]
**Next:** [[02_Data_Preparation|Data Preparation]]
