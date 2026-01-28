## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[01_Fine_Tuning_Basics|Fine-Tuning Basics]]
**Next:** [[03_Techniques_and_Evaluation|Techniques and Evaluation]]

---

# Data Preparation for Fine-Tuning

## Importance of Data Quality

The model learns to reproduce patterns from examples. If the examples contain errors, the model will learn to make errors. If the examples are inconsistent, the model will be inconsistent.

Data preparation is typically the most labor-intensive stage of fine-tuning. From experience, 70-80% of time is spent working with data: collection, cleaning, labeling, validation.

## Data Sources

**Historical logs** are the gold standard. Chat logs, support tickets, customer correspondence. Advantage: authenticity and diversity. Limitation: the quality of original responses requires thorough filtering.

**Expert labeling** provides controlled quality. You create queries and engage experts to write reference responses. Advantage: quality control. Limitation: cost and speed.

**Synthetic generation** uses a strong model (GPT-4) to create examples. Advantage: scalability and speed. Limitation: risk of inheriting issues from the generating model.

**Knowledge distillation** involves running real queries through a powerful model and using its responses to train a smaller model.

**Combined approach** is usually optimal: filtered historical data + expert labeling + synthetic generation.

## Data Formats

**Chat format** is the standard for modern models. A sequence of messages with roles: system, user, assistant.

**Instruction format** consists of triplets: instruction, input, output.

**Completion format** is the simplest: prompt-completion pairs without role structuring.

**JSONL** is a technical storage format where each line is a separate JSON object containing one example.

## Structure of a Quality Example

**System prompt** sets the context and expectations. It must be consistent across all examples.

**User query** must be realistic. Real users make typos, phrase things vaguely, and omit context.

**Assistant response** demonstrates reference behavior. It must fully comply with format, style, and structure requirements.

**Example length** matters. Balance between informativeness and training cost.

## Data Quality Criteria

**Correctness** — responses are factually accurate and match the questions.

**Completeness** — responses contain all necessary information.

**Consistency** — identical questions receive aligned responses.

**Formatting** — all examples follow the same format.

**Tone and style** — language matches the desired tone of voice.

**Relevance** — examples reflect real usage scenarios.

## Cleaning and Filtering

**Duplicate removal** — exact matching and similarity-based deduplication.

**Quality filtering** — empty responses, overly short responses, obvious errors.

**Format normalization** — standardizing to a uniform format (HTML, markdown, plain text).

**Structure validation** — verifying compliance with the expected schema.

**Personal data removal** — PII must not end up in training data.

## Data Augmentation

**Paraphrasing** creates variations of existing examples while preserving meaning.

**Adding variability** — typos, informal language, abbreviations (if typical for the audience).

**Creating edge cases** fills gaps in coverage.

**Multilingual support** can be achieved through translation. Translation quality is critical.

Augmented data must undergo the same quality validation as original data.

## Splitting into Train/Validation/Test

**Training set** — data for model training (80-90%).

**Validation set** — monitoring training and tuning hyperparameters (5-10%).

**Test set** — held-out data for final evaluation (5-10%). Used only once.

**Random splitting** may be insufficient when structure is present (sessions, users).

**Stratified splitting** preserves category distribution.

## Handling Imbalance

**Understanding imbalance** — analyzing distribution across categories, topics, and query types.

**Upsampling rare categories** — duplication or augmentation. Risk of overfitting.

**Downsampling frequent categories** — reducing dominance. Loss of information.

**Targeted data collection** — the best approach for balancing.

## Dataset Documentation

**Source description** — where the data came from, when and how it was collected.

**Cleaning process documentation** — what transformations were applied and why.

**Dataset statistics** — number of examples, category distribution, length statistics.

**Known issues** — honest description of data limitations.

## Key Takeaways

Data quality determines the ceiling of model quality.

Data preparation accounts for 70-80% of fine-tuning work.

Combine sources: historical logs, expert labeling, synthetic generation.

Validate at every stage.

Document everything.

A held-out test set is mandatory.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Fine-Tuning
**Previous:** [[01_Fine_Tuning_Basics|Fine-Tuning Basics]]
**Next:** [[03_Techniques_and_Evaluation|Techniques and Evaluation]]
