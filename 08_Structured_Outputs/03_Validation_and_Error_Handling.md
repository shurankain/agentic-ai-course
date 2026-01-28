# Validation and Error Handling

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Structured Output
**Previous:** [[02_Data_Extraction|Data Extraction from Unstructured Text]]
**Next:** [[../09_Conversational_AI/01_Dialog_Management|Dialog Management]]

---

## Introduction: Why the Model Is Not Always Right

There is a temptation to treat language model output as truth. The model is trained on trillions of tokens, knows more than any individual person, and its answers sound confident and authoritative. But this temptation is dangerous.

Language models hallucinate. They make mistakes. They confuse facts, fabricate nonexistent information, and misinterpret context. This is not a bug — it is a fundamental property of generative models. They generate the most probable continuation, and probable does not always mean correct.

For structured output: even syntactically correct JSON with a perfect structure can contain incorrect values. A negative age. A date from a future millennium. An email without @. An amount as a string instead of a number.

Validation is not paranoia — it is professionalism. In critical systems, it is the only path to reliability.

## Validation Levels

Validation is a multi-layered defense. Each level catches its own class of errors.

**Syntactic validation** — format checking. Is the JSON valid? Not truncated? No syntax errors? With provider JSON Mode, this is guaranteed automatically. With prompts or weaker models, problems occur: unclosed brackets (exceeded the token limit), single quotes (influence of Python code), trailing commas (valid in some languages, invalid in JSON).

**Structural validation** — conformance to the expected structure. The JSON is valid, but are the fields correct, are the types right? Expected an array — got an object. Expected a required field "name" — it is missing. Expected a number in "age" — got the string "thirty-two". Verified against a JSON schema: presence of required fields, value types, nesting. With Function Calling, this is partially guaranteed but not fully.

**Semantic validation** — meaning of values. The JSON is valid, the structure is correct, but do the values make sense? An age of -5 is technically a number and matches the schema, but it is impossible. A birth date of 2050-01-01 has the correct format, but a future birth date is clearly invalid. "john.smith" is a string, but not an email. Business rules are checked: value ranges, string formats, logical relationships between fields.

**Consistency** — internal coherence of data. Birth year 1990, university graduation year 1985 — did the person graduate before being born? Position Junior Developer, 15 years of experience — a fifteen-year junior? Company Google, location Paris — plausible. Company Acme Widget Corp, location Mars — questionable. Verified that data does not contradict itself or common sense.

**External validation** — verification against external sources. Email john.smith@google.com — does it exist? You can send a verification email or check via API. Company TechCorp Inc — does it exist? Check through a business registry. Address 123 Main Street, Springfield — valid? A Geocoding API will confirm. The most reliable but also the most expensive. Reserved for critical fields.

## Error Handling Strategies

**Rejection and retry** — reject the invalid result, ask the model to try again. You can retry with the same prompt (sometimes helps due to randomness), but it is more effective to add feedback about the error. Example: "Your previous answer contained an error: age cannot be negative. Re-read the text carefully and extract the correct data." The model, upon receiving feedback, often self-corrects. It understands what it did wrong and adjusts its approach. Three to five attempts are usually sufficient.

**Partial acceptance** — accept the valid portion, re-request only the problematic fields. Example: "Name and company were extracted correctly. But the 'age' field contains -5. Find the person's actual age in the text." This saves tokens and time while preserving correct information.

**Default values** — use fallback values for non-critical fields. If an optional field is unrecognized or invalid — substitute null or a default value. Caution: silent substitution masks systemic problems. If 50% of fields are filled with default values, something is wrong with the extraction process.

**Escalation** — when automated handling is not possible. Generate a problem report, queue the task for manual processing, notify an operator. For critical data (finance, medicine, legal documents), escalation is preferable to automatic "correction."

## Feedback Loops

The most powerful improvement mechanism is feedback loops with the model.

**Immediate feedback** — after a failed validation, inform the model about the error. Formulate clearly and specifically: which field, what error, what was expected. Good example: "The 'email' field contains 'john.smith', but this is not a valid email — it is missing @ and a domain. Find the full email address in the text." Bad example: "Validation error." The model will not understand what to fix.

**Cumulative feedback** — for complex documents, accumulate context from all previous errors. "On the first attempt, you missed the date. On the second attempt, the date was found but in the wrong format. As a reminder, the format must be YYYY-MM-DD." This helps the model see the error pattern and avoid repeating it.

**Self-check** — ask the model to review its own answer before submitting. Example: "Now look at your answer critically. Are all fields filled? Do the values make sense? Are there any contradictions? If you find problems — fix them." Self-reflection often catches obvious errors missed during generation but noticed during careful review.

## Handling Edge Cases

**Missing data** — the needed value is not in the text. "Extract the age" — but age is not mentioned. Strategies: return null for optional fields, return a special value NOT_FOUND, require explicit indication for mandatory fields with escalation when absent. It is important to distinguish between "could not find" and "value is empty." An empty contact list (the person has no contacts) differs from null (contact information was not extracted).

**Multiple values** — "The company has offices in New York, London, and Singapore." — Which address to extract? If the schema expects a single value — a selection strategy is needed: first, primary (if determinable), random. Or extend the schema to an array. If not anticipated in advance — a reason to reconsider the schema design.

**Ambiguous data** — "The amount is about a million dollars." — What number to record? 1000000? But it is approximate, not exact. For approximate values, use ranges (900000-1100000), separate fields for confidence, or a text representation marked "approximate."

**Noisy data** — OCR recognition with errors, automatic speech transcription with gaps, text with typos and grammatical errors. Models are remarkably resilient to noise — they are trained on real data, including errors. But for critical fields, add checks for suspicious patterns (strange characters in names, unreadable sequences).

## Logging and Monitoring

**Validation metrics** — track statistics: percentage of successful validations, distribution of error types, number of retries, time to successful validation. If the error rate is increasing — it signals a problem. The input data format may have changed. Or the model was updated and behaves differently. Or load is causing timeouts and truncated responses.

**Error logging** — for each validation error, save the context: input text, model response, error type, expected vs. received. This is material for analysis and improvements. Recurring errors point to systemic problems that can be resolved by improving prompts or the schema.

**Alerts** — configure notifications for anomalies: error spikes, too many retries, complete validation failure for critical documents. The sooner you learn about a problem, the smaller the consequences.

## Practical Example: Validation with Retries

Architecture: ValidationService implements the extraction-with-retries pattern. The extractWithValidation method takes the source text and the target type class.

The process works in six stages:

1. **First attempt** — the model receives the text and extraction instructions, returns a structured result.

2. **Multi-level validation** — the result passes through three levels: structural validation via Bean Validation (checking @NotNull, @Min, @Max annotations), semantic validation (value ranges, string formats, business rules), and consistency checks (field coherence).

3. **Error handling** — all discovered errors are collected into a list with a detailed description of each problem.

4. **Feedback generation** — if errors are found, a detailed message is created for the model listing all problems and requesting corrections.

5. **Retry** — the model receives the original text plus information about previous errors and makes a new extraction attempt.

6. **Attempt limit** — the process repeats a maximum of 3 times. If validation is not passed after all attempts — an exception is thrown.

An alternative approach is the Self-reflection pattern via the SelfCheckingExtractor interface. After extraction, the model checks completeness (are all fields filled), logical consistency (are there contradictions), format correctness (do values match their types), and fixes any issues found before returning the result. This reduces the number of external validation iterations by moving part of the checks inside the generation process.

Key patterns: cumulative feedback helps avoid repeating errors, detailed problem descriptions improve correction quality, attempt limits prevent infinite loops, self-checking saves tokens and time.

## Key Takeaways

Validation is an essential component of a reliable system. Language models make mistakes, and those mistakes must be caught and handled.

Validation levels: syntactic (valid JSON), structural (correct fields and types), semantic (meaningful values), consistency, and external validation.

Handling strategies: retries with feedback, partial acceptance of results, fallback values, escalation to a human.

Feedback loops are a powerful mechanism. Inform the model about errors, ask for corrections, use self-checks.

Edge cases are inevitable: missing data, multiple values, ambiguities, noise. Each requires thoughtful handling.

Production monitoring is critical. Track metrics, log errors, configure alerts. Understanding what is happening is the key to reliability.

---

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Structured Output
**Previous:** [[02_Data_Extraction|Data Extraction from Unstructured Text]]
**Next:** [[../09_Conversational_AI/01_Dialog_Management|Dialog Management]]

---
