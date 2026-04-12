# Verification Summary

## Evaluation System

- `Format` (10): exactly one Python code block, no missing code.
- `Grounding` (25): case loading correctness plus prompt-specific literals and required API usage.
- `Continuity` (15): follow-up turns preserve required earlier modifications.
- `Execution` (20): generated code runs successfully in the ANDES runtime.
- `Semantic` (25): `RESULT_JSON` matches the oracle result within tolerance.
- `Artifact` (5): required plot files are created and reported correctly.

## Model Summary

| model_key | model_label | scenarios | scenario_pass_rate | avg_conversation_score |
| --- | --- | --- | --- | --- |
| rag | RAG | 100 | 100.0 | 100.0 |
| fine_tuned_rag | Fine-tuned + RAG | 100 | 100.0 | 100.0 |
| fine_tuned | Fine-tuned | 100 | 58.0 | 86.44 |
| base_openai | Base OpenAI | 100 | 0.0 | 46.35 |

## Turn Summary

| model_label | turn_id | turn_pass_rate | avg_turn_score |
| --- | --- | --- | --- |
| Base OpenAI | 1 | 0.0 | 52.16 |
| Base OpenAI | 2 | 0.0 | 47.75 |
| Base OpenAI | 3 | 0.0 | 39.12 |
| Fine-tuned | 1 | 85.0 | 94.4 |
| Fine-tuned | 2 | 68.0 | 82.83 |
| Fine-tuned | 3 | 66.0 | 82.09 |
| Fine-tuned + RAG | 1 | 100.0 | 100.0 |
| Fine-tuned + RAG | 2 | 100.0 | 100.0 |
| Fine-tuned + RAG | 3 | 100.0 | 100.0 |
| RAG | 1 | 100.0 | 100.0 |
| RAG | 2 | 100.0 | 100.0 |
| RAG | 3 | 100.0 | 100.0 |

## Family Summary

| model_label | case_family | case_source | scenario_pass_rate | avg_conversation_score |
| --- | --- | --- | --- | --- |
| Base OpenAI | ieee14 | builtin | 0.0 | 48.68 |
| Base OpenAI | ieee14 | uploaded | 0.0 | 47.99 |
| Base OpenAI | ieee39 | builtin | 0.0 | 45.38 |
| Base OpenAI | ieee39 | uploaded | 0.0 | 46.5 |
| Base OpenAI | kundur | builtin | 0.0 | 43.24 |
| Base OpenAI | kundur | uploaded | 0.0 | 48.0 |
| Base OpenAI | pjm5 | builtin | 0.0 | 43.72 |
| Base OpenAI | pjm5 | uploaded | 0.0 | 45.95 |
| Fine-tuned | ieee14 | builtin | 85.71 | 93.93 |
| Fine-tuned | ieee14 | uploaded | 71.43 | 88.48 |
| Fine-tuned | ieee39 | builtin | 64.29 | 85.08 |
| Fine-tuned | ieee39 | uploaded | 42.86 | 85.41 |
| Fine-tuned | kundur | builtin | 57.14 | 80.45 |
| Fine-tuned | kundur | uploaded | 50.0 | 87.19 |
| Fine-tuned | pjm5 | builtin | 37.5 | 84.76 |
| Fine-tuned | pjm5 | uploaded | 37.5 | 84.76 |
| Fine-tuned + RAG | ieee14 | builtin | 100.0 | 100.0 |
| Fine-tuned + RAG | ieee14 | uploaded | 100.0 | 100.0 |
| Fine-tuned + RAG | ieee39 | builtin | 100.0 | 100.0 |
| Fine-tuned + RAG | ieee39 | uploaded | 100.0 | 100.0 |
| Fine-tuned + RAG | kundur | builtin | 100.0 | 100.0 |
| Fine-tuned + RAG | kundur | uploaded | 100.0 | 100.0 |
| Fine-tuned + RAG | pjm5 | builtin | 100.0 | 100.0 |
| Fine-tuned + RAG | pjm5 | uploaded | 100.0 | 100.0 |
| RAG | ieee14 | builtin | 100.0 | 100.0 |
| RAG | ieee14 | uploaded | 100.0 | 100.0 |
| RAG | ieee39 | builtin | 100.0 | 100.0 |
| RAG | ieee39 | uploaded | 100.0 | 100.0 |
| RAG | kundur | builtin | 100.0 | 100.0 |
| RAG | kundur | uploaded | 100.0 | 100.0 |
| RAG | pjm5 | builtin | 100.0 | 100.0 |
| RAG | pjm5 | uploaded | 100.0 | 100.0 |

## Failure Summary

| model_label | failure_category | count |
| --- | --- | --- |
| Base OpenAI | execution | 300 |
| Base OpenAI | semantic | 300 |
| Base OpenAI | grounding | 240 |
| Base OpenAI | continuity | 88 |
| Base OpenAI | artifact | 68 |
| Base OpenAI | format | 39 |
| Fine-tuned | semantic | 81 |
| Fine-tuned | execution | 69 |
| Fine-tuned | grounding | 39 |
| Fine-tuned | continuity | 17 |
| Fine-tuned | format | 15 |
| Fine-tuned | artifact | 10 |