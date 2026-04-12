# PFAGENT Reviewer Guide

Date: 2026-04-03

This branch keeps one final verification package for reviewer inspection.

Use this page as the entry point instead of navigating the raw logs directly.

## 1. Recommended Reading Order

1. Final report: [current_test_report_20260328.md](current_test_report_20260328.md)
2. User manual: [pfagent_user_manual_20260328.md](pfagent_user_manual_20260328.md)
3. Final benchmark package: [final/README.md](final/README.md)

## 2. What Is Preserved On This Branch

This branch preserves:

- one final benchmark package under `verification/final/`
- the final report and PDF
- the user manual and PDF

The final package includes both:

- high-level summaries and figures
- full turn-level raw logs

## 3. Final Benchmark Entry Points

Start here for the final benchmark evidence:

- Final package root:
  [final](/home/bshe/Documents/git-research/pfagent/verification/final)
- Final summary:
  [verification_summary.md](/home/bshe/Documents/git-research/pfagent/verification/final/reports/verification_summary.md)
- Final report:
  [current_test_report_20260328.md](/home/bshe/Documents/git-research/pfagent/verification/current_test_report_20260328.md)

## 4. Where To Inspect Raw Turn-Level Logs

For turn-by-turn evidence, open:

- full raw root:
  [raw](/home/bshe/Documents/git-research/pfagent/verification/final/raw)
- `Base OpenAI` logs:
  [base_openai](/home/bshe/Documents/git-research/pfagent/verification/final/raw/base_openai)
- `RAG` logs:
  [rag](/home/bshe/Documents/git-research/pfagent/verification/final/raw/rag)
- pure `Fine-tuned` logs:
  [fine_tuned](/home/bshe/Documents/git-research/pfagent/verification/final/raw/fine_tuned)
- `Fine-tuned + RAG` logs:
  [fine_tuned_rag](/home/bshe/Documents/git-research/pfagent/verification/final/raw/fine_tuned_rag)

Each turn folder typically includes:

- `prompt.txt`
- `response.md`
- `generated_code.py`
- `execution_output.txt`
- `expected_result.json`
- `actual_result.json`
- `turn_result.json`
- `workspace/`

## 5. Figures And Tables

Quick figure entry points:

- overall score:
  [overall_score_by_model.png](/home/bshe/Documents/git-research/pfagent/verification/final/reports/figures/overall_score_by_model.png)
- scenario pass rate:
  [scenario_pass_rate_by_model.png](/home/bshe/Documents/git-research/pfagent/verification/final/reports/figures/scenario_pass_rate_by_model.png)
- turn pass rate:
  [turn_pass_rate.png](/home/bshe/Documents/git-research/pfagent/verification/final/reports/figures/turn_pass_rate.png)
- failure categories:
  [failure_categories.png](/home/bshe/Documents/git-research/pfagent/verification/final/reports/figures/failure_categories.png)

Quick table entry points:

- model summary:
  [model_summary.csv](/home/bshe/Documents/git-research/pfagent/verification/final/reports/tables/model_summary.csv)
- turn summary:
  [turn_summary.csv](/home/bshe/Documents/git-research/pfagent/verification/final/reports/tables/turn_summary.csv)
- failure summary:
  [failure_summary.csv](/home/bshe/Documents/git-research/pfagent/verification/final/reports/tables/failure_summary.csv)
- scenario-level results:
  [scenario_level_results.csv](/home/bshe/Documents/git-research/pfagent/verification/final/reports/tables/scenario_level_results.csv)

## 6. Representative Examples

For a concrete pure `Fine-tuned` partial-failure example:

- response:
  [response.md](/home/bshe/Documents/git-research/pfagent/verification/final/raw/fine_tuned/scenario_001/turn_01/response.md)
- scored result:
  [turn_result.json](/home/bshe/Documents/git-research/pfagent/verification/final/raw/fine_tuned/scenario_001/turn_01/turn_result.json)

For a concrete `RAG` success example:

- response:
  [response.md](/home/bshe/Documents/git-research/pfagent/verification/final/raw/rag/scenario_001/turn_01/response.md)
- scored result:
  [turn_result.json](/home/bshe/Documents/git-research/pfagent/verification/final/raw/rag/scenario_001/turn_01/turn_result.json)

For a concrete `Fine-tuned + RAG` success example:

- response:
  [response.md](/home/bshe/Documents/git-research/pfagent/verification/final/raw/fine_tuned_rag/scenario_001/turn_01/response.md)
- scored result:
  [turn_result.json](/home/bshe/Documents/git-research/pfagent/verification/final/raw/fine_tuned_rag/scenario_001/turn_01/turn_result.json)

## 7. Notes

- This branch intentionally keeps the final generated benchmark artifacts for reviewer access.
- The final retained story is a single canonical package, not multiple old/new comparison trees.
