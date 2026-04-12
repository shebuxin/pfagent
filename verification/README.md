# Verification Suite

This folder contains a full comparison harness for PFAGENT generation quality across four modes:

- `Base OpenAI`
- `RAG`
- `Fine-tuned`
- `Fine-tuned + RAG`

The suite is designed for multi-turn ANDES verification, not just one-shot prompt checks.

## Reviewer Access

If you are reviewing the benchmark evidence rather than developing locally, start with:

- [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md)

That guide links the final report, final package, and the full raw logs.

## Holdout Positioning

This directory is the benchmark-facing evaluation layer of the repository.

It should be treated differently from `knowledge/finetuning/`:

- `knowledge/finetuning/` contains training and fine-tuning data assets
- `verification/` contains holdout scenarios, oracle logic, and scoring

As a repository standard, future fine-tuning changes should not casually pull scenarios directly from this suite without an explicit data-governance decision.

## What it does

- Builds `132` deterministic conversation scenarios.
- Each scenario has `3` turns:
  - turn 1: baseline study request
  - turn 2: first follow-up with a case modification
  - turn 3: second follow-up with an additional case modification
- Covers both:
  - built-in ANDES cases
  - uploaded-case workflows
- Stores, for every turn:
  - full prompt
  - full model response
  - extracted Python code
  - execution output
  - parsed `RESULT_JSON`
  - expected oracle result
  - score breakdown

## Scenario design

The current suite spans:

- `ieee14`
- `ieee39`
- `kundur`
- `pjm5`

and tests both `builtin` and `uploaded` sources.

Follow-up modifications are limited to operations that were locally verified against the current ANDES runtime:

- add PQ load before setup
- scale all PQ loads after setup
- set slack-bus voltage target
- set first PV voltage target
- locate an existing PQ device by bus and modify that specific load
- locate an existing PV device by bus and modify that specific setpoint
- open a specific line identified by bus pair
- perform one-at-a-time N-1 screening over a candidate line list
- voltage ranking / threshold checks
- line-angle ranking / threshold checks
- voltage plots saved to file

## Evaluation rubric

Each turn is scored out of `100`:

- `Format` = `10`
- `Grounding` = `25`
- `Continuity` = `15`
- `Execution` = `20`
- `Semantic` = `25`
- `Artifact` = `5`

Conversation score is the mean of its three turns.

The semantic score compares model-produced `RESULT_JSON` against an ANDES oracle generated from the same scenario state and cumulative modifications.

## Running it

Run inside the `pfagent` conda environment:

```bash
conda activate pfagent
python verification/runner.py
```

If `OPENAI_API_KEY` is not already set:

```bash
conda activate pfagent
python verification/runner.py --api-key YOUR_KEY
```

Optional overrides:

```bash
python verification/runner.py \
  --models base_openai rag fine_tuned fine_tuned_rag \
  --base-model gpt-4o-mini \
  --fine-tuned-model ft:gpt-4.1-mini-2025-04-14:personal:pfagent:DOXJbJmU
```

## Outputs

Each run creates a timestamped folder under:

```text
verification/artifacts/
```

Inside each run:

- `scenario_suite.json`
- `verification_results.json`
- `verification_manifest.json`
- `raw/<model>/<scenario>/turn_xx/...`
- `reports/verification_summary.md`
- `reports/tables/*.csv`
- `reports/figures/*.png`

These outputs are generated artifacts. They are useful locally and for release evidence, but they should not be treated as routine source files in normal development commits.

The canonical retained final benchmark package on this branch lives under:

```text
verification/final/
```

## Main files

- `suite.py`: scenario generation
- `oracle.py`: ANDES oracle execution
- `runner.py`: model execution + scoring + artifact capture
- `reporting.py`: tables and figures
