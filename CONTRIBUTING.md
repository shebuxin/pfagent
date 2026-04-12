# Contributing to PFAGENT

## Scope

PFAGENT combines application code, data-generation utilities, and a holdout verification suite. Changes should preserve the distinction between those layers.

## Development Setup

```bash
conda env create -f environment.pfagent.yml
conda activate pfagent
```

## Repository Rules

1. Do not commit secrets, API keys, or local `.env` files.
2. Do not commit runtime outputs such as `code_executions/`, feedback logs, or new benchmark artifact folders.
3. Do not mix holdout verification scenarios into fine-tuning data without an explicit data-governance decision.
4. Keep generated benchmark evidence in local or release artifacts, not in routine source-control changes.
5. Prefer small, reviewable changes that keep the Streamlit app runnable.

## Before Opening a PR

Run:

```bash
conda activate pfagent
python -m unittest discover -s text-to-sim/tests -p "test_*.py"
python -m unittest discover -s verification/tests -p "test_*.py"
python knowledge/finetuning/scripts/smoke_test_finetune_examples.py
```

If your change affects benchmark logic, also run:

```bash
conda activate pfagent
python verification/runner.py --scenario-count 100
```

If your change affects training data generation, also run:

```bash
conda activate pfagent
python knowledge/finetuning/scripts/build_clean_finetune_dataset.py
```

## Pull Request Checklist

- The change aligns with the PFAGENT product goal.
- The user-facing behavior is documented if it changed.
- The benchmark/holdout boundary is still respected.
- Generated artifacts are not included unintentionally.
- The relevant tests or smoke tests were run.

## Coding Conventions

- Use Python 3.11-compatible code.
- Favor small utilities and explicit naming over hidden side effects.
- Keep prompt contracts and evaluation contracts precise and machine-checkable.
- When modifying the agent, prefer changes that preserve reproducibility in verification.

## Documentation Expectations

Update docs when you change:

- repository layout
- user workflow
- verification policy
- fine-tuning data pipeline
- release workflow

