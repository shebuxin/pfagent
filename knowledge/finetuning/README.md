# PFAGENT Fine-Tuning Pipeline

This directory contains the ANDES-specific data pipeline used to build, validate, and package fine-tuning data.

## Responsibilities

- maintain canonical code examples
- generate synthetic prompt/code pairs
- add curated hard cases
- validate strict executable scenarios
- validate generalized multi-turn conversations
- write cleaned train/validation JSONL files

## Important Files

- `build_clean_finetune_dataset.py`
- `fine_tuning_dataset_utils.py`
- `generate_verified_finetune_examples.py`
- `generate_generalized_verified_finetune_examples.py`
- `run_processes.py`
- `smoke_test_finetune_examples.py`
- `train.cleaned.jsonl`
- `validation.cleaned.jsonl`

## Current Data Policy

- verified training examples in this directory are training assets
- the main `verification/` suite remains the benchmark-facing holdout layer

See [docs/DATA_GOVERNANCE.md](../docs/DATA_GOVERNANCE.md) for the boundary.

## Typical Workflow

```bash
conda activate pfagent
python generate_verified_finetune_examples.py
python generate_generalized_verified_finetune_examples.py
python build_clean_finetune_dataset.py
python smoke_test_finetune_examples.py
```
