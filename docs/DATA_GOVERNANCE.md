# Data Governance

## Purpose

PFAGENT uses both training data and benchmark data. They must not be treated as the same thing.

## Data Classes

### 1. Training Data

Training data lives in [knowledge/finetuning/](../knowledge/finetuning/).
Deployment-stage RAG assets (manual PDF, few-shot code examples)
live in [knowledge/rag/](../knowledge/rag/), and unprocessed original
sources live in [knowledge/raw/](../knowledge/raw/).

Examples:

- canonical tasks
- generated prompt/code pairs
- curated hard cases
- verified training scenarios
- generalized verified conversations

These may be used for:

- fine-tuning
- training-data audits
- training-data regeneration

### 2. Holdout Verification Data

Holdout benchmark logic lives in [verification/](../verification/).

Examples:

- scenario definitions in `suite.py`
- oracle logic in `oracle.py`
- benchmark scoring in `runner.py` and `reporting.py`

These are intended for:

- evaluation
- release validation
- regression detection

They should not be casually folded into future fine-tuning data.

## Current Boundary

Some strictly validated training datasets already exist in `knowledge/finetuning/data/`, but the main `verification` suite remains the benchmark-facing layer.

Going forward, any proposal to derive training data from benchmark scenarios should be explicit, documented, and reviewed.

## Generated Artifacts

The following are treated as generated and should not be routinely committed:

- `text-to-sim/code_executions/`
- `text-to-sim/regression_runtime/`
- `text-to-sim/regression_results*.json`
- `verification/artifacts*/`
- local feedback logs
- Python caches

## Policy

1. Source code and reproducible dataset definitions belong in git.
2. Local runtime outputs do not belong in routine commits.
3. Holdout benchmark logic should remain stable enough to compare releases over time.
4. Any benchmark-to-training migration must be documented as a deliberate governance decision.
