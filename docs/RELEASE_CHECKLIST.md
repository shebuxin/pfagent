# Release Checklist

Use this checklist before tagging or publishing a release candidate.

## Product

- The repository README reflects the current product scope.
- The recommended runtime mode is documented.
- User-facing docs are up to date.

## Code and Runtime

- The Streamlit app launches successfully in the `pfagent` environment.
- The default model configuration is intentional and documented.
- No local secrets or runtime data are included in the release commit.

## Verification

- Unit tests pass.
- Fine-tune smoke tests pass.
- The verification suite has been run on the intended release candidate.
- Benchmark results are summarized in a stable report, not only in raw artifact folders.

## Data Governance

- No accidental holdout leakage into the training set was introduced.
- Newly generated training datasets are validated and auditable.
- Benchmark scenarios used for release claims are still reproducible.

## Repository Hygiene

- No `__pycache__`, `.DS_Store`, or local scratch outputs are committed.
- New generated artifact directories are not included unintentionally.
- Docs, scripts, and file references point to live paths.

