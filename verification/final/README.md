# Final Verification Package

Date: 2026-04-03

This directory is the canonical retained verification package for the current PFAGENT release candidate.

It contains one final benchmark package rather than multiple historical comparison variants.

## What is inside

- `raw/`: full turn-level prompt / response / code / execution logs
- `reports/verification_summary.md`: summary markdown
- `reports/tables/`: CSV summaries
- `reports/figures/`: benchmark figures
- `scenario_suite.json`: the evaluated scenario definitions
- `verification_results.json`: structured run results
- `verification_manifest.json`: run metadata

## Validated scope

The retained final run covers:

- `100` deterministic scenarios
- `3` turns per scenario
- `Base OpenAI`
- `RAG`
- `Fine-tuned`
- `Fine-tuned + RAG`
- built-in and uploaded cases
- multi-turn case modification workflows

## Key result

The strongest product paths, `RAG` and `Fine-tuned + RAG`, both achieved:

- `100/100` scenario pass rate
- `300/300` turn pass rate
- `100.0` average conversation score

Pure `Fine-tuned` reached `58/100` scenario pass rate, while `Base OpenAI` remained a baseline-only comparison path.

Start with:

- [verification_summary.md](/home/bshe/Documents/git-research/pfagent/verification/final/reports/verification_summary.md)
- [scenario_pass_rate_by_model.png](/home/bshe/Documents/git-research/pfagent/verification/final/reports/figures/scenario_pass_rate_by_model.png)
- [raw](/home/bshe/Documents/git-research/pfagent/verification/final/raw)
