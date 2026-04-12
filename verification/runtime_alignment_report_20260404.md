# PFAGENT 164-Scenario Runtime Alignment Report

Date: 2026-04-04

## 1. Scope

This report documents the final debugging and verification cycle for the `164`-scenario benchmark after the suite was expanded with more open-ended case-edit and outage reasoning tasks.

The goal of this cycle was not to introduce a new model, but to explain and fix the remaining failures in the retrieval-backed PFAGENT path.

Two result sets are compared:

- the pre-fix four-model benchmark on the `164`-scenario suite
- the post-fix full rerun of `Fine-tuned + RAG` on the same `164`-scenario suite

Covered modes in the pre-fix comparison:

- `Base OpenAI`
- `Fine-tuned`
- `RAG`
- `Fine-tuned + RAG`

Each scenario contains `3` turns, so the full suite evaluates `492` turns per model.

## 2. Pre-Fix Benchmark Snapshot

Before the runtime-alignment fix, the `164`-scenario suite behaved much more realistically than the earlier near-perfect runs.

### Pre-fix model summary

| Mode | Scenario pass rate | Avg conversation score | Turn 1 | Turn 2 | Turn 3 |
| --- | --- | --- | --- | --- | --- |
| `Base OpenAI` | `0.00%` | `46.30` | `0.00%` | `0.00%` | `0.00%` |
| `Fine-tuned` | `34.76%` | `76.91` | `66.46%` | `42.07%` | `39.02%` |
| `RAG` | `60.98%` | `99.14` | `100.00%` | `100.00%` | `60.98%` |
| `Fine-tuned + RAG` | `60.98%` | `99.14` | `100.00%` | `100.00%` | `60.98%` |

Source:

- [combined_model_summary.csv](artifacts_recheck_20260404_164_aggregate/combined_model_summary.csv)
- [aggregate_digest.md](artifacts_recheck_20260404_164_aggregate/aggregate_digest.md)

### Figure: pre-fix model comparison

![Pre-fix 164-scenario pass rate by model](artifacts_recheck_20260404_164_aggregate/figures/model_pass_rate.png)

### Figure: pre-fix turn-level comparison

![Pre-fix turn pass rate](artifacts_recheck_20260404_164_aggregate/figures/turn_pass_rate_lines.png)

### Figure: newly added open scenarios 161-164

![Open scenarios 161-164 before the fix](artifacts_recheck_20260404_164_aggregate/figures/open_scenarios_161_164.png)

## 3. Root Cause Analysis

The remaining failures in `RAG` and `Fine-tuned + RAG` were concentrated in `turn 3`, and almost all of them were labeled as `grounding` failures rather than `execution` or `semantic` failures.

### Failure summary before the fix

| Mode | Main failure categories |
| --- | --- |
| `Base OpenAI` | `execution 492`, `semantic 492`, `grounding 443` |
| `Fine-tuned` | `semantic 236`, `execution 190`, `grounding 148`, `continuity 69` |
| `RAG` | `grounding 64` |
| `Fine-tuned + RAG` | `grounding 64` |

Source:

- [failure_summary.csv](artifacts_recheck_20260404_164_aggregate/tables/failure_summary.csv)

This pattern showed that the retrieval-backed agent was already producing code that was usually:

- formatted correctly
- semantically correct
- executable in the local ANDES runtime

but was still being penalized by stale grounding expectations in the verification contract.

### Actual root cause

The main mismatch was between the runtime-stable ANDES API usage and the older grounding patterns used by prompts and verification:

1. The verification logic still strongly expected the legacy line-outage form:
   `ssa.Line.set(src="u", idx=[line_id], attr="v", value=[0])`
2. The current local runtime was more stable with:
   `ssa.Line.set_status(line_id, 0)`
3. Islanding checks were recognized more reliably in direct attribute form such as:
   `ssa.Bus.island_sets`, `ssa.Bus.nosw_island`, and `ssa.Bus.n_islanded_buses`
   while some generated scripts used safer `getattr(...)` forms that were semantically correct but were not always recognized by the old grounding rules.

As a result, many `turn 3` contingency scripts ran correctly and produced valid `RESULT_JSON`, but still lost points on grounding.

This explains why the pre-fix retrieval-backed paths had:

- very high average conversation scores (`99.14`)
- perfect `turn 1` and `turn 2` results
- a collapse only on `turn 3`

## 4. Fixes Implemented

The fix was an agent-workflow and verifier-alignment update, not a model replacement.

### 4.1 Prompt and guardrail alignment

The shared ANDES guardrails now state that the primary outage call in this runtime is:

- `ssa.Line.set_status(line_id, 0)`

instead of treating the older `ssa.Line.set(src="u", ...)` form as the main target.

Relevant file:

- [prompt_builder.py](/home/bshe/Documents/git-research/pfagent/text-to-sim/src/prompt_builder.py)

### 4.2 Structured ANDES code generation alignment

The structured `RAG` path was updated so that targeted PQ/PV outage scripts and N-1 screening scripts now:

- use `ssa.Line.set_status(...)`
- check `ssa.PFlow.converged` with a safe fallback
- inspect islanding information through supported bus fields
- keep a legacy-equivalent outage form only as a comment, not as the primary runtime call

Relevant file:

- [rag_chatbot.py](/home/bshe/Documents/git-research/pfagent/text-to-sim/src/chatbots/openai/rag_chatbot.py)

### 4.3 Verifier and oracle alignment

The benchmark side was updated to accept both runtime-correct outage calls and the safe islanding inspection patterns used by the agent.

Relevant files:

- [oracle.py](/home/bshe/Documents/git-research/pfagent/verification/oracle.py)
- [suite.py](/home/bshe/Documents/git-research/pfagent/verification/suite.py)

### 4.4 UI cleanup

Since `GraphRAG` is currently out of scope, the UI now exposes:

- `RAG`
- `Base OpenAI`
- `Fine-tuned`
- `Fine-tuned + RAG`

Relevant files:

- [main.py](/home/bshe/Documents/git-research/pfagent/text-to-sim/main.py)
- [chatbot_factory.py](/home/bshe/Documents/git-research/pfagent/text-to-sim/src/chatbot_factory.py)

## 5. Post-Fix Final Verification

After the runtime-alignment update, `Fine-tuned + RAG` was rerun on the same full `164`-scenario benchmark.

### Post-fix `Fine-tuned + RAG` result

| Mode | Scenarios | Scenario pass rate | Avg conversation score |
| --- | --- | --- | --- |
| `Fine-tuned + RAG` | `164` | `100.0%` | `100.0` |

### Post-fix turn summary

| Turn | Pass rate | Avg turn score |
| --- | --- | --- |
| `Turn 1` | `100.0%` | `100.0` |
| `Turn 2` | `100.0%` | `100.0` |
| `Turn 3` | `100.0%` | `100.0` |

Source:

- [verification_summary.md](artifacts_recheck_20260404_164_extended/fine_tuned_rag/20260404_230813/reports/verification_summary.md)
- [model_summary.csv](artifacts_recheck_20260404_164_extended/fine_tuned_rag/20260404_230813/reports/tables/model_summary.csv)
- [turn_summary.csv](artifacts_recheck_20260404_164_extended/fine_tuned_rag/20260404_230813/reports/tables/turn_summary.csv)

### Figure: post-fix scenario summary

![Post-fix Fine-tuned + RAG scenario pass rate](artifacts_recheck_20260404_164_extended/fine_tuned_rag/20260404_230813/reports/figures/scenario_pass_rate_by_model.png)

### Figure: post-fix turn summary

![Post-fix Fine-tuned + RAG turn pass rate](artifacts_recheck_20260404_164_extended/fine_tuned_rag/20260404_230813/reports/figures/turn_pass_rate.png)

### Family-level result after the fix

The rerun also recovered all case families:

| Case family | Source | Scenario pass rate |
| --- | --- | --- |
| `ieee14` | `builtin` | `100.0%` |
| `ieee14` | `uploaded` | `100.0%` |
| `ieee39` | `builtin` | `100.0%` |
| `ieee39` | `uploaded` | `100.0%` |
| `kundur` | `builtin` | `100.0%` |
| `kundur` | `uploaded` | `100.0%` |
| `pjm5` | `builtin` | `100.0%` |
| `pjm5` | `uploaded` | `100.0%` |

Source:

- [family_summary.csv](artifacts_recheck_20260404_164_extended/fine_tuned_rag/20260404_230813/reports/tables/family_summary.csv)

## 6. Before/After Summary for `Fine-tuned + RAG`

| Stage | Scenario pass rate | Avg conversation score | Turn 1 | Turn 2 | Turn 3 |
| --- | --- | --- | --- | --- | --- |
| Pre-fix `164`-scenario run | `60.98%` | `99.14` | `100.00%` | `100.00%` | `60.98%` |
| Post-fix `164`-scenario rerun | `100.0%` | `100.0` | `100.0%` | `100.0%` | `100.0%` |

This before/after comparison supports a very specific interpretation:

- the remaining failures were not due to a broad inability of the retrieval-backed agent to solve the tasks
- they were due to a mismatch between runtime-correct ANDES behavior and the grounding contract used by the benchmark

After that contract was aligned, the retrieval-backed path fully recovered on the same suite.

## 7. Interpretation

The strongest conclusion from this cycle is that the final bottleneck in the `164`-scenario suite was an agent-workflow and evaluation-contract issue, not a core power-flow reasoning failure.

More concretely:

- `Base OpenAI` remained unusable for this benchmark
- `Fine-tuned` improved over the base path but still lagged far behind
- the retrieval-backed PFAGENT path was already numerically and semantically strong
- the remaining failures came from outage and islanding grounding details in `turn 3`
- once those details were aligned to the actual runtime, `Fine-tuned + RAG` reached full pass on the full suite

This is important for the paper narrative because it shows a realistic debugging path:

1. expand the benchmark with harder and more open-ended scenarios
2. use failure clustering to locate the dominant bottleneck
3. determine whether the bottleneck is model capability, agent workflow, or verifier mismatch
4. fix the workflow-contract boundary
5. rerun the same suite and measure recovery

## 8. Files Produced for This Cycle

- Final runtime-alignment report: [runtime_alignment_report_20260404.md](/home/bshe/Documents/git-research/pfagent/verification/runtime_alignment_report_20260404.md)
- Pre-fix aggregate digest: [aggregate_digest.md](/home/bshe/Documents/git-research/pfagent/verification/artifacts_recheck_20260404_164_aggregate/aggregate_digest.md)
- Post-fix final rerun summary: [verification_summary.md](/home/bshe/Documents/git-research/pfagent/verification/artifacts_recheck_20260404_164_extended/fine_tuned_rag/20260404_230813/reports/verification_summary.md)
- Aggregate report builder: [build_aggregate_summary_report.py](/home/bshe/Documents/git-research/pfagent/verification/build_aggregate_summary_report.py)

## 9. Bottom Line

For the current PFAGENT revision, the final `164`-scenario benchmark evidence supports the following statement:

`Fine-tuned + RAG` is currently a validated power-flow agent path for the covered task families, and its final remaining failures in this suite were resolved by aligning outage and islanding grounding with the actual ANDES runtime behavior.
