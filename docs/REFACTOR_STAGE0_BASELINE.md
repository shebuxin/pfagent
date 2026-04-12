# Stage 0 Baseline — rag_chatbot.py Refactor Safety Net

**Date captured:** 2026-04-05
**Branch:** `reviewer`
**Conda env used:** `poweragent` (local), project canonical is `pfagent`

This document records the test-suite state **before** the rag_chatbot.py
decomposition planned in Stage 1. Any Stage 1 commit that changes these
numbers (beyond the single pre-existing failure noted below) must be
reverted or investigated.

## Unit-test baseline (text-to-sim/tests)

```
CONDA_ENV=poweragent make test-app
Ran 71 tests in ~85s
FAILED (failures=1)
```

- **70 passing / 1 failing** — `make test-app`
- **All 25 snapshot tests passing** — `make snapshot-check` (new in this stage)

### Known pre-existing failure (local-env only, fixed in CI)

`test_structured_codegen_matches_oracle_for_representative_scenarios`
fails with:

```
AttributeError: 'Line' object has no attribute 'set_status'
```

Root cause: the `poweragent` conda env ships ANDES
`1.10.0.post9+gd2cf9714c`, which exposes `Line.set(...)` but not
`Line.set_status(...)`. The structured codegen templates were upgraded
to `set_status()` as part of the runtime alignment work (see
`verification/runtime_alignment_report_20260404.md`, §4.2).

**CI status:** resolved. ANDES is now pinned to `>=2.0.0,<3.0.0` in
`environment.pfagent.yml` (2.0.0 is the release that introduced
`Line.set_status`). CI will always pick up a working ANDES and this
test will pass.

**Local status:** still fails for anyone whose conda env has a pre-2.0
ANDES install. To fix locally:

```bash
conda activate <your_env>
pip install --upgrade 'andes>=2.0.0,<3.0.0'
```

**Scope:** environment/version mismatch, **not** a code bug. Stage 1+
refactor commits must not introduce a second failure beyond this one
on a pre-2.0 ANDES env.

## Snapshot guard (new in Stage 0)

- **File:** `text-to-sim/tests/test_rag_chatbot_snapshots.py`
- **Snapshots:** `text-to-sim/tests/snapshots/rag_chatbot_snapshots.json`
- **Coverage:** 25 pinned outputs across 9 public functions:
  - `is_code_only_request`
  - `is_explanatory_followup_request`
  - `extract_effective_user_context`
  - `infer_requested_builtin_case`
  - `extract_python_code_blocks`
  - `validate_response_code`
  - `normalize_andes_response`
  - `build_andes_fallback_response`   (3 cases; 2 hit real template branches)
  - `build_andes_explanation_fallback_response`  (2 cases)

These are the functions currently imported by external modules from
`src.chatbots.openai.rag_chatbot`:

```
text-to-sim/src/chatbot_factory.py             -> RAGChatbot, RAGConfig
text-to-sim/scripts/andes_regression_check.py  -> RAGChatbot, RAGConfig,
                                                  build_andes_fallback_response,
                                                  validate_response_code
text-to-sim/tests/test_prompt_builder.py       -> RAGChatbot, RAGConfig
text-to-sim/tests/test_conversation_compaction.py -> RAGChatbot, RAGConfig
text-to-sim/tests/test_andes_response_guardrails.py -> 7 names (all covered)
text-to-sim/tests/test_structured_andes_codegen.py  -> RAGChatbot, RAGConfig,
                                                       StructuredAndesState,
                                                       build_structured_andes_response,
                                                       extract_python_code_blocks
verification/runner.py                         -> RAGChatbot, RAGConfig,
                                                  validate_response_code
```

After any Stage 1 file move, **every one of those import paths must still
resolve** (covered implicitly by the snapshot test importing them at module
load) and **`make snapshot-check` must pass with zero diffs**.

## Verification baseline (deferred)

A full `python verification/runner.py --scenario-count 100` run was
**not** captured here because:

1. It requires live OpenAI API calls and takes significant wall time.
2. The LLM responses have non-determinism, so "zero diff" is not the right
   contract.
3. The authoritative RAG baseline already lives in
   `verification/final/` (100% scenario pass rate, see
   `verification/current_test_report_20260328.md`).

**Contract for Stage 1:** after each refactor PR, run
`CONDA_ENV=<env> make verify` and confirm RAG/`Fine-tuned + RAG` scenario
pass rate ≥ 99% (allowing 1% for LLM nondeterminism).

## Refactor-safety workflow

During Stage 1 file-moves, per commit:

```bash
CONDA_ENV=poweragent make snapshot-check   # byte-for-byte pin (< 1 s)
CONDA_ENV=poweragent make test-app         # full unit suite    (~85 s)
```

If snapshot-check fails and the diff is intentional (semantic change
outside the refactor scope), first review, then:

```bash
CONDA_ENV=poweragent make snapshot-update
git diff text-to-sim/tests/snapshots/rag_chatbot_snapshots.json
```

## Stage 0 deliverables

| #   | Item                                                     | Status                  |
| --- | -------------------------------------------------------- | ----------------------- |
| 0.1 | Baseline test status captured (this doc)                 | √                      |
| 0.2 | Snapshot tests for 9 public exports (25 cases)           | √                      |
| 0.3 | Makefile targets:`snapshot-check`, `snapshot-update` | √                      |
| 0.4 | Remove duplicate `_outage_status`                      | -> Deferred to Stage 1 |

### Why 0.4 was deferred

Initial analysis flagged two definitions of `_outage_status` at
rag_chatbot.py:1481 and :1552. Closer inspection shows both live
**inside f-string template literals** (emitted into user-facing
generated Python code) — the outer functions are the two branches of
`build_andes_fallback_response` (uploaded-case vs builtin-case). The
duplication is real but in template text, not Python scope; removing it
requires extracting a shared template helper, which is exactly the
refactor targeted by Stage 1's `fallback.py` extraction.
