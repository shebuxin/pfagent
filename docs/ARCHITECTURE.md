# PFAGENT Architecture

## Product Definition

PFAGENT is a tool-augmented LLM system for translating natural-language study requests into executable ANDES power-flow workflows.

At a high level, the system has three layers:

1. application runtime
2. training-data production
3. holdout verification

## 1. Application Runtime

The runtime layer lives in [text-to-sim/](../text-to-sim/).

Main responsibilities:

- Streamlit UI
- agent initialization and mode selection
- manual bootstrapping
- uploaded-file handling
- generated-code execution
- plot capture and display

The main agent modes are:

- `Base OpenAI`
- `Fine-tuned`
- `GraphRAG`
- `Fine-tuned + RAG`

Current recommended mode:

- `Fine-tuned + RAG`

## 2. Knowledge Layer

All training-stage and RAG-stage knowledge lives in
[knowledge/](../knowledge/), organized into three sub-layers:

- [knowledge/raw/](../knowledge/raw/) — original unprocessed sources:
  manual-extraction CSVs, GitHub-issue crawls, sample notebooks.
- [knowledge/finetuning/](../knowledge/finetuning/) — development-stage
  fine-tuning data plus the generators, audits, and smoke tests that
  produce it:
  - `data/` — `.jsonl` / `.json` training datasets and audit reports
  - `scripts/` — generators, audit, smoke test, trainer helpers
  - `generated/` — intermediate experiment outputs
- [knowledge/rag/](../knowledge/rag/) — deployment-stage assets loaded
  by the runtime RAG pipeline:
  - `andes_manual.pdf` — indexed by `src/andes_manual.py` into FAISS
  - `code_examples/` — few-shot / repo-aware fixer context

Design principle:

- training data should be reproducible, validated, and auditable
- deployment (RAG) assets should be the minimal artifact set the
  runtime actually loads; nothing else

## 3. Holdout Verification Layer

The verification layer lives in [verification/](../verification/).

It exists to answer:

- does the model return executable code?
- is the code grounded in the correct ANDES workflow?
- does it preserve multi-turn state?
- does it match oracle-computed results?

Design principle:

- verification data is benchmark-facing, not casually recycled into training

## Data Flow

```text
User prompt
  -> Streamlit UI
  -> selected agent mode
  -> ANDES-manual grounding and optional document context
  -> generated Python
  -> runtime execution
  -> outputs, plots, and follow-up state
```

Training and evaluation should remain separate:

```text
ANDES source corpus / examples
  -> cleaned and validated fine-tune data
  -> fine-tuned model

Holdout verification scenarios
  -> benchmark run
  -> scores, reports, and release validation
```

## Repository Standards

Professional repository standards for this project mean:

- clean separation between source and generated artifacts
- explicit benchmark governance
- reproducible scripts for data generation
- stable docs for contributors and users
- repeatable tests and release checks
