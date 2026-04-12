# Knowledge Layer

This directory consolidates everything PFAGENT learns from, trains on,
or loads at runtime. It is the merged successor of the legacy
`ANDES_Data/` and `PowerFlow_Specific/` directories (refactored
2026-04-10).

## Three sub-layers

```
knowledge/
├── raw/          ← unprocessed original sources
├── finetuning/   ← development-stage fine-tuning data + generators
└── rag/          ← deployment-stage runtime-loaded assets
```

The split mirrors the agent lifecycle:

| Stage       | Sub-layer       | Touched by                      |
|-------------|-----------------|---------------------------------|
| Source      | `raw/`          | one-off extraction / crawling   |
| Development | `finetuning/`   | data generators, audits, smoke  |
| Deployment  | `rag/`          | runtime FAISS index + few-shot  |

## `raw/` — original unprocessed sources

Reference material in its original form. Nothing in this directory is
loaded by the runtime.

| Path | Origin | Contents |
|---|---|---|
| `raw/manual_extraction/` | `ANDES_Data/Extract_From_Manual/` | ANDES manual PDF + extracted API/example CSVs and the script that produced them |
| `raw/github_issues/` | `ANDES_Data/GitHub_Issues_Crawler/` | crawler + crawled `andes_github_issues.csv` |
| `raw/sample_notebooks/` | `ANDES_Data/Sample_Code/` | Jupyter notebooks demonstrating ANDES workflows |

## `finetuning/` — development stage

Everything used to **build** the fine-tuning dataset. Scripts here run
offline (development-time only) and are NOT loaded by the runtime.

```
finetuning/
├── README.md          legacy PowerFlow_Specific README
├── guide.txt          notes
├── data/              .jsonl / .json training datasets + audit reports
├── scripts/           generators, audit, smoke test, trainer helpers
└── generated/
    ├── current/       active experiment outputs
    └── old/           archived experiment outputs
```

Key scripts (under `finetuning/scripts/`):

| Script | Purpose |
|---|---|
| `fine_tuning_dataset_utils.py` | shared dataset helpers + path constants |
| `build_clean_finetune_dataset.py` | builds the cleaned `.jsonl` from raw + curated sources |
| `audit_fine_tuning_data.py` | runs audits on a fine-tuning JSONL |
| `generate_verified_finetune_examples.py` | strict-verified single-turn generator |
| `generate_generalized_verified_finetune_examples.py` | multi-turn generator |
| `separate_finetune_data.py` | train/validation split |
| `smoke_test_finetune_examples.py` | CI smoke test (run on every PR) |
| `finetune.py` | uploads data and creates an OpenAI fine-tune job |
| `run_processes.py` / `run_processes.sh` | top-level orchestrator |

Key datasets (under `finetuning/data/`):

| File | Purpose |
|---|---|
| `fine_tuning_data.jsonl` | raw concatenated dataset |
| `fine_tuning_data.cleaned.jsonl` | cleaned + audited version |
| `train.cleaned.jsonl` / `validation.cleaned.jsonl` | train/val split |
| `verified_training_examples.json` | strictly verified single-turn examples |
| `generalized_verified_training_examples.json` | verified multi-turn conversations |
| `curated_training_examples.json` | curated hard cases |
| `*.audit.json` | dataset audit reports |
| `examples.csv` | canonical task list |

## `rag/` — deployment stage

The minimal artifact set the **runtime** actually loads. Everything
here is consumed by the live Streamlit app via
`text-to-sim/src/`.

| Path | Loaded by | Purpose |
|---|---|---|
| `rag/andes_manual.pdf` | `src/andes_manual.py` | Indexed into FAISS for RAG retrieval |
| `rag/code_examples/` | `src/codex_fixer.py` | Repo-aware fixer context + few-shot examples |

Touch this directory only when you want to change the assets a
deployed agent uses at request time. Do NOT mix in development-only
data or raw sources here.

## Data governance

The training/holdout boundary still applies:

- `knowledge/finetuning/` may be used for fine-tuning
- `verification/` is the holdout benchmark and must NOT be folded
  back into training data without an explicit governance decision

See [docs/DATA_GOVERNANCE.md](../docs/DATA_GOVERNANCE.md).
