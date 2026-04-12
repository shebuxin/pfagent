#!/usr/bin/env python3

import argparse
from pathlib import Path

try:
    from fine_tuning_dataset_utils import (
        DEFAULT_AUDIT_JSON,
        DEFAULT_CLEAN_JSONL,
        build_clean_dataset,
        write_audit,
        write_jsonl,
    )
except ModuleNotFoundError:  # pragma: no cover - import fallback for package-style execution
    from knowledge.finetuning.scripts.fine_tuning_dataset_utils import (
        DEFAULT_AUDIT_JSON,
        DEFAULT_CLEAN_JSONL,
        build_clean_dataset,
        write_audit,
        write_jsonl,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a cleaned/enhanced fine-tuning dataset for PFAGENT.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_CLEAN_JSONL),
        help="Output JSONL path for the cleaned dataset.",
    )
    parser.add_argument(
        "--audit-output",
        default=str(DEFAULT_AUDIT_JSON),
        help="Output JSON path for the audit report.",
    )
    parser.add_argument(
        "--include-high-level",
        action="store_true",
        help="Include high-level generated prompts. Disabled by default because they are often too vague.",
    )
    parser.add_argument(
        "--include-generated-summaries",
        action="store_true",
        help="Include summary-style prompt/answer pairs from generated_output_summary.csv. Disabled by default.",
    )
    args = parser.parse_args()

    dataset, audit = build_clean_dataset(
        include_high_level=args.include_high_level,
        include_generated_summaries=args.include_generated_summaries,
    )

    output_path = Path(args.output)
    audit_path = Path(args.audit_output)
    write_jsonl(dataset, output_path)
    write_audit(audit, audit_path)

    print(f"Clean dataset written to: {output_path}")
    print(f"Audit report written to: {audit_path}")
    print(f"Accepted pairs: {audit['accepted_pairs']} / {audit['candidate_pairs']}")
    print(f"Accepted by source: {audit['accepted_by_source']}")
    print(f"Rejections: {audit['rejections']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
