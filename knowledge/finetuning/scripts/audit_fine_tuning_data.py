#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

try:
    from fine_tuning_dataset_utils import DEFAULT_RAW_JSONL, audit_jsonl
except ModuleNotFoundError:  # pragma: no cover - import fallback for package-style execution
    from knowledge.finetuning.scripts.fine_tuning_dataset_utils import DEFAULT_RAW_JSONL, audit_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a fine-tuning JSONL dataset for common PFAGENT data issues.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_RAW_JSONL),
        help="JSONL file to audit.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the audit report as JSON.",
    )
    args = parser.parse_args()

    report = audit_jsonl(Path(args.input))
    print(json.dumps(report, indent=2, ensure_ascii=True))

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
