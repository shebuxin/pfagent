#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path
from typing import List

try:
    from fine_tuning_dataset_utils import DEFAULT_CLEAN_JSONL
except ModuleNotFoundError:  # pragma: no cover - import fallback for package-style execution
    from knowledge.finetuning.scripts.fine_tuning_dataset_utils import DEFAULT_CLEAN_JSONL


def load_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Split a fine-tuning dataset into train/validation JSONL files.")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_CLEAN_JSONL),
        help="Input JSONL file. Defaults to the cleaned dataset.",
    )
    parser.add_argument(
        "--train-output",
        default=str(DEFAULT_CLEAN_JSONL.with_name("train.cleaned.jsonl")),
        help="Output path for the training split.",
    )
    parser.add_argument(
        "--validation-output",
        default=str(DEFAULT_CLEAN_JSONL.with_name("validation.cleaned.jsonl")),
        help="Output path for the validation split.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Training fraction between 0 and 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    rows = load_jsonl(input_path)
    random.Random(args.seed).shuffle(rows)

    train_ratio = min(max(args.train_ratio, 0.1), 0.99)
    split_index = int(len(rows) * train_ratio)
    train_rows = rows[:split_index]
    validation_rows = rows[split_index:]

    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.validation_output), validation_rows)

    print(f"Input dataset: {input_path}")
    print(f"Total entries: {len(rows)}")
    print(f"Training set: {len(train_rows)} entries")
    print(f"Validation set: {len(validation_rows)} entries")
    print(f"Train output: {Path(args.train_output)}")
    print(f"Validation output: {Path(args.validation_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
