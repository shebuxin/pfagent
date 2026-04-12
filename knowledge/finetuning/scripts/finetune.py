#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

from openai import OpenAI


SCRIPTS_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPTS_DIR.parent / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload PFAGENT fine-tuning data and create a fine-tune job.")
    parser.add_argument(
        "--train-file",
        default=str(DATA_DIR / "train.cleaned.jsonl"),
        help="Training JSONL file to upload.",
    )
    parser.add_argument(
        "--validation-file",
        default=str(DATA_DIR / "validation.cleaned.jsonl"),
        help="Optional validation JSONL file to upload.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini-2024-07-18",
        help="Base model for the fine-tune job.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Create a training-only fine-tune job without a validation file.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    client = OpenAI(api_key=api_key)

    with open(args.train_file, "rb") as handle:
        train_file = client.files.create(file=handle, purpose="fine-tune")

    validation_file = None
    if not args.skip_validation:
        with open(args.validation_file, "rb") as handle:
            validation_file = client.files.create(file=handle, purpose="fine-tune")

    job_kwargs = {
        "training_file": train_file.id,
        "model": args.model,
    }
    if validation_file is not None:
        job_kwargs["validation_file"] = validation_file.id

    job = client.fine_tuning.jobs.create(**job_kwargs)

    print(f"Training file id: {train_file.id}")
    if validation_file is not None:
        print(f"Validation file id: {validation_file.id}")
    print(f"Fine-tune job id: {job.id}")
    print(f"Base model: {args.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
