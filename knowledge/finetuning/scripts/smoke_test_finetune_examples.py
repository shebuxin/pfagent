#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple


SCRIPTS_DIR = Path(__file__).resolve().parent
FINETUNING_ROOT = SCRIPTS_DIR.parent
KNOWLEDGE_ROOT = FINETUNING_ROOT.parent
DATA_DIR = FINETUNING_ROOT / "data"
CODE_EXAMPLES_DIR = KNOWLEDGE_ROOT / "rag" / "code_examples"

CURATED_JSON = DATA_DIR / "curated_training_examples.json"
GENERALIZED_JSON = DATA_DIR / "generalized_verified_training_examples.json"
DEFAULT_CODE_EXAMPLES = [
    CODE_EXAMPLES_DIR / "powerflow-ex1.py",
    CODE_EXAMPLES_DIR / "powerflow-ex9.py",
    CODE_EXAMPLES_DIR / "powerflow-ex12.py",
    CODE_EXAMPLES_DIR / "powerflow-ex15.py",
    CODE_EXAMPLES_DIR / "powerflow-ex29.py",
]


def run_script(script_path: Path, cwd: Path, env: dict) -> Tuple[bool, str]:
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = result.stdout
    if result.stderr:
        output += f"\nSTDERR:\n{result.stderr}"
    return result.returncode == 0, output.strip()


def iter_builtin_curated_examples() -> Iterable[Tuple[str, str]]:
    payload = json.loads(CURATED_JSON.read_text(encoding="utf-8"))
    for item in payload.get("examples", []):
        assistant = item.get("assistant", "")
        if "andes.get_case(" in assistant:
            yield item.get("id", "curated_example"), assistant


def iter_generalized_builtin_turns(max_conversations: int = 2) -> Iterable[Tuple[str, str]]:
    if not GENERALIZED_JSON.exists():
        return

    payload = json.loads(GENERALIZED_JSON.read_text(encoding="utf-8"))
    yielded_conversations = 0
    for item in payload.get("examples", []):
        messages = item.get("messages", [])
        built_in_turns = [
            (index // 2 + 1, message.get("content", ""))
            for index, message in enumerate(messages)
            if message.get("role") == "assistant" and "andes.get_case(" in message.get("content", "")
        ]
        if not built_in_turns:
            continue

        yielded_conversations += 1
        example_id = item.get("id", f"generalized_{yielded_conversations:02d}")
        for turn_id, assistant in built_in_turns:
            yield f"{example_id}_turn_{turn_id:02d}", assistant

        if yielded_conversations >= max_conversations:
            break


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test PFAGENT fine-tuning examples that should run locally.")
    parser.add_argument(
        "--skip-curated",
        action="store_true",
        help="Skip curated built-in examples and only run selected code_examples scripts.",
    )
    parser.add_argument(
        "--skip-generalized",
        action="store_true",
        help="Skip generalized built-in conversation turns.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-pfagent-smoke")

    failures: List[str] = []

    for script_path in DEFAULT_CODE_EXAMPLES:
        ok, output = run_script(script_path, cwd=KNOWLEDGE_ROOT.parent, env=env)
        status = "PASS" if ok else "FAIL"
        print(f"{status} code_example {script_path.name}")
        if not ok:
            failures.append(f"{script_path.name}\n{output}")

    if not args.skip_curated:
        with tempfile.TemporaryDirectory(prefix="pfagent-curated-smoke-", dir="/tmp") as tmpdir:
            tmp_root = Path(tmpdir)
            for example_id, assistant_code in iter_builtin_curated_examples():
                script_path = tmp_root / f"{example_id}.py"
                script_path.write_text(assistant_code, encoding="utf-8")
                ok, output = run_script(script_path, cwd=tmp_root, env=env)
                status = "PASS" if ok else "FAIL"
                print(f"{status} curated {example_id}")
                if not ok:
                    failures.append(f"{example_id}\n{output}")

    if not args.skip_generalized:
        with tempfile.TemporaryDirectory(prefix="pfagent-generalized-smoke-", dir="/tmp") as tmpdir:
            tmp_root = Path(tmpdir)
            for example_id, assistant_code in iter_generalized_builtin_turns():
                script_path = tmp_root / f"{example_id}.py"
                script_path.write_text(assistant_code, encoding="utf-8")
                ok, output = run_script(script_path, cwd=tmp_root, env=env)
                status = "PASS" if ok else "FAIL"
                print(f"{status} generalized {example_id}")
                if not ok:
                    failures.append(f"{example_id}\n{output}")

    if failures:
        print("\nFailures:")
        for failure in failures:
            print(failure)
            print("-" * 60)
        return 1

    print("\nAll selected fine-tuning smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
