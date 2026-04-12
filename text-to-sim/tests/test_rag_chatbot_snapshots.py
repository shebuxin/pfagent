"""Character-level snapshot tests for rag_chatbot public exports.

These tests pin the exact output of every function that is currently imported
from ``src.chatbots.openai.rag_chatbot`` by external modules (other source
files, tests, scripts, and verification/runner.py). They exist to guard the
upcoming refactor that will break up ``rag_chatbot.py`` into smaller modules
(see the Stage 1 plan). After each refactor step, re-running this suite must
produce zero diffs against ``tests/snapshots/rag_chatbot_snapshots.json``.

Usage
-----
Normal run (compare mode, used in CI and by ``make test``)::

    python -m unittest text-to-sim/tests/test_rag_chatbot_snapshots.py

Regenerate snapshots (only after a deliberate behavioral change)::

    UPDATE_SNAPSHOTS=1 python -m unittest text-to-sim/tests/test_rag_chatbot_snapshots.py

A missing snapshot file triggers a hard failure so the refactor never
silently loses coverage. To bootstrap the file initially, run once with
``UPDATE_SNAPSHOTS=1``.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.chatbots.openai.rag_chatbot import (  # noqa: E402
    build_andes_explanation_fallback_response,
    build_andes_fallback_response,
    extract_effective_user_context,
    extract_python_code_blocks,
    infer_requested_builtin_case,
    is_code_only_request,
    is_explanatory_followup_request,
    normalize_andes_response,
    validate_response_code,
)


SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "rag_chatbot_snapshots.json"
UPDATE_MODE = os.environ.get("UPDATE_SNAPSHOTS") == "1"


# Shared fixtures -----------------------------------------------------------

CODE_ONLY_CTX = (
    "Generate runnable Python code only. Use an ANDES built-in case. "
    "Compute bus voltages after power flow convergence and print a line "
    "beginning with 'RESULT_JSON:' containing key 'bus_voltages'."
)

EXPLAIN_CTX = "Explain the results I just received from the previous run."

IEEE14_PROMPT = (
    "Generate runnable Python code only. Use the IEEE 14-bus case. "
    "Run power flow and print RESULT_JSON with key 'bus_voltages'."
)

# Prompts crafted to hit specific fallback template branches so their body
# (big multi-line code generators) gets pinned by snapshot.
IEEE14_SLACK_TOP3_PROMPT = (
    "Use the IEEE 14 case. Print the slack bus voltage and the top-3 "
    "highest bus voltages after running power flow."
)
IEEE39_PQ_BUS_PROMPT = (
    "Use the IEEE 39 case. Add a PQ load at bus 12 and report the bus "
    "voltage at that bus after power flow."
)
EXPLAIN_LINE_TRIP_VOLTAGE_PROMPT = (
    "Explain why the voltage distribution barely changes after the line "
    "trip on a line in the IEEE 39 network."
)

# A response with several patterns that the normalizer is expected to repair:
#   - andes.load("<xlsx>") without andes.get_case() wrapping
#   - ssa.add(model=..., param_dict=...) -> ssa.add("<model>", param_dict=...)
#   - ssa.Bus.v.vn attribute path that must collapse to ssa.Bus.v.v
#   - setup=False followed by explicit ssa.setup()
NORMALIZABLE_RESPONSE = (
    "```python\n"
    "import andes\n"
    "\n"
    "ssa = andes.load(\"kundur/kundur_full.xlsx\", setup=False, no_output=True, log=False)\n"
    "ssa.add(model=\"PQ\", param_dict={\"bus\": 8, \"idx\": \"PQ_NEW_1\", \"p0\": 0.01, \"q0\": 0.01})\n"
    "ssa.setup()\n"
    "ssa.PFlow.run()\n"
    "print(ssa.Bus.v.vn[0])\n"
    "```"
)

VALID_RUNNABLE_RESPONSE = (
    "```python\n"
    "import andes\n"
    "import json\n"
    "ssa = andes.load(andes.get_case('ieee14/ieee14.raw'), setup=True, no_output=True, log=False)\n"
    "ssa.PFlow.run()\n"
    "print('RESULT_JSON:' + json.dumps({'bus_voltages': list(ssa.Bus.v.v)}))\n"
    "```"
)

UNITTEST_RESPONSE = (
    "```python\n"
    "import unittest\n"
    "\n"
    "class Demo(unittest.TestCase):\n"
    "    def test_nothing(self):\n"
    "        self.assertTrue(True)\n"
    "```"
)


# Snapshot case registry ----------------------------------------------------
# Each entry: (case_id, callable, args_tuple) -> deterministic output.
# Output types accepted: str, bool, list[str], tuple of the former.

SnapshotCase = Tuple[str, Callable[..., Any], Tuple[Any, ...]]

CASES: Dict[str, List[SnapshotCase]] = {
    "is_code_only_request": [
        ("code_only_explicit", is_code_only_request, (CODE_ONLY_CTX,)),
        ("explanation_request", is_code_only_request, (EXPLAIN_CTX,)),
        ("empty", is_code_only_request, ("",)),
    ],
    "is_explanatory_followup_request": [
        ("explain_result", is_explanatory_followup_request, (EXPLAIN_CTX,)),
        ("code_only", is_explanatory_followup_request, (CODE_ONLY_CTX,)),
        ("walk_me_through", is_explanatory_followup_request,
         ("Walk me through what the script did step by step.",)),
    ],
    "extract_effective_user_context": [
        ("plain_text", extract_effective_user_context,
         ("Run a power flow on the IEEE 14 bus case.",)),
        ("empty", extract_effective_user_context, ("",)),
    ],
    "infer_requested_builtin_case": [
        ("ieee14_explicit", infer_requested_builtin_case,
         ("Please use the IEEE 14-bus case.",)),
        ("kundur_explicit", infer_requested_builtin_case,
         ("Use the Kundur two-area system.",)),
        ("ieee39", infer_requested_builtin_case,
         ("Run power flow on the IEEE 39 bus system.",)),
        ("none", infer_requested_builtin_case,
         ("Run a power flow analysis.",)),
    ],
    "extract_python_code_blocks": [
        ("single_block", extract_python_code_blocks,
         ("Preamble\n```python\nimport andes\nprint(1)\n```\nTail",)),
        ("no_block", extract_python_code_blocks, ("Just plain text without code.",)),
        ("two_blocks", extract_python_code_blocks,
         ("```python\na = 1\n```\n\n```python\nb = 2\n```",)),
    ],
    "validate_response_code": [
        ("valid_runnable", validate_response_code,
         (VALID_RUNNABLE_RESPONSE, CODE_ONLY_CTX)),
        ("unittest_rejected", validate_response_code,
         (UNITTEST_RESPONSE, CODE_ONLY_CTX)),
        ("no_code_block", validate_response_code,
         ("No python block here at all.", CODE_ONLY_CTX)),
    ],
    "normalize_andes_response": [
        ("common_patterns", normalize_andes_response, (NORMALIZABLE_RESPONSE, "")),
        ("already_clean", normalize_andes_response, (VALID_RUNNABLE_RESPONSE, "")),
    ],
    "build_andes_fallback_response": [
        # Non-matching prompt -> empty string (negative-path coverage).
        ("unmatched_prompt", build_andes_fallback_response, (IEEE14_PROMPT,)),
        # Matches the IEEE-14 slack/top-3 template branch.
        ("ieee14_slack_top3", build_andes_fallback_response,
         (IEEE14_SLACK_TOP3_PROMPT,)),
        # Matches the IEEE-39 PQ-at-bus template branch (parametric).
        ("ieee39_pq_bus", build_andes_fallback_response, (IEEE39_PQ_BUS_PROMPT,)),
    ],
    "build_andes_explanation_fallback_response": [
        # Non-matching prompt -> empty string.
        ("unmatched_prompt", build_andes_explanation_fallback_response,
         ("Explain the voltage results from the previous run.",)),
        # Matches the "line trip with unchanged voltage distribution" branch.
        ("line_trip_voltage", build_andes_explanation_fallback_response,
         (EXPLAIN_LINE_TRIP_VOLTAGE_PROMPT,)),
    ],
}


# Serialization helpers -----------------------------------------------------

def _normalize_for_json(value: Any) -> Any:
    """Convert tuples to lists so JSON round-trips deterministically."""
    if isinstance(value, tuple):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, list):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _serialize_case(func_name: str, case_id: str, args: Tuple[Any, ...], output: Any) -> Dict[str, Any]:
    return {
        "func": func_name,
        "case_id": case_id,
        "args": _normalize_for_json(args),
        "output": _normalize_for_json(output),
    }


def _load_snapshots() -> Dict[str, Dict[str, Any]]:
    if not SNAPSHOT_PATH.exists():
        return {}
    with SNAPSHOT_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return {f"{entry['func']}::{entry['case_id']}": entry for entry in data}


def _write_snapshots(entries: List[Dict[str, Any]]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries, key=lambda e: (e["func"], e["case_id"]))
    with SNAPSHOT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")


# Test cases ----------------------------------------------------------------

class RagChatbotSnapshotTests(unittest.TestCase):
    """One test method per pinned function; each iterates its fixture cases."""

    maxDiff = None

    @classmethod
    def setUpClass(cls) -> None:
        cls._existing = _load_snapshots()
        cls._produced: List[Dict[str, Any]] = []

    @classmethod
    def tearDownClass(cls) -> None:
        if UPDATE_MODE:
            _write_snapshots(cls._produced)
            print(
                f"\n[snapshots] wrote {len(cls._produced)} entries to {SNAPSHOT_PATH}",
                file=sys.stderr,
            )

    def _run_case(self, func_name: str, case_id: str, func: Callable[..., Any], args: Tuple[Any, ...]) -> None:
        actual = func(*args)
        entry = _serialize_case(func_name, case_id, args, actual)
        key = f"{func_name}::{case_id}"

        if UPDATE_MODE:
            type(self)._produced.append(entry)
            return

        if not self._existing:
            self.fail(
                f"Snapshot file missing at {SNAPSHOT_PATH}. "
                "Bootstrap it with UPDATE_SNAPSHOTS=1."
            )
        self.assertIn(
            key, self._existing,
            msg=f"No saved snapshot for {key}. Run with UPDATE_SNAPSHOTS=1 to add it.",
        )
        expected_output = self._existing[key]["output"]
        self.assertEqual(
            _normalize_for_json(actual),
            expected_output,
            msg=(
                f"Snapshot mismatch for {key}. "
                f"If this change is intentional, re-run with UPDATE_SNAPSHOTS=1."
            ),
        )


def _install_tests() -> None:
    """Dynamically attach one test method per (func, case) pair."""
    for func_name, cases in CASES.items():
        for case_id, func, args in cases:
            def make(func_name=func_name, case_id=case_id, func=func, args=args):
                def test(self):
                    self._run_case(func_name, case_id, func, args)
                return test
            method_name = f"test_{func_name}__{case_id}"
            setattr(RagChatbotSnapshotTests, method_name, make())


_install_tests()


if __name__ == "__main__":
    unittest.main()
