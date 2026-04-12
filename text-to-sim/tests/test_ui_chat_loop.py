"""Unit tests for build_contextual_user_input (from src.ui_chat_loop).

Before Stage 5, the prompt-composition logic was inline in main.py's
chat submission handler and had zero test coverage. Extracting it as
a pure function enabled these tests, which pin:

- the exact wording of the three optional context blocks (runtime
  files, uploaded-case preview, ANDES continuity)
- the block ordering (runtime files first, then preview, then
  continuity)
- the no-fire conditions (empty runtime_files, empty preview, missing
  active_case keys)
- the concatenation seams ("\\n\\n" between blocks)

These are part of the prompt contract the tuned model was trained
against, so any change here requires a matching prompt/regression run.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.ui_chat_loop import build_contextual_user_input  # noqa: E402


class NoContextTests(unittest.TestCase):
    def test_empty_everything_returns_user_input_unchanged(self):
        result = build_contextual_user_input(
            user_input="run power flow",
            runtime_files=[],
            runtime_data_dir="/tmp/x",
            uploaded_case_preview="",
            active_case=None,
        )
        self.assertEqual(result, "run power flow")

    def test_active_case_missing_keys_is_noop(self):
        # An active_case dict without source+value keys should add no
        # continuity block (mirrors the `active_source and active_value`
        # truthiness guard in the implementation).
        result = build_contextual_user_input(
            user_input="hi",
            runtime_files=[],
            runtime_data_dir="/tmp/x",
            uploaded_case_preview="",
            active_case={"source": "", "value": ""},
        )
        self.assertEqual(result, "hi")

    def test_active_case_non_dict_is_noop(self):
        # Guard clause: the implementation double-checks `isinstance(dict)`.
        result = build_contextual_user_input(
            user_input="hi",
            runtime_files=[],
            runtime_data_dir="/tmp/x",
            uploaded_case_preview="",
            active_case="not a dict",  # type: ignore[arg-type]
        )
        self.assertEqual(result, "hi")


class RuntimeFileBlockTests(unittest.TestCase):
    def test_runtime_files_produces_exact_wording(self):
        result = build_contextual_user_input(
            user_input="plot voltages",
            runtime_files=["ieee14.xlsx", "ieee39.xlsx"],
            runtime_data_dir="./code_executions/abc/data",
            uploaded_case_preview="",
            active_case=None,
        )
        self.assertIn("plot voltages\n\nRuntime file context:", result)
        self.assertIn(
            "- Working directory for execution: ./code_executions/abc/data",
            result,
        )
        self.assertIn("- ieee14.xlsx", result)
        self.assertIn("- ieee39.xlsx", result)
        self.assertIn("Use these filenames directly in generated Python code", result)

    def test_runtime_files_preserves_case_loading_rules(self):
        # The two case-loading rules are load-bearing prompt contract
        # for the tuned model (uploaded vs built-in case classification).
        result = build_contextual_user_input(
            user_input="hi",
            runtime_files=["case.xlsx"],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case=None,
        )
        self.assertIn(
            'if using an uploaded file, load it directly with andes.load("<exact_filename>", ...), and do NOT wrap it with andes.get_case(...)',
            result,
        )
        self.assertIn(
            "only use andes.get_case(...) for ANDES built-in benchmark cases",
            result,
        )

    def test_runtime_files_preserves_uploaded_case_template(self):
        # This exact template is what the model was tuned to emit for
        # uploaded-case loading.
        result = build_contextual_user_input(
            user_input="hi",
            runtime_files=["c.xlsx"],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case=None,
        )
        self.assertIn(
            'script_dir=os.getcwd(); case=os.path.join(script_dir, "<exact_filename>"); ssa=andes.load(case, setup=True, no_output=True, log=False)',
            result,
        )

    def test_empty_runtime_files_skips_block(self):
        result = build_contextual_user_input(
            user_input="hi",
            runtime_files=[],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case=None,
        )
        self.assertNotIn("Runtime file context:", result)


class UploadedCasePreviewTests(unittest.TestCase):
    def test_preview_appended_with_blank_line_separator(self):
        preview = "Selected ANDES case preview for network.xlsx:\n- Sheet Bus columns: idx, name"
        result = build_contextual_user_input(
            user_input="summarize",
            runtime_files=[],
            runtime_data_dir="/tmp",
            uploaded_case_preview=preview,
            active_case=None,
        )
        # Exactly one blank line between user_input and preview.
        self.assertEqual(result, f"summarize\n\n{preview}")

    def test_empty_preview_skips_block(self):
        result = build_contextual_user_input(
            user_input="x",
            runtime_files=[],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case=None,
        )
        self.assertEqual(result, "x")


class ContinuityBlockTests(unittest.TestCase):
    def test_continuity_exact_wording_builtin(self):
        result = build_contextual_user_input(
            user_input="re-plot",
            runtime_files=[],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case={"source": "builtin", "value": "ieee14/ieee14_full.xlsx"},
        )
        self.assertIn("ANDES continuity context:", result)
        self.assertIn("Last successfully executed case source: builtin", result)
        self.assertIn(
            "Last successfully executed case identifier: ieee14/ieee14_full.xlsx",
            result,
        )
        self.assertIn(
            "If the user is asking a follow-up (for example: plot/summarize/analyze)",
            result,
        )
        self.assertIn("reuse this same case", result)

    def test_continuity_uploaded_source(self):
        result = build_contextual_user_input(
            user_input="x",
            runtime_files=[],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case={"source": "uploaded", "value": "my_case.xlsx"},
        )
        self.assertIn("Last successfully executed case source: uploaded", result)
        self.assertIn("Last successfully executed case identifier: my_case.xlsx", result)

    def test_continuity_requires_both_source_and_value(self):
        # Only source set: no block.
        r1 = build_contextual_user_input(
            user_input="x", runtime_files=[], runtime_data_dir="/tmp",
            uploaded_case_preview="", active_case={"source": "builtin", "value": ""},
        )
        self.assertNotIn("ANDES continuity context:", r1)
        # Only value set: no block.
        r2 = build_contextual_user_input(
            user_input="x", runtime_files=[], runtime_data_dir="/tmp",
            uploaded_case_preview="", active_case={"source": "", "value": "x.xlsx"},
        )
        self.assertNotIn("ANDES continuity context:", r2)


class BlockOrderingTests(unittest.TestCase):
    def test_all_three_blocks_in_correct_order(self):
        result = build_contextual_user_input(
            user_input="run flow",
            runtime_files=["case.xlsx"],
            runtime_data_dir="/tmp/data",
            uploaded_case_preview="Selected ANDES case preview for case.xlsx:",
            active_case={"source": "uploaded", "value": "case.xlsx"},
        )
        # Order: raw input -> runtime files -> preview -> continuity.
        input_pos = result.find("run flow")
        runtime_pos = result.find("Runtime file context:")
        preview_pos = result.find("Selected ANDES case preview")
        continuity_pos = result.find("ANDES continuity context:")

        self.assertGreaterEqual(input_pos, 0)
        self.assertGreater(runtime_pos, input_pos)
        self.assertGreater(preview_pos, runtime_pos)
        self.assertGreater(continuity_pos, preview_pos)

    def test_only_runtime_and_continuity_no_preview(self):
        result = build_contextual_user_input(
            user_input="x",
            runtime_files=["a.xlsx"],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case={"source": "builtin", "value": "b.xlsx"},
        )
        self.assertIn("Runtime file context:", result)
        self.assertNotIn("Selected ANDES case preview", result)
        self.assertIn("ANDES continuity context:", result)

    def test_only_preview_and_continuity_no_runtime(self):
        result = build_contextual_user_input(
            user_input="x",
            runtime_files=[],
            runtime_data_dir="/tmp",
            uploaded_case_preview="Selected ANDES case preview for z.xlsx:",
            active_case={"source": "builtin", "value": "z.xlsx"},
        )
        self.assertNotIn("Runtime file context:", result)
        self.assertIn("Selected ANDES case preview for z.xlsx:", result)
        self.assertIn("ANDES continuity context:", result)


class CaseIdxInventoryBlockTests(unittest.TestCase):
    def test_inventory_appended_with_blank_line_separator(self):
        inventory = (
            "ANDES case idx inventory for ieee14/ieee14.raw:\n"
            "Line (20 entries):\n  idx  = [\"Line_1\", \"Line_2\"]"
        )
        result = build_contextual_user_input(
            user_input="trip line 18",
            runtime_files=[],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case=None,
            case_idx_inventory=inventory,
        )
        # Exactly one blank line between user_input and inventory.
        self.assertEqual(result, f"trip line 18\n\n{inventory}")

    def test_empty_inventory_skips_block(self):
        result = build_contextual_user_input(
            user_input="x",
            runtime_files=[],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case=None,
            case_idx_inventory="",
        )
        self.assertEqual(result, "x")

    def test_inventory_comes_after_continuity_block(self):
        # Ordering contract: raw -> runtime -> preview -> continuity ->
        # inventory. Changing this drifts prompt behavior.
        inventory = "ANDES case idx inventory for x.xlsx:\nLine (1 entry):"
        result = build_contextual_user_input(
            user_input="trip line 5",
            runtime_files=["x.xlsx"],
            runtime_data_dir="/tmp",
            uploaded_case_preview="Selected ANDES case preview for x.xlsx:",
            active_case={"source": "uploaded", "value": "x.xlsx"},
            case_idx_inventory=inventory,
        )
        runtime_pos = result.find("Runtime file context:")
        preview_pos = result.find("Selected ANDES case preview")
        continuity_pos = result.find("ANDES continuity context:")
        inventory_pos = result.find("ANDES case idx inventory for")

        self.assertGreaterEqual(runtime_pos, 0)
        self.assertGreater(preview_pos, runtime_pos)
        self.assertGreater(continuity_pos, preview_pos)
        self.assertGreater(inventory_pos, continuity_pos)

    def test_inventory_param_is_optional_default_empty(self):
        # Existing callers that don't pass case_idx_inventory continue
        # to work unchanged.
        result = build_contextual_user_input(
            user_input="hi",
            runtime_files=[],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case=None,
        )
        self.assertEqual(result, "hi")


class EdgeCaseTests(unittest.TestCase):
    def test_multiline_user_input_preserved(self):
        result = build_contextual_user_input(
            user_input="line 1\nline 2\nline 3",
            runtime_files=[],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case=None,
        )
        self.assertEqual(result, "line 1\nline 2\nline 3")

    def test_runtime_files_single_file(self):
        result = build_contextual_user_input(
            user_input="x",
            runtime_files=["only_one.xlsx"],
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case=None,
        )
        self.assertIn("- only_one.xlsx", result)

    def test_runtime_files_many_files_listed_with_dashes(self):
        files = [f"case_{i}.xlsx" for i in range(5)]
        result = build_contextual_user_input(
            user_input="x",
            runtime_files=files,
            runtime_data_dir="/tmp",
            uploaded_case_preview="",
            active_case=None,
        )
        for f in files:
            self.assertIn(f"- {f}", result)


if __name__ == "__main__":
    unittest.main()
