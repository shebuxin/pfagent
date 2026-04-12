"""Per-rule unit tests for src.andes_code.validators.

validate_andes_case_loading bundles ~40 individual ANDES API rules.
The snapshot tests catch broad drift; these tests pin each rule group
so a future prompt/API change fails exactly the test that matches.

Rule tests check both sides:
- the asserting side: a violating snippet triggers an error message
  that mentions the specific guardrail
- the non-firing side: a clean snippet does NOT trigger that rule
  (to guard against false-positive over-reach)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.andes_code.validators import (  # noqa: E402
    check_python_code_compilation,
    validate_andes_case_loading,
    validate_response_code,
)


# --- check_python_code_compilation --------------------------------------

class CheckPythonCodeCompilationTests(unittest.TestCase):
    def test_valid_code_returns_ok(self):
        ok, msg = check_python_code_compilation("x = 1\nprint(x)\n")
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_syntax_error_returns_helpful_message(self):
        ok, msg = check_python_code_compilation("def :\n    return 1")
        self.assertFalse(ok)
        self.assertIn("Syntax Error", msg)
        self.assertIn("line 1", msg)

    def test_empty_code_is_valid(self):
        ok, msg = check_python_code_compilation("")
        self.assertTrue(ok)


# --- validate_response_code ---------------------------------------------

class ValidateResponseCodeTests(unittest.TestCase):
    def _valid_block(self, code: str) -> str:
        return f"```python\n{code}\n```"

    def test_code_only_prompt_without_fenced_block_fails(self):
        ok, errors = validate_response_code(
            "some plain text without a fence",
            user_context="Return runnable Python code only.",
        )
        self.assertFalse(ok)
        self.assertTrue(errors)

    def test_non_code_prompt_with_no_blocks_is_valid(self):
        ok, errors = validate_response_code("here is my explanation", user_context="")
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_valid_block_passes(self):
        # Use the exact case path that infer_requested_builtin_case
        # resolves to for "IEEE 14" prompts (ieee14/ieee14_full.xlsx),
        # otherwise the prompt-matching rule fires.
        body = self._valid_block(
            "import andes\n"
            "ssa = andes.load(andes.get_case('ieee14/ieee14_full.xlsx'), setup=True, no_output=True, log=False)\n"
            "ssa.PFlow.run()\n"
        )
        ok, errors = validate_response_code(body, user_context="Generate runnable Python code only. Use the IEEE 14 built-in case.")
        self.assertTrue(ok, f"should pass, got errors={errors}")


# --- Case-loading rules --------------------------------------------------

class AndesPackageNameRules(unittest.TestCase):
    def test_anodes_typo_flagged(self):
        errors = validate_andes_case_loading("import anodes\n")
        self.assertTrue(any("Use 'andes' package, not 'anodes'" in e for e in errors))

    def test_uppercase_ANDES_flagged(self):
        errors = validate_andes_case_loading("import ANDES\nANDES.load('x')")
        self.assertTrue(any("Use lowercase `andes`" in e for e in errors))

    def test_lowercase_andes_is_clean(self):
        errors = validate_andes_case_loading("import andes\nandes.load('x')")
        self.assertFalse(any("Use lowercase `andes`" in e for e in errors))


class GetCasePathRules(unittest.TestCase):
    def test_invalid_get_case_path_yields_suggestions(self):
        errors = validate_andes_case_loading(
            'ssa = andes.load(andes.get_case("ieee14/typo.raw"), setup=True)'
        )
        self.assertTrue(
            any("not a valid ANDES built-in case path" in e for e in errors),
            msg=f"expected an invalid-path error, got {errors}",
        )

    def test_known_builtin_case_path_is_clean(self):
        # ieee14/ieee14.raw is a real ANDES built-in.
        errors = validate_andes_case_loading(
            'ssa = andes.load(andes.get_case("ieee14/ieee14.raw"), setup=True)'
        )
        # Some other rules may fire, but the invalid-path one must not.
        self.assertFalse(any("not a valid ANDES built-in case path" in e for e in errors))


class BusAccessorRules(unittest.TestCase):
    def test_bus_v_vn_flagged(self):
        errors = validate_andes_case_loading("v = ssa.Bus.v.vn")
        self.assertTrue(any("Bus.v.v" in e and "Bus.v.vn" in e for e in errors))

    def test_bus_v_mag_flagged(self):
        errors = validate_andes_case_loading("v = ssa.Bus.v.mag")
        self.assertTrue(any("Bus.v.mag" in e for e in errors))

    def test_bare_bus_v_flagged(self):
        errors = validate_andes_case_loading("v = ssa.Bus.v")
        self.assertTrue(any("Use `ssa.Bus.v.v`, not `ssa.Bus.v`" in e for e in errors))

    def test_bare_bus_idx_flagged(self):
        errors = validate_andes_case_loading("ids = ssa.Bus.idx")
        self.assertTrue(any("Use `ssa.Bus.idx.v`, not `ssa.Bus.idx`" in e for e in errors))

    def test_bus_v_v_is_clean(self):
        errors = validate_andes_case_loading("v = ssa.Bus.v.v")
        self.assertFalse(any("Bus.v.v`, not `ssa.Bus.v`" in e for e in errors))

    def test_iterate_over_bus_idx_without_v_flagged(self):
        errors = validate_andes_case_loading("for b in ssa.Bus.idx:\n    pass")
        self.assertTrue(any("Iterate over `ssa.Bus.idx.v`" in e for e in errors))


class AddCallRules(unittest.TestCase):
    def test_add_model_kwarg_flagged(self):
        errors = validate_andes_case_loading('ssa.add(model="PQ", param_dict={"bus": 1})')
        self.assertTrue(any("ssa.add(model=" in e for e in errors))

    def test_pq_using_p_kwarg_flagged(self):
        errors = validate_andes_case_loading(
            'ssa.add("PQ", param_dict={"bus": 1, p=0.1, q=0.1})'
        )
        self.assertTrue(any("p0=` and `q0=" in e for e in errors))

    def test_pq_with_p0_q0_is_clean(self):
        errors = validate_andes_case_loading(
            'ssa.add("PQ", param_dict={"bus": 1, "p0": 0.1, "q0": 0.1})'
        )
        self.assertFalse(any("p0=` and `q0=" in e for e in errors))


class SlackAccessorRules(unittest.TestCase):
    def test_bus_slack_attr_flagged(self):
        errors = validate_andes_case_loading("s = ssa.Bus.slack")
        self.assertTrue(any("Bus.slack" in e and "not a valid" in e for e in errors))

    def test_slack_bus_idx_flagged(self):
        errors = validate_andes_case_loading("s = ssa.Slack.bus.idx")
        self.assertTrue(any("ssa.Slack.bus.v[0]" in e for e in errors))


class CaseStyleRules(unittest.TestCase):
    def test_lowercase_pflow_flagged(self):
        errors = validate_andes_case_loading("ssa.pflow.run()")
        self.assertTrue(any("ssa.PFlow`, not `ssa.pflow" in e for e in errors))

    def test_plot_voltage_flagged(self):
        errors = validate_andes_case_loading("ssa.PFlow.plot_voltage()")
        self.assertTrue(any("plot_voltage()" in e for e in errors))

    def test_private_numpy_helper_flagged(self):
        errors = validate_andes_case_loading("x = np._something(1, 2)")
        self.assertTrue(any("np._" in e for e in errors))


class BuiltinCaseEnforcementRules(unittest.TestCase):
    def test_builtin_prompt_without_get_case_flagged(self):
        ctx = "Use the IEEE 14 built-in case."
        errors = validate_andes_case_loading(
            'ssa = andes.load("ieee14/ieee14.raw", setup=True)', user_context=ctx,
        )
        self.assertTrue(any("andes.get_case" in e for e in errors))

    def test_wrong_builtin_case_flagged(self):
        ctx = "Use the IEEE 39 built-in case."
        errors = validate_andes_case_loading(
            'ssa = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True)',
            user_context=ctx,
        )
        self.assertTrue(
            any("prompt asks for the built-in case" in e for e in errors),
            msg=f"got errors={errors}",
        )

    def test_matching_builtin_case_is_clean(self):
        ctx = "Use the IEEE 14 built-in case."
        errors = validate_andes_case_loading(
            'ssa = andes.load(andes.get_case("ieee14/ieee14_full.xlsx"), setup=True)',
            user_context=ctx,
        )
        self.assertFalse(any("prompt asks for the built-in case" in e for e in errors))


class SetupWorkflowRules(unittest.TestCase):
    def test_setup_false_without_setup_call_flagged(self):
        errors = validate_andes_case_loading(
            "ssa = andes.load(x, setup=False)\nssa.PFlow.run()"
        )
        self.assertTrue(any("When using `setup=False`" in e for e in errors))

    def test_setup_false_with_setup_call_is_clean(self):
        errors = validate_andes_case_loading(
            "ssa = andes.load(x, setup=False)\nssa.setup()\nssa.PFlow.run()"
        )
        self.assertFalse(any("When using `setup=False`" in e for e in errors))

    def test_before_setup_with_setup_true_flagged(self):
        ctx = "Add one new PQ load before setup, then run power flow."
        errors = validate_andes_case_loading(
            "ssa = andes.load(x, setup=True)\nssa.add('PQ', param_dict={})\nssa.setup()",
            user_context=ctx,
        )
        self.assertTrue(any("load the case with `setup=False`" in e for e in errors))


class TestScaffoldingRules(unittest.TestCase):
    def test_unittest_in_code_only_flagged(self):
        ctx = "Generate runnable Python code only."
        errors = validate_andes_case_loading(
            "import unittest\nclass T(unittest.TestCase):\n    pass",
            user_context=ctx,
        )
        self.assertTrue(any("unittest/pytest scaffolding" in e for e in errors))

    def test_pytest_in_code_only_flagged(self):
        ctx = "Generate runnable Python code only."
        errors = validate_andes_case_loading("import pytest", user_context=ctx)
        self.assertTrue(any("unittest/pytest" in e for e in errors))

    def test_unittest_allowed_outside_code_only(self):
        errors = validate_andes_case_loading("import unittest", user_context="")
        self.assertFalse(any("unittest/pytest" in e for e in errors))


class PlotPromptRules(unittest.TestCase):
    def test_plot_prompt_without_plot_call_flagged(self):
        ctx = "Plot the voltage magnitudes."
        errors = validate_andes_case_loading("print('hi')", user_context=ctx)
        self.assertTrue(any("actually create a plot" in e for e in errors))

    def test_plot_prompt_with_plot_call_is_clean(self):
        ctx = "Plot the voltage magnitudes."
        errors = validate_andes_case_loading(
            "import matplotlib.pyplot as plt\nplt.plot([1,2])", user_context=ctx,
        )
        self.assertFalse(any("actually create a plot" in e for e in errors))


class SlackPromptRules(unittest.TestCase):
    def test_slack_bus_prompt_without_Slack_model_flagged(self):
        ctx = "Report the slack bus voltage."
        errors = validate_andes_case_loading("print(ssa.Bus.v.v[0])", user_context=ctx)
        self.assertTrue(any("ANDES Slack model" in e for e in errors))


class LineAngleRules(unittest.TestCase):
    def test_line_angle_prompt_without_a1_e_flagged(self):
        ctx = "Show line angles."
        errors = validate_andes_case_loading(
            "print(ssa.Bus.a.v)", user_context=ctx,
        )
        self.assertTrue(any("ssa.Line.a1.e" in e for e in errors))


class LineStatusRules(unittest.TestCase):
    def test_line_status_accessor_flagged(self):
        errors = validate_andes_case_loading("flag = ssa.Line.status")
        self.assertTrue(any("does not expose `status`" in e for e in errors))

    def test_line_set_status_src_flagged(self):
        errors = validate_andes_case_loading(
            'ssa.Line.set(src="status", idx=["L1"], attr="v", value=[0])'
        )
        self.assertTrue(any('src="u"' in e for e in errors))


class StemCollectionRules(unittest.TestCase):
    def test_use_line_collection_flag_flagged(self):
        errors = validate_andes_case_loading(
            "plt.stem(x, y, use_line_collection=True)"
        )
        self.assertTrue(any("use_line_collection" in e for e in errors))


class LineIdxNumericCastRules(unittest.TestCase):
    def test_line_idx_int_cast_flagged(self):
        errors = validate_andes_case_loading(
            "ids = np.asarray(ssa.Line.idx.v, dtype=int)"
        )
        self.assertTrue(any("Line.idx.v" in e and "string device IDs" in e for e in errors))

    def test_line_idx_str_wrap_is_clean(self):
        errors = validate_andes_case_loading(
            "ids = [str(item) for item in np.asarray(ssa.Line.idx.v)]"
        )
        self.assertFalse(any("string device IDs" in e for e in errors))


class N1ContingencyRules(unittest.TestCase):
    def test_line_outage_without_set_call_flagged(self):
        ctx = "Trip one line, then rerun power flow."
        code = "import andes\nssa.Line.idx.v\nssa.PFlow.run()"
        errors = validate_andes_case_loading(code, user_context=ctx)
        self.assertTrue(
            any("should actually open a line" in e for e in errors)
            or any("Line.set_status" in e for e in errors),
            msg=f"expected trip-line error; got {errors}",
        )

    def test_n1_contingency_status_markers_flagged(self):
        # N-1 prompt with almost no contingency-status tracking markers
        # should trigger the markers rule.
        ctx = "Run an N-1 contingency screen on every line."
        code = (
            "for l in ssa.Line.idx.v:\n"
            "    ssa.Line.set_status(l, 0)\n"
            "    ssa.PFlow.run()\n"
        )
        errors = validate_andes_case_loading(code, user_context=ctx)
        self.assertTrue(
            any("post-contingency convergence and islanding" in e for e in errors),
            msg=f"expected contingency-status error; got {errors}",
        )


class BranchFlowRules(unittest.TestCase):
    def test_active_power_branch_p1_usage_flagged(self):
        ctx = "Plot line active power flow."
        errors = validate_andes_case_loading("p = ssa.Line.p1.e", user_context=ctx)
        self.assertTrue(any("p1` / `p2" in e for e in errors))

    def test_active_power_branch_no_a1e_flagged(self):
        ctx = "Plot line active power flow."
        errors = validate_andes_case_loading("p = 0", user_context=ctx)
        self.assertTrue(any("ssa.Line.a1.e`" in e or "ssa.Line.a2.e" in e for e in errors))

    def test_reactive_power_branch_q1_usage_flagged(self):
        ctx = "Plot line reactive power flow."
        errors = validate_andes_case_loading("q = ssa.Line.q1.v", user_context=ctx)
        self.assertTrue(any("q1` / `q2" in e for e in errors))


class RankingKeywordRules(unittest.TestCase):
    def test_top_3_without_sorting_flagged(self):
        ctx = "Report top-3 highest bus voltages."
        errors = validate_andes_case_loading("print(ssa.Bus.v.v)", user_context=ctx)
        self.assertTrue(any("three ranked results" in e for e in errors))

    def test_top_2_without_sorting_flagged(self):
        ctx = "Report top-2 highest bus voltages."
        errors = validate_andes_case_loading("print(ssa.Bus.v.v)", user_context=ctx)
        self.assertTrue(any("two ranked results" in e for e in errors))

    def test_argsort_present_satisfies_ranking_rule(self):
        ctx = "Report top-3 highest bus voltages."
        errors = validate_andes_case_loading(
            "import numpy as np\nidx = np.argsort(ssa.Bus.v.v)[-3:]", user_context=ctx,
        )
        self.assertFalse(any("three ranked results" in e for e in errors))

    def test_count_prompt_without_len_flagged(self):
        ctx = "Count the number of PQ loads."
        errors = validate_andes_case_loading("print('hi')", user_context=ctx)
        self.assertTrue(any("compute and print an explicit count" in e for e in errors))


class ModificationIdxResolutionRules(unittest.TestCase):
    def test_pq_set_at_bus_without_idx_lookup_flagged(self):
        ctx = "Modify the PQ load at bus 8."
        errors = validate_andes_case_loading(
            'ssa.PQ.set(src="p0", idx=["PQ_1"], attr="v", value=[0.2])',
            user_context=ctx,
        )
        self.assertTrue(
            any("resolve the real device idx" in e for e in errors),
            msg=f"expected idx-resolution error; got {errors}",
        )

    def test_pq_set_with_idx_lookup_is_clean(self):
        ctx = "Modify the PQ load at bus 8."
        errors = validate_andes_case_loading(
            "buses = ssa.PQ.bus.v\n"
            "ids = ssa.PQ.idx.v\n"
            'ssa.PQ.set(src="p0", idx=[ids[0]], attr="v", value=[0.2])',
            user_context=ctx,
        )
        self.assertFalse(any("resolve the real device idx" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
