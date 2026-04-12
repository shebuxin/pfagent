"""Per-replacer unit tests for src.andes_code.normalizer.normalize_andes_code_block.

The existing snapshot tests cover ~2 broad cases, which catch drift but
don't pinpoint which replacer broke. These tests exercise each regex /
string-replacement group in isolation with the minimum input needed to
trigger it, and verify:

1. The transformation produces the expected output.
2. The ``notes`` list contains a recognizable marker for that replacer
   (so agent-evolution logs stay attributable).

When a future change in normalizer.py breaks one replacer, exactly one
test here fails -- not 25 at once via a snapshot miss.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.andes_code.normalizer import normalize_andes_code_block  # noqa: E402


class ApiCasingTests(unittest.TestCase):
    def test_uppercase_andes_becomes_lowercase(self):
        out, notes = normalize_andes_code_block("import ANDES\nANDES.load(...)")
        self.assertIn("import andes", out)
        self.assertIn("andes.load", out)
        self.assertNotIn("ANDES", out)
        self.assertTrue(any("lowercase `andes`" in n for n in notes))

    def test_lowercase_pflow_becomes_PFlow(self):
        out, notes = normalize_andes_code_block("ssa.pflow.run()")
        self.assertIn(".PFlow", out)
        self.assertNotIn(".pflow", out)
        self.assertTrue(any("PFlow" in n for n in notes))


class NumpyFilesystemHelperTests(unittest.TestCase):
    def test_np_os_rewritten_to_os(self):
        out, notes = normalize_andes_code_block("path = np.os.getcwd()")
        self.assertIn("os.getcwd()", out)
        self.assertNotIn("np.os.", out)
        self.assertTrue(any("NumPy filesystem" in n for n in notes))

    def test_np_path_rewritten_to_os_path(self):
        out, _ = normalize_andes_code_block("p = np.path.join('a', 'b')")
        self.assertIn("os.path.join", out)
        self.assertNotIn("np.path.", out)


class NonPythonCommentTests(unittest.TestCase):
    def test_c_style_block_comment_stripped(self):
        out, notes = normalize_andes_code_block("x = 1 /* not python */ + 2")
        self.assertNotIn("/*", out)
        self.assertNotIn("*/", out)
        self.assertTrue(any("non-Python comment" in n for n in notes))

    def test_slash_slash_line_comment_becomes_hash(self):
        out, _ = normalize_andes_code_block("// this is C++ style\nx = 1")
        self.assertIn("#", out)
        self.assertNotIn("//", out)


class BusAttributeNormalizationTests(unittest.TestCase):
    def test_bus_v_vn_collapses_to_bus_v_v(self):
        out, notes = normalize_andes_code_block("print(ssa.Bus.v.vn[0])")
        self.assertIn("Bus.v.v[0]", out)
        self.assertNotIn("Bus.v.vn", out)
        self.assertTrue(any("Bus.v.vn" in n for n in notes))

    def test_bus_v_mag_collapses_to_bus_v_v(self):
        out, notes = normalize_andes_code_block("v = ssa.Bus.v.mag")
        self.assertIn("Bus.v.v", out)
        self.assertNotIn("Bus.v.mag", out)
        self.assertTrue(any("Bus.v.mag" in n for n in notes))

    def test_bus_v_expanded_to_bus_v_v(self):
        out, notes = normalize_andes_code_block("x = ssa.Bus.v")
        self.assertIn("ssa.Bus.v.v", out)
        self.assertTrue(any("ssa.Bus.v.v" in n for n in notes))

    def test_bus_idx_expanded_to_bus_idx_v(self):
        out, notes = normalize_andes_code_block("ids = ssa.Bus.idx")
        self.assertIn("ssa.Bus.idx.v", out)
        self.assertTrue(any("ssa.Bus.idx.v" in n for n in notes))

    def test_already_bus_v_v_not_double_wrapped(self):
        out, _ = normalize_andes_code_block("x = ssa.Bus.v.v\n")
        self.assertNotIn("ssa.Bus.v.v.v", out)


class LineStatusAttributeTests(unittest.TestCase):
    def test_line_status_v_rewritten_to_line_u_v(self):
        out, notes = normalize_andes_code_block("flag = ssa.Line.status.v")
        self.assertIn(".Line.u.v", out)
        self.assertNotIn(".Line.status.v", out)
        self.assertTrue(any("in-service flag" in n for n in notes))

    def test_line_set_status_becomes_u(self):
        out, notes = normalize_andes_code_block(
            'ssa.Line.set(src="status", idx=["L1"], attr="v", value=[0])'
        )
        self.assertIn('.Line.set(src="u"', out)
        self.assertNotIn('src="status"', out)
        self.assertTrue(any('src="u"' in n for n in notes))

    def test_false_value_converted_to_zero_and_unwrapped(self):
        # [False] -> [0] (boolean fix) -> 0 (unwrap to scalar)
        out, notes = normalize_andes_code_block(
            'ssa.Line.set(src="u", idx=["L1"], attr="v", value=[False])'
        )
        self.assertIn("value=0", out)
        self.assertNotIn("[False]", out)
        self.assertNotIn("value=[", out)
        self.assertTrue(any("numeric in-service" in n for n in notes))

    def test_false_scalar_converted_to_zero(self):
        out, _ = normalize_andes_code_block(
            'ssa.Line.set(src="u", idx="L1", attr="v", value=False)'
        )
        self.assertIn("value=0", out)
        self.assertNotIn("value=False", out)

    def test_set_status_rewritten_to_backward_compatible_set(self):
        # ANDES 2.0+ only API -> backward-compat form with scalar args.
        out, notes = normalize_andes_code_block("ssa.Line.set_status(line_id, 0)")
        self.assertIn(
            'ssa.Line.set(src="u", idx=line_id, attr="v", value=0)', out
        )
        self.assertNotIn("set_status", out)
        self.assertTrue(any("backward-compatible" in n for n in notes))

    def test_set_status_with_subscripted_id_is_preserved(self):
        out, _ = normalize_andes_code_block(
            "ssa.Line.set_status(line_ids[0], 0)"
        )
        self.assertIn(
            'ssa.Line.set(src="u", idx=line_ids[0], attr="v", value=0)', out
        )

    def test_set_status_value_one_also_rewritten(self):
        # Restore-in-service path ("set_status(..., 1)").
        out, _ = normalize_andes_code_block(
            "contingency_ssa.Line.set_status(candidate_line, 1)"
        )
        self.assertIn(
            'contingency_ssa.Line.set(src="u", idx=candidate_line, attr="v", value=1)',
            out,
        )

    def test_list_wrapped_idx_and_value_unwrapped_to_scalar(self):
        # Model emits idx=[X], value=[0]; normalizer strips brackets.
        out, notes = normalize_andes_code_block(
            'ssa.Line.set(src="u", idx=[line_id], attr="v", value=[0])'
        )
        self.assertIn('idx=line_id', out)
        self.assertIn('value=0', out)
        self.assertNotIn('idx=[', out)
        self.assertNotIn('value=[', out)
        self.assertTrue(any("Unwrapped" in n for n in notes))


class PltStemCollectionTests(unittest.TestCase):
    def test_use_line_collection_flag_stripped(self):
        out, notes = normalize_andes_code_block(
            "plt.stem(x, y, use_line_collection=True)"
        )
        self.assertNotIn("use_line_collection=True", out)
        self.assertIn("plt.stem(x, y)", out)
        self.assertTrue(any("use_line_collection" in n for n in notes))


class ResultAccessorTests(unittest.TestCase):
    def test_bus_v_v_e_collapses(self):
        out, notes = normalize_andes_code_block("val = ssa.Bus.v.v.e")
        self.assertIn("ssa.Bus.v.v", out)
        self.assertNotIn(".Bus.v.v.e", out)
        self.assertTrue(any("accessors" in n for n in notes))

    def test_line_sn_e_becomes_line_sn_v(self):
        out, _ = normalize_andes_code_block("s = ssa.Line.Sn.e")
        self.assertIn(".Line.Sn.v", out)

    def test_bus_idx_v_name_stripped(self):
        out, _ = normalize_andes_code_block("n = ssa.Bus.idx.v.name")
        self.assertIn("ssa.Bus.idx.v", out)
        self.assertNotIn(".name", out)


class BranchFlowApiMappingTests(unittest.TestCase):
    def test_active_power_p1_mapped_to_a1_e(self):
        ctx = "Plot line active power flow."
        out, notes = normalize_andes_code_block("p = ssa.Line.p1.v", user_context=ctx)
        self.assertIn("ssa.Line.a1.e", out)
        self.assertTrue(any("active-power flow" in n for n in notes))

    def test_active_power_a1_v_mapped_to_a1_e(self):
        ctx = "Plot line active power flow."
        out, _ = normalize_andes_code_block("p = ssa.Line.a1.v", user_context=ctx)
        self.assertIn("ssa.Line.a1.e", out)
        self.assertNotIn("Line.a1.v", out)

    def test_reactive_power_q1_mapped_to_v1_e(self):
        ctx = "Plot line reactive power flow."
        out, notes = normalize_andes_code_block("q = ssa.Line.q1.v", user_context=ctx)
        self.assertIn("ssa.Line.v1.e", out)
        self.assertTrue(any("reactive-power flow" in n for n in notes))

    def test_no_branch_context_leaves_p1_alone(self):
        # Without branch-flow intent in user_context, we must not touch
        # it (someone might legitimately reference a different p1).
        out, _ = normalize_andes_code_block("p = ssa.Line.p1.v")
        self.assertIn("Line.p1.v", out)


class LineIdxNumericCastTests(unittest.TestCase):
    def test_numeric_cast_replaced_with_string_ids(self):
        code = "ids = np.asarray(ssa.Line.idx.v, dtype=int)"
        out, notes = normalize_andes_code_block(code)
        self.assertIn("[str(item) for item in np.asarray(ssa.Line.idx.v)]", out)
        self.assertNotIn("dtype=int", out)
        self.assertTrue(any("string line-device IDs" in n for n in notes))

    def test_float_cast_also_replaced(self):
        code = "ids = np.asarray(ssa.Line.idx.v, dtype=float)"
        out, _ = normalize_andes_code_block(code)
        self.assertIn("[str(item) for item in np.asarray(ssa.Line.idx.v)]", out)


class SlackBusAccessorTests(unittest.TestCase):
    def test_slack_bus_idx_v_collapses(self):
        out, notes = normalize_andes_code_block("s = ssa.Slack.bus.idx.v")
        self.assertIn("ssa.Slack.bus.v", out)
        self.assertNotIn(".Slack.bus.idx.v", out)
        self.assertTrue(any("Slack.bus.idx.v" in n for n in notes))

    def test_slack_bus_idx_direct_collapses(self):
        out, notes = normalize_andes_code_block("s = ssa.Slack.bus.idx")
        self.assertIn("ssa.Slack.bus.v", out)
        self.assertNotIn(".Slack.bus.idx", out)
        self.assertTrue(any("Slack.bus.idx" in n for n in notes))


class TypoRepairTests(unittest.TestCase):
    def test_buse_idx_typo_fixed(self):
        out, notes = normalize_andes_code_block("for buse_idx in bus_list: pass")
        self.assertIn("bus_idx", out)
        self.assertNotIn("buse_idx", out)
        self.assertTrue(any("buse_idx" in n for n in notes))


class PFlowRunReturnCodeTests(unittest.TestCase):
    def test_rc_assignment_to_bare_call(self):
        out, notes = normalize_andes_code_block("rc = ssa.PFlow.run()")
        self.assertIn("ssa.PFlow.run()", out)
        self.assertNotIn("rc = ", out)
        self.assertTrue(any("convergence code" in n for n in notes))

    def test_rc_raise_guard_stripped(self):
        code = "rc = ssa.PFlow.run()\nif rc != 0:\n    raise RuntimeError('x')\n"
        out, _ = normalize_andes_code_block(code)
        self.assertNotIn("raise RuntimeError", out)


class AddModelKwargTests(unittest.TestCase):
    def test_add_model_kwarg_rewritten_to_positional(self):
        out, notes = normalize_andes_code_block(
            'ssa.add(model="PQ", param_dict={"bus": 1})'
        )
        self.assertIn('ssa.add("PQ", param_dict=', out)
        self.assertNotIn('model="PQ"', out)
        self.assertTrue(any(".add(model=...)" in n for n in notes))


class PlotVoltageReplacementTests(unittest.TestCase):
    def test_pflow_plot_voltage_replaced_with_matplotlib(self):
        out, notes = normalize_andes_code_block("ssa.PFlow.plot_voltage()")
        self.assertNotIn("plot_voltage()", out)
        self.assertIn("plt.plot(", out)
        self.assertIn("ssa.Bus.v.v", out)
        self.assertIn("import matplotlib.pyplot as plt", out)
        self.assertTrue(any("plot_voltage()" in n for n in notes))


class TDSPlotterReplacementTests(unittest.TestCase):
    def test_tds_plotter_and_run_replaced(self):
        code = "ssa.TDS.run()\nssa.TDS.plotter.plot(ssa.Bus.v)"
        out, notes = normalize_andes_code_block(code)
        self.assertNotIn(".TDS.run()", out)
        self.assertNotIn(".TDS.plotter.plot(", out)
        self.assertIn("plt.plot(bus_id, bus_v", out)
        self.assertTrue(any("TDS plotting" in n for n in notes))


class VoltageProfileBarToLineTests(unittest.TestCase):
    def test_bar_becomes_plot_when_voltage_profile_requested(self):
        ctx = "Generate a voltage profile plot."
        out, notes = normalize_andes_code_block(
            "plt.bar(bus_ids, bus_v)", user_context=ctx
        )
        self.assertIn("plt.plot(bus_ids, bus_v)", out)
        self.assertTrue(any("line-style voltage profile" in n for n in notes))

    def test_bar_kept_without_voltage_profile_context(self):
        out, _ = normalize_andes_code_block("plt.bar(bus_ids, bus_v)")
        self.assertIn("plt.bar(", out)


class ImportInjectionTests(unittest.TestCase):
    def test_os_import_added_when_os_getcwd_used(self):
        out, _ = normalize_andes_code_block("p = os.getcwd()")
        self.assertIn("import os", out)

    def test_numpy_import_added_when_np_asarray_used(self):
        out, _ = normalize_andes_code_block("x = ssa.Bus.v.v\n")
        # This test assumes the _wrap_simple_rhs_with_numpy replacer
        # fired, which inserted np.asarray(...), which pulls in the np
        # import.
        self.assertIn("np.asarray", out)
        self.assertIn("import numpy as np", out)


class AddBeforeSetupWorkflowTests(unittest.TestCase):
    def test_setup_true_becomes_false_with_add_call(self):
        ctx = "Add one new PQ load before setup, then run power flow."
        code = (
            "ssa = andes.load(andes.get_case('kundur/kundur_full.xlsx'), setup=True)\n"
            "ssa.add('PQ', param_dict={'bus': 1, 'idx': 'X', 'p0': 0.1, 'q0': 0.1})\n"
            "ssa.PFlow.run()\n"
        )
        out, notes = normalize_andes_code_block(code, user_context=ctx)
        self.assertIn("setup=False", out)
        self.assertNotIn("setup=True", out)
        self.assertTrue(any("add-before-setup" in n for n in notes))

    def test_setup_call_inserted_before_pflow_run(self):
        ctx = "Add one new PQ load before setup, then run power flow."
        code = (
            "ssa = andes.load(andes.get_case('kundur/kundur_full.xlsx'), setup=True)\n"
            "ssa.add('PQ', param_dict={'bus': 1, 'idx': 'X', 'p0': 0.1, 'q0': 0.1})\n"
            "ssa.PFlow.run()\n"
        )
        out, _ = normalize_andes_code_block(code, user_context=ctx)
        self.assertIn("ssa.setup()", out)
        # Setup should appear before PFlow.run in the code
        setup_pos = out.find("ssa.setup()")
        pflow_pos = out.find("ssa.PFlow.run()")
        self.assertGreaterEqual(setup_pos, 0)
        self.assertGreater(pflow_pos, setup_pos)


class NumpyWrapperTests(unittest.TestCase):
    def test_bus_voltage_rhs_wrapped_with_asarray_float(self):
        out, notes = normalize_andes_code_block("voltages = ssa.Bus.v.v\n")
        self.assertIn("np.asarray(ssa.Bus.v.v, dtype=float)", out)
        self.assertTrue(any("np.asarray" in n for n in notes))

    def test_bus_idx_rhs_wrapped_with_asarray_int(self):
        out, notes = normalize_andes_code_block("ids = ssa.Bus.idx.v\n")
        self.assertIn("np.asarray(ssa.Bus.idx.v, dtype=int)", out)
        self.assertTrue(any("np.asarray" in n for n in notes))

    def test_line_idx_rhs_wrapped_with_str_items(self):
        out, _ = normalize_andes_code_block("names = ssa.Line.idx.v\n")
        self.assertIn("[str(item) for item in np.asarray(ssa.Line.idx.v)]", out)


class IdempotenceTests(unittest.TestCase):
    def test_already_clean_code_is_untouched(self):
        code = (
            "import andes\n"
            "ssa = andes.load(andes.get_case('ieee14/ieee14.raw'), setup=True, no_output=True, log=False)\n"
            "ssa.PFlow.run()\n"
            "print(ssa.Bus.v.v[0])\n"
        )
        out, notes = normalize_andes_code_block(code)
        # Some lightweight notes may fire (e.g. bus.v -> bus.v.v
        # expansion if there's a bare Bus.v anywhere); but the output
        # should not change meaningfully.
        self.assertIn("andes.get_case('ieee14/ieee14.raw')", out)
        self.assertIn("ssa.PFlow.run()", out)
        self.assertIn("ssa.Bus.v.v[0]", out)

    def test_normalizer_is_idempotent(self):
        # A second pass should not change what a first pass produced.
        code = "ssa.add(model='PQ', param_dict={'bus': 1})\nval = ssa.Bus.v"
        once, _ = normalize_andes_code_block(code)
        twice, _ = normalize_andes_code_block(once)
        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
