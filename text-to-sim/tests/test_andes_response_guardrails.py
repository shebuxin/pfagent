import sys
import unittest
from pathlib import Path


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.chatbots.openai.rag_chatbot import (
    build_andes_explanation_fallback_response,
    build_andes_fallback_response,
    extract_effective_user_context,
    infer_requested_builtin_case,
    is_explanatory_followup_request,
    normalize_andes_response,
    validate_response_code,
)


class AndesResponseGuardrailTests(unittest.TestCase):
    def test_normalization_repairs_common_andes_patterns(self):
        response = """```python
import andes

ssa = andes.load("kundur/kundur_full.xlsx", setup=False, no_output=True, log=False)
ssa.add(model="PQ", param_dict={"bus": 8, "idx": "PQ_NEW_1", "p0": 0.01, "q0": 0.01})
ssa.setup()
ssa.PFlow.run()
print(ssa.Bus.v.vn[0])
```"""

        normalized, notes = normalize_andes_response(response)

        self.assertIn('andes.get_case("kundur/kundur_full.xlsx")', normalized)
        self.assertIn('ssa.add("PQ", param_dict=', normalized)
        self.assertIn("ssa.Bus.v.v[0]", normalized)
        self.assertTrue(notes)

    def test_validation_rejects_unittest_for_code_only_prompt(self):
        response = """```python
import unittest

class Demo(unittest.TestCase):
    pass
```"""

        is_valid, errors = validate_response_code(
            response,
            user_context="Generate runnable Python code only. Use an ANDES built-in case.",
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("plain runnable Python script" in error for error in errors))

    def test_validation_rejects_missing_plot_when_prompt_requests_plot(self):
        response = """```python
import andes

ssa = andes.load("uploaded_ieee39.xlsx", setup=True, no_output=True, log=False)
ssa.PFlow.run()
print("done")
```"""

        is_valid, errors = validate_response_code(
            response,
            user_context="Generate runnable Python code only. Use my uploaded file uploaded_ieee39.xlsx and plot the voltage profile.",
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("should actually create a plot" in error for error in errors))

    def test_validation_rejects_missing_code_block_for_code_only_prompt(self):
        response = "import andes\nprint('hello')"

        is_valid, errors = validate_response_code(
            response,
            user_context="Generate runnable Python code only. Use my uploaded file uploaded_ieee39.xlsx.",
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("```python``` code block" in error for error in errors))

    def test_normalization_wraps_plain_code_only_response(self):
        response = "import andes\nprint('hello')"

        normalized, notes = normalize_andes_response(
            response,
            user_context="Generate runnable Python code only. Use my uploaded file uploaded_ieee39.xlsx.",
        )

        self.assertTrue(normalized.startswith("```python"))
        self.assertTrue(normalized.rstrip().endswith("```"))
        self.assertTrue(any("Wrapped a plain Python response" in note for note in notes))

    def test_normalization_repairs_builtin_join_path_and_vector_access(self):
        response = """```python
import os
import ANDES

script_dir = os.getcwd()
case = os.path.join(script_dir, "kundur_full.xlsx")
ssa = ANDES.load(case, setup=True, no_output=True, log=False)
bus_idx = ssa.Bus.idx
bus_v = ssa.Bus.v
```"""

        normalized, notes = normalize_andes_response(
            response,
            user_context="Generate runnable Python code only. Use the built-in kundur_full case and print bus voltages.",
        )

        self.assertIn('andes.get_case("kundur/kundur_full.xlsx")', normalized)
        self.assertIn("import andes", normalized)
        self.assertIn("np.asarray(ssa.Bus.idx.v, dtype=int)", normalized)
        self.assertIn("np.asarray(ssa.Bus.v.v, dtype=float)", normalized)
        self.assertTrue(any("Normalized `ANDES` imports" in note for note in notes))

    def test_validation_rejects_missing_line_angle_field(self):
        response = """```python
import andes

ssa = andes.load(andes.get_case("5bus/pjm5bus.json"), setup=True, no_output=True, log=False)
ssa.PFlow.run()
line_ids = ssa.Line.idx.v
print(line_ids)
```"""

        is_valid, errors = validate_response_code(
            response,
            user_context="Generate runnable Python code only. Use the built-in pjm5bus case, run power flow, and print the top-2 lines by absolute sending-end phase angle.",
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("Line.a1.e" in error for error in errors))

    def test_normalization_repairs_branch_power_fields_and_line_ids(self):
        response = """```python
import andes
import numpy as np
import matplotlib.pyplot as plt

ssa = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)
ssa.PFlow.run()
line_id = np.asarray(ssa.Line.idx.v, dtype=float)
line_p1 = np.asarray(ssa.Line.p1.v, dtype=float)
plt.plot(line_id, line_p1)
```"""

        normalized, notes = normalize_andes_response(
            response,
            user_context="Generate runnable Python code only. Use ieee39 and plot active power of all the branches.",
        )

        self.assertIn("[str(item) for item in np.asarray(ssa.Line.idx.v)]", normalized)
        self.assertIn("np.asarray(ssa.Line.a1.e, dtype=float)", normalized)
        self.assertTrue(notes)

    def test_validation_rejects_unsupported_branch_active_power_fields(self):
        response = """```python
import andes
import numpy as np

ssa = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)
ssa.PFlow.run()
line_id = np.asarray(ssa.Line.idx.v, dtype=float)
line_p1 = np.asarray(ssa.Line.p1.v, dtype=float)
print(line_id[:3], line_p1[:3])
```"""

        is_valid, errors = validate_response_code(
            response,
            user_context="Generate runnable Python code only. Use ieee39 and plot active power of all the branches.",
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("Line.idx.v" in error for error in errors))
        self.assertTrue(any("Line` does not expose `p1`" in error or "Line.a1.e" in error for error in errors))

    def test_normalization_repairs_plain_python_branch_flow_script(self):
        response = """import andes
import numpy as np

ssa = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)
ssa.PFlow.run()
line_id = np.asarray(ssa.Line.idx.v, dtype=float)
line_p1 = np.asarray(ssa.Line.p1.v, dtype=float)
print(line_id[:3], line_p1[:3])
"""

        normalized, notes = normalize_andes_response(
            response,
            user_context="Use ieee39 and plot active power of all the branches.",
        )

        self.assertIn("[str(item) for item in np.asarray(ssa.Line.idx.v)]", normalized)
        self.assertIn("np.asarray(ssa.Line.a1.e, dtype=float)", normalized)
        self.assertTrue(notes)

    def test_validation_checks_unfenced_plain_python_script(self):
        response = """import andes
import numpy as np

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=True,
    no_output=True,
    log=False,
)
ssa.PFlow.run()

bus_ids = np.asarray(ssa.PQ.bus.v, dtype=int)
bus_v = float(np.asarray(ssa.Bus.v.v, dtype=float)[int(np.where(np.asarray(ssa.Bus.idx.v, dtype=int) == 15)[0][0)])
if 15 in bus_ids:
    print(f"PQ load found at bus 15. Bus 15 voltage: {bus_v:.4f} p.u.")
"""

        is_valid, errors = validate_response_code(
            response,
            user_context="Use ieee39 to run a power flow. Tell me whether there is a PQ load associated with bus 15 and its bus voltage",
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("Syntax Error" in error for error in errors))

    def test_fallback_covers_ieee39_pq_bus_voltage_prompt(self):
        response = build_andes_fallback_response(
            "Use ieee39 to run a power flow. Tell me whether there is a PQ load associated with bus 15 and its bus voltage",
        )

        self.assertIn('andes.get_case("ieee39/ieee39.xlsx")', response)
        self.assertIn("pq_buses = np.asarray(ssa.PQ.bus.v, dtype=int)", response)
        self.assertIn("Bus {target_bus} voltage", response)

    def test_effective_user_context_ignores_repo_snippet_noise(self):
        prompt = """A generated Python script failed inside the PFAGENT Streamlit app.

## Original user request
plot branch flow

## ANDES continuity context
- Last successful case source: builtin
- Last successful case identifier: ieee39/ieee39.xlsx

## Retrieved repository context
### Repo snippet 1: text-to-sim/tests/test_andes_response_guardrails.py:1-20
```text
Use the built-in pjm5bus case and print the top-2 lines by absolute sending-end phase angle.
```"""

        effective = extract_effective_user_context(prompt)

        self.assertIn("plot branch flow", effective)
        self.assertIn("ieee39/ieee39.xlsx", effective)
        self.assertNotIn("pjm5bus", effective)

    def test_infer_requested_builtin_case_prefers_original_request_and_continuity(self):
        prompt = """A generated Python script failed inside the PFAGENT Streamlit app.

## Original user request
plot branch flow

## ANDES continuity context
- Last successful case source: builtin
- Last successful case identifier: ieee39/ieee39.xlsx

## Retrieved repository context
```text
Use the built-in pjm5bus case and print the top-2 lines by absolute sending-end phase angle.
```"""

        inferred_case = infer_requested_builtin_case(prompt)
        self.assertEqual("ieee39/ieee39.xlsx", inferred_case)

    def test_fallback_branch_flow_template_uses_continuity_case(self):
        prompt = """A generated Python script failed inside the PFAGENT Streamlit app.

## Original user request
plot branch flow

## ANDES continuity context
- Last successful case source: builtin
- Last successful case identifier: ieee39/ieee39.xlsx

## Retrieved repository context
```text
Use the built-in pjm5bus case and print the top-2 lines by absolute sending-end phase angle.
```"""

        response = build_andes_fallback_response(prompt)

        self.assertIn('andes.get_case("ieee39/ieee39.xlsx")', response)
        self.assertIn("p_ij = np.asarray(ssa.Line.a1.e, dtype=float)", response)
        self.assertIn('plt.plot(line_ids, p_ij, marker="o", label="Pij")', response)
        self.assertNotIn("pjm5bus", response)

    def test_fallback_covers_trip_line_voltage_change_prompt(self):
        response = build_andes_fallback_response(
            """trip one line and compare the bus voltage change

ANDES continuity context:
- Last successfully executed case source: builtin
- Last successfully executed case identifier: ieee39/ieee39.xlsx""",
        )

        self.assertIn('andes.get_case("ieee39/ieee39.xlsx")', response)
        self.assertIn(
            'contingency_ssa.Line.set(src="u", idx=candidate_line, attr="v", value=0)',
            response,
        )
        self.assertIn('getattr(ssa.Bus, "nosw_island"', response)
        self.assertIn('getattr(ssa.Bus, "island_sets"', response)
        self.assertIn("Maximum |ΔV| across buses", response)

    def test_validation_rejects_segment_style_generic_branch_flow_plot(self):
        response = """```python
import andes
import numpy as np
import matplotlib.pyplot as plt

ssa = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)
ssa.PFlow.run()
bus_id = np.asarray(ssa.Bus.idx.v, dtype=float)
line_bus1 = np.asarray(ssa.Line.bus1.v, dtype=float)
line_bus2 = np.asarray(ssa.Line.bus2.v, dtype=float)
line_a1 = np.asarray(ssa.Line.a1.e, dtype=float)

for i in range(len(line_bus1)):
    plt.plot([line_bus1[i], line_bus2[i]], [0, 0] + 4 * [i])
```"""

        is_valid, errors = validate_response_code(
            response,
            user_context="plot branch flow",
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("plot branch metrics against line IDs" in error for error in errors))
        self.assertTrue(any("synthetic y-vectors" in error for error in errors))

    def test_validation_rejects_status_based_line_trip_script(self):
        response = """```python
import andes
import numpy as np
import matplotlib.pyplot as plt

ssa = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)
ssa.PFlow.run()
bus_ids = np.asarray(ssa.Bus.idx.v, dtype=int)
bus_v0 = np.asarray(ssa.Bus.v.v, dtype=float)
line_ids = np.asarray(ssa.Line.idx.v)
line_status = np.asarray(ssa.Line.status.v, dtype=bool)
trip_line = andes.get_case("ieee39/ieee39.xlsx")
ssa.Line.set(src="status", idx=trip_line, attr="v", value=False)
plt.stem(bus_ids, bus_v0, use_line_collection=True)
```"""

        is_valid, errors = validate_response_code(
            response,
            user_context="trip one line and compare the bus voltage change",
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("Line` does not expose `status`" in error or "Line.status" in error for error in errors))
        self.assertTrue(any("src=\"u\"" in error for error in errors))
        self.assertTrue(any("use_line_collection=True" in error for error in errors))
        self.assertTrue(any("do not assign `andes.get_case(...)` to a line ID" in error for error in errors))
        self.assertTrue(any("post-contingency convergence and islanding" in error for error in errors))

    def test_normalization_repairs_status_based_line_trip_script(self):
        response = """```python
import andes
import numpy as np
import matplotlib.pyplot as plt

ssa = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)
ssa.PFlow.run()
line_status = np.asarray(ssa.Line.status.v, dtype=bool)
ssa.Line.set(src="status", idx=["Line_1"], attr="v", value=[False])
plt.stem([1, 2], [0.0, 0.1], use_line_collection=True)
```"""

        normalized, notes = normalize_andes_response(
            response,
            user_context="trip one line and compare the bus voltage change",
        )

        self.assertIn("ssa.Line.u.v", normalized)
        self.assertIn('ssa.Line.set(src="u", idx="Line_1", attr="v", value=0)', normalized)
        self.assertNotIn("use_line_collection=True", normalized)
        self.assertTrue(notes)

    def test_validation_rejects_n1_screen_without_islanding_checks(self):
        response = """```python
import andes
import numpy as np

ssa = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)
ssa.PFlow.run()
line_ids = [str(item) for item in np.asarray(ssa.Line.idx.v)]
for line_id in line_ids[:3]:
    contingency_ssa = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)
    contingency_ssa.PFlow.run()
    contingency_ssa.Line.set(src="u", idx=[line_id], attr="v", value=[0])
    contingency_ssa.PFlow.run()
    print(line_id, float(np.min(np.asarray(contingency_ssa.Bus.v.v, dtype=float))))
```"""

        is_valid, errors = validate_response_code(
            response,
            user_context="Run an N-1 screening on ieee39 and identify the worst contingency.",
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("post-contingency convergence and islanding" in error for error in errors))

    def test_validation_accepts_set_status_with_getattr_islanding_checks(self):
        response = """```python
import andes
import numpy as np

ssa = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)
ssa.PFlow.run()
line_ids = [str(item) for item in np.asarray(ssa.Line.idx.v)]
for line_id in line_ids[:3]:
    contingency_ssa = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)
    contingency_ssa.PFlow.run()
    contingency_ssa.Line.set_status(line_id, 0)
    run_result = contingency_ssa.PFlow.run()
    try:
        converged = bool(contingency_ssa.PFlow.converged)
    except Exception:
        converged = bool(run_result)
    island_sets = list(getattr(contingency_ssa.Bus, "island_sets", []) or [])
    no_slack_islands = int(len(getattr(contingency_ssa.Bus, "nosw_island", []) or []))
    islanded_bus_count = int(getattr(contingency_ssa.Bus, "n_islanded_buses", 0) or 0)
    exit_code = int(getattr(contingency_ssa, "exit_code", 1))
    print(line_id, converged, exit_code, len(island_sets), no_slack_islands, islanded_bus_count)
```"""

        is_valid, errors = validate_response_code(
            response,
            user_context="Run an N-1 screening on ieee39 and identify the worst contingency.",
        )

        self.assertTrue(is_valid, msg=str(errors))

    def test_explanatory_followup_detection_prefers_prose(self):
        self.assertTrue(
            is_explanatory_followup_request("Why the voltage distribution doesn't change at all after trpping the line")
        )
        self.assertFalse(
            is_explanatory_followup_request("trip one line and compare the bus voltage change")
        )

    def test_explanation_fallback_covers_line_trip_voltage_question(self):
        response = build_andes_explanation_fallback_response(
            "Why the voltage distribution doesn't change at all after trpping the line",
        )

        self.assertIn("meshed system", response)
        self.assertIn("branch flows or angles", response)
        self.assertNotIn("```python", response)

    def test_validation_rejects_hard_coded_pv_idx_for_bus_specific_update(self):
        response = """```python
import andes
import os

case = os.path.join(os.getcwd(), "grid39_for_review.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)
ssa.setup()
ssa.PV.set(src="v0", idx=[1], attr="v", value=[1.045])
ssa.PFlow.run()
```"""

        is_valid, errors = validate_response_code(
            response,
            user_context=(
                "Generate runnable Python code only. Use my uploaded file grid39_for_review.xlsx, "
                "inspect the case to find the PV device connected to bus 31, set its voltage target "
                "to 1.045, and rerun power flow."
            ),
        )

        self.assertFalse(is_valid)
        self.assertTrue(any("ssa.PV.bus.v" in error for error in errors))

    def test_validation_accepts_case_resolved_pv_update(self):
        response = """```python
import andes
import numpy as np
import os

case = os.path.join(os.getcwd(), "grid39_for_review.xlsx")
ssa = andes.load(case, setup=False, no_output=True, log=False)
ssa.setup()
target_bus = 31
pv_buses = np.asarray(ssa.PV.bus.v, dtype=int)
pv_idx = np.asarray(ssa.PV.idx.v)
match = np.where(pv_buses == target_bus)[0]
if len(match) == 0:
    raise ValueError(f"No PV device found at bus {target_bus}.")
ssa.PV.set(src="v0", idx=[pv_idx[int(match[0])]], attr="v", value=[1.045])
ssa.PFlow.run()
```"""

        is_valid, errors = validate_response_code(
            response,
            user_context=(
                "Generate runnable Python code only. Use my uploaded file grid39_for_review.xlsx, "
                "inspect the case to find the PV device connected to bus 31, set its voltage target "
                "to 1.045, and rerun power flow."
            ),
        )

        self.assertTrue(is_valid, msg=str(errors))
        self.assertEqual([], errors)


if __name__ == "__main__":
    unittest.main()
