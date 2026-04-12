import shutil
import sys
import tempfile
import unittest
from pathlib import Path
import asyncio

import andes


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verification.oracle import compute_oracle_turn_result
from verification.runner import _build_uploaded_runtime_context, _execute_generated_code, _extract_python_code
from verification.suite import build_open_generalization_suite, build_verification_suite
from src.chatbots.openai.rag_chatbot import (
    RAGChatbot,
    RAGConfig,
    StructuredAndesState,
    build_structured_andes_response,
    extract_python_code_blocks,
)


class StructuredAndesCodegenTests(unittest.TestCase):
    def _runtime_prompt(self, scenario, turn, workspace_dir: Path) -> str:
        prompt = turn["prompt"]
        if scenario["case_source"] == "uploaded":
            prompt = _build_uploaded_runtime_context(prompt, scenario["uploaded_filename"], workspace_dir)
        return prompt

    def test_structured_codegen_covers_all_verification_prompts(self):
        scenarios = build_verification_suite()

        for scenario in scenarios:
            state = StructuredAndesState()
            for turn in scenario["turns"]:
                prompt = self._runtime_prompt(scenario, turn, Path("/tmp/pfagent-structured-prompt"))
                response, state, _notes = build_structured_andes_response(prompt, state)

                self.assertTrue(
                    response,
                    msg=f"Structured codegen did not trigger for {scenario['scenario_id']} turn {turn['turn_id']}",
                )
                self.assertEqual(
                    1,
                    len(extract_python_code_blocks(response)),
                    msg=f"Expected exactly one Python block for {scenario['scenario_id']} turn {turn['turn_id']}",
                )

    def test_structured_codegen_matches_oracle_for_representative_scenarios(self):
        selected = {}
        for scenario in build_verification_suite():
            key = (scenario["blueprint"], scenario["case_source"])
            selected.setdefault(key, scenario)

        for scenario in selected.values():
            state = StructuredAndesState()
            for turn in scenario["turns"]:
                with tempfile.TemporaryDirectory(prefix="pfagent-structured-test-") as tmp:
                    workspace = Path(tmp)
                    if scenario["case_source"] == "uploaded":
                        source_case = Path(andes.get_case(scenario["source_case_path"]))
                        shutil.copyfile(source_case, workspace / scenario["uploaded_filename"])

                    prompt = self._runtime_prompt(scenario, turn, workspace)
                    response, state, _notes = build_structured_andes_response(prompt, state)
                    self.assertTrue(
                        response,
                        msg=f"Structured codegen did not trigger for {scenario['scenario_id']} turn {turn['turn_id']}",
                    )

                    code = _extract_python_code(response)
                    execution = _execute_generated_code(code, workspace)
                    self.assertTrue(
                        execution["execution_passed"],
                        msg=(
                            f"Execution failed for {scenario['scenario_id']} turn {turn['turn_id']}:\n"
                            f"{execution['execution_output']}"
                        ),
                    )

                    expected = compute_oracle_turn_result(scenario, turn)
                    self.assertEqual(
                        expected,
                        execution["result_json"],
                        msg=(
                            f"RESULT_JSON mismatch for {scenario['scenario_id']} turn {turn['turn_id']}\n"
                            f"expected={expected}\nactual={execution['result_json']}"
                        ),
                    )

    def test_chatbot_preserves_bus_rank_count_across_followups(self):
        scenario = next(
            item for item in build_verification_suite()
            if item["scenario_id"] == "scenario_013"
        )
        chatbot = RAGChatbot(
            RAGConfig(
                openai_api_key="test",
                chat_model="gpt-4.1-nano",
                code_compilation_check=True,
                allow_template_fallback=False,
            )
        )
        chatbot.load_system_prompt(session_id="structured_rank_count", custom_instructions="")

        async def _run() -> str:
            response = ""
            for turn in scenario["turns"]:
                response = await chatbot.chat(turn["prompt"])
            return response

        final_response = asyncio.run(_run())
        self.assertIn("top_k = 3", final_response)
        self.assertEqual(3, chatbot.structured_andes_state.bus_rank_count)
        self.assertIsNone(chatbot.structured_andes_state.line_rank_count)

    def test_chatbot_preserves_targeted_pq_state_for_n1_followup(self):
        scenario = next(
            item for item in build_verification_suite()
            if item["scenario_id"] == "scenario_101"
        )
        chatbot = RAGChatbot(
            RAGConfig(
                openai_api_key="test",
                chat_model="gpt-4.1-nano",
                code_compilation_check=True,
                allow_template_fallback=False,
            )
        )
        chatbot.load_system_prompt(session_id="structured_targeted_pq", custom_instructions="")

        async def _run() -> str:
            response = ""
            for turn in scenario["turns"]:
                response = await chatbot.chat(turn["prompt"])
            return response

        final_response = asyncio.run(_run())
        self.assertIn("target_pq_bus = 2", final_response)
        self.assertIn('ssa.Line.set(src="u"', final_response)
        self.assertIn("candidate_pairs = [(1, 2), (1, 5), (2, 3)]", final_response)
        self.assertEqual(2, chatbot.structured_andes_state.target_pq_bus)
        self.assertAlmostEqual(1.03, chatbot.structured_andes_state.target_pq_scale_factor)
        self.assertEqual([(1, 2), (1, 5), (2, 3)], chatbot.structured_andes_state.n1_candidate_lines)

    def test_chatbot_preserves_targeted_pv_state_for_line_outage_followup(self):
        scenario = next(
            item for item in build_verification_suite()
            if item["scenario_id"] == "scenario_103"
        )
        chatbot = RAGChatbot(
            RAGConfig(
                openai_api_key="test",
                chat_model="gpt-4.1-nano",
                code_compilation_check=True,
                allow_template_fallback=False,
            )
        )
        chatbot.load_system_prompt(session_id="structured_targeted_pv", custom_instructions="")

        async def _run() -> str:
            response = ""
            for turn in scenario["turns"]:
                response = await chatbot.chat(turn["prompt"])
            return response

        final_response = asyncio.run(_run())
        self.assertIn("target_pv_bus = 2", final_response)
        self.assertIn('result_json["opened_line_bus_pair"] = [opened_bus1, opened_bus2]', final_response)
        self.assertIn('ssa.Line.set(src="u"', final_response)
        self.assertEqual(2, chatbot.structured_andes_state.target_pv_bus)
        self.assertAlmostEqual(1.01, chatbot.structured_andes_state.target_pv_setpoint)
        self.assertEqual((1, 2), chatbot.structured_andes_state.opened_line_pair)

    def test_structured_codegen_handles_generalized_case_edit_wording(self):
        pq_scenario = next(
            item for item in build_verification_suite()
            if item["blueprint"] == "generalized_targeted_pq_then_branch_trip"
        )
        pv_scenario = next(
            item for item in build_verification_suite()
            if item["blueprint"] == "generalized_targeted_pv_then_branch_trip"
        )

        scenario_states = []
        for scenario in [pq_scenario, pv_scenario]:
            state = StructuredAndesState()
            for turn in scenario["turns"]:
                prompt = self._runtime_prompt(scenario, turn, Path("/tmp/pfagent-structured-generalized"))
                response, state, _notes = build_structured_andes_response(prompt, state)
                self.assertTrue(
                    response,
                    msg=f"Structured codegen did not trigger for {scenario['scenario_id']} turn {turn['turn_id']}",
                )
            scenario_states.append((scenario, state))

        pq_state = scenario_states[0][1]
        pv_state = scenario_states[1][1]
        self.assertIsNotNone(pq_state.case_reference)
        self.assertIsNotNone(pq_state.target_pq_bus)
        self.assertIsNotNone(pq_state.target_pq_scale_factor)
        self.assertIsNotNone(pq_state.opened_line_pair)
        self.assertIsNotNone(pv_state.case_reference)
        self.assertIsNotNone(pv_state.target_pv_bus)
        self.assertIsNotNone(pv_state.target_pv_setpoint)
        self.assertIsNotNone(pv_state.opened_line_pair)

    def test_structured_codegen_handles_open_generalization_suite(self):
        for scenario in build_open_generalization_suite():
            state = StructuredAndesState()
            for turn in scenario["turns"]:
                prompt = self._runtime_prompt(scenario, turn, Path("/tmp/pfagent-open-generalization"))
                response, state, _notes = build_structured_andes_response(prompt, state)
                self.assertTrue(
                    response,
                    msg=f"Structured codegen did not trigger for {scenario['scenario_id']} turn {turn['turn_id']}",
                )


if __name__ == "__main__":
    unittest.main()
