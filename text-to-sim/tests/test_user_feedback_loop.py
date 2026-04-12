import json
import sys
import tempfile
import unittest
from pathlib import Path


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.user_feedback_loop import (
    analyze_session_feedback_loop,
    load_session_log,
    record_chat_turn,
    record_code_feedback,
    record_execution_result,
)


class UserFeedbackLoopTests(unittest.TestCase):
    def test_user_feedback_failure_updates_profile_with_runtime_root_causes(self):
        with tempfile.TemporaryDirectory(prefix="pfagent-user-feedback-") as tmp:
            base_dir = Path(tmp) / "user_feedback"
            profile_path = Path(tmp) / "agent_profile.json"

            record_chat_turn(
                "session-alpha",
                turn_id=1,
                user_message="Modify the existing load at bus 2 and rerun the case.",
                contextual_user_message="Modify the existing load at bus 2 and rerun the case.",
                assistant_message="Please change idx=2 manually and run it.",
                chatbot_type="Fine-tuned + RAG",
                base_dir=base_dir,
            )
            record_code_feedback(
                "session-alpha",
                turn_id=1,
                message_index=0,
                code_id="code_0_0",
                verdict="failure",
                feedback_text="The response was not runnable and did not inspect the case.",
                root_cause_hint="wrong device idx",
                assistant_message="Please change idx=2 manually and run it.",
                base_dir=base_dir,
            )

            analysis = analyze_session_feedback_loop(
                "session-alpha",
                profile_path=profile_path,
                base_dir=base_dir,
            )

            self.assertEqual(1, analysis["failure_turn_count"])
            signature_ids = {item["signature_id"] for item in analysis["root_cause_summary"]}
            self.assertIn("positional_idx_used_as_device_idx", signature_ids)
            self.assertIn("response_not_runnable", signature_ids)

            merged_profile = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertIn("targeted_device_resolution", merged_profile["active_mutation_packs"])
            self.assertIn("runnable_code_contract", merged_profile["active_mutation_packs"])

    def test_execution_failures_flow_into_runtime_analysis(self):
        with tempfile.TemporaryDirectory(prefix="pfagent-user-feedback-exec-") as tmp:
            base_dir = Path(tmp) / "user_feedback"
            profile_path = Path(tmp) / "agent_profile.json"

            record_chat_turn(
                "session-beta",
                turn_id=1,
                user_message="Make that same demand record 4% heavier and rerun.",
                contextual_user_message="Make that same demand record 4% heavier and rerun.",
                assistant_message="```python\nprint('placeholder')\n```",
                chatbot_type="Fine-tuned + RAG",
                base_dir=base_dir,
            )
            record_execution_result(
                "session-beta",
                turn_id=1,
                message_index=0,
                code_id="code_0_0",
                executed_code="print('placeholder')",
                output="Error (exit code 1):\nValueError: invalid literal for int() with base 10: 'PQ_2'",
                base_dir=base_dir,
            )

            analysis = analyze_session_feedback_loop(
                "session-beta",
                profile_path=profile_path,
                base_dir=base_dir,
            )

            self.assertEqual(1, analysis["failure_turn_count"])
            signature_ids = {item["signature_id"] for item in analysis["root_cause_summary"]}
            self.assertIn("device_idx_cast_to_int", signature_ids)
            self.assertIn("open_ended_pq_percentage_language", signature_ids)

            session_log = load_session_log("session-beta", base_dir=base_dir)
            self.assertEqual(1, len(session_log["analysis_history"]))


if __name__ == "__main__":
    unittest.main()
