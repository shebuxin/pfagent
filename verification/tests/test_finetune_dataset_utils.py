import json
import tempfile
import unittest
from pathlib import Path

from knowledge.finetuning.scripts.fine_tuning_dataset_utils import DatasetPair, load_structured_pairs, write_jsonl


class FineTuneDatasetUtilsTests(unittest.TestCase):
    def test_load_structured_pairs_supports_multi_turn_messages(self) -> None:
        payload = {
            "examples": [
                {
                    "id": "conversation_example",
                    "messages": [
                        {"role": "user", "content": "Use the built-in IEEE 14 case and return code only."},
                        {
                            "role": "assistant",
                            "content": "```python\nimport andes\nssa = andes.load(andes.get_case('ieee14/ieee14_full.xlsx'), setup=True, no_output=True)\nssa.PFlow.run()\n```",
                        },
                        {"role": "user", "content": "Follow up and return a fresh script."},
                        {
                            "role": "assistant",
                            "content": "import andes\nssa = andes.load(andes.get_case('ieee14/ieee14_full.xlsx'), setup=True, no_output=True)\nssa.PFlow.run()",
                        },
                    ],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "conversation_examples.json"
            json_path.write_text(json.dumps(payload), encoding="utf-8")
            pairs = load_structured_pairs(json_path, "generalized_verified_conversation")

        self.assertEqual(len(pairs), 1)
        self.assertIsNotNone(pairs[0].messages)
        self.assertEqual([message["role"] for message in pairs[0].messages], ["user", "assistant", "user", "assistant"])
        self.assertNotIn("```", pairs[0].messages[1]["content"])

    def test_write_jsonl_preserves_conversation_messages(self) -> None:
        example = DatasetPair(
            source="generalized_verified_conversation",
            messages=[
                {"role": "user", "content": "Use the built-in IEEE 14 case."},
                {
                    "role": "assistant",
                    "content": "import andes\nssa = andes.load(andes.get_case('ieee14/ieee14_full.xlsx'), setup=True, no_output=True)\nssa.PFlow.run()",
                },
                {"role": "user", "content": "Now revise the study."},
                {
                    "role": "assistant",
                    "content": "import andes\nssa = andes.load(andes.get_case('ieee14/ieee14_full.xlsx'), setup=True, no_output=True)\nssa.PFlow.run()",
                },
            ],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dataset.jsonl"
            write_jsonl([example], output_path)
            rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]["messages"]), 4)
        self.assertEqual(rows[0]["messages"][2]["role"], "user")
        self.assertIn("andes.get_case", rows[0]["messages"][3]["content"])


if __name__ == "__main__":
    unittest.main()
