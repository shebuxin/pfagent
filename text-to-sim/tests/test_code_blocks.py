import sys
import unittest
from pathlib import Path


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.code_blocks import (
    ensure_python_code_block,
    extract_python_code_blocks,
    replace_python_code_block,
    strip_python_code_from_message,
)


class CodeBlockHelpersTests(unittest.TestCase):
    def test_extracts_python_from_plain_script(self):
        message = "import andes\nssa = andes.load('demo')\nprint('ok')"
        self.assertEqual(
            extract_python_code_blocks(message),
            [message],
        )

    def test_extracts_python_from_generic_fence_when_code_looks_pythonic(self):
        message = "Here you go:\n``` \nimport andes\nprint('ok')\n```"
        blocks = extract_python_code_blocks(message)
        self.assertEqual(len(blocks), 1)
        self.assertIn("import andes", blocks[0])

    def test_wraps_plain_code_for_code_only_requests(self):
        normalized, notes = ensure_python_code_block(
            "import andes\nprint('ok')",
            user_context="Return exactly one runnable Python code block and no prose.",
        )
        self.assertTrue(normalized.startswith("```python"))
        self.assertTrue(any("Wrapped a plain Python response" in note for note in notes))

    def test_strips_code_from_message_body(self):
        message = "I prepared the script below.\n```python\nimport andes\nprint('ok')\n```"
        self.assertEqual(strip_python_code_from_message(message), "I prepared the script below.")

    def test_replaces_plain_python_message_with_canonical_fence(self):
        updated = replace_python_code_block(
            "import andes\nprint('ok')",
            0,
            "import andes\nprint('updated')",
        )
        self.assertIn("```python", updated)
        self.assertIn("updated", updated)


if __name__ == "__main__":
    unittest.main()
