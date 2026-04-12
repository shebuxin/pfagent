"""Unit tests for src.chatbots.openai.chat_feedback.

These pin the exact wording of the three prompt pieces extracted from
rag_chatbot.chat in Stage 2. The strings are part of the prompt contract
the model has been tuned against, so changing them requires a deliberate
prompt/regression run, not a drive-by edit.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


TEXT_TO_SIM_ROOT = Path(__file__).resolve().parents[1]
if str(TEXT_TO_SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(TEXT_TO_SIM_ROOT))

from src.chatbots.openai.chat_feedback import (  # noqa: E402
    PROSE_RESPONSE_GUARDRAIL,
    PROSE_RETRY_NUDGE,
    build_compilation_error_feedback,
)


class ProseResponseGuardrailTests(unittest.TestCase):
    def test_starts_with_two_newlines_for_concatenation(self):
        # PROSE_RESPONSE_GUARDRAIL is appended to the existing system
        # prompt via `+=`, so the leading blank line must already be
        # baked into the constant.
        self.assertTrue(PROSE_RESPONSE_GUARDRAIL.startswith("\n\n"))

    def test_explicitly_forbids_code_fences(self):
        body = PROSE_RESPONSE_GUARDRAIL.lower()
        self.assertIn("plain prose only", body)
        self.assertIn("do not return python", body)
        self.assertIn("markdown code fences", body)

    def test_does_not_mention_retry_mechanics(self):
        # This guard is appended to the *first* system message, not
        # injected on retry, so it should describe the turn's intent
        # rather than retry behavior.
        self.assertNotIn("retry", PROSE_RESPONSE_GUARDRAIL.lower())


class ProseRetryNudgeTests(unittest.TestCase):
    def test_is_short_single_paragraph(self):
        # The nudge is injected as a HumanMessage content after the
        # model produced code for a prose-only follow-up. Keep it
        # short so it doesn't dilute the chat history.
        self.assertLess(len(PROSE_RETRY_NUDGE), 200)
        self.assertNotIn("\n\n", PROSE_RETRY_NUDGE)

    def test_explicitly_forbids_new_script(self):
        body = PROSE_RETRY_NUDGE.lower()
        self.assertIn("not a new script", body)
        self.assertIn("plain prose", body)
        self.assertIn("no python", body)


class BuildCompilationErrorFeedbackTests(unittest.TestCase):
    def test_lists_each_error_on_its_own_line(self):
        errors = [
            "Code block 1: Use `ssa.Bus.v.v`, not `ssa.Bus.v`.",
            "Code block 1: Use lowercase `andes`, not `ANDES`.",
        ]
        body = build_compilation_error_feedback(errors)
        self.assertIn(errors[0], body)
        self.assertIn(errors[1], body)
        # Errors are joined by a single newline (chr(10)), so each
        # error lands on its own line in the bulleted block.
        combined = "\n".join(errors)
        self.assertIn(combined, body)

    def test_preserves_python_fence_instruction(self):
        body = build_compilation_error_feedback(["irrelevant"])
        # This phrase is a load-bearing prompt contract -- removing the
        # backticks or the "exactly one" quantifier drifts the model.
        self.assertIn("exactly one ```python fenced code block", body)
        self.assertIn("no prose before or after it", body)

    def test_preserves_manual_aligned_execution_order(self):
        body = build_compilation_error_feedback(["irrelevant"])
        self.assertIn(
            "imports -> case loading -> optional edits -> setup -> run routine -> inspect results -> plotting/printing",
            body,
        )

    def test_preserves_idx_resolution_guardrail(self):
        body = build_compilation_error_feedback(["irrelevant"])
        self.assertIn("inspect the loaded case and resolve the real `idx`", body)
        self.assertIn("before calling `.set(...)`", body)

    def test_preserves_no_sample_output_guardrail(self):
        body = build_compilation_error_feedback(["irrelevant"])
        self.assertIn("Do not paste sample output", body)
        self.assertIn("Markdown comments", body)
        self.assertIn("C-style comments", body)

    def test_empty_error_list_still_produces_wellformed_body(self):
        # Defensive: the caller never passes [] today (it guards on
        # `if not is_valid`) but the function should not raise.
        body = build_compilation_error_feedback([])
        self.assertIn("ANDES-specific issues", body)
        self.assertIn("```python fenced code block", body)

    def test_output_starts_with_newline(self):
        # The feedback body is passed as HumanMessage content verbatim;
        # the leading newline lets the "has ANDES-specific issues:"
        # header render as its own line in the chat transcript.
        body = build_compilation_error_feedback(["x"])
        self.assertTrue(body.startswith("\n"))


if __name__ == "__main__":
    unittest.main()
