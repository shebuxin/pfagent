import asyncio
import tempfile
import unittest
from pathlib import Path

from src.code_blocks import extract_python_code_segments
from src.codex_fixer import (
    CodexFixerConfig,
    RepoSnippet,
    RepoAwareCodexFixer,
    RepoContextRetriever,
    _extract_response_text,
    _response_hit_output_cap,
    _summarize_response_for_debug,
    build_basic_error_fix_prompt,
    build_codex_fix_user_message,
    build_repo_aware_fix_prompt,
    clear_repo_context_cache,
    normalize_error_fix_response,
    run_isolated_chat_repair,
)


class RepoContextRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.tempdir.name)
        (self.repo_root / "text-to-sim" / "src").mkdir(parents=True)
        (self.repo_root / "docs").mkdir(parents=True)

        (self.repo_root / "README.md").write_text("# Test Repo\n", encoding="utf-8")
        (self.repo_root / "text-to-sim" / "src" / "rag_chatbot.py").write_text(
            "\n".join(
                [
                    "def open_line(ssa, line_id):",
                    "    ssa.Line.set(src='u', idx=[line_id], attr='v', value=[0])",
                    "    return ssa.Line.get(src='u', idx=[line_id], attr='v')",
                ]
            ),
            encoding="utf-8",
        )
        (self.repo_root / "text-to-sim" / "src" / "analysis.py").write_text(
            "def analyze(text):\n    return text\n",
            encoding="utf-8",
        )
        clear_repo_context_cache()

    def tearDown(self):
        clear_repo_context_cache()
        self.tempdir.cleanup()

    def test_retriever_prefers_matching_andes_api_pattern(self):
        retriever = RepoContextRetriever(str(self.repo_root))
        snippets = retriever.retrieve(
            "AttributeError Line object has no attribute set_status use ssa.Line.set src u",
            k=2,
        )

        self.assertTrue(snippets)
        self.assertEqual("text-to-sim/src/rag_chatbot.py", snippets[0].path)
        self.assertIn("ssa.Line.set", snippets[0].content)

    def test_build_repo_aware_fix_prompt_uses_repo_context_for_fallback(self):
        prompt = build_repo_aware_fix_prompt(
            {
                "user_message": "Trip one line and compare the bus voltage change.",
                "failed_code": "ssa.Line.set(src='status', idx=line_id, attr='v', value=[0])",
                "error_output": "KeyError: 'status'",
            },
            str(self.repo_root),
            fallback_reason="NotFoundError: gpt-5.2-codex",
        )

        self.assertIn("Original user request", prompt)
        self.assertIn("Fallback repair mode", prompt)
        self.assertIn("ssa.Line.set(src='u'", prompt)
        self.assertIn("NotFoundError: gpt-5.2-codex", prompt)


class CodexFixPromptTests(unittest.TestCase):
    def test_basic_prompt_includes_code_and_output(self):
        prompt = build_basic_error_fix_prompt("print('hi')", "Traceback...")
        self.assertIn("Original Code", prompt)
        self.assertIn("Traceback...", prompt)

    def test_repo_aware_prompt_includes_runtime_repo_and_case_context(self):
        prompt = build_codex_fix_user_message(
            {
                "user_message": "Trip the first line and save a plot.",
                "failed_code": "ssa.Line.set_status(first_line, 0)",
                "error_output": "AttributeError: 'Line' object has no attribute 'set_status'",
                "runtime_data_dir": "./code_executions/session/data",
                "runtime_files": ["ieee14.raw", "notes.txt"],
                "uploaded_case_preview": "Uploaded ANDES case summary: ieee14.raw",
                "active_case": {"source": "uploaded", "value": "ieee14.raw"},
                "recent_chat_history": [
                    ("Run a study on ieee14", "Here is a script."),
                    ("Trip the first line", "Try this updated script."),
                ],
                "message_index": 1,
            },
            [
                RepoSnippet(
                    path="text-to-sim/src/rag_chatbot.py",
                    start_line=10,
                    end_line=20,
                    content="ssa.Line.set(src='u', idx=[line_id], attr='v', value=[0])",
                    score=15.0,
                )
            ],
        )

        self.assertIn("Original user request", prompt)
        self.assertIn("Runtime file context", prompt)
        self.assertIn("Uploaded case preview", prompt)
        self.assertIn("ANDES continuity context", prompt)
        self.assertIn("Retrieved repository context", prompt)
        self.assertIn("ssa.Line.set(src='u'", prompt)

    def test_repo_aware_prompt_includes_local_validation_feedback(self):
        prompt = build_codex_fix_user_message(
            {
                "user_message": "Fix the line outage script.",
                "failed_code": "ssa.Line.set_status(first_line, 0)",
                "error_output": "AttributeError: 'Line' object has no attribute 'set_status'",
                "validation_attempt": 1,
                "validation_output": "Error (exit code 1):\nAttributeError: 'Line' object has no attribute 'set_status'",
                "previous_candidate_code": "ssa.Line.set_status(first_line, 0)",
            },
            [],
        )

        self.assertIn("Local validation feedback", prompt)
        self.assertIn("retry attempt 1", prompt)
        self.assertIn("Validation failure output", prompt)
        self.assertIn("previously generated candidate".lower(), prompt.lower())

    def test_repo_aware_prompt_can_mark_fallback_mode(self):
        prompt = build_codex_fix_user_message(
            {
                "user_message": "Trip one line and compare the bus voltage change.",
                "failed_code": "ssa.Line.set(src='status', idx=line_id, attr='v', value=[0])",
                "error_output": "KeyError: 'status'",
            },
            [],
            fallback_reason="AuthenticationError: model unavailable",
        )

        self.assertIn("Fallback repair mode", prompt)
        self.assertIn("Ignore earlier unrelated turns", prompt)
        self.assertIn("AuthenticationError: model unavailable", prompt)

    def test_extract_response_text_handles_multi_block_responses_api_content(self):
        class ResponseBlock:
            def __init__(self, text: str):
                self.text = text

        response = type(
            "FakeResponse",
            (),
            {
                "content": [
                    {"type": "reasoning", "summary": "thinking"},
                    {"type": "output_text", "text": "Here is the fix:"},
                    ResponseBlock("```python\nprint('ok')\n```"),
                ]
            },
        )()

        text = _extract_response_text(response)
        self.assertIn("Here is the fix:", text)
        self.assertIn("```python", text)

    def test_extract_response_text_prefers_output_text_when_present(self):
        response = type(
            "FakeResponse",
            (),
            {
                "output_text": "```python\nprint('ok')\n```",
                "content": [],
            },
        )()

        text = _extract_response_text(response)
        self.assertEqual("```python\nprint('ok')\n```", text)

    def test_extract_response_text_reads_message_blocks_from_output_items(self):
        class OutputText:
            def __init__(self, text: str):
                self.text = text

        class OutputMessage:
            def __init__(self, content):
                self.type = "message"
                self.content = content

        response = type(
            "FakeResponse",
            (),
            {
                "output": [
                    OutputMessage([OutputText("```python\nprint('from output')\n```")]),
                ]
            },
        )()

        text = _extract_response_text(response)
        self.assertIn("from output", text)

    def test_response_debug_summary_mentions_item_types_and_preview(self):
        class ResponseBlock:
            def __init__(self, text: str):
                self.text = text

        response = type(
            "FakeResponse",
            (),
            {
                "content": [
                    {"type": "output_text", "text": "Explaining the fix"},
                    ResponseBlock("```python\nprint('ok')\n```"),
                ]
            },
        )()

        summary = _summarize_response_for_debug(response)
        self.assertIn("response_type=FakeResponse", summary)
        self.assertIn("content_type=list", summary)
        self.assertIn("preview=", summary)
        self.assertIn("Explaining the fix", summary)

    def test_response_hit_output_cap_detects_incomplete_reason(self):
        response = type(
            "FakeResponse",
            (),
            {
                "incomplete_details": type("Incomplete", (), {"reason": "max_output_tokens"})(),
            },
        )()

        self.assertTrue(_response_hit_output_cap(response))

    def test_normalize_error_fix_response_applies_shared_andes_guardrails(self):
        request = {
            "user_message": "Use ieee39 and plot active power of all the branches.",
            "failed_code": "line_id = np.asarray(ssa.Line.idx.v, dtype=float)\nline_p1 = np.asarray(ssa.Line.p1.v, dtype=float)",
            "error_output": "ValueError: could not convert string to float: 'Line_1'",
        }
        response_text = """```python
import andes
import numpy as np

ssa = andes.load(andes.get_case("ieee39/ieee39.xlsx"), setup=True, no_output=True, log=False)
ssa.PFlow.run()
line_id = np.asarray(ssa.Line.idx.v, dtype=float)
line_p1 = np.asarray(ssa.Line.p1.v, dtype=float)
print(line_id[:3], line_p1[:3])
```"""

        normalized, notes = normalize_error_fix_response(response_text, request)

        self.assertIn("[str(item) for item in np.asarray(ssa.Line.idx.v)]", normalized)
        self.assertIn("np.asarray(ssa.Line.a1.e, dtype=float)", normalized)
        self.assertTrue(notes)

    def test_plain_fixer_response_keeps_code_separate_from_validation_markdown(self):
        request = {
            "user_message": "Use ieee39 to run a power flow. Tell me whether there is a PQ load associated with bus 15 and its bus voltage",
            "failed_code": "print('broken')",
            "error_output": "SyntaxError: ...",
        }
        response_text = """# required_dependencies: andes,numpy
import andes
import numpy as np

ssa = andes.load(
    andes.get_case("ieee39/ieee39.xlsx"),
    setup=True,
    no_output=True,
    log=False,
)
ssa.PFlow.run()
print("ok")

**Local validation: passed after 1 attempt(s) in the current session environment.**

```text
ok
```"""

        normalized, _ = normalize_error_fix_response(response_text, request)
        segments = extract_python_code_segments(normalized)

        self.assertEqual(1, len(segments))
        self.assertIn('print("ok")', segments[0].code)
        self.assertNotIn("Local validation:", segments[0].code)

    def test_run_isolated_chat_repair_ignores_prior_history_and_restores_it(self):
        class DummyChatbot:
            def __init__(self):
                self.conversation_history = ["old unrelated turn"]

            async def chat(self, user_message: str) -> str:
                if self.conversation_history:
                    return "drifted to old context"
                self.conversation_history.append(user_message)
                self.conversation_history.append("assistant repair")
                return "focused repair"

        chatbot = DummyChatbot()
        original_history = list(chatbot.conversation_history)

        result = asyncio.run(run_isolated_chat_repair(chatbot, "fix only this failing script"))

        self.assertEqual("focused repair", result)
        self.assertEqual(original_history, chatbot.conversation_history)

    def test_request_fix_retries_with_medium_verbosity_when_model_rejects_low(self):
        class DummyResponses:
            def __init__(self):
                self.calls = []

            async def create(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    raise Exception(
                        "BadRequestError: Error code: 400 - {'error': {'message': "
                        "\"Unsupported value: 'low' is not supported with the 'gpt-5.2-codex' model. "
                        "Supported values are: 'medium'.\", 'param': 'text.verbosity'}}"
                    )
                return {"ok": True}

        fixer = RepoAwareCodexFixer.__new__(RepoAwareCodexFixer)
        fixer.config = CodexFixerConfig(
            openai_api_key="test-key",
            repo_root=".",
            model="gpt-5.2-codex",
            text_verbosity="low",
        )
        dummy_responses = DummyResponses()
        fixer.client = type("DummyClient", (), {"responses": dummy_responses})()

        result = asyncio.run(
            fixer._request_fix(
                "system",
                "user",
                max_output_tokens=100,
                reasoning_effort="medium",
            )
        )

        self.assertEqual({"ok": True}, result)
        self.assertEqual(2, len(dummy_responses.calls))
        self.assertEqual("low", dummy_responses.calls[0]["text"]["verbosity"])
        self.assertEqual("medium", dummy_responses.calls[1]["text"]["verbosity"])


if __name__ == "__main__":
    unittest.main()
