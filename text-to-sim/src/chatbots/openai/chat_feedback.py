"""Prompt snippets and feedback builders used by the RAGChatbot chat loop.

Extracted from ``src.chatbots.openai.rag_chatbot`` in Stage 2. Keeping
these strings in one place deduplicates the inline copies that lived in
both ``chat`` and ``_chat_without_compilation_check``, and makes the
exact wording unit-testable without spinning up the LangChain stack.

Every exported name is byte-identical to what it replaces.
"""

from __future__ import annotations

from typing import List


# Appended to the system message when the user's turn looks like a
# conceptual follow-up rather than a code request. Starts with two
# newlines so it concatenates cleanly after the existing system prompt.
PROSE_RESPONSE_GUARDRAIL: str = (
    "\n\nFor this turn, the user is asking for an explanation or conceptual answer, not runnable code. "
    "Answer in plain prose only. Do not return Python, pseudocode, or Markdown code fences."
)


# Injected as a HumanMessage when the model returned code for an
# explanatory follow-up and we want to nudge it back to prose before
# retrying.
PROSE_RETRY_NUDGE: str = (
    "The user asked for an explanation, not a new script. "
    "Answer briefly in plain prose only, with no Python or code fences."
)


def build_compilation_error_feedback(error_messages: List[str]) -> str:
    """Build the HumanMessage body that feeds validator errors back to the model.

    The leading/trailing newlines, the bullet list joining convention, and
    the six trailing manual-aligned instructions are all part of the
    prompt contract the model has been tuned against -- do not reorder
    or rewrap them without a matching prompt/regression run.
    """
    return f"""
The Python code in your previous response has ANDES-specific issues:

{chr(10).join(error_messages)}

Please fix these issues and provide one corrected runnable Python script only.
Return exactly one ```python fenced code block with no prose before or after it.
Keep the execution order manual-aligned: imports -> case loading -> optional edits -> setup -> run routine -> inspect results -> plotting/printing.
Do not invent undocumented ANDES helpers or rename uploaded files.
If you are modifying an existing device, inspect the loaded case and resolve the real `idx` from the case arrays before calling `.set(...)`.
Do not paste sample output, Markdown comments, C-style comments, or explanatory text inside the Python script.
"""
