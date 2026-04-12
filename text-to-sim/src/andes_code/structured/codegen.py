"""Top-level structured ANDES codegen orchestrator.

Extracted from ``src.chatbots.openai.rag_chatbot`` in Stage 1. Glues
the applicability gate, state reducer, report-kind classifier,
state-completeness check, and script builder into a single entry point
used by ``RAGChatbot`` (and by ``test_structured_andes_codegen``).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.andes_code.structured.scripts import build_structured_andes_script
from src.andes_code.structured.state import (
    StructuredAndesState,
    extract_result_json_keys,
    infer_structured_report_kind,
    merge_structured_andes_state,
    structured_codegen_is_applicable,
    structured_report_has_required_state,
)


def build_structured_andes_response(
    user_context: str,
    current_state: Optional[StructuredAndesState],
) -> Tuple[str, StructuredAndesState, List[str]]:
    if not structured_codegen_is_applicable(user_context):
        return "", current_state or StructuredAndesState(), []

    updated_state = merge_structured_andes_state(current_state, user_context)
    report_kind = infer_structured_report_kind(extract_result_json_keys(user_context), user_context)
    if not report_kind or not updated_state.case_reference:
        return "", updated_state, []
    if not structured_report_has_required_state(report_kind, updated_state):
        return "", updated_state, []

    script = build_structured_andes_script(report_kind, user_context, updated_state)
    return (
        f"```python\n{script}\n```",
        updated_state,
        [f"Generated a structured ANDES script for `{report_kind}`."],
    )
