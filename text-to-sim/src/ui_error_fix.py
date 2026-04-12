"""Pending error-fix handler: Codex fixer + local validation retry loop.

Extracted from main.py in Stage 5. When the user clicks "Fix Error
with AI" on a failed code block, the request is queued on
``st.session_state.pending_error_fix``. Every Streamlit rerun pops one
request off the queue and runs the repair pipeline:

1. Build the fix-request payload (failed code + error output + chat
   context + runtime file listing + uploaded-case preview + active
   ANDES case).
2. Ask the Codex error fixer for a repair (falls back to the current
   chatbot with a repo-aware prompt if the Codex path fails).
3. If local validation is enabled, execute the fixed code in the
   session workspace; if it still errors, loop back to the Codex
   fixer with the new error output (up to
   ``max_error_fix_validation_retries`` rounds).
4. Append the full repair turn to chat history + the recorder.

All session-state reads/writes stay against ``st.session_state`` --
this is a behavior-preserving extraction, not a state refactor.
"""

from __future__ import annotations

import asyncio
import logging

import streamlit as st

from src.code_blocks import extract_python_code_segments
from src.codex_fixer import (
    DEFAULT_CODEX_FIX_MODEL,
    build_repo_aware_fix_prompt,
    create_codex_error_fixer,
    normalize_error_fix_response,
    run_isolated_chat_repair,
)
from src.files import (
    build_uploaded_case_prompt_context,
    execute_python_code,
    get_files_in_directory,
)
from src.user_feedback_loop import execution_output_has_error, record_chat_turn


logger = logging.getLogger(__name__)


def _extract_first_python_candidate(response_text: str) -> str:
    segments = extract_python_code_segments(response_text or "")
    if not segments:
        return ""
    return segments[0].code.strip()


def _append_local_validation_note(response_text: str, validation_output: str, passed: bool, attempts_used: int) -> str:
    status_line = (
        f"Local validation: passed after {attempts_used} attempt(s) in the current session environment."
        if passed
        else f"Local validation: still failing after {attempts_used} attempt(s)."
    )
    normalized_output = (validation_output or "").strip()
    if not normalized_output:
        return f"{response_text}\n\n{status_line}"

    output_excerpt = normalized_output
    if len(output_excerpt) > 1800:
        output_excerpt = output_excerpt[:1797].rstrip() + "..."

    return (
        f"{response_text}\n\n"
        f"**{status_line}**\n\n"
        "```text\n"
        f"{output_excerpt}\n"
        "```"
    )


def handle_pending_error_fixes(repo_root: str) -> None:
    """Process up to one pending error-fix request from the queue.

    ``repo_root`` is passed in rather than computed from ``__file__``
    because this module lives under ``src/``, so
    ``os.path.dirname(__file__)/..`` would point to ``text-to-sim/``
    instead of the repository root. The caller (main.py) owns the
    location knowledge.
    """
    if not (st.session_state.get('pending_error_fix') and len(st.session_state.pending_error_fix) > 0):
        return

    error_fix_request = st.session_state.pending_error_fix.pop(0)

    with st.spinner("🔧 AI is fixing the error..."):
        try:
            response = ""
            contextual_fix_prompt = error_fix_request["prompt"]
            fix_chatbot_type = st.session_state.get("chatbot_type", "")
            fixer = None
            used_codex_fixer = False
            validation_attempts_used = 0
            latest_validation_output = ""
            validation_passed = False

            runtime_data_dir = f"./code_executions/{st.session_state.session_id}/data"
            runtime_files = get_files_in_directory(runtime_data_dir)
            active_case = st.session_state.get("active_andes_case")

            message_index = error_fix_request.get("message_index")
            user_message = ""
            assistant_message = ""
            if isinstance(message_index, int) and 0 <= message_index < len(st.session_state.chat_history):
                user_message, assistant_message = st.session_state.chat_history[message_index]

            uploaded_case_preview = build_uploaded_case_prompt_context(
                runtime_data_dir,
                user_input=user_message,
                active_case=active_case if isinstance(active_case, dict) else None,
            )

            fix_request_payload = {
                "failed_code": error_fix_request["failed_code"],
                "error_output": error_fix_request["error_output"],
                "user_message": user_message,
                "assistant_message": assistant_message,
                "recent_chat_history": st.session_state.chat_history,
                "message_index": message_index,
                "runtime_data_dir": runtime_data_dir,
                "runtime_files": runtime_files,
                "uploaded_case_preview": uploaded_case_preview,
                "active_case": active_case if isinstance(active_case, dict) else None,
                "custom_instructions": st.session_state.get("custom_system_prompt", ""),
            }

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                try:
                    if st.session_state.get("use_codex_error_fixer", True):
                        fixer = create_codex_error_fixer(
                            openai_api_key=st.session_state.openai_api_key,
                            repo_root=repo_root,
                            model=st.session_state.get("codex_error_fix_model", DEFAULT_CODEX_FIX_MODEL),
                        )
                        used_codex_fixer = True
                        response, contextual_fix_prompt = loop.run_until_complete(
                            fixer.fix_error(fix_request_payload)
                        )
                        fix_chatbot_type = (
                            f"{fix_chatbot_type} + CodexFixer"
                            if fix_chatbot_type
                            else "CodexFixer"
                        )
                    else:
                        response = loop.run_until_complete(
                            st.session_state.chatbot.chat(error_fix_request["prompt"])
                        )
                except Exception as fixer_error:
                    fixer_error_text = f"{type(fixer_error).__name__}: {fixer_error}"
                    logger.warning(
                        "Codex fixer failed; falling back to the current chatbot: %s",
                        fixer_error_text,
                        exc_info=True,
                    )
                    fixer = None
                    used_codex_fixer = False
                    contextual_fix_prompt = build_repo_aware_fix_prompt(
                        fix_request_payload,
                        repo_root=repo_root,
                        fallback_reason=fixer_error_text,
                    )
                    response = loop.run_until_complete(
                        run_isolated_chat_repair(
                            st.session_state.chatbot,
                            contextual_fix_prompt,
                        )
                    )
                    fix_chatbot_type = (
                        f"{fix_chatbot_type} + RepoAwareFallbackFix"
                        if fix_chatbot_type
                        else "RepoAwareFallbackFix"
                    )
                    st.warning(
                        "⚠️ The Codex fixer was unavailable, so the app fell back to the current chat agent for this repair.\n\n"
                        f"Reason: `{fixer_error_text}`"
                    )

                if not used_codex_fixer:
                    response, fallback_notes = normalize_error_fix_response(
                        response,
                        fix_request_payload,
                    )
                    if fallback_notes:
                        logger.info(
                            "Applied shared guardrails to fallback error-fix response: %s",
                            "; ".join(fallback_notes),
                        )

                if st.session_state.get("validate_error_fix_locally", True):
                    candidate_code = _extract_first_python_candidate(response)
                    validation_attempts_used = 1

                    if candidate_code:
                        latest_validation_output = execute_python_code(candidate_code)
                        validation_passed = not execution_output_has_error(latest_validation_output)
                    else:
                        latest_validation_output = "No runnable Python code block found in the fixer response."

                    retry_budget = int(st.session_state.get("max_error_fix_validation_retries", 2))
                    retry_count = 0
                    while (
                        used_codex_fixer
                        and fixer is not None
                        and not validation_passed
                        and retry_count < retry_budget
                    ):
                        retry_count += 1
                        retry_payload = dict(fix_request_payload)
                        retry_payload.update(
                            {
                                "failed_code": candidate_code or fix_request_payload["failed_code"],
                                "error_output": latest_validation_output,
                                "validation_attempt": retry_count,
                                "validation_output": latest_validation_output,
                                "previous_candidate_code": candidate_code,
                            }
                        )
                        response, contextual_fix_prompt = loop.run_until_complete(
                            fixer.fix_error(retry_payload)
                        )
                        candidate_code = _extract_first_python_candidate(response)
                        validation_attempts_used += 1
                        if not candidate_code:
                            latest_validation_output = "No runnable Python code block found in the fixer response."
                            validation_passed = False
                            break

                        latest_validation_output = execute_python_code(candidate_code)
                        validation_passed = not execution_output_has_error(latest_validation_output)

                    response = _append_local_validation_note(
                        response,
                        latest_validation_output,
                        validation_passed,
                        validation_attempts_used,
                    )
            finally:
                loop.close()
                asyncio.set_event_loop(None)

            # Add the error fix conversation to chat history
            st.session_state.chat_history.append((
                f"🔧 **Error Fix Request:** Please fix this code error:\n\n{error_fix_request['prompt']}",
                response
            ))
            record_chat_turn(
                st.session_state.session_id,
                turn_id=len(st.session_state.chat_history),
                user_message=error_fix_request["prompt"],
                assistant_message=response,
                contextual_user_message=contextual_fix_prompt,
                chatbot_type=fix_chatbot_type,
                turn_type="error_fix",
                active_case=st.session_state.get("active_andes_case") if isinstance(st.session_state.get("active_andes_case"), dict) else None,
            )

            # Show success message with more context
            if st.session_state.get("validate_error_fix_locally", True):
                if validation_passed:
                    st.success("✅ AI provided a fix and it passed local execution in the current session.")
                else:
                    st.warning("⚠️ AI provided a fix, but local execution still failed. The latest validation output is included in the response.")
            else:
                st.success("✅ AI has analyzed the error and provided a fix. Check the new response above!")
            st.info("💡 You can still run or edit the corrected code by clicking the ▶ Run button on the new code block.")

            # Rerun to update the display
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error generating fix: {str(e)}")
            logger.error(f"Error fix generation error: {str(e)}", exc_info=True)
