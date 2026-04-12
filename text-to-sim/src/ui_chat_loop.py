"""Chat-area UI rendering + chat-submission handler.

Extracted from main.py in Stage 5. Three entry points:

- ``build_contextual_user_input`` -- pure function that composes the
  user's raw message with runtime-file context, an uploaded-case
  preview, and the active ANDES case continuity block. Unit-testable.
- ``render_chat_area`` -- renders the chat header, message history,
  and the input form. Returns ``(user_input, submitted)`` from the
  form.
- ``handle_chat_submission`` -- async-invokes the chatbot on the
  contextual input, appends the turn to history, and reruns.

All session-state reads/writes stay against ``st.session_state`` --
this is a pure extraction, not a state-isolation refactor.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional, Tuple

import streamlit as st

from src.andes_case_inspector import build_case_idx_inventory
from src.andes_code.extractors import (
    extract_effective_user_context,
    infer_requested_builtin_case,
)
from src.files import build_uploaded_case_prompt_context, get_files_in_directory
from src.ui import display_chat_message
from src.user_feedback_loop import record_chat_turn


logger = logging.getLogger(__name__)


def build_contextual_user_input(
    user_input: str,
    runtime_files,
    runtime_data_dir: str,
    uploaded_case_preview: str,
    active_case: Optional[Dict[str, str]],
    case_idx_inventory: str = "",
) -> str:
    """Compose the chatbot prompt from the raw input plus runtime context.

    Wrapping order matches the sections documented in the prompt
    contract (runtime file list -> uploaded-case preview -> continuity
    block -> case idx inventory). Changing the order, the wording,
    or the fence template will drift the tuned model behavior --
    update the snapshot tests first if you need to change any of
    these strings.
    """
    contextual_user_input = user_input
    if runtime_files:
        available_files = "\n".join(f"- {name}" for name in runtime_files)
        contextual_user_input = (
            f"{user_input}\n\n"
            "Runtime file context:\n"
            f"- Working directory for execution: {runtime_data_dir}\n"
            "- Uploaded files available during execution:\n"
            f"{available_files}\n"
            "- Use these filenames directly in generated Python code when needed.\n"
            "- Case-loading rule: if using an uploaded file, load it directly with andes.load(\"<exact_filename>\", ...), and do NOT wrap it with andes.get_case(...).\n"
            "- Case-loading rule: only use andes.get_case(...) for ANDES built-in benchmark cases.\n"
            "- Preferred uploaded-case template: script_dir=os.getcwd(); case=os.path.join(script_dir, \"<exact_filename>\"); ssa=andes.load(case, setup=True, no_output=True, log=False)"
        )

    if uploaded_case_preview:
        contextual_user_input = (
            f"{contextual_user_input}\n\n"
            f"{uploaded_case_preview}"
        )

    if active_case and isinstance(active_case, dict):
        active_source = active_case.get("source", "")
        active_value = active_case.get("value", "")
        if active_source and active_value:
            contextual_user_input = (
                f"{contextual_user_input}\n\n"
                "ANDES continuity context:\n"
                f"- Last successfully executed case source: {active_source}\n"
                f"- Last successfully executed case identifier: {active_value}\n"
                "- If the user is asking a follow-up (for example: plot/summarize/analyze) and does not specify a new case, reuse this same case."
            )

    if case_idx_inventory:
        contextual_user_input = (
            f"{contextual_user_input}\n\n"
            f"{case_idx_inventory}"
        )

    return contextual_user_input


def _resolve_active_case_for_inventory(
    user_input: str,
    active_case: Optional[Dict[str, str]],
) -> tuple[str, str]:
    """Decide which case to inspect for the idx inventory.

    Prefers ``active_case`` (from a successful prior execution); falls
    back to inferring a built-in case name from the current prompt.
    Returns ``(case_source, case_reference)``, possibly ``("", "")``.
    """
    if active_case and isinstance(active_case, dict):
        source = active_case.get("source", "")
        reference = active_case.get("value", "")
        if source and reference:
            return source, reference

    # First turn in the session: try to read a built-in case name out
    # of the prompt so the user's very first "trip line N" attempt
    # still sees an inventory.
    effective_ctx = extract_effective_user_context(user_input) or user_input
    inferred = infer_requested_builtin_case(effective_ctx)
    if inferred:
        return "builtin", inferred

    return "", ""


def render_no_chatbot_banner() -> None:
    """Render the placeholder banner shown when no chatbot is initialized."""
    st.markdown("# Power Flow Agent")
    st.markdown("**Ready to chat! Initialize the agent to get started.**")
    st.info("ℹ️ Please initialize the agent using the sidebar")


def render_chat_area() -> Tuple[str, bool]:
    """Render chat header, message history, and input form.

    Returns ``(user_input, submitted)`` from the form. Assumes the
    chatbot is already initialized -- callers must check
    ``st.session_state.chatbot`` and early-return to
    ``render_no_chatbot_banner()`` before calling this.
    """
    # Chat header
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        # Show API key status in header
        if st.session_state.openai_api_key:
            masked_key = st.session_state.openai_api_key[:8] + "..." + st.session_state.openai_api_key[-4:]
            st.caption(f"🔑 {masked_key}")

    with col2:
        st.markdown("# Power Flow Agent")
        if st.session_state.get('default_manual_loaded', False):
            st.caption("Ask me anything about ANDES. The official ANDES manual is loaded by default.")
        else:
            st.caption("Ask me anything about your documents")

        # Show pending error fix status
        if st.session_state.get('pending_error_fix') and len(st.session_state.pending_error_fix) > 0:
            st.info(f"🔧 {len(st.session_state.pending_error_fix)} error fix(es) pending...")

    with col3:
        pass

    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display chat history
        for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):
            display_chat_message(user_msg, is_user=True, message_index=i)
            display_chat_message(assistant_msg, is_user=False, message_index=i)

    # Chat input area
    # Add spacing before input
    st.markdown("")
    st.markdown("")
    st.markdown("")

    # Chat input area
    input_container = st.container()
    with input_container:
        # Create form for input
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                label="Message",
                placeholder="Type your message here...",
                height=70,
                max_chars=4000,
                key="user_input",
                label_visibility="collapsed"
            )
            submitted = st.form_submit_button("Send", type="primary")

    return user_input, submitted


def handle_chat_submission(user_input: str) -> None:
    """Process a form submission: build the contextual input, run the
    chatbot, append the turn to history, and rerun the app.
    """
    with st.spinner("Generating..."):
        try:
            runtime_data_dir = f"./code_executions/{st.session_state.session_id}/data"
            runtime_files = get_files_in_directory(runtime_data_dir)
            active_case = st.session_state.get("active_andes_case")

            uploaded_case_preview = build_uploaded_case_prompt_context(
                runtime_data_dir,
                user_input=user_input,
                active_case=active_case if isinstance(active_case, dict) else None,
            )

            # Ground-truth device idx values for the active case.
            # Feeds the model real idx strings ("Line_18") and bus
            # linkage so it doesn't hardcode guesses like "18".
            inventory_source, inventory_reference = _resolve_active_case_for_inventory(
                user_input=user_input,
                active_case=active_case if isinstance(active_case, dict) else None,
            )
            case_idx_inventory = build_case_idx_inventory(
                case_source=inventory_source,
                case_reference=inventory_reference,
                uploaded_dir=runtime_data_dir,
            )
            if case_idx_inventory:
                logger.info(
                    "Injected ANDES idx inventory for %s:%s (%d chars)",
                    inventory_source, inventory_reference, len(case_idx_inventory),
                )

            contextual_user_input = build_contextual_user_input(
                user_input=user_input,
                runtime_files=runtime_files,
                runtime_data_dir=runtime_data_dir,
                uploaded_case_preview=uploaded_case_preview,
                active_case=active_case if isinstance(active_case, dict) else None,
                case_idx_inventory=case_idx_inventory,
            )

            # Get response from chatbot (async)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                st.session_state.chatbot.chat(contextual_user_input)
            )
            loop.close()

            # Add to chat history
            st.session_state.chat_history.append((user_input, response))
            record_chat_turn(
                st.session_state.session_id,
                turn_id=len(st.session_state.chat_history),
                user_message=user_input,
                assistant_message=response,
                contextual_user_message=contextual_user_input,
                chatbot_type=st.session_state.get("chatbot_type", ""),
                turn_type="user",
                active_case=active_case if isinstance(active_case, dict) else None,
            )

            # Rerun to update the display
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            # Log more detailed error for debugging
            logger.error(f"Chat error details: {str(e)}", exc_info=True)
