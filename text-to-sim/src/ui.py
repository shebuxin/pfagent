import os
from datetime import datetime
from typing import List, Tuple

import streamlit as st
from streamlit_ace import st_ace

from .files import execute_python_code, extract_generated_image_paths
from .analysis import create_analyzer
from .code_blocks import (
    extract_python_code_segments,
    replace_python_code_block,
    strip_python_code_from_message,
)
from .codex_fixer import build_basic_error_fix_prompt
from .user_feedback_loop import (
    execution_output_has_error,
    record_code_feedback,
    record_execution_result,
)


def load_css():
    """Load custom CSS for styling - now using native Streamlit components"""
    pass


def extract_python_code(text: str, msg_idx: int) -> List[Tuple[str, str]]:
    segments = extract_python_code_segments(text)
    return [(f"code_{msg_idx}_{j}", segment.code) for j, segment in enumerate(segments)]


def display_chat_message(message: str, is_user: bool = True, message_index: int = 0):
    code_blocks = []
    if not is_user:
        code_blocks = extract_python_code(message, message_index)
        message = strip_python_code_from_message(message)
    if is_user:
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
            if message:
                st.markdown(message)
    for code_id, code in code_blocks:
        with st.container():
            st.markdown("**Python Code:**")
            current_code = st.session_state.edited_codes.get(code_id, code)
            reset_counter = st.session_state.code_reset_counters.get(code_id, 0)
            edited_code = st_ace(
                value=current_code,
                language='python',
                key=f"edit_{code_id}_{reset_counter}",
                height=min(300, max(150, len(current_code.split('\n')) * 18)),
                auto_update=False,
                font_size=14,
                tab_size=4,
                show_gutter=True,
                show_print_margin=False,
                wrap=False,
                annotations=None,
                markers=None,
                placeholder="Type your Python code here...",
            )
            if edited_code != current_code:
                st.session_state.edited_codes[code_id] = edited_code
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("▶ Run", key=f"run_{code_id}", help="Execute this Python code"):
                    with st.spinner("Executing code..."):
                        output = execute_python_code(edited_code)
                        st.session_state.code_outputs[code_id] = output
                        update_conversation_history_with_edited_code(message_index, code_id, edited_code)
                        if st.session_state.get("session_id"):
                            record_execution_result(
                                st.session_state.session_id,
                                turn_id=message_index + 1,
                                message_index=message_index,
                                code_id=code_id,
                                executed_code=edited_code,
                                output=output,
                                active_case=st.session_state.get("active_andes_case") if isinstance(st.session_state.get("active_andes_case"), dict) else None,
                            )
                        st.session_state.refresh_files = True
                        st.rerun()
            with col2:
                if st.button("🔄 Reset", key=f"reset_{code_id}", help="Reset to original code"):
                    st.session_state.edited_codes[code_id] = code
                    st.session_state.code_reset_counters[code_id] = st.session_state.code_reset_counters.get(code_id, 0) + 1
                    st.rerun()
            if code_id in st.session_state.code_outputs:
                output = st.session_state.code_outputs[code_id]
                st.markdown("**Output:**")
                st.code(output, language="text")

                generated_images = extract_generated_image_paths(output, st.session_state.session_id)
                if generated_images:
                    st.markdown("**Generated Plot(s):**")
                    for image_path in generated_images:
                        st.image(
                            image_path,
                            caption=os.path.basename(image_path),
                            use_container_width=True,
                        )
                
                # Action buttons row
                is_error = execution_output_has_error(output)
                
                # Create columns for action buttons
                if is_error:
                    col_fix, col_analyze, col_spacer = st.columns([2, 2, 2])
                else:
                    col_analyze, col_spacer = st.columns([2, 4])
                
                # AI Analysis button (always shown)
                analysis_key = f"analysis_{code_id}"
                with (col_analyze if not is_error else col_analyze):
                    if st.button("🧠 AI Analysis", key=f"analyze_{code_id}", help="Get AI analysis of the code and output", type="secondary"):
                        current_code = st.session_state.edited_codes.get(code_id, code)
                        
                        # Initialize analysis storage if not exists
                        if 'code_analyses' not in st.session_state:
                            st.session_state.code_analyses = {}
                        
                        # Show spinner while analyzing
                        with st.spinner("🧠 AI is analyzing the code and output..."):
                            try:
                                # Create analyzer using the current OpenAI API key
                                if st.session_state.get('openai_api_key'):
                                    analyzer = create_analyzer(st.session_state.openai_api_key)
                                    
                                    # Get analysis (using sync wrapper since we're in Streamlit)
                                    analysis = analyzer.analyze_code_output_sync(current_code, output)
                                    
                                    # Store analysis in session state
                                    st.session_state.code_analyses[code_id] = analysis
                                    
                                    st.success("✅ Analysis complete! Check below for insights.")
                                    st.rerun()
                                else:
                                    st.error("❌ OpenAI API key not found. Please check your configuration.")
                            except Exception as e:
                                st.error(f"❌ Error during analysis: {str(e)}")
                
                # Error fix button (only shown for errors)
                if is_error:
                    with col_fix:
                        if st.button("🔧 Fix Error with AI", key=f"fix_{code_id}", help="Send error to AI for fixing", type="secondary"):
                            current_code = st.session_state.edited_codes.get(code_id, code)
                            error_prompt = build_basic_error_fix_prompt(current_code, output)
                            if 'pending_error_fix' not in st.session_state:
                                st.session_state.pending_error_fix = []
                            st.session_state.pending_error_fix.append(
                                {
                                    'prompt': error_prompt,
                                    'original_code_id': code_id,
                                    'failed_code': current_code,
                                    'error_output': output,
                                    'message_index': message_index,
                                }
                            )
                            st.info("🔧 Error fix request added to queue. The repo-aware fixer will process it shortly...")
                            st.rerun()
                
                # Display analysis if available
                if st.session_state.get('code_analyses', {}).get(code_id):
                    st.markdown("---")
                    st.markdown("**🧠 AI Analysis:**")
                    with st.expander("View Analysis", expanded=True):
                        st.markdown(st.session_state.code_analyses[code_id])
                        
                        # Option to clear analysis
                        if st.button("🗑️ Clear Analysis", key=f"clear_analysis_{code_id}", help="Remove this analysis"):
                            if code_id in st.session_state.code_analyses:
                                del st.session_state.code_analyses[code_id]
                            st.rerun()

                st.markdown("**Feedback Loop:**")
                with st.expander("Mark whether this code/result was correct", expanded=is_error):
                    verdict_key = f"feedback_verdict_{code_id}"
                    details_key = f"feedback_details_{code_id}"
                    root_cause_key = f"feedback_root_{code_id}"
                    verdict = st.radio(
                        "Result quality",
                        options=["success", "failure"],
                        horizontal=True,
                        key=verdict_key,
                        index=1 if is_error else 0,
                    )
                    root_cause_hint = st.text_input(
                        "Root cause hint",
                        key=root_cause_key,
                        placeholder="Example: wrong idx, did not inspect the case, forgot previous change, not runnable",
                    )
                    feedback_text = st.text_area(
                        "What happened?",
                        key=details_key,
                        height=80,
                        placeholder="Optional note to help the agent evolution loop learn from this run.",
                    )
                    if st.button("💾 Save Result Feedback", key=f"save_feedback_{code_id}"):
                        assistant_message = ""
                        if message_index < len(st.session_state.chat_history):
                            _user_msg, assistant_message = st.session_state.chat_history[message_index]
                        record_code_feedback(
                            st.session_state.session_id,
                            turn_id=message_index + 1,
                            message_index=message_index,
                            code_id=code_id,
                            verdict=verdict,
                            feedback_text=feedback_text,
                            root_cause_hint=root_cause_hint,
                            assistant_message=assistant_message,
                        )
                        if verdict == "failure":
                            st.warning("Saved as a failure case. This session will feed the evolution profile when you finish the session.")
                        else:
                            st.success("Saved as a successful result.")


def update_conversation_history_with_edited_code(message_index: int, code_id: str, edited_code: str):
    if message_index < len(st.session_state.chat_history):
        user_msg, assistant_msg = st.session_state.chat_history[message_index]
        original_code_blocks = extract_python_code(assistant_msg, message_index)
        for i, (original_code_id, _) in enumerate(original_code_blocks):
            if original_code_id == code_id:
                updated_msg = replace_python_code_block(assistant_msg, i, edited_code)
                st.session_state.chat_history[message_index] = (user_msg, updated_msg)
                break


def prepare_conversation_for_download():
    conversation_data = {
        "timestamp": datetime.now().isoformat(),
        "total_messages": len(st.session_state.chat_history),
        "documents_processed": st.session_state.documents_processed,
        "conversation": [],
    }
    for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):
        code_blocks = extract_python_code(assistant_msg, i)
        code_data = []
        for code_id, code in code_blocks:
            code_info = {
                "code_id": code_id,
                "original_code": code,
                "edited_code": st.session_state.edited_codes.get(code_id, code),
                "output": st.session_state.code_outputs.get(code_id, None),
                "ai_analysis": st.session_state.get('code_analyses', {}).get(code_id, None),
            }
            code_data.append(code_info)
        conversation_item = {
            "message_index": i,
            "user_message": user_msg,
            "assistant_message": assistant_msg,
            "code_blocks": code_data,
            "timestamp": datetime.now().isoformat(),
        }
        conversation_data["conversation"].append(conversation_item)
    return conversation_data


def show_feedback_screen():
    from .metadata import save_feedback
    from .session import end_session_cleanup
    from .user_feedback_loop import analyze_session_feedback_loop, record_session_feedback

    st.markdown("# 📝 Session Feedback")
    st.markdown("### Thank you for using LLM Sandbox!")
    st.markdown("Your feedback helps us improve the experience.")
    
    with st.form("feedback_form"):
            st.markdown("#### How was your experience?")
            rating = st.select_slider(
                "Overall Rating", options=[1, 2, 3, 4, 5], value=3, format_func=lambda x: "⭐" * x, help="Rate your overall experience"
            )
            st.markdown("**What worked well?** (Select all that apply)")
            col_a, col_b = st.columns(2)
            with col_a:
                easy_to_use = st.checkbox("Easy to use")
                accurate_responses = st.checkbox("Accurate responses")
                helpful_features = st.checkbox("Helpful features")
            with col_b:
                fast_responses = st.checkbox("Fast responses")
                good_code_execution = st.checkbox("Code execution worked well")
                useful_documents = st.checkbox("Document processing was useful")
            st.markdown("**What could be improved?**")
            improvements = st.multiselect(
                "Select areas for improvement",
                [
                    "Response accuracy",
                    "Response speed",
                    "User interface",
                    "Document processing",
                    "Code execution",
                    "File management",
                    "API key setup",
                    "Error handling",
                    "Documentation",
                ],
            )
            feedback_text = st.text_area(
                "Additional Comments",
                placeholder="Tell us more about your experience, suggestions for improvement, or any issues you encountered...",
                height=150,
            )
            col_submit1, col_submit2 = st.columns(2)
            with col_submit1:
                submitted = st.form_submit_button("📤 Submit Feedback", type="primary", use_container_width=True)
            with col_submit2:
                skip_feedback = st.form_submit_button("⏭️ Skip", use_container_width=True)
    
    if submitted or skip_feedback:
            if submitted:
                positive_aspects = []
                if easy_to_use:
                    positive_aspects.append("Easy to use")
                if accurate_responses:
                    positive_aspects.append("Accurate responses")
                if helpful_features:
                    positive_aspects.append("Helpful features")
                if fast_responses:
                    positive_aspects.append("Fast responses")
                if good_code_execution:
                    positive_aspects.append("Code execution worked well")
                if useful_documents:
                    positive_aspects.append("Document processing was useful")
                compiled_feedback = {
                    "rating": rating,
                    "positive_aspects": positive_aspects,
                    "improvement_areas": improvements,
                    "additional_comments": feedback_text,
                }
                feedback_success = save_feedback(st.session_state.session_id, __import__('json').dumps(compiled_feedback), rating)
                record_session_feedback(
                    st.session_state.session_id,
                    feedback_payload=compiled_feedback,
                )
                if feedback_success:
                    st.success("✅ Thank you for your feedback!")
                else:
                    st.error("❌ Error saving feedback, but thank you for trying!")
            elif st.session_state.session_id:
                record_session_feedback(
                    st.session_state.session_id,
                    feedback_payload={"skipped": True},
                )

            if st.session_state.session_id:
                analysis = analyze_session_feedback_loop(st.session_state.session_id)
                st.session_state.feedback_loop_last_analysis = analysis
                if analysis["failure_turn_count"] > 0:
                    root_cause_labels = ", ".join(
                        item.get("label", item.get("signature_id", "unknown"))
                        for item in analysis.get("root_cause_summary", [])[:3]
                    )
                    st.info(
                        f"🔁 Feedback loop analyzed {analysis['failure_turn_count']} failed turn(s) and updated the evolution profile."
                        + (f" Top root causes: {root_cause_labels}" if root_cause_labels else "")
                    )
                else:
                    st.success("🔁 Feedback loop found no failed turns to merge into the evolution profile.")
            if st.session_state.session_id:
                cleanup_success, chat_history_saved = end_session_cleanup(st.session_state.session_id)
                if cleanup_success:
                    if chat_history_saved:
                        st.success("🧹 Session data cleaned up successfully and chat history saved!")
                    else:
                        st.info("🧹 Session data cleaned up successfully")
                else:
                    st.warning("⚠️ Some session data may not have been cleaned up properly")
            st.session_state.session_ended = True
            st.session_state.show_feedback_screen = False
            st.session_state.show_chatbot = False
            st.session_state.api_key_validated = False
            st.session_state.openai_api_key = None
            st.session_state.chatbot = None
            st.session_state.session_id = None
            st.session_state.chat_history = []
            st.session_state.documents_processed = False
            st.session_state.code_outputs = {}
            st.session_state.edited_codes = {}
            st.session_state.code_reset_counters = {}
            st.session_state.pending_error_fix = []
            st.session_state.code_analyses = {}  # Clear AI analyses
            st.session_state.active_andes_case = None
            st.session_state.persistent_docs = set()
            st.session_state.session_docs = set()
            st.balloons()
            with st.spinner("Redirecting to home page..."):
                import time
                time.sleep(2)
            st.rerun()


def show_introduction_screen():
    from .auth import validate_openai_api_key
    from .chatbot_factory import DEFAULT_BASE_MODEL, DEFAULT_FINETUNED_MODEL

    st.markdown("# Power Flow Agent")
    st.markdown("### A Tractable and Self-Evolving Power-Flow Agent for Interactive Grid Analysis")
    st.markdown("---")
    st.markdown("### 🔑 Get Started")
    st.info(
        "To use this chatbot you need: (1) an OpenAI API key "
        "[get one here](https://platform.openai.com/api-keys), and "
        "(2) a chat model id. The default base model `gpt-4o-mini` works "
        "out of the box. To use the **Fine-tuned** or **Fine-tuned + RAG** "
        "modes, also paste your own fine-tune model id — fine-tuned "
        "models on OpenAI can only be invoked by the API key that owns "
        "them, so this app cannot ship a shared fine-tune."
    )

    with st.form("api_key_form"):
        st.markdown("#### Enter your OpenAI API Key")
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="sk-...",
            help="Your OpenAI API key will be used only for this session and is not stored permanently.",
            label_visibility="collapsed",
        )

        st.markdown("#### Base chat model")
        base_model = st.text_input(
            "Base model id",
            value=st.session_state.get("base_chat_model") or DEFAULT_BASE_MODEL,
            placeholder="gpt-4o-mini",
            help="Any OpenAI chat-completion model your API key can call (e.g. `gpt-4o-mini`, `gpt-4o`, `gpt-4.1`, `gpt-4.1-mini`).",
            label_visibility="collapsed",
        )

        st.markdown("#### Fine-tuned chat model id  *(optional — only if you want Fine-tuned modes)*")
        finetuned_model = st.text_input(
            "Fine-tuned model id",
            value=st.session_state.get("finetuned_chat_model") or DEFAULT_FINETUNED_MODEL,
            placeholder="ft:gpt-4o-mini-2024-07-18:your-org:your-suffix:abc123",
            help=(
                "OpenAI fine-tune model identifier (starts with `ft:`). "
                "Required for the Fine-tuned and Fine-tuned + RAG modes. "
                "Leave blank if you only want to use Base OpenAI / RAG."
            ),
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("🚀 Start Chatbot", type="primary", use_container_width=True)
        if submitted:
            if not api_key.strip():
                st.error("⚠️ Please enter your OpenAI API key.")
                return
            if not api_key.startswith('sk-'):
                st.error("⚠️ Invalid API key format. OpenAI API keys start with 'sk-'.")
                return
            if not base_model.strip():
                st.error("⚠️ Please enter a base chat model id (e.g. `gpt-4o-mini`).")
                return
            with st.spinner("🔍 Validating API key..."):
                is_valid, message = validate_openai_api_key(api_key.strip())
                if is_valid:
                    st.session_state.openai_api_key = api_key.strip()
                    st.session_state.api_key_validated = True
                    st.session_state.show_chatbot = True
                    st.session_state.base_chat_model = base_model.strip()
                    st.session_state.finetuned_chat_model = finetuned_model.strip()
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
