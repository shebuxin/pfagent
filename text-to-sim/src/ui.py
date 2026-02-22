from datetime import datetime
from typing import List, Tuple

import streamlit as st
from streamlit_ace import st_ace

from .files import execute_python_code
from .analysis import create_analyzer


def load_css():
    """Load custom CSS for styling - now using native Streamlit components"""
    pass


def extract_python_code(text: str, msg_idx: int) -> List[Tuple[str, str]]:
    pattern = r'```python\s*\n(.*?)```'
    import re
    matches = re.findall(pattern, text, re.DOTALL)
    return [(f"code_{msg_idx}_{j}", code.strip()) for j, code in enumerate(matches)]


def display_chat_message(message: str, is_user: bool = True, message_index: int = 0):
    code_blocks = []
    if not is_user:
        code_blocks = extract_python_code(message, message_index)
        for _, code in code_blocks:
            message = message.replace(f"```python\n{code}\n```", "").replace("\n\n", "\n").strip()
    if is_user:
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
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
                
                # Action buttons row
                is_error = (
                    "Error" in output
                    or "Exception" in output
                    or "Traceback" in output
                    or output.startswith("Error (exit code")
                    or "SyntaxError" in output
                    or "NameError" in output
                    or "TypeError" in output
                    or "ValueError" in output
                    or "ImportError" in output
                    or "ModuleNotFoundError" in output
                    or "AttributeError" in output
                    or "KeyError" in output
                    or "IndexError" in output
                    or "FileNotFoundError" in output
                    or "PermissionError" in output
                    or "ZeroDivisionError" in output
                )
                
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
                            error_prompt = f"""I encountered an error while running Python code. Please analyze the error and provide a corrected version of the code.

**Original Code:**
```python
{current_code}
```

**Error Output:**
```
{output}
```

**Request:**
Please provide the corrected Python code that fixes this error. Focus on:
1. Identifying the root cause of the error
2. Providing syntactically correct code
3. Adding any necessary imports or dependencies
4. Including proper error handling if applicable
5. Adding comments explaining the fix if it's not obvious

Please respond with the corrected code in a ```python code block, and briefly explain what was wrong and how you fixed it."""
                            if 'pending_error_fix' not in st.session_state:
                                st.session_state.pending_error_fix = []
                            st.session_state.pending_error_fix.append(
                                {
                                    'prompt': error_prompt,
                                    'original_code_id': code_id,
                                    'failed_code': current_code,
                                    'error_output': output,
                                }
                            )
                            st.info("🔧 Error fix request added to queue. The AI will process it shortly...")
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


def update_conversation_history_with_edited_code(message_index: int, code_id: str, edited_code: str):
    if message_index < len(st.session_state.chat_history):
        user_msg, assistant_msg = st.session_state.chat_history[message_index]
        original_code_blocks = extract_python_code(assistant_msg, message_index)
        for i, (original_code_id, original_code) in enumerate(original_code_blocks):
            if original_code_id == code_id:
                updated_msg = assistant_msg.replace(
                    f"```python\n{original_code}\n```",
                    f"```python\n{edited_code}\n```",
                )
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
                if feedback_success:
                    st.success("✅ Thank you for your feedback!")
                else:
                    st.error("❌ Error saving feedback, but thank you for trying!")
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
            st.session_state.persistent_docs = set()
            st.session_state.session_docs = set()
            st.balloons()
            with st.spinner("Redirecting to home page..."):
                import time
                time.sleep(2)
            st.rerun()


def show_introduction_screen():
    from .auth import validate_openai_api_key

    st.markdown("# Power Flow Agent")
    st.markdown("### PowerFlow-Agent: Tool-Augmented LLM for Text-to-Power-Flow Simulation")
    st.markdown("---")
    st.markdown("### 🔑 Get Started")
    st.info("To use this chatbot, you'll need an OpenAI API key. [Get your API key here](https://platform.openai.com/api-keys)")
    
    with st.form("api_key_form"):
        st.markdown("#### Enter your OpenAI API Key")
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="sk-...",
            help="Your OpenAI API key will be used only for this session and is not stored permanently.",
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
            with st.spinner("🔍 Validating API key..."):
                is_valid, message = validate_openai_api_key(api_key.strip())
                if is_valid:
                    st.session_state.openai_api_key = api_key.strip()
                    st.session_state.api_key_validated = True
                    st.session_state.show_chatbot = True
                    st.success(f"✅ {message}")
                    st.rerun()
                else:
                    st.error(f"❌ {message}")
