"""Streamlit sidebar: configuration, agent init, uploads, chat controls.

Extracted from main.py in Stage 5. The sidebar was a single 508-line
function; Stage 5.5 decomposes it into 13 per-section ``_render_*``
helpers. The public ``render_sidebar()`` entry point is now a short
orchestrator that makes the sidebar structure readable top-down.

Data flow: most sections only touch ``st.session_state`` (reads and
writes), but three values are threaded explicitly between helpers so
the dependency is visible at the orchestrator:

- ``custom_prompt``           (custom-instructions -> init-agent button)
- ``chatbot_type``            (selector -> init-agent button + header)
- ``code_check_enabled``,
  ``max_retries``             (compilation-settings -> status panel
                               + apply-settings button)

All other cross-section coordination flows through ``st.session_state``
because that's how the original code was wired.

This is a behavior-preserving decomposition: every ``st.*`` call,
every session_state key, every spinner/error message, the ``with
st.sidebar:`` context manager, and the ordering of separator /
section-header markdown calls are all byte-identical to the
pre-decomposition version.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from uuid import uuid4

import streamlit as st

from src.andes_manual import bootstrap_default_andes_manual
from src.chatbot_factory import create_chatbot
from src.codex_fixer import DEFAULT_CODEX_FIX_MODEL
from src.documents import (
    process_documents_with_session_async,
    process_uploaded_files,
)
from src.files import display_file_section, save_uploaded_file
from src.metadata import (
    display_csv_excel_files,
    display_saved_chat_histories,
    get_csv_excel_column_names,
    prepare_conversation_for_download,
    process_csv_excel_files,
    save_csv_excel_file,
)
from src.session import cleanup_inactive_sessions, register_session_cleanup


# --- API key ------------------------------------------------------------

def _render_api_key_section() -> None:
    st.markdown("# 🔑 API Configuration")
    if st.session_state.openai_api_key:
        masked_key = st.session_state.openai_api_key[:8] + "..." + st.session_state.openai_api_key[-4:]
        st.success(f"✅ API Key: {masked_key}")

        if st.button("🔄 Change API Key", use_container_width=True):
            st.session_state.show_chatbot = False
            st.session_state.api_key_validated = False
            st.session_state.openai_api_key = None
            st.session_state.chatbot = None
            st.rerun()


# --- Configuration group ------------------------------------------------

def _render_custom_instructions_section() -> str:
    """Render the custom-instructions text area.

    Returns the current custom_prompt value. Also writes it to
    ``st.session_state.custom_system_prompt`` so downstream code paths
    that read session state (e.g. the init-agent bootstrap) continue
    to work.
    """
    # Custom system prompt text box
    st.markdown("## 🎯 Custom Instructions")
    custom_prompt = st.text_area(
        "Additional instructions for the AI assistant:",
        value=st.session_state.get('custom_system_prompt', ''),
        height=120,
        placeholder="Examples:\n• 'Act as a Python expert focused on data science'\n• 'Be concise and provide code examples'\n• 'Focus on business analysis and insights'\n• 'Explain concepts in simple terms for beginners'\n• 'Prioritize security and best practices'",
        help="This text will be added to the AI's system prompt and will influence how it responds to your questions."
    )

    # Update session state when the text changes
    st.session_state.custom_system_prompt = custom_prompt

    # Show info about current custom prompt
    if custom_prompt.strip():
        st.info(f"📝 **Custom instructions active:** {len(custom_prompt)} characters")

        # Show preview of custom instructions
        with st.expander("👀 Preview Custom Instructions", expanded=False):
            st.markdown("**Your custom instructions:**")
            st.code(custom_prompt, language="text")
            st.caption("💡 These instructions will be applied when you initialize the agent")
    else:
        st.caption("💡 Add custom instructions to personalize the AI's behavior")

    # Show warning if chatbot is already initialized
    if st.session_state.chatbot and custom_prompt != st.session_state.get('last_applied_prompt', ''):
        st.warning("⚠️ To apply new custom instructions, you need to re-initialize the agent")

    return custom_prompt


def _render_chatbot_type_selector() -> str:
    """Render the agent-type selector and info banner.

    Returns the selected chatbot_type and mirrors it to
    ``st.session_state.chatbot_type``.
    """
    st.markdown("## 🤖 Agent Type")

    # Chatbot type selection
    chatbot_type = st.selectbox(
        "Select Agent Type",
        options=["RAG", "Base OpenAI", "Fine-tuned", "Fine-tuned + RAG"],
        index=0,
        help="Choose the type of agent:\n"
             "• RAG: Uses FAISS retrieval with a base OpenAI model\n"
             "• Base OpenAI: Uses your base OpenAI model without document retrieval\n"
             "• Fine-tuned: Uses your fine-tuned model without document retrieval\n"
             "• Fine-tuned + RAG: Uses your fine-tuned model with FAISS retrieval"
    )

    # Store chatbot type in session state
    st.session_state.chatbot_type = chatbot_type

    # Show info about selected chatbot type
    if chatbot_type == "RAG":
        st.info("🔍 **RAG**: Uses FAISS retrieval with a base OpenAI model")
    elif chatbot_type == "Fine-tuned + RAG":
        st.info("🔍 **Fine-tuned + RAG**: Uses FAISS retrieval with your fine-tuned model")
    elif chatbot_type == "Fine-tuned":
        st.info("🧪 **Fine-tuned**: Uses your fine-tuned model without document retrieval")
    else:  # Base OpenAI
        st.info("💬 **Base OpenAI**: Uses a base OpenAI model without document retrieval")

    return chatbot_type


def _render_code_compilation_settings() -> tuple[bool, int]:
    """Render the code-compilation-checking toggle + max-retries input.

    Returns ``(code_check_enabled, max_retries)``; mirrors them to
    session state. The companion _render_current_settings_status
    helper reuses both values to compare against the live chatbot
    config and render the apply-settings button.
    """
    st.markdown("## 🔍 Code Compilation Checking")

    # Code compilation checking toggle
    code_check_enabled = st.checkbox(
        "Enable code compilation checking",
        value=True,
        help="When enabled, the AI will check generated Python code for syntax errors and retry if needed"
    )

    # Max retries setting
    max_retries = st.number_input(
        "Maximum compilation retries",
        min_value=0,
        max_value=5,
        value=2,
        help="Number of times the AI will retry if code has compilation errors"
    )

    # Store settings in session state
    st.session_state.code_compilation_check = code_check_enabled
    st.session_state.max_compilation_retries = max_retries

    return code_check_enabled, max_retries


def _render_error_fixing_settings() -> None:
    """Render the 'Fix Error with AI' config block (Codex fixer +
    local validation settings). All values land in session state
    for ui_error_fix.handle_pending_error_fixes to read later.
    """
    st.markdown("## 🛠 Error Fixing")
    use_codex_error_fixer = st.checkbox(
        "Use Codex repo-aware fixer",
        value=st.session_state.get("use_codex_error_fixer", True),
        help="When enabled, 'Fix Error with AI' uses a dedicated Codex model with repository-aware retrieval and falls back to the current agent if needed.",
    )
    st.session_state.use_codex_error_fixer = use_codex_error_fixer

    codex_error_fix_model = st.text_input(
        "Error fixer model",
        value=st.session_state.get("codex_error_fix_model", DEFAULT_CODEX_FIX_MODEL),
        help="Best results are expected with a Codex-capable Responses API model such as gpt-5.2-codex.",
    ).strip()
    st.session_state.codex_error_fix_model = codex_error_fix_model or DEFAULT_CODEX_FIX_MODEL
    validate_error_fix_locally = st.checkbox(
        "Run fixed code locally before returning it",
        value=st.session_state.get("validate_error_fix_locally", True),
        help="After Codex proposes a fix, the app will execute it in the current session and feed any new error back for another repair attempt.",
    )
    st.session_state.validate_error_fix_locally = validate_error_fix_locally

    max_error_fix_validation_retries = st.number_input(
        "Local debug retries",
        min_value=0,
        max_value=3,
        value=int(st.session_state.get("max_error_fix_validation_retries", 2)),
        help="How many extra repair rounds to run when local validation still fails.",
    )
    st.session_state.max_error_fix_validation_retries = int(max_error_fix_validation_retries)
    st.caption("This setting only affects the 'Fix Error with AI' button. It does not change the main chat agent.")


def _render_current_settings_status(code_check_enabled: bool, max_retries: int) -> None:
    """Show whether the sidebar settings match the running chatbot's
    config, and expose an apply-settings button if they diverged.
    """
    if not st.session_state.chatbot:
        return

    current_check = st.session_state.chatbot.config.code_compilation_check
    current_retries = st.session_state.chatbot.config.max_compilation_retries

    if current_check != code_check_enabled or current_retries != max_retries:
        st.info("💡 Settings will be applied to current agent")
        if st.button("🔄 Apply Settings", use_container_width=True):
            st.session_state.chatbot.set_code_compilation_check(code_check_enabled)
            st.session_state.chatbot.set_max_compilation_retries(max_retries)
            st.success("✅ Settings applied!")
            st.rerun()
    else:
        status_icon = "✅" if current_check else "❌"
        st.success(f"{status_icon} Compilation checking: {'Enabled' if current_check else 'Disabled'} (Max retries: {current_retries})")


def _render_initialize_agent_button(chatbot_type: str) -> None:
    """Render the 'Initialize Agent' primary button + its async
    bootstrap handler (session setup, create_chatbot, ANDES manual
    preload, system prompt load).
    """
    if not st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
        return

    with st.spinner(f"Initializing {chatbot_type} agent..."):
        st.session_state.session_id = str(uuid4())
        st.session_state.active_andes_case = None
        st.session_state.default_manual_loaded = False
        st.session_state.default_manual_doc_count = 0
        st.session_state.documents_processed = False
        st.session_state.processing_status = None

        # Create chatbot based on selected type
        chatbot = create_chatbot(
            data_directory=f"./code_executions/{st.session_state.session_id}/data",
            openai_api_key=st.session_state.openai_api_key,
            chatbot_type=chatbot_type
        )

        if chatbot:
            os.makedirs("code_executions", exist_ok=True)
            os.makedirs(f"./code_executions/{st.session_state.session_id}", exist_ok=True)
            os.makedirs(f"./code_executions/{st.session_state.session_id}/data", exist_ok=True)

            default_manual_doc_count = 0
            bootstrap_failed = False
            loop = None

            try:
                if chatbot_type in {"RAG", "Fine-tuned + RAG"}:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    default_manual_doc_count = loop.run_until_complete(
                        bootstrap_default_andes_manual(chatbot)
                    )

                chatbot.load_system_prompt(
                    session_id=st.session_state.session_id,
                    custom_instructions=st.session_state.get('custom_system_prompt', '')
                )
            except Exception as e:
                bootstrap_failed = True
                st.error(f"❌ Failed to preload the official ANDES manual: {str(e)}")
            finally:
                if loop is not None:
                    loop.close()
                    asyncio.set_event_loop(None)

            if bootstrap_failed:
                chatbot = None

        if chatbot:
            st.session_state.chatbot = chatbot
            st.session_state.default_manual_loaded = chatbot_type in {"RAG", "Fine-tuned + RAG"}
            st.session_state.default_manual_doc_count = default_manual_doc_count
            if st.session_state.default_manual_loaded:
                st.session_state.documents_processed = True
                st.session_state.processing_status = (
                    f"Loaded the full official ANDES manual by default ({default_manual_doc_count} pages available for retrieval)"
                )

            # Store the prompt that was applied
            st.session_state.last_applied_prompt = st.session_state.get('custom_system_prompt', '')

            # Register session cleanup
            register_session_cleanup()
            cleanup_inactive_sessions()

            st.success(f"✅ {chatbot_type} agent initialized!")
        else:
            st.error(f"❌ Failed to initialize {chatbot_type} agent")


def _render_chat_history_settings() -> None:
    st.markdown("## 💬 Chat History")

    # Persistent chat history checkbox
    persistent_chat = st.checkbox(
        "💬 Persistent Chat History",
        value=st.session_state.get('persistent_chat_history', False),
        help="If checked, your chat history will be saved as a file on the server after session ends. Otherwise, it will be deleted."
    )

    # Update session state
    st.session_state.persistent_chat_history = persistent_chat

    # Info about chat persistence
    if persistent_chat:
        st.info("💾 **Chat:** Will be saved")
    else:
        st.warning("🗑️ **Chat:** Will be deleted")


# --- Document upload + status ------------------------------------------

def _render_document_upload_section() -> None:
    """Render the document-upload block.

    Only renders the upload widgets for RAG-enabled chatbot types;
    otherwise shows a short info banner explaining why it's hidden.
    """
    # Document upload section (only show for RAG-enabled modes)
    if st.session_state.get('chatbot_type', 'RAG') in ['RAG', 'Fine-tuned + RAG']:
        if st.session_state.get('default_manual_loaded', False):
            st.success(
                f"📘 Full official ANDES manual loaded by default ({st.session_state.get('default_manual_doc_count', 0)} pages available for retrieval)"
            )

        st.markdown("# 📁 Document Upload")

        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'pdf', 'csv', 'xlsx', 'xls', 'py', 'cpp', 'c', 'h', 'hpp', 'rs', 'js', 'ts', 'jsx', 'tsx', 'java', 'go', 'rb', 'php', 'swift', 'kt', 'scala', 'r', 'sql', 'html', 'css', 'json', 'xml', 'yaml', 'yml', 'md', 'ipynb', 'sh', 'bat', 'ps1', 'pl', 'lua', 'dart', 'cs', 'vb', 'f90', 'f95', 'jl', 'clj', 'hs', 'elm', 'nim', 'zig', 'v', 'toml', 'ini', 'cfg', 'conf'],
            accept_multiple_files=True,
            help="Upload documents and code files to add to the knowledge base. Supports text documents, PDFs, spreadsheets, and code files in various programming languages including Python, C++, Rust, JavaScript, Jupyter notebooks, and more."
        )

        # Persistent document checkbox
        persistent_docs = st.checkbox(
            "🔒 Persistent Documents",
            value=False,
            help="If checked, documents will remain in the knowledge base even after session ends. Otherwise, they will be removed when you disconnect."
        )

        # Info about document persistence
        if persistent_docs:
            st.info("📌 **Documents:** Persistent")
        else:
            st.warning("⚠️ **Documents:** Temporary")

        if uploaded_files and st.session_state.chatbot:
            # Check if any files are CSV/Excel and need metadata
            other_files, has_csv_excel = process_csv_excel_files(uploaded_files, persistent_docs, st.session_state.session_id)

            if has_csv_excel and st.session_state.pending_csv_excel_files:
                # Show CSV/Excel metadata collection interface
                st.markdown("### 📊 CSV/Excel File Information")
                st.info("Please provide additional information about your CSV/Excel files:")

                csv_excel_metadata = {}
                all_metadata_collected = True

                for i, (uploaded_file, is_persistent, session_id) in enumerate(st.session_state.pending_csv_excel_files):
                    with st.expander(f"📄 {uploaded_file.name}", expanded=True):
                        # Get column information
                        columns_info = get_csv_excel_column_names(uploaded_file, uploaded_file.name)

                        # Show column information
                        st.markdown("**Column Information:**")
                        if isinstance(columns_info, dict):  # Excel with multiple sheets
                            for sheet_name, columns in columns_info.items():
                                st.markdown(f"*Sheet '{sheet_name}':*")
                                st.code(", ".join(columns))
                        else:  # CSV or single sheet
                            st.code(", ".join(columns_info))

                        # User description input
                        user_description = st.text_area(
                            f"Describe the content and purpose of {uploaded_file.name}:",
                            key=f"desc_{i}_{uploaded_file.name}",
                            placeholder="e.g., This file contains customer data with contact information and purchase history...",
                            height=100
                        )

                        if not user_description.strip():
                            all_metadata_collected = False
                            st.warning("⚠️ Please provide a description for this file.")

                        csv_excel_metadata[uploaded_file.name] = {
                            'file': uploaded_file,
                            'description': user_description,
                            'columns': columns_info,
                            'is_persistent': is_persistent,
                            'session_id': session_id
                        }

                # Process CSV/Excel files if all metadata is collected
                if all_metadata_collected and st.button("💾 Save CSV/Excel Files", type="primary"):
                    with st.spinner("Saving CSV/Excel files..."):
                        saved_files = []
                        for file_name, file_info in csv_excel_metadata.items():
                            file_path, saved_filename = save_csv_excel_file(
                                file_info['file'],
                                file_info['is_persistent'],
                                file_info['session_id'],
                                file_info['description'],
                                file_info['columns']
                            )
                            if file_path:
                                saved_files.append((file_name, file_path))

                        if saved_files:
                            st.success(f"✅ Saved {len(saved_files)} CSV/Excel files with metadata!")
                            st.session_state.pending_csv_excel_files = []  # Clear pending files

                            # Reload system prompt to include new CSV/Excel files
                            st.session_state.chatbot.load_system_prompt(
                                st.session_state.session_id,
                                st.session_state.get('custom_system_prompt', '')
                            )

                            # Show saved files
                            st.markdown("**Saved Files:**")
                            for original_name, file_path in saved_files:
                                st.markdown(f"- {original_name} → `{file_path}`")
                        else:
                            st.error("❌ Failed to save CSV/Excel files")

                # If there are other files, show option to process them
                if other_files:
                    st.markdown("---")
                    st.markdown("### 📄 Other Documents")
                    st.info(f"Found {len(other_files)} other document(s) ready for processing.")

            # Process non-CSV/Excel files or all files if no CSV/Excel metadata needed
            files_to_process = other_files if has_csv_excel else uploaded_files

            if files_to_process and st.button("📊 Process Documents", type="secondary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    try:
                        documents, doc_ids = process_uploaded_files(files_to_process)

                        if documents:
                            # Create persistent flags for all documents
                            persistent_flags = [persistent_docs] * len(documents)

                            # Track document persistence in session state
                            for doc_id in doc_ids:
                                if persistent_docs:
                                    st.session_state.persistent_docs.add(doc_id)
                                else:
                                    st.session_state.session_docs.add(doc_id)

                            # Run async processing with session tracking
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(
                                process_documents_with_session_async(
                                    st.session_state.chatbot,
                                    documents,
                                    doc_ids,
                                    st.session_state.session_id,
                                    persistent_flags
                                )
                            )
                            loop.close()

                            st.session_state.documents_processed = True
                            persistence_status = "persistent" if persistent_docs else "temporary"
                            st.session_state.processing_status = f"Successfully processed {len(documents)} documents as {persistence_status}"
                            st.success(f"✅ Documents processed as {persistence_status}!")
                        else:
                            st.error("❌ No documents processed")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
    else:
        # Show info when a non-RAG mode is selected
        st.info("💬 **Non-RAG Mode Selected**: Document upload is not available. Switch to 'RAG' or 'Fine-tuned + RAG' to enable document processing.")


def _render_documents_ready_status() -> None:
    """Render the 'Documents Ready' summary panel (persistent vs
    temporary doc counts, chat-history persistence status).
    """
    if not st.session_state.documents_processed:
        return

    st.success("✅ Documents Ready")

    # Show document persistence status
    if st.session_state.persistent_docs or st.session_state.session_docs:
        st.markdown("**Document Status:**")
        if st.session_state.persistent_docs:
            st.markdown(f"🔒 **Persistent:** {len(st.session_state.persistent_docs)} documents")
        if st.session_state.session_docs:
            st.markdown(f"🔄 **Temporary:** {len(st.session_state.session_docs)} documents")
        st.caption("Temporary documents will be removed when you end the session")

    # Show chat history persistence status
    if st.session_state.chat_history:
        st.markdown("**Chat History Status:**")
        if st.session_state.get('persistent_chat_history', False):
            st.markdown(f"💾 **Will be saved:** {len(st.session_state.chat_history)} messages")
            st.caption("Chat history will be saved as a file when you end the session")
        else:
            st.markdown(f"🗑️ **Will be deleted:** {len(st.session_state.chat_history)} messages")
            st.caption("Chat history will be deleted when you end the session")


# --- Tools group --------------------------------------------------------

def _render_code_execution_uploads() -> None:
    """Render the code-execution file-upload panel.

    Only visible when a chatbot session is live.
    """
    if not (st.session_state.session_id and st.session_state.chatbot):
        return

    st.markdown("## 📁 Code Execution Files")
    st.markdown("Upload files needed for Python code execution:")

    code_files = st.file_uploader(
        "Upload files for code execution",
        accept_multiple_files=True,
        help="Upload files that your Python code needs to read",
        key="code_files"
    )

    if code_files:
        for uploaded_file in code_files:
            target_path = f"./code_executions/{st.session_state.session_id}/data/{uploaded_file.name}"

            if st.button(f"💾 Save {uploaded_file.name}", key=f"save_{uploaded_file.name}"):
                try:
                    save_uploaded_file(uploaded_file, target_path)
                    st.success(f"✅ Saved {uploaded_file.name} to {target_path}")
                    st.session_state.refresh_files = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving {uploaded_file.name}: {str(e)}")


def _render_file_sections() -> None:
    """Render the My Files / Output Files / CSV-Excel files /
    saved chat-histories file listings. Only visible once a
    chatbot session is live.
    """
    if not (st.session_state.session_id and st.session_state.chatbot):
        return

    display_file_section("📥 My Files", f"./code_executions/{st.session_state.session_id}/data", "file")

    # Output Files Section
    display_file_section("📤 Output Files", f"./code_executions/{st.session_state.session_id}/data/output", "output")

    st.markdown("---")

    # CSV/Excel files section
    display_csv_excel_files()

    # Saved chat histories section
    display_saved_chat_histories()


# --- Session controls --------------------------------------------------

def _render_session_controls() -> None:
    """Render Clear-Chat / End-Session / Show-Summary / Download
    Conversation buttons plus the trailing session-info banners.
    """
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.code_outputs = {}
        st.session_state.edited_codes = {}
        st.session_state.code_reset_counters = {}
        st.session_state.pending_error_fix = []
        st.session_state.code_analyses = {}  # Clear AI analyses
        st.session_state.active_andes_case = None
        st.success("Chat cleared!")

    # End Session button
    if st.button("🛑 End Session", use_container_width=True, type="secondary"):
        st.session_state.show_feedback_screen = True
        st.session_state.show_chatbot = False
        st.rerun()

    # Session information
    if st.session_state.chatbot and st.session_state.session_docs:
        st.info(f"📄 You have {len(st.session_state.session_docs)} temporary documents that will be removed when you end the session.")

    # Chat history information
    if st.session_state.chat_history and not st.session_state.get('persistent_chat_history', False):
        st.warning(f"💬 Your chat history ({len(st.session_state.chat_history)} messages) will be deleted when you end the session. Enable 'Persistent Chat History' above to save it.")

    if st.button("📋 Show Summary", use_container_width=True) and st.session_state.chatbot:
        summary = st.session_state.chatbot.get_conversation_summary()
        st.text_area("Conversation Summary", summary, height=200)

    # Download conversation history button
    if st.session_state.chat_history:
        conversation_data = prepare_conversation_for_download()
        st.download_button(
            label="📥 Download Conversation",
            data=json.dumps(conversation_data, indent=2),
            file_name=f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
            help="Download the complete conversation history including code edits and outputs"
        )


# --- Orchestrator ------------------------------------------------------

def render_sidebar() -> None:
    """Render the entire configuration + tools sidebar."""
    # Sidebar for configuration and document upload
    with st.sidebar:
        _render_api_key_section()

        st.markdown("---")
        st.markdown("# ⚙️ Configuration")

        custom_prompt = _render_custom_instructions_section()
        # Signal that custom_prompt flows through st.session_state to
        # the init-agent bootstrap; keep the local var live so the
        # reference is visible at the orchestrator.
        _ = custom_prompt

        st.markdown("---")
        chatbot_type = _render_chatbot_type_selector()

        st.markdown("---")
        code_check_enabled, max_retries = _render_code_compilation_settings()

        st.markdown("---")
        _render_error_fixing_settings()

        _render_current_settings_status(code_check_enabled, max_retries)

        _render_initialize_agent_button(chatbot_type)

        st.markdown("---")
        _render_chat_history_settings()

        st.markdown("---")

        _render_document_upload_section()

        _render_documents_ready_status()

        st.markdown("---")

        # Additional features
        st.markdown("# 🔧 Tools")

        _render_code_execution_uploads()

        st.markdown("---")

        _render_file_sections()

        st.markdown("---")

        _render_session_controls()
