import streamlit as st
import asyncio
import os
import json
import logging
from uuid import uuid4
from datetime import datetime

from src.session import (
    initialize_session_state,
    update_session_activity,
    cleanup_inactive_sessions,
    register_session_cleanup,
)
from src.ui import (
    show_feedback_screen,
    show_introduction_screen,
    display_chat_message,
)
from src.chatbot_factory import create_chatbot
from src.metadata import (
    process_csv_excel_files,
    get_csv_excel_column_names,
    save_csv_excel_file,
    display_csv_excel_files,
    display_saved_chat_histories,
    prepare_conversation_for_download,
)
from src.documents import (
    process_uploaded_files,
    process_documents_with_session_async,
)
from src.files import (
    display_file_section,
    save_uploaded_file,
)

# Logger for this module
logger = logging.getLogger(__name__)


def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Show feedback screen if session is ending
    if st.session_state.show_feedback_screen:
        show_feedback_screen()
        return
    
    # Show introduction screen if API key not provided or chatbot not shown yet
    if not st.session_state.show_chatbot or not st.session_state.api_key_validated:
        show_introduction_screen()
        return
    
    # Update session activity and cleanup inactive sessions
    update_session_activity()
    cleanup_inactive_sessions()
    
    # Sidebar for configuration and document upload
    with st.sidebar:
        # Show API key status
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
        
        st.markdown("---")
        st.markdown("# ⚙️ Configuration")
        
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
        
        st.markdown("---")
        st.markdown("## 🤖 Agent Type")
        
        # Chatbot type selection
        chatbot_type = st.selectbox(
            "Select Agent Type",
            options=["GraphRAG", "RAG", "No RAG"],
            index=0,
            help="Choose the type of agent:\n"
                 "• GraphRAG: Uses knowledge graphs + vector search (requires Neo4j)\n"
                 "• RAG: Uses only vector search with FAISS\n"
                 "• No RAG: Simple OpenAI agent without document processing"
        )
        
        # Store chatbot type in session state
        st.session_state.chatbot_type = chatbot_type
        
        # Show info about selected chatbot type
        if chatbot_type == "GraphRAG":
            st.info("🔗 **GraphRAG**: Combines vector search with knowledge graphs for enhanced understanding")
        elif chatbot_type == "RAG":
            st.info("🔍 **RAG**: Uses vector similarity search to find relevant information from documents")
        else:  # No RAG
            st.info("💬 **No RAG**: Simple conversation without document context")
        
        st.markdown("---")
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
        
        # Show current status if chatbot is initialized
        if st.session_state.chatbot:
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
        
        # Initialize chatbot button
        if st.button("🚀 Initialize Agent", type="primary", use_container_width=True):
            with st.spinner(f"Initializing {chatbot_type} agent..."):
                st.session_state.session_id = str(uuid4())
                
                # Create chatbot based on selected type
                chatbot = create_chatbot(
                    data_directory=f"./code_executions/{st.session_state.session_id}/data",
                    openai_api_key=st.session_state.openai_api_key,
                    chatbot_type=chatbot_type
                )
                
                if chatbot:
                    # Load system prompt with custom instructions
                    chatbot.load_system_prompt(
                        session_id=st.session_state.session_id,
                        custom_instructions=st.session_state.get('custom_system_prompt', '')
                    )
                    
                    st.session_state.chatbot = chatbot
                    # Store the prompt that was applied
                    st.session_state.last_applied_prompt = st.session_state.get('custom_system_prompt', '')

                    os.makedirs("code_executions", exist_ok=True)
                    os.makedirs(f"./code_executions/{st.session_state.session_id}", exist_ok=True)
                    os.makedirs(f"./code_executions/{st.session_state.session_id}/data", exist_ok=True)
                    
                    # Register session cleanup
                    register_session_cleanup()
                    cleanup_inactive_sessions()
                    
                    st.success(f"✅ {chatbot_type} agent initialized!")
                else:
                    st.error(f"❌ Failed to initialize {chatbot_type} agent")
        
        st.markdown("---")
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
        
        st.markdown("---")
        
        # Document upload section (only show for RAG/GraphRAG)
        if st.session_state.get('chatbot_type', 'GraphRAG') in ['RAG', 'GraphRAG']:
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
            # Show info when No RAG is selected
            st.info("💬 **No RAG Mode Selected**: Document upload is not available in No RAG mode. Switch to 'RAG' or 'GraphRAG' to enable document processing.")
        
        # Status display
        if st.session_state.documents_processed:
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
        
        st.markdown("---")
        
        # Additional features
        st.markdown("# 🔧 Tools")
        
        # File upload for code execution (only show when chatbot is initialized)
        if st.session_state.session_id and st.session_state.chatbot:
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
        
        st.markdown("---")
        
        # Input Files Section (only show when chatbot is initialized)
        if st.session_state.session_id and st.session_state.chatbot:
            display_file_section("📥 My Files", f"./code_executions/{st.session_state.session_id}/data", "file")

            # Output Files Section  
            # display_file_section("📤 Output Files", f"./code_executions/{st.session_state.session_id}/data/output", "output")
            
            st.markdown("---")
            
            # CSV/Excel files section
            display_csv_excel_files()
            
            # Saved chat histories section
            display_saved_chat_histories()
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.code_outputs = {}
            st.session_state.edited_codes = {}
            st.session_state.code_reset_counters = {}
            st.session_state.pending_error_fix = []
            st.session_state.code_analyses = {}  # Clear AI analyses
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
    
    # Main chat interface
    if not st.session_state.chatbot:
        st.markdown("# LLM Sandbox")
        st.markdown("**Ready to chat! Initialize the agent to get started.**")
        st.info("ℹ️ Please initialize the agent using the sidebar")
        return
    
    # Chat header
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Show API key status in header
        if st.session_state.openai_api_key:
            masked_key = st.session_state.openai_api_key[:8] + "..." + st.session_state.openai_api_key[-4:]
            st.caption(f"🔑 {masked_key}")
    
    with col2:
        st.markdown("# LLM Sandbox")
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
    
    # Handle pending error fixes first
    if st.session_state.get('pending_error_fix') and len(st.session_state.pending_error_fix) > 0:
        error_fix_request = st.session_state.pending_error_fix.pop(0)
        
        with st.spinner("🔧 AI is fixing the error..."):
            try:
                # Get response from chatbot for error fixing
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    st.session_state.chatbot.chat(error_fix_request['prompt'])
                )
                loop.close()
                
                # Add the error fix conversation to chat history
                st.session_state.chat_history.append((
                    f"🔧 **Error Fix Request:** Please fix this code error:\n\n{error_fix_request['prompt']}", 
                    response
                ))
                
                # Show success message with more context
                st.success("✅ AI has analyzed the error and provided a fix. Check the new response above!")
                st.info("💡 You can run the corrected code by clicking the ▶ Run button on the new code block.")
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating fix: {str(e)}")
                logger.error(f"Error fix generation error: {str(e)}", exc_info=True)
    
    # Handle chat submission
    if submitted and user_input.strip():
        with st.spinner("Generating..."):
            try:
                # Get response from chatbot (async)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(
                    st.session_state.chatbot.chat(user_input)
                )
                loop.close()
                
                # Add to chat history
                st.session_state.chat_history.append((user_input, response))
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                # Log more detailed error for debugging
                logger.error(f"Chat error details: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()