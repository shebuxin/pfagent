import os
import shutil
from datetime import datetime

import streamlit as st

from .metadata import cleanup_session_csv_excel_files, save_chat_history


def initialize_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = None
    if 'api_key_validated' not in st.session_state:
        st.session_state.api_key_validated = False
    if 'show_chatbot' not in st.session_state:
        st.session_state.show_chatbot = False
    if 'show_feedback_screen' not in st.session_state:
        st.session_state.show_feedback_screen = False
    if 'session_ended' not in st.session_state:
        st.session_state.session_ended = False

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = None
    if 'code_outputs' not in st.session_state:
        st.session_state.code_outputs = {}
    if 'edited_codes' not in st.session_state:
        st.session_state.edited_codes = {}
    if 'code_reset_counters' not in st.session_state:
        st.session_state.code_reset_counters = {}
    if 'pending_error_fix' not in st.session_state:
        st.session_state.pending_error_fix = []
    if 'refresh_files' not in st.session_state:
        st.session_state.refresh_files = False
    if 'persistent_docs' not in st.session_state:
        st.session_state.persistent_docs = set()
    if 'session_docs' not in st.session_state:
        st.session_state.session_docs = set()
    if 'session_cleanup_registered' not in st.session_state:
        st.session_state.session_cleanup_registered = False
    if 'csv_excel_metadata' not in st.session_state:
        st.session_state.csv_excel_metadata = {}
    if 'pending_csv_excel_files' not in st.session_state:
        st.session_state.pending_csv_excel_files = []
    if 'persistent_chat_history' not in st.session_state:
        st.session_state.persistent_chat_history = False
    if 'custom_system_prompt' not in st.session_state:
        st.session_state.custom_system_prompt = ""
    if 'last_applied_prompt' not in st.session_state:
        st.session_state.last_applied_prompt = ""
    if 'code_compilation_check' not in st.session_state:
        st.session_state.code_compilation_check = True
    if 'max_compilation_retries' not in st.session_state:
        st.session_state.max_compilation_retries = 2
    if 'active_andes_case' not in st.session_state:
        st.session_state.active_andes_case = None


def register_session_cleanup():
    """Register session cleanup using Streamlit's session state"""
    if not st.session_state.get('session_cleanup_registered', False):
        st.session_state.session_cleanup_registered = True
        if 'active_sessions' not in st.session_state:
            st.session_state.active_sessions = {}
        if st.session_state.session_id:
            st.session_state.active_sessions[st.session_state.session_id] = {
                'chatbot': st.session_state.chatbot,
                'last_active': datetime.now(),
                'session_docs': st.session_state.session_docs.copy()
            }


def cleanup_inactive_sessions():
    """Clean up sessions that have been inactive for too long"""
    if 'active_sessions' not in st.session_state:
        return
    current_time = datetime.now()
    inactive_sessions = []
    for session_id, session_info in st.session_state.active_sessions.items():
        if (current_time - session_info['last_active']).total_seconds() > 86400:
            if session_id != st.session_state.get('session_id'):
                inactive_sessions.append(session_id)
    for session_id in inactive_sessions:
        session_info = st.session_state.active_sessions[session_id]
        try:
            if session_info['chatbot'] and session_info['session_docs']:
                session_info['chatbot'].cleanup_session(session_id)
            cleanup_session_csv_excel_files(session_id)
        except Exception as e:
            print(f"Error cleaning up session {session_id}: {e}")
        del st.session_state.active_sessions[session_id]


def update_session_activity():
    """Update the last activity time for current session"""
    if 'active_sessions' in st.session_state and st.session_state.get('session_id'):
        if st.session_state.session_id in st.session_state.active_sessions:
            st.session_state.active_sessions[st.session_state.session_id]['last_active'] = datetime.now()


def end_session_cleanup(session_id):
    """Clean up session data when user explicitly ends session"""
    try:
        chat_history_saved = False
        if st.session_state.get('persistent_chat_history', False) and st.session_state.chat_history:
            success, file_path = save_chat_history(session_id, st.session_state.chat_history)
            if success:
                chat_history_saved = True
                print(f"Chat history saved to: {file_path}")
        if st.session_state.chatbot and st.session_state.session_docs:
            st.session_state.chatbot.cleanup_session(session_id)
        cleanup_session_csv_excel_files(session_id)
        session_code_dir = f"./code_executions/{session_id}"
        if os.path.exists(session_code_dir):
            shutil.rmtree(session_code_dir)
        if 'active_sessions' in st.session_state and session_id in st.session_state.active_sessions:
            del st.session_state.active_sessions[session_id]
        return True, chat_history_saved
    except Exception as e:
        print(f"Error during session cleanup: {str(e)}")
        return False, False
