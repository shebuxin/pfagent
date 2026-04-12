import logging
import os

import streamlit as st

from src.session import (
    cleanup_inactive_sessions,
    initialize_session_state,
    update_session_activity,
)
from src.ui import show_feedback_screen, show_introduction_screen
from src.ui_chat_loop import (
    handle_chat_submission,
    render_chat_area,
    render_no_chatbot_banner,
)
from src.ui_error_fix import handle_pending_error_fixes
from src.ui_sidebar import render_sidebar

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
    
    # Sidebar (configuration, agent init, uploads, tools, session controls)
    # lives in src/ui_sidebar.py (Stage 5 refactor).
    render_sidebar()

    # Main chat interface
    if not st.session_state.chatbot:
        render_no_chatbot_banner()
        return

    user_input, submitted = render_chat_area()
    
    # Handle pending error fixes first
    handle_pending_error_fixes(
        repo_root=os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    )

    # Handle chat submission
    if submitted and user_input.strip():
        handle_chat_submission(user_input)


if __name__ == "__main__":
    main()
