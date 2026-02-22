import os
import json
import re
from datetime import datetime
from typing import List, Tuple, Dict
from uuid import uuid4

import pandas as pd
import streamlit as st


def create_directories():
    os.makedirs("./data_files", exist_ok=True)
    os.makedirs("./data_files/persistent", exist_ok=True)
    os.makedirs("./data_files/sessions", exist_ok=True)
    os.makedirs("./data_files/chat_history", exist_ok=True)


def get_metadata_file_path():
    return "./data_files/metadata.json"


def load_metadata():
    metadata_file = get_metadata_file_path()
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading metadata: {str(e)}")
            return {"persistent": {}, "sessions": {}}
    return {"persistent": {}, "sessions": {}}


def save_metadata(metadata):
    metadata_file = get_metadata_file_path()
    try:
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving metadata: {str(e)}")
        return False


def get_csv_excel_column_names(file_obj, filename):
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file_obj)
            return list(df.columns)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            excel_file = pd.ExcelFile(file_obj)
            sheet_names = excel_file.sheet_names
            columns_dict = {}
            for sheet in sheet_names:
                df = pd.read_excel(file_obj, sheet_name=sheet)
                columns_dict[sheet] = list(df.columns)
            return columns_dict
        return []
    except Exception as e:
        st.error(f"Error extracting columns from {filename}: {str(e)}")
        return []


def save_csv_excel_file(uploaded_file, is_persistent, session_id, user_description, columns_info):
    create_directories()
    if is_persistent:
        file_dir = "./data_files/persistent"
    else:
        file_dir = f"./data_files/sessions/{session_id}"
        os.makedirs(file_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_extension = uploaded_file.name.split('.')[-1]
    filename = f"{uuid4().hex}.{file_extension}"
    raw_id = filename.split('.')[0]
    if not re.match(r'^[A-Za-z]', raw_id):
        raw_id = 'f_' + raw_id
    file_id = re.sub(r'[^A-Za-z0-9_]','_', raw_id)
    file_path = os.path.join(file_dir, filename)
    try:
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        metadata = load_metadata()
        file_metadata = {
            "original_filename": uploaded_file.name,
            "saved_filename": filename,
            "file_path": file_path,
            "user_description": user_description,
            "columns_info": columns_info,
            "file_type": file_extension,
            "upload_timestamp": timestamp,
            "file_size": uploaded_file.size if hasattr(uploaded_file, 'size') else 0
        }
        if is_persistent:
            metadata["persistent"][file_id] = file_metadata
        else:
            if session_id not in metadata["sessions"]:
                metadata["sessions"][session_id] = {}
            metadata["sessions"][session_id][file_id] = file_metadata
        if save_metadata(metadata):
            return file_path, file_id
        else:
            os.remove(file_path)
            return None, None
    except Exception as e:
        st.error(f"Error saving file {uploaded_file.name}: {str(e)}")
        return None, None


def process_csv_excel_files(uploaded_files, is_persistent, session_id):
    csv_excel_files = []
    other_files = []
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(('.csv', '.xlsx', '.xls')):
            csv_excel_files.append(uploaded_file)
        else:
            other_files.append(uploaded_file)
    if csv_excel_files:
        st.session_state.pending_csv_excel_files = [(f, is_persistent, session_id) for f in csv_excel_files]
        return other_files, True
    return other_files, False


def get_saved_chat_histories():
    try:
        chat_dir = "./data_files/chat_history"
        if not os.path.exists(chat_dir):
            return []
        chat_files = []
        for filename in os.listdir(chat_dir):
            if filename.startswith('chat_history_') and filename.endswith('.json'):
                file_path = os.path.join(chat_dir, filename)
                try:
                    mtime = os.path.getmtime(file_path)
                    chat_files.append({
                        'filename': filename,
                        'path': file_path,
                        'modified': datetime.fromtimestamp(mtime)
                    })
                except Exception:
                    continue
        chat_files.sort(key=lambda x: x['modified'], reverse=True)
        return chat_files
    except Exception as e:
        print(f"Error getting chat histories: {str(e)}")
        return []


def delete_chat_history_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception as e:
        print(f"Error deleting chat history file: {str(e)}")
        return False


def prepare_conversation_for_download():
    conversation_data = {
        "timestamp": datetime.now().isoformat(),
        "total_messages": len(st.session_state.chat_history),
        "documents_processed": st.session_state.documents_processed,
        "conversation": [],
    }
    for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):
        # Local import to avoid circular dependency on UI-only helpers
        from .ui import extract_python_code
        code_blocks = extract_python_code(assistant_msg, i)
        code_data = []
        for code_id, code in code_blocks:
            code_info = {
                "code_id": code_id,
                "original_code": code,
                "edited_code": st.session_state.edited_codes.get(code_id, code),
                "output": st.session_state.code_outputs.get(code_id, None),
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


def save_chat_history(session_id, chat_history):
    try:
        chat_dir = "./data_files/chat_history"
        os.makedirs(chat_dir, exist_ok=True)
        conversation_data = prepare_conversation_for_download()
        conversation_data["session_id"] = session_id
        conversation_data["saved_timestamp"] = datetime.now().isoformat()
        chat_file = os.path.join(chat_dir, f"chat_history_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(chat_file, 'w') as f:
            json.dump(conversation_data, f, indent=2)
        return True, chat_file
    except Exception as e:
        print(f"Error saving chat history: {str(e)}")
        return False, None


def display_saved_chat_histories():
    chat_histories = get_saved_chat_histories()
    if not chat_histories:
        return
    st.markdown("## 💬 Saved Chat Histories")
    with st.expander(f"📚 Chat History Files ({len(chat_histories)})", expanded=False):
        for chat_file in chat_histories:
            col1, col2 = st.columns([3, 1])
            with col1:
                filename_parts = chat_file['filename'].replace('chat_history_', '').replace('.json', '')
                session_id_part = filename_parts.split('_')[0][:8]
                st.markdown(f"**Session:** `{session_id_part}...`")
                st.caption(f"📅 Saved: {chat_file['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                try:
                    file_size = os.path.getsize(chat_file['path'])
                    if file_size < 1024:
                        size_str = f"{file_size} B"
                    elif file_size < 1024 * 1024:
                        size_str = f"{file_size / 1024:.1f} KB"
                    else:
                        size_str = f"{file_size / (1024 * 1024):.1f} MB"
                    st.caption(f"📊 Size: {size_str}")
                except Exception:
                    pass
            with col2:
                try:
                    with open(chat_file['path'], 'r') as f:
                        chat_data = f.read()
                    st.download_button(
                        "📥",
                        data=chat_data,
                        file_name=chat_file['filename'],
                        mime="application/json",
                        help="Download chat history",
                        key=f"download_{chat_file['filename']}",
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                if st.button("🗑️", help="Delete chat history", key=f"delete_{chat_file['filename']}"):
                    if delete_chat_history_file(chat_file['path']):
                        st.success(f"Deleted {chat_file['filename']}")
                        st.rerun()
                    else:
                        st.error("Failed to delete file")
            st.markdown("---")


def display_csv_excel_files():
    metadata = load_metadata()
    if not metadata["persistent"] and not metadata["sessions"]:
        return
    st.markdown("## 📊 CSV/Excel Files")
    if metadata["persistent"]:
        with st.expander(f"🔒 Persistent Files ({len(metadata['persistent'])})", expanded=False):
            for filename, file_info in metadata['persistent'].items():
                st.markdown(f"**📄 {file_info['original_filename']}**")
                st.caption(f"Saved as: {filename}")
                st.markdown(f"*Description:* {file_info['user_description']}")
                if isinstance(file_info['columns_info'], dict):
                    st.markdown("*Columns:*")
                    for sheet, columns in file_info['columns_info'].items():
                        st.markdown(f"  - {sheet}: {', '.join(columns)}")
                else:
                    st.markdown(f"*Columns:* {', '.join(file_info['columns_info'])}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Delete", key=f"del_persistent_{filename}"):
                        if delete_csv_excel_file(filename, True, None):
                            st.success(f"✅ Deleted {filename}")
                            st.rerun()
                with col2:
                    try:
                        with open(file_info['file_path'], 'rb') as f:
                            file_content = f.read()
                        st.download_button(
                            "⬇️ Download",
                            data=file_content,
                            file_name=file_info['original_filename'],
                            key=f"download_persistent_{filename}",
                        )
                    except Exception as e:
                        st.error(f"Error reading file: {str(e)}")
                st.markdown("---")
    current_session = st.session_state.get('session_id')
    if current_session and current_session in metadata["sessions"]:
        session_files = metadata["sessions"][current_session]
        if session_files:
            with st.expander(f"🔄 Session Files ({len(session_files)})", expanded=False):
                for filename, file_info in session_files.items():
                    st.markdown(f"**📄 {file_info['original_filename']}**")
                    st.caption(f"Saved as: {filename}")
                    st.markdown(f"*Description:* {file_info['user_description']}")
                    if isinstance(file_info['columns_info'], dict):
                        st.markdown("*Columns:*")
                        for sheet, columns in file_info['columns_info'].items():
                            st.markdown(f"  - {sheet}: {', '.join(columns)}")
                    else:
                        st.markdown(f"*Columns:* {', '.join(file_info['columns_info'])}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🗑️ Delete", key=f"del_session_{filename}"):
                            if delete_csv_excel_file(filename, False, current_session):
                                st.success(f"✅ Deleted {filename}")
                                st.rerun()
                    with col2:
                        try:
                            with open(file_info['file_path'], 'rb') as f:
                                file_content = f.read()
                            st.download_button(
                                "⬇️ Download",
                                data=file_content,
                                file_name=file_info['original_filename'],
                                key=f"download_session_{filename}",
                            )
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
                    st.markdown("---")


def delete_csv_excel_file(filename, is_persistent, session_id):
    try:
        metadata = load_metadata()
        if is_persistent:
            file_info = metadata["persistent"].get(filename)
            if file_info:
                if os.path.exists(file_info['file_path']):
                    os.remove(file_info['file_path'])
                del metadata["persistent"][filename]
        else:
            if session_id in metadata["sessions"]:
                file_info = metadata["sessions"][session_id].get(filename)
                if file_info:
                    if os.path.exists(file_info['file_path']):
                        os.remove(file_info['file_path'])
                    del metadata["sessions"][session_id][filename]
                    if not metadata["sessions"][session_id]:
                        del metadata["sessions"][session_id]
        return save_metadata(metadata)
    except Exception as e:
        st.error(f"Error deleting file: {str(e)}")
        return False


def cleanup_session_csv_excel_files(session_id):
    try:
        metadata = load_metadata()
        if session_id in metadata["sessions"]:
            session_files = metadata["sessions"][session_id]
            for filename, file_info in session_files.items():
                if os.path.exists(file_info['file_path']):
                    os.remove(file_info['file_path'])
            del metadata["sessions"][session_id]
            session_dir = f"./data_files/sessions/{session_id}"
            if os.path.exists(session_dir):
                try:
                    os.rmdir(session_dir)
                except OSError:
                    pass
            save_metadata(metadata)
        return True
    except Exception as e:
        print(f"Error cleaning up session CSV/Excel files: {str(e)}")
        return False


def save_feedback(session_id, feedback_text, rating):
    try:
        feedback_dir = "./data_files/feedback"
        os.makedirs(feedback_dir, exist_ok=True)
        feedback_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "rating": rating,
            "feedback": feedback_text,
            "user_agent": "streamlit_app",
        }
        feedback_file = os.path.join(feedback_dir, f"feedback_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(feedback_file, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        master_log = os.path.join(feedback_dir, "feedback_log.jsonl")
        with open(master_log, 'a') as f:
            f.write(json.dumps(feedback_data) + '\n')
        return True
    except Exception as e:
        print(f"Error saving feedback: {str(e)}")
        return False
