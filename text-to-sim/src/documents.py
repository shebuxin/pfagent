import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import PyPDF2
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


def process_uploaded_files(uploaded_files):
    """Process uploaded files and extract text content"""
    documents = []
    doc_ids = []
    
    for uploaded_file in uploaded_files:
        uploaded_file: UploadedFile = uploaded_file
        try:
            # Read file content based on file type
            if uploaded_file.type == "text/plain":
                content = str(uploaded_file.read(), "utf-8")
            elif uploaded_file.name.endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                content = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            elif uploaded_file.name.endswith('.xlsx'):
                sheet_names = pd.ExcelFile(uploaded_file).sheet_names
                content = ""
                for sheet in sheet_names:
                    df = pd.read_excel(uploaded_file, sheet_name=sheet)
                    content += df.to_csv(index=False) + "\n"
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                content = df.to_csv(index=False)
            else:
                # Try to read as text
                content = str(uploaded_file.read(), "utf-8")
            
            documents.append(content)
            doc_ids.append(f"doc_{uploaded_file.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
    
    return documents, doc_ids


async def process_documents_async(chatbot, documents, doc_ids):
    """Async wrapper for document processing"""
    await chatbot.process_documents(documents, doc_ids)


async def process_documents_with_session_async(chatbot, documents, doc_ids, session_id, persistent_flags):
    """Async wrapper for document processing with session tracking"""
    await chatbot.process_documents(documents, doc_ids, session_id, persistent_flags)
