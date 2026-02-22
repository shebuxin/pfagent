import os
import streamlit as st
from src.chatbots.openai.graphrag_chatbot import GraphRAGChatbot, GraphRAGConfig
from src.chatbots.openai.rag_chatbot import RAGChatbot, RAGConfig
from src.chatbots.openai.simple_chatbot import SimpleChatbot, SimpleChatConfig

DEFAULT_BASE_MODEL = os.environ.get("OPENAI_BASE_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_FINETUNED_MODEL = os.environ.get(
    "OPENAI_FINETUNED_MODEL",
    "ft:gpt-4.1-nano-2025-04-14:personal:test-finetuning:DBSr7e2r",
)


def create_chatbot(data_directory, openai_api_key, chatbot_type="GraphRAG"):
    """Create and initialize the chatbot based on selected type"""
    try:
        if chatbot_type == "GraphRAG":
            print("Creating GraphRAG Chatbot...")
            config = GraphRAGConfig(
                openai_api_key=openai_api_key,
                neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
                neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
                neo4j_password=os.environ.get("NEO4J_PASSWORD"),
                data_directory=data_directory,
                code_compilation_check=st.session_state.get('code_compilation_check', True),
                max_compilation_retries=st.session_state.get('max_compilation_retries', 2)
            )
            
            if not all([config.openai_api_key, config.neo4j_uri, config.neo4j_user, config.neo4j_password]):
                st.error("Missing required Neo4j environment variables. Please check your configuration.")
                return None
                
            return GraphRAGChatbot(config)
            
        elif chatbot_type in {"Fine-tuned + RAG", "RAG"}:
            print("Creating RAG Chatbot...")
            config = RAGConfig(
                openai_api_key=openai_api_key,
                chat_model=DEFAULT_FINETUNED_MODEL if chatbot_type == "Fine-tuned + RAG" else os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
                data_directory=data_directory,
                code_compilation_check=st.session_state.get('code_compilation_check', True),
                max_compilation_retries=st.session_state.get('max_compilation_retries', 2)
            )
            
            if not config.openai_api_key:
                st.error("Missing OpenAI API key.")
                return None
                
            return RAGChatbot(config)
            
        elif chatbot_type in {"Base OpenAI", "No RAG", "Fine-tuned"}:
            print("Creating Simple Chatbot...")
            config = SimpleChatConfig(
                openai_api_key=openai_api_key,
                chat_model=DEFAULT_FINETUNED_MODEL if chatbot_type == "Fine-tuned" else DEFAULT_BASE_MODEL,
                code_compilation_check=st.session_state.get('code_compilation_check', True),
                max_compilation_retries=st.session_state.get('max_compilation_retries', 2)
            )
            
            if not config.openai_api_key:
                st.error("Missing OpenAI API key.")
                return None
                
            return SimpleChatbot(config)
            
        else:
            st.error(f"Unknown chatbot type: {chatbot_type}")
            return None
            
    except Exception as e:
        st.error(f"Error creating {chatbot_type} chatbot: {str(e)}")
        return None
