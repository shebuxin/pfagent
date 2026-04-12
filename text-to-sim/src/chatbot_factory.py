import os
import streamlit as st
from src.chatbots.openai.graphrag_chatbot import GraphRAGChatbot, GraphRAGConfig
from src.chatbots.openai.rag_chatbot import RAGChatbot, RAGConfig
from src.chatbots.openai.simple_chatbot import SimpleChatbot, SimpleChatConfig

# Built-in default for the unmodified base path. Anyone can run this
# with their own OpenAI API key.
DEFAULT_BASE_MODEL = os.environ.get("OPENAI_BASE_CHAT_MODEL", "gpt-4o-mini")

# Fine-tuned models are user-specific assets on OpenAI's side and can
# only be invoked by the API key that owns the fine-tune job. There is
# no shared default; users must supply their own fine-tune model id
# through the introduction screen (or the OPENAI_FINETUNED_MODEL env
# var as a fallback for power users).
DEFAULT_FINETUNED_MODEL = os.environ.get("OPENAI_FINETUNED_MODEL", "")


def _resolve_base_model() -> str:
    return st.session_state.get("base_chat_model") or DEFAULT_BASE_MODEL


def _resolve_finetuned_model() -> str:
    return st.session_state.get("finetuned_chat_model") or DEFAULT_FINETUNED_MODEL


def create_chatbot(data_directory, openai_api_key, chatbot_type="RAG"):
    """Create and initialize the chatbot based on selected type"""
    base_model = _resolve_base_model()
    finetuned_model = _resolve_finetuned_model()

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
            if chatbot_type == "Fine-tuned + RAG" and not finetuned_model:
                st.error(
                    "❌ Fine-tuned + RAG requires a fine-tuned model id, but none was "
                    "provided. Re-launch and supply your fine-tune model id (or set the "
                    "`OPENAI_FINETUNED_MODEL` env var). Note: a fine-tuned model can only "
                    "be invoked by the OpenAI API key that owns the fine-tune job."
                )
                return None

            print("Creating RAG Chatbot...")
            config = RAGConfig(
                openai_api_key=openai_api_key,
                chat_model=finetuned_model if chatbot_type == "Fine-tuned + RAG" else base_model,
                data_directory=data_directory,
                code_compilation_check=st.session_state.get('code_compilation_check', True),
                max_compilation_retries=st.session_state.get('max_compilation_retries', 2)
            )

            if not config.openai_api_key:
                st.error("Missing OpenAI API key.")
                return None

            return RAGChatbot(config)

        elif chatbot_type in {"Base OpenAI", "No RAG", "Fine-tuned"}:
            if chatbot_type == "Fine-tuned" and not finetuned_model:
                st.error(
                    "❌ Fine-tuned mode requires a fine-tuned model id, but none was "
                    "provided. Re-launch and supply your fine-tune model id (or set the "
                    "`OPENAI_FINETUNED_MODEL` env var). Note: a fine-tuned model can only "
                    "be invoked by the OpenAI API key that owns the fine-tune job."
                )
                return None

            print("Creating Simple Chatbot...")
            config = SimpleChatConfig(
                openai_api_key=openai_api_key,
                chat_model=finetuned_model if chatbot_type == "Fine-tuned" else base_model,
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
