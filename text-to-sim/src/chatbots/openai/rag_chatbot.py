from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import os
import json
import re
import ast
import sqlite3
import csv
import logging

# Core LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import TextLoader
from langchain.schema.document import Document

# Additional utilities
import faiss
from uuid import uuid4

from dotenv import load_dotenv
from src.agent_evolution import (
    get_profile_marker_overrides,
    get_profile_pattern_overrides,
)
from src.andes_manual import RAG_ANDES_MANUAL_POLICY, build_retrieved_andes_manual_context
from src.andes_case_catalog import get_andes_builtin_case_paths, suggest_andes_case_paths
from src.code_blocks import extract_python_code_segments
from src.conversation_compaction import (
    ConversationCompactionConfig,
    build_compacted_message_window,
    build_readable_conversation_summary,
)
from src.prompt_builder import AndesPromptBuilderConfig, AndesSystemPromptBuilder

# Stage 1 refactor: pure detectors/extractors now live in src.andes_code.
# Re-exported here so every external import path keeps working.
from src.andes_code.detectors import (  # noqa: F401
    is_code_only_request,
    is_explanatory_followup_request,
    looks_like_python_script,
    prompt_explicitly_mentions_idx,
)
from src.andes_code.extractors import (  # noqa: F401
    _extract_markdown_section,
    _extract_voltage_bounds,
    extract_continuity_case_identifier,
    extract_continuity_case_source,
    extract_effective_user_context,
    extract_python_code_blocks,
    extract_requested_bus_number,
    extract_requested_bus_numbers,
    extract_uploaded_files_from_context,
    infer_requested_builtin_case,
)
from src.andes_code.normalizer import (  # noqa: F401
    ensure_import,
    ensure_python_code_block,
    ensure_result_json_output,
    normalize_andes_code_block,
    normalize_andes_response,
    resolve_builtin_case_path,
    transform_python_code_blocks,
)
from src.andes_code.validators import (  # noqa: F401
    check_python_code_compilation,
    validate_andes_case_loading,
    validate_response_code,
)
from src.chatbots.openai.chat_feedback import (
    PROSE_RESPONSE_GUARDRAIL,
    PROSE_RETRY_NUDGE,
    build_compilation_error_feedback,
)
from src.chatbots.openai.embedding_config import (  # noqa: F401
    EMBEDDING_MODEL_DIMENSIONS,
    resolve_embedding_dimension,
)
from src.andes_code.fallback import (  # noqa: F401
    build_andes_explanation_fallback_response,
    build_andes_fallback_response,
)
from src.andes_code.structured.codegen import (  # noqa: F401
    build_structured_andes_response,
)
from src.andes_code.structured.scripts import (  # noqa: F401
    _build_structured_case_load_lines,
    build_structured_andes_script,
    build_structured_n1_screening_script,
    build_structured_targeted_pq_script,
    build_structured_targeted_pv_script,
)
from src.andes_code.structured.state import (  # noqa: F401
    StructuredAndesState,
    extract_first_float,
    extract_first_float_from_patterns,
    extract_first_int,
    extract_first_int_from_patterns,
    extract_high_voltage_threshold,
    extract_low_voltage_threshold,
    extract_plot_filename,
    extract_result_json_keys,
    extract_top_k_from_prompt,
    infer_structured_report_kind,
    merge_structured_andes_state,
    parse_add_pq_operation,
    parse_candidate_line_pairs,
    parse_line_outage_by_pair,
    parse_scale_pq_at_bus_operation,
    parse_set_pv_bus_v0_operation,
    parse_target_pq_bus,
    parse_target_pq_scale_factor,
    parse_target_pv_bus,
    parse_target_pv_v0_value,
    structured_codegen_is_applicable,
    structured_report_has_required_state,
)

load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# EMBEDDING_MODEL_DIMENSIONS + resolve_embedding_dimension now live in
# src.chatbots.openai.embedding_config (Stage 3 refactor). Re-exported
# below to keep any external import paths stable.


def _env_int(name: str, default: int) -> int:
    """Parse an int env var; fall back to default on unset or malformed."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring non-integer env var %s=%r; using default %d", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Parse a bool env var ('1'/'true'/'yes' -> True, '0'/'false'/'no' -> False)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off", ""):
        return False
    logger.warning("Ignoring non-bool env var %s=%r; using default %s", name, raw, default)
    return default


@dataclass
class RAGConfig:
    """Configuration for the RAG chatbot.

    Every numeric / boolean field accepts a ``PFAGENT_*`` env-var
    override at class-definition time so operators can tune chunk
    sizes, retry counts, or compaction thresholds without editing
    code. Model names continue to honor the pre-existing
    ``OPENAI_EMBEDDING_MODEL`` / ``OPENAI_CHAT_MODEL`` env vars.
    """
    openai_api_key: str
    embedding_model: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    chunk_size: int = _env_int("PFAGENT_CHUNK_SIZE", 1000)
    chunk_overlap: int = _env_int("PFAGENT_CHUNK_OVERLAP", 200)
    max_tokens: int = _env_int("PFAGENT_MAX_TOKENS", 2000)
    data_directory: str = os.environ.get("PFAGENT_DATA_DIR", ".")
    code_compilation_check: bool = _env_bool("PFAGENT_CODE_COMPILATION_CHECK", True)
    max_compilation_retries: int = _env_int("PFAGENT_MAX_COMPILATION_RETRIES", 3)
    allow_template_fallback: bool = _env_bool("PFAGENT_ALLOW_TEMPLATE_FALLBACK", True)
    conversation_compaction_enabled: bool = _env_bool("PFAGENT_COMPACTION_ENABLED", True)
    conversation_compaction_trigger_messages: int = _env_int("PFAGENT_COMPACTION_TRIGGER_MESSAGES", 14)
    conversation_compaction_keep_recent_messages: int = _env_int("PFAGENT_COMPACTION_KEEP_RECENT", 8)
    conversation_compaction_max_summary_chars: int = _env_int("PFAGENT_COMPACTION_MAX_SUMMARY_CHARS", 3200)

@tool
def query_database(sql_query: str) -> str:
    """Execute an SQL query on the database and return the results."""
    return None

def extract_response_text(response: Any) -> str:
    """Normalize LangChain/OpenAI responses into plain text."""
    if isinstance(response, str):
        return response

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "\n".join(part for part in parts if part).strip()

    return str(content)


# is_code_only_request, is_explanatory_followup_request, looks_like_python_script
# now live in src.andes_code.detectors (Stage 1 refactor). Re-exported below.


# ensure_python_code_block, ensure_import, ensure_result_json_output,
# resolve_builtin_case_path, transform_python_code_blocks,
# normalize_andes_code_block, normalize_andes_response now live in
# src.andes_code.normalizer (Stage 1 refactor). Re-exported from top.


# check_python_code_compilation, validate_response_code,
# validate_andes_case_loading now live in src.andes_code.validators
# (Stage 1 refactor). Re-exported from top.


# _extract_voltage_bounds now lives in src.andes_code.extractors
# (Stage 1 refactor). Re-exported below.


# build_andes_fallback_response and build_andes_explanation_fallback_response
# now live in src.andes_code.fallback (Stage 1 refactor). Re-exported from top.


# StructuredAndesState, all parse_*/extract_* helpers, merge_structured_andes_state,
# infer_structured_report_kind, structured_codegen_is_applicable, and
# structured_report_has_required_state now live in
# src.andes_code.structured.state (Stage 1 refactor). Re-exported from top.


# _build_structured_case_load_lines, build_structured_targeted_pq_script,
# build_structured_targeted_pv_script, build_structured_n1_screening_script,
# build_structured_andes_script now live in
# src.andes_code.structured.scripts (Stage 1 refactor). Re-exported from top.


# build_structured_andes_response now lives in
# src.andes_code.structured.codegen (Stage 1 refactor). Re-exported from top.


class RAGChatbot:
    """Main chatbot class using RAG with FAISS vector store and OpenAI chat completions"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.system_message = None
        
        # Initialize OpenAI components
        self.chat_model = ChatOpenAI(
            api_key=config.openai_api_key,
            model_name=config.chat_model,
            max_tokens=config.max_tokens,
            use_responses_api=True
        )
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.openai_api_key,
        )
        
        # Initialize FAISS vector store following the sample pattern
        self.index = faiss.IndexFlatL2(resolve_embedding_dimension(config.embedding_model))
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=self.index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        
        # Text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Conversation history
        self.conversation_history: List[BaseMessage] = []
        self.last_response_used_template_fallback = False
        self.structured_andes_state = StructuredAndesState()
        self.compaction_config = ConversationCompactionConfig(
            enabled=config.conversation_compaction_enabled,
            trigger_message_count=config.conversation_compaction_trigger_messages,
            keep_recent_messages=config.conversation_compaction_keep_recent_messages,
            max_summary_chars=config.conversation_compaction_max_summary_chars,
        )
        self.last_compaction_summary = ""

        # ANDES manual is retrieved separately from uploaded documents to avoid
        # stitching together small vector fragments from the official manual.
        self.default_andes_manual_loaded = False
        self.default_andes_manual_count = 0
        self.default_andes_manual_page_count = 0
        
        # Document tracking for session management
        self.persistent_documents: Dict[str, str] = {}  # doc_id -> content
        self.session_documents: Dict[str, Dict[str, str]] = {}  # session_id -> {doc_id -> content}
        
        # SQLite database for CSV/Excel queries
        self.db_conn = sqlite3.connect(':memory:', check_same_thread=False)
        
        self.prompt_builder = AndesSystemPromptBuilder(
            AndesPromptBuilderConfig(
                include_context_placeholder=True,
                include_tools_info=True,
                enforce_code_only_fence_rule=True,
                include_andes_guardrails=True,
                ban_test_wrappers=True,
            )
        )

    def _build_model_messages(self, system_message_content: str) -> List[BaseMessage]:
        messages, summary = build_compacted_message_window(
            system_message=system_message_content,
            conversation_history=self.conversation_history,
            config=self.compaction_config,
        )
        self.last_compaction_summary = summary
        if summary:
            logger.info(
                "Using compacted conversation memory for RAG chat (%d history messages).",
                len(self.conversation_history),
            )
        return messages

    def query_database_execute(self, sql_query: str) -> str:
        """Execute an SQL query on the database and return the results."""
        cursor = self.db_conn.cursor()
        try:
            cursor.execute(sql_query)
            results = cursor.fetchall()
            return f"Query executed successfully. Results: {results}"
        except sqlite3.Error as e:
            return f"An error occurred: {e}"
        finally:
            cursor.close()
    
    async def process_documents(
        self,
        documents: List[str],
        doc_ids: List[str] = None,
        session_id: str = None,
        persistent_flags: List[bool] = None,
        split_flags: List[bool] = None,
        document_metadata: List[Dict[str, Any]] = None,
    ):
        """Process documents and build FAISS vector store"""
        logger.info("Processing documents...")
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        if persistent_flags is None:
            persistent_flags = [False] * len(documents)

        if split_flags is None:
            split_flags = [True] * len(documents)

        if document_metadata is None:
            document_metadata = [{} for _ in documents]
        
        # Track documents for session management
        filtered_documents: List[str] = []
        filtered_doc_ids: List[str] = []
        filtered_persistent_flags: List[bool] = []
        filtered_split_flags: List[bool] = []
        filtered_metadata: List[Dict[str, Any]] = []

        for doc_id, document, is_persistent, should_split, extra_metadata in zip(
            doc_ids,
            documents,
            persistent_flags,
            split_flags,
            document_metadata,
        ):
            metadata_payload = extra_metadata or {}
            if metadata_payload.get("source") == "andes_manual":
                self.default_andes_manual_loaded = True
                continue

            filtered_documents.append(document)
            filtered_doc_ids.append(doc_id)
            filtered_persistent_flags.append(is_persistent)
            filtered_split_flags.append(should_split)
            filtered_metadata.append(metadata_payload)

        # Track non-manual documents for session management
        for doc_id, document, is_persistent in zip(filtered_doc_ids, filtered_documents, filtered_persistent_flags):
            if is_persistent:
                self.persistent_documents[doc_id] = document
            else:
                if session_id:
                    if session_id not in self.session_documents:
                        self.session_documents[session_id] = {}
                    self.session_documents[session_id][doc_id] = document
        
        # Prepare documents for processing
        all_documents = []
        document_uuids = []
        
        for doc_id, document, should_split, extra_metadata in zip(
            filtered_doc_ids,
            filtered_documents,
            filtered_split_flags,
            filtered_metadata,
        ):
            base_metadata = {"doc_id": doc_id}
            if extra_metadata:
                base_metadata.update(extra_metadata)

            doc_obj = Document(page_content=document, metadata=base_metadata)
            chunks = self.text_splitter.split_documents([doc_obj]) if should_split else [doc_obj]
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk.metadata["chunk_id"] = chunk_id
                all_documents.append(chunk)
                document_uuids.append(str(uuid4()))
        
        # Add documents to FAISS vector store following the sample pattern
        if all_documents:
            self.vector_store.add_documents(documents=all_documents, ids=document_uuids)
            logger.info(f"Processed {len(all_documents)} chunks into FAISS vector store")

    def retrieve_manual_context(self, query: str) -> str:
        """Retrieve larger contiguous manual windows from the full ANDES manual."""
        if not self.default_andes_manual_loaded:
            return ""
        return build_retrieved_andes_manual_context(query)

    async def retrieve_document_context(self, query: str, k: int = 4) -> List[str]:
        """Retrieve relevant uploaded-document context using FAISS similarity search."""
        context_chunks = []

        try:
            similar_docs = self.vector_store.similarity_search(query, k=k)
            for doc in similar_docs:
                context_chunks.append(doc.page_content)
        except Exception as e:
            logger.warning(f"Error during similarity search: {e}")

        return context_chunks

    async def retrieve_context(self, query: str, k: int = 4) -> Tuple[str, List[str]]:
        """Retrieve manual context from the full manual and document context from FAISS."""
        manual_context = self.retrieve_manual_context(query)
        document_context = await self.retrieve_document_context(query, k=k)
        return manual_context, document_context

    def format_context(self, manual_context: str, context_chunks: List[str]) -> str:
        """Format manual-first context for the prompt."""
        sections: List[str] = []
        if manual_context:
            sections.append(manual_context)

        if context_chunks:
            document_section = [
                "## Relevant Information from uploaded documents and other retrieved files:"
            ]
            for index, chunk in enumerate(context_chunks[:2], 1):
                document_section.append(f"### Retrieved Document Context {index}:\n{chunk}")
            sections.append("\n\n".join(document_section))

        return "\n\n".join(sections)

    def _try_direct_prose_fallback(
        self, effective_user_context: str, user_message: str
    ) -> str:
        """Attempt the deterministic prose explanation template.

        Returns the fallback body (possibly empty). When non-empty, the
        body is also appended to conversation_history so callers can
        ``return`` it directly.
        """
        fallback_text = build_andes_explanation_fallback_response(
            effective_user_context or user_message
        )
        if fallback_text:
            self.conversation_history.append(AIMessage(content=fallback_text))
        return fallback_text

    def _try_structured_codegen(
        self, effective_user_context: str, user_message: str
    ) -> Optional[str]:
        """Attempt structured deterministic ANDES codegen.

        On success, updates ``structured_andes_state``, logs the
        structured-codegen note, appends to conversation_history, and
        returns the response. Returns ``None`` when no structured
        template fires for this turn.
        """
        structured_response, updated_state, structured_notes = build_structured_andes_response(
            effective_user_context or user_message,
            self.structured_andes_state,
        )
        if not structured_response:
            return None
        self.structured_andes_state = updated_state
        if structured_notes:
            logger.info("Applied structured ANDES code generation: %s", "; ".join(structured_notes))
        self.conversation_history.append(AIMessage(content=structured_response))
        return structured_response

    async def chat(self, user_message: str, max_retries: int = None) -> str:
        """Main chat function with code compilation checking"""
        # Use config values if not specified
        if max_retries is None:
            max_retries = self.config.max_compilation_retries
        
        # Skip compilation checking if disabled in config
        if not self.config.code_compilation_check:
            return await self._chat_without_compilation_check(user_message)
        
        # Add user message to conversation
        user_msg = HumanMessage(content=user_message)
        self.conversation_history.append(user_msg)
        self.last_response_used_template_fallback = False
        effective_user_context = extract_effective_user_context(user_message)
        prefer_prose_response = is_explanatory_followup_request(effective_user_context or user_message)
        direct_explanation_fallback = ""

        if prefer_prose_response:
            direct_explanation_fallback = self._try_direct_prose_fallback(
                effective_user_context, user_message
            )
            if direct_explanation_fallback:
                return direct_explanation_fallback
        else:
            structured_response = self._try_structured_codegen(
                effective_user_context, user_message
            )
            if structured_response:
                return structured_response

        # Retrieve relevant context
        manual_context, context_chunks = await self.retrieve_context(effective_user_context or user_message)
        
        # Format context
        context = self.format_context(manual_context, context_chunks)
        
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Create system message with context
                system_message_content = self.system_message.replace('<context>', context)
                if prefer_prose_response:
                    system_message_content += PROSE_RESPONSE_GUARDRAIL
                system_msg = SystemMessage(content=system_message_content)
                
                # Get response from OpenAI
                response = await self.chat_model.ainvoke(
                    self._build_model_messages(system_msg.content)
                )

                # Handle tool calls if present
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for tool_call in response.tool_calls:
                        if tool_call['name'] == 'query_database':
                            tool_result = self.query_database_execute(tool_call['args']['sql_query'])
                            tool_msg = ToolMessage(content=tool_result, tool_call_id=tool_call['id'])
                            self.conversation_history.append(tool_msg)
                    
                    # Get final response after tool calls
                    response = await self.chat_model.ainvoke(
                        self._build_model_messages(system_msg.content)
                    )

                response_text = extract_response_text(response)
                if prefer_prose_response:
                    prose_response_text = (response_text or "").strip()
                    if extract_python_code_segments(prose_response_text) or looks_like_python_script(prose_response_text):
                        if retry_count < max_retries:
                            self.conversation_history.append(AIMessage(content=prose_response_text))
                            self.conversation_history.append(
                                HumanMessage(content=PROSE_RETRY_NUDGE)
                            )
                            retry_count += 1
                            logger.warning(
                                "Model returned code for an explanatory follow-up, retrying in prose-only mode (%d/%d)",
                                retry_count,
                                max_retries,
                            )
                            continue
                        fallback_explanation = direct_explanation_fallback or build_andes_explanation_fallback_response(
                            effective_user_context or user_message
                        )
                        if fallback_explanation:
                            self.conversation_history.append(AIMessage(content=fallback_explanation))
                            return fallback_explanation
                    self.conversation_history.append(AIMessage(content=prose_response_text))
                    return prose_response_text

                normalized_response_text, normalization_notes = normalize_andes_response(
                    response_text,
                    user_context=user_message,
                )
                if normalization_notes:
                    logger.info("Applied ANDES response normalization: %s", "; ".join(normalization_notes))

                # Check if code compilation checking is enabled
                if self.config.code_compilation_check:
                    # Validate any Python code in the response
                    is_valid, error_messages = validate_response_code(
                        normalized_response_text,
                        user_context=user_message,
                    )

                    if not is_valid and self.config.allow_template_fallback:
                        fallback_response_text = build_andes_fallback_response(effective_user_context or user_message)
                        if fallback_response_text:
                            normalized_fallback_text, fallback_notes = normalize_andes_response(
                                fallback_response_text,
                                user_context=user_message,
                            )
                            fallback_is_valid, fallback_errors = validate_response_code(
                                normalized_fallback_text,
                                user_context=user_message,
                            )
                            if fallback_is_valid:
                                if fallback_notes:
                                    logger.info(
                                        "Applied ANDES fallback normalization: %s",
                                        "; ".join(fallback_notes),
                                    )
                                logger.info("Using ANDES fallback template for the current power-flow task.")
                                self.last_response_used_template_fallback = True
                                self.conversation_history.append(AIMessage(content=normalized_fallback_text))
                                return normalized_fallback_text
                            error_messages.extend(fallback_errors)
                    
                    if not is_valid and retry_count < max_retries:
                        self.conversation_history.append(AIMessage(content=normalized_response_text))
                        error_feedback = build_compilation_error_feedback(error_messages)
                        error_msg = HumanMessage(content=error_feedback)
                        self.conversation_history.append(error_msg)
                        
                        retry_count += 1
                        logger.warning(f"Code compilation failed, retrying ({retry_count}/{max_retries})")
                        continue

                # Update conversation history with successful response
                self.conversation_history.append(AIMessage(content=normalized_response_text))
                return normalized_response_text
                
            except Exception as e:
                logger.error(f"Error in chat function: {e}")
                if retry_count >= max_retries:
                    return f"I apologize, but I encountered an error: {str(e)}"
                retry_count += 1
        
        # This should never be reached, but just in case
        raise Exception("Unexpected error in chat function")

    async def _chat_without_compilation_check(self, user_message: str) -> str:
        """Original chat function without compilation checking (for backward compatibility)"""
        # Add user message to conversation
        user_msg = HumanMessage(content=user_message)
        self.conversation_history.append(user_msg)
        effective_user_context = extract_effective_user_context(user_message)
        prefer_prose_response = is_explanatory_followup_request(effective_user_context or user_message)

        if prefer_prose_response:
            direct_explanation_fallback = self._try_direct_prose_fallback(
                effective_user_context, user_message
            )
            if direct_explanation_fallback:
                return direct_explanation_fallback
        else:
            structured_response = self._try_structured_codegen(
                effective_user_context, user_message
            )
            if structured_response:
                return structured_response

        # Retrieve relevant context
        manual_context, context_chunks = await self.retrieve_context(effective_user_context or user_message)
        
        # Format context
        context = self.format_context(manual_context, context_chunks)

        # Create system message with context
        system_message_content = self.system_message.replace('<context>', context)
        if prefer_prose_response:
            system_message_content += PROSE_RESPONSE_GUARDRAIL
        system_msg = SystemMessage(content=system_message_content)
        
        # Get response from OpenAI
        response = await self.chat_model.ainvoke(
            self._build_model_messages(system_msg.content)
        )

        # Handle tool calls if present
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'query_database':
                    tool_result = self.query_database_execute(tool_call['args']['sql_query'])
                    tool_msg = ToolMessage(content=tool_result, tool_call_id=tool_call['id'])
                    self.conversation_history.append(tool_msg)
            
            # Get final response after tool calls
            response = await self.chat_model.ainvoke(
                self._build_model_messages(system_msg.content)
            )
        
        response_text = extract_response_text(response)
        if prefer_prose_response:
            normalized_response_text = (response_text or "").strip()
        else:
            normalized_response_text, normalization_notes = normalize_andes_response(
                response_text,
                user_context=user_message,
            )
            if normalization_notes:
                logger.info("Applied ANDES response normalization: %s", "; ".join(normalization_notes))

        # Update conversation history
        self.conversation_history.append(AIMessage(content=normalized_response_text))
        return normalized_response_text

    def load_system_prompt(self, session_id: str = None, custom_instructions: str = ""):
        """Load system prompt with CSV/Excel database information and custom instructions"""
        try:
            with open(os.path.join("data_files", "metadata.json"), "r") as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            metadata = {"persistent": {}, "sessions": {}}
        
        tools_info = "We have a few sqlite databases (in CSV files) that user can ask questions about. Whenever user asks something related to those tables, you need to call query_database function with sql_query, and table_name in arguments. The SQL query must be compatible with sqlite3. Here is list and details about the databases:\n\n"
        tools_found = False

        cursor = self.db_conn.cursor()
        
        # Load persistent CSV/Excel files
        for file_id, metad in metadata.get("persistent", {}).items():
            tools_found = True
            tools_info += f"- **Table name: {file_id}**:\nColumns: {', '.join(metad['columns_info'])}\nDescription: {metad.get('user_description', 'No description available')}\n\n"
            
            # Create table with columns
            csv_file_path = metad['file_path']
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {file_id} ({', '.join(metad['columns_info'])})")
            
            # Load CSV data into the table
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    placeholders = ', '.join(['?' for _ in row])
                    cursor.execute(f"INSERT INTO {file_id} VALUES ({placeholders})", list(row.values()))
            self.db_conn.commit()

        # Load session-specific CSV/Excel files
        if session_id:
            for file_id, metad in metadata.get('sessions', {}).get(session_id, {}).items():
                tools_found = True
                tools_info += f"- **Table name: {file_id}**:\nColumns: {', '.join(metad['columns_info'])}\nDescription: {metad.get('user_description', 'No description available')}\n\n"
                
                # Create table with columns
                csv_file_path = metad['file_path']
                cursor.execute(f"CREATE TABLE IF NOT EXISTS {file_id} ({', '.join(metad['columns_info'])})")
                
                # Load CSV data into the table
                with open(csv_file_path, 'r') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        placeholders = ', '.join(['?' for _ in row])
                        cursor.execute(f"INSERT INTO {file_id} VALUES ({placeholders})", list(row.values()))

        cursor.close()
        self.db_conn.commit()

        self.system_message = self.prompt_builder.build_prompt(
            andes_manual_policy=RAG_ANDES_MANUAL_POLICY,
            tools_info=tools_info if tools_found else "",
            custom_instructions=custom_instructions,
        )

        # Enable query_database tool if tools are found
        if tools_found:
            self.chat_model = self.chat_model.bind_tools([query_database])
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation history"""
        return build_readable_conversation_summary(
            conversation_history=self.conversation_history,
            config=self.compaction_config,
        )
    
    def set_code_compilation_check(self, enabled: bool):
        """Enable or disable code compilation checking"""
        self.config.code_compilation_check = enabled
        logger.info(f"Code compilation checking {'enabled' if enabled else 'disabled'}")
    
    def set_max_compilation_retries(self, max_retries: int):
        """Set maximum number of compilation retries"""
        if max_retries < 0:
            max_retries = 0
        self.config.max_compilation_retries = max_retries
        logger.info(f"Maximum compilation retries set to {max_retries}")
    
    def cleanup_session(self, session_id: str):
        """Clean up non-persistent documents for a session"""
        if session_id in self.session_documents:
            # Get document IDs to remove
            doc_ids_to_remove = list(self.session_documents[session_id].keys())
            
            # Remove documents from FAISS vector store
            # Note: FAISS doesn't have direct document removal by doc_id in metadata
            # In a production system, you might want to rebuild the index
            # For now, we'll just remove from our tracking
            del self.session_documents[session_id]
            
            logger.info(f"Cleaned up {len(doc_ids_to_remove)} session documents for session {session_id}")
    
    def close(self):
        """Clean up resources"""
        self.db_conn.close()

# Example usage and testing
async def main():
    """Example usage of the RAG chatbot"""
    
    # Configuration
    config = RAGConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Initialize chatbot
    chatbot = RAGChatbot(config)
    
    # Sample documents
    documents = [
        """
        Python is a high-level, interpreted programming language with dynamic semantics.
        Its high-level built-in data structures, combined with dynamic typing and dynamic binding,
        make it very attractive for Rapid Application Development, as well as for use as a scripting
        or glue language to connect existing components together.
        """,
        """
        Machine learning is a method of data analysis that automates analytical model building.
        It is a branch of artificial intelligence based on the idea that systems can learn from data,
        identify patterns and make decisions with minimal human intervention.
        """,
        """
        FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and
        clustering of dense vectors. It contains algorithms that search in sets of vectors of any size,
        up to ones that possibly do not fit in RAM.
        """
    ]
    
    # Process documents
    await chatbot.process_documents(documents)
    
    # Interactive chat loop
    print("\n=== RAG Chatbot Ready ===")
    print("Ask questions about the processed documents. Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        
        if not user_input:
            continue
        
        try:
            response = await chatbot.chat(user_input)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Show conversation summary
    print("\n=== Conversation Summary ===")
    print(chatbot.get_conversation_summary())
    
    # Cleanup
    chatbot.close()
    print("\nGoodbye!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
