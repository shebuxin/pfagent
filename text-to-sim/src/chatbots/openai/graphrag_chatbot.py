import csv
import os
import asyncio
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging
import csv

# Core LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, BaseMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.schema.document import Document

import sqlite3

# Neo4j imports
from neo4j import GraphDatabase

# Additional utilities
import numpy as np
import json
import re
import ast
import traceback

logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

from dotenv import load_dotenv
from src.andes_manual import RAG_ANDES_MANUAL_POLICY, build_retrieved_andes_manual_context
from src.andes_case_catalog import get_andes_builtin_case_paths, suggest_andes_case_paths
from src.code_blocks import ensure_python_code_block
from src.conversation_compaction import (
    ConversationCompactionConfig,
    build_compacted_message_window,
    build_readable_conversation_summary,
)
from src.prompt_builder import AndesPromptBuilderConfig, AndesSystemPromptBuilder
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GraphRAGConfig:
    """Configuration for Graph RAG system"""
    openai_api_key: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    embedding_model: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-5-mini")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 2000
    data_directory: str = "."
    code_compilation_check: bool = True  # Enable/disable code compilation checking
    max_compilation_retries: int = 2     # Maximum retries for compilation errors
    conversation_compaction_enabled: bool = True
    conversation_compaction_trigger_messages: int = 14
    conversation_compaction_keep_recent_messages: int = 8
    conversation_compaction_max_summary_chars: int = 3200

class Neo4jGraphStore:
    """Neo4j graph database interface for storing and querying knowledge graphs"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._create_constraints()
    
    def _create_constraints(self):
        """Create necessary constraints and indexes"""
        with self.driver.session() as session:
            # Create unique constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE")
            
            # Create indexes for better performance
            session.run("CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE")
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None):
        with self.driver.session() as session:
            # Serialize metadata dict to a JSON string
            metadata_str = json.dumps(metadata or {})
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.content = $content,
                    d.created_at = datetime(),
                    d.metadata = $metadata
            """, doc_id=doc_id, content=content, metadata=metadata_str)
    
    def add_chunk(self, chunk_id: str, doc_id: str, content: str, embedding: List[float]):
        """Add a text chunk with its embedding"""
        with self.driver.session() as session:
            session.run("""
                MATCH (d:Document {id: $doc_id})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.content = $content,
                    c.embedding = $embedding
                MERGE (d)-[:CONTAINS]->(c)
            """, chunk_id=chunk_id, doc_id=doc_id, content=content, embedding=embedding)
    
    def add_entity(self, name: str, entity_type: str, properties: Dict[str, Any] = None):
        with self.driver.session() as session:
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.type = $entity_type
            """, name=name, entity_type=entity_type)
    
    def add_relationship(self, entity1: str, entity2: str, relation_type: str, properties: Dict[str, Any] = None):
        """Add a relationship between entities"""
        with self.driver.session() as session:
            session.run(f"""
                MATCH (e1:Entity {{name: $entity1}})
                MATCH (e2:Entity {{name: $entity2}})
                MERGE (e1)-[r:{relation_type}]->(e2)
                SET r.properties = $properties
            """, entity1=entity1, entity2=entity2, properties=properties or {})
    
    def link_entity_to_chunk(self, entity_name: str, chunk_id: str):
        """Link an entity to a text chunk"""
        with self.driver.session() as session:
            session.run("""
                MATCH (e:Entity {name: $entity_name})
                MATCH (c:Chunk {id: $chunk_id})
                MERGE (c)-[:MENTIONS]->(e)
            """, entity_name=entity_name, chunk_id=chunk_id)
    
    def get_related_entities(self, entity_name: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Get entities related to a given entity"""
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (e1:Entity {{name: $entity_name}})
                MATCH (e1)-[*1..{max_depth}]-(e2:Entity)
                RETURN DISTINCT e2.name as name, e2.type as type, e2.properties as properties
            """, entity_name=entity_name)
            return [dict(record) for record in result]
    
    def get_entity_context(self, entity_name: str) -> List[str]:
        """Get text chunks that mention an entity"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:Entity {name: $entity_name})<-[:MENTIONS]-(c:Chunk)
                RETURN c.content as content
            """, entity_name=entity_name)
            return [record["content"] for record in result]
    
    def cypher_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute a custom Cypher query"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def close(self):
        """Close the database connection"""
        self.driver.close()

    def remove_session_documents(self, session_id: str):
        """Remove documents that were added in a specific session (non-persistent)"""
        with self.driver.session() as session:
            # Remove chunks and their relationships for session documents
            session.run("""
                MATCH (d:Document)
                WHERE d.session_id = $session_id AND (d.persistent IS NULL OR d.persistent = false)
                MATCH (d)-[:CONTAINS]->(c:Chunk)
                DETACH DELETE c
            """, session_id=session_id)
            
            # Remove entities that are only connected to session documents
            session.run("""
                MATCH (d:Document)
                WHERE d.session_id = $session_id AND (d.persistent IS NULL OR d.persistent = false)
                MATCH (d)-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE NOT EXISTS {
                    MATCH (other_d:Document)-[:CONTAINS]->(other_c:Chunk)-[:MENTIONS]->(e)
                    WHERE other_d.session_id <> $session_id OR other_d.persistent = true
                }
                DETACH DELETE e
            """, session_id=session_id)
            
            # Remove the session documents themselves
            session.run("""
                MATCH (d:Document)
                WHERE d.session_id = $session_id AND (d.persistent IS NULL OR d.persistent = false)
                DETACH DELETE d
            """, session_id=session_id)

    def mark_document_persistent(self, doc_id: str, is_persistent: bool = True):
        """Mark a document as persistent or non-persistent"""
        with self.driver.session() as session:
            session.run("""
                MATCH (d:Document {id: $doc_id})
                SET d.persistent = $is_persistent
            """, doc_id=doc_id, is_persistent=is_persistent)

    def add_document_with_session(self, doc_id: str, content: str, session_id: str, is_persistent: bool = False, metadata: Dict[str, Any] = None):
        """Add a document with session tracking"""
        with self.driver.session() as session:
            # Serialize metadata dict to a JSON string
            metadata_str = json.dumps(metadata or {})
            session.run("""
                MERGE (d:Document {id: $doc_id})
                SET d.content = $content,
                    d.created_at = datetime(),
                    d.metadata = $metadata,
                    d.session_id = $session_id,
                    d.persistent = $is_persistent
            """, doc_id=doc_id, content=content, metadata=metadata_str, session_id=session_id, is_persistent=is_persistent)
    
@tool
def query_database(sql_query: str) -> str:
    """Execute an SQL query on the database and return the results."""
    return None

def extract_python_code_blocks(text: str) -> List[str]:
    """Extract Python code blocks from text"""
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [code.strip() for code in matches]

def check_python_code_compilation(code: str) -> Tuple[bool, str]:
    """
    Check if Python code compiles without syntax errors.
    Returns (is_valid, error_message)
    """
    try:
        # Parse the code to check for syntax errors
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax Error on line {e.lineno}: {e.msg}"
        if e.text:
            error_msg += f"\nProblematic line: {e.text.strip()}"
        return False, error_msg
    except Exception as e:
        return False, f"Compilation Error: {str(e)}"

def validate_response_code(response: str, user_context: str = "") -> Tuple[bool, List[str]]:
    """
    Validate all Python code blocks in a response.
    Returns (all_valid, error_messages)
    """
    code_blocks = extract_python_code_blocks(response)
    
    if not code_blocks:
        return True, []  # No code to validate
    
    error_messages = []
    all_valid = True
    
    for i, code in enumerate(code_blocks):
        is_valid, error_msg = check_python_code_compilation(code)
        if not is_valid:
            all_valid = False
            error_messages.append(f"Code block {i+1}: {error_msg}")

        rule_errors = validate_andes_case_loading(code, user_context=user_context)
        if rule_errors:
            all_valid = False
            for rule_error in rule_errors:
                error_messages.append(f"Code block {i+1}: {rule_error}")
    
    return all_valid, error_messages


def extract_uploaded_files_from_context(user_context: str) -> List[str]:
    """Extract uploaded filenames from runtime context injected by the app."""
    if not user_context:
        return []

    uploaded_files: List[str] = []
    in_uploaded_section = False
    for raw_line in user_context.splitlines():
        line = raw_line.strip()
        if "Uploaded files available during execution" in line:
            in_uploaded_section = True
            continue

        if not in_uploaded_section:
            continue

        if not line.startswith("- "):
            if line:
                in_uploaded_section = False
            continue

        candidate = line[2:].strip()
        lower_candidate = candidate.lower()
        if lower_candidate.startswith("use these filenames"):
            continue
        if lower_candidate.startswith("case-loading rule"):
            continue
        if lower_candidate.startswith("preferred uploaded-case template"):
            continue
        if "." not in candidate:
            continue
        uploaded_files.append(candidate)

    return uploaded_files


def validate_andes_case_loading(code: str, user_context: str = "") -> List[str]:
    """Validate common ANDES case-loading mistakes."""
    errors: List[str] = []
    uploaded_files = extract_uploaded_files_from_context(user_context)
    uploaded_file_set = {os.path.basename(name) for name in uploaded_files}

    if re.search(r"\bimport\s+anodes\b", code) or re.search(r"\banodes\.", code):
        errors.append("Use 'andes' package, not 'anodes'.")

    get_case_args = re.findall(r'andes\.get_case\(\s*["\']([^"\']+)["\']\s*\)', code)
    invalid_uploaded_args = set()
    if uploaded_file_set and get_case_args:
        for arg in get_case_args:
            arg_basename = os.path.basename(arg)
            if arg_basename in uploaded_file_set:
                errors.append(
                    f"Uploaded case '{arg_basename}' must be loaded directly with andes.load(...), "
                    "not andes.get_case(...)."
                )
                invalid_uploaded_args.add(arg)
                break
            if "/" not in arg and "\\" not in arg and arg_basename.lower().endswith((".xlsx", ".xls", ".csv")):
                errors.append(
                    "When uploaded files are available, do not call andes.get_case('<filename>'). "
                    "Use andes.load('<exact_filename>', ...)."
                )
                invalid_uploaded_args.add(arg)
                break

    builtin_case_paths = set(get_andes_builtin_case_paths())
    if builtin_case_paths and get_case_args:
        for arg in get_case_args:
            if arg in invalid_uploaded_args:
                continue
            normalized_arg = arg.replace("\\", "/")
            if normalized_arg not in builtin_case_paths:
                suggestions = suggest_andes_case_paths(normalized_arg, max_suggestions=3)
                if suggestions:
                    errors.append(
                        f"'{arg}' is not a valid ANDES built-in case path for andes.get_case(...). "
                        f"Try one of: {', '.join(suggestions)}."
                    )
                else:
                    errors.append(
                        f"'{arg}' is not a valid ANDES built-in case path for andes.get_case(...). "
                        "Use an exact relative path under andes/cases."
                    )

    return errors

class GraphRAGChatbot:
    """Main chatbot class combining Graph RAG with OpenAI chat completions"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config

        self.system_message = None
        
        # Initialize OpenAI components
        self.chat_model = ChatOpenAI(
            model_name=config.chat_model,
            max_tokens=config.max_tokens,
            api_key=config.openai_api_key,
            use_responses_api=True
        )
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            api_key=config.openai_api_key,
        )
        
        # Initialize graph store
        self.graph_store = Neo4jGraphStore(
            config.neo4j_uri,
            config.neo4j_user,
            config.neo4j_password
        )
        
        # Initialize FAISS vector store
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Conversation history
        self.conversation_history: List[BaseMessage] = []
        self.compaction_config = ConversationCompactionConfig(
            enabled=config.conversation_compaction_enabled,
            trigger_message_count=config.conversation_compaction_trigger_messages,
            keep_recent_messages=config.conversation_compaction_keep_recent_messages,
            max_summary_chars=config.conversation_compaction_max_summary_chars,
        )
        self.last_compaction_summary = ""

        # The official manual is retrieved separately from uploaded documents so
        # we can search over the full manual and return larger contiguous windows.
        self.default_andes_manual_loaded = False
        self.default_andes_manual_count = 0
        self.default_andes_manual_page_count = 0
        
        # Entity extraction patterns optimized for coding documentation
        self.entity_patterns = {
            'CLASS': r'\bclass\s+([A-Za-z_][A-Za-z0-9_]*)',
            'FUNCTION': r'\bdef\s+([A-Za-z_][A-Za-z0-9_]*)',
            'METHOD': r'\.([A-Za-z_][A-Za-z0-9_]*)\s*\(',
            'VARIABLE': r'\b([A-Za-z_][A-Za-z0-9_]*)\s*=',
            'MODULE': r'\bimport\s+([A-Za-z_][A-Za-z0-9_\.]*)',
            'PACKAGE': r'\bfrom\s+([A-Za-z_][A-Za-z0-9_\.]*)',
            'CONSTANT': r'\b([A-Z][A-Z0-9_]*)\b',
            'PARAMETER': r'\b([A-Za-z_][A-Za-z0-9_]*)\s*:',
            'DECORATOR': r'@([A-Za-z_][A-Za-z0-9_]*)',
            'EXCEPTION': r'\b([A-Za-z_][A-Za-z0-9_]*Error|[A-Za-z_][A-Za-z0-9_]*Exception)\b'
        }

        self.db_conn = sqlite3.connect(':memory:', check_same_thread=False)

        self.system_message = None
        self.prompt_builder = AndesSystemPromptBuilder(
            AndesPromptBuilderConfig(
                include_context_placeholder=True,
                include_tools_info=True,
                enforce_code_only_fence_rule=False,
                include_andes_guardrails=False,
                ban_test_wrappers=False,
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
                "Using compacted conversation memory for GraphRAG (%d history messages).",
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
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities from text using simple patterns"""
        entities = []
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append((match.strip(), entity_type))
        return entities
    
    async def process_documents(
        self,
        documents: List[str],
        doc_ids: List[str] = None,
        session_id: str = None,
        persistent_flags: List[bool] = None,
        split_flags: List[bool] = None,
        document_metadata: List[Dict[str, Any]] = None,
    ):
        """Process documents and build both vector and graph stores"""
        print("Processing documents...")
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        if persistent_flags is None:
            persistent_flags = [False] * len(documents)

        if split_flags is None:
            split_flags = [True] * len(documents)

        if document_metadata is None:
            document_metadata = [{} for _ in documents]
        
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

        all_chunks = []
        
        for doc_id, document, is_persistent, should_split, extra_metadata in zip(
            filtered_doc_ids,
            filtered_documents,
            filtered_persistent_flags,
            filtered_split_flags,
            filtered_metadata,
        ):
            metadata_payload = extra_metadata or {}
            # Add document to graph with session tracking
            if session_id:
                self.graph_store.add_document_with_session(
                    doc_id,
                    document,
                    session_id,
                    is_persistent,
                    metadata=metadata_payload,
                )
            else:
                self.graph_store.add_document(doc_id, document, metadata=metadata_payload)
            
            # Split document into chunks
            base_metadata = {"doc_id": doc_id}
            if metadata_payload:
                base_metadata.update(metadata_payload)
            doc_obj = Document(page_content=document, metadata=base_metadata)
            chunks = self.text_splitter.split_documents([doc_obj]) if should_split else [doc_obj]

            if not chunks:
                continue

            embeddings = await self.embeddings.aembed_documents(
                [chunk.page_content for chunk in chunks]
            )
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_content = chunk.page_content
                
                # Add chunk to graph with embedding
                self.graph_store.add_chunk(chunk_id, doc_id, chunk_content, embedding)
                
                # Extract entities from chunk
                entities = self.extract_entities(chunk_content)
                
                for entity_name, entity_type in entities:
                    # Add entity to graph
                    self.graph_store.add_entity(entity_name, entity_type)
                    
                    # Link entity to chunk
                    self.graph_store.link_entity_to_chunk(entity_name, chunk_id)
                
                # Prepare for FAISS
                all_chunks.append(chunk)
        
        # Build FAISS vector store
        if all_chunks:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
            else:
                self.vector_store.add_documents(documents=all_chunks)
            logger.info(f"Processed {len(all_chunks)} chunks into vector store")

    def retrieve_manual_context(self, query: str) -> str:
        """Retrieve larger contiguous manual windows from the full ANDES manual."""
        if not self.default_andes_manual_loaded:
            return ""
        return build_retrieved_andes_manual_context(query)

    async def retrieve_context(self, query: str, k: int = 4) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        """Retrieve manual context plus graph/vector context for uploaded documents."""
        context_chunks = []
        graph_context = []

        manual_context = self.retrieve_manual_context(query)

        if self.vector_store:
            similar_docs = self.vector_store.similarity_search(query, k=k)
            for doc in similar_docs:
                context_chunks.append(doc.page_content)

        query_entities = self.extract_entities(query)

        for entity_name, entity_type in query_entities:
            related_entities = self.graph_store.get_related_entities(entity_name)
            graph_context.extend(related_entities)

            entity_chunks = self.graph_store.get_entity_context(entity_name)
            context_chunks.extend(entity_chunks)

        return manual_context, context_chunks[:k], graph_context

    def format_context(self, manual_context: str, context_chunks: List[str], graph_context: List[Dict[str, Any]]) -> str:
        """Format manual-first context for the prompt."""
        sections: List[str] = []
        if manual_context:
            sections.append(manual_context)

        if context_chunks:
            text_section = ["## Relevant Information from uploaded documents and retrieved files:"]
            for index, chunk in enumerate(context_chunks[:2], 1):
                text_section.append(f"### Retrieved Document Context {index}:\n{chunk[:2400]}")
            sections.append("\n\n".join(text_section))

        if graph_context:
            graph_section = ["## Related graph entities from uploaded documents:"]
            for entity in graph_context[:5]:
                graph_section.append(f"- {entity['name']} ({entity['type']})")
            sections.append("\n".join(graph_section))

        return "\n\n".join(sections)
    
    async def chat(self, user_message: str, max_retries: int = None) -> str:
        """Main chat function with code compilation checking"""
        # Use config values if not specified
        if max_retries is None:
            max_retries = self.config.max_compilation_retries
        
        # Skip compilation checking if disabled in config
        if not self.config.code_compilation_check:
            return await self._chat_without_compilation_check(user_message)
        
        # Retrieve relevant context
        manual_context, context_chunks, graph_context = await self.retrieve_context(user_message)
        
        # Format context
        context = self.format_context(manual_context, context_chunks, graph_context)

        # Add user message to conversation
        user_msg = HumanMessage(content=user_message)
        self.conversation_history.append(user_msg)
        
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Get response from OpenAI
                system_msg = SystemMessage(content=self.system_message.replace('<context>', context))
                response = await self.chat_model.ainvoke(
                    self._build_model_messages(system_msg.content)
                )

                # Handle tool calls first
                if hasattr(response, "tool_calls") and response.tool_calls:
                    for tool_call in response.tool_calls:
                        function_name = tool_call['name']
                        if function_name == "query_database":
                            # Call the tool method directly with the sql_query argument
                            result = self.query_database_execute(tool_call['args']['sql_query'])
                            tool_call_result = ToolMessage(
                                content=result,
                                tool_call_id=tool_call['id']
                            )
                            self.conversation_history.append(tool_call_result)
                    response = await self.chat_model.ainvoke(
                        self._build_model_messages(system_msg.content)
                    )
                
                # Check if response contains Python code
                response_content = response.content[0].get("text", "") if isinstance(response.content, list) else response.content
                response_content, normalization_notes = ensure_python_code_block(
                    response_content,
                    user_context=user_message,
                )
                if normalization_notes:
                    logger.info("Applied code-block normalization: %s", "; ".join(normalization_notes))
                code_blocks = extract_python_code_blocks(response_content)
                
                if code_blocks:
                    # Validate code compilation
                    is_valid, error_messages = validate_response_code(
                        response_content,
                        user_context=user_message,
                    )
                    
                    if not is_valid:
                        retry_count += 1
                        logger.warning(f"Code compilation failed (attempt {retry_count}/{max_retries + 1}): {error_messages}")
                        
                        if retry_count <= max_retries:
                            # Create error feedback message for the LLM
                            error_feedback = (
                                f"The Python code in your previous response has compilation errors:\n\n"
                                + "\n".join(error_messages) + 
                                "\n\nPlease fix these compilation errors and provide corrected code. "
                                "Make sure all syntax is valid Python code."
                            )
                            
                            # Remove the failed response from conversation history
                            if self.conversation_history and isinstance(self.conversation_history[-1], AIMessage):
                                self.conversation_history.pop()
                            
                            # Add error feedback as a system message for retry
                            error_msg = SystemMessage(content=error_feedback)
                            self.conversation_history.append(error_msg)
                            
                            # Continue to next iteration for retry
                            continue
                        else:
                            # Max retries exceeded, return the response with a warning
                            warning_msg = (
                                "\n\n⚠️ **Note**: The generated code above may contain compilation errors. "
                                "Please review and test the code before executing it."
                            )
                            response_content += warning_msg
                            logger.error(f"Max retries exceeded. Returning response with compilation errors: {error_messages}")
                
                # Update conversation history with successful response
                self.conversation_history.append(AIMessage(content=response_content))
                return response_content
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Error in chat function (attempt {retry_count}/{max_retries + 1}): {str(e)}")
                
                if retry_count <= max_retries:
                    # Add a brief delay before retry
                    await asyncio.sleep(1)
                    continue
                else:
                    # Max retries exceeded, raise the exception
                    raise e
        
        # This should never be reached, but just in case
        raise Exception("Unexpected error in chat function")

    async def _chat_without_compilation_check(self, user_message: str) -> str:
        """Original chat function without compilation checking (for backward compatibility)"""
        # Retrieve relevant context
        manual_context, context_chunks, graph_context = await self.retrieve_context(user_message)
        
        # Format context
        context = self.format_context(manual_context, context_chunks, graph_context)

        # Add user message to conversation
        user_msg = HumanMessage(content=user_message)
        self.conversation_history.append(user_msg)
        
        # Get response from OpenAI
        system_msg = SystemMessage(content=self.system_message.replace('<context>', context))
        response = await self.chat_model.ainvoke(
            self._build_model_messages(system_msg.content)
        )

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tool_call in response.tool_calls:
                function_name = tool_call['name']
                if function_name == "query_database":
                    # Call the tool method directly with the sql_query argument
                    result = self.query_database_execute(tool_call['args']['sql_query'])
                    tool_call_result = ToolMessage(
                        content=result,
                        tool_call_id=tool_call['id']
                    )
                    self.conversation_history.append(tool_call_result)
            response = await self.chat_model.ainvoke(
                self._build_model_messages(system_msg.content)
            )

        response_content = response.content[0].get("text", "") if isinstance(response.content, list) else response.content
        response_content, normalization_notes = ensure_python_code_block(
            response_content,
            user_context=user_message,
        )
        if normalization_notes:
            logger.info("Applied code-block normalization: %s", "; ".join(normalization_notes))

        self.conversation_history.append(AIMessage(content=response_content))
        return response_content

    def load_system_prompt(self, session_id: str = None, custom_instructions: str = ""):
        try:
            with open(os.path.join("data_files", "metadata.json"), "r") as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            metadata = {"persistent": {}, "sessions": {}}
        tools_info = "We have a few sqlite databases (in CSV files) that user can ask questions about. Whenever user asks something related to those tables, you need to call query_database function with sql_query, and table_name in arguments. The SQL query must be compatible with sqlite3. Here is list and details about the databases:\n\n"
        tools_found = False

        cursor = self.db_conn.cursor()
        for file_id, metad in metadata.get("persistent", {}).items():
            tools_found = True
            tools_info += f"- **Table name: {file_id}**:\nColumns: {', '.join(metad['columns_info'])}\nDescription: {metad.get('user_description', 'No description available')}\n\n"
            # Create in memory sqlite connections for each file
            csv_file_path = metad['file_path']

            # Create table with columns
            cursor.execute(f"CREATE TABLE {file_id} ({', '.join(metad['columns_info'])})")
            # Load CSV data into the table
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    placeholders = ', '.join(['?'] * len(row))
                    cursor.execute(f"INSERT INTO {file_id} VALUES ({placeholders})", tuple(row.values()))
            self.db_conn.commit()

        for file_id, metad in metadata['sessions'].get(session_id, {}).items():
            tools_found = True
            tools_info += f"- **Table name: {file_id}**:\nColumns: {', '.join(metad['columns_info'])}\nDescription: {metad.get('user_description', 'No description available')}\n\n"
            # Create in memory sqlite connections for each file
            csv_file_path = metad['file_path']

            # Create table with columns
            cursor.execute(f"CREATE TABLE {file_id} ({', '.join(metad['columns_info'])})")
            # Load CSV data into the table
            with open(csv_file_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    placeholders = ', '.join(['?'] * len(row))
                    cursor.execute(f"INSERT INTO {file_id} VALUES ({placeholders})", tuple(row.values()))
            self.db_conn.commit()

        cursor.close()

        self.system_message = self.prompt_builder.build_prompt(
            andes_manual_policy=RAG_ANDES_MANUAL_POLICY,
            tools_info=tools_info if tools_found else "",
            custom_instructions=custom_instructions,
        )

        if tools_found:
            self.chat_model = self.chat_model.bind_tools(
                tools=[query_database]
            )
    def add_custom_relationship(self, entity1: str, entity2: str, relation_type: str, properties: Dict[str, Any] = None):
        """Add custom relationships to the knowledge graph"""
        self.graph_store.add_relationship(entity1, entity2, relation_type, properties)
    
    def query_graph(self, cypher_query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Execute custom graph queries"""
        return self.graph_store.cypher_query(cypher_query, parameters)
    
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
            raise ValueError("max_retries must be non-negative")
        self.config.max_compilation_retries = max_retries
        logger.info(f"Maximum compilation retries set to {max_retries}")
    
    def close(self):
        """Clean up resources"""
        self.graph_store.close()

    def cleanup_session(self, session_id: str):
        """Clean up non-persistent documents for a session"""
        self.graph_store.remove_session_documents(session_id)

# Example usage and testing
async def main():
    """Example usage of the GraphRAG chatbot"""
    
    # Configuration
    config = GraphRAGConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        neo4j_uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.environ.get("NEO4J_USER", "neo4j"),
        neo4j_password=os.environ.get("NEO4J_PASSWORD")
    )
    
    # Initialize chatbot
    chatbot = GraphRAGChatbot(config)
    
    # Sample documents
    documents = [
        """
        Priya Patel is the CEO of Innovatech Solutions, a leading technology company based in Austin.
        The company specializes in artificial intelligence and machine learning solutions.
        Priya has been working in the tech industry for over 15 years and previously worked at Microsoft.
        Innovatech Solutions was founded in 2018 and has grown to over 500 employees.
        """,
        """
        Carlos Rivera is the CTO of QuantumLeap Labs, a leading technology company based in Seattle.
        The company specializes in artificial intelligence and machine learning solutions.
        Carlos has been working in the tech industry for over 15 years and previously worked at Amazon.
        QuantumLeap Labs was founded in 2018 and has grown to over 500 employees.
        """,
        """
        Emily Chen is Sales head of NeuralNext Technologies, a leading technology company based in New York.
        The company specializes in artificial intelligence and machine learning solutions.
        Emily has been working in the tech industry for over 15 years and previously worked at IBM.
        NeuralNext Technologies was founded in 2018 and has grown to over 500 employees.
        """
    ]
    
    # Uncomment the following line any time you want to process a new set of documents
    await chatbot.process_documents(documents)
    
    # Add some custom relationships
    chatbot.add_custom_relationship("Priya Patel", "TechCorp Inc", "LEADS")
    chatbot.add_custom_relationship("Carlos Rivera", "QuantumLeap Labs", "WORKS_AT")
    chatbot.add_custom_relationship("Emily Chen", "NeuralNext Technologies", "SALES_HEAD")
    
    # Interactive chat loop
    print("\n=== GraphRAG Chatbot Ready ===")
    print("Ask questions about the processed documents. Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            break
        
        if not user_input:
            continue
        
        try:
            response = await chatbot.chat(user_input)
            print(f"\nAssistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    # Show conversation summary
    print("\n=== Conversation Summary ===")
    print(chatbot.get_conversation_summary())
    
    # Cleanup
    chatbot.close()
    print("\nGoodbye!")

if __name__ == "__main__":
    asyncio.run(main())
