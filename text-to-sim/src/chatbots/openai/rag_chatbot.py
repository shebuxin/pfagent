from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
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
load_dotenv(override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    openai_api_key: str
    embedding_model: str = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chat_model: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 2000
    data_directory: str = "."
    code_compilation_check: bool = True  # Enable/disable code compilation checking
    max_compilation_retries: int = 2     # Maximum retries for compilation errors

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

def validate_response_code(response: str) -> Tuple[bool, List[str]]:
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
    
    return all_valid, error_messages

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
        self.embeddings = OpenAIEmbeddings(model=config.embedding_model)
        
        # Initialize FAISS vector store following the sample pattern
        self.index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
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
        
        # Document tracking for session management
        self.persistent_documents: Dict[str, str] = {}  # doc_id -> content
        self.session_documents: Dict[str, Dict[str, str]] = {}  # session_id -> {doc_id -> content}
        
        # SQLite database for CSV/Excel queries
        self.db_conn = sqlite3.connect(':memory:', check_same_thread=False)
        
        # Base system message
        self._system_message = SystemMessage(content="""
You are a helpful AI assistant with access to a knowledge base. Use the provided context to answer questions accurately and comprehensively.

<context>

When answering:
1. Use information from the provided context when relevant
2. If the context doesn't contain enough information, say so clearly
3. Be conversational and helpful
4. When your response includes python code, ensure it is well-formatted inside triple backticks (```python) for clarity. Also make sure to add required dependencies on top of the code block if those aren't python standard libraries.
5. IMPORTANT: All Python code you generate must be syntactically correct and compile without errors. Double-check your code for syntax errors, proper indentation, matching parentheses/brackets, and valid Python syntax before responding.

For example:
```python
# required_dependencies: numpy,pandas
import numpy as np
import pandas as pd

x = np.array([1, 2, 3])
print(x)
```

IMPORTANT: Always add required_dependencies on top of the code block if those aren't python standard libraries.

Code Quality Requirements:
- Ensure proper indentation (4 spaces per level)  
- Check that all parentheses, brackets, and quotes are properly matched
- Verify function/class definitions are syntactically correct
- Make sure import statements are valid
- Test variable names and function calls for typos

{tools_info}
""")

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
    
    async def process_documents(self, documents: List[str], doc_ids: List[str] = None, session_id: str = None, persistent_flags: List[bool] = None):
        """Process documents and build FAISS vector store"""
        logger.info("Processing documents...")
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(documents))]
        
        if persistent_flags is None:
            persistent_flags = [False] * len(documents)
        
        # Track documents for session management
        for doc_id, document, is_persistent in zip(doc_ids, documents, persistent_flags):
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
        
        for doc_id, document in zip(doc_ids, documents):
            # Split document into chunks following the FAISS sample pattern
            doc_obj = Document(page_content=document, metadata={"doc_id": doc_id})
            chunks = self.text_splitter.split_documents([doc_obj])
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk.metadata["chunk_id"] = chunk_id
                all_documents.append(chunk)
                document_uuids.append(str(uuid4()))
        
        # Add documents to FAISS vector store following the sample pattern
        if all_documents:
            self.vector_store.add_documents(documents=all_documents, ids=document_uuids)
            logger.info(f"Processed {len(all_documents)} chunks into FAISS vector store")
    
    async def retrieve_context(self, query: str, k: int = 5) -> List[str]:
        """Retrieve relevant context using FAISS similarity search"""
        context_chunks = []
        
        # Perform similarity search following the FAISS sample pattern
        try:
            similar_docs = self.vector_store.similarity_search(query, k=k)
            for doc in similar_docs:
                context_chunks.append(doc.page_content)
        except Exception as e:
            logger.warning(f"Error during similarity search: {e}")
        
        return context_chunks
    
    def format_context(self, context_chunks: List[str]) -> str:
        """Format retrieved context for the prompt"""
        if not context_chunks:
            return ""
        
        formatted_context = "## Relevant Information:\n\n"
        
        # Add text chunks
        for i, chunk in enumerate(context_chunks[:3], 1):
            formatted_context += f"### Context {i}:\n{chunk}\n\n"
        
        return formatted_context
    
    async def chat(self, user_message: str, max_retries: int = None) -> str:
        """Main chat function with code compilation checking"""
        # Use config values if not specified
        if max_retries is None:
            max_retries = self.config.max_compilation_retries
        
        # Skip compilation checking if disabled in config
        if not self.config.code_compilation_check:
            return await self._chat_without_compilation_check(user_message)
        
        # Retrieve relevant context
        context_chunks = await self.retrieve_context(user_message)
        
        # Format context
        context = self.format_context(context_chunks)

        # Add user message to conversation
        user_msg = HumanMessage(content=user_message)
        self.conversation_history.append(user_msg)
        
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Create system message with context
                system_msg = SystemMessage(content=self.system_message.replace('<context>', context))
                
                # Get response from OpenAI
                response = await self.chat_model.ainvoke(
                    [system_msg] + self.conversation_history[-20:]
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
                        [system_msg] + self.conversation_history[-20:]
                    )

                # Check if code compilation checking is enabled
                if self.config.code_compilation_check:
                    # Validate any Python code in the response
                    is_valid, error_messages = validate_response_code(response.content[0].get("text", "") if isinstance(response.content, list) else response.content)
                    
                    if not is_valid and retry_count < max_retries:
                        # Create error feedback message
                        error_feedback = f"""
The Python code in your previous response has compilation errors:

{chr(10).join(error_messages)}

Please fix these errors and provide a corrected response with syntactically valid Python code.
"""
                        
                        # Add error feedback to conversation
                        error_msg = HumanMessage(content=error_feedback)
                        self.conversation_history.append(error_msg)
                        
                        retry_count += 1
                        logger.warning(f"Code compilation failed, retrying ({retry_count}/{max_retries})")
                        continue

                # Update conversation history with successful response
                self.conversation_history.append(response)
                return response.content[0].get("text", "") if isinstance(response.content, list) else response.content
                
            except Exception as e:
                logger.error(f"Error in chat function: {e}")
                if retry_count >= max_retries:
                    return f"I apologize, but I encountered an error: {str(e)}"
                retry_count += 1
        
        # This should never be reached, but just in case
        raise Exception("Unexpected error in chat function")

    async def _chat_without_compilation_check(self, user_message: str) -> str:
        """Original chat function without compilation checking (for backward compatibility)"""
        # Retrieve relevant context
        context_chunks = await self.retrieve_context(user_message)
        
        # Format context
        context = self.format_context(context_chunks)

        # Add user message to conversation
        user_msg = HumanMessage(content=user_message)
        self.conversation_history.append(user_msg)
        
        # Create system message with context
        system_msg = SystemMessage(content=self.system_message.replace('<context>', context))
        
        # Get response from OpenAI
        response = await self.chat_model.ainvoke(
            [system_msg] + self.conversation_history[-20:]
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
                [system_msg] + self.conversation_history[-20:]
            )
        
        # Update conversation history
        self.conversation_history.append(response)
        return response.content[0].get("text", "") if isinstance(response.content, list) else response.content

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

        # Prepare the base system message
        base_system_message = self._system_message.content.format(
            tools_info=tools_info if tools_found else ""
        )
        
        # Append custom instructions if provided
        if custom_instructions.strip():
            base_system_message += f"\n\nAdditional Instructions:\n{custom_instructions}"
        
        self.system_message = base_system_message

        # Enable query_database tool if tools are found
        if tools_found:
            self.chat_model = self.chat_model.bind_tools([query_database])
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation history"""
        if not self.conversation_history:
            return "No conversation history available."
        
        messages_text = []
        for msg in self.conversation_history[-6:]:  # Last 6 messages
            if isinstance(msg, HumanMessage):
                messages_text.append(f"User: {msg.content[:100]}...")
            elif isinstance(msg, AIMessage):
                messages_text.append(f"Assistant: {msg.content[:100]}...")
        
        return "\n".join(messages_text)
    
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