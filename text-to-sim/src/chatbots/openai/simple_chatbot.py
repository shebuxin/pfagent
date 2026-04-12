from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import os
import json
import logging

# Core LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

# Additional utilities
import re
import ast

from dotenv import load_dotenv
from src.andes_manual import BASE_ANDES_MANUAL_POLICY
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
class SimpleChatConfig:
    """Configuration for Simple Chat system"""
    openai_api_key: str
    chat_model: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    max_tokens: int = 2000
    code_compilation_check: bool = True  # Enable/disable code compilation checking
    max_compilation_retries: int = 2     # Maximum retries for compilation errors
    conversation_compaction_enabled: bool = True
    conversation_compaction_trigger_messages: int = 14
    conversation_compaction_keep_recent_messages: int = 8
    conversation_compaction_max_summary_chars: int = 3200

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

class SimpleChatbot:
    """Simple chatbot using only OpenAI without RAG"""
    
    def __init__(self, config: SimpleChatConfig):
        self.config = config
        self.system_message = None
        
        # Initialize OpenAI components
        self.chat_model = ChatOpenAI(
            api_key=config.openai_api_key,
            model_name=config.chat_model,
            max_tokens=config.max_tokens
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
        
        self.prompt_builder = AndesSystemPromptBuilder(
            AndesPromptBuilderConfig(
                include_context_placeholder=False,
                include_tools_info=False,
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
                "Using compacted conversation memory for simple chat (%d history messages).",
                len(self.conversation_history),
            )
        return messages

    async def process_documents(self, documents: List[str], doc_ids: List[str] = None, session_id: str = None, persistent_flags: List[bool] = None):
        """Process documents - No-op for simple chatbot since it doesn't use RAG"""
        logger.info("Simple chatbot doesn't process documents - skipping document processing")
        pass
    
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
        
        retry_count = 0
        while retry_count <= max_retries:
            try:
                # Get response from OpenAI
                response = await self.chat_model.ainvoke(
                    self._build_model_messages(self.system_message)
                )
                response_text = response.content[0].get("text", "") if isinstance(response.content, list) else response.content
                normalized_response_text, normalization_notes = ensure_python_code_block(
                    response_text,
                    user_context=user_message,
                )
                if normalization_notes:
                    logger.info("Applied code-block normalization: %s", "; ".join(normalization_notes))

                # Check if code compilation checking is enabled
                if self.config.code_compilation_check:
                    # Validate any Python code in the response
                    is_valid, error_messages = validate_response_code(
                        normalized_response_text,
                        user_context=user_message,
                    )
                    
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
        
        # Get response from OpenAI
        response = await self.chat_model.ainvoke(
            self._build_model_messages(self.system_message)
        )
        response_text = response.content[0].get("text", "") if isinstance(response.content, list) else response.content
        normalized_response_text, normalization_notes = ensure_python_code_block(
            response_text,
            user_context=user_message,
        )
        if normalization_notes:
            logger.info("Applied code-block normalization: %s", "; ".join(normalization_notes))
        
        # Update conversation history
        self.conversation_history.append(AIMessage(content=normalized_response_text))
        return normalized_response_text

    def load_system_prompt(self, session_id: str = None, custom_instructions: str = ""):
        """Load system prompt with custom instructions"""
        self.system_message = self.prompt_builder.build_prompt(
            andes_manual_policy=BASE_ANDES_MANUAL_POLICY,
            custom_instructions=custom_instructions,
        )
    
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
        """Clean up session - No-op for simple chatbot"""
        logger.info(f"Simple chatbot doesn't need session cleanup for session {session_id}")
        pass
    
    def close(self):
        """Clean up resources - No-op for simple chatbot"""
        pass

# Example usage and testing
async def main():
    """Example usage of the Simple chatbot"""
    
    # Configuration
    config = SimpleChatConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY")
    )
    
    # Initialize chatbot
    chatbot = SimpleChatbot(config)
    
    # Load system prompt
    chatbot.load_system_prompt(custom_instructions="Be helpful and concise.")
    
    # Interactive chat loop
    print("\n=== Simple Chatbot Ready ===")
    print("Ask any questions. Type 'quit' to exit.\n")
    
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
