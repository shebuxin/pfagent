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
        
        # Base system message
        self._system_message = SystemMessage(content="""
You are a helpful AI assistant. Be conversational and helpful in your responses.

When answering:
1. Be clear and comprehensive in your explanations
2. If you don't know something, say so clearly
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

{custom_instructions}
""")

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
                    [SystemMessage(content=self.system_message)] + self.conversation_history[-20:]
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
        # Add user message to conversation
        user_msg = HumanMessage(content=user_message)
        self.conversation_history.append(user_msg)
        
        # Get response from OpenAI
        response = await self.chat_model.ainvoke(
            [SystemMessage(content=self.system_message)] + self.conversation_history[-20:]
        )
        
        # Update conversation history
        self.conversation_history.append(response)
        return response.content[0].get("text", "") if isinstance(response.content, list) else response.content

    def load_system_prompt(self, session_id: str = None, custom_instructions: str = ""):
        """Load system prompt with custom instructions"""
        # Prepare the base system message with custom instructions
        base_system_message = self._system_message.content.format(
            custom_instructions=f"\n\nAdditional Instructions:\n{custom_instructions}" if custom_instructions.strip() else ""
        )
        
        self.system_message = base_system_message
    
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