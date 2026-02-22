"""
AI Analysis module for code execution outputs
"""
import logging
from typing import Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class CodeOutputAnalyzer:
    """Analyzes code execution outputs using OpenAI API"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        """Initialize the analyzer with OpenAI API key"""
        self.chat_model = ChatOpenAI(
            api_key=openai_api_key,
            model_name=model,
            max_tokens=1500,
            temperature=0.2  # Lower temperature for more focused analysis
        )
    
    async def analyze_code_output(self, code: str, output: str) -> str:
        """
        Analyze code and its execution output using AI
        
        Args:
            code: The Python code that was executed
            output: The output from the code execution
            
        Returns:
            AI analysis and insights about the code and output
        """
        try:
            # Create system message for analysis
            system_message = SystemMessage(content="""
You are an expert code analyst and data scientist. Your task is to analyze Python code and its execution output to provide valuable insights.

When analyzing code and output, focus on:

1. **Code Quality & Logic:**
   - Review the code structure and logic
   - Identify any potential improvements or best practices
   - Note any inefficiencies or optimization opportunities

2. **Output Analysis:**
   - Interpret what the output means
   - Identify patterns, trends, or significant findings in the results
   - Explain any errors or warnings if present

3. **Data Insights (if applicable):**
   - Summarize key findings from data analysis or visualizations
   - Highlight important statistics, patterns, or outliers
   - Suggest potential next steps for further analysis

4. **Practical Recommendations:**
   - Suggest improvements to the code or approach
   - Recommend additional analyses that might be valuable
   - Point out any potential issues or limitations

5. **Context & Significance:**
   - Explain the significance of the results
   - Provide context for interpreting the output
   - Connect findings to potential real-world applications

Format your response clearly with sections using markdown headers. Be concise but thorough.
If the output contains errors, focus on explaining what went wrong and how to fix it.
If the output contains data or results, focus on interpreting and explaining their significance.
""")
            
            # Create user message with code and output
            user_message = HumanMessage(content=f"""
Please analyze the following Python code and its execution output:

**Code:**
```python
{code}
```

**Output:**
```
{output}
```

Please provide a comprehensive analysis covering code quality, output interpretation, insights, and recommendations.
""")
            
            # Get analysis from OpenAI
            response = await self.chat_model.ainvoke([system_message, user_message])
            
            analysis_content = response.content
            if isinstance(analysis_content, list):
                analysis_content = analysis_content[0].get("text", "") if analysis_content else ""
            
            return analysis_content
            
        except Exception as e:
            logger.error(f"Error in AI analysis: {str(e)}")
            return f"❌ **Analysis Error**\n\nSorry, I encountered an error while analyzing the code output: {str(e)}\n\nPlease try again or check your API key configuration."
    
    def analyze_code_output_sync(self, code: str, output: str) -> str:
        """
        Synchronous wrapper for analyze_code_output
        This is for compatibility with synchronous contexts
        """
        import asyncio
        
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to handle this differently
                # This shouldn't normally happen in Streamlit, but just in case
                return "❌ **Analysis Error**: Cannot run synchronous analysis from async context. Please use the async version."
            else:
                return loop.run_until_complete(self.analyze_code_output(code, output))
        except RuntimeError:
            # No event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.analyze_code_output(code, output))
                return result
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous analysis wrapper: {str(e)}")
            return f"❌ **Analysis Error**\n\nSorry, I encountered an error while analyzing the code output: {str(e)}"

def create_analyzer(openai_api_key: str) -> CodeOutputAnalyzer:
    """
    Factory function to create a CodeOutputAnalyzer instance
    
    Args:
        openai_api_key: OpenAI API key for authentication
        
    Returns:
        Configured CodeOutputAnalyzer instance
    """
    return CodeOutputAnalyzer(openai_api_key)