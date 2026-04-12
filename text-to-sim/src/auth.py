from langchain_openai import ChatOpenAI


def validate_openai_api_key(api_key):
    """Validate OpenAI API key by making a simple test request"""
    try:
        test_model = ChatOpenAI(
            model_name="o3-mini",
            max_tokens=10,
            api_key=api_key,
        )
        _ = test_model.invoke("Test")
        return True, "API key is valid!"
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid api key" in error_msg or "incorrect api key" in error_msg:
            return False, "Invalid API key. Please check your OpenAI API key."
        elif "quota" in error_msg or "billing" in error_msg:
            return False, "API key is valid but you may have exceeded your quota or have billing issues."
        elif "rate limit" in error_msg:
            return True, "API key is valid (rate limited, but that's expected)."
        else:
            return False, f"Error validating API key: {str(e)}"
