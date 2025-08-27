import os
import logging

API_KEY_MAPPING = {
    "openai/": "OPENAI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}

def get_api_key(model_name: str, api_key: str = None) -> str:
    """
    Get the API key for a given model name.
    """
    if api_key:
        return api_key

    for prefix, env_var in API_KEY_MAPPING.items():
        if model_name.startswith(prefix):
            api_key = os.environ.get(env_var)
            if api_key:
                return api_key

    # Default to OPENAI_API_KEY for backward compatibility
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    logging.error("No API key found for model %s. Please set the corresponding environment variable.", model_name)
    raise ValueError(f"No API key found for model {model_name}")

