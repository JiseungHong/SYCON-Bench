
"""
Configuration module for SYCON-Bench.

This module provides a unified way to handle API keys and configuration
for different model providers.
"""

import os
from typing import Optional

# Mapping of model prefixes to environment variable names
API_KEY_ENV_MAP = {
    "openai/": "OPENAI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",  # Support for claude* models
    # Add more providers as needed
    "azure/": "AZURE_API_KEY",
    "google/": "GOOGLE_API_KEY",
    "mistral/": "MISTRAL_API_KEY",
    "cohere/": "COHERE_API_KEY",
}

def get_api_key_for_model(model_id: str, provided_api_key: Optional[str] = None) -> Optional[str]:
    """
    Get the appropriate API key for a given model.

    Args:
        model_id: The model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-opus")
        provided_api_key: An explicitly provided API key (takes precedence over environment variables)

    Returns:
        The API key if found, None otherwise
    """
    # If an API key is explicitly provided, use it
    if provided_api_key is not None:
        return provided_api_key

    # Check for model-specific environment variables
    for prefix, env_var in API_KEY_ENV_MAP.items():
        if model_id.startswith(prefix):
            return os.environ.get(env_var)

    # Fallback to OPENAI_API_KEY for backward compatibility
    return os.environ.get("OPENAI_API_KEY")

def get_required_api_key_for_model(model_id: str, provided_api_key: Optional[str] = None) -> str:
    """
    Get the required API key for a given model, raising an error if not found.

    Args:
        model_id: The model identifier
        provided_api_key: An explicitly provided API key

    Returns:
        The API key

    Raises:
        ValueError: If no API key is found
    """
    api_key = get_api_key_for_model(model_id, provided_api_key)
    if api_key is None:
        # Determine which environment variable should be set
        env_var = "OPENAI_API_KEY"  # Default fallback
        for prefix, var_name in API_KEY_ENV_MAP.items():
            if model_id.startswith(prefix):
                env_var = var_name
                break

        raise ValueError(
            f"No API key provided for model '{model_id}'. "
            f"Please provide via api_key parameter or set {env_var} environment variable."
        )

    return api_key

def get_provider_name(model_id: str) -> str:
    """
    Get the provider name for a given model ID.

    Args:
        model_id: The model identifier

    Returns:
        The provider name
    """
    for prefix, _ in API_KEY_ENV_MAP.items():
        if model_id.startswith(prefix):
            # Extract provider name from prefix
            if prefix.endswith("/"):
                return prefix[:-1]  # Remove trailing slash
            return prefix
    return "unknown"
