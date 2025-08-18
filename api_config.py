"""
Unified API configuration system for SYCON-Bench.

This module provides a consistent way to handle API keys and configurations
for different model providers across all settings.
"""
import os
import logging
from typing import Dict, Optional, Tuple


class APIConfig:
    """Unified API configuration manager for different model providers."""

    # Mapping of model prefixes to their respective environment variables
    MODEL_ENV_MAPPING = {
        # OpenAI models
        "openai/": "OPENAI_API_KEY",
        "gpt-": "OPENAI_API_KEY",

        # Anthropic models
        "anthropic/": "ANTHROPIC_API_KEY",
        "claude": "ANTHROPIC_API_KEY",

        # Azure OpenAI models
        "azure/": "AZURE_OPENAI_API_KEY",
        "azure-openai/": "AZURE_OPENAI_API_KEY",

        # Google models
        "google/": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY",

        # Cohere models
        "cohere/": "COHERE_API_KEY",

        # Hugging Face models
        "huggingface/": "HUGGINGFACE_API_KEY",
        "hf/": "HUGGINGFACE_API_KEY",
    }

    # Additional environment variables that might be needed for specific providers
    PROVIDER_ADDITIONAL_VARS = {
        "azure": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"],
        "google": ["GOOGLE_PROJECT_ID"],
    }

    @classmethod
    def get_api_key_env_var(cls, model_id: str) -> str:
        """
        Get the appropriate environment variable name for a given model ID.

        Args:
            model_id (str): The model identifier (e.g., "openai/gpt-4o", "claude-3-sonnet")

        Returns:
            str: The environment variable name for the API key

        Raises:
            ValueError: If no matching environment variable is found
        """
        model_id_lower = model_id.lower()

        # Check each prefix mapping
        for prefix, env_var in cls.MODEL_ENV_MAPPING.items():
            if model_id_lower.startswith(prefix):
                return env_var

        # Default fallback for unknown models
        logging.warning(f"No specific API key mapping found for model '{model_id}'. Using OPENAI_API_KEY as fallback.")
        return "OPENAI_API_KEY"

    @classmethod
    def get_api_key(cls, model_id: str, api_key: Optional[str] = None) -> str:
        """
        Get the API key for a given model, with fallback to environment variables.

        Args:
            model_id (str): The model identifier
            api_key (Optional[str]): Explicitly provided API key

        Returns:
            str: The API key to use

        Raises:
            ValueError: If no API key is found
        """
        if api_key:
            return api_key

        env_var = cls.get_api_key_env_var(model_id)
        api_key_from_env = os.environ.get(env_var)

        if not api_key_from_env:
            # Try common fallbacks
            fallback_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "API_KEY"]
            for fallback_var in fallback_vars:
                fallback_key = os.environ.get(fallback_var)
                if fallback_key:
                    logging.warning(f"Using {fallback_var} as fallback for model '{model_id}'")
                    return fallback_key

            raise ValueError(
                f"No API key found for model '{model_id}'. "
                f"Please provide via api_key parameter or set {env_var} environment variable. "
                f"Supported environment variables: {', '.join(cls.MODEL_ENV_MAPPING.values())}"
            )

        return api_key_from_env

    @classmethod
    def get_provider_config(cls, model_id: str, api_key: Optional[str] = None,
                          base_url: Optional[str] = None) -> Dict[str, str]:
        """
        Get complete configuration for a model provider.

        Args:
            model_id (str): The model identifier
            api_key (Optional[str]): Explicitly provided API key
            base_url (Optional[str]): Custom base URL

        Returns:
            Dict[str, str]: Configuration dictionary with api_key and other provider-specific settings
        """
        config = {
            "api_key": cls.get_api_key(model_id, api_key),
            "model_id": model_id
        }

        if base_url:
            config["base_url"] = base_url

        # Add provider-specific configurations
        model_id_lower = model_id.lower()

        if model_id_lower.startswith(("azure/", "azure-openai/")):
            # Azure OpenAI specific configurations
            config["azure_endpoint"] = os.environ.get("AZURE_OPENAI_ENDPOINT")
            config["api_version"] = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

            if not config["azure_endpoint"]:
                logging.warning("AZURE_OPENAI_ENDPOINT not set. Azure OpenAI calls may fail.")

        elif model_id_lower.startswith("google/"):
            # Google specific configurations
            config["project_id"] = os.environ.get("GOOGLE_PROJECT_ID")

        return config

    @classmethod
    def validate_config(cls, model_id: str, api_key: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validate that the configuration for a model is complete.

        Args:
            model_id (str): The model identifier
            api_key (Optional[str]): Explicitly provided API key

        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            config = cls.get_provider_config(model_id, api_key)

            # Check for required fields based on provider
            model_id_lower = model_id.lower()

            if model_id_lower.startswith(("azure/", "azure-openai/")):
                if not config.get("azure_endpoint"):
                    return False, "Azure OpenAI endpoint (AZURE_OPENAI_ENDPOINT) is required for Azure models"

            return True, "Configuration is valid"

        except ValueError as e:
            return False, str(e)

    @classmethod
    def list_supported_providers(cls) -> Dict[str, str]:
        """
        List all supported providers and their environment variables.

        Returns:
            Dict[str, str]: Mapping of provider names to environment variables
        """
        providers = {}
        for prefix, env_var in cls.MODEL_ENV_MAPPING.items():
            provider_name = prefix.rstrip("/").replace("/", " ").title()
            if provider_name not in providers:
                providers[provider_name] = env_var

        return providers
