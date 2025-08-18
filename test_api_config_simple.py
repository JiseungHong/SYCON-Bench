#!/usr/bin/env python3
"""
Simple test suite for the unified API configuration system.

This test suite focuses only on the APIConfig class functionality
without importing heavy dependencies like torch.
"""
import os
import sys
import unittest
from unittest.mock import patch

# Add the workspace directory to the path
sys.path.insert(0, '/workspace')

from api_config import APIConfig


class TestAPIConfigCore(unittest.TestCase):
    """Test cases for the core APIConfig functionality."""

    def setUp(self):
        """Set up test environment."""
        # Clear any existing environment variables
        self.original_env = os.environ.copy()
        for key in list(os.environ.keys()):
            if key.endswith('_API_KEY') or key.startswith('AZURE_') or key.startswith('GOOGLE_'):
                del os.environ[key]

    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_openai_models(self):
        """Test OpenAI model detection and configuration."""
        test_cases = [
            "openai/gpt-4o",
            "openai/gpt-3.5-turbo",
            "gpt-4",
            "gpt-3.5-turbo"
        ]

        for model_id in test_cases:
            with self.subTest(model_id=model_id):
                env_var = APIConfig.get_api_key_env_var(model_id)
                self.assertEqual(env_var, "OPENAI_API_KEY")

    def test_anthropic_models(self):
        """Test Anthropic model detection and configuration."""
        test_cases = [
            "anthropic/claude-3-sonnet",
            "claude-3-haiku",
            "claude-2"
        ]

        for model_id in test_cases:
            with self.subTest(model_id=model_id):
                env_var = APIConfig.get_api_key_env_var(model_id)
                self.assertEqual(env_var, "ANTHROPIC_API_KEY")

    def test_azure_models(self):
        """Test Azure model detection and configuration."""
        test_cases = [
            "azure/gpt-4o",
            "azure-openai/gpt-35-turbo"
        ]

        for model_id in test_cases:
            with self.subTest(model_id=model_id):
                env_var = APIConfig.get_api_key_env_var(model_id)
                self.assertEqual(env_var, "AZURE_OPENAI_API_KEY")

    def test_api_key_retrieval(self):
        """Test API key retrieval from environment."""
        os.environ["OPENAI_API_KEY"] = "test-openai-key"
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

        # Test direct retrieval
        openai_key = APIConfig.get_api_key("openai/gpt-4o")
        self.assertEqual(openai_key, "test-openai-key")

        anthropic_key = APIConfig.get_api_key("anthropic/claude-3-sonnet")
        self.assertEqual(anthropic_key, "test-anthropic-key")

    def test_api_key_explicit_override(self):
        """Test that explicit API key overrides environment."""
        os.environ["OPENAI_API_KEY"] = "env-key"

        explicit_key = "explicit-key"
        result = APIConfig.get_api_key("openai/gpt-4o", explicit_key)
        self.assertEqual(result, explicit_key)

    def test_fallback_mechanism(self):
        """Test API key fallback mechanism."""
        # Set only Anthropic key
        os.environ["ANTHROPIC_API_KEY"] = "fallback-key"

        with patch('api_config.logging') as mock_logging:
            # Request OpenAI key, should fallback to Anthropic
            result = APIConfig.get_api_key("openai/gpt-4o")
            self.assertEqual(result, "fallback-key")
            mock_logging.warning.assert_called()

    def test_no_api_key_error(self):
        """Test error when no API key is available."""
        with self.assertRaises(ValueError) as context:
            APIConfig.get_api_key("openai/gpt-4o")

        error_msg = str(context.exception)
        self.assertIn("No API key found", error_msg)
        self.assertIn("OPENAI_API_KEY", error_msg)

    def test_azure_configuration(self):
        """Test Azure-specific configuration."""
        os.environ["AZURE_OPENAI_API_KEY"] = "azure-key"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com/"
        os.environ["AZURE_OPENAI_API_VERSION"] = "2023-12-01-preview"

        config = APIConfig.get_provider_config("azure/gpt-4o")

        self.assertEqual(config["api_key"], "azure-key")
        self.assertEqual(config["azure_endpoint"], "https://test.openai.azure.com/")
        self.assertEqual(config["api_version"], "2023-12-01-preview")

    def test_azure_missing_endpoint_warning(self):
        """Test warning when Azure endpoint is missing."""
        os.environ["AZURE_OPENAI_API_KEY"] = "azure-key"

        with patch('api_config.logging') as mock_logging:
            config = APIConfig.get_provider_config("azure/gpt-4o")
            mock_logging.warning.assert_called_with(
                "AZURE_OPENAI_ENDPOINT not set. Azure OpenAI calls may fail."
            )

    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid OpenAI configuration
        os.environ["OPENAI_API_KEY"] = "test-key"
        is_valid, msg = APIConfig.validate_config("openai/gpt-4o")
        self.assertTrue(is_valid)
        self.assertEqual(msg, "Configuration is valid")

        # Invalid configuration (no API key)
        del os.environ["OPENAI_API_KEY"]
        is_valid, msg = APIConfig.validate_config("openai/gpt-4o")
        self.assertFalse(is_valid)
        self.assertIn("No API key found", msg)

    def test_azure_validation(self):
        """Test Azure configuration validation."""
        # Missing endpoint
        os.environ["AZURE_OPENAI_API_KEY"] = "azure-key"
        is_valid, msg = APIConfig.validate_config("azure/gpt-4o")
        self.assertFalse(is_valid)
        self.assertIn("Azure OpenAI endpoint", msg)

        # Valid Azure configuration
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com/"
        is_valid, msg = APIConfig.validate_config("azure/gpt-4o")
        self.assertTrue(is_valid)
        self.assertEqual(msg, "Configuration is valid")

    def test_supported_providers_list(self):
        """Test listing supported providers."""
        providers = APIConfig.list_supported_providers()

        self.assertIsInstance(providers, dict)
        self.assertIn("Openai", providers)
        self.assertIn("Anthropic", providers)
        self.assertIn("Azure", providers)
        self.assertIn("Google", providers)

        # Check that environment variables are correct
        self.assertEqual(providers["Openai"], "OPENAI_API_KEY")
        self.assertEqual(providers["Anthropic"], "ANTHROPIC_API_KEY")

    def test_unknown_model_fallback(self):
        """Test handling of unknown model types."""
        with patch('api_config.logging') as mock_logging:
            env_var = APIConfig.get_api_key_env_var("unknown/weird-model")
            self.assertEqual(env_var, "OPENAI_API_KEY")
            mock_logging.warning.assert_called_once()

    def test_case_insensitive_matching(self):
        """Test that model matching is case insensitive."""
        test_cases = [
            ("OpenAI/GPT-4o", "OPENAI_API_KEY"),
            ("ANTHROPIC/CLAUDE-3-SONNET", "ANTHROPIC_API_KEY"),
            ("Azure/gpt-4o", "AZURE_OPENAI_API_KEY"),
        ]

        for model_id, expected_env_var in test_cases:
            with self.subTest(model_id=model_id):
                result = APIConfig.get_api_key_env_var(model_id)
                self.assertEqual(result, expected_env_var)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
