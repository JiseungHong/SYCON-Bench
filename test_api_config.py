#!/usr/bin/env python3
"""
Test suite for the unified API configuration system.

This test suite verifies that the APIConfig class correctly handles
different model providers and their respective environment variables.
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add the workspace directory to the path
sys.path.insert(0, '/workspace')

from api_config import APIConfig


class TestAPIConfig(unittest.TestCase):
    """Test cases for the APIConfig class."""

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

    def test_get_api_key_env_var_openai(self):
        """Test getting environment variable for OpenAI models."""
        test_cases = [
            ("openai/gpt-4o", "OPENAI_API_KEY"),
            ("openai/gpt-3.5-turbo", "OPENAI_API_KEY"),
            ("gpt-4", "OPENAI_API_KEY"),
            ("gpt-3.5-turbo", "OPENAI_API_KEY"),
        ]

        for model_id, expected_env_var in test_cases:
            with self.subTest(model_id=model_id):
                result = APIConfig.get_api_key_env_var(model_id)
                self.assertEqual(result, expected_env_var)

    def test_get_api_key_env_var_anthropic(self):
        """Test getting environment variable for Anthropic models."""
        test_cases = [
            ("anthropic/claude-3-sonnet", "ANTHROPIC_API_KEY"),
            ("claude-3-haiku", "ANTHROPIC_API_KEY"),
            ("claude-2", "ANTHROPIC_API_KEY"),
        ]

        for model_id, expected_env_var in test_cases:
            with self.subTest(model_id=model_id):
                result = APIConfig.get_api_key_env_var(model_id)
                self.assertEqual(result, expected_env_var)

    def test_get_api_key_env_var_azure(self):
        """Test getting environment variable for Azure models."""
        test_cases = [
            ("azure/gpt-4o", "AZURE_OPENAI_API_KEY"),
            ("azure-openai/gpt-35-turbo", "AZURE_OPENAI_API_KEY"),
        ]

        for model_id, expected_env_var in test_cases:
            with self.subTest(model_id=model_id):
                result = APIConfig.get_api_key_env_var(model_id)
                self.assertEqual(result, expected_env_var)

    def test_get_api_key_env_var_google(self):
        """Test getting environment variable for Google models."""
        test_cases = [
            ("google/gemini-pro", "GOOGLE_API_KEY"),
            ("gemini-1.5-pro", "GOOGLE_API_KEY"),
        ]

        for model_id, expected_env_var in test_cases:
            with self.subTest(model_id=model_id):
                result = APIConfig.get_api_key_env_var(model_id)
                self.assertEqual(result, expected_env_var)

    def test_get_api_key_env_var_unknown(self):
        """Test getting environment variable for unknown models."""
        with patch('api_config.logging') as mock_logging:
            result = APIConfig.get_api_key_env_var("unknown/model")
            self.assertEqual(result, "OPENAI_API_KEY")
            mock_logging.warning.assert_called_once()

    def test_get_api_key_with_explicit_key(self):
        """Test getting API key when explicitly provided."""
        api_key = "test-api-key-123"
        result = APIConfig.get_api_key("openai/gpt-4o", api_key)
        self.assertEqual(result, api_key)

    def test_get_api_key_from_environment(self):
        """Test getting API key from environment variable."""
        os.environ["OPENAI_API_KEY"] = "env-api-key-456"
        result = APIConfig.get_api_key("openai/gpt-4o")
        self.assertEqual(result, "env-api-key-456")

    def test_get_api_key_fallback(self):
        """Test API key fallback mechanism."""
        os.environ["ANTHROPIC_API_KEY"] = "fallback-key-789"

        with patch('api_config.logging') as mock_logging:
            result = APIConfig.get_api_key("openai/gpt-4o")
            self.assertEqual(result, "fallback-key-789")
            mock_logging.warning.assert_called()

    def test_get_api_key_no_key_found(self):
        """Test error when no API key is found."""
        with self.assertRaises(ValueError) as context:
            APIConfig.get_api_key("openai/gpt-4o")

        self.assertIn("No API key found", str(context.exception))
        self.assertIn("OPENAI_API_KEY", str(context.exception))

    def test_get_provider_config_openai(self):
        """Test getting provider configuration for OpenAI models."""
        os.environ["OPENAI_API_KEY"] = "test-openai-key"

        config = APIConfig.get_provider_config("openai/gpt-4o")

        self.assertEqual(config["api_key"], "test-openai-key")
        self.assertEqual(config["model_id"], "openai/gpt-4o")
        self.assertNotIn("azure_endpoint", config)

    def test_get_provider_config_azure(self):
        """Test getting provider configuration for Azure models."""
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com/"

        config = APIConfig.get_provider_config("azure/gpt-4o")

        self.assertEqual(config["api_key"], "test-azure-key")
        self.assertEqual(config["model_id"], "azure/gpt-4o")
        self.assertEqual(config["azure_endpoint"], "https://test.openai.azure.com/")
        self.assertEqual(config["api_version"], "2024-02-15-preview")

    def test_get_provider_config_azure_missing_endpoint(self):
        """Test Azure configuration with missing endpoint."""
        os.environ["AZURE_OPENAI_API_KEY"] = "test-azure-key"

        with patch('api_config.logging') as mock_logging:
            config = APIConfig.get_provider_config("azure/gpt-4o")
            mock_logging.warning.assert_called_with("AZURE_OPENAI_ENDPOINT not set. Azure OpenAI calls may fail.")

    def test_get_provider_config_with_base_url(self):
        """Test getting provider configuration with custom base URL."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        base_url = "https://custom.api.endpoint.com"

        config = APIConfig.get_provider_config("openai/gpt-4o", base_url=base_url)

        self.assertEqual(config["base_url"], base_url)

    def test_validate_config_valid_openai(self):
        """Test configuration validation for valid OpenAI setup."""
        os.environ["OPENAI_API_KEY"] = "test-key"

        is_valid, error_msg = APIConfig.validate_config("openai/gpt-4o")

        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "Configuration is valid")

    def test_validate_config_invalid_azure(self):
        """Test configuration validation for invalid Azure setup."""
        os.environ["AZURE_OPENAI_API_KEY"] = "test-key"
        # Missing AZURE_OPENAI_ENDPOINT

        is_valid, error_msg = APIConfig.validate_config("azure/gpt-4o")

        self.assertFalse(is_valid)
        self.assertIn("Azure OpenAI endpoint", error_msg)

    def test_validate_config_valid_azure(self):
        """Test configuration validation for valid Azure setup."""
        os.environ["AZURE_OPENAI_API_KEY"] = "test-key"
        os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com/"

        is_valid, error_msg = APIConfig.validate_config("azure/gpt-4o")

        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "Configuration is valid")

    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        is_valid, error_msg = APIConfig.validate_config("openai/gpt-4o")

        self.assertFalse(is_valid)
        self.assertIn("No API key found", error_msg)

    def test_list_supported_providers(self):
        """Test listing supported providers."""
        providers = APIConfig.list_supported_providers()

        self.assertIsInstance(providers, dict)
        self.assertIn("Openai", providers)
        self.assertIn("Anthropic", providers)
        self.assertIn("Azure", providers)
        self.assertEqual(providers["Openai"], "OPENAI_API_KEY")
        self.assertEqual(providers["Anthropic"], "ANTHROPIC_API_KEY")


class TestModelIntegration(unittest.TestCase):
    """Test integration with model classes."""

    def setUp(self):
        """Set up test environment."""
        self.original_env = os.environ.copy()
        # Clear any existing environment variables
        for key in list(os.environ.keys()):
            if key.endswith('_API_KEY'):
                del os.environ[key]

    def tearDown(self):
        """Clean up test environment."""
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_closed_model_initialization_openai(self):
        """Test ClosedModel initialization with OpenAI model."""
        os.environ["OPENAI_API_KEY"] = "test-openai-key"

        # Import here to avoid import issues during test discovery
        sys.path.insert(0, '/workspace/debate-setting')
        from models import ClosedModel

        model = ClosedModel("openai/gpt-4o")
        self.assertEqual(model.api_key, "test-openai-key")
        self.assertEqual(model.model_id, "openai/gpt-4o")

        # Test setup
        result = model.setup()
        self.assertTrue(result)

    def test_closed_model_initialization_anthropic(self):
        """Test ClosedModel initialization with Anthropic model."""
        os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"

        sys.path.insert(0, '/workspace/debate-setting')
        from models import ClosedModel

        model = ClosedModel("anthropic/claude-3-sonnet")
        self.assertEqual(model.api_key, "test-anthropic-key")
        self.assertEqual(model.model_id, "anthropic/claude-3-sonnet")

    def test_closed_model_initialization_no_key(self):
        """Test ClosedModel initialization without API key."""
        sys.path.insert(0, '/workspace/debate-setting')
        from models import ClosedModel

        model = ClosedModel("openai/gpt-4o")
        self.assertIsNone(model.api_key)

        # Test setup should raise error
        with self.assertRaises(ValueError) as context:
            model.setup()

        self.assertIn("No API key found", str(context.exception))

    def test_model_factory_openai(self):
        """Test ModelFactory with OpenAI model."""
        os.environ["OPENAI_API_KEY"] = "test-key"

        sys.path.insert(0, '/workspace/debate-setting')
        from models import ModelFactory, ClosedModel

        model = ModelFactory.create_model("openai/gpt-4o")
        self.assertIsInstance(model, ClosedModel)

    def test_model_factory_anthropic(self):
        """Test ModelFactory with Anthropic model."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"

        sys.path.insert(0, '/workspace/debate-setting')
        from models import ModelFactory, ClosedModel

        model = ModelFactory.create_model("anthropic/claude-3-sonnet")
        self.assertIsInstance(model, ClosedModel)

    def test_model_factory_open_model(self):
        """Test ModelFactory with open-source model."""
        sys.path.insert(0, '/workspace/debate-setting')
        from models import ModelFactory, OpenModel

        model = ModelFactory.create_model("meta-llama/Llama-2-7b-hf")
        self.assertIsInstance(model, OpenModel)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)
