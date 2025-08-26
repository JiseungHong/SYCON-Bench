
#!/usr/bin/env python3
"""
Test script for the configuration system.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add the workspace directory to the path so we can import config
sys.path.insert(0, '/workspace')

from config import get_api_key_for_model, get_required_api_key_for_model, get_provider_name

class TestConfig(unittest.TestCase):

    def setUp(self):
        # Clear environment variables before each test
        for var in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'AZURE_API_KEY']:
            if var in os.environ:
                del os.environ[var]

    def test_get_api_key_for_model_with_provided_key(self):
        """Test that provided API key takes precedence."""
        api_key = get_api_key_for_model("openai/gpt-4o", "test-key")
        self.assertEqual(api_key, "test-key")

    def test_get_api_key_for_model_openai_from_env(self):
        """Test getting OpenAI API key from environment."""
        os.environ['OPENAI_API_KEY'] = 'test-openai-key'
        api_key = get_api_key_for_model("openai/gpt-4o")
        self.assertEqual(api_key, 'test-openai-key')

    def test_get_api_key_for_model_anthropic_from_env(self):
        """Test getting Anthropic API key from environment."""
        os.environ['ANTHROPIC_API_KEY'] = 'test-anthropic-key'
        api_key = get_api_key_for_model("anthropic/claude-3-opus")
        self.assertEqual(api_key, 'test-anthropic-key')

    def test_get_api_key_for_model_claude_from_env(self):
        """Test getting Claude API key from environment."""
        os.environ['ANTHROPIC_API_KEY'] = 'test-anthropic-key'
        api_key = get_api_key_for_model("claude-3-sonnet")
        self.assertEqual(api_key, 'test-anthropic-key')

    def test_get_api_key_for_model_fallback_to_openai(self):
        """Test fallback to OPENAI_API_KEY for unknown models."""
        os.environ['OPENAI_API_KEY'] = 'fallback-key'
        api_key = get_api_key_for_model("unknown/model")
        self.assertEqual(api_key, 'fallback-key')

    def test_get_api_key_for_model_no_key(self):
        """Test when no API key is available."""
        api_key = get_api_key_for_model("openai/gpt-4o")
        self.assertIsNone(api_key)

    def test_get_required_api_key_for_model_with_provided_key(self):
        """Test that provided API key works with required function."""
        api_key = get_required_api_key_for_model("openai/gpt-4o", "test-key")
        self.assertEqual(api_key, "test-key")

    def test_get_required_api_key_for_model_success(self):
        """Test successful retrieval of required API key."""
        os.environ['OPENAI_API_KEY'] = 'test-openai-key'
        api_key = get_required_api_key_for_model("openai/gpt-4o")
        self.assertEqual(api_key, 'test-openai-key')

    def test_get_required_api_key_for_model_missing(self):
        """Test error when required API key is missing."""
        with self.assertRaises(ValueError) as context:
            get_required_api_key_for_model("openai/gpt-4o")

        self.assertIn("No API key provided for model 'openai/gpt-4o'", str(context.exception))
        self.assertIn("OPENAI_API_KEY", str(context.exception))

    def test_get_provider_name(self):
        """Test getting provider name from model ID."""
        self.assertEqual(get_provider_name("openai/gpt-4o"), "openai")
        self.assertEqual(get_provider_name("anthropic/claude-3-opus"), "anthropic")
        self.assertEqual(get_provider_name("claude-3-sonnet"), "claude")
        self.assertEqual(get_provider_name("unknown/model"), "unknown")

if __name__ == '__main__':
    unittest.main()

