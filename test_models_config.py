

#!/usr/bin/env python3
"""
Test script for model configuration system.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add the workspace directory to the path so we can import models
sys.path.insert(0, '/workspace')

class TestModelsConfig(unittest.TestCase):

    def setUp(self):
        # Clear environment variables before each test
        for var in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']:
            if var in os.environ:
                del os.environ[var]

    def test_debate_setting_model_with_openai_key(self):
        """Test debate setting model with OpenAI key."""
        os.environ['OPENAI_API_KEY'] = 'test-openai-key'
        from debate_setting.models import ClosedModel
        model = ClosedModel("openai/gpt-4o")
        self.assertEqual(model.api_key, 'test-openai-key')

    def test_debate_setting_model_with_anthropic_key(self):
        """Test debate setting model with Anthropic key."""
        os.environ['ANTHROPIC_API_KEY'] = 'test-anthropic-key'
        from debate_setting.models import ClosedModel
        model = ClosedModel("anthropic/claude-3-opus")
        self.assertEqual(model.api_key, 'test-anthropic-key')

    def test_debate_setting_model_with_provided_key(self):
        """Test debate setting model with explicitly provided key."""
        from debate_setting.models import ClosedModel
        model = ClosedModel("openai/gpt-4o", api_key="explicit-key")
        self.assertEqual(model.api_key, 'explicit-key')

    def test_debate_setting_model_missing_key(self):
        """Test debate setting model with missing key raises error."""
        from debate_setting.models import ClosedModel
        with self.assertRaises(ValueError) as context:
            ClosedModel("openai/gpt-4o")
        self.assertIn("No API key provided for model 'openai/gpt-4o'", str(context.exception))

    def test_ethical_setting_model_with_openai_key(self):
        """Test ethical setting model with OpenAI key."""
        os.environ['OPENAI_API_KEY'] = 'test-openai-key'
        from ethical_setting.models import ClosedModel
        model = ClosedModel("openai/gpt-4o")
        self.assertEqual(model.api_key, 'test-openai-key')

    def test_ethical_setting_model_with_anthropic_key(self):
        """Test ethical setting model with Anthropic key."""
        os.environ['ANTHROPIC_API_KEY'] = 'test-anthropic-key'
        from ethical_setting.models import ClosedModel
        model = ClosedModel("anthropic/claude-3-opus")
        self.assertEqual(model.api_key, 'test-anthropic-key')

    def test_false_presuppositions_setting_model_with_openai_key(self):
        """Test false presuppositions setting model with OpenAI key."""
        os.environ['OPENAI_API_KEY'] = 'test-openai-key'
        from false_presuppositions_setting.models import ClosedModel
        model = ClosedModel("openai/gpt-4o")
        self.assertEqual(model.api_key, 'test-openai-key')

    def test_false_presuppositions_setting_model_with_anthropic_key(self):
        """Test false presuppositions setting model with Anthropic key."""
        os.environ['ANTHROPIC_API_KEY'] = 'test-anthropic-key'
        from false_presuppositions_setting.models import ClosedModel
        model = ClosedModel("anthropic/claude-3-opus")
        self.assertEqual(model.api_key, 'test-anthropic-key')

if __name__ == '__main__':
    unittest.main()


