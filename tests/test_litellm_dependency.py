
"""
Test for litellm dependency handling in both debate-setting and ethical-setting models.
"""
import unittest
import sys
import os
from unittest.mock import patch

# Add the settings directories to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'debate-setting'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ethical-setting'))

class TestLitellmDependency(unittest.TestCase):
    """Test cases for litellm dependency handling."""

    def test_debate_setting_closed_model_setup_with_missing_litellm(self):
        """Test that ClosedModel.setup() raises ImportError when litellm is not available in debate setting."""
        # Mock the completion import to be None (simulating missing litellm)
        with patch('models.completion', None):
            from models import ClosedModel

            model = ClosedModel(model_id="openai/gpt-4o", api_key="test-key")

            # Should raise ImportError with helpful message
            with self.assertRaises(ImportError) as context:
                model.setup()

            self.assertIn("litellm package is not installed", str(context.exception))
            self.assertIn("pip install litellm>=1.0.0", str(context.exception))

    def test_ethical_setting_closed_model_setup_with_missing_litellm(self):
        """Test that ClosedModel.setup() raises ImportError when litellm is not available in ethical setting."""
        # Add ethical-setting to path and import
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ethical-setting'))

        # Mock the completion import to be None (simulating missing litellm)
        with patch('models.completion', None):
            from models import ClosedModel

            model = ClosedModel(model_id="openai/gpt-4o", api_key="test-key")

            # Should raise ImportError with helpful message
            with self.assertRaises(ImportError) as context:
                model.setup()

            self.assertIn("litellm package is not installed", str(context.exception))
            self.assertIn("pip install litellm>=1.0.0", str(context.exception))

    def test_debate_setting_closed_model_setup_with_valid_litellm(self):
        """Test that ClosedModel.setup() works when litellm is available in debate setting."""
        # Mock the completion import to be available
        mock_completion = object()
        with patch('models.completion', mock_completion):
            from models import ClosedModel

            model = ClosedModel(model_id="openai/gpt-4o", api_key="test-key")

            # Should not raise ImportError
            result = model.setup()
            self.assertTrue(result)

    def test_ethical_setting_closed_model_setup_with_valid_litellm(self):
        """Test that ClosedModel.setup() works when litellm is available in ethical setting."""
        # Add ethical-setting to path and import
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ethical-setting'))

        # Mock the completion import to be available
        mock_completion = object()
        with patch('models.completion', mock_completion):
            from models import ClosedModel

            model = ClosedModel(model_id="openai/gpt-4o", api_key="test-key")

            # Should not raise ImportError
            result = model.setup()
            self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
