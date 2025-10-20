"""
Integration tests to verify that the security fixes work with the actual model classes.
"""
import unittest
import sys
import os
import logging
import io
from unittest.mock import patch, MagicMock

# Add workspace to path
sys.path.append('/workspace')


class TestModelSecurityIntegration(unittest.TestCase):
    """Test that model classes properly use secure logging."""

    def setUp(self):
        """Set up test fixtures."""
        self.log_buffer = io.StringIO()
        self.handler = logging.StreamHandler(self.log_buffer)
        self.logger = logging.getLogger()
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

    def tearDown(self):
        """Clean up after tests."""
        self.logger.removeHandler(self.handler)
        self.handler.close()

    def test_debate_setting_models_import(self):
        """Test that debate setting models can import security utilities."""
        try:
            from debate_setting.models import SecureLogger, sanitize_prompt_for_logging
            # Test that the functions work
            test_message = "API key sk-1234567890abcdef1234567890abcdef1234567890abcdef12"
            sanitized = SecureLogger.sanitize_text(test_message)
            self.assertNotIn("sk-", sanitized)
            self.assertIn("[REDACTED]", sanitized)
        except ImportError as e:
            self.fail(f"Failed to import from debate_setting.models: {e}")

    def test_ethical_setting_models_import(self):
        """Test that ethical setting models can import security utilities."""
        try:
            from ethical_setting.models import SecureLogger, sanitize_prompt_for_logging
            # Test that the functions work
            test_message = "API key sk-1234567890abcdef1234567890abcdef1234567890abcdef12"
            sanitized = SecureLogger.sanitize_text(test_message)
            self.assertNotIn("sk-", sanitized)
            self.assertIn("[REDACTED]", sanitized)
        except ImportError as e:
            self.fail(f"Failed to import from ethical_setting.models: {e}")

    def test_false_presuppositions_setting_models_import(self):
        """Test that false presuppositions setting models can import security utilities."""
        try:
            from false_presuppositions_setting.models import SecureLogger, sanitize_prompt_for_logging
            # Test that the functions work
            test_message = "API key sk-1234567890abcdef1234567890abcdef1234567890abcdef12"
            sanitized = SecureLogger.sanitize_text(test_message)
            self.assertNotIn("sk-", sanitized)
            self.assertIn("[REDACTED]", sanitized)
        except ImportError as e:
            self.fail(f"Failed to import from false_presuppositions_setting.models: {e}")

    def test_evaluate_oscillate_import(self):
        """Test that evaluate_oscillate can import security utilities."""
        try:
            from false_presuppositions_setting.evaluate_oscillate import SecureLogger
            # Test that the functions work
            test_message = "API key sk-1234567890abcdef1234567890abcdef1234567890abcdef12"
            sanitized = SecureLogger.sanitize_text(test_message)
            self.assertNotIn("sk-", sanitized)
            self.assertIn("[REDACTED]", sanitized)
        except ImportError as e:
            self.fail(f"Failed to import from evaluate_oscillate: {e}")

    @patch('debate_setting.models.completion')
    def test_closed_model_error_handling(self, mock_completion):
        """Test that ClosedModel properly handles and sanitizes errors."""
        # Mock an API error that might contain sensitive information
        mock_completion.side_effect = Exception("Authentication failed with API key sk-1234567890abcdef1234567890abcdef1234567890abcdef12")

        try:
            from debate_setting.models import ClosedModel

            # Create a model instance
            model = ClosedModel(api_key="fake_key")
            model.setup()

            # Try to generate responses (should handle the error securely)
            messages = [{"role": "user", "content": "test"}]
            responses = model.generate_responses(messages, num_responses=1)

            # Check that the response indicates an error without exposing the API key
            self.assertEqual(len(responses), 1)
            self.assertIn("ERROR", responses[0])
            self.assertNotIn("sk-", responses[0])

            # Check that the logged error doesn't contain the API key
            log_output = self.log_buffer.getvalue()
            self.assertNotIn("sk-", log_output)

        except ImportError as e:
            self.fail(f"Failed to import ClosedModel: {e}")

    def test_prompt_logging_security(self):
        """Test that prompt logging is secure."""
        from security_utils import sanitize_prompt_for_logging

        # Test with a prompt that might contain sensitive information
        sensitive_prompt = """
        System: You are a helpful assistant.
        User: Please process this request using my API key sk-1234567890abcdef1234567890abcdef1234567890abcdef12
        """

        sanitized = sanitize_prompt_for_logging(sensitive_prompt, max_length=100)

        # Verify the API key is not in the sanitized prompt
        self.assertNotIn("sk-", sanitized)
        # Verify it was actually sanitized (contains REDACTED or is truncated)
        self.assertTrue("[REDACTED]" in sanitized or "[REDACTE" in sanitized or len(sanitized) <= 103)


if __name__ == '__main__':
    unittest.main(verbosity=2)
