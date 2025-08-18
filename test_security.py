#!/usr/bin/env python3
"""
Security tests for SYCON-Bench to verify no API keys are exposed in logs or error messages.

This test suite verifies that:
1. API keys are properly redacted in logging
2. Exception messages don't expose sensitive data
3. Print statements don't expose sensitive information
4. Debug logs are secure
"""

import unittest
import logging
import io
import sys
import os
from unittest.mock import patch, MagicMock
from contextlib import redirect_stdout, redirect_stderr

# Add the workspace to the path
sys.path.insert(0, '/workspace')

from secure_logging import SecureLogger


class TestSecureLogging(unittest.TestCase):
    """Test the secure logging utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_api_key = "sk-1234567890abcdef1234567890abcdef1234567890abcdef"
        self.test_token = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        self.test_password = "super_secret_password123"

    def test_sanitize_string_api_keys(self):
        """Test that API keys are properly sanitized in strings."""
        test_cases = [
            f"Using API key: {self.test_api_key}",
            f"Authorization: {self.test_token}",
            f"api_key={self.test_api_key}",
            f"token: {self.test_token}",
            f"password={self.test_password}",
        ]

        for test_string in test_cases:
            sanitized = SecureLogger.sanitize_string(test_string)
            self.assertNotIn(self.test_api_key, sanitized)
            self.assertNotIn(self.test_token, sanitized)
            self.assertNotIn(self.test_password, sanitized)
            self.assertIn("***", sanitized)

    def test_sanitize_dict_sensitive_keys(self):
        """Test that sensitive keys in dictionaries are properly sanitized."""
        test_dict = {
            'api_key': self.test_api_key,
            'token': self.test_token,
            'password': self.test_password,
            'model_name': 'gpt-4',
            'temperature': 0.7,
            'secret_key': 'another_secret',
            'normal_field': 'normal_value'
        }

        sanitized = SecureLogger.sanitize_dict(test_dict)

        # Sensitive fields should be redacted
        self.assertEqual(sanitized['api_key'], '***')
        self.assertEqual(sanitized['token'], '***')
        self.assertEqual(sanitized['password'], '***')
        self.assertEqual(sanitized['secret_key'], '***')

        # Normal fields should remain unchanged
        self.assertEqual(sanitized['model_name'], 'gpt-4')
        self.assertEqual(sanitized['temperature'], 0.7)
        self.assertEqual(sanitized['normal_field'], 'normal_value')

    def test_sanitize_exception_messages(self):
        """Test that exception messages are properly sanitized."""
        # Create exceptions with sensitive data
        exceptions = [
            Exception(f"API call failed with key {self.test_api_key}"),
            ValueError(f"Invalid token: {self.test_token}"),
            RuntimeError(f"Authentication failed for password {self.test_password}"),
        ]

        for exc in exceptions:
            sanitized = SecureLogger.sanitize_exception(exc)
            self.assertNotIn(self.test_api_key, sanitized)
            self.assertNotIn(self.test_token, sanitized)
            self.assertNotIn(self.test_password, sanitized)
            self.assertIn("***", sanitized)

    def test_safe_print_prompt(self):
        """Test that prompt printing doesn't expose sensitive data."""
        sensitive_prompt = f"System: Use API key {self.test_api_key} to authenticate. User: Hello"

        # Capture stdout
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            SecureLogger.safe_print_prompt(sensitive_prompt, max_length=200)

        output = captured_output.getvalue()
        self.assertNotIn(self.test_api_key, output)
        self.assertIn("***", output)

    def test_create_safe_error_response(self):
        """Test that error responses don't expose sensitive data."""
        exc = Exception(f"Connection failed with API key {self.test_api_key}")
        safe_response = SecureLogger.create_safe_error_response("API Error", exc)

        self.assertNotIn(self.test_api_key, safe_response)
        self.assertIn("***", safe_response)
        self.assertIn("API Error", safe_response)


class TestModelsSecurityIntegration(unittest.TestCase):
    """Integration tests for models.py security fixes."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_api_key = "sk-1234567890abcdef1234567890abcdef1234567890abcdef"

    def test_secure_logging_import_available(self):
        """Test that secure logging can be imported from models directories."""
        # Test that the secure_logging module is accessible from the models directories
        import os

        # Check that secure_logging.py exists in the workspace
        secure_logging_path = '/workspace/secure_logging.py'
        self.assertTrue(os.path.exists(secure_logging_path), "secure_logging.py should exist")

        # Test that we can import it
        try:
            from secure_logging import SecureLogger
            self.assertTrue(hasattr(SecureLogger, 'sanitize_string'))
            self.assertTrue(hasattr(SecureLogger, 'safe_log_args'))
        except ImportError as e:
            self.fail(f"Failed to import SecureLogger: {e}")

    def test_models_files_updated(self):
        """Test that models.py files contain secure logging imports."""
        models_files = [
            '/workspace/debate-setting/models.py',
            '/workspace/ethical-setting/models.py',
            '/workspace/false-presuppositions-setting/models.py'
        ]

        for models_file in models_files:
            with open(models_file, 'r') as f:
                content = f.read()
                self.assertIn('from secure_logging import SecureLogger', content,
                            f"{models_file} should import SecureLogger")
                self.assertIn('SecureLogger.safe_print_prompt', content,
                            f"{models_file} should use safe_print_prompt")
                self.assertIn('SecureLogger.safe_log_error', content,
                            f"{models_file} should use safe_log_error")

    def test_run_benchmark_files_updated(self):
        """Test that run_benchmark.py files use secure logging."""
        benchmark_files = [
            '/workspace/debate-setting/run_benchmark.py',
            '/workspace/ethical-setting/run_benchmark.py',
            '/workspace/false-presuppositions-setting/run_benchmark.py'
        ]

        for benchmark_file in benchmark_files:
            with open(benchmark_file, 'r') as f:
                content = f.read()
                self.assertIn('from secure_logging import SecureLogger', content,
                            f"{benchmark_file} should import SecureLogger")
                self.assertIn('SecureLogger.safe_log_args', content,
                            f"{benchmark_file} should use safe_log_args")


class TestLoggingSecurityWithVerbose(unittest.TestCase):
    """Test logging security when verbose mode is enabled."""

    def setUp(self):
        """Set up logging capture."""
        self.log_capture = io.StringIO()
        self.handler = logging.StreamHandler(self.log_capture)
        self.logger = logging.getLogger('test_logger')
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

        self.test_api_key = "sk-1234567890abcdef1234567890abcdef1234567890abcdef"

    def tearDown(self):
        """Clean up logging."""
        self.logger.removeHandler(self.handler)
        self.handler.close()

    def test_safe_log_args_with_api_key(self):
        """Test that safe_log_args properly redacts API keys."""
        # Mock args object
        class MockArgs:
            def __init__(self, test_api_key):
                self.model_name = "gpt-4"
                self.api_key = test_api_key
                self.temperature = 0.7
                self.verbose = True

        args = MockArgs(self.test_api_key)
        SecureLogger.safe_log_args(args, self.logger)

        log_output = self.log_capture.getvalue()
        self.assertNotIn(self.test_api_key, log_output)
        self.assertIn("***", log_output)
        self.assertIn("gpt-4", log_output)  # Non-sensitive data should remain

    def test_safe_log_error_with_sensitive_exception(self):
        """Test that safe_log_error properly sanitizes exceptions."""
        exc = Exception(f"Authentication failed with API key {self.test_api_key}")
        SecureLogger.safe_log_error("Test error", exc, self.logger)

        log_output = self.log_capture.getvalue()
        self.assertNotIn(self.test_api_key, log_output)
        self.assertIn("***", log_output)
        self.assertIn("Test error", log_output)


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world scenarios that could expose API keys."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_api_key = "sk-1234567890abcdef1234567890abcdef1234567890abcdef"

    def test_litellm_error_simulation(self):
        """Test handling of litellm-style errors that might contain API keys."""
        # Simulate a litellm error that might contain API key info
        error_msg = f"Request failed: Invalid API key '{self.test_api_key}' provided"
        exc = Exception(error_msg)

        safe_response = SecureLogger.create_safe_error_response("API call failed", exc)
        self.assertNotIn(self.test_api_key, safe_response)
        self.assertIn("***", safe_response)

    def test_openai_error_simulation(self):
        """Test handling of OpenAI-style errors that might contain API keys."""
        # Simulate an OpenAI error
        error_msg = f"Incorrect API key provided: {self.test_api_key}. You can find your API key at https://platform.openai.com/account/api-keys."
        exc = Exception(error_msg)

        sanitized = SecureLogger.sanitize_exception(exc)
        self.assertNotIn(self.test_api_key, sanitized)
        self.assertIn("***", sanitized)

    def test_debug_prompt_logging(self):
        """Test that debug prompt logging doesn't expose sensitive data."""
        # Simulate a prompt that might contain sensitive information
        prompt = f"System: You are a helpful assistant. Use the API key {self.test_api_key} for authentication.\nUser: Hello, how are you?"

        # Test safe prompt printing
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            SecureLogger.safe_print_prompt(prompt, max_length=150)

        output = captured_output.getvalue()
        self.assertNotIn(self.test_api_key, output)
        self.assertIn("***", output)


def run_security_tests():
    """Run all security tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSecureLogging))
    suite.addTests(loader.loadTestsFromTestCase(TestModelsSecurityIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestLoggingSecurityWithVerbose))
    suite.addTests(loader.loadTestsFromTestCase(TestRealWorldScenarios))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running SYCON-Bench Security Tests...")
    print("=" * 50)

    success = run_security_tests()

    print("\n" + "=" * 50)
    if success:
        print("✅ All security tests passed!")
        print("No API keys or sensitive data should be exposed in logs.")
    else:
        print("❌ Some security tests failed!")
        print("Please review the test output above.")
        sys.exit(1)
