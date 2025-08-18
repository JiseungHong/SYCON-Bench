"""
Tests for litellm dependency handling in SYCON-Bench models.

This test module verifies that proper error handling is in place when litellm
is not available, and that closed-source models fail gracefully with clear
error messages.
"""
import unittest
import tempfile
import os
import sys


class TestLitellmDependency(unittest.TestCase):
    """Test litellm dependency handling across all settings."""

    def test_litellm_import_handling(self):
        """Test that the import handling code works correctly."""
        # Create a temporary test module to simulate the import behavior
        test_code_with_litellm = '''
try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    completion = None

class TestClosedModel:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def setup(self):
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is required for closed-source models but is not installed. "
                "Please install it with: pip install litellm>=1.0.0"
            )
        if self.api_key is None:
            raise ValueError("No API key provided. Please provide via api_key parameter or set OPENAI_API_KEY environment variable.")
        return True
'''

        # Write the test code to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code_with_litellm)
            temp_module_path = f.name

        try:
            # Load the module
            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", temp_module_path)
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)

            # Test when litellm is not available (which should be the case in test environment)
            if not test_module.LITELLM_AVAILABLE:
                # Test that ClosedModel setup fails with proper error
                closed_model = test_module.TestClosedModel(api_key="test_key")
                with self.assertRaises(ImportError) as context:
                    closed_model.setup()

                error_message = str(context.exception)
                self.assertIn("litellm is required", error_message)
                self.assertIn("pip install litellm>=1.0.0", error_message)

                # Test that completion is None when litellm is not available
                self.assertIsNone(test_module.completion)

            # Test API key validation still works
            closed_model_no_key = test_module.TestClosedModel()  # No API key
            if test_module.LITELLM_AVAILABLE:
                # If litellm is available, should fail on API key validation
                with self.assertRaises(ValueError) as context:
                    closed_model_no_key.setup()
                error_message = str(context.exception)
                self.assertIn("No API key provided", error_message)
            else:
                # If litellm is not available, should fail on litellm check first
                with self.assertRaises(ImportError) as context:
                    closed_model_no_key.setup()
                error_message = str(context.exception)
                self.assertIn("litellm is required", error_message)

        finally:
            # Clean up the temporary file
            os.unlink(temp_module_path)

    def test_requirements_includes_litellm(self):
        """Test that litellm is included in requirements.txt."""
        requirements_path = '/workspace/requirements.txt'
        self.assertTrue(os.path.exists(requirements_path), "requirements.txt should exist")

        with open(requirements_path, 'r') as f:
            requirements_content = f.read()

        self.assertIn('litellm>=1.0.0', requirements_content,
                     "litellm>=1.0.0 should be in requirements.txt")

    def test_models_files_have_proper_import_handling(self):
        """Test that all models.py files have proper import handling."""
        models_files = [
            '/workspace/debate-setting/models.py',
            '/workspace/ethical-setting/models.py',
            '/workspace/false-presuppositions-setting/models.py'
        ]

        for models_file in models_files:
            with self.subTest(models_file=models_file):
                self.assertTrue(os.path.exists(models_file), f"{models_file} should exist")

                with open(models_file, 'r') as f:
                    content = f.read()

                # Check that the import handling is present
                self.assertIn('LITELLM_AVAILABLE = True', content,
                             f"{models_file} should set LITELLM_AVAILABLE = True on successful import")
                self.assertIn('LITELLM_AVAILABLE = False', content,
                             f"{models_file} should set LITELLM_AVAILABLE = False on ImportError")
                self.assertIn('completion = None', content,
                             f"{models_file} should set completion = None on ImportError")

                # Check that setup method validates litellm availability
                self.assertIn('if not LITELLM_AVAILABLE:', content,
                             f"{models_file} should check LITELLM_AVAILABLE in setup method")
                self.assertIn('litellm is required for closed-source models', content,
                             f"{models_file} should have proper error message")
                self.assertIn('pip install litellm>=1.0.0', content,
                             f"{models_file} should provide installation instructions")


if __name__ == '__main__':
    unittest.main()
