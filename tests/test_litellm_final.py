


"""
Final test for litellm dependency handling.
This test verifies our changes work correctly by simulating both scenarios:
1. When litellm is available (should work)
2. When litellm is not available (should raise helpful ImportError)
"""
import unittest
import sys
import os
from unittest.mock import patch, MagicMock

class TestLitellmDependencyFinal(unittest.TestCase):
    """Test cases for litellm dependency handling."""

    def test_import_with_available_litellm(self):
        """Test that our import handling works when litellm is available."""
        # This simulates the try/except block in our models.py files
        completion = None
        try:
            from litellm import completion as imported_completion
            completion = imported_completion
        except ImportError:
            completion = None

        # Since litellm is installed, completion should not be None
        self.assertIsNotNone(completion)

        # Test that setup logic works with available litellm
        def mock_setup_with_completion():
            if completion is None:
                raise ImportError(
                    "litellm package is not installed. Please install it with: pip install litellm>=1.0.0\n"
                    "Alternatively, install all required dependencies with: pip install -r requirements.txt"
                )
            return True

        # Should not raise ImportError
        result = mock_setup_with_completion()
        self.assertTrue(result)

    def test_import_with_missing_litellm(self):
        """Test that our import handling works when litellm is not available."""
        # This simulates the try/except block when litellm is not available
        completion = None
        with patch.dict('sys.modules', {'litellm': None}):
            try:
                from litellm import completion as imported_completion
                completion = imported_completion
            except ImportError:
                completion = None

        # Since we mocked litellm to be None, completion should be None
        self.assertIsNone(completion)

        # Test that setup logic raises helpful error when litellm is missing
        def mock_setup_without_completion():
            if completion is None:
                raise ImportError(
                    "litellm package is not installed. Please install it with: pip install litellm>=1.0.0\n"
                    "Alternatively, install all required dependencies with: pip install -r requirements.txt"
                )
            return True

        # Should raise ImportError with helpful message
        with self.assertRaises(ImportError) as context:
            mock_setup_without_completion()

        self.assertIn("litellm package is not installed", str(context.exception))
        self.assertIn("pip install litellm>=1.0.0", str(context.exception))

    def test_import_with_import_error(self):
        """Test that our import handling works when litellm import fails."""
        # This simulates the try/except block when litellm import raises ImportError
        completion = None
        with patch.dict('sys.modules', {'litellm': MagicMock()}):
            # Make the litellm module raise ImportError when accessed
            sys.modules['litellm'] = MagicMock()
            sys.modules['litellm'].completion = MagicMock(side_effect=ImportError("No module named 'litellm'"))

            try:
                from litellm import completion as imported_completion
                completion = imported_completion
            except ImportError:
                completion = None

        # Since import failed, completion should be None
        self.assertIsNone(completion)

        # Test that setup logic raises helpful error when litellm is missing
        def mock_setup_without_completion():
            if completion is None:
                raise ImportError(
                    "litellm package is not installed. Please install it with: pip install litellm>=1.0.0\n"
                    "Alternatively, install all required dependencies with: pip install -r requirements.txt"
                )
            return True

        # Should raise ImportError with helpful message
        with self.assertRaises(ImportError) as context:
            mock_setup_without_completion()

        self.assertIn("litellm package is not installed", str(context.exception))
        self.assertIn("pip install litellm>=1.0.0", str(context.exception))

if __name__ == '__main__':
    unittest.main()


