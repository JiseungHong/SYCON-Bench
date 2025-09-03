"""
Tests for litellm dependency handling in ethical-setting models.py
"""
import pytest
import sys
from unittest.mock import patch, MagicMock
import os

# Test the import behavior
def test_litellm_import_available():
    """Test that when litellm is available, LITELLM_AVAILABLE is True"""
    with patch.dict('sys.modules', {'litellm': MagicMock()}):
        # Force reimport to test the import logic
        if 'ethical-setting.models' in sys.modules:
            del sys.modules['ethical-setting.models']

        # Import using the module path
        sys.path.insert(0, '/workspace/ethical-setting')
        try:
            from models import LITELLM_AVAILABLE, completion
            assert LITELLM_AVAILABLE is True
            assert completion is not None
        finally:
            sys.path.remove('/workspace/ethical-setting')


def test_litellm_import_unavailable():
    """Test that when litellm is not available, LITELLM_AVAILABLE is False"""
    # Mock the import to raise ImportError
    with patch.dict('sys.modules', {'litellm': None}):
        with patch('builtins.__import__', side_effect=ImportError("No module named 'litellm'")):
            # Force reimport to test the import logic
            if 'models' in sys.modules:
                del sys.modules['models']

            sys.path.insert(0, '/workspace/ethical-setting')
            try:
                from models import LITELLM_AVAILABLE, completion
                assert LITELLM_AVAILABLE is False
                assert completion is None
            finally:
                sys.path.remove('/workspace/ethical-setting')


def test_closed_model_setup_without_litellm():
    """Test that ClosedModel.setup() raises ImportError when litellm is not available"""
    sys.path.insert(0, '/workspace/ethical-setting')
    try:
        # Mock LITELLM_AVAILABLE to be False
        with patch('models.LITELLM_AVAILABLE', False):
            from models import ClosedModel

            model = ClosedModel(api_key="test_key")

            with pytest.raises(ImportError) as exc_info:
                model.setup()

            assert "litellm is required for closed-source models" in str(exc_info.value)
            assert "pip install litellm>=1.0.0" in str(exc_info.value)
    finally:
        sys.path.remove('/workspace/ethical-setting')


def test_closed_model_setup_with_litellm():
    """Test that ClosedModel.setup() works when litellm is available"""
    sys.path.insert(0, '/workspace/ethical-setting')
    try:
        with patch('models.LITELLM_AVAILABLE', True):
            from models import ClosedModel

            model = ClosedModel(api_key="test_key")
            result = model.setup()
            assert result is True
    finally:
        sys.path.remove('/workspace/ethical-setting')


def test_closed_model_setup_without_api_key():
    """Test that ClosedModel.setup() raises ValueError when no API key is provided"""
    sys.path.insert(0, '/workspace/ethical-setting')
    try:
        with patch('models.LITELLM_AVAILABLE', True):
            from models import ClosedModel

            # Test without api_key and without environment variable
            with patch.dict(os.environ, {}, clear=True):
                model = ClosedModel()

                with pytest.raises(ValueError) as exc_info:
                    model.setup()

                assert "No API key provided" in str(exc_info.value)
    finally:
        sys.path.remove('/workspace/ethical-setting')


def test_closed_model_generate_responses_without_litellm():
    """Test that generate_responses raises ImportError when litellm is not available"""
    sys.path.insert(0, '/workspace/ethical-setting')
    try:
        with patch('models.LITELLM_AVAILABLE', False):
            from models import ClosedModel

            model = ClosedModel(api_key="test_key")
            messages = [{"role": "user", "content": "test"}]

            with pytest.raises(ImportError) as exc_info:
                model.generate_responses(messages)

            assert "litellm is required for closed-source models" in str(exc_info.value)
            assert "pip install litellm>=1.0.0" in str(exc_info.value)
    finally:
        sys.path.remove('/workspace/ethical-setting')


def test_closed_model_generate_responses_with_litellm():
    """Test that generate_responses works when litellm is available"""
    # Mock the completion function
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5

    mock_completion = MagicMock(return_value=mock_response)

    sys.path.insert(0, '/workspace/ethical-setting')
    try:
        with patch('models.LITELLM_AVAILABLE', True):
            with patch('models.completion', mock_completion):
                from models import ClosedModel

                model = ClosedModel(api_key="test_key")
                messages = [{"role": "user", "content": "test"}]

                response = model.generate_responses(messages, num_responses=1)

                assert response == "Test response"
                mock_completion.assert_called_once()
    finally:
        sys.path.remove('/workspace/ethical-setting')


def test_open_model_not_affected():
    """Test that OpenModel is not affected by litellm availability"""
    sys.path.insert(0, '/workspace/ethical-setting')
    try:
        with patch('models.LITELLM_AVAILABLE', False):
            from models import OpenModel

            # OpenModel should work regardless of litellm availability
            model = OpenModel("test-model")
            assert model.model_name == "test-model"
            # We don't test setup() here as it requires actual model loading
    finally:
        sys.path.remove('/workspace/ethical-setting')
