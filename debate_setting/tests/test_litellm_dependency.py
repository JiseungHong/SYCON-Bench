"""
Tests for litellm dependency handling in models.py
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

def test_closed_model_setup_without_litellm():
    """Test that ClosedModel.setup() raises ImportError when litellm is not available"""
    # Add the debate_setting directory to Python path
    sys.path.insert(0, '/workspace/debate_setting')

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
        sys.path.remove('/workspace/debate_setting')


def test_closed_model_setup_with_litellm():
    """Test that ClosedModel.setup() works when litellm is available"""
    sys.path.insert(0, '/workspace/debate_setting')

    try:
        with patch('models.LITELLM_AVAILABLE', True):
            from models import ClosedModel

            model = ClosedModel(api_key="test_key")
            result = model.setup()
            assert result is True
    finally:
        sys.path.remove('/workspace/debate_setting')


def test_closed_model_setup_without_api_key():
    """Test that ClosedModel.setup() raises ValueError when no API key is provided"""
    sys.path.insert(0, '/workspace/debate_setting')

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
        sys.path.remove('/workspace/debate_setting')


def test_closed_model_generate_responses_without_litellm():
    """Test that generate_responses raises ImportError when litellm is not available"""
    sys.path.insert(0, '/workspace/debate_setting')

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
        sys.path.remove('/workspace/debate_setting')


def test_closed_model_generate_responses_with_litellm():
    """Test that generate_responses works when litellm is available"""
    sys.path.insert(0, '/workspace/debate_setting')

    try:
        # Mock the completion function
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5

        mock_completion = MagicMock(return_value=mock_response)

        with patch('models.LITELLM_AVAILABLE', True):
            with patch('models.completion', mock_completion):
                from models import ClosedModel

                model = ClosedModel(api_key="test_key")
                messages = [{"role": "user", "content": "test"}]

                responses = model.generate_responses(messages, num_responses=1)

                assert len(responses) == 1
                assert responses[0] == "Test response"
                mock_completion.assert_called_once()
    finally:
        sys.path.remove('/workspace/debate_setting')


def test_open_model_not_affected():
    """Test that OpenModel is not affected by litellm availability"""
    sys.path.insert(0, '/workspace/debate_setting')

    try:
        with patch('models.LITELLM_AVAILABLE', False):
            from models import OpenModel

            # OpenModel should work regardless of litellm availability
            model = OpenModel("test-model")
            assert model.model_name == "test-model"
            # We don't test setup() here as it requires actual model loading
    finally:
        sys.path.remove('/workspace/debate_setting')
