"""
Unit tests for the base model classes.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import torch

from model_registry.base_models import RegistryOpenModel, RegistryClosedModel, create_model
from model_registry.registry import ModelFamily, QuantizationStrategy


class TestRegistryOpenModel(unittest.TestCase):
    """Test cases for the RegistryOpenModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = RegistryOpenModel("meta-llama/Llama-2-7b-chat-hf")

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_name, "meta-llama/Llama-2-7b-chat-hf")
        self.assertEqual(self.model.config.family, ModelFamily.LLAMA)
        self.assertEqual(self.model.config.size_category, "small")
        self.assertIsNone(self.model.model)
        self.assertIsNone(self.model.tokenizer)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_setup_llama_model(self, mock_tokenizer, mock_model):
        """Test setup for Llama model."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        # Mock model
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance

        # Setup model
        model, tokenizer = self.model.setup()

        # Verify tokenizer was loaded with correct kwargs
        mock_tokenizer.assert_called_once_with(
            "meta-llama/Llama-2-7b-chat-hf",
            padding_side="right",
            add_eos_token=True
        )

        # Verify pad token was set
        self.assertEqual(mock_tokenizer_instance.pad_token, "<eos>")

        # Verify model was loaded with correct kwargs
        expected_model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16
        }
        mock_model.assert_called_once_with("meta-llama/Llama-2-7b-chat-hf", **expected_model_kwargs)

        # Verify return values
        self.assertEqual(model, mock_model_instance)
        self.assertEqual(tokenizer, mock_tokenizer_instance)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_setup_gemma_model(self, mock_tokenizer, mock_model):
        """Test setup for Gemma model."""
        gemma_model = RegistryOpenModel("google/gemma-2b-it")

        # Mock tokenizer and model
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()

        # Setup model
        gemma_model.setup()

        # Verify tokenizer was loaded with trust_remote_code
        mock_tokenizer.assert_called_once_with(
            "google/gemma-2b-it",
            trust_remote_code=True
        )

        # Verify model was loaded with trust_remote_code and torch_dtype
        expected_model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "torch_dtype": torch.float16
        }
        mock_model.assert_called_once_with("google/gemma-2b-it", **expected_model_kwargs)

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_setup_large_model_quantization(self, mock_tokenizer, mock_model):
        """Test setup for large model with 8-bit quantization."""
        large_model = RegistryOpenModel("meta-llama/Llama-2-70b-chat-hf")

        # Mock tokenizer and model
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_model.return_value = Mock()

        # Setup model
        large_model.setup()

        # Verify model was loaded with 4-bit quantization (xlarge model)
        expected_model_kwargs = {
            "device_map": "auto",
            "padding_side": "right",
            "add_eos_token": True,
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": torch.float16
        }
        mock_model.assert_called_once_with("meta-llama/Llama-2-70b-chat-hf", **expected_model_kwargs)

    def test_format_messages_fallback(self):
        """Test fallback message formatting."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        formatted = self.model.format_messages_fallback(messages)

        self.assertIn("<s>\nYou are a helpful assistant.\n</s>", formatted)
        self.assertIn("<user>\nHello!\n</user>", formatted)
        self.assertIn("<assistant>\nHi there!\n</assistant>", formatted)

    def test_apply_template_gemma(self):
        """Test chat template application for Gemma models."""
        gemma_model = RegistryOpenModel("google/gemma-2b-it")

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]

        # Mock tokenizer with apply_chat_template
        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_prompt"

        result = gemma_model.apply_template(messages, mock_tokenizer, "google/gemma-2b-it")

        # Verify that apply_chat_template was called with transformed messages
        mock_tokenizer.apply_chat_template.assert_called_once()
        call_args = mock_tokenizer.apply_chat_template.call_args[0][0]

        # Should have combined system and user messages
        self.assertEqual(len(call_args), 1)
        self.assertEqual(call_args[0]["role"], "user")
        self.assertIn("You are a helpful assistant.", call_args[0]["content"])
        self.assertIn("Hello!", call_args[0]["content"])

    @patch('transformers.pipeline')
    def test_generate_responses(self, mock_pipeline):
        """Test response generation."""
        # Mock pipeline
        mock_generator = Mock()
        mock_generator.return_value = [{"generated_text": "Test response"}]
        mock_pipeline.return_value = mock_generator

        # Mock tokenizer and model
        self.model.tokenizer = Mock()
        self.model.tokenizer.eos_token_id = 2
        self.model.tokenizer.apply_chat_template.return_value = "formatted_prompt"
        self.model.model = Mock()

        messages = [{"role": "user", "content": "Test question"}]

        responses = self.model.generate_responses(messages, num_responses=2)

        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0], "Test response")
        self.assertEqual(responses[1], "Test response")

    def test_get_chat_messages(self):
        """Test chat message generation for different prompt types."""
        question = "What do you think about AI?"
        argument = "AI is beneficial"

        # Test individual_thinker prompt
        messages = self.model.get_chat_messages(question, argument, "individual_thinker")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("independent thinker", messages[0]["content"])
        self.assertIn(argument, messages[0]["content"])

        # Test spt prompt
        messages = self.model.get_chat_messages(question, argument, "spt")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Andrew", messages[0]["content"])

        # Test invalid prompt type
        with self.assertRaises(ValueError):
            self.model.get_chat_messages(question, argument, "invalid_type")


class TestRegistryClosedModel(unittest.TestCase):
    """Test cases for the RegistryClosedModel class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = RegistryClosedModel("openai/gpt-4o", api_key="test_key")

    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.model_name, "openai/gpt-4o")
        self.assertEqual(self.model.api_key, "test_key")
        self.assertEqual(self.model.config.family, ModelFamily.GPT)
        self.assertTrue(self.model.config.api_based)

    def test_setup(self):
        """Test model setup."""
        result = self.model.setup()
        self.assertTrue(result)

    def test_setup_no_api_key(self):
        """Test setup without API key."""
        model_no_key = RegistryClosedModel("openai/gpt-4o")

        with self.assertRaises(ValueError):
            model_no_key.setup()

    def test_estimate_cost(self):
        """Test cost estimation."""
        count = {"input": 1000, "output": 500}
        cost = self.model.estimate_cost(count)

        # Should calculate cost based on registry pricing
        expected_cost = (1000 * 5 / 1e6) + (500 * 15 / 1e6)
        self.assertEqual(cost, expected_cost)

    @patch('model_registry.base_models.completion')
    def test_generate_responses(self, mock_completion):
        """Test response generation via API."""
        # Mock API response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test API response"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_completion.return_value = mock_response

        messages = [{"role": "user", "content": "Test question"}]

        responses = self.model.generate_responses(messages, num_responses=2)

        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0], "Test API response")
        self.assertEqual(responses[1], "Test API response")

        # Verify API was called correctly
        self.assertEqual(mock_completion.call_count, 2)


class TestCreateModel(unittest.TestCase):
    """Test cases for the create_model function."""

    def test_create_open_model(self):
        """Test creation of open-source model."""
        model = create_model("meta-llama/Llama-2-7b-chat-hf")
        self.assertIsInstance(model, RegistryOpenModel)
        self.assertEqual(model.model_name, "meta-llama/Llama-2-7b-chat-hf")

    def test_create_closed_model_openai(self):
        """Test creation of OpenAI model."""
        model = create_model("openai/gpt-4o", api_key="test_key")
        self.assertIsInstance(model, RegistryClosedModel)
        self.assertEqual(model.model_name, "openai/gpt-4o")
        self.assertEqual(model.api_key, "test_key")

    def test_create_closed_model_anthropic(self):
        """Test creation of Anthropic model."""
        model = create_model("anthropic/claude-3-sonnet-20240229", api_key="test_key")
        self.assertIsInstance(model, RegistryClosedModel)
        self.assertEqual(model.model_name, "anthropic/claude-3-sonnet-20240229")

    def test_create_model_with_registry_detection(self):
        """Test model creation using registry detection."""
        # Test a model that should be detected as API-based by registry
        model = create_model("claude-3-opus", api_key="test_key")
        self.assertIsInstance(model, RegistryClosedModel)


if __name__ == '__main__':
    unittest.main()
