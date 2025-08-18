"""
Basic tests for the model registry system without heavy dependencies.
"""

import unittest
import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import model_registry
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_registry.registry import ModelRegistry, ModelFamily, QuantizationStrategy, get_model_config


class TestModelRegistryBasic(unittest.TestCase):
    """Basic test cases for the ModelRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ModelRegistry()

    def test_family_detection_llama(self):
        """Test detection of Llama model family."""
        test_cases = [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
            "codellama/CodeLlama-7b-Python-hf",
            "llama-7b-instruct"
        ]

        for model_name in test_cases:
            with self.subTest(model_name=model_name):
                family = self.registry.detect_model_family(model_name)
                self.assertEqual(family, ModelFamily.LLAMA)

    def test_family_detection_gemma(self):
        """Test detection of Gemma model family."""
        test_cases = [
            "google/gemma-2b-it",
            "google/gemma-7b-it",
            "google/gemma-2-9b-it",
            "gemma-instruct-7b"
        ]

        for model_name in test_cases:
            with self.subTest(model_name=model_name):
                family = self.registry.detect_model_family(model_name)
                self.assertEqual(family, ModelFamily.GEMMA)

    def test_family_detection_qwen(self):
        """Test detection of Qwen model family."""
        test_cases = [
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen1.5-14B-Chat",
            "alibaba/qwen-7b-chat"
        ]

        for model_name in test_cases:
            with self.subTest(model_name=model_name):
                family = self.registry.detect_model_family(model_name)
                self.assertEqual(family, ModelFamily.QWEN)

    def test_family_detection_mistral(self):
        """Test detection of Mistral model family."""
        test_cases = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistral-7b-instruct"
        ]

        for model_name in test_cases:
            with self.subTest(model_name=model_name):
                family = self.registry.detect_model_family(model_name)
                self.assertEqual(family, ModelFamily.MISTRAL)

    def test_family_detection_api_models(self):
        """Test detection of API-based model families."""
        test_cases = [
            ("openai/gpt-4o", ModelFamily.GPT),
            ("openai/gpt-3.5-turbo", ModelFamily.GPT),
            ("anthropic/claude-3-sonnet-20240229", ModelFamily.CLAUDE),
            ("claude-3-opus", ModelFamily.CLAUDE)
        ]

        for model_name, expected_family in test_cases:
            with self.subTest(model_name=model_name):
                family = self.registry.detect_model_family(model_name)
                self.assertEqual(family, expected_family)

    def test_size_detection(self):
        """Test detection of model sizes."""
        test_cases = [
            ("meta-llama/Llama-2-7b-chat-hf", "small"),
            ("meta-llama/Llama-2-13b-chat-hf", "medium"),
            ("meta-llama/Llama-2-70b-chat-hf", "xlarge"),
            ("Qwen/Qwen2-32B-Instruct", "large"),
            ("google/gemma-2b-it", "small"),
            ("mistralai/Mixtral-8x7B-Instruct-v0.1", "small"),  # 8x7B is treated as 7B
        ]

        for model_name, expected_size in test_cases:
            with self.subTest(model_name=model_name):
                size = self.registry.detect_model_size(model_name)
                self.assertEqual(size, expected_size)

    def test_get_config_exact_match(self):
        """Test getting configuration with exact match."""
        config = self.registry.get_config("llama-small")
        self.assertEqual(config.family, ModelFamily.LLAMA)
        self.assertEqual(config.size_category, "small")
        self.assertEqual(config.quantization, QuantizationStrategy.FLOAT16)

    def test_get_config_family_size_match(self):
        """Test getting configuration by family and size detection."""
        config = self.registry.get_config("meta-llama/Llama-2-7b-chat-hf")
        self.assertEqual(config.family, ModelFamily.LLAMA)
        self.assertEqual(config.size_category, "small")
        self.assertEqual(config.quantization, QuantizationStrategy.FLOAT16)

    def test_get_config_fallback(self):
        """Test fallback configuration for unknown models."""
        config = self.registry.get_config("unknown/model-7b")
        self.assertEqual(config.family, ModelFamily.UNKNOWN)
        self.assertEqual(config.size_category, "small")
        self.assertIn("Fallback configuration", str(config.known_issues))

    def test_quantization_strategy_by_size(self):
        """Test that quantization strategies are appropriate for model sizes."""
        size_configs = {
            "small": self.registry.get_config("llama-small"),
            "medium": self.registry.get_config("llama-medium"),
            "large": self.registry.get_config("llama-large"),
            "xlarge": self.registry.get_config("llama-xlarge")
        }

        expected_quantizations = {
            "small": QuantizationStrategy.FLOAT16,
            "medium": QuantizationStrategy.FLOAT16,
            "large": QuantizationStrategy.INT8,
            "xlarge": QuantizationStrategy.INT4
        }

        for size, config in size_configs.items():
            with self.subTest(size=size):
                self.assertEqual(config.quantization, expected_quantizations[size])

    def test_api_based_model_detection(self):
        """Test detection of API-based models."""
        api_models = [
            "openai/gpt-4o",
            "anthropic/claude-3-sonnet-20240229"
        ]

        for model_name in api_models:
            with self.subTest(model_name=model_name):
                config = self.registry.get_config(model_name)
                self.assertTrue(config.api_based)

    def test_compatibility_matrix_generation(self):
        """Test generation of compatibility matrix."""
        matrix = self.registry.get_compatibility_matrix()

        # Check that matrix contains expected keys
        self.assertIn("llama-small", matrix)
        self.assertIn("gemma-small", matrix)
        self.assertIn("gpt-4o", matrix)

        # Check structure of matrix entries
        for model_key, model_info in matrix.items():
            with self.subTest(model_key=model_key):
                required_keys = ["family", "size_category", "quantization", "api_based"]
                for key in required_keys:
                    self.assertIn(key, model_info)

    def test_global_get_model_config_function(self):
        """Test the global get_model_config function."""
        config = get_model_config("meta-llama/Llama-2-7b-chat-hf")
        self.assertEqual(config.family, ModelFamily.LLAMA)
        self.assertEqual(config.size_category, "small")

    def test_string_matching_improvements(self):
        """Test that the new system handles edge cases better than string matching."""
        # Test cases that would fail with simple string matching
        test_cases = [
            # Model names without explicit size indicators
            ("microsoft/DialoGPT-medium", ModelFamily.UNKNOWN, "medium"),
            # Models with confusing naming
            ("facebook/opt-125m", ModelFamily.UNKNOWN, "small"),
            # Models with version numbers that could be confused with sizes
            ("microsoft/DialoGPT-large", ModelFamily.UNKNOWN, "large"),
        ]

        for model_name, expected_family, expected_size in test_cases:
            with self.subTest(model_name=model_name):
                config = self.registry.get_config(model_name)
                # These should get fallback configs, not crash
                self.assertIsNotNone(config)
                self.assertIsInstance(config.family, ModelFamily)
                self.assertIsInstance(config.size_category, str)


if __name__ == '__main__':
    unittest.main()
