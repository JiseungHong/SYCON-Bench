"""
Unit tests for the model registry system.
"""

import unittest
from unittest.mock import Mock, patch
import torch

from model_registry.registry import ModelRegistry, ModelFamily, QuantizationStrategy, get_model_config
from model_registry.compatibility import ModelCompatibilityTester, CompatibilityTestResult


class TestModelRegistry(unittest.TestCase):
    """Test cases for the ModelRegistry class."""

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


class TestModelCompatibilityTester(unittest.TestCase):
    """Test cases for the ModelCompatibilityTester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tester = ModelCompatibilityTester()

    def test_config_detection_test(self):
        """Test the config detection test."""
        result = self.tester.test_model_config_detection("meta-llama/Llama-2-7b-chat-hf")
        self.assertIsInstance(result, CompatibilityTestResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.test_name, "config_detection")

    def test_family_detection_test(self):
        """Test the family detection test."""
        result = self.tester.test_family_detection("meta-llama/Llama-2-7b-chat-hf", ModelFamily.LLAMA)
        self.assertIsInstance(result, CompatibilityTestResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.test_name, "family_detection")

    def test_family_detection_test_wrong_family(self):
        """Test family detection test with wrong expected family."""
        result = self.tester.test_family_detection("meta-llama/Llama-2-7b-chat-hf", ModelFamily.GEMMA)
        self.assertIsInstance(result, CompatibilityTestResult)
        self.assertFalse(result.passed)
        self.assertIn("Expected gemma, got llama", result.error_message)

    def test_size_detection_test(self):
        """Test the size detection test."""
        result = self.tester.test_size_detection("meta-llama/Llama-2-7b-chat-hf", "small")
        self.assertIsInstance(result, CompatibilityTestResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.test_name, "size_detection")

    def test_quantization_config_test(self):
        """Test the quantization configuration test."""
        result = self.tester.test_quantization_config("meta-llama/Llama-2-7b-chat-hf")
        self.assertIsInstance(result, CompatibilityTestResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.test_name, "quantization_config")

    @patch('transformers.AutoTokenizer.from_pretrained')
    def test_tokenizer_compatibility_test(self, mock_tokenizer):
        """Test the tokenizer compatibility test."""
        mock_tokenizer.return_value = Mock()

        result = self.tester.test_tokenizer_compatibility("meta-llama/Llama-2-7b-chat-hf")
        self.assertIsInstance(result, CompatibilityTestResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.test_name, "tokenizer_compatibility")

    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    def test_model_loading_compatibility_test(self, mock_model):
        """Test the model loading compatibility test."""
        mock_model.return_value = Mock()

        result = self.tester.test_model_loading_compatibility("meta-llama/Llama-2-7b-chat-hf")
        self.assertIsInstance(result, CompatibilityTestResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.test_name, "model_loading_compatibility")

    def test_chat_template_support_test(self):
        """Test the chat template support test."""
        result = self.tester.test_chat_template_support("google/gemma-2b-it")
        self.assertIsInstance(result, CompatibilityTestResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.test_name, "chat_template_support")

    def test_full_compatibility_test(self):
        """Test running full compatibility test suite."""
        results = self.tester.run_full_compatibility_test("meta-llama/Llama-2-7b-chat-hf")

        # Should have results for all test methods
        expected_tests = [
            "test_model_config_detection",
            "test_family_detection",
            "test_size_detection",
            "test_quantization_config",
            "test_tokenizer_compatibility",
            "test_model_loading_compatibility",
            "test_chat_template_support"
        ]

        self.assertEqual(len(results), len(expected_tests))

        for result in results:
            self.assertIsInstance(result, CompatibilityTestResult)
            self.assertEqual(result.model_name, "meta-llama/Llama-2-7b-chat-hf")

    def test_batch_compatibility_test(self):
        """Test running compatibility tests for multiple models."""
        model_names = [
            "meta-llama/Llama-2-7b-chat-hf",
            "google/gemma-2b-it"
        ]

        results = self.tester.run_batch_compatibility_test(model_names)

        self.assertEqual(len(results), 2)
        for model_name in model_names:
            self.assertIn(model_name, results)
            self.assertIsInstance(results[model_name], list)

    def test_compatibility_report_generation(self):
        """Test generation of compatibility report."""
        # Create some mock results
        results = {
            "test-model": [
                CompatibilityTestResult("test-model", "test1", True),
                CompatibilityTestResult("test-model", "test2", True, warnings=["Warning message"]),
                CompatibilityTestResult("test-model", "test3", False, error_message="Error message")
            ]
        }

        report = self.tester.generate_compatibility_report(results)

        self.assertIsInstance(report, str)
        self.assertIn("# Model Compatibility Report", report)
        self.assertIn("test-model", report)
        self.assertIn("✅ Passed", report)
        self.assertIn("⚠️ Passed with Warnings", report)
        self.assertIn("❌ Failed", report)


class TestGlobalFunctions(unittest.TestCase):
    """Test cases for global functions."""

    def test_get_model_config_function(self):
        """Test the global get_model_config function."""
        config = get_model_config("meta-llama/Llama-2-7b-chat-hf")
        self.assertEqual(config.family, ModelFamily.LLAMA)
        self.assertEqual(config.size_category, "small")


if __name__ == '__main__':
    unittest.main()
