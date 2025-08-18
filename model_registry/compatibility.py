"""
Model Compatibility Testing Framework

This module provides tools for testing model compatibility and
validating model configurations.
"""

import os
import sys
import torch
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from unittest.mock import Mock, patch

from .registry import ModelRegistry, ModelConfig, ModelFamily, get_model_config


@dataclass
class CompatibilityTestResult:
    """Result of a compatibility test."""
    model_name: str
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    warnings: List[str] = None
    execution_time: Optional[float] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ModelCompatibilityTester:
    """Framework for testing model compatibility."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        self.registry = registry or ModelRegistry()
        self.logger = logging.getLogger(__name__)

    def test_model_config_detection(self, model_name: str) -> CompatibilityTestResult:
        """Test if model configuration can be detected correctly."""
        try:
            config = self.registry.get_config(model_name)

            # Check if we got a fallback config
            warnings = []
            if "Fallback configuration" in str(config.known_issues):
                warnings.append("Using fallback configuration - consider adding explicit config")

            return CompatibilityTestResult(
                model_name=model_name,
                test_name="config_detection",
                passed=True,
                warnings=warnings
            )
        except Exception as e:
            return CompatibilityTestResult(
                model_name=model_name,
                test_name="config_detection",
                passed=False,
                error_message=str(e)
            )

    def test_family_detection(self, model_name: str, expected_family: Optional[ModelFamily] = None) -> CompatibilityTestResult:
        """Test if model family is detected correctly."""
        try:
            detected_family = self.registry.detect_model_family(model_name)

            if expected_family and detected_family != expected_family:
                return CompatibilityTestResult(
                    model_name=model_name,
                    test_name="family_detection",
                    passed=False,
                    error_message=f"Expected {expected_family.value}, got {detected_family.value}"
                )

            warnings = []
            if detected_family == ModelFamily.UNKNOWN:
                warnings.append("Model family could not be detected")

            return CompatibilityTestResult(
                model_name=model_name,
                test_name="family_detection",
                passed=True,
                warnings=warnings
            )
        except Exception as e:
            return CompatibilityTestResult(
                model_name=model_name,
                test_name="family_detection",
                passed=False,
                error_message=str(e)
            )

    def test_size_detection(self, model_name: str, expected_size: Optional[str] = None) -> CompatibilityTestResult:
        """Test if model size is detected correctly."""
        try:
            detected_size = self.registry.detect_model_size(model_name)

            if expected_size and detected_size != expected_size:
                return CompatibilityTestResult(
                    model_name=model_name,
                    test_name="size_detection",
                    passed=False,
                    error_message=f"Expected {expected_size}, got {detected_size}"
                )

            return CompatibilityTestResult(
                model_name=model_name,
                test_name="size_detection",
                passed=True
            )
        except Exception as e:
            return CompatibilityTestResult(
                model_name=model_name,
                test_name="size_detection",
                passed=False,
                error_message=str(e)
            )

    def test_quantization_config(self, model_name: str) -> CompatibilityTestResult:
        """Test if quantization configuration is appropriate."""
        try:
            config = self.registry.get_config(model_name)

            warnings = []

            # Check if quantization strategy matches size category
            size_quant_map = {
                "small": ["float16", "bfloat16"],
                "medium": ["float16", "bfloat16"],
                "large": ["int8", "float16"],
                "xlarge": ["int4", "int8"]
            }

            expected_quants = size_quant_map.get(config.size_category, [])
            if expected_quants and config.quantization.value not in expected_quants:
                warnings.append(f"Quantization {config.quantization.value} may not be optimal for {config.size_category} model")

            return CompatibilityTestResult(
                model_name=model_name,
                test_name="quantization_config",
                passed=True,
                warnings=warnings
            )
        except Exception as e:
            return CompatibilityTestResult(
                model_name=model_name,
                test_name="quantization_config",
                passed=False,
                error_message=str(e)
            )

    def test_tokenizer_compatibility(self, model_name: str) -> CompatibilityTestResult:
        """Test tokenizer loading with mock objects."""
        try:
            config = self.registry.get_config(model_name)

            # Mock the tokenizer loading
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                mock_tokenizer.return_value = Mock()

                # Try to load with the config's tokenizer kwargs
                try:
                    mock_tokenizer.from_pretrained(model_name, **config.tokenizer_kwargs)
                    return CompatibilityTestResult(
                        model_name=model_name,
                        test_name="tokenizer_compatibility",
                        passed=True
                    )
                except Exception as e:
                    return CompatibilityTestResult(
                        model_name=model_name,
                        test_name="tokenizer_compatibility",
                        passed=False,
                        error_message=f"Tokenizer kwargs failed: {str(e)}"
                    )
        except Exception as e:
            return CompatibilityTestResult(
                model_name=model_name,
                test_name="tokenizer_compatibility",
                passed=False,
                error_message=str(e)
            )

    def test_model_loading_compatibility(self, model_name: str) -> CompatibilityTestResult:
        """Test model loading configuration with mock objects."""
        try:
            config = self.registry.get_config(model_name)

            if config.api_based:
                # For API-based models, just check if we have the right config
                return CompatibilityTestResult(
                    model_name=model_name,
                    test_name="model_loading_compatibility",
                    passed=True,
                    warnings=["API-based model - skipping loading test"]
                )

            # Mock the model loading
            with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
                mock_model.return_value = Mock()

                # Build model kwargs based on config
                model_kwargs = {"device_map": "auto"}
                model_kwargs.update(config.model_kwargs)

                # Add quantization settings
                if config.quantization.value == "int4":
                    model_kwargs.update({
                        "load_in_4bit": True,
                        "bnb_4bit_compute_dtype": torch.float16
                    })
                elif config.quantization.value == "int8":
                    model_kwargs.update({"load_in_8bit": True})
                else:
                    model_kwargs.update({"torch_dtype": torch.float16})

                try:
                    mock_model.from_pretrained(model_name, **model_kwargs)
                    return CompatibilityTestResult(
                        model_name=model_name,
                        test_name="model_loading_compatibility",
                        passed=True
                    )
                except Exception as e:
                    return CompatibilityTestResult(
                        model_name=model_name,
                        test_name="model_loading_compatibility",
                        passed=False,
                        error_message=f"Model kwargs failed: {str(e)}"
                    )
        except Exception as e:
            return CompatibilityTestResult(
                model_name=model_name,
                test_name="model_loading_compatibility",
                passed=False,
                error_message=str(e)
            )

    def test_chat_template_support(self, model_name: str) -> CompatibilityTestResult:
        """Test chat template handling."""
        try:
            config = self.registry.get_config(model_name)

            warnings = []

            # Check if model family has known chat template issues
            if config.family == ModelFamily.GEMMA:
                if config.chat_template_type != "custom":
                    warnings.append("Gemma models require custom chat template handling")

            # Check for fallback template usage
            if config.chat_template_type == "fallback":
                warnings.append("Model uses fallback chat template - may not be optimal")

            return CompatibilityTestResult(
                model_name=model_name,
                test_name="chat_template_support",
                passed=True,
                warnings=warnings
            )
        except Exception as e:
            return CompatibilityTestResult(
                model_name=model_name,
                test_name="chat_template_support",
                passed=False,
                error_message=str(e)
            )

    def run_full_compatibility_test(self, model_name: str) -> List[CompatibilityTestResult]:
        """Run all compatibility tests for a model."""
        tests = [
            self.test_model_config_detection,
            self.test_family_detection,
            self.test_size_detection,
            self.test_quantization_config,
            self.test_tokenizer_compatibility,
            self.test_model_loading_compatibility,
            self.test_chat_template_support
        ]

        results = []
        for test_func in tests:
            try:
                result = test_func(model_name)
                results.append(result)
            except Exception as e:
                results.append(CompatibilityTestResult(
                    model_name=model_name,
                    test_name=test_func.__name__,
                    passed=False,
                    error_message=f"Test execution failed: {str(e)}"
                ))

        return results

    def run_batch_compatibility_test(self, model_names: List[str]) -> Dict[str, List[CompatibilityTestResult]]:
        """Run compatibility tests for multiple models."""
        results = {}

        for model_name in model_names:
            self.logger.info(f"Testing compatibility for {model_name}")
            results[model_name] = self.run_full_compatibility_test(model_name)

        return results

    def generate_compatibility_report(self, results: Dict[str, List[CompatibilityTestResult]]) -> str:
        """Generate a human-readable compatibility report."""
        report = ["# Model Compatibility Report\n"]

        for model_name, test_results in results.items():
            report.append(f"## {model_name}\n")

            passed_tests = sum(1 for r in test_results if r.passed)
            total_tests = len(test_results)

            report.append(f"**Overall**: {passed_tests}/{total_tests} tests passed\n")

            # Group results by status
            passed = [r for r in test_results if r.passed and not r.warnings]
            passed_with_warnings = [r for r in test_results if r.passed and r.warnings]
            failed = [r for r in test_results if not r.passed]

            if passed:
                report.append("### ✅ Passed")
                for result in passed:
                    report.append(f"- {result.test_name}")
                report.append("")

            if passed_with_warnings:
                report.append("### ⚠️ Passed with Warnings")
                for result in passed_with_warnings:
                    report.append(f"- {result.test_name}")
                    for warning in result.warnings:
                        report.append(f"  - ⚠️ {warning}")
                report.append("")

            if failed:
                report.append("### ❌ Failed")
                for result in failed:
                    report.append(f"- {result.test_name}: {result.error_message}")
                report.append("")

            report.append("---\n")

        return "\n".join(report)


def test_common_models() -> Dict[str, List[CompatibilityTestResult]]:
    """Test compatibility for commonly used models."""
    tester = ModelCompatibilityTester()

    common_models = [
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "meta-llama/Llama-2-70b-chat-hf",
        "google/gemma-2b-it",
        "google/gemma-7b-it",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "Qwen/Qwen2-7B-Instruct",
        "openai/gpt-4o",
        "anthropic/claude-3-sonnet-20240229"
    ]

    return tester.run_batch_compatibility_test(common_models)
