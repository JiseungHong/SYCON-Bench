"""
Model Registry for SYCON-Bench

This module provides a centralized registry for model configurations,
including quantization settings, chat templates, and model-specific quirks.
"""

import re
import torch
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum


class ModelFamily(Enum):
    """Enumeration of supported model families."""
    LLAMA = "llama"
    QWEN = "qwen"
    GEMMA = "gemma"
    MISTRAL = "mistral"
    GPT = "gpt"
    CLAUDE = "claude"
    FALCON = "falcon"
    MPT = "mpt"
    STARCODER = "starcoder"
    RWKV = "rwkv"
    UNKNOWN = "unknown"


class QuantizationStrategy(Enum):
    """Quantization strategies for different model sizes."""
    FLOAT16 = "float16"
    INT8 = "int8"
    INT4 = "int4"
    BFLOAT16 = "bfloat16"


@dataclass
class ModelConfig:
    """Configuration for a specific model or model family."""
    family: ModelFamily
    size_category: str  # e.g., "small", "medium", "large", "xlarge"
    quantization: QuantizationStrategy
    requires_trust_remote_code: bool = False
    chat_template_type: str = "auto"  # "auto", "custom", "fallback"
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    known_issues: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    api_based: bool = False
    pricing: Optional[Dict[str, float]] = None


class ModelRegistry:
    """Central registry for model configurations and compatibility."""

    def __init__(self):
        self._registry = {}
        self._family_patterns = {}
        self._size_patterns = {}
        self._initialize_registry()

    def _initialize_registry(self):
        """Initialize the model registry with known configurations."""

        # Family detection patterns
        self._family_patterns = {
            ModelFamily.LLAMA: [
                r"llama",
                r"meta-llama",
                r"codellama"
            ],
            ModelFamily.QWEN: [
                r"qwen",
                r"alibaba.*qwen"
            ],
            ModelFamily.GEMMA: [
                r"gemma",
                r"google.*gemma"
            ],
            ModelFamily.MISTRAL: [
                r"mistral",
                r"mixtral"
            ],
            ModelFamily.GPT: [
                r"gpt-[34]",
                r"openai.*gpt"
            ],
            ModelFamily.CLAUDE: [
                r"claude",
                r"anthropic.*claude"
            ],
            ModelFamily.FALCON: [
                r"falcon"
            ],
            ModelFamily.MPT: [
                r"mpt"
            ],
            ModelFamily.STARCODER: [
                r"starcoder",
                r"bigcode.*starcoder"
            ],
            ModelFamily.RWKV: [
                r"rwkv"
            ]
        }

        # Size detection patterns (in order of precedence)
        self._size_patterns = [
            (r"(?:70|72|65)b", "xlarge"),
            (r"(?:30|32|33|34|27)b", "large"),
            (r"(?:13|14|15|16|20)b", "medium"),
            (r"(?:[1-9]b|10b|11b|12b)", "small"),
            (r"(?:1|2|3|4|5|6|7|8|9)b", "small")
        ]

        # Register base configurations for each family
        self._register_family_configs()

    def _register_family_configs(self):
        """Register base configurations for each model family."""

        # Llama family configurations
        self.register_config("llama-small", ModelConfig(
            family=ModelFamily.LLAMA,
            size_category="small",
            quantization=QuantizationStrategy.FLOAT16,
            tokenizer_kwargs={
                "padding_side": "right",
                "add_eos_token": True
            },
            generation_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.0,
                "top_p": 0.9,
                "do_sample": False
            }
        ))

        self.register_config("llama-medium", ModelConfig(
            family=ModelFamily.LLAMA,
            size_category="medium",
            quantization=QuantizationStrategy.FLOAT16,
            tokenizer_kwargs={
                "padding_side": "right",
                "add_eos_token": True
            },
            generation_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.0,
                "top_p": 0.9,
                "do_sample": False
            }
        ))

        self.register_config("llama-large", ModelConfig(
            family=ModelFamily.LLAMA,
            size_category="large",
            quantization=QuantizationStrategy.INT8,
            tokenizer_kwargs={
                "padding_side": "right",
                "add_eos_token": True
            },
            generation_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.0,
                "top_p": 0.9,
                "do_sample": False
            }
        ))

        self.register_config("llama-xlarge", ModelConfig(
            family=ModelFamily.LLAMA,
            size_category="xlarge",
            quantization=QuantizationStrategy.INT4,
            tokenizer_kwargs={
                "padding_side": "right",
                "add_eos_token": True
            },
            generation_kwargs={
                "max_new_tokens": 512,
                "temperature": 0.0,
                "top_p": 0.9,
                "do_sample": False
            }
        ))

        # Gemma family configurations
        for size in ["small", "medium", "large", "xlarge"]:
            quant = {
                "small": QuantizationStrategy.FLOAT16,
                "medium": QuantizationStrategy.FLOAT16,
                "large": QuantizationStrategy.INT8,
                "xlarge": QuantizationStrategy.INT4
            }[size]

            self.register_config(f"gemma-{size}", ModelConfig(
                family=ModelFamily.GEMMA,
                size_category=size,
                quantization=quant,
                requires_trust_remote_code=True,
                chat_template_type="custom",
                tokenizer_kwargs={"trust_remote_code": True},
                model_kwargs={"trust_remote_code": True, "torch_dtype": torch.float16}
            ))

        # Qwen family configurations
        for size in ["small", "medium", "large", "xlarge"]:
            quant = {
                "small": QuantizationStrategy.FLOAT16,
                "medium": QuantizationStrategy.FLOAT16,
                "large": QuantizationStrategy.INT8,
                "xlarge": QuantizationStrategy.INT4
            }[size]

            self.register_config(f"qwen-{size}", ModelConfig(
                family=ModelFamily.QWEN,
                size_category=size,
                quantization=quant,
                requires_trust_remote_code=True,
                tokenizer_kwargs={"trust_remote_code": True},
                model_kwargs={"trust_remote_code": True}
            ))

        # Mistral family configurations
        for size in ["small", "medium", "large", "xlarge"]:
            quant = {
                "small": QuantizationStrategy.FLOAT16,
                "medium": QuantizationStrategy.FLOAT16,
                "large": QuantizationStrategy.INT8,
                "xlarge": QuantizationStrategy.INT4
            }[size]

            self.register_config(f"mistral-{size}", ModelConfig(
                family=ModelFamily.MISTRAL,
                size_category=size,
                quantization=quant,
                tokenizer_kwargs={
                    "padding_side": "right",
                    "add_eos_token": True
                },
                generation_kwargs={
                    "max_new_tokens": 512,
                    "temperature": 0.0,
                    "top_p": 0.9,
                    "do_sample": False
                }
            ))

        # API-based models
        self.register_config("gpt-4o", ModelConfig(
            family=ModelFamily.GPT,
            size_category="xlarge",
            quantization=QuantizationStrategy.FLOAT16,  # Not applicable for API
            api_based=True,
            pricing={"input": 5 / 1e6, "output": 15 / 1e6}
        ))

        self.register_config("claude-3", ModelConfig(
            family=ModelFamily.CLAUDE,
            size_category="xlarge",
            quantization=QuantizationStrategy.FLOAT16,  # Not applicable for API
            api_based=True,
            pricing={"input": 3 / 1e6, "output": 15 / 1e6}
        ))

        # Models requiring trust_remote_code
        trust_remote_families = {
            "falcon": ModelFamily.FALCON,
            "mpt": ModelFamily.MPT,
            "starcoder": ModelFamily.STARCODER,
            "rwkv": ModelFamily.RWKV
        }

        for family_name, family_enum in trust_remote_families.items():
            for size in ["small", "medium", "large", "xlarge"]:
                quant = {
                    "small": QuantizationStrategy.FLOAT16,
                    "medium": QuantizationStrategy.FLOAT16,
                    "large": QuantizationStrategy.INT8,
                    "xlarge": QuantizationStrategy.INT4
                }[size]

                self.register_config(f"{family_name}-{size}", ModelConfig(
                    family=family_enum,
                    size_category=size,
                    quantization=quant,
                    requires_trust_remote_code=True,
                    model_kwargs={"trust_remote_code": True}
                ))

    def register_config(self, key: str, config: ModelConfig):
        """Register a model configuration."""
        self._registry[key] = config

    def detect_model_family(self, model_name: str) -> ModelFamily:
        """Detect the model family from the model name."""
        model_name_lower = model_name.lower()

        for family, patterns in self._family_patterns.items():
            for pattern in patterns:
                if re.search(pattern, model_name_lower):
                    return family

        return ModelFamily.UNKNOWN

    def detect_model_size(self, model_name: str) -> str:
        """Detect the model size category from the model name."""
        model_name_lower = model_name.lower()

        for pattern, size_category in self._size_patterns:
            if re.search(pattern, model_name_lower):
                return size_category

        return "medium"  # Default fallback

    def get_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        # First try exact match
        if model_name in self._registry:
            return self._registry[model_name]

        # Try to find by family and size
        family = self.detect_model_family(model_name)
        size = self.detect_model_size(model_name)

        config_key = f"{family.value}-{size}"
        if config_key in self._registry:
            return self._registry[config_key]

        # Fallback to a generic configuration
        return self._create_fallback_config(model_name, family, size)

    def _create_fallback_config(self, model_name: str, family: ModelFamily, size: str) -> ModelConfig:
        """Create a fallback configuration for unknown models."""
        quantization_map = {
            "small": QuantizationStrategy.FLOAT16,
            "medium": QuantizationStrategy.FLOAT16,
            "large": QuantizationStrategy.INT8,
            "xlarge": QuantizationStrategy.INT4
        }

        # Check if it's an API-based model
        api_based = any(prefix in model_name.lower() for prefix in ["openai/", "anthropic/", "claude"])

        return ModelConfig(
            family=family,
            size_category=size,
            quantization=quantization_map.get(size, QuantizationStrategy.FLOAT16),
            api_based=api_based,
            known_issues=["Fallback configuration - may need manual tuning"]
        )

    def list_supported_families(self) -> List[ModelFamily]:
        """List all supported model families."""
        return list(self._family_patterns.keys())

    def get_compatibility_matrix(self) -> Dict[str, Dict[str, Any]]:
        """Get the full compatibility matrix."""
        matrix = {}

        for key, config in self._registry.items():
            matrix[key] = {
                "family": config.family.value,
                "size_category": config.size_category,
                "quantization": config.quantization.value,
                "api_based": config.api_based,
                "requires_trust_remote_code": config.requires_trust_remote_code,
                "known_issues": config.known_issues,
                "dependencies": config.dependencies
            }

        return matrix


# Global registry instance
_global_registry = ModelRegistry()


def get_model_config(model_name: str) -> ModelConfig:
    """Get model configuration from the global registry."""
    return _global_registry.get_config(model_name)


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return _global_registry
