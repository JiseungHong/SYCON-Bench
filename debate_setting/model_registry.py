
import re
from dataclasses import dataclass

@dataclass
class ModelConfig:
    family: str
    quantization: dict
    chat_template: str
    dependencies: list[str]

class ModelRegistry:
    _models = {
        "llama": ModelConfig(
            family="Llama",
            quantization={"70B": "4bit", "default": "8bit"},
            chat_template="llama-2",
            dependencies=["transformers", "bitsandbytes", "accelerate"]
        ),
        "qwen": ModelConfig(
            family="Qwen",
            quantization={"14B": "4bit", "72B": "4bit", "default": "8bit"},
            chat_template="chatml",
            dependencies=["transformers", "accelerate"]
        ),
        "gemma": ModelConfig(
            family="Gemma",
            quantization={"default": "8bit"},
            chat_template="gemma",
            dependencies=["transformers", "accelerate"]
        ),
        "mistral": ModelConfig(
            family="Mistral",
            quantization={"default": "8bit"},
            chat_template="mistral",
            dependencies=["transformers", "accelerate"]
        ),
        # Add other model families here
    }

    @classmethod
    def get_model_family(cls, model_name: str) -> str:
        """Identify model family from model name using patterns"""
        model_name = model_name.lower()
        for pattern in cls._models:
            if re.search(rf"\b{pattern}\b", model_name):
                return pattern
        return "unknown"

    @classmethod
    def get_quantization_config(cls, model_name: str) -> dict:
        """Get quantization config for a given model name"""
        family = cls.get_model_family(model_name)
        model_config = cls._models.get(family)

        if not model_config:
            return {}

        # Check for size-specific quantization
        for size, quant_type in model_config.quantization.items():
            if size != "default" and size in model_name:
                return {"quant_type": quant_type}

        # Fallback to default quantization
        return {"quant_type": model_config.quantization.get("default")}

    @classmethod
    def get_supported_families(cls) -> list:
        """Return list of supported model families"""
        return list(cls._models.keys())
