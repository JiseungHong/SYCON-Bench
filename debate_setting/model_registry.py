
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
            quantization={"load_in_4bit": True, "bnb_4bit_compute_dtype": "float16"},
            chat_template="llama-2",
            dependencies=["transformers", "bitsandbytes"]
        ),
        "qwen": ModelConfig(
            family="Qwen",
            quantization={"load_in_8bit": True},
            chat_template="chatml",
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
    def get_quantization_config(cls, family: str) -> dict:
        """Get quantization config for model family"""
        model_config = cls._models.get(family)
        return model_config.quantization if model_config else {}

    @classmethod
    def get_supported_families(cls) -> list:
        """Return list of supported model families"""
        return list(cls._models.keys())
