

import pytest
from debate_setting.model_registry import ModelRegistry, ModelConfig

def test_get_model_family():
    # Test known model families
    assert ModelRegistry.get_model_family("Llama-2-70B") == "llama"
    assert ModelRegistry.get_model_family("Qwen-72B") == "qwen"
    # Test unknown model
    assert ModelRegistry.get_model_family("UnknownModel-1B") == "unknown"

def test_get_quantization_config():
    # Test known configs
    llama_config = ModelRegistry.get_quantization_config("llama")
    assert llama_config == {"load_in_4bit": True, "bnb_4bit_compute_dtype": "float16"}

    qwen_config = ModelRegistry.get_quantization_config("qwen")
    assert qwen_config == {"load_in_8bit": True}

    # Test unknown config
    assert ModelRegistry.get_quantization_config("unknown") == {}

def test_supported_families():
    supported = ModelRegistry.get_supported_families()
    assert "llama" in supported
    assert "qwen" in supported

