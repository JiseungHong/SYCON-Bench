

import pytest
from debate_setting.model_registry import ModelRegistry, ModelConfig

def test_get_model_family():
    # Test known model families
    assert ModelRegistry.get_model_family("Llama-2-70B") == "llama"
    assert ModelRegistry.get_model_family("Qwen-72B") == "qwen"
    assert ModelRegistry.get_model_family("gemma-7b") == "gemma"
    assert ModelRegistry.get_model_family("mistral-7b") == "mistral"
    # Test unknown model
    assert ModelRegistry.get_model_family("UnknownModel-1B") == "unknown"

def test_get_quantization_config():
    # Test size-specific quantization for Llama
    assert ModelRegistry.get_quantization_config("Llama-2-70B") == {"quant_type": "4bit"}
    # Test default quantization for Llama
    assert ModelRegistry.get_quantization_config("Llama-2-13B") == {"quant_type": "8bit"}
    
    # Test size-specific quantization for Qwen
    assert ModelRegistry.get_quantization_config("Qwen-14B") == {"quant_type": "4bit"}
    assert ModelRegistry.get_quantization_config("Qwen-72B") == {"quant_type": "4bit"}
    # Test default quantization for Qwen
    assert ModelRegistry.get_quantization_config("Qwen-7B") == {"quant_type": "8bit"}
    
    # Test default quantization for Gemma
    assert ModelRegistry.get_quantization_config("gemma-7b") == {"quant_type": "8bit"}

    # Test unknown model
    assert ModelRegistry.get_quantization_config("UnknownModel-1B") == {}

def test_supported_families():
    supported = ModelRegistry.get_supported_families()
    assert "llama" in supported
    assert "qwen" in supported


@pytest.mark.xfail(reason="Not yet implemented")
def test_get_chat_template():
    assert ModelRegistry.get_chat_template("llama") == "llama-2"
    assert ModelRegistry.get_chat_template("qwen") == "chatml"
    assert ModelRegistry.get_chat_template("gemma") == "gemma"
    assert ModelRegistry.get_chat_template("unknown") is None

