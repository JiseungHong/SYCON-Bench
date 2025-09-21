
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pytest
from model_registry import ModelRegistry, ModelConfig

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


def test_gemma_config():
    gemma_config = ModelRegistry.get_quantization_config("gemma")
    assert gemma_config == {"torch_dtype": "float16"}
    assert ModelRegistry.get_model_family("google/gemma-3-12b-it") == "gemma"

def test_gpt_claude_config():
    gpt_config = ModelRegistry.get_quantization_config("gpt")
    claude_config = ModelRegistry.get_quantization_config("claude")
    assert gpt_config == {}
    assert claude_config == {}
    assert ModelRegistry.get_model_family("openai/gpt-4o") == "gpt"
    assert ModelRegistry.get_model_family("anthropic/claude-3-opus") == "claude"

