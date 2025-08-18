# Model Compatibility Matrix

This document provides a comprehensive overview of model compatibility in SYCON-Bench, including supported model families, configuration requirements, and known issues.

## Overview

The SYCON-Bench model registry system provides centralized configuration management for different model families, ensuring consistent and optimal settings across all benchmark settings (debate, ethical, false-presuppositions).

## Supported Model Families

### 1. Llama Family
- **Models**: Llama-2, Llama-3, CodeLlama
- **Sizes**: 7B (small), 13B (medium), 30B+ (large), 65B+ (xlarge)
- **Quantization**:
  - Small/Medium: FP16
  - Large: INT8
  - XLarge: INT4
- **Special Requirements**:
  - Padding token configuration
  - EOS token handling
- **Chat Template**: Auto-detection with fallback

### 2. Gemma Family
- **Models**: Gemma-2B, Gemma-7B, Gemma-2-9B
- **Sizes**: 2B (small), 7B+ (medium/large)
- **Quantization**:
  - Small: FP16
  - Medium+: INT8/INT4 based on size
- **Special Requirements**:
  - `trust_remote_code=True`
  - Custom chat template (system → user message transformation)
- **Chat Template**: Custom handling for system messages

### 3. Qwen Family
- **Models**: Qwen2, Qwen1.5
- **Sizes**: 7B (small), 14B (medium), 32B+ (large)
- **Quantization**: Size-based (FP16 → INT8 → INT4)
- **Special Requirements**:
  - `trust_remote_code=True`
- **Chat Template**: Auto-detection

### 4. Mistral Family
- **Models**: Mistral-7B, Mixtral-8x7B
- **Sizes**: 7B (small), 8x7B (treated as small)
- **Quantization**: Size-based
- **Special Requirements**:
  - Padding token configuration
  - EOS token handling
- **Chat Template**: Auto-detection

### 5. API-Based Models

#### GPT Family (OpenAI)
- **Models**: GPT-4o, GPT-3.5-turbo
- **Pricing**: Input $5/1M tokens, Output $15/1M tokens (GPT-4o)
- **Requirements**: OpenAI API key
- **Rate Limits**: Managed by litellm

#### Claude Family (Anthropic)
- **Models**: Claude-3 (Sonnet, Opus, Haiku)
- **Pricing**: Input $3/1M tokens, Output $15/1M tokens (estimated)
- **Requirements**: Anthropic API key
- **Rate Limits**: Managed by litellm

### 6. Other Families
- **Falcon**: Requires `trust_remote_code=True`
- **MPT**: Requires `trust_remote_code=True`
- **StarCoder**: Requires `trust_remote_code=True`
- **RWKV**: Requires `trust_remote_code=True`

## Quantization Strategies

The registry automatically selects quantization strategies based on model size:

| Size Category | Parameter Range | Quantization | Memory Usage |
|---------------|----------------|--------------|--------------|
| Small         | 1B-12B         | FP16         | ~24GB        |
| Medium        | 13B-20B        | FP16         | ~40GB        |
| Large         | 27B-34B        | INT8         | ~35GB        |
| XLarge        | 65B+           | INT4         | ~40GB        |

## Configuration Examples

### Llama-2 7B Chat
```python
from model_registry import get_model_config

config = get_model_config("meta-llama/Llama-2-7b-chat-hf")
# Returns:
# - family: LLAMA
# - size_category: small
# - quantization: FLOAT16
# - tokenizer_kwargs: {"padding_side": "right", "add_eos_token": True}
```

### Gemma 2B IT
```python
config = get_model_config("google/gemma-2b-it")
# Returns:
# - family: GEMMA
# - size_category: small
# - quantization: FLOAT16
# - requires_trust_remote_code: True
# - chat_template_type: custom
```

### GPT-4o
```python
config = get_model_config("openai/gpt-4o")
# Returns:
# - family: GPT
# - api_based: True
# - pricing: {"input": 5e-6, "output": 15e-6}
```

## Usage in Benchmark Settings

### Using Registry-Based Models

Replace the old model loading code:

```python
# Old approach (scattered across files)
if "70B" in model_name:
    quantization_config = {"load_in_4bit": True}
elif "gemma" in model_name.lower():
    model_kwargs["trust_remote_code"] = True

# New approach (centralized)
from model_registry.base_models import create_model

model = create_model(model_name, api_key=api_key)
model.setup()
```

### Compatibility Testing

Test model compatibility before running benchmarks:

```python
from model_registry.compatibility import ModelCompatibilityTester

tester = ModelCompatibilityTester()
results = tester.run_full_compatibility_test("meta-llama/Llama-2-7b-chat-hf")

for result in results:
    if not result.passed:
        print(f"❌ {result.test_name}: {result.error_message}")
    elif result.warnings:
        print(f"⚠️ {result.test_name}: {result.warnings}")
```

## Known Issues and Limitations

### Model-Specific Issues

1. **Gemma Models**
   - System messages must be merged with user messages
   - Assistant role must be transformed to "model" role
   - Requires `trust_remote_code=True`

2. **Large Models (70B+)**
   - Require 4-bit quantization for single GPU
   - May need multiple GPUs for optimal performance
   - Longer loading times

3. **API Models**
   - Rate limiting may affect batch processing
   - Cost accumulation with large datasets
   - Network dependency

### General Limitations

1. **Memory Requirements**
   - Quantization reduces precision
   - Large models may not fit on single GPU
   - Batch size limitations

2. **Performance Variations**
   - Quantized models may have different outputs
   - API models have variable latency
   - Temperature=0 may not be truly deterministic

## Troubleshooting Guide

### Common Issues

1. **"Model not found" errors**
   - Check model name spelling
   - Verify model exists on Hugging Face Hub
   - Check internet connection

2. **Out of memory errors**
   - Try smaller batch size
   - Use higher quantization (INT8 → INT4)
   - Clear GPU cache between runs

3. **Chat template errors**
   - Check if model supports chat templates
   - Verify message format
   - Use fallback template if needed

4. **API errors**
   - Verify API key is set
   - Check rate limits
   - Ensure sufficient credits

### Getting Help

1. **Check compatibility first**:
   ```bash
   python tests/run_compatibility_tests.py --models your-model-name
   ```

2. **Enable verbose logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

3. **Review model configuration**:
   ```python
   from model_registry import get_model_config
   config = get_model_config("your-model-name")
   print(config)
   ```

## Adding New Models

To add support for a new model:

1. **Identify model family and size**
2. **Test compatibility**:
   ```bash
   python tests/run_compatibility_tests.py --models new-model-name
   ```
3. **Add explicit configuration if needed**:
   ```python
   from model_registry.registry import get_registry
   registry = get_registry()
   registry.register_config("new-model", ModelConfig(...))
   ```

## Performance Benchmarks

| Model Family | Size | Load Time | Memory Usage | Inference Speed |
|--------------|------|-----------|--------------|-----------------|
| Llama-2      | 7B   | ~30s      | ~14GB        | ~50 tokens/s    |
| Llama-2      | 13B  | ~45s      | ~26GB        | ~35 tokens/s    |
| Llama-2      | 70B  | ~120s     | ~40GB (4bit) | ~15 tokens/s    |
| Gemma        | 2B   | ~15s      | ~4GB         | ~80 tokens/s    |
| Gemma        | 7B   | ~25s      | ~14GB        | ~55 tokens/s    |
| GPT-4o       | API  | ~1s       | N/A          | ~40 tokens/s    |

*Note: Benchmarks are approximate and may vary based on hardware and configuration.*

## Migration Guide

### From Old Models.py to Registry System

1. **Replace imports**:
   ```python
   # Old
   from models import OpenModel, ClosedModel

   # New
   from model_registry.base_models import create_model
   ```

2. **Update model creation**:
   ```python
   # Old
   if model_name.startswith("openai/"):
       model = ClosedModel(model_name, api_key)
   else:
       model = OpenModel(model_name)

   # New
   model = create_model(model_name, api_key=api_key)
   ```

3. **Remove manual configuration**:
   ```python
   # Old - manual quantization logic
   if "70B" in model_name:
       quantization_config = {"load_in_4bit": True}

   # New - automatic via registry
   # No manual configuration needed
   ```

## Future Enhancements

1. **Dynamic model discovery** from Hugging Face Hub
2. **Automatic hardware optimization** based on available GPU memory
3. **Model performance profiling** and recommendation system
4. **Integration with model serving frameworks** (vLLM, TensorRT-LLM)
5. **Support for fine-tuned models** and custom configurations
