# Migration Guide: From Old Models.py to Registry System

This guide explains how to migrate from the old fragile string-matching approach to the new registry-based model system.

## Overview of Changes

### Before (Old System)
- Scattered quantization logic across multiple files
- Fragile string matching for model detection
- Inconsistent model-specific handling
- No systematic testing

### After (New System)
- Centralized model registry with structured configurations
- Robust family and size detection using regex patterns
- Systematic compatibility testing framework
- Easy addition of new models

## Step-by-Step Migration

### 1. Update Imports

**Old:**
```python
from models import OpenModel, ClosedModel
```

**New:**
```python
from model_registry.base_models import create_model
# Or for backward compatibility:
from model_registry.base_models import RegistryOpenModel as OpenModel, RegistryClosedModel as ClosedModel
```

### 2. Replace Model Creation Logic

**Old:**
```python
def create_model(model_name, api_key=None, base_url=None):
    """Create and return the appropriate model instance based on model name."""
    if model_name.startswith(("openai/", "anthropic/", "claude")):
        return ClosedModel(model_id=model_name, api_key=api_key, base_url=base_url)
    else:
        return OpenModel(model_name=model_name)
```

**New:**
```python
from model_registry.base_models import create_model
# The function is already implemented and handles all the logic automatically
```

### 3. Remove Manual Quantization Logic

**Old:**
```python
# Set up quantization parameters based on model size
if "70B" in self.model_name or "65B" in self.model_name or "72B" in self.model_name:
    # 4-bit quantization for very large models
    quantization_config = {"load_in_4bit": True, "bnb_4bit_compute_dtype": torch.float16}
elif any(size in self.model_name for size in ["32B", "33B", "27B", "34B", "30B"]):
    # 8-bit quantization for large models
    quantization_config = {"load_in_8bit": True}
else:
    # 16-bit for smaller models that can fit in memory
    quantization_config = {"torch_dtype": torch.float16}
```

**New:**
```python
# No manual logic needed - handled automatically by registry
# Configuration is determined by model family and size detection
```

### 4. Remove Model-Specific Handling

**Old:**
```python
# Handle Gemma models
if "gemma" in model_name_lower:
    print("Using Gemma-specific configurations")
    tokenizer_kwargs["trust_remote_code"] = True
    model_kwargs["trust_remote_code"] = True
    model_kwargs["torch_dtype"] = torch.float16

# Llama models need padding token configured
elif any(name in model_name_lower for name in ["llama", "mistral"]):
    tokenizer_kwargs["padding_side"] = "right"
    tokenizer_kwargs["add_eos_token"] = True
```

**New:**
```python
# All model-specific handling is centralized in the registry
# No manual configuration needed
```

### 5. Update Model Setup

**Old:**
```python
class OpenModel(BaseModel):
    def setup(self):
        # Manual configuration logic...
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **tokenizer_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
```

**New:**
```python
# The RegistryOpenModel class handles all setup automatically
model = create_model(model_name)
model.setup()  # Uses registry configuration
```

## File-by-File Migration

### debate-setting/models.py

1. **Create new file** `debate-setting/models_registry.py`:
   ```python
   from model_registry.base_models import create_model

   # Backward compatibility
   def create_model_instance(model_name, api_key=None, base_url=None):
       return create_model(model_name, api_key=api_key, base_url=base_url)
   ```

2. **Update run_benchmark.py**:
   ```python
   # Old
   from models import create_model

   # New
   from models_registry import create_model_instance as create_model
   ```

3. **Test the migration**:
   ```bash
   python tests/run_compatibility_tests.py --models meta-llama/Llama-2-7b-chat-hf
   ```

### ethical-setting/models.py

Follow the same pattern as debate-setting.

### false-presuppositions-setting/models.py

Follow the same pattern as debate-setting.

## Testing Your Migration

### 1. Run Compatibility Tests

```bash
# Test specific models
python tests/run_compatibility_tests.py --models meta-llama/Llama-2-7b-chat-hf google/gemma-2b-it

# Test all common models
python tests/run_compatibility_tests.py

# Generate detailed report
python tests/run_compatibility_tests.py --output migration_test_report.md
```

### 2. Compare Outputs

Run the same benchmark with both old and new systems to ensure consistent results:

```bash
# Old system
python run_benchmark.py meta-llama/Llama-2-7b-chat-hf --output_dir old_results

# New system (after migration)
python run_benchmark.py meta-llama/Llama-2-7b-chat-hf --output_dir new_results

# Compare results
diff -r old_results new_results
```

### 3. Performance Testing

Monitor memory usage and loading times:

```python
import time
import psutil

# Test model loading time
start_time = time.time()
model = create_model("meta-llama/Llama-2-7b-chat-hf")
model.setup()
load_time = time.time() - start_time

# Monitor memory usage
memory_usage = psutil.virtual_memory().used / 1024**3  # GB

print(f"Load time: {load_time:.2f}s")
print(f"Memory usage: {memory_usage:.2f}GB")
```

## Common Issues and Solutions

### 1. Import Errors

**Issue:** `ModuleNotFoundError: No module named 'model_registry'`

**Solution:** Add the workspace root to Python path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### 2. Model Not Found

**Issue:** Model configuration not found in registry

**Solution:** Check if model is supported or add custom configuration:
```python
from model_registry.registry import get_registry
registry = get_registry()

# Check if model is supported
config = registry.get_config("your-model-name")
if "Fallback configuration" in str(config.known_issues):
    print("Model needs explicit configuration")
```

### 3. Quantization Issues

**Issue:** Out of memory errors with new quantization settings

**Solution:** Override quantization strategy:
```python
from model_registry.registry import ModelConfig, QuantizationStrategy
from model_registry.base_models import RegistryOpenModel

class CustomModel(RegistryOpenModel):
    def __init__(self, model_name):
        super().__init__(model_name)
        # Override quantization for this specific case
        self.config.quantization = QuantizationStrategy.INT4
```

### 4. Chat Template Issues

**Issue:** Chat template not working correctly

**Solution:** Check model family and template type:
```python
config = get_model_config("your-model-name")
print(f"Chat template type: {config.chat_template_type}")
print(f"Model family: {config.family.value}")

# For Gemma models, system messages are automatically handled
# For other models, check if fallback template is being used
```

## Rollback Plan

If you need to rollback to the old system:

1. **Keep old files**: Don't delete the original `models.py` files
2. **Use git branches**: Create a migration branch
3. **Gradual migration**: Migrate one setting at a time
4. **Fallback imports**: Use conditional imports:

```python
try:
    from model_registry.base_models import create_model
    USE_REGISTRY = True
except ImportError:
    from models import create_model
    USE_REGISTRY = False
```

## Benefits After Migration

### 1. Reduced Code Duplication
- Single source of truth for model configurations
- No more copy-pasted quantization logic

### 2. Better Error Handling
- Systematic compatibility testing
- Clear error messages and warnings

### 3. Easier Maintenance
- Add new models by updating registry
- Centralized configuration management

### 4. Improved Reliability
- Robust model detection using regex patterns
- Fallback configurations for unknown models

### 5. Better Documentation
- Comprehensive compatibility matrix
- Clear troubleshooting guides

## Next Steps

1. **Complete migration** for all three settings
2. **Add custom configurations** for any missing models
3. **Set up CI/CD** to run compatibility tests automatically
4. **Monitor performance** and adjust configurations as needed
5. **Contribute back** any new model configurations to the registry

## Getting Help

1. **Check compatibility first**: Run compatibility tests
2. **Review documentation**: Check the compatibility matrix
3. **Enable debug logging**: Use verbose mode for troubleshooting
4. **Test incrementally**: Migrate one setting at a time

For additional support, refer to the [Model Compatibility Matrix](model_compatibility_matrix.md) and run the demonstration script:

```bash
python demo_model_registry.py
```
