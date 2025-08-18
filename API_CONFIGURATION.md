# API Configuration Guide

This document explains the unified API configuration system for SYCON-Bench, which provides consistent environment variable handling across all settings.

## Overview

The unified API configuration system (`api_config.py`) standardizes how API keys and provider-specific configurations are handled across all three settings:
- `debate-setting`
- `ethical-setting`
- `false-presuppositions-setting`

## Supported Providers

The system supports the following model providers and their respective environment variables:

| Provider | Model Prefixes | Environment Variable | Additional Variables |
|----------|----------------|---------------------|---------------------|
| OpenAI | `openai/`, `gpt-` | `OPENAI_API_KEY` | - |
| Anthropic | `anthropic/`, `claude` | `ANTHROPIC_API_KEY` | - |
| Azure OpenAI | `azure/`, `azure-openai/` | `AZURE_OPENAI_API_KEY` | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION` |
| Google | `google/`, `gemini` | `GOOGLE_API_KEY` | `GOOGLE_PROJECT_ID` |
| Cohere | `cohere/` | `COHERE_API_KEY` | - |
| Hugging Face | `huggingface/`, `hf/` | `HUGGINGFACE_API_KEY` | - |

## Environment Variable Setup

### OpenAI Models
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### Anthropic Models
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### Azure OpenAI Models
```bash
export AZURE_OPENAI_API_KEY="your-azure-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"  # Optional, defaults to this version
```

### Google Models
```bash
export GOOGLE_API_KEY="your-google-api-key"
export GOOGLE_PROJECT_ID="your-google-project-id"  # Optional
```

## Usage Examples

### Using the ModelFactory

The `ModelFactory` class automatically detects the model type and creates the appropriate model instance:

```python
from models import ModelFactory

# OpenAI model - requires OPENAI_API_KEY
model = ModelFactory.create_model("openai/gpt-4o")

# Anthropic model - requires ANTHROPIC_API_KEY
model = ModelFactory.create_model("anthropic/claude-3-sonnet")

# Azure OpenAI model - requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT
model = ModelFactory.create_model("azure/gpt-4o")

# Open-source model - no API key required
model = ModelFactory.create_model("meta-llama/Llama-2-7b-hf")
```

### Direct API Configuration Usage

You can also use the `APIConfig` class directly:

```python
from api_config import APIConfig

# Get the appropriate environment variable name for a model
env_var = APIConfig.get_api_key_env_var("openai/gpt-4o")
print(env_var)  # Output: OPENAI_API_KEY

# Get API key with fallback
api_key = APIConfig.get_api_key("anthropic/claude-3-sonnet")

# Get complete provider configuration
config = APIConfig.get_provider_config("azure/gpt-4o")
print(config)
# Output: {
#     'api_key': 'your-azure-key',
#     'model_id': 'azure/gpt-4o',
#     'azure_endpoint': 'https://your-resource.openai.azure.com/',
#     'api_version': '2024-02-15-preview'
# }

# Validate configuration
is_valid, error_msg = APIConfig.validate_config("openai/gpt-4o")
if not is_valid:
    print(f"Configuration error: {error_msg}")
```

## Error Handling

The system provides clear error messages when API keys are missing or configurations are invalid:

### Missing API Key
```
ValueError: No API key found for model 'openai/gpt-4o'.
Please provide via api_key parameter or set OPENAI_API_KEY environment variable.
Supported environment variables: OPENAI_API_KEY, ANTHROPIC_API_KEY, AZURE_OPENAI_API_KEY, GOOGLE_API_KEY, COHERE_API_KEY, HUGGINGFACE_API_KEY
```

### Invalid Azure Configuration
```
ValueError: Configuration validation failed: Azure OpenAI endpoint (AZURE_OPENAI_ENDPOINT) is required for Azure models
```

## Fallback Mechanism

If the specific environment variable for a model is not found, the system will try common fallback variables in this order:
1. `OPENAI_API_KEY`
2. `ANTHROPIC_API_KEY`
3. `API_KEY`

A warning will be logged when using fallback variables.

## Migration from Old System

### Before (Inconsistent)
```python
# Different files used different approaches
self.api_key = api_key or os.environ.get("OPENAI_API_KEY")  # Always OpenAI
```

### After (Unified)
```python
# Automatically detects the correct environment variable
self.api_key = APIConfig.get_api_key(model_id, api_key)
```

## Command Line Usage

All evaluation scripts now support the unified system:

```bash
# OpenAI model
python evaluate_ToF.py --model_name openai/gpt-4o --api_key your-key

# Anthropic model
python evaluate_ToF.py --model_name anthropic/claude-3-sonnet --api_key your-key

# Using environment variables (recommended)
export ANTHROPIC_API_KEY="your-key"
python evaluate_ToF.py --model_name anthropic/claude-3-sonnet
```

## Testing

Run the comprehensive test suite to verify the configuration system:

```bash
python test_api_config.py
```

The test suite covers:
- Environment variable detection for all supported providers
- API key retrieval with fallbacks
- Provider-specific configuration handling
- Integration with model classes
- Error handling scenarios

## Best Practices

1. **Use Environment Variables**: Set API keys as environment variables rather than passing them as parameters
2. **Provider-Specific Variables**: Use the correct environment variable for each provider (e.g., `ANTHROPIC_API_KEY` for Claude models)
3. **Validation**: Always validate configurations before making API calls
4. **Error Handling**: Handle configuration errors gracefully in your applications
5. **Security**: Never commit API keys to version control

## Troubleshooting

### Common Issues

1. **Wrong Environment Variable**: Make sure you're using the correct environment variable for your model provider
2. **Missing Azure Endpoint**: Azure OpenAI models require both `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`
3. **Model Prefix Mismatch**: Ensure your model name starts with a recognized prefix (e.g., `openai/`, `anthropic/`)

### Debug Mode

Enable debug logging to see detailed configuration information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your code here
```

This will show which environment variables are being used and any fallback mechanisms that are triggered.
