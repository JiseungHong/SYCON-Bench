
# SYCON-Bench Configuration System

This document explains how to configure API keys for different model providers in SYCON-Bench.

## Environment Variables

SYCON-Bench now supports a consistent way to handle API keys for different model providers:

| Model Prefix | Environment Variable | Provider |
|--------------|---------------------|----------|
| `openai/` | `OPENAI_API_KEY` | OpenAI |
| `anthropic/` | `ANTHROPIC_API_KEY` | Anthropic |
| `claude` | `ANTHROPIC_API_KEY` | Anthropic (Claude models) |
| `azure/` | `AZURE_API_KEY` | Azure OpenAI |
| `google/` | `GOOGLE_API_KEY` | Google |
| `mistral/` | `MISTRAL_API_KEY` | Mistral AI |
| `cohere/` | `COHERE_API_KEY` | Cohere |

## Usage Examples

### Setting Environment Variables

```bash
# For OpenAI models
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic/Claude models
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Azure OpenAI models
export AZURE_API_KEY="your-azure-api-key"
```

### Using Models

The system automatically detects which API key to use based on the model identifier:

```python
# OpenAI model - will use OPENAI_API_KEY
model = ModelFactory.create_model("openai/gpt-4o")

# Anthropic model - will use ANTHROPIC_API_KEY
model = ModelFactory.create_model("anthropic/claude-3-opus")

# Claude model - will use ANTHROPIC_API_KEY
model = ModelFactory.create_model("claude-3-sonnet")
```

### Explicit API Key

You can also provide an API key explicitly:

```python
model = ModelFactory.create_model("openai/gpt-4o", api_key="your-api-key")
```

## Backward Compatibility

For backward compatibility, if no specific environment variable is found for a model, the system will fall back to `OPENAI_API_KEY`.

## Error Handling

If no API key is found for a model, a clear error message will be displayed indicating which environment variable should be set:

```
ValueError: No API key provided for model 'anthropic/claude-3-opus'. Please provide via api_key parameter or set ANTHROPIC_API_KEY environment variable.
```

## Adding New Providers

To add support for new providers, update the `API_KEY_ENV_MAP` in `config.py`:

```python
API_KEY_ENV_MAP = {
    "openai/": "OPENAI_API_KEY",
    "anthropic/": "ANTHROPIC_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    # Add new providers here
    "newprovider/": "NEWPROVIDER_API_KEY",
}
```
