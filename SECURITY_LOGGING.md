# Secure Logging Practices for SYCON-Bench

This document outlines the secure logging practices implemented in SYCON-Bench to prevent API keys and other sensitive data from being exposed in logs and error messages.

## Overview

SYCON-Bench has been enhanced with secure logging utilities to prevent accidental exposure of sensitive information such as:
- API keys (OpenAI, Anthropic, etc.)
- Authentication tokens
- Passwords and secrets
- Sensitive prompt content

## Security Measures Implemented

### 1. Secure Logging Utility (`secure_logging.py`)

A comprehensive secure logging utility has been implemented with the following features:

#### Key Components:
- **`SecureLogger` class**: Main utility class for secure logging operations
- **Pattern-based sanitization**: Automatically detects and redacts common sensitive data patterns
- **Dictionary sanitization**: Safely handles configuration objects and argument dictionaries
- **Exception sanitization**: Prevents sensitive data exposure in error messages
- **Safe prompt printing**: Allows debugging without exposing sensitive prompt content

#### Sensitive Data Patterns Detected:
- OpenAI API keys (`sk-...`)
- Generic API keys starting with `sk-`
- Bearer tokens
- API key assignments (`api_key=...`)
- Token assignments (`token=...`)
- Password assignments (`password=...`)
- Secret assignments (`secret=...`)

### 2. Models Security Enhancements

All `models.py` files have been updated with:

#### Secure Print Statements:
```python
# Before (VULNERABLE):
print(f"prompt {i}: {prompt}")

# After (SECURE):
SecureLogger.safe_print_prompt(prompt, max_length=100, prefix=f"Prompt {i+1}")
```

#### Secure Exception Handling:
```python
# Before (VULNERABLE):
except Exception as e:
    logging.error(f"Error generating response: {e}")
    return f"ERROR: Failed to generate response: {str(e)}"

# After (SECURE):
except Exception as e:
    SecureLogger.safe_log_error("Error generating response", e)
    safe_error = SecureLogger.create_safe_error_response("Failed to generate response", e)
    return f"ERROR: {safe_error}"
```

### 3. Benchmark Script Security

All `run_benchmark.py` files have been updated to use secure argument logging:

```python
# Before (VULNERABLE - though already partially handled):
log_args = vars(args).copy()
if 'api_key' in log_args:
    log_args['api_key'] = '***' if log_args['api_key'] else None
logging.info(f"Arguments: {log_args}")

# After (SECURE):
from secure_logging import SecureLogger
SecureLogger.safe_log_args(args)
```

## Usage Guidelines

### For Developers

#### 1. Logging Arguments/Configuration
```python
from secure_logging import SecureLogger

# Safe argument logging
SecureLogger.safe_log_args(args, logger)

# Safe dictionary logging
config = {"api_key": "sk-...", "model": "gpt-4"}
safe_config = SecureLogger.sanitize_dict(config)
logging.info(f"Configuration: {safe_config}")
```

#### 2. Error Handling
```python
try:
    # API call or sensitive operation
    result = api_call(api_key=secret_key)
except Exception as e:
    # Safe error logging
    SecureLogger.safe_log_error("API call failed", e, logger)

    # Safe error response
    safe_response = SecureLogger.create_safe_error_response("Operation failed", e)
    return safe_response
```

#### 3. Debug Printing
```python
# Safe prompt/content printing
SecureLogger.safe_print_prompt(prompt_text, max_length=100, prefix="Processing")

# Manual string sanitization
sensitive_text = "Using API key sk-1234567890..."
safe_text = SecureLogger.sanitize_string(sensitive_text)
print(f"Debug: {safe_text}")
```

### For Users

#### Running with Verbose Logging
When using the `--verbose` flag, all sensitive data will be automatically redacted:

```bash
python run_benchmark.py --model_name gpt-4 --api_key sk-your-key --verbose
```

Output will show:
```
Arguments: {'model_name': 'gpt-4', 'api_key': '***', 'verbose': True, ...}
```

## Security Testing

A comprehensive test suite (`test_security.py`) has been implemented to verify:

1. **API Key Redaction**: Ensures API keys are properly redacted in all contexts
2. **Exception Sanitization**: Verifies exception messages don't expose sensitive data
3. **Print Statement Security**: Confirms debug prints are safe
4. **Integration Testing**: Tests real-world scenarios and edge cases

### Running Security Tests
```bash
python test_security.py
```

## Best Practices

### DO:
✅ Use `SecureLogger` utilities for all logging operations
✅ Test with verbose logging to ensure no sensitive data is exposed
✅ Use safe error responses for user-facing error messages
✅ Sanitize any user input that might contain sensitive data before logging

### DON'T:
❌ Log raw exception messages without sanitization
❌ Print full prompts or API responses without sanitization
❌ Include API keys in debug print statements
❌ Log configuration dictionaries without sanitization

## Verification Steps

To verify the security measures are working:

1. **Run with verbose logging**:
   ```bash
   python run_benchmark.py --model_name gpt-4 --api_key your-key --verbose
   ```

2. **Check log outputs**: Ensure no partial API keys appear in logs

3. **Test error scenarios**: Trigger errors and verify no keys appear in error messages

4. **Run security tests**:
   ```bash
   python test_security.py
   ```

## Implementation Details

### Files Modified:
- `/workspace/secure_logging.py` - New secure logging utility
- `/workspace/debate-setting/models.py` - Updated with secure logging
- `/workspace/ethical-setting/models.py` - Updated with secure logging
- `/workspace/false-presuppositions-setting/models.py` - Updated with secure logging
- `/workspace/debate-setting/run_benchmark.py` - Updated argument logging
- `/workspace/ethical-setting/run_benchmark.py` - Updated argument logging
- `/workspace/false-presuppositions-setting/run_benchmark.py` - Updated argument logging

### Security Patterns Addressed:
1. **Log exposure**: API keys redacted in debug logs
2. **Error message exposure**: Exception handling sanitizes sensitive data
3. **Cost estimation logging**: Token usage logs remain safe (no sensitive request details)
4. **Print statement exposure**: Debug prints sanitized

## Future Considerations

1. **Environment Variable Security**: Consider implementing checks for accidentally logged environment variables
2. **Response Content**: Monitor for sensitive data in API response content
3. **File Path Security**: Ensure file paths don't inadvertently expose sensitive information
4. **Third-party Library Logging**: Monitor third-party libraries (like litellm) for potential logging issues

## Contact

For security concerns or questions about these implementations, please review the code and test suite, or create an issue in the repository.
