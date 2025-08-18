#!/usr/bin/env python3
"""
Demonstration script showing the security fixes implemented in SYCON-Bench.

This script demonstrates how API keys and other sensitive data are now properly
redacted in logs and error messages.
"""

import logging
import sys
from secure_logging import SecureLogger

def demo_before_and_after():
    """Demonstrate the before and after of security fixes."""

    print("🔒 SYCON-Bench Security Fixes Demonstration")
    print("=" * 50)

    # Test API key
    test_api_key = "sk-1234567890abcdef1234567890abcdef1234567890abcdef"

    print("\n1. ARGUMENT LOGGING SECURITY")
    print("-" * 30)

    # Simulate command line arguments
    class MockArgs:
        def __init__(self):
            self.model_name = "gpt-4"
            self.api_key = test_api_key
            self.temperature = 0.7
            self.verbose = True
            self.batch_size = 10

    args = MockArgs()

    print("❌ BEFORE (Vulnerable):")
    print(f"   Arguments: {vars(args)}")

    print("\n✅ AFTER (Secure):")
    print("   ", end="")
    SecureLogger.safe_log_args(args)

    print("\n2. PROMPT PRINTING SECURITY")
    print("-" * 30)

    sensitive_prompt = f"System: You are a helpful assistant. Use API key {test_api_key} for authentication.\nUser: What is the weather like today?"

    print("❌ BEFORE (Vulnerable):")
    print(f"   Processing prompt: {sensitive_prompt[:100]}...")

    print("\n✅ AFTER (Secure):")
    print("   ", end="")
    SecureLogger.safe_print_prompt(sensitive_prompt, max_length=100, prefix="Processing prompt")

    print("\n3. ERROR HANDLING SECURITY")
    print("-" * 30)

    # Simulate an API error
    api_error = Exception(f"Authentication failed: Invalid API key '{test_api_key}' provided")

    print("❌ BEFORE (Vulnerable):")
    print(f"   ERROR: Failed to generate response: {str(api_error)}")

    print("\n✅ AFTER (Secure):")
    safe_error = SecureLogger.create_safe_error_response("Failed to generate response", api_error)
    print(f"   ERROR: {safe_error}")

    print("\n4. LOGGING SECURITY")
    print("-" * 30)

    # Set up logging
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
    logger = logging.getLogger('demo')

    print("❌ BEFORE (Vulnerable):")
    print(f"   DEBUG: API call with key {test_api_key}")

    print("\n✅ AFTER (Secure):")
    print("   ", end="")
    SecureLogger.safe_log_error("API call failed", api_error, logger)

    print("\n5. DICTIONARY SANITIZATION")
    print("-" * 30)

    config = {
        'model_name': 'gpt-4',
        'api_key': test_api_key,
        'temperature': 0.7,
        'secret_token': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9',
        'password': 'super_secret_password123',
        'normal_field': 'this is fine'
    }

    print("❌ BEFORE (Vulnerable):")
    print(f"   Config: {config}")

    print("\n✅ AFTER (Secure):")
    safe_config = SecureLogger.sanitize_dict(config)
    print(f"   Config: {safe_config}")

    print("\n" + "=" * 50)
    print("🎉 SECURITY DEMONSTRATION COMPLETE")
    print("\nKey Security Improvements:")
    print("✅ API keys are automatically redacted in all logs")
    print("✅ Exception messages are sanitized before logging")
    print("✅ Debug prints are safe and don't expose sensitive data")
    print("✅ Configuration logging is secure")
    print("✅ All sensitive patterns are detected and redacted")

    print("\nFiles Updated:")
    print("• /workspace/secure_logging.py - New secure logging utility")
    print("• /workspace/debate-setting/models.py - Updated with secure logging")
    print("• /workspace/ethical-setting/models.py - Updated with secure logging")
    print("• /workspace/false-presuppositions-setting/models.py - Updated with secure logging")
    print("• All run_benchmark.py files - Updated argument logging")

    print("\nTo verify security:")
    print("1. Run: python test_security.py")
    print("2. Use --verbose flag with any benchmark script")
    print("3. Check that no API keys appear in logs or error messages")


if __name__ == "__main__":
    demo_before_and_after()
