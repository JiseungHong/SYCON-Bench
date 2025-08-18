#!/usr/bin/env python3
"""
Integration test to demonstrate the litellm dependency fix.

This script simulates the original problem scenario and shows that
the fix provides clear error messages when litellm is not available.
"""
import sys
import os
import tempfile
import importlib.util

def test_litellm_error_handling():
    """Test that the error handling works correctly."""
    print("Testing litellm error handling...")

    # Create a minimal test version of the ClosedModel class
    test_code = '''
# Simulate the import handling from the actual models.py
try:
    from litellm import completion
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    completion = None

class ClosedModel:
    def __init__(self, model_id="openai/gpt-4o", api_key=None):
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def setup(self):
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is required for closed-source models but is not installed. "
                "Please install it with: pip install litellm>=1.0.0"
            )
        if self.api_key is None:
            raise ValueError("No API key provided. Please provide via api_key parameter or set OPENAI_API_KEY environment variable.")
        return True

import os
'''

    # Write the test code to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_code)
        temp_module_path = f.name

    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("test_models", temp_module_path)
        test_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_models)

        print(f"LITELLM_AVAILABLE: {test_models.LITELLM_AVAILABLE}")
        print(f"completion function: {test_models.completion}")

        # Test with API key
        print("\nTrying to create ClosedModel with API key...")
        closed_model = test_models.ClosedModel(model_id="openai/gpt-4o", api_key="test_key")

        print("Calling setup()...")
        try:
            result = closed_model.setup()
            print(f"Setup successful: {result}")
            print("✅ litellm is available and setup succeeded")
        except ImportError as e:
            print(f"❌ ImportError (expected if litellm not installed): {e}")
            print("✅ Clear error message provided with installation instructions")

            # Verify the error message contains the expected content
            error_msg = str(e)
            if "litellm is required" in error_msg and "pip install litellm>=1.0.0" in error_msg:
                print("✅ Error message contains proper instructions")
            else:
                print("❌ Error message missing proper instructions")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

        # Test without API key
        print("\nTrying to create ClosedModel without API key...")
        closed_model_no_key = test_models.ClosedModel(model_id="openai/gpt-4o")

        print("Calling setup()...")
        try:
            result = closed_model_no_key.setup()
            print(f"Setup successful: {result}")
        except ImportError as e:
            print(f"❌ ImportError (expected if litellm not installed): {e}")
            print("✅ litellm check happens before API key check")
        except ValueError as e:
            print(f"❌ ValueError (expected if litellm available but no API key): {e}")
            print("✅ API key validation works when litellm is available")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    finally:
        # Clean up the temporary file
        os.unlink(temp_module_path)

def test_requirements_file():
    """Test that requirements.txt includes litellm."""
    print("\nTesting requirements.txt...")

    requirements_path = '/workspace/requirements.txt'
    with open(requirements_path, 'r') as f:
        content = f.read()

    if 'litellm>=1.0.0' in content:
        print("✅ litellm>=1.0.0 found in requirements.txt")
    else:
        print("❌ litellm>=1.0.0 not found in requirements.txt")

def test_models_files():
    """Test that all models.py files have the proper error handling."""
    print("\nTesting models.py files...")

    models_files = [
        '/workspace/debate-setting/models.py',
        '/workspace/ethical-setting/models.py',
        '/workspace/false-presuppositions-setting/models.py'
    ]

    for models_file in models_files:
        print(f"\nChecking {models_file}...")

        if not os.path.exists(models_file):
            print(f"❌ {models_file} does not exist")
            continue

        with open(models_file, 'r') as f:
            content = f.read()

        # Check for proper import handling
        checks = [
            ('LITELLM_AVAILABLE = True', 'Sets LITELLM_AVAILABLE = True on successful import'),
            ('LITELLM_AVAILABLE = False', 'Sets LITELLM_AVAILABLE = False on ImportError'),
            ('completion = None', 'Sets completion = None on ImportError'),
            ('if not LITELLM_AVAILABLE:', 'Checks LITELLM_AVAILABLE in setup method'),
            ('litellm is required for closed-source models', 'Has proper error message'),
            ('pip install litellm>=1.0.0', 'Provides installation instructions')
        ]

        for check_text, description in checks:
            if check_text in content:
                print(f"  ✅ {description}")
            else:
                print(f"  ❌ {description}")

if __name__ == '__main__':
    print("=" * 60)
    print("SYCON-Bench litellm Dependency Fix Integration Test")
    print("=" * 60)

    test_litellm_error_handling()
    test_requirements_file()
    test_models_files()

    print("\n" + "=" * 60)
    print("Integration test completed!")
    print("=" * 60)
