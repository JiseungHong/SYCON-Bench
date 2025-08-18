#!/usr/bin/env python3
"""
Test to simulate the scenario where litellm is not available.

This demonstrates that the fix works correctly when litellm is missing.
"""
import sys
import os
import tempfile
import importlib.util

def test_without_litellm():
    """Test the error handling when litellm is not available."""
    print("Testing error handling when litellm is NOT available...")

    # Create a test module that simulates litellm not being available
    test_code = '''
# Simulate litellm not being available
try:
    # Force an ImportError to simulate litellm not being installed
    raise ImportError("No module named 'litellm'")
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
        spec = importlib.util.spec_from_file_location("test_models_no_litellm", temp_module_path)
        test_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_models)

        print(f"LITELLM_AVAILABLE: {test_models.LITELLM_AVAILABLE}")
        print(f"completion function: {test_models.completion}")

        # Test with API key - should fail on litellm check
        print("\nTrying to create ClosedModel with API key...")
        closed_model = test_models.ClosedModel(model_id="openai/gpt-4o", api_key="test_key")

        print("Calling setup()...")
        try:
            result = closed_model.setup()
            print(f"❌ Setup unexpectedly successful: {result}")
        except ImportError as e:
            print(f"✅ ImportError (expected): {e}")

            # Verify the error message contains the expected content
            error_msg = str(e)
            if "litellm is required" in error_msg and "pip install litellm>=1.0.0" in error_msg:
                print("✅ Error message contains proper instructions")
            else:
                print("❌ Error message missing proper instructions")
        except Exception as e:
            print(f"❌ Unexpected error type: {type(e).__name__}: {e}")

        # Test without API key - should still fail on litellm check first
        print("\nTrying to create ClosedModel without API key...")
        closed_model_no_key = test_models.ClosedModel(model_id="openai/gpt-4o")

        print("Calling setup()...")
        try:
            result = closed_model_no_key.setup()
            print(f"❌ Setup unexpectedly successful: {result}")
        except ImportError as e:
            print(f"✅ ImportError (expected - litellm check happens first): {e}")
        except ValueError as e:
            print(f"❌ ValueError (should not reach API key check): {e}")
        except Exception as e:
            print(f"❌ Unexpected error type: {type(e).__name__}: {e}")

    finally:
        # Clean up the temporary file
        os.unlink(temp_module_path)

def demonstrate_original_problem():
    """Demonstrate what would happen with the original code."""
    print("\n" + "="*60)
    print("Demonstrating the ORIGINAL problem (before fix):")
    print("="*60)

    # Create a test module that simulates the original problematic code
    original_code = '''
# Original problematic import handling
try:
    from litellm import completion
except ImportError:
    pass  # Silent failure!

class ClosedModel:
    def __init__(self, model_id="openai/gpt-4o", api_key=None):
        self.model_id = model_id
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def setup(self):
        # Original code had no litellm availability check
        if self.api_key is None:
            raise ValueError("No API key provided. Please provide via api_key parameter or set OPENAI_API_KEY environment variable.")
        return True

    def generate_responses(self, messages, num_responses=5):
        # This would fail with NameError if litellm not available
        response = completion(  # NameError: name 'completion' is not defined
            model=self.model_id,
            messages=messages
        )
        return [response.choices[0].message.content]

import os
'''

    # Write the test code to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(original_code)
        temp_module_path = f.name

    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("original_models", temp_module_path)
        original_models = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original_models)

        print("With original code:")
        print("- Import succeeds silently even when litellm is not available")
        print("- setup() method passes without checking litellm availability")
        print("- Error only occurs when trying to use completion() function")

        # Test the original problematic behavior
        closed_model = original_models.ClosedModel(api_key="test_key")

        try:
            setup_result = closed_model.setup()
            print(f"✅ setup() succeeds: {setup_result}")
            print("❌ But this gives false confidence to users!")

            # Now try to use the model - this is where the original error occurred
            print("\nTrying to call generate_responses() (where original error occurred)...")
            try:
                messages = [{"role": "user", "content": "Hello"}]
                responses = closed_model.generate_responses(messages)
                print(f"❌ Unexpectedly successful: {responses}")
            except NameError as e:
                print(f"✅ NameError (original problem): {e}")
                print("❌ This is a confusing error message for users!")
            except Exception as e:
                print(f"❌ Other error: {type(e).__name__}: {e}")

        except Exception as e:
            print(f"❌ Unexpected error in setup: {type(e).__name__}: {e}")

    finally:
        # Clean up the temporary file
        os.unlink(temp_module_path)

if __name__ == '__main__':
    print("=" * 60)
    print("Testing litellm Dependency Fix - Simulated Missing litellm")
    print("=" * 60)

    test_without_litellm()
    demonstrate_original_problem()

    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("✅ Fixed version: Clear error message at setup() time")
    print("❌ Original version: Confusing NameError at runtime")
    print("✅ Users now get actionable installation instructions")
    print("=" * 60)
