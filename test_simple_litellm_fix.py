#!/usr/bin/env python3
"""
Simple test to verify the litellm dependency fix works correctly.
"""

import sys
import os

def test_litellm_fix():
    """Test the litellm dependency fix"""
    print("Testing litellm dependency fix...")

    # Add the debate_setting directory to Python path
    sys.path.insert(0, '/workspace/debate_setting')

    try:
        # Import the models module
        import models

        # Check that LITELLM_AVAILABLE is properly set
        print(f"LITELLM_AVAILABLE: {models.LITELLM_AVAILABLE}")

        # Test ClosedModel with litellm available
        if models.LITELLM_AVAILABLE:
            print("✓ litellm is available, testing ClosedModel...")
            model = models.ClosedModel(api_key="test_key")
            result = model.setup()
            print(f"✓ ClosedModel.setup() returned: {result}")

            # Test that the error message is clear when no API key is provided
            try:
                model_no_key = models.ClosedModel()
                model_no_key.setup()
                print("ERROR: Should have raised ValueError for missing API key")
                return False
            except ValueError as e:
                if "No API key provided" in str(e):
                    print("✓ Correct error message for missing API key")
                else:
                    print(f"ERROR: Wrong error message: {e}")
                    return False
        else:
            print("litellm is not available, testing error handling...")
            model = models.ClosedModel(api_key="test_key")

            # Test setup() error handling
            try:
                model.setup()
                print("ERROR: Should have raised ImportError")
                return False
            except ImportError as e:
                if "litellm is required for closed-source models" in str(e) and "pip install litellm>=1.0.0" in str(e):
                    print("✓ Correct error message for setup() without litellm")
                else:
                    print(f"ERROR: Wrong error message: {e}")
                    return False

            # Test generate_responses() error handling
            try:
                model.generate_responses([{"role": "user", "content": "test"}])
                print("ERROR: Should have raised ImportError")
                return False
            except ImportError as e:
                if "litellm is required for closed-source models" in str(e) and "pip install litellm>=1.0.0" in str(e):
                    print("✓ Correct error message for generate_responses() without litellm")
                else:
                    print(f"ERROR: Wrong error message: {e}")
                    return False

        # Test that OpenModel is not affected
        open_model = models.OpenModel("test-model")
        print(f"✓ OpenModel created successfully: {open_model.model_name}")

        return True

    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.path.remove('/workspace/debate_setting')


def test_requirements_updated():
    """Test that litellm was added to requirements.txt"""
    print("Testing requirements.txt update...")

    with open('/workspace/requirements.txt', 'r') as f:
        content = f.read()

    if 'litellm>=1.0.0' in content:
        print("✓ litellm>=1.0.0 found in requirements.txt")
        return True
    else:
        print("✗ litellm>=1.0.0 not found in requirements.txt")
        return False


def test_all_settings():
    """Test that all three settings have been updated"""
    print("Testing all settings have been updated...")

    settings = [
        '/workspace/debate_setting/models.py',
        '/workspace/ethical-setting/models.py',
        '/workspace/false-presuppositions-setting/models.py'
    ]

    all_good = True
    for setting_file in settings:
        with open(setting_file, 'r') as f:
            content = f.read()

        # Check for the improved import pattern
        if 'LITELLM_AVAILABLE = True' in content and 'LITELLM_AVAILABLE = False' in content:
            print(f"✓ {setting_file} has been updated with proper litellm handling")
        else:
            print(f"✗ {setting_file} missing proper litellm handling")
            all_good = False

        # Check for error messages in setup method
        if 'litellm is required for closed-source models' in content:
            print(f"✓ {setting_file} has proper error messages")
        else:
            print(f"✗ {setting_file} missing proper error messages")
            all_good = False

    return all_good


if __name__ == "__main__":
    print("Running tests for litellm dependency fix...\n")

    all_passed = True

    # Test requirements.txt update
    if not test_requirements_updated():
        all_passed = False

    print()

    # Test all settings updated
    if not test_all_settings():
        all_passed = False

    print()

    # Test the actual functionality
    if not test_litellm_fix():
        all_passed = False

    if all_passed:
        print("\n🎉 All tests passed! The litellm dependency fix is working correctly.")
        print("\nSummary of changes made:")
        print("1. ✓ Added litellm>=1.0.0 to requirements.txt")
        print("2. ✓ Updated import handling in all three settings")
        print("3. ✓ Added proper error messages with installation instructions")
        print("4. ✓ Added validation in setup() methods")
        print("5. ✓ Added validation in generate_responses() methods")
        print("6. ✓ OpenModel classes are unaffected by litellm availability")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)
