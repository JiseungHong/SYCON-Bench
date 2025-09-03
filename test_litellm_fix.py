#!/usr/bin/env python3
"""
Integration test to verify the litellm dependency fix works correctly.
This test simulates the scenario described in the issue.
"""

import sys
import os
import subprocess
import tempfile

def test_without_litellm():
    """Test that the error message is clear when litellm is not available"""
    print("Testing behavior without litellm...")

    # Create a test script that tries to use ClosedModel without litellm
    test_script = '''
import sys
sys.path.insert(0, '/workspace/debate_setting')

# Mock the litellm import to fail
import sys
from unittest.mock import patch

with patch.dict('sys.modules', {'litellm': None}):
    with patch('builtins.__import__', side_effect=lambda name, *args, **kwargs:
               __import__(name, *args, **kwargs) if name != 'litellm' else exec('raise ImportError("No module named litellm")')):

        # Now try to import and use the models
        try:
            # Temporarily fix the relative import issue for testing
            import models
            models.ModelRegistry = type('MockRegistry', (), {
                'get_model_family': lambda x: 'unknown',
                'get_quantization_config': lambda x: {}
            })()

            # Test ClosedModel setup without litellm
            model = models.ClosedModel(api_key="test_key")
            try:
                model.setup()
                print("ERROR: Expected ImportError was not raised!")
                sys.exit(1)
            except ImportError as e:
                if "litellm is required for closed-source models" in str(e):
                    print("✓ Correct error message for setup() without litellm")
                else:
                    print(f"ERROR: Wrong error message: {e}")
                    sys.exit(1)

            # Test generate_responses without litellm
            try:
                model.generate_responses([{"role": "user", "content": "test"}])
                print("ERROR: Expected ImportError was not raised!")
                sys.exit(1)
            except ImportError as e:
                if "litellm is required for closed-source models" in str(e):
                    print("✓ Correct error message for generate_responses() without litellm")
                else:
                    print(f"ERROR: Wrong error message: {e}")
                    sys.exit(1)

        except Exception as e:
            print(f"ERROR: Unexpected error: {e}")
            sys.exit(1)

print("✓ All tests passed - litellm dependency handling works correctly!")
'''

    # Write the test script to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_script = f.name

    try:
        # Run the test script
        result = subprocess.run([sys.executable, temp_script],
                              capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✓ Test passed:", result.stdout.strip())
            return True
        else:
            print("✗ Test failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    finally:
        # Clean up
        os.unlink(temp_script)


def test_with_litellm():
    """Test that the code works when litellm is available"""
    print("\nTesting behavior with litellm available...")

    # Install litellm for this test
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'litellm>=1.0.0'],
                      check=True, capture_output=True)
        print("✓ litellm installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install litellm: {e}")
        return False

    # Test that ClosedModel works with litellm available
    test_script = '''
import sys
sys.path.insert(0, '/workspace/debate_setting')

try:
    import models
    models.ModelRegistry = type('MockRegistry', (), {
        'get_model_family': lambda x: 'unknown',
        'get_quantization_config': lambda x: {}
    })()

    # Test ClosedModel setup with litellm available
    model = models.ClosedModel(api_key="test_key")
    result = model.setup()
    if result is True:
        print("✓ ClosedModel.setup() works with litellm available")
    else:
        print("ERROR: setup() should return True")
        sys.exit(1)

except Exception as e:
    print(f"ERROR: Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✓ ClosedModel works correctly with litellm available!")
'''

    # Write the test script to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        temp_script = f.name

    try:
        # Run the test script
        result = subprocess.run([sys.executable, temp_script],
                              capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print("✓ Test passed:", result.stdout.strip())
            return True
        else:
            print("✗ Test failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
    finally:
        # Clean up
        os.unlink(temp_script)


def test_requirements_updated():
    """Test that litellm was added to requirements.txt"""
    print("\nTesting requirements.txt update...")

    with open('/workspace/requirements.txt', 'r') as f:
        content = f.read()

    if 'litellm>=1.0.0' in content:
        print("✓ litellm>=1.0.0 found in requirements.txt")
        return True
    else:
        print("✗ litellm>=1.0.0 not found in requirements.txt")
        return False


if __name__ == "__main__":
    print("Running integration tests for litellm dependency fix...\n")

    all_passed = True

    # Test requirements.txt update
    if not test_requirements_updated():
        all_passed = False

    # Test behavior without litellm
    if not test_without_litellm():
        all_passed = False

    # Test behavior with litellm
    if not test_with_litellm():
        all_passed = False

    if all_passed:
        print("\n🎉 All integration tests passed! The litellm dependency fix is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        sys.exit(1)
