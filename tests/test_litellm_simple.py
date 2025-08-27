

"""
Simple test for litellm dependency handling.
"""
import sys
import os

def test_debate_setting_import():
    """Test that debate-setting models can be imported and handle missing litellm."""
    # Add debate-setting to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'debate-setting'))

    # Import the module
    import models

    # Check that completion is either available or set to None
    # This verifies our try/except handling works
    completion_var = getattr(models, 'completion', 'NOT_FOUND')
    print(f"Debate setting completion: {type(completion_var)}")

    # Test creating a ClosedModel
    from models import ClosedModel
    model = ClosedModel(model_id="openai/gpt-4o", api_key="test-key")

    # This should raise ImportError due to missing litellm
    try:
        model.setup()
        print("ERROR: Should have raised ImportError")
        return False
    except ImportError as e:
        if "litellm package is not installed" in str(e):
            print("SUCCESS: Correct ImportError raised for debate setting")
            return True
        else:
            print(f"ERROR: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        return False

def test_ethical_setting_import():
    """Test that ethical-setting models can be imported and handle missing litellm."""
    # Add ethical-setting to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ethical-setting'))

    # Import the module
    import models

    # Check that completion is either available or set to None
    # This verifies our try/except handling works
    completion_var = getattr(models, 'completion', 'NOT_FOUND')
    print(f"Ethical setting completion: {type(completion_var)}")

    # Test creating a ClosedModel
    from models import ClosedModel
    model = ClosedModel(model_id="openai/gpt-4o", api_key="test-key")

    # This should raise ImportError due to missing litellm
    try:
        model.setup()
        print("ERROR: Should have raised ImportError")
        return False
    except ImportError as e:
        if "litellm package is not installed" in str(e):
            print("SUCCESS: Correct ImportError raised for ethical setting")
            return True
        else:
            print(f"ERROR: Wrong error message: {e}")
            return False
    except Exception as e:
        print(f"ERROR: Unexpected exception: {e}")
        return False

if __name__ == '__main__':
    print("Testing litellm dependency handling...")

    # Temporarily make completion None to simulate missing dependency
    import sys
    sys.modules['litellm'] = None

    success1 = test_debate_setting_import()
    success2 = test_ethical_setting_import()

    if success1 and success2:
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)

