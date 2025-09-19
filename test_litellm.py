import unittest
import os
import sys

# Add the 'debate_setting' and 'ethical-setting' directories to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'debate_setting')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'ethical-setting')))

class TestLiteLLM(unittest.TestCase):
    def test_debate_setting_import(self):
        try:
            from debate_setting import models
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import debate_setting.models: {e}")

    def test_ethical_setting_import(self):
        try:
            from ethical_setting import models
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import ethical_setting.models: {e}")

    def test_debate_setting_closed_model_setup(self):
        from debate_setting.models import ClosedModel
        # Test without API key
        with self.assertRaises(ValueError):
            model = ClosedModel()
            model.setup()
        # Test with API key
        os.environ["OPENAI_API_KEY"] = "test"
        model = ClosedModel()
        self.assertTrue(model.setup())
        del os.environ["OPENAI_API_KEY"]

    def test_ethical_setting_closed_model_setup(self):
        from ethical_setting.models import ClosedModel
        # Test without API key
        with self.assertRaises(ValueError):
            model = ClosedModel()
            model.setup()
        # Test with API key
        os.environ["OPENAI_API_KEY"] = "test"
        model = ClosedModel()
        self.assertTrue(model.setup())
        del os.environ["OPENAI_API_KEY"]

if __name__ == '__main__':
    unittest.main()
