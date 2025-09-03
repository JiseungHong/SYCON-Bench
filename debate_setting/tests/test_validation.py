import os
import sys
import unittest
from unittest.mock import patch, mock_open

# Add the parent directory to the sys.path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_benchmark import validate_data

class TestValidation(unittest.TestCase):
    def setUp(self):
        self.data_dir = "test_data"
        os.makedirs(self.data_dir, exist_ok=True)

    def tearDown(self):
        for f in os.listdir(self.data_dir):
            os.remove(os.path.join(self.data_dir, f))
        os.rmdir(self.data_dir)

    def test_missing_files(self):
        with self.assertLogs(level='ERROR') as cm:
            self.assertFalse(validate_data(self.data_dir))
            self.assertIn("not found", cm.output[0])

    def test_mismatched_lines(self):
        with open(os.path.join(self.data_dir, "questions.txt"), "w") as f:
            f.write("Q1\nQ2\n")
        with open(os.path.join(self.data_dir, "arguments.txt"), "w") as f:
            f.write("A1\n")
        with self.assertLogs(level='ERROR') as cm:
            self.assertFalse(validate_data(self.data_dir))
            self.assertIn("does not match", cm.output[0])

    def test_empty_lines(self):
        with open(os.path.join(self.data_dir, "questions.txt"), "w") as f:
            f.write("Q1\n\nQ3")
        with open(os.path.join(self.data_dir, "arguments.txt"), "w") as f:
            f.write("A1\nA2\n")
        with self.assertLogs(level='WARNING') as cm:
            self.assertTrue(validate_data(self.data_dir))
            self.assertIn("is empty", cm.output[0])

if __name__ == "__main__":
    unittest.main()
