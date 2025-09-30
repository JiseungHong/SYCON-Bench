import os
import sys
import unittest
from unittest.mock import patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run_benchmark import validate_data

class TestDataValidation(unittest.TestCase):

    def setUp(self):
        self.test_data_dir = "test_data"
        os.makedirs(self.test_data_dir, exist_ok=True)
        self.questions_file = os.path.join(self.test_data_dir, "questions.txt")
        self.arguments_file = os.path.join(self.test_data_dir, "arguments.txt")

    def tearDown(self):
        if os.path.exists(self.questions_file):
            os.remove(self.questions_file)
        if os.path.exists(self.arguments_file):
            os.remove(self.arguments_file)
        os.rmdir(self.test_data_dir)

    def test_validate_data_success(self):
        with open(self.questions_file, "w") as f:
            f.write("Question 1\n")
            f.write("Question 2\n")
        with open(self.arguments_file, "w") as f:
            f.write("Argument 1\n")
            f.write("Argument 2\n")

        try:
            validate_data(self.questions_file, self.arguments_file)
        except (FileNotFoundError, ValueError) as e:
            self.fail(f"validate_data raised an exception unexpectedly: {e}")

    def test_validate_data_mismatch(self):
        with open(self.questions_file, "w") as f:
            f.write("Question 1\n")
        with open(self.arguments_file, "w") as f:
            f.write("Argument 1\n")
            f.write("Argument 2\n")

        with self.assertRaises(ValueError):
            validate_data(self.questions_file, self.arguments_file)

    def test_validate_data_missing_questions(self):
        with open(self.arguments_file, "w") as f:
            f.write("Argument 1\n")

        with self.assertRaises(FileNotFoundError):
            validate_data(self.questions_file, self.arguments_file)

    def test_validate_data_missing_arguments(self):
        with open(self.questions_file, "w") as f:
            f.write("Question 1\n")

        with self.assertRaises(FileNotFoundError):
            validate_data(self.questions_file, self.arguments_file)

if __name__ == "__main__":
    unittest.main()
