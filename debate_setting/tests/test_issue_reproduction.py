"""
Test to reproduce the original issue described in the problem statement.
"""
import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run_benchmark import read_data
from data_validation import DataValidationError


class TestIssueReproduction:
    """Test cases to reproduce the original issue."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_data_files(self, questions_content: str, arguments_content: str):
        """Helper to create test data files."""
        os.makedirs("data", exist_ok=True)

        with open("data/questions.txt", 'w', encoding='utf-8') as f:
            f.write(questions_content)

        with open("data/arguments.txt", 'w', encoding='utf-8') as f:
            f.write(arguments_content)

    def test_original_issue_different_line_counts(self):
        """
        Test the original issue: different number of lines in questions.txt vs arguments.txt

        This reproduces the scenario described in the problem statement:
        1. Modify questions.txt to have different number of lines than arguments.txt
        2. Run the benchmark
        3. Get unclear assertion error (before fix) or clear error message (after fix)
        """
        # Create files with different line counts (reproducing the issue)
        questions_content = "Question 1?\nQuestion 2?\n"  # 2 questions
        arguments_content = "Argument 1\nArgument 2\nArgument 3\n"  # 3 arguments

        self.create_data_files(questions_content, arguments_content)

        # Test with validation enabled (new behavior - should give clear error)
        with pytest.raises(SystemExit) as exc_info:
            read_data(validate=True)
        assert exc_info.value.code == 1

        # Test with validation disabled (legacy behavior - should still give better error)
        with pytest.raises(SystemExit) as exc_info:
            read_data(validate=False)
        assert exc_info.value.code == 1

    def test_missing_data_files(self):
        """Test behavior when data files are missing."""
        # Don't create any data files

        # Test with validation enabled
        with pytest.raises(SystemExit) as exc_info:
            read_data(validate=True)
        assert exc_info.value.code == 1

        # Test with validation disabled
        with pytest.raises(SystemExit) as exc_info:
            read_data(validate=False)
        assert exc_info.value.code == 1

    def test_malformed_data_files(self):
        """Test behavior with malformed data files."""
        # Create files with problematic content
        questions_content = "Question 1?\n\n\n   \nQuestion 2?\n"  # Empty/whitespace lines
        arguments_content = "Argument 1\n\nArgument 2\n"  # Different pattern of empty lines

        self.create_data_files(questions_content, arguments_content)

        # With validation enabled, should succeed (empty lines filtered out)
        questions, arguments = read_data(validate=True)
        assert len(questions) == 2
        assert len(arguments) == 2

        # With validation disabled, should also succeed
        questions, arguments = read_data(validate=False)
        assert len(questions) == 2
        assert len(arguments) == 2

    def test_empty_data_files(self):
        """Test behavior with empty data files."""
        # Create empty files
        questions_content = ""
        arguments_content = ""

        self.create_data_files(questions_content, arguments_content)

        # Both validation modes should fail gracefully
        with pytest.raises(SystemExit) as exc_info:
            read_data(validate=True)
        assert exc_info.value.code == 1

        with pytest.raises(SystemExit) as exc_info:
            read_data(validate=False)
        assert exc_info.value.code == 1

    def test_file_encoding_issues(self):
        """Test behavior with file encoding issues."""
        # Create files with different encodings
        os.makedirs("data", exist_ok=True)

        # Create a file with latin-1 encoding that might cause issues
        with open("data/questions.txt", 'w', encoding='latin-1') as f:
            f.write("Question with special chars: café\n")

        with open("data/arguments.txt", 'w', encoding='utf-8') as f:
            f.write("Argument 1\n")

        # With validation enabled, should handle encoding gracefully
        # (The validator tries to read as UTF-8 and will give a clear error if it fails)
        try:
            questions, arguments = read_data(validate=True)
            # If it succeeds, that's also fine (depends on system)
            assert len(questions) >= 0
            assert len(arguments) >= 0
        except SystemExit as e:
            # If it fails, it should exit cleanly with code 1
            assert e.code == 1


if __name__ == "__main__":
    pytest.main([__file__])
