
"""
Tests for data validation in the debate setting benchmark.
"""
import os
import pytest
import tempfile
import shutil
import sys
from unittest.mock import patch, MagicMock

# Mock the ModelFactory import before importing run_benchmark
with patch.dict('sys.modules', {'models': MagicMock()}):
    # Add the debate_setting directory to the path
    sys.path.insert(0, '/workspace/debate_setting')

    # Import the function we want to test
    from run_benchmark import read_data


class TestDataValidation:
    """Test data validation functionality."""

    def test_read_data_valid_files(self):
        """Test reading valid data files."""
        # Create temporary directory with valid data
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid questions file
            questions_file = os.path.join(temp_dir, "questions.txt")
            with open(questions_file, "w") as f:
                f.write("Question 1\nQuestion 2\nQuestion 3\n")

            # Create valid arguments file
            arguments_file = os.path.join(temp_dir, "arguments.txt")
            with open(arguments_file, "w") as f:
                f.write("Argument 1\nArgument 2\nArgument 3\n")

            # Test reading valid data
            questions, arguments = read_data(temp_dir)

            assert len(questions) == 3
            assert len(arguments) == 3
            assert questions == ["Question 1", "Question 2", "Question 3"]
            assert arguments == ["Argument 1", "Argument 2", "Argument 3"]

    def test_read_data_missing_directory(self):
        """Test error when data directory doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Data directory 'nonexistent_dir' does not exist"):
            read_data("nonexistent_dir")

    def test_read_data_missing_questions_file(self):
        """Test error when questions file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only arguments file
            arguments_file = os.path.join(temp_dir, "arguments.txt")
            with open(arguments_file, "w") as f:
                f.write("Argument 1\n")

            with pytest.raises(FileNotFoundError, match="Questions file"):
                read_data(temp_dir)

    def test_read_data_missing_arguments_file(self):
        """Test error when arguments file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only questions file
            questions_file = os.path.join(temp_dir, "questions.txt")
            with open(questions_file, "w") as f:
                f.write("Question 1\n")

            with pytest.raises(FileNotFoundError, match="Arguments file"):
                read_data(temp_dir)

    def test_read_data_empty_questions_file(self):
        """Test error when questions file is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty questions file
            questions_file = os.path.join(temp_dir, "questions.txt")
            with open(questions_file, "w") as f:
                f.write("")

            # Create valid arguments file
            arguments_file = os.path.join(temp_dir, "arguments.txt")
            with open(arguments_file, "w") as f:
                f.write("Argument 1\n")

            with pytest.raises(ValueError, match="is empty or contains no valid data"):
                read_data(temp_dir)

    def test_read_data_empty_arguments_file(self):
        """Test error when arguments file is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid questions file
            questions_file = os.path.join(temp_dir, "questions.txt")
            with open(questions_file, "w") as f:
                f.write("Question 1\n")

            # Create empty arguments file
            arguments_file = os.path.join(temp_dir, "arguments.txt")
            with open(arguments_file, "w") as f:
                f.write("")

            with pytest.raises(ValueError, match="is empty or contains no valid data"):
                read_data(temp_dir)

    def test_read_data_mismatched_lengths(self):
        """Test error when question and argument counts don't match."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create questions file with 2 items
            questions_file = os.path.join(temp_dir, "questions.txt")
            with open(questions_file, "w") as f:
                f.write("Question 1\nQuestion 2\n")

            # Create arguments file with 3 items
            arguments_file = os.path.join(temp_dir, "arguments.txt")
            with open(arguments_file, "w") as f:
                f.write("Argument 1\nArgument 2\nArgument 3\n")

            with pytest.raises(ValueError, match="Number of questions.*does not match number of arguments"):
                read_data(temp_dir)

    def test_read_data_with_blank_lines(self):
        """Test that blank lines are properly filtered out."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create questions file with blank lines
            questions_file = os.path.join(temp_dir, "questions.txt")
            with open(questions_file, "w") as f:
                f.write("Question 1\n\nQuestion 2\n  \nQuestion 3\n")

            # Create arguments file with blank lines
            arguments_file = os.path.join(temp_dir, "arguments.txt")
            with open(arguments_file, "w") as f:
                f.write("Argument 1\n\nArgument 2\n  \nArgument 3\n")

            questions, arguments = read_data(temp_dir)

            assert len(questions) == 3
            assert len(arguments) == 3
            assert questions == ["Question 1", "Question 2", "Question 3"]
            assert arguments == ["Argument 1", "Argument 2", "Argument 3"]
