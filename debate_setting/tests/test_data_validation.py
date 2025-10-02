
"""
Tests for data validation functionality.
"""
import os
import tempfile
import pytest
import sys

# Add the debate_setting directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_validation import validate_data_files, validate_data_directory

def test_validate_data_files_success():
    """Test that validate_data_files works correctly with valid data files."""
    questions, arguments = validate_data_files("debate_setting/data")
    assert len(questions) == 100
    assert len(arguments) == 100
    assert isinstance(questions, list)
    assert isinstance(arguments, list)

def test_validate_data_files_file_not_found_questions():
    """Test that validate_data_files raises appropriate error when questions file doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create only arguments file
        arguments_file = os.path.join(temp_dir, "arguments.txt")
        with open(arguments_file, "w") as f:
            f.write("Argument 1\nArgument 2\n")

        with pytest.raises(FileNotFoundError, match="questions.txt"):
            validate_data_files(temp_dir)

def test_validate_data_files_file_not_found_arguments():
    """Test that validate_data_files raises appropriate error when arguments file doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create only questions file
        questions_file = os.path.join(temp_dir, "questions.txt")
        with open(questions_file, "w") as f:
            f.write("Question 1\nQuestion 2\n")

        with pytest.raises(FileNotFoundError, match="arguments.txt"):
            validate_data_files(temp_dir)

def test_validate_data_files_mismatched_lengths():
    """Test that validate_data_files raises appropriate error when files have different lengths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        questions_file = os.path.join(temp_dir, "questions.txt")
        arguments_file = os.path.join(temp_dir, "arguments.txt")

        # Write different number of lines to each file
        with open(questions_file, "w") as f:
            f.write("Question 1\nQuestion 2\nQuestion 3\n")

        with open(arguments_file, "w") as f:
            f.write("Argument 1\nArgument 2\n")

        # Test that the function raises an appropriate error
        with pytest.raises(ValueError, match="Data mismatch"):
            validate_data_files(temp_dir)

def test_validate_data_files_empty_files():
    """Test that validate_data_files handles empty files appropriately."""
    with tempfile.TemporaryDirectory() as temp_dir:
        questions_file = os.path.join(temp_dir, "questions.txt")
        arguments_file = os.path.join(temp_dir, "arguments.txt")

        # Write empty files
        with open(questions_file, "w") as f:
            f.write("")

        with open(arguments_file, "w") as f:
            f.write("")

        # Test that the function raises an appropriate error
        with pytest.raises(ValueError, match="No valid data"):
            validate_data_files(temp_dir)

def test_validate_data_files_empty_lines_filtered():
    """Test that validate_data_files filters out empty lines."""
    with tempfile.TemporaryDirectory() as temp_dir:
        questions_file = os.path.join(temp_dir, "questions.txt")
        arguments_file = os.path.join(temp_dir, "arguments.txt")

        # Write files with empty lines
        with open(questions_file, "w") as f:
            f.write("Question 1\n\nQuestion 2\n\n")

        with open(arguments_file, "w") as f:
            f.write("Argument 1\n\nArgument 2\n\n")

        # Test that empty lines are filtered out
        questions, arguments = validate_data_files(temp_dir)
        assert len(questions) == 2
        assert len(arguments) == 2
        assert "Question 1" in questions
        assert "Question 2" in questions
        assert "Argument 1" in arguments
        assert "Argument 2" in arguments

def test_validate_data_directory_success():
    """Test that validate_data_directory works correctly with valid data files."""
    result = validate_data_directory("debate_setting/data")
    assert result is True

def test_validate_data_directory_file_not_found():
    """Test that validate_data_directory raises appropriate error when files don't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Don't create any files
        with pytest.raises(FileNotFoundError, match="questions.txt"):
            validate_data_directory(temp_dir)
