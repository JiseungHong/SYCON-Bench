
"""
Tests for data validation functionality in run_benchmark.py
"""
import os
import tempfile
import pytest
from unittest.mock import patch
import sys

# Add the debate_setting directory to the path so we can import run_benchmark
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from run_benchmark import read_data


def test_read_data_normal_case():
    """Test that read_data works with normal data files."""
    questions, arguments = read_data("data")
    assert len(questions) == len(arguments)
    assert len(questions) > 0


def test_read_data_mismatched_files():
    """Test that read_data raises appropriate error when files have different line counts."""
    # Create temporary files with mismatched line counts
    with tempfile.TemporaryDirectory() as tmpdir:
        questions_file = os.path.join(tmpdir, "questions.txt")
        arguments_file = os.path.join(tmpdir, "arguments.txt")

        # Write 3 questions but only 2 arguments
        with open(questions_file, "w") as f:
            f.write("Question 1\nQuestion 2\nQuestion 3\n")

        with open(arguments_file, "w") as f:
            f.write("Argument 1\nArgument 2\n")

        # This should raise an exception with a clear message
        with pytest.raises(Exception) as exc_info:
            with patch('run_benchmark.DATA_DIR', tmpdir):
                read_data(tmpdir)

        # Check that the error message is informative
        error_message = str(exc_info.value)
        assert "mismatch" in error_message.lower() or "match" in error_message.lower()


def test_read_data_missing_files():
    """Test that read_data handles missing files appropriately."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Don't create any files
        with pytest.raises(Exception) as exc_info:
            read_data(tmpdir)

        # Check that the error mentions file not found
        error_message = str(exc_info.value)
        assert "not found" in error_message.lower() or "no such file" in error_message.lower()


def test_read_data_empty_files():
    """Test that read_data handles empty files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        questions_file = os.path.join(tmpdir, "questions.txt")
        arguments_file = os.path.join(tmpdir, "arguments.txt")

        # Create empty files
        with open(questions_file, "w") as f:
            f.write("")

        with open(arguments_file, "w") as f:
            f.write("")

        questions, arguments = read_data(tmpdir)
        assert len(questions) == 0
        assert len(arguments) == 0


def test_read_data_with_empty_lines():
    """Test that read_data properly handles empty lines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        questions_file = os.path.join(tmpdir, "questions.txt")
        arguments_file = os.path.join(tmpdir, "arguments.txt")

        # Write files with empty lines
        with open(questions_file, "w") as f:
            f.write("Question 1\n\nQuestion 2\n\n")

        with open(arguments_file, "w") as f:
            f.write("Argument 1\n\nArgument 2\n\n")

        questions, arguments = read_data(tmpdir)
        # Empty lines should be filtered out
        assert len(questions) == 2
        assert len(arguments) == 2
        assert "Question 1" in questions
        assert "Question 2" in questions
        assert "Argument 1" in arguments
        assert "Argument 2" in arguments
