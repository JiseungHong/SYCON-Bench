

#!/usr/bin/env python3
"""
Test script to reproduce the data validation issue by copying the read_data function.
"""
import tempfile
import os

def read_data_original(data_dir="data"):
    """Original read_data function from run_benchmark.py"""
    # Read the questions
    with open(f"{data_dir}/questions.txt", "r") as f:
        questions = [line.strip() for line in f if line.strip()]

    # Read the arguments
    with open(f"{data_dir}/arguments.txt", "r") as f:
        arguments = [line.strip() for line in f if line.strip()]

    assert len(questions) == len(arguments), "Number of questions must match number of arguments"
    return questions, arguments

def read_data_improved(data_dir="data"):
    """Improved read_data function with better validation"""
    questions_file = os.path.join(data_dir, "questions.txt")
    arguments_file = os.path.join(data_dir, "arguments.txt")

    # Check if files exist
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    if not os.path.exists(arguments_file):
        raise FileNotFoundError(f"Arguments file not found: {arguments_file}")

    # Check if files are readable
    if not os.access(questions_file, os.R_OK):
        raise PermissionError(f"Cannot read questions file: {questions_file}")

    if not os.access(arguments_file, os.R_OK):
        raise PermissionError(f"Cannot read arguments file: {arguments_file}")

    # Read the questions
    try:
        with open(questions_file, "r") as f:
            questions = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise IOError(f"Error reading questions file {questions_file}: {e}")

    # Read the arguments
    try:
        with open(arguments_file, "r") as f:
            arguments = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise IOError(f"Error reading arguments file {arguments_file}: {e}")

    # Validate that we have the same number of questions and arguments
    if len(questions) != len(arguments):
        raise ValueError(
            f"Mismatch between questions and arguments: "
            f"found {len(questions)} questions but {len(arguments)} arguments. "
            f"Files {questions_file} and {arguments_file} must have the same number of non-empty lines."
        )

    return questions, arguments

def test_original_behavior():
    """Test the original behavior with mismatched files."""
    print("Testing original behavior with mismatched files...")

    # Create temporary files with mismatched line counts
    with tempfile.TemporaryDirectory() as tmpdir:
        questions_file = os.path.join(tmpdir, "questions.txt")
        arguments_file = os.path.join(tmpdir, "arguments.txt")

        # Write 3 questions but only 2 arguments
        with open(questions_file, "w") as f:
            f.write("Question 1\nQuestion 2\nQuestion 3\n")

        with open(arguments_file, "w") as f:
            f.write("Argument 1\nArgument 2\n")

        print(f"Created test files in {tmpdir}")

        try:
            questions, arguments = read_data_original(tmpdir)
            print(f"Success: Got {len(questions)} questions and {len(arguments)} arguments")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")

def test_improved_behavior():
    """Test the improved behavior with mismatched files."""
    print("\nTesting improved behavior with mismatched files...")

    # Create temporary files with mismatched line counts
    with tempfile.TemporaryDirectory() as tmpdir:
        questions_file = os.path.join(tmpdir, "questions.txt")
        arguments_file = os.path.join(tmpdir, "arguments.txt")

        # Write 3 questions but only 2 arguments
        with open(questions_file, "w") as f:
            f.write("Question 1\nQuestion 2\nQuestion 3\n")

        with open(arguments_file, "w") as f:
            f.write("Argument 1\nArgument 2\n")

        print(f"Created test files in {tmpdir}")

        try:
            questions, arguments = read_data_improved(tmpdir)
            print(f"Success: Got {len(questions)} questions and {len(arguments)} arguments")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")

def test_improved_behavior_missing_files():
    """Test the improved behavior with missing files."""
    print("\nTesting improved behavior with missing files...")

    # Create temporary directory with no files
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Testing in empty directory: {tmpdir}")

        try:
            questions, arguments = read_data_improved(tmpdir)
            print(f"Success: Got {len(questions)} questions and {len(arguments)} arguments")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_original_behavior()
    test_improved_behavior()
    test_improved_behavior_missing_files()

