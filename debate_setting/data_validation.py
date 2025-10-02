
"""
Data validation module for SYCON-Bench debate setting.

This module provides functions for validating data files used in the benchmark.
"""
import os
from typing import List, Tuple


def validate_data_files(data_dir: str = "data") -> Tuple[List[str], List[str]]:
    """
    Validate and read questions and arguments from data files.

    Args:
        data_dir: Directory containing the data files

    Returns:
        Tuple of (questions, arguments) lists

    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If data files have format issues or mismatched lengths
    """
    questions_file = os.path.join(data_dir, "questions.txt")
    arguments_file = os.path.join(data_dir, "arguments.txt")

    # Check file existence
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    if not os.path.exists(arguments_file):
        raise FileNotFoundError(f"Arguments file not found: {arguments_file}")

    # Read the questions
    try:
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise ValueError(f"Error reading questions file {questions_file}: {str(e)}")

    # Read the arguments
    try:
        with open(arguments_file, "r", encoding="utf-8") as f:
            arguments = [line.strip() for line in f if line.strip()]
    except Exception as e:
        raise ValueError(f"Error reading arguments file {arguments_file}: {str(e)}")

    # Validate data integrity
    if len(questions) != len(arguments):
        raise ValueError(
            f"Data mismatch: questions.txt has {len(questions)} entries "
            f"but arguments.txt has {len(arguments)} entries. "
            f"Files must have the same number of non-empty lines."
        )

    # Check for empty data
    if len(questions) == 0:
        raise ValueError("No valid data found in questions.txt or arguments.txt")

    return questions, arguments


def validate_data_directory(data_dir: str = "data") -> bool:
    """
    Validate data directory without reading the full content.

    Args:
        data_dir: Directory containing the data files

    Returns:
        True if validation passes

    Raises:
        FileNotFoundError: If required data files are missing
        ValueError: If data files have format issues
    """
    questions_file = os.path.join(data_dir, "questions.txt")
    arguments_file = os.path.join(data_dir, "arguments.txt")

    # Check file existence
    if not os.path.exists(questions_file):
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    if not os.path.exists(arguments_file):
        raise FileNotFoundError(f"Arguments file not found: {arguments_file}")

    # Check file readability
    try:
        with open(questions_file, "r", encoding="utf-8") as f:
            pass
    except Exception as e:
        raise ValueError(f"Cannot read questions file {questions_file}: {str(e)}")

    try:
        with open(arguments_file, "r", encoding="utf-8") as f:
            pass
    except Exception as e:
        raise ValueError(f"Cannot read arguments file {arguments_file}: {str(e)}")

    return True

