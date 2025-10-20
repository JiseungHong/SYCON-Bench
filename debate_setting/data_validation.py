"""
Data validation utilities for SYCON-Bench debate setting.

This module provides comprehensive validation for data files used in the benchmark,
including file existence checks, format validation, and data integrity checks.
"""
import os
import csv
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DataValidator:
    """Comprehensive data validator for benchmark data files."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data validator.

        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

    def validate_file_exists(self, filepath: str) -> None:
        """
        Validate that a file exists and is readable.

        Args:
            filepath: Path to the file to validate

        Raises:
            DataValidationError: If file doesn't exist or isn't readable
        """
        full_path = self.data_dir / filepath
        if not full_path.exists():
            raise DataValidationError(
                f"Required data file not found: {full_path}\n"
                f"Please ensure the file exists in the data directory."
            )

        if not full_path.is_file():
            raise DataValidationError(
                f"Path exists but is not a file: {full_path}\n"
                f"Please ensure this is a valid data file."
            )

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                f.read(1)  # Try to read one character
        except PermissionError:
            raise DataValidationError(
                f"Permission denied reading file: {full_path}\n"
                f"Please check file permissions."
            )
        except UnicodeDecodeError as e:
            raise DataValidationError(
                f"File encoding error in {full_path}: {e}\n"
                f"Please ensure the file is UTF-8 encoded."
            )
        except Exception as e:
            raise DataValidationError(
                f"Error reading file {full_path}: {e}\n"
                f"Please check if the file is corrupted."
            )

    def validate_text_file_format(self, filepath: str, min_lines: int = 1) -> List[str]:
        """
        Validate a text file format and return non-empty lines.

        Args:
            filepath: Path to the text file
            min_lines: Minimum number of non-empty lines required

        Returns:
            List of non-empty lines from the file

        Raises:
            DataValidationError: If file format is invalid
        """
        full_path = self.data_dir / filepath

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            raise DataValidationError(
                f"Error reading text file {full_path}: {e}"
            )

        if len(lines) < min_lines:
            raise DataValidationError(
                f"File {full_path} contains only {len(lines)} non-empty lines, "
                f"but at least {min_lines} are required.\n"
                f"Please check the file content and ensure it has sufficient data."
            )

        # Check for common issues
        empty_line_numbers = []
        for i, line in enumerate(lines, 1):
            if not line or line.isspace():
                empty_line_numbers.append(i)

        if empty_line_numbers:
            self.logger.warning(
                f"File {full_path} contains empty or whitespace-only lines at positions: "
                f"{empty_line_numbers[:5]}{'...' if len(empty_line_numbers) > 5 else ''}"
            )

        return lines

    def validate_csv_file_format(self, filepath: str, required_columns: List[str] = None) -> List[Dict[str, Any]]:
        """
        Validate a CSV file format and return the data.

        Args:
            filepath: Path to the CSV file
            required_columns: List of required column names

        Returns:
            List of dictionaries representing the CSV data

        Raises:
            DataValidationError: If CSV format is invalid
        """
        full_path = self.data_dir / filepath
        required_columns = required_columns or []

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                # Try to detect the CSV format
                sample = f.read(1024)
                f.seek(0)

                # Check if it looks like a CSV
                if ',' not in sample and '\t' not in sample:
                    raise DataValidationError(
                        f"File {full_path} does not appear to be a valid CSV file.\n"
                        f"No comma or tab delimiters found in the first 1024 characters."
                    )

                reader = csv.DictReader(f)
                data = list(reader)

        except csv.Error as e:
            raise DataValidationError(
                f"CSV parsing error in {full_path}: {e}\n"
                f"Please check the CSV format and ensure proper escaping of special characters."
            )
        except Exception as e:
            raise DataValidationError(
                f"Error reading CSV file {full_path}: {e}"
            )

        if not data:
            raise DataValidationError(
                f"CSV file {full_path} is empty or contains no data rows.\n"
                f"Please ensure the file contains valid data."
            )

        # Validate required columns
        if required_columns:
            actual_columns = set(data[0].keys()) if data else set()
            missing_columns = set(required_columns) - actual_columns

            if missing_columns:
                raise DataValidationError(
                    f"CSV file {full_path} is missing required columns: {sorted(missing_columns)}\n"
                    f"Available columns: {sorted(actual_columns)}\n"
                    f"Please ensure all required columns are present."
                )

        return data

    def validate_data_integrity(self, questions: List[str], arguments: List[str]) -> None:
        """
        Validate the integrity of questions and arguments data.

        Args:
            questions: List of questions
            arguments: List of arguments

        Raises:
            DataValidationError: If data integrity issues are found
        """
        # Check length matching
        if len(questions) != len(arguments):
            raise DataValidationError(
                f"Data integrity error: Number of questions ({len(questions)}) "
                f"does not match number of arguments ({len(arguments)}).\n"
                f"Files: questions.txt has {len(questions)} lines, "
                f"arguments.txt has {len(arguments)} lines.\n"
                f"Please ensure both files have the same number of non-empty lines."
            )

        # Check for empty entries
        empty_questions = [i+1 for i, q in enumerate(questions) if not q or q.isspace()]
        empty_arguments = [i+1 for i, a in enumerate(arguments) if not a or a.isspace()]

        if empty_questions:
            raise DataValidationError(
                f"Data integrity error: Empty questions found at line numbers: "
                f"{empty_questions[:10]}{'...' if len(empty_questions) > 10 else ''}\n"
                f"Please ensure all questions contain meaningful content."
            )

        if empty_arguments:
            raise DataValidationError(
                f"Data integrity error: Empty arguments found at line numbers: "
                f"{empty_arguments[:10]}{'...' if len(empty_arguments) > 10 else ''}\n"
                f"Please ensure all arguments contain meaningful content."
            )

        # Check for suspiciously short entries
        short_questions = [(i+1, q) for i, q in enumerate(questions) if len(q.strip()) < 10]
        short_arguments = [(i+1, a) for i, a in enumerate(arguments) if len(a.strip()) < 10]

        if short_questions:
            self.logger.warning(
                f"Found {len(short_questions)} suspiciously short questions (< 10 characters) "
                f"at lines: {[line_num for line_num, _ in short_questions[:5]]}"
                f"{'...' if len(short_questions) > 5 else ''}"
            )

        if short_arguments:
            self.logger.warning(
                f"Found {len(short_arguments)} suspiciously short arguments (< 10 characters) "
                f"at lines: {[line_num for line_num, _ in short_arguments[:5]]}"
                f"{'...' if len(short_arguments) > 5 else ''}"
            )

        # Check for duplicates
        question_duplicates = self._find_duplicates(questions)
        argument_duplicates = self._find_duplicates(arguments)

        if question_duplicates:
            self.logger.warning(
                f"Found {len(question_duplicates)} duplicate questions. "
                f"First few duplicates at lines: {list(question_duplicates.values())[:3]}"
            )

        if argument_duplicates:
            self.logger.warning(
                f"Found {len(argument_duplicates)} duplicate arguments. "
                f"First few duplicates at lines: {list(argument_duplicates.values())[:3]}"
            )

    def _find_duplicates(self, items: List[str]) -> Dict[str, List[int]]:
        """Find duplicate items and their line numbers."""
        seen = {}
        duplicates = {}

        for i, item in enumerate(items):
            item_lower = item.lower().strip()
            if item_lower in seen:
                if item_lower not in duplicates:
                    duplicates[item_lower] = [seen[item_lower] + 1]
                duplicates[item_lower].append(i + 1)
            else:
                seen[item_lower] = i

        return duplicates

    def validate_debate_data(self, questions_file: str = "questions.txt",
                           arguments_file: str = "arguments.txt") -> Tuple[List[str], List[str]]:
        """
        Comprehensive validation of debate setting data files.

        Args:
            questions_file: Name of the questions file
            arguments_file: Name of the arguments file

        Returns:
            Tuple of (questions, arguments) lists

        Raises:
            DataValidationError: If any validation fails
        """
        self.logger.info("Starting comprehensive data validation...")

        # Step 1: Check file existence
        self.logger.info("Checking file existence...")
        self.validate_file_exists(questions_file)
        self.validate_file_exists(arguments_file)

        # Step 2: Validate file formats
        self.logger.info("Validating file formats...")
        questions = self.validate_text_file_format(questions_file, min_lines=1)
        arguments = self.validate_text_file_format(arguments_file, min_lines=1)

        # Step 3: Validate data integrity
        self.logger.info("Validating data integrity...")
        self.validate_data_integrity(questions, arguments)

        self.logger.info(f"Data validation completed successfully. "
                        f"Found {len(questions)} question-argument pairs.")

        return questions, arguments

    def validate_ethical_data(self, data_file: str = "stereoset_intra_user_queries_api_over45.csv",
                            required_columns: List[str] = None) -> List[Dict[str, Any]]:
        """
        Comprehensive validation of ethical setting data file.

        Args:
            data_file: Name of the CSV data file
            required_columns: List of required column names

        Returns:
            List of data dictionaries

        Raises:
            DataValidationError: If any validation fails
        """
        self.logger.info("Starting ethical data validation...")

        # Step 1: Check file existence
        self.logger.info("Checking file existence...")
        self.validate_file_exists(data_file)

        # Step 2: Validate CSV format
        self.logger.info("Validating CSV format...")
        data = self.validate_csv_file_format(data_file, required_columns)

        self.logger.info(f"Ethical data validation completed successfully. "
                        f"Found {len(data)} data entries.")

        return data

    def validate_presupposition_data(self, questions_file: str = "questions.txt") -> List[str]:
        """
        Comprehensive validation of false presuppositions setting data file.

        Args:
            questions_file: Name of the questions file

        Returns:
            List of questions

        Raises:
            DataValidationError: If any validation fails
        """
        self.logger.info("Starting presupposition data validation...")

        # Step 1: Check file existence
        self.logger.info("Checking file existence...")
        self.validate_file_exists(questions_file)

        # Step 2: Validate file format
        self.logger.info("Validating file format...")
        questions = self.validate_text_file_format(questions_file, min_lines=1)

        self.logger.info(f"Presupposition data validation completed successfully. "
                        f"Found {len(questions)} questions.")

        return questions


def validate_data_files(data_dir: str = "data", setting_type: str = "debate") -> Any:
    """
    Convenience function to validate data files for different settings.

    Args:
        data_dir: Directory containing the data files
        setting_type: Type of setting ("debate", "ethical", "presupposition")

    Returns:
        Validated data (format depends on setting type)

    Raises:
        DataValidationError: If validation fails
    """
    validator = DataValidator(data_dir)

    if setting_type == "debate":
        return validator.validate_debate_data()
    elif setting_type == "ethical":
        return validator.validate_ethical_data()
    elif setting_type == "presupposition":
        return validator.validate_presupposition_data()
    else:
        raise ValueError(f"Unknown setting type: {setting_type}")
