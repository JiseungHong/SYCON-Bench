"""
Data validation utilities for SYCON-Bench false presuppositions setting.

This module provides comprehensive validation for data files used in the false presuppositions benchmark.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'debate_setting'))

from data_validation import DataValidator, DataValidationError, validate_data_files


class PresuppositionDataValidator(DataValidator):
    """Specialized data validator for false presuppositions setting."""

    def validate_presupposition_data(self, questions_file: str = "questions.txt") -> list:
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

        # Step 3: Additional presupposition-specific validation
        self.logger.info("Performing presupposition-specific validation...")
        self._validate_presupposition_content(questions)

        self.logger.info(f"Presupposition data validation completed successfully. "
                        f"Found {len(questions)} questions.")

        return questions

    def _validate_presupposition_content(self, questions: list) -> None:
        """Perform presupposition-specific content validation."""
        if not questions:
            return

        # Check for questions that might not contain presuppositions
        non_presupposition_indicators = [
            "is it true that",
            "do you think",
            "what is",
            "how do",
            "can you",
            "will you"
        ]

        potential_issues = []
        for i, question in enumerate(questions):
            question_lower = question.lower().strip()

            # Check if question starts with common non-presupposition patterns
            for indicator in non_presupposition_indicators:
                if question_lower.startswith(indicator):
                    potential_issues.append((i+1, question[:50] + "..."))
                    break

        if potential_issues:
            self.logger.warning(
                f"Found {len(potential_issues)} questions that might not contain "
                f"false presuppositions. First few examples: {potential_issues[:3]}"
            )

        # Check for questions that are too short to contain meaningful presuppositions
        short_questions = [(i+1, q) for i, q in enumerate(questions) if len(q.strip()) < 20]
        if short_questions:
            self.logger.warning(
                f"Found {len(short_questions)} very short questions (< 20 characters) "
                f"that might not contain meaningful presuppositions"
            )


def validate_presupposition_data_files(data_dir: str = "data") -> list:
    """
    Convenience function to validate false presuppositions setting data files.

    Args:
        data_dir: Directory containing the data files

    Returns:
        Validated questions list

    Raises:
        DataValidationError: If validation fails
    """
    validator = PresuppositionDataValidator(data_dir)
    return validator.validate_presupposition_data()
