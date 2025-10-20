"""
Data validation utilities for SYCON-Bench ethical setting.

This module provides comprehensive validation for data files used in the ethical benchmark.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'debate_setting'))

from data_validation import DataValidator, DataValidationError, validate_data_files


class EthicalDataValidator(DataValidator):
    """Specialized data validator for ethical setting."""

    def validate_ethical_data(self, data_file: str = "stereoset_intra_user_queries_api_over45.csv",
                            required_columns: list = None) -> list:
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
        if required_columns is None:
            # Common columns expected in ethical data
            required_columns = []  # Will be determined from actual file

        self.logger.info("Starting ethical data validation...")

        # Step 1: Check file existence
        self.logger.info("Checking file existence...")
        self.validate_file_exists(data_file)

        # Step 2: Validate CSV format
        self.logger.info("Validating CSV format...")
        data = self.validate_csv_file_format(data_file, required_columns)

        # Step 3: Additional ethical-specific validation
        self.logger.info("Performing ethical-specific validation...")
        self._validate_ethical_content(data)

        self.logger.info(f"Ethical data validation completed successfully. "
                        f"Found {len(data)} data entries.")

        return data

    def _validate_ethical_content(self, data: list) -> None:
        """Perform ethical-specific content validation."""
        if not data:
            return

        # Check for required fields in ethical data
        sample_row = data[0]
        expected_fields = ['query', 'response', 'bias_type']  # Common ethical fields

        missing_fields = []
        for field in expected_fields:
            if field not in sample_row:
                # Check for similar field names
                similar_fields = [k for k in sample_row.keys() if field.lower() in k.lower()]
                if similar_fields:
                    self.logger.warning(f"Field '{field}' not found, but similar fields exist: {similar_fields}")
                else:
                    missing_fields.append(field)

        if missing_fields:
            self.logger.warning(f"Some expected ethical fields are missing: {missing_fields}")

        # Check for empty or suspicious entries
        empty_entries = []
        for i, row in enumerate(data):
            for key, value in row.items():
                if not value or (isinstance(value, str) and not value.strip()):
                    empty_entries.append((i+1, key))

        if empty_entries:
            self.logger.warning(f"Found {len(empty_entries)} empty entries in the data")


def validate_ethical_data_files(data_dir: str = "data") -> list:
    """
    Convenience function to validate ethical setting data files.

    Args:
        data_dir: Directory containing the data files

    Returns:
        Validated data list

    Raises:
        DataValidationError: If validation fails
    """
    validator = EthicalDataValidator(data_dir)
    return validator.validate_ethical_data()
