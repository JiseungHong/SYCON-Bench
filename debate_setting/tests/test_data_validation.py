"""
Tests for data validation functionality in the debate setting.
"""
import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add the parent directory to the path to import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_validation import DataValidator, DataValidationError, validate_data_files


class TestDataValidator:
    """Test cases for DataValidator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = DataValidator(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_file(self, filename: str, content: str):
        """Helper to create test files."""
        filepath = Path(self.temp_dir) / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(filepath)

    def test_validate_file_exists_success(self):
        """Test successful file existence validation."""
        self.create_test_file("test.txt", "test content")
        # Should not raise an exception
        self.validator.validate_file_exists("test.txt")

    def test_validate_file_exists_missing_file(self):
        """Test file existence validation with missing file."""
        with pytest.raises(DataValidationError, match="Required data file not found"):
            self.validator.validate_file_exists("nonexistent.txt")

    def test_validate_text_file_format_success(self):
        """Test successful text file format validation."""
        content = "Line 1\nLine 2\nLine 3\n"
        self.create_test_file("test.txt", content)

        lines = self.validator.validate_text_file_format("test.txt")
        assert lines == ["Line 1", "Line 2", "Line 3"]

    def test_validate_text_file_format_empty_file(self):
        """Test text file format validation with empty file."""
        self.create_test_file("empty.txt", "")

        with pytest.raises(DataValidationError, match="contains only 0 non-empty lines"):
            self.validator.validate_text_file_format("empty.txt", min_lines=1)

    def test_validate_text_file_format_insufficient_lines(self):
        """Test text file format validation with insufficient lines."""
        content = "Only one line"
        self.create_test_file("short.txt", content)

        with pytest.raises(DataValidationError, match="contains only 1 non-empty lines"):
            self.validator.validate_text_file_format("short.txt", min_lines=5)

    def test_validate_data_integrity_success(self):
        """Test successful data integrity validation."""
        questions = ["Question 1?", "Question 2?", "Question 3?"]
        arguments = ["Argument 1", "Argument 2", "Argument 3"]

        # Should not raise an exception
        self.validator.validate_data_integrity(questions, arguments)

    def test_validate_data_integrity_length_mismatch(self):
        """Test data integrity validation with length mismatch."""
        questions = ["Question 1?", "Question 2?"]
        arguments = ["Argument 1", "Argument 2", "Argument 3"]

        with pytest.raises(DataValidationError, match="Number of questions.*does not match"):
            self.validator.validate_data_integrity(questions, arguments)

    def test_validate_data_integrity_empty_questions(self):
        """Test data integrity validation with empty questions."""
        questions = ["Question 1?", "", "Question 3?"]
        arguments = ["Argument 1", "Argument 2", "Argument 3"]

        with pytest.raises(DataValidationError, match="Empty questions found"):
            self.validator.validate_data_integrity(questions, arguments)

    def test_validate_data_integrity_empty_arguments(self):
        """Test data integrity validation with empty arguments."""
        questions = ["Question 1?", "Question 2?", "Question 3?"]
        arguments = ["Argument 1", "", "Argument 3"]

        with pytest.raises(DataValidationError, match="Empty arguments found"):
            self.validator.validate_data_integrity(questions, arguments)

    def test_validate_debate_data_success(self):
        """Test successful debate data validation."""
        questions_content = "Question 1?\nQuestion 2?\nQuestion 3?\n"
        arguments_content = "Argument 1\nArgument 2\nArgument 3\n"

        self.create_test_file("questions.txt", questions_content)
        self.create_test_file("arguments.txt", arguments_content)

        questions, arguments = self.validator.validate_debate_data()

        assert len(questions) == 3
        assert len(arguments) == 3
        assert questions[0] == "Question 1?"
        assert arguments[0] == "Argument 1"

    def test_validate_debate_data_missing_files(self):
        """Test debate data validation with missing files."""
        with pytest.raises(DataValidationError, match="Required data file not found"):
            self.validator.validate_debate_data()

    def test_find_duplicates(self):
        """Test duplicate detection."""
        items = ["Item 1", "Item 2", "Item 1", "Item 3", "item 1"]
        duplicates = self.validator._find_duplicates(items)

        assert "item 1" in duplicates
        assert duplicates["item 1"] == [1, 3, 5]  # Line numbers (1-indexed)


class TestValidateDataFiles:
    """Test cases for the validate_data_files convenience function."""

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

    def test_validate_data_files_debate_success(self):
        """Test validate_data_files for debate setting (success case)."""
        questions_content = "Question 1?\nQuestion 2?\nQuestion 3?\n"
        arguments_content = "Argument 1\nArgument 2\nArgument 3\n"

        self.create_data_files(questions_content, arguments_content)

        questions, arguments = validate_data_files(data_dir="data", setting_type="debate")

        assert len(questions) == 3
        assert len(arguments) == 3
        assert questions[0] == "Question 1?"
        assert arguments[0] == "Argument 1"

    def test_validate_data_files_debate_failure(self):
        """Test validate_data_files for debate setting (failure case)."""
        # Create mismatched files
        questions_content = "Question 1?\nQuestion 2?\n"
        arguments_content = "Argument 1\nArgument 2\nArgument 3\n"

        self.create_data_files(questions_content, arguments_content)

        with pytest.raises(DataValidationError, match="Number of questions.*does not match"):
            validate_data_files(data_dir="data", setting_type="debate")

    def test_validate_data_files_unknown_setting(self):
        """Test validate_data_files with unknown setting type."""
        with pytest.raises(ValueError, match="Unknown setting type"):
            validate_data_files(data_dir="data", setting_type="unknown")


class TestIntegrationScenarios:
    """Integration test scenarios for common data issues."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.validator = DataValidator(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_file(self, filename: str, content: str):
        """Helper to create test files."""
        filepath = Path(self.temp_dir) / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return str(filepath)

    def test_scenario_different_line_counts(self):
        """Test scenario: files have different number of lines."""
        questions_content = "Question 1?\nQuestion 2?\n"
        arguments_content = "Argument 1\nArgument 2\nArgument 3\n"

        self.create_test_file("questions.txt", questions_content)
        self.create_test_file("arguments.txt", arguments_content)

        with pytest.raises(DataValidationError, match="Number of questions.*does not match"):
            self.validator.validate_debate_data()

    def test_scenario_empty_lines_in_middle(self):
        """Test scenario: files have empty lines in the middle."""
        questions_content = "Question 1?\n\nQuestion 2?\n"
        arguments_content = "Argument 1\n\nArgument 2\n"

        self.create_test_file("questions.txt", questions_content)
        self.create_test_file("arguments.txt", arguments_content)

        # Should succeed (empty lines are filtered out)
        questions, arguments = self.validator.validate_debate_data()
        assert len(questions) == 2
        assert len(arguments) == 2

    def test_scenario_unicode_content(self):
        """Test scenario: files contain unicode characters."""
        questions_content = "¿Pregunta 1?\n问题 2?\nQuestion 3 with émojis 🤔?\n"
        arguments_content = "Argumento 1\n论点 2\nArgument 3 with émojis 💭\n"

        self.create_test_file("questions.txt", questions_content)
        self.create_test_file("arguments.txt", arguments_content)

        # Should succeed
        questions, arguments = self.validator.validate_debate_data()
        assert len(questions) == 3
        assert len(arguments) == 3
        assert "🤔" in questions[2]
        assert "💭" in arguments[2]

    def test_scenario_very_short_content(self):
        """Test scenario: files have very short content."""
        questions_content = "Q1?\nQ2?\nQ3?\n"
        arguments_content = "A1\nA2\nA3\n"

        self.create_test_file("questions.txt", questions_content)
        self.create_test_file("arguments.txt", arguments_content)

        # Should succeed but generate warnings
        questions, arguments = self.validator.validate_debate_data()
        assert len(questions) == 3
        assert len(arguments) == 3


if __name__ == "__main__":
    pytest.main([__file__])
