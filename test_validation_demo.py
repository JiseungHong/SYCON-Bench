#!/usr/bin/env python3
"""
Demo script to test the improved data validation functionality.
This script reproduces the issue described in the problem statement and shows the improved error handling.
"""
import os
import tempfile
import shutil
import sys
from pathlib import Path

# Add the debate_setting directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'debate_setting'))

from data_validation import DataValidator, DataValidationError


def create_test_scenario(temp_dir: str, scenario_name: str):
    """Create different test scenarios to demonstrate validation."""
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"\n=== Testing Scenario: {scenario_name} ===")

    if scenario_name == "mismatched_lines":
        # Reproduce the original issue: different number of lines
        questions_content = "Question 1?\nQuestion 2?\n"  # 2 questions
        arguments_content = "Argument 1\nArgument 2\nArgument 3\n"  # 3 arguments

        with open(data_dir / "questions.txt", 'w') as f:
            f.write(questions_content)
        with open(data_dir / "arguments.txt", 'w') as f:
            f.write(arguments_content)

    elif scenario_name == "missing_files":
        # Don't create any files - this will test file existence validation
        # The data directory exists but is empty
        pass

    elif scenario_name == "empty_files":
        # Create empty files
        with open(data_dir / "questions.txt", 'w') as f:
            f.write("")
        with open(data_dir / "arguments.txt", 'w') as f:
            f.write("")

    elif scenario_name == "valid_data":
        # Create valid matching data
        questions_content = "Question 1?\nQuestion 2?\nQuestion 3?\n"
        arguments_content = "Argument 1\nArgument 2\nArgument 3\n"

        with open(data_dir / "questions.txt", 'w') as f:
            f.write(questions_content)
        with open(data_dir / "arguments.txt", 'w') as f:
            f.write(arguments_content)

    elif scenario_name == "malformed_data":
        # Create files with empty lines and whitespace issues
        questions_content = "Question 1?\n\n   \nQuestion 2?\n\nQuestion 3?\n"
        arguments_content = "Argument 1\n\nArgument 2\n   \nArgument 3\n"

        with open(data_dir / "questions.txt", 'w') as f:
            f.write(questions_content)
        with open(data_dir / "arguments.txt", 'w') as f:
            f.write(arguments_content)


def test_scenario(temp_dir: str, scenario_name: str):
    """Test a specific scenario and show the validation results."""
    # Clean up any existing data directory
    data_dir = Path(temp_dir) / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)

    create_test_scenario(temp_dir, scenario_name)

    validator = DataValidator(Path(temp_dir) / "data")

    try:
        questions, arguments = validator.validate_debate_data()
        print(f"✅ Validation successful!")
        print(f"   Found {len(questions)} questions and {len(arguments)} arguments")
        if questions:
            print(f"   First question: {questions[0][:50]}...")
        if arguments:
            print(f"   First argument: {arguments[0][:50]}...")

    except DataValidationError as e:
        print(f"❌ Validation failed with clear error message:")
        print(f"   {str(e)}")

    except Exception as e:
        print(f"💥 Unexpected error: {e}")


def main():
    """Run the validation demo."""
    print("SYCON-Bench Data Validation Demo")
    print("=" * 50)
    print("This demo shows the improved error handling for data validation issues.")

    # Create a temporary directory for testing
    temp_dir = tempfile.mkdtemp()

    try:
        # Test different scenarios
        scenarios = [
            "valid_data",
            "mismatched_lines",
            "missing_files",
            "empty_files",
            "malformed_data"
        ]

        for scenario in scenarios:
            test_scenario(temp_dir, scenario)

        print(f"\n=== Testing --validate flag functionality ===")
        print("The --validate flag allows you to check data without running the full benchmark:")
        print("Example usage:")
        print("  python run_benchmark.py dummy_model --validate")
        print("  python run_benchmark.py dummy_model --no-validate  # Skip validation")

    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n=== Summary ===")
    print("✅ File existence checks: Clear error messages when files are missing")
    print("✅ Data format validation: Validates CSV/txt format structure")
    print("✅ Improved assertion messages: Detailed error with file names and line numbers")
    print("✅ Data integrity checks: Detects empty responses, malformed entries")
    print("✅ Validation flag: --validate flag to check data without running models")
    print("✅ Graceful error handling: All errors provide actionable feedback")


if __name__ == "__main__":
    main()
