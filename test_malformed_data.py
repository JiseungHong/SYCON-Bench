
#!/usr/bin/env python3
"""
Test script to reproduce the data validation issue.
"""
import tempfile
import os
import sys

# Add the debate_setting directory to the path
sys.path.insert(0, '/workspace/debate_setting')

def test_current_behavior():
    """Test the current behavior with mismatched files."""
    from run_benchmark import read_data

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
        print(f"Questions file: {questions_file}")
        print(f"Arguments file: {arguments_file}")

        try:
            questions, arguments = read_data(tmpdir)
            print(f"Success: Got {len(questions)} questions and {len(arguments)} arguments")
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_current_behavior()
