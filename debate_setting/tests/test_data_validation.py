import os
import pytest
from debate_setting import run_benchmark

def test_missing_files(tmp_path):
    # Remove questions.txt and arguments.txt if present
    data_dir = tmp_path
    q_path = data_dir / "questions.txt"
    a_path = data_dir / "arguments.txt"
    # No files created
    with pytest.raises(FileNotFoundError):
        run_benchmark.read_data(str(data_dir))

def test_mismatched_lines(tmp_path):
    data_dir = tmp_path
    q_path = data_dir / "questions.txt"
    a_path = data_dir / "arguments.txt"
    q_path.write_text("Q1\nQ2\n")
    a_path.write_text("A1\n")
    with pytest.raises(ValueError) as e:
        run_benchmark.read_data(str(data_dir))
    assert "Number of questions" in str(e.value)

def test_empty_lines(tmp_path):
    data_dir = tmp_path
    q_path = data_dir / "questions.txt"
    a_path = data_dir / "arguments.txt"
    q_path.write_text("\n\nQ1\n\nQ2\n\n")
    a_path.write_text("A1\n\nA2\n\n")
    questions, arguments = run_benchmark.read_data(str(data_dir))
    assert questions == ["Q1", "Q2"]
    assert arguments == ["A1", "A2"]

def test_malformed_file(tmp_path):
    data_dir = tmp_path
    q_path = data_dir / "questions.txt"
    a_path = data_dir / "arguments.txt"
    # Simulate binary or unreadable file
    q_path.write_bytes(b"\x00\x01\x02")
    a_path.write_text("A1\nA2\n")
    with pytest.raises(ValueError) as e:
        run_benchmark.read_data(str(data_dir))
    assert "questions.txt" in str(e.value)
