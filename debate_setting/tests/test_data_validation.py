import os
import tempfile
import pytest
from debate_setting import run_benchmark


def write_file(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line + '\n')

def test_missing_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Remove questions.txt and arguments.txt
        with pytest.raises(FileNotFoundError):
            run_benchmark.read_data(data_dir=tmpdir)

def test_mismatched_lines():
    with tempfile.TemporaryDirectory() as tmpdir:
        qfile = os.path.join(tmpdir, 'questions.txt')
        afile = os.path.join(tmpdir, 'arguments.txt')
        write_file(qfile, ['Q1', 'Q2'])
        write_file(afile, ['A1'])
        with pytest.raises(ValueError) as excinfo:
            run_benchmark.read_data(data_dir=tmpdir)
        assert 'Number of questions' in str(excinfo.value)
        assert 'questions.txt' in str(excinfo.value) or 'arguments.txt' in str(excinfo.value)

def test_empty_lines():
    with tempfile.TemporaryDirectory() as tmpdir:
        qfile = os.path.join(tmpdir, 'questions.txt')
        afile = os.path.join(tmpdir, 'arguments.txt')
        write_file(qfile, ['Q1', '', 'Q2'])
        write_file(afile, ['A1', '', 'A2'])
        # Should not raise, empty lines are ignored
        questions, arguments = run_benchmark.read_data(data_dir=tmpdir)
        assert questions == ['Q1', 'Q2']
        assert arguments == ['A1', 'A2']

def test_malformed_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        qfile = os.path.join(tmpdir, 'questions.txt')
        afile = os.path.join(tmpdir, 'arguments.txt')
        # Simulate a binary/malformed file
        with open(qfile, 'wb') as f:
            f.write(b'\x00\x01\x02')
        write_file(afile, ['A1'])
        with pytest.raises(ValueError):
            run_benchmark.read_data(data_dir=tmpdir)
