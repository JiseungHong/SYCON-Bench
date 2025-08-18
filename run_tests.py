#!/usr/bin/env python3
"""
Test runner for SYCON-Bench tests.
"""
import sys
import unittest
import os

# Add the workspace to the Python path
sys.path.insert(0, '/workspace')

if __name__ == '__main__':
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = '/workspace/tests'
    suite = loader.discover(start_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Exit with non-zero code if tests failed
    sys.exit(not result.wasSuccessful())
