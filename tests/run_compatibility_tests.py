#!/usr/bin/env python3
"""
Comprehensive compatibility test runner for SYCON-Bench models.

This script runs compatibility tests for common models and generates
a detailed report.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the parent directory to the path so we can import model_registry
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_registry.compatibility import ModelCompatibilityTester, test_common_models


def setup_logging(verbose=False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function to run compatibility tests."""
    parser = argparse.ArgumentParser(description="Run model compatibility tests")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific models to test (default: test common models)"
    )
    parser.add_argument(
        "--output",
        default="compatibility_report.md",
        help="Output file for the report (default: compatibility_report.md)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all registered models"
    )

    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    tester = ModelCompatibilityTester()

    if args.models:
        # Test specific models
        logger.info(f"Testing specific models: {args.models}")
        results = tester.run_batch_compatibility_test(args.models)
    elif args.test_all:
        # Test all registered models
        from model_registry.registry import get_registry
        registry = get_registry()
        all_models = list(registry.get_compatibility_matrix().keys())
        logger.info(f"Testing all registered models: {len(all_models)} models")
        results = tester.run_batch_compatibility_test(all_models)
    else:
        # Test common models
        logger.info("Testing common models")
        results = test_common_models()

    # Generate and save report
    report = tester.generate_compatibility_report(results)

    with open(args.output, 'w') as f:
        f.write(report)

    logger.info(f"Compatibility report saved to {args.output}")

    # Print summary to console
    total_models = len(results)
    total_tests = sum(len(test_results) for test_results in results.values())
    passed_tests = sum(
        sum(1 for r in test_results if r.passed)
        for test_results in results.values()
    )

    print(f"\n{'='*60}")
    print(f"COMPATIBILITY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Models tested: {total_models}")
    print(f"Total tests: {total_tests}")
    print(f"Passed tests: {passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    print(f"Report saved to: {args.output}")
    print(f"{'='*60}")

    # Exit with error code if any tests failed
    if passed_tests < total_tests:
        sys.exit(1)


if __name__ == "__main__":
    main()
