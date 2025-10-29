#!/usr/bin/env python3
"""
Test runner script for the LangGraph CSV Analysis platform.
Provides different test execution modes and reporting options.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_type="all", verbose=False, coverage=False, report_format="terminal"):
    """Run tests with specified configuration."""
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test files based on type
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration", "test_integration.py"])
    elif test_type == "api":
        cmd.extend(["-m", "api", "test_api.py"])
    elif test_type == "system":
        cmd.extend(["-m", "system", "test_installation.py"])
    elif test_type == "quick":
        cmd.extend(["-m", "not slow"])
    # "all" runs everything by default
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=../backend",
            "--cov-report=term-missing"
        ])
        
        if report_format == "html":
            cmd.append("--cov-report=html:htmlcov")
        elif report_format == "xml":
            cmd.append("--cov-report=xml:coverage.xml")
    
    # Add other useful options
    cmd.extend([
        "--tb=short",
        "--strict-markers"
    ])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Execute tests
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description="Run tests for LangGraph CSV Analysis platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Types:
  unit         Run only unit tests
  integration  Run integration tests
  api          Run API tests
  system       Run system/installation tests
  quick        Run all tests except slow ones
  all          Run all tests (default)

Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --type api -v      # Run API tests with verbose output
  python run_tests.py --coverage --html  # Run with HTML coverage report
  python run_tests.py --type quick       # Run quick tests only
        """
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["unit", "integration", "api", "system", "quick", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage reports"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report (requires --coverage)"
    )
    
    parser.add_argument(
        "--xml",
        action="store_true",
        help="Generate XML coverage report (requires --coverage)"
    )
    
    args = parser.parse_args()
    
    # Determine report format
    report_format = "terminal"
    if args.html:
        report_format = "html"
        if not args.coverage:
            print("Warning: --html requires --coverage, enabling coverage")
            args.coverage = True
    elif args.xml:
        report_format = "xml"
        if not args.coverage:
            print("Warning: --xml requires --coverage, enabling coverage")
            args.coverage = True
    
    # Run tests
    exit_code = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        coverage=args.coverage,
        report_format=report_format
    )
    
    # Print summary
    print("-" * 60)
    if exit_code == 0:
        print("✅ All tests passed!")
        if args.coverage and report_format == "html":
            print("📊 HTML coverage report generated in htmlcov/index.html")
        elif args.coverage and report_format == "xml":
            print("📊 XML coverage report generated in coverage.xml")
    else:
        print("❌ Some tests failed!")
        print(f"Exit code: {exit_code}")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()