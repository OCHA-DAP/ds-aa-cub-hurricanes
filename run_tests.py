#!/usr/bin/env python3
"""
Test runner script for the Cuba hurricane monitoring system.
Provides convenient ways to run different types of tests.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 50)

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test runner for Cuba hurricane monitoring"
    )
    parser.add_argument(
        "--type",
        choices=["unit", "integration", "all", "coverage", "lint"],
        default="unit",
        help="Type of tests to run",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Run tests in verbose mode",
    )
    parser.add_argument(
        "--parallel", "-p", action="store_true", help="Run tests in parallel"
    )
    parser.add_argument(
        "--fail-fast", "-x", action="store_true", help="Stop on first failure"
    )

    args = parser.parse_args()

    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]

    if args.verbose:
        pytest_cmd.append("-v")

    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])

    if args.fail_fast:
        pytest_cmd.append("-x")

    success = True

    if args.type == "lint":
        # Run linting checks
        commands = [
            (
                ["python", "-m", "black", "--check", "src/", "tests/"],
                "Black code formatting check",
            ),
            (
                ["python", "-m", "isort", "--check-only", "src/", "tests/"],
                "Import sorting check",
            ),
            (["python", "-m", "flake8", "src/", "tests/"], "Flake8 linting"),
        ]

        for cmd, desc in commands:
            if not run_command(cmd, desc):
                success = False

    elif args.type == "unit":
        # Run unit tests
        cmd = pytest_cmd + ["tests/", "-m", "unit"]
        success = run_command(cmd, "Unit tests")

    elif args.type == "integration":
        # Run integration tests
        cmd = pytest_cmd + ["tests/", "-m", "integration"]
        success = run_command(cmd, "Integration tests")

    elif args.type == "coverage":
        # Run tests with coverage
        cmd = pytest_cmd + [
            "tests/",
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-report=xml",
        ]
        success = run_command(cmd, "Tests with coverage")

        if success:
            print("\n📊 Coverage report generated in htmlcov/index.html")

    elif args.type == "all":
        # Run all tests and checks
        test_types = [
            (pytest_cmd + ["tests/", "-m", "unit"], "Unit tests"),
            (
                pytest_cmd + ["tests/", "-m", "integration"],
                "Integration tests",
            ),
            (
                ["python", "-m", "black", "--check", "src/", "tests/"],
                "Code formatting check",
            ),
            (
                ["python", "-m", "isort", "--check-only", "src/", "tests/"],
                "Import sorting check",
            ),
        ]

        for cmd, desc in test_types:
            if not run_command(cmd, desc):
                success = False

    if success:
        print("\n🎉 All checks passed!")
        sys.exit(0)
    else:
        print("\n💥 Some checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
