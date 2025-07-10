.PHONY: help test test-unit test-integration test-coverage lint format clean install-dev

# Default target
help:
	@echo "Available targets:"
	@echo "  install-dev    Install development dependencies"
	@echo "  test          Run unit tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-coverage Run tests with coverage report"
	@echo "  test-all      Run all tests and linting"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  clean         Clean up test artifacts"

# Install development dependencies
install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

# Run unit tests
test:
	python -m pytest tests/ -v -m unit

test-unit:
	python -m pytest tests/ -v -m unit

# Run integration tests  
test-integration:
	python -m pytest tests/ -v -m integration

# Run tests with coverage
test-coverage:
	python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml

# Run all tests and checks
test-all:
	python run_tests.py --type all --verbose

# Linting
lint:
	python -m black --check src/ tests/
	python -m isort --check-only src/ tests/
	python -m flake8 src/ tests/ --max-line-length=88

# Format code
format:
	python -m black src/ tests/
	python -m isort src/ tests/

# Clean up
clean:
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Quick test for monitoring utils specifically
test-monitoring:
	python -m pytest tests/monitoring/ -v

# Run tests in parallel
test-parallel:
	python -m pytest tests/ -v -n auto

# Run tests and open coverage report
test-coverage-open: test-coverage
	@echo "Opening coverage report..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open htmlcov/index.html; \
	elif command -v open > /dev/null; then \
		open htmlcov/index.html; \
	else \
		echo "Coverage report available at htmlcov/index.html"; \
	fi
