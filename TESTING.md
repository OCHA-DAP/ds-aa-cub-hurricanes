# Testing Framework for Cuba Hurricane Monitoring

This document describes the comprehensive testing framework set up for the `monitoring_utils.py` module and the Cuba hurricane monitoring system.

## Quick Start

### Install Testing Dependencies

```bash
# Install all dependencies including testing tools
make install-dev

# Or manually:
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

### Run Tests

```bash
# Run unit tests (fastest)
make test

# Run all tests with coverage
make test-coverage

# Run linting and formatting checks
make lint

# Run everything
make test-all
```

## Testing Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and configuration
├── monitoring/
│   ├── __init__.py
│   ├── test_monitoring_utils.py   # Core functionality tests
│   ├── test_rainfall_processors.py # Rainfall processor tests
│   ├── test_factory_functions.py  # Factory function tests
│   └── test_integration.py        # Integration tests
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Test individual functions and methods in isolation
- Use mocked dependencies 
- Fast execution (< 1 second per test)
- Run with: `make test-unit` or `pytest -m unit`

### Integration Tests (`@pytest.mark.integration`)
- Test complete workflows with multiple components
- May use external dependencies (mocked for CI)
- Run with: `make test-integration` or `pytest -m integration`

### Slow Tests (`@pytest.mark.slow`)
- Performance tests and large dataset processing
- Excluded from regular CI runs
- Run manually when needed

## Key Test Files

### `conftest.py`
Contains shared fixtures:
- `mock_codab_data`: Mock Cuba boundary data
- `mock_nhc_forecast_data`: Mock NHC forecast tracks
- `mock_nhc_obsv_data`: Mock NHC observational tracks
- `mock_imerg_data`: Mock rainfall data
- `cuba_monitor_*`: Pre-configured monitor instances

### `test_monitoring_utils.py`
Tests core `CubaHurricaneMonitor` functionality:
- Utility methods (ID creation, data filtering)
- Geometric operations (distance calculations, interpolation)
- Wind calculations and threshold checking
- Data processing workflows

### `test_rainfall_processors.py`
Tests rainfall processor classes:
- `IMERGProcessor` functionality
- Abstract base class compliance
- Integration with monitoring system

### `test_factory_functions.py`
Tests factory functions and module-level features:
- `create_cuba_hurricane_monitor()` function
- End-to-end workflow testing
- Configuration validation

### `test_integration.py`
Tests complete system workflows:
- Full forecast processing pipeline
- Observational data with rainfall integration
- Error handling scenarios
- Performance benchmarks

## Running Tests

### Command Line Options

```bash
# Basic test run
pytest tests/

# Verbose output
pytest tests/ -v

# Run specific test file
pytest tests/monitoring/test_monitoring_utils.py

# Run specific test function
pytest tests/monitoring/test_monitoring_utils.py::TestCubaHurricaneMonitorUtilities::test_create_monitor_id

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run tests in parallel
pytest tests/ -n auto

# Stop on first failure
pytest tests/ -x

# Run only unit tests
pytest tests/ -m unit

# Run only integration tests  
pytest tests/ -m integration

# Exclude slow tests
pytest tests/ -m "not slow"
```

### Using the Test Runner Script

```bash
# Run unit tests
python run_tests.py --type unit --verbose

# Run with coverage
python run_tests.py --type coverage

# Run all checks
python run_tests.py --type all

# Run in parallel
python run_tests.py --type unit --parallel

# Stop on first failure
python run_tests.py --type unit --fail-fast
```

### Using Make Commands

```bash
make test              # Unit tests
make test-coverage     # Tests with coverage report
make test-integration  # Integration tests only
make test-all          # All tests and linting
make lint              # Linting only
make format            # Code formatting
make clean             # Clean test artifacts
```

## Continuous Integration

### GitHub Actions Workflow
Located in `.github/workflows/ci.yml`, the CI pipeline:

1. **Test Matrix**: Tests on Python 3.9, 3.10, 3.11
2. **Unit Tests**: Fast unit test execution
3. **Integration Tests**: Full workflow testing (main branch only)
4. **Code Quality**: Linting, formatting, security checks
5. **Coverage**: Upload coverage reports to Codecov

### CI Triggers
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

## Configuration Files

### `pytest.ini`
Pytest configuration including:
- Coverage settings
- Test discovery patterns
- Marker definitions
- Report formats

### `pyproject.toml` (Coverage section)
Coverage reporting configuration:
- Source directories
- Exclusion patterns
- Report formats

## Mocking Strategy

### External Dependencies
All external dependencies are mocked in tests:
- **Azure Blob Storage**: `stratus.load_parquet_from_blob()`, `stratus.upload_parquet_to_blob()`
- **NHC Data**: `nhc.load_recent_glb_forecasts()`, `nhc.load_recent_glb_obsv()`
- **IMERG Data**: `imerg.load_imerg_recent()`
- **Boundary Data**: `codab.load_codab_from_blob()`

### Mock Data Characteristics
- **Realistic structure**: Matches actual data schemas
- **Edge cases**: Empty data, boundary conditions
- **Temporal consistency**: Proper time series for track data
- **Geospatial validity**: Valid coordinates for Cuba region

## Coverage Goals

- **Target**: 90%+ line coverage
- **Focus areas**: Core business logic, data processing workflows
- **Exclusions**: Error handling, logging, `__main__` blocks

## Writing New Tests

### Guidelines
1. **One test per behavior**: Each test should verify one specific behavior
2. **Descriptive names**: Test names should clearly describe what is being tested
3. **Arrange-Act-Assert**: Structure tests with clear setup, execution, and verification
4. **Use fixtures**: Leverage shared fixtures for consistent test data
5. **Mock external calls**: Mock all external dependencies and I/O operations

### Example Test Structure
```python
def test_should_skip_existing_with_data_no_clobber(
    self, cuba_monitor_no_rainfall, sample_existing_data
):
    """Test skip logic with existing data, no clobber."""
    # Arrange
    monitor_id = "al012024_fcast_2024-07-01T12:00:00"
    
    # Act
    result = cuba_monitor_no_rainfall._should_skip_existing(
        monitor_id, sample_existing_data, clobber=False
    )
    
    # Assert
    assert result is True
```

### Adding New Fixtures
Add fixtures to `conftest.py` for reusable test data:

```python
@pytest.fixture
def my_test_data():
    """Description of the fixture."""
    return pd.DataFrame({
        "column1": [1, 2, 3],
        "column2": ["a", "b", "c"]
    })
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure package is installed in development mode: `pip install -e .`
2. **Missing Dependencies**: Install test dependencies: `pip install -r requirements-dev.txt`
3. **Slow Tests**: Use `-m "not slow"` to exclude performance tests
4. **Mock Issues**: Check that all external dependencies are properly mocked

### Debug Mode
Run tests with `--pdb` to drop into debugger on failure:
```bash
pytest tests/monitoring/test_monitoring_utils.py::test_name --pdb
```

### Verbose Logging
Enable debug logging in tests:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extending the Framework

### Adding New Test Categories
1. Add marker to `pytest.ini`
2. Mark tests with `@pytest.mark.new_category`
3. Update CI workflow if needed

### Performance Testing
- Use `@pytest.mark.slow` for performance tests
- Benchmark against baseline metrics
- Consider memory usage and execution time

### Security Testing
- Add security-focused tests
- Test input validation and sanitization
- Verify proper handling of sensitive data

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Fast Feedback**: Keep unit tests under 1 second each
3. **Realistic Data**: Use data that reflects real-world scenarios
4. **Error Cases**: Test both success and failure paths
5. **Documentation**: Keep test documentation up to date
6. **Regular Maintenance**: Review and update tests as code evolves
