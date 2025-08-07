# Cuba Anticipatory Action: Hurricanes

[![CI](https://github.com/OCHA-DAP/ds-aa-cub-hurricanes/actions/workflows/ci.yml/badge.svg?branch=forecast-monitor)](https://github.com/OCHA-DAP/ds-aa-cub-hurricanes/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/OCHA-DAP/ds-aa-cub-hurricanes/branch/main/graph/badge.svg)](https://codecov.io/gh/OCHA-DAP/ds-aa-cub-hurricanes)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Status](https://img.shields.io/badge/status-under%20development-orange.svg)](https://github.com/OCHA-DAP/ds-aa-cub-hurricanes)

Analysis for Cuba Hurricanes Anticipatory Action framework.

## Developer Setup

1. Create and activate a virtual environment:

Using `venv`:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Or you may want to use `pyenv`:

```
pyenv virtualenv <PYTHON_VERSION> virtualenv <VIRTUALENV_NAME>
pyenv local <VIRTUALENV_NAME>
pyenv activate <VIRTUALENV_NAME>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. Install package in development mode:
```bash
pip install -e .
```

4. Set up the following environment variables in a `.env` file:
```
DS_AZ_BLOB_DEV_SAS=<provided on request>
DS_AZ_BLOB_PROD_SAS=<provided on request>

DS_AZ_DB_DEV_PW=<provided on request>
DS_AZ_DB_DEV_UID=<provided on request>

DS_AZ_DB_PROD_PW=<provided on request>
DS_AZ_DB_PROD_UID=<provided on request>

DS_AZ_DB_DEV_HOST=<provided on request>
DS_AZ_DB_PROD_HOST=<provided on request>

```

### Pre-Commit

All code is formatted according to black and flake8 guidelines. The repo is set-up to use pre-commit. Before you start developing in this repository, you will need to run

```
pre-commit install
```

You can run all hooks against all your files using

```
pre-commit run --all-files
```

## Testing

This project uses pytest for testing with comprehensive unit and integration test coverage.

### Running Tests

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
```

Run only unit tests:
```bash
pytest tests/ -m unit
```

Run only integration tests:
```bash
pytest tests/ -m integration
```

### Using the Makefile

For convenience, you can use the provided Makefile commands:

```bash
# Run all tests
make test

# Run tests with coverage
make test-coverage

# Run linting
make lint

# Run formatting
make format

# Run all quality checks
make quality
```

### Code Quality

The project enforces code quality through:
- **Black** for code formatting
- **isort** for import sorting
- **pytest** for testing with >65% coverage
- **GitHub Actions CI** for automated quality checks

All code is automatically checked for formatting, import sorting, and test coverage in the CI pipeline.

### Manual Steps

There were several manual steps taken to set up the monitoring system for the 
project that are not immediately obvious from the codebase. These include:

1. Creation of map legend - was done manually in this [gslide](https://docs.google.com/presentation/d/1NlUxI7ZTlKH05CI2w7c-rUrkryD7ItBnNncx_ls3pJY/edit?slide=id.p#slide=id.p)
2. Creation of initial maps/plots for testing. Done "manually" in book_cub_hurricanes/email_setup_prep.qmd
3. Creation of `distribution_list.csv` and `test_distribution_list.csv` dropped into blob storage via Azure Storage Explorer.
4. Creation/initiation of email_record.csv

## Monitoring Run States

The monitoring system uses three environment variables to control runtime behavior. These can be set as environment variables or overridden for testing by monkey-patching the constants in [`src/constants.py`](src/constants.py).

### Control Flags

| Flag | Default | Purpose | 
|------|---------|---------|
| `DRY_RUN` | `true` | Prevents external side effects (email sending, record saving) |
| `TEST_EMAIL` | `true` | Controls distribution list selection |
| `FORCE_ALERT` | `false` | Enables test data injection and email prefixes |

### Environment Variable Usage

Set flags via environment variables:
```bash
export DRY_RUN=false
export TEST_EMAIL=true
export FORCE_ALERT=true
python your_script.py
```
or like:

```bash
TEST_EMAIL=true DRY_RUN=false FORCE_ALERT=true python pipelines/01_update_fcast_monitor.py
```


### Testing Override

For testing, monkey-patch the constants directly:
```python
import src.constants as constants

# Override flags for testing
constants.DRY_RUN = False
constants.TEST_EMAIL = False  
constants.FORCE_ALERT = True

# Now run your functions - they'll use the overridden values
from src.email.send_emails import send_info_email
send_info_email(monitor_id, "fcast")
```

### Common Run State Combinations

#### 🔧 **Development/Testing** (Default)
```python
DRY_RUN = True      # Safe - no emails sent, no records saved
TEST_EMAIL = True   # Use test distribution list
FORCE_ALERT = False # Use real monitoring data
```
- **Use case**: Safe development and testing
- **Behavior**: Functions run but no external side effects occur

#### 🧪 **Preview Generation** 
```python
DRY_RUN = True      # No emails sent
TEST_EMAIL = True   # Test distribution (but not used in preview)
FORCE_ALERT = True  # Add "PRUEBA" prefix and test data
```
- **Use case**: Generate email previews with test data
- **Behavior**: Creates HTML previews with test storm data and "PRUEBA" subject prefix

#### ⚠️ **Test Email Sending**
```python
DRY_RUN = False     # Actually send emails
TEST_EMAIL = True   # Send to test distribution list only
FORCE_ALERT = True  # Mark emails as test with "PRUEBA" prefix
```
- **Use case**: End-to-end testing with real email delivery
- **Behavior**: Sends actual emails but only to test recipients with test markers

#### 🚀 **Production**
```python
DRY_RUN = False     # Send emails and save records
TEST_EMAIL = False  # Use production distribution list
FORCE_ALERT = False # Use real data, no test markers
```
- **Use case**: Live production monitoring
- **Behavior**: Full production behavior with real recipients and data

### Flag Details

#### `DRY_RUN`
- **Purpose**: Master safety switch
- **When `true`**: 
  - `send_info_email()` and `send_trigger_email()` print messages instead of sending
  - `save_email_record()` prints message instead of saving to blob storage
- **When `false`**: Full execution with external side effects

#### `TEST_EMAIL` 
- **Purpose**: Controls email distribution lists
- **When `true`**: Uses `test_distribution_list.csv` from blob storage
- **When `false`**: Uses `distribution_list.csv` from blob storage
- **Note**: Only affects actual email sending, not preview functions

#### `FORCE_ALERT`
- **Purpose**: Test data injection and email marking
- **When `true`**: 
  - Adds "PRUEBA : " prefix to email subjects
  - Injects test monitoring data (triggers activated)
  - Email templates show test indicators
- **When `false`**: Uses real monitoring data and standard email formatting
