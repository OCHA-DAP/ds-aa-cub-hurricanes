# Cuba Anticipatory Action: Hurricanes

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
