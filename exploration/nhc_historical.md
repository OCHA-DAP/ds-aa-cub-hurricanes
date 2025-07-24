---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.1
  kernelspec:
    display_name: ds-aa-cub-hurricanes
    language: python
    name: ds-aa-cub-hurricanes
---

# Historical NHC

Download and process historical NHC forecasts

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from src.datasources import nhc
```

```python
nhc.download_historical_forecasts(start_year=2023, end_year=2024)
```

```python
nhc.process_historical_forecasts()
```
