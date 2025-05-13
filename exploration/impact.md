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

# Impact

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus

from src.constants import *
```

```python
blob_name = f"{PROJECT_PREFIX}/raw/impact/cerf-storms-with-sids-2024-02-27.csv"
```

```python
df_cerf = stratus.load_csv_from_blob(blob_name)
```

```python
df_cerf_cub =
```
