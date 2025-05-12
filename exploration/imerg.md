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

# IMERG
<!-- markdownlint-disable MD013 -->

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd

from src.datasources import imerg
```

```python
pcode = "CU"
```

```python
df_imerg = imerg.load_imerg(pcode)
```

```python
df_imerg["valid_date"] = pd.to_datetime(df_imerg["valid_date"])
```

```python
df_imerg["mean"].hist()
```

```python
df_imerg["roll2_mean"] = df_imerg["mean"].rolling(2).sum()
```

```python
df_imerg.groupby(df_imerg["valid_date"].dt.year)[
    "roll2_mean"
].max().reset_index()
```

```python

```
