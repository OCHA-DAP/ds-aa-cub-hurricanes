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

# IMERG quantiles

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import math

import ocha_stratus as stratus
import matplotlib.pyplot as plt

from src.datasources import ibtracs
from src.constants import *
```

## Load data

```python
df_storms = ibtracs.load_storms()
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/impact/emdat_cerf_upto2024.parquet"
df_impact = stratus.load_parquet_from_blob(blob_name)
df_impact["cerf"] = ~df_impact["Amount in US$"].isnull()
cols = ["sid", "cerf", "Total Affected"]
df_impact = df_impact[cols]
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/storm_stats/zma_stats.parquet"

df_stats_raw_meanonly = stratus.load_parquet_from_blob(blob_name)
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/storm_stats/zma_stats_imerg_quantiles.parquet"
)

df_stats_raw = stratus.load_parquet_from_blob(blob_name)
```

```python
df_stats_raw
```

```python
df_stats = (
    df_stats_raw.merge(df_stats_raw_meanonly[["sid", "max_roll2_mean"]])
    .merge(df_storms)
    .merge(df_impact, how="left")
)
# note we have to set type as "boolean" (NOT bool) to get the desired behaviour here,
# without throwing warnings
df_stats["cerf"] = df_stats["cerf"].astype("boolean").fillna(False)
```

```python
df_stats
```

```python
df_stats.corr(numeric_only=True)[["cerf", "Total Affected"]].plot(kind="bar")
```

```python
rain_cols = [x for x in df_stats.columns if x.startswith("q")] + [
    "max_roll2_mean"
]
```

```python
target_rp = 4
total_years = df_stats["season"].nunique()
target_year_count = math.floor((total_years + 1) / target_rp)
```

```python
target_year_count
```

```python

```
