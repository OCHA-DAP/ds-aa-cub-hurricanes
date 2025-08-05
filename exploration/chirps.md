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

# CHIRPS

Aggregating CHIRPS to storm

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import pandas as pd

from src.constants import *
```

## Load data

```python
blob_name = f"{PROJECT_PREFIX}/processed/chirps/20240524_chirps_daily_historical_cuba.csv"
df_daily = stratus.load_csv_from_blob(blob_name, parse_dates=["date"])
```

```python
df_daily = df_daily.sort_values("date")
```

```python
df_daily["roll2_mean"] = df_daily["value"].rolling(2).sum()
```

```python
df_daily
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/storm_stats/zma_stats_imerg_quantiles.parquet"
)

df_stats = stratus.load_parquet_from_blob(blob_name)
```

```python
df_stats
```

## Aggregate to storm

```python
def get_storm_rainfall(storm_row):
    min_date = storm_row["valid_time_min"].date()
    max_date = storm_row["valid_time_max"].date() + pd.DateOffset(days=1)
    dff_imerg = df_daily[
        (df_daily["date"] >= pd.Timestamp(min_date))
        & (df_daily["date"] <= pd.Timestamp(max_date))
    ]
    storm_row["chirps_roll2_mean"] = dff_imerg["roll2_mean"].max()
    return storm_row
```

```python
df_stats = df_stats.apply(get_storm_rainfall, axis=1)
```

## Save

```python
blob_name = f"{PROJECT_PREFIX}/processed/chirps/chirps_stats.parquet"
df_stats[["sid", "chirps_roll2_mean"]]
stratus.upload_parquet_to_blob(
    df_stats[["sid", "chirps_roll2_mean"]], blob_name
)
```
