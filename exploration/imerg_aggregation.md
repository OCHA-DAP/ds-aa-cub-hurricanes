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

# IMERG aggregation

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ocha_stratus as stratus
from tqdm.auto import tqdm

from src.datasources import imerg, zma, ibtracs, codab
from src.utils.raster import upsample_dataarray
from src.constants import *
```

```python
adm0 = codab.load_codab_from_blob()
```

```python
adm0.plot()
```

```python
gdf_zma = zma.load_zma()
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/zma_tracks_2000-2024.parquet"
df_ibtracs = stratus.load_parquet_from_blob(blob_name)
```

```python
df_agg_zma = (
    df_ibtracs.groupby("sid")
    .agg(
        valid_time_min=("valid_time", "min"),
        valid_time_max=("valid_time", "max"),
        wind_speed_max=("wind_speed", "max"),
    )
    .reset_index()
)
```

```python
df_agg_landfall = (
    df_ibtracs[df_ibtracs["landfall"]]
    .groupby("sid")
    .agg(
        wind_speed_max_landfall=("wind_speed", "max"),
    )
    .reset_index()
)
```

```python
df_agg = df_agg_zma.merge(df_agg_landfall, how="left")
```

```python
da_test = imerg.open_imerg_raster_dates(
    pd.date_range("2024-10-20", "2024-10-23")
)
```

```python
da_test
```

```python
da_test_clip = da_test.rio.clip(adm0.geometry)
```

```python
fig, ax = plt.subplots(dpi=300)

da_test_clip.isel(date=1).plot(ax=ax)
adm0.boundary.plot(ax=ax, linewidth=0.5)
```

```python
fig, ax = plt.subplots(dpi=300)

(da_test_clip.isel(date=1) > da_test_clip.isel(date=1).quantile(0.9)).where(
    ~da_test_clip.isel(date=1).isnull()
).plot(ax=ax)
adm0.boundary.plot(ax=ax, linewidth=0.5)
```

```python
fig, ax = plt.subplots(dpi=300)

(da_test_clip.isel(date=1) > da_test_clip.isel(date=1).quantile(0.99)).where(
    ~da_test_clip.isel(date=1).isnull()
).plot(ax=ax)
adm0.boundary.plot(ax=ax, linewidth=0.5)
```

```python
df_agg
```

```python
da_test_clip
```

```python
da_sum_test = da_test_clip.sum(dim="date").where(
    ~da_test_clip.isel(date=0).isnull()
)
```

```python
da_sum_test.plot()
```

```python
da_test_clip
```

```python
da_rolling_test = da_test_clip.rolling(date=2).sum()
```

```python
da_rolling_test.isel(date=0).plot()
```

```python
da_rolling_test.isel(date=1).plot()
```

```python
da_rolling_test.isel(date=2).plot()
```

```python
da_rolling_test.isel(date=3).plot()
```

```python
da_rolling_test
```

```python
q_thresh = float(da_rolling_test.quantile(0.8, dim=["x", "y"]).max())
```

```python
q_thresh
```

```python
q_threshs = da_rolling_test.quantile(0.8, dim=["x", "y"])
```

```python
q_threshs
```

```python
float(q_threshs.max())
```

```python
float(
    da_rolling_test.where(da_rolling_test >= q_threshs)
    .mean(dim=["x", "y"])
    .max()
)
```

```python
da_sum_test.quantile(0.8, dim=["x", "y"]).max()
```

```python
da_sum_test.plot()
```

```python
start_date = "2000-06-01"
df_agg_recent = df_agg[df_agg["valid_time_min"] >= start_date]
```

```python
quantiles = [0.5, 0.8, 0.9, 0.95, 0.99]


def get_storm_rainfall_aggregations(row):
    row = row.copy()
    min_date = row["valid_time_min"].date() - pd.DateOffset(days=1)
    max_date = row["valid_time_max"].date() + pd.DateOffset(days=1)
    dates = pd.date_range(min_date, max_date)
    da = imerg.open_imerg_raster_dates(dates)
    da_clip = da.rio.clip(adm0.geometry)

    # sum over whole time
    da_sum = da_clip.sum(dim="date").where(~da_clip.isel(date=0).isnull())
    # 2-day rolling sum
    da_rolling2 = da_clip.rolling(date=2).sum()
    # 3-day rolling sum
    da_rolling3 = da_clip.rolling(date=3).sum()

    # take quantiles
    for quantile in quantiles:
        for da_agg, agg_str in [
            (da_sum, "total"),
            (da_rolling2, "roll2"),
            (da_rolling3, "roll3"),
        ]:
            # get quantile threshs
            quantile_threshs = da_agg.quantile(quantile, dim=["x", "y"])
            # get max value
            row[f"q{quantile*100:.0f}_{agg_str}"] = float(
                quantile_threshs.max()
            )
            # mask by values above quantile threshs
            row[f"q{quantile*100:.0f}_{agg_str}_mean_abv"] = float(
                da_agg.where(da_agg >= quantile_threshs)
                .mean(dim=["x", "y"])
                .max()
            )

    return row
```

```python
get_storm_rainfall_aggregations(df_agg_recent.iloc[0])
```

```python
tqdm.pandas()
```

```python
df_agg_recent = df_agg_recent.progress_apply(
    get_storm_rainfall_aggregations, axis=1
)
```

```python
df_agg_recent
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/storm_stats/zma_stats_imerg_quantiles.parquet"
)
stratus.upload_parquet_to_blob(df_agg_recent, blob_name)
```
