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

# Melissa CHIRPS-GEFS

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import xarray as xr
from tqdm.auto import tqdm
from dask.diagnostics import ProgressBar

from src.datasources import ibtracs, codab, chirps_gefs
from src.datasources.ibtracs import knots2cat
from src.constants import *
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fcast_obsv_combined_stats.parquet"
df_stats = stratus.load_parquet_from_blob(blob_name)
```

```python
adm0 = codab.load_codab_from_blob()
```

```python
adm0.plot()
```

```python
issued_date = datetime(2025, 10, 28).date()

das = []
for lt in tqdm(range(16)):
    valid_date = issued_date + pd.Timedelta(days=lt)
    da_in = chirps_gefs.open_chirps_gefs(issued_date, valid_date)
    da_in["valid_date"] = valid_date
    das.append(da_in)
```

```python
da_gefs = xr.concat(das, dim="valid_date").squeeze(drop=True)
```

```python
total_bounds = adm0.total_bounds
```

```python
da_gefs_clip_box = da_gefs.rio.clip_box(*total_bounds)
```

```python
with ProgressBar():
    da_gefs_clip_box_computed = da_gefs_clip_box.compute()
```

```python
da_gefs_clip = da_gefs_clip_box_computed.rio.clip(adm0.geometry)
```

```python
da_gefs_clip.isel(valid_date=0).plot()
```

```python
da_gefs_q80 = da_gefs_clip.quantile(q=0.8, dim=["x", "y"])
```

```python
da_gefs_roll2 = da_gefs_clip.rolling(valid_date=2).sum()
```

```python
da_gefs_q80 = da_gefs_roll2.quantile(q=0.8, dim=["x", "y"])
```

```python
da_gefs_q80.max()
```

```python
da_gefs_clip.mean(dim=["x", "y"]).plot()
```

```python

```
