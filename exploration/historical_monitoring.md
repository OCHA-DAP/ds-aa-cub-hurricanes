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

# Historical monitoring
<!-- markdownlint-disable MD013 -->
Checking values from NHC and CHIRPS-GEFS historical forecasts

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm.auto import tqdm

from src.datasources import nhc, ibtracs, chirps_gefs, zma
from src.constants import *
```

## Load data

```python
gdf_zma = zma.load_zma()
```

```python
df_gefs = chirps_gefs.load_processed_chirps_gefs()
```

```python
df_gefs
```

```python
gdf_nhc = nhc.load_historical_forecasts(include_geometry=True)
```

```python
gdf_nhc["in_zma"] = gdf_nhc.within(gdf_zma.iloc[0].geometry)
```

```python
gdf_nhc
```

```python
gdf_nhc_zma = gdf_nhc[gdf_nhc["in_zma"]]
```

## Run historical simulation

Iterate over all past forecasts, checking the rainfall forecast if the forecasted track enters the ZMA.

```python
dicts = []

lts = {
    "readiness": pd.Timedelta(days=5),
    "action": pd.Timedelta(days=3),
}

for atcf_id, storm_group in tqdm(gdf_nhc_zma.groupby("atcf_id")):
    for issue_time, issue_group in storm_group.groupby("issue_time"):
        df_gefs_i = df_gefs[df_gefs["issued_date"] == issue_time.date()]
        for lt_name, lt in lts.items():
            lt_group = issue_group[
                issue_group["valid_time"] <= issue_time + lt
            ]
            start_date = lt_group["valid_time"].min().date()
            end_date = lt_group["valid_time"].max().date() + pd.Timedelta(
                days=1
            )
            df_gefs_v = df_gefs_i[
                (df_gefs_i["valid_date"] >= start_date)
                & (df_gefs_i["valid_date"] <= end_date)
            ]
            dict_out = df_gefs_v.groupby("variable")["value"].max().to_dict()
            dict_out.update(
                {
                    "atcf_id": atcf_id,
                    "issue_time": issue_time,
                    "lt_name": lt_name,
                    "wind": lt_group["windspeed"].max(),
                }
            )
            dicts.append(dict_out)
```

```python
df_monitors = pd.DataFrame(dicts)
```

```python
df_monitors
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/nhc/monitors_nhc_chirpsgefs.parquet"
stratus.upload_parquet_to_blob(df_monitors, blob_name)
```

## Basic plotting

Get the maximum value for each historical monitoring point.

```python
df_monitors_max = (
    df_monitors.groupby(["atcf_id", "lt_name"]).max().reset_index().dropna()
)
```

```python
df_monitors_max
```

```python
df_monitors_max["year"] = df_monitors_max["atcf_id"].str[-4:].astype(int)
```

```python
df_monitors_max
```

```python
df_monitors_max["issue_time"].max().year
```

```python
n_years = (
    df_monitors_max["issue_time"].max().year
    - df_monitors_max["issue_time"].min().year
    + 1
)
```

```python
# set RPs based on rough numbers approved by CERF
target_rps = {
    "action": 3.7,
    "readiness": 3,
}
```

```python
# set target number of years to trigger for based on RP
# note that int() takes the floor, so will be conservative
target_trig_years = {
    lt_name: int((n_years + 1) / rp) for lt_name, rp in target_rps.items()
}
```

```python
target_trig_years
```

```python
[x for x in df_monitors_max.columns if "q" in x or x == "mean"]
```

```python
# iterate over all possible trigger combinations
# keeping only the ones that have the correct return period
rain_cols = [x for x in df_monitors_max.columns if "q" in x or x == "mean"]

dicts = []

for lt_name, lt_trig_years in target_trig_years.items():
    for wind_thresh in df_monitors_max["wind"].unique():
        for rain_col in rain_cols:
            for rain_thresh in df_monitors_max[rain_col].unique():
                dff = df_monitors_max[
                    (df_monitors_max["wind"] >= wind_thresh)
                    & (df_monitors_max[rain_col] >= rain_thresh)
                ]
                if dff["year"].nunique() == lt_trig_years:
                    dicts.append(
                        {
                            "lt_name": lt_name,
                            "wind_thresh": wind_thresh,
                            "rain_thresh": rain_thresh,
                            "rain_col": rain_col,
                        }
                    )
```

```python
df_threshs = pd.DataFrame(dicts)
```

```python
# quick plot of distribution of trigger combos with correct return periods
df_plot = df_threshs[df_threshs["rain_col"] == "mean"]

for rain_col, rain_group in df_threshs.groupby("rain_col"):
    fig, ax = plt.subplots()
    ax.set_title(rain_col)
    for lt_name, group in rain_group.groupby("lt_name"):
        group.plot(
            x="wind_thresh",
            y="rain_thresh",
            ax=ax,
            label=lt_name,
            linewidth=0,
            marker=".",
        )
```
