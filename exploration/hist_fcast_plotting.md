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

# Historical forecast plotting
<!-- markdownlint-disable MD013 -->
Plotting historical forecasts to compare:

- The first time (issue time) a forecasted track enters the ZMA (i.e., the earliest time we could trigger)
- The time the storm makes landfall (i.e., the time we really care about)

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

from src.datasources import nhc, imerg, codab, zma
from src.constants import *
```

## Load data

```python
blob_name = blob_name = (
    f"{PROJECT_PREFIX}/processed/nhc/monitors_nhc_chirpsgefs.parquet"
)
df_monitors = stratus.load_parquet_from_blob(blob_name)
```

```python
filename = "temp/ibtracs.NA.list.v04r01.csv"
df_ibtracs = pd.read_csv(filename, skiprows=[1])
```

```python
df_storms = (
    df_ibtracs.groupby("SID")[["NAME", "USA_ATCF_ID"]].last().reset_index()
)
df_storms = df_storms.rename(
    columns={"SID": "sid", "NAME": "name", "USA_ATCF_ID": "atcf_id"}
)
df_storms = df_storms[df_storms["sid"].str[:4].astype(int) >= 2000]
df_storms["atcf_id"] = df_storms["atcf_id"].str.lower()
```

```python
gdf_zma = zma.load_zma()
```

```python
adm0 = codab.load_codab_from_blob()
```

```python
adm0_buffer = gpd.GeoDataFrame(
    geometry=adm0.simplify(0.01).to_crs(3857).buffer(100_000).to_crs(4326)
)
```

```python
gdf_nhc = nhc.load_historical_forecasts(include_geometry=True)
```

## Process data

```python
# get points inside ZMA and making landfall
gdf_nhc["in_zma"] = gdf_nhc.within(gdf_zma.iloc[0].geometry)
gdf_nhc["landfall"] = gdf_nhc.within(adm0.iloc[0].geometry)
```

```python
gdf_nhc = gdf_nhc.merge(df_storms, how="left")
```

```python
gdf_nhc["season"] = gdf_nhc["valid_time"].dt.year
```

```python
gdf_nhc["name_season"] = (
    gdf_nhc["name"].str.capitalize() + " " + gdf_nhc["season"].astype(str)
)
```

```python
# observational is just the forecast at lt=0
gdf_obsv = gdf_nhc[gdf_nhc["issue_time"] == gdf_nhc["valid_time"]]
```

```python
gdf_obsv[gdf_obsv["sid"] == RAFAEL]
```

```python
# interpolating observational to make sure we get the right landfall time
dfs_interp = []
str_cols = ["atcf_id", "sid", "name", "season", "name_season"]
interp_cols = ["windspeed", "lat", "lon"]
for sid, group in gdf_obsv.groupby("sid"):
    group = group.sort_values("valid_time").set_index("valid_time")
    group_resampled = group.drop(columns="geometry").resample("30min").asfreq()
    group_resampled = group_resampled[interp_cols].interpolate().reset_index()
    # group_resampled[str_cols] = group.iloc[0][str_cols]
    for col in str_cols:
        group_resampled[col] = group.iloc[0][col]
    dfs_interp.append(group_resampled)
```

```python
df_obsv_interp = pd.concat(dfs_interp, ignore_index=True)
```

```python
gdf_obsv_interp = gpd.GeoDataFrame(
    data=df_obsv_interp,
    geometry=gpd.points_from_xy(df_obsv_interp["lon"], df_obsv_interp["lat"]),
    crs=4326,
)
```

```python
# have to re-check landfall and ZMA after interpolating
gdf_obsv_interp["in_zma"] = gdf_obsv_interp.within(gdf_zma.iloc[0].geometry)
gdf_obsv_interp["landfall"] = gdf_obsv_interp.within(adm0.iloc[0].geometry)
```

## Plotting

```python
def format_timedelta(td: pd.Timedelta) -> str:
    days = td.days
    hours = td.components.hours
    if days and hours:
        return f"{days} days, {hours} hours"
    elif days:
        return f"{days} days"
    else:
        return f"{hours} hours"
```

```python
lonmin, latmin, lonmax, latmax = gdf_zma.total_bounds
```

```python
def plot_sid_leadtime(sid):
    # filter by SID
    gdf_nhc_sid = gdf_nhc[gdf_nhc["sid"] == sid]

    # get earliest "triggering" forecast
    issue_time = gdf_nhc_sid[gdf_nhc_sid["in_zma"]]["issue_time"].min()
    gdf_issuetime = gdf_nhc_sid[gdf_nhc_sid["issue_time"] == issue_time]

    # get landfall time
    gdf_obsv_sid = gdf_obsv_interp[gdf_obsv_interp["sid"] == sid]
    landfall_time = gdf_obsv_sid[gdf_obsv_sid["landfall"]]["valid_time"].min()

    # filter observational to plot, otherwise plot is super zoomed out
    gdf_obsv_sid_plot = gdf_obsv_sid[
        (gdf_obsv_sid["lon"] >= lonmin - 1)
        & (gdf_obsv_sid["lat"] <= latmax + 1)
        & (gdf_obsv_sid["lat"] >= gdf_issuetime["lat"].min() - 1)
        & (gdf_obsv_sid["lon"] <= gdf_issuetime["lon"].max() + 1)
    ]

    first_fcast_time = gdf_nhc[gdf_nhc["sid"] == sid]["issue_time"].min()

    leadtime = landfall_time - issue_time
    leadtime_str = format_timedelta(leadtime)

    fig, ax = plt.subplots(dpi=200, figsize=(10, 10))

    adm0.boundary.plot(color="grey", linewidth=0.5, ax=ax)
    gdf_zma.boundary.plot(color="grey", linewidth=0.5, ax=ax, linestyle="--")

    fcast_color = "darkorange"
    obsv_color = "crimson"

    gdf_issuetime.drop(columns="geometry").plot(
        x="lon",
        y="lat",
        ax=ax,
        color=fcast_color,
        label="forecast",
        linewidth=0.5,
    )

    for _, row in gdf_issuetime.iterrows():
        ax.annotate(
            row["windspeed"],
            (row["lon"], row["lat"]),
            fontsize=10,
            va="center",
            ha="center",
            color=fcast_color,
            fontweight="bold",
        )

    gdf_obsv_sid_plot.drop(columns="geometry").plot(
        x="lon",
        y="lat",
        ax=ax,
        label="actual",
        color=obsv_color,
        linewidth=0.5,
    )

    landfall_row = gdf_obsv_sid_plot.set_index("valid_time").loc[landfall_time]
    ax.annotate(
        f'{landfall_row["windspeed"]:.0f}',
        (landfall_row["lon"], landfall_row["lat"]),
        fontsize=10,
        va="center",
        ha="center",
        color=obsv_color,
        fontweight="bold",
    )

    ax.set_title(
        f'{gdf_issuetime.iloc[0]["name_season"]}: forecast issued {issue_time}\n'
        f"Landfall at {landfall_time}\n"
        f"Leadtime to landfall: {leadtime_str}\n"
        f"(First ever forecast issued at {first_fcast_time})"
    )

    ax.legend(title="Track and\nwind speed (knots)")

    ax.axis("off")
```

```python
plot_sid_leadtime(GUSTAV)
```

```python
plot_sid_leadtime(IKE)
```

```python
plot_sid_leadtime(IRMA)
```

```python
plot_sid_leadtime(OSCAR)
```

```python
plot_sid_leadtime(IAN)
```

```python
plot_sid_leadtime(RAFAEL)
```

```python

```
