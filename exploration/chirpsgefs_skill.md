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

# CHIRPS-GEFS / CHIRPS / IMERG comparison

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import matplotlib.pyplot as plt
import pandas as pd

from src.datasources import chirps_gefs
from src.constants import *
```

## For storms specifically

```python
blob_name = f"{PROJECT_PREFIX}/processed/fcast_obsv_combined_stats.parquet"
df_stats = stratus.load_parquet_from_blob(blob_name)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/chirps/chirps_stats.parquet"
df_stats_chirps = stratus.load_parquet_from_blob(blob_name)
```

```python
df_stats = df_stats.merge(df_stats_chirps)
```

```python
df_stats
```

```python
df_stats_corr = df_stats.corr(numeric_only=True)
```

```python
def plot_rain_comparison(xcol, ycol):
    fig, ax = plt.subplots(dpi=150, figsize=(7, 7))
    df_stats.plot(x=xcol, y=ycol, ax=ax, linewidth=0, legend=False)
    xthresh = df_stats[xcol].quantile(2 / 3)
    ythresh = df_stats[ycol].quantile(2 / 3)

    df_stats["P"] = df_stats[xcol] >= xthresh
    df_stats["TP"] = (df_stats[ycol] >= ythresh) & df_stats["P"]

    tpr = df_stats["TP"].sum() / df_stats["P"].sum()

    ax.axhline(ythresh)
    ax.axvline(xthresh)
    for _, row in df_stats.iterrows():
        ax.annotate(
            row["name"].capitalize() + "\n" + str(row["year"]),
            (row[xcol], row[ycol]),
            ha="center",
            va="center",
        )
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    corr = df_stats_corr.loc[xcol, ycol]
    ax.set_title(f"{xcol}\n{corr=:.02f}\n{tpr=:.02f}")
    return fig, ax
```

```python
aggs = ["mean"] + [f"q{x}" for x in [50, 80, 90, 95]]

for agg in aggs:
    xcol, ycol = agg, f"{agg}_obsv"
    fig, ax = plot_rain_comparison(xcol, ycol)
    ax.set_xlabel("CHIRPS-GEFS")
    ax.set_ylabel("IMERG")
```

```python
xcol, ycol = "mean", "chirps_roll2_mean"
fig, ax = plot_rain_comparison(xcol, ycol)
ax.set_xlabel("CHIRPS-GEFS")
ax.set_ylabel("CHIRPS")
```

```python
xcol, ycol = "mean_obsv", "chirps_roll2_mean"
fig, ax = plot_rain_comparison(xcol, ycol)
ax.set_xlabel("IMERG")
ax.set_ylabel("CHIRPS")
```

## In general daily

```python
blob_name = f"{PROJECT_PREFIX}/processed/chirps/20240524_chirps_daily_historical_cuba.csv"
df_daily_chirps = stratus.load_csv_from_blob(blob_name, parse_dates=["date"])
df_daily_chirps = df_daily_chirps[["value", "date"]].rename(
    columns={"date": "valid_date"}
)
```

```python
df_daily_chirps = df_daily_chirps.sort_values("valid_date")
```

```python
df_daily_chirps["roll2"] = df_daily_chirps["value"].rolling(2).sum()
```

```python
df_daily_gefs = chirps_gefs.load_processed_chirps_gefs(variable_name="mean")
df_daily_gefs = df_daily_gefs.drop(columns="variable").rename(
    columns={"value": "roll2"}
)
```

```python
df_daily = df_daily_chirps.merge(
    df_daily_gefs, on="valid_date", suffixes=("_o", "_f")
)
```

```python
df_daily["leadtime"] = df_daily["valid_date"] - df_daily["issued_date"]
```

```python
xymax = df_daily[["roll2_f", "roll2_o"]].max().max()

dicts = []

xcol = "roll2_f"
ycol = "roll2_o"

for lt, group in df_daily.groupby("leadtime"):
    if lt.days == 0:
        continue
    fig, ax = plt.subplots(figsize=(7, 7))
    group.plot(
        x=xcol,
        y=ycol,
        ax=ax,
        linewidth=0,
        marker=".",
        markersize=1,
        alpha=0.5,
        legend=False,
    )

    xthresh = group[xcol].quantile(2 / 3)
    ythresh = group[ycol].quantile(2 / 3)

    group["P"] = group[xcol] >= xthresh
    group["TP"] = (group[ycol] >= ythresh) & group["P"]

    tpr = group["TP"].sum() / group["P"].sum()
    corr = group.corr().loc[xcol, ycol]
    ax.set_xlim((0, xymax))
    ax.set_ylim((0, xymax))
    ax.set_xlabel("CHIRPS-GEFS")
    ax.set_ylabel("CHIRPS")
    ax.set_title(f"leadtime={lt.days} days\n{corr=:0.2f}")

    dicts.append({"lt": lt.days, "corr": corr, "tpr": tpr})
```

```python
df_metrics = pd.DataFrame(dicts)
```

```python
df_metrics[df_metrics["lt"] <= 9].set_index("lt").plot()
```

```python
df_daily.groupby("leadtime")["roll2_f"].mean().plot()
```
