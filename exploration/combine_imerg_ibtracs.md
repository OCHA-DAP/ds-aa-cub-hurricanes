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

# Combine IMERG and IBTrACS

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import math

import ocha_stratus as stratus
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.datasources import ibtracs
from src.constants import *
```

```python
df_storms = ibtracs.load_storms()
```

```python
df_storms
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/impact/emdat_cerf_upto2024.parquet"
df_impact = stratus.load_parquet_from_blob(blob_name)
```

```python
df_impact["cerf"] = ~df_impact["Amount in US$"].isnull()
```

```python
cols = ["sid", "cerf", "Total Affected"]
df_impact[cols]
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/storm_stats/zma_stats.parquet"

df_stats_raw = stratus.load_parquet_from_blob(blob_name)
```

```python
df_stats = df_stats_raw.merge(df_storms)
df_stats
```

```python
cols = ["sid", "cerf", "Total Affected"]
df_stats = df_stats.merge(df_impact[cols], how="left")
df_stats["cerf"] = df_stats["cerf"].fillna(False)
```

```python
target_rp = 4
```

```python
df_stats
```

```python
df_stats_complete = df_stats.dropna(subset=["max_roll2_mean"])
```

```python
df_stats_complete
```

```python
total_years = df_stats_complete["season"].nunique()
```

```python
total_years
```

```python
target_year_count = math.floor((total_years + 1) / target_rp)
```

```python
target_year_count
```

```python
(total_years + 1) / target_year_count
```

```python
dicts = []

# check for each Cat limit
for cat_limit in CAT_LIMITS + [(0, None)]:
    dff = df_stats_complete[
        df_stats_complete["wind_speed_max"] >= cat_limit[0]
    ].copy()
    dff = dff.sort_values("max_roll2_mean", ascending=False)

    for rain_thresh in dff["max_roll2_mean"]:
        dfff = dff[dff["max_roll2_mean"] >= rain_thresh]
        trigger_year_count = dfff["season"].nunique()
        if trigger_year_count > target_year_count:
            break
        dict_out = {
            "max_roll2_mean": rain_thresh,
            "cat": cat_limit[1],
            "wind_speed_max": cat_limit[0],
            "trigger_year_count": trigger_year_count,
            "overall_rp": (total_years + 1) / dfff["season"].nunique(),
        }
    if dict_out["trigger_year_count"] == target_year_count:
        dicts.append(dict_out)

# check for windspeed-only
dff = df_stats_complete.sort_values("wind_speed_max", ascending=False).copy()
for wind_thresh in dff["wind_speed_max"]:
    dfff = dff[dff["wind_speed_max"] >= wind_thresh]
    trigger_year_count = dfff["season"].nunique()
    if trigger_year_count > target_year_count:
        break
    dict_out = {
        "max_roll2_mean": None,
        "cat": None,
        "wind_speed_max": wind_thresh,
        "trigger_year_count": trigger_year_count,
        "overall_rp": (total_years + 1) / dfff["season"].nunique(),
    }
dicts.append(dict_out)

df_threshs = pd.DataFrame(dicts)
```

```python
df_threshs
```

```python
df_threshs["wind_speed_max_kph"] = df_threshs["wind_speed_max"] * 1.852
```

```python
df_threshs
```

```python
CAT_LIMITS
```

```python
cat_colors = {
    "Trop. Storm": "dodgerblue",
    "Cat. 1": "gold",
    "Cat. 2": "darkorange",
    "Cat. 3": "orangered",
    "Cat. 4": "darkred",
    "Cat. 5": "purple",
}
```

```python
def plot_rain_wind_impact(lang: str = "EN"):
    if lang == "EN":
        title_text = f"{target_rp}-year return period trigger options"
        xlabel_text = "Max. wind speed while in ZMA (knots)"
        ylabel_text = (
            "Total 2-day precipitation, average over whole country (mm)"
        )
    elif lang == "ES":
        title_text = f"Opciones de activación del período de retorno de {target_rp} años"
        xlabel_text = "Velocidad máxima del viento en la ZMA (nudos)"
        ylabel_text = (
            "Precipitación total en 2 días, media en todo el país (mm)"
        )
    fig, ax = plt.subplots(dpi=200, figsize=(7, 7))

    ymax = df_stats_complete["max_roll2_mean"].max() * 1.1
    xmax = df_stats_complete["wind_speed_max"].max() * 1.1

    # Bubble sizes (handle NaNs as zero)
    bubble_sizes = df_stats_complete["Total Affected"].fillna(0)
    # Optional: scale for visual clarity
    bubble_sizes_scaled = (
        bubble_sizes / bubble_sizes.max() * 5000
    )  # Adjust 300 as needed

    # Plot bubbles
    ax.scatter(
        df_stats_complete["wind_speed_max"],
        df_stats_complete["max_roll2_mean"],
        s=bubble_sizes_scaled,
        alpha=0.3,
        color="crimson",
        edgecolor="none",
        zorder=1,
    )

    for _, row in df_stats_complete.iterrows():
        ax.annotate(
            row["name"].capitalize() + "\n" + str(row["season"]),
            (row["wind_speed_max"], row["max_roll2_mean"]),
            ha="center",
            va="center",
            fontsize=6,
            color="crimson" if row["cerf"] else "k",
            zorder=10 if row["cerf"] else 9,
        )

    for cat_name, row in df_threshs.set_index("cat").iterrows():
        if cat_name is None:
            continue
        color = cat_colors[cat_name]
        ax.axhline(row["max_roll2_mean"], color=color, linewidth=0.5)
        ax.axvline(row["wind_speed_max"], color=color, linewidth=0.5)
        ax.add_patch(
            patches.Rectangle(
                (row["wind_speed_max"], row["max_roll2_mean"]),  # bottom left
                xmax - row["wind_speed_max"],  # width
                ymax - row["max_roll2_mean"],  # height
                facecolor=color,
                alpha=0.1,
                zorder=0,
            )
        )
        ax.annotate(
            cat_name,
            (row["wind_speed_max"], 1),
            va="bottom",
            ha="right",
            color=color,
            rotation=90,
            fontweight="bold",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0",
                facecolor="white",  # Highlight color
                edgecolor="none",  # No border
                alpha=0.8,  # Transparency
            ),
        )

    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(bottom=0, top=ymax)

    ax.set_xlabel(xlabel_text)
    ax.set_ylabel(ylabel_text)
    ax.set_title(title_text)

    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
```

```python
plot_rain_wind_impact()
```

```python
plot_rain_wind_impact(lang="ES")
```

```python

```
