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

# Historical monitor plotting
<!-- markdownlint-disable MD013 -->
Plotting historical triggers based on historical forecasts

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

from src.datasources import ibtracs
from src.datasources.ibtracs import knots2cat
from src.constants import *
```

## Load and merge data

```python
df_storms = ibtracs.load_storms()
cols = ["sid", "atcf_id", "name"]
df_storms = df_storms[
    (df_storms["sid"].str[:4].astype(int) >= 2000)
    & (df_storms["genesis_basin"] == "NA")
][cols]
df_storms["atcf_id"] = df_storms["atcf_id"].str.lower()
```

```python
df_storms
```

```python
df_storms
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/impact/emdat_cerf_upto2024.parquet"
df_impact = stratus.load_parquet_from_blob(blob_name)
df_impact["cerf"] = ~df_impact["Amount in US$"].isnull()
cols = [
    "sid",
    "cerf",
    "Total Affected",
    "Total Deaths",
    "Total Damage, Adjusted ('000 US$)",
    "Amount in US$",
]
df_impact = df_impact[cols]
```

```python
df_impact.loc[df_impact["sid"] == IKE, "Total Affected"] = 2.6e6
```

```python
blob_name = blob_name = (
    f"{PROJECT_PREFIX}/processed/nhc/monitors_nhc_chirpsgefs.parquet"
)
df_monitors = stratus.load_parquet_from_blob(blob_name)
```

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
n_years = (
    df_monitors_max["issue_time"].max().year
    - df_monitors_max["issue_time"].min().year
    + 1
)
```

```python
target_rps = {
    "action": 3.7,
    "readiness": 3,
}
```

```python
target_trig_years = {
    lt_name: int((n_years + 1) / rp) for lt_name, rp in target_rps.items()
}
```

```python
target_trig_years
```

```python
lt_name = "action"
df_stats = (
    df_monitors_max[df_monitors_max["lt_name"] == lt_name]
    .merge(df_storms, how="left")
    .merge(df_impact, how="left")
)
df_stats
```

## Comparing options

```python
f"{6**2 * 70**6:,}"
```

```python
f"{6**2 * 70**4:,}"
```

```python
df_stats["cat"] = df_stats["wind"].apply(knots2cat)
```

```python
df_stats["cerf_num"] = df_stats["cerf"].apply(lambda x: 1 if x == True else 0)
```

```python
df_stats[df_stats["Total Affected"] > 0].corr(numeric_only=True)[
    "Total Affected"
]
```

```python
df_stats["Total Affected sq"] = df_stats["Total Affected"] ** 2
```

```python
df_stats.corr(numeric_only=True)["Total Affected sq"]
```

Above checks correlation with impact to get rough idea of what rainfall aggregation could be good. Maybe the median?

```python
rain_cols = [x for x in df_stats.columns if "q" in x or x == "mean"]
```

```python
df_stats[df_stats["cerf"] == True][
    ["name", "Total Affected"] + rain_cols
].sort_values("Total Affected", ascending=False).style.background_gradient(
    cmap="Reds",  # use a colormap where higher = darker
    axis=0,  # apply column-wise normalization
)
```

Plot above compares relative severity of different rainfall aggregations for each storm. Unfortunately, Oscar isn't the "worst" for any of them, and Rafael isn't even close.

```python
cat2knots = {5: 137, 4: 113, 3: 96, 2: 83, 1: 64}
```

```python
# iterate over categories to get the appropriate rainfall threshold for each one
rain_col = "q80"

rain_threshs = {}
for cat in [1, 2, 3, 4]:
    df_stats_cat = df_stats[df_stats["cat"] >= cat].sort_values(
        rain_col, ascending=False
    )
    rain_thresh = None
    for check_thresh in df_stats_cat[rain_col]:
        dff = df_stats_cat[df_stats_cat[rain_col] >= check_thresh]
        if dff["year"].nunique() > 7:
            break
        rain_thresh = check_thresh

    rain_thresh = (rain_thresh + check_thresh) / 2

    rain_threshs.update({cat: rain_thresh})
```

```python
rain_threshs
```

## Plotting

Plotting rain vs. wind for historical storms, with rain threshold set to get correct return period from a wind threshold fixed at a Category level.

```python
def plot_threshs(rain_thresh, wind_cat_thresh, rain_col):
    wind_thresh = cat2knots[wind_cat_thresh]
    ymax = df_stats[rain_col].max() * 1.1
    xmax = df_stats["wind"].max() * 1.1

    fig, ax = plt.subplots(dpi=200, figsize=(7, 7))

    bubble_sizes = df_stats["Total Affected"].fillna(0)
    # Optional: scale for visual clarity
    bubble_sizes_scaled = (
        bubble_sizes / bubble_sizes.max() * 5000
    )  # Adjust 300 as needed

    # Plot bubbles
    ax.scatter(
        df_stats["wind"],
        df_stats[rain_col],
        s=bubble_sizes_scaled,
        alpha=0.3,
        color="crimson",
        edgecolor="none",
        zorder=1,
    )

    for _, row in df_stats.iterrows():
        triggered = (row[rain_col] >= rain_thresh) & (
            row["wind"] >= wind_thresh
        )
        ax.annotate(
            row["name"].capitalize() + "\n" + str(row["year"]),
            (row["wind"], row[rain_col]),
            ha="center",
            va="center",
            fontsize=6,
            color="crimson" if row["cerf"] == True else "k",
            zorder=10 if row["cerf"] else 9,
            alpha=0.8,
            fontstyle="italic" if triggered else "normal",
            fontweight="bold" if triggered else "normal",
        )

    trig_color = "orange"
    ax.axvline(
        wind_thresh,
        color=trig_color,
        linewidth=0.5,
        zorder=0,
    )
    ax.annotate(
        f"  Cat. {wind_cat_thresh} \n({wind_thresh} kn) ",
        (wind_thresh, 0),
        va="top",
        ha="center",
        rotation=90,
        color=trig_color,
        fontsize=8,
        fontstyle="italic",
    )
    ax.axhline(
        rain_thresh,
        color=trig_color,
        linewidth=0.5,
        zorder=0,
    )
    ax.annotate(
        f"{rain_thresh:.0f} mm",
        (0, rain_thresh),
        va="center",
        ha="right",
        color=trig_color,
        fontsize=8,
        fontstyle="italic",
    )
    ax.add_patch(
        mpatches.Rectangle(
            (wind_thresh, rain_thresh),  # bottom left
            xmax - wind_thresh,  # width
            ymax - rain_thresh,  # height
            facecolor=trig_color,
            alpha=0.07,
            zorder=0,
        )
    )

    ax.annotate(
        "\n"
        "    Size of bubble corresponds to\n"
        "    total number of people affected [EM-DAT]\n\n"
        "    Red text indicates CERF allocation",
        (0, ymax),
        va="top",
        fontsize=6,
        fontstyle="italic",
        color="grey",
    )
    ax.annotate(
        "\n    Triggered    \n    storms    ",
        (wind_thresh, ymax),
        ha="left",
        va="top",
        color=trig_color,
        fontstyle="italic",
    )

    if rain_col == "q50":
        ylabel = (
            "Two-day rainfall, median over whole country (mm) [CHIRPS-GEFS]"
        )
    else:
        ylabel = rain_col
    ax.set_ylabel(ylabel)
    ax.set_xlabel("\nMax. wind speed while in 250 km buffer (knots) [NHC]")
    ax.set_title(
        "Cuba triggered storms based on forecasts (since 2000)\n"
        f"Cat. {cat} and {rain_thresh:.0f} mm rainfall trigger"
    )

    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(bottom=0, top=ymax)

    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    return fig, ax
```

```python
cat = 1
plot_threshs(rain_threshs[cat], cat, rain_col)
```

```python
cat = 2
fig, ax = plot_threshs(rain_threshs[cat], cat, rain_col)
```

```python
cat = 3
plot_threshs(rain_threshs[cat], cat, rain_col)
```

```python
cat = 4
plot_threshs(rain_threshs[cat], cat, rain_col)
```

```python

```
