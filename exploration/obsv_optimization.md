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

# Observational trigger optimization

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import math

import numpy as np
import pandas as pd
import ocha_stratus as stratus
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from tqdm.auto import tqdm

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
df_storms["name_season"] = (
    df_storms["name"].str.capitalize() + " " + df_storms["season"].astype(str)
)
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/storm_stats/zma_stats_imerg_quantiles.parquet"
)

df_stats_raw = stratus.load_parquet_from_blob(blob_name)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/impact/emdat_cerf_upto2024.parquet"
df_impact = stratus.load_parquet_from_blob(blob_name)
df_impact["cerf"] = ~df_impact["Amount in US$"].isnull()
cols = ["sid", "cerf", "Total Affected", "Amount in US$"]
df_impact = df_impact[cols]
```

```python
df_impact["Amount in US$"].mean()
```

```python
df_stats = df_stats_raw.merge(df_impact, how="left").merge(
    df_storms[["sid", "season", "name", "name_season"]], how="left"
)


def set_cerf(row):
    if row["season"] < 2006:
        return np.nan
    else:
        if row["cerf"] == True:
            return True
        else:
            return False


df_stats["cerf"] = df_stats.apply(set_cerf, axis=1)
df_stats["cerf_str"] = df_stats["cerf"].astype(str)
df_stats["Total Affected"] = df_stats["Total Affected"].fillna(0).astype(int)
df_stats
```

```python
target_rp = 4
total_years = 2024 - 2000 + 1
target_year_count = math.floor((total_years + 1) / target_rp)
```

```python
target_year_count
```

```python
total_affected_thresh = None
for total_affected in sorted(df_stats["Total Affected"], reverse=True):
    dff = df_stats[df_stats["Total Affected"] >= total_affected]
    display(dff)
    print(total_affected)
    if dff["season"].nunique() <= target_year_count:
        total_affected_thresh = total_affected
    else:
        break
```

```python
total_affected_thresh
```

```python
df_stats["target"] = df_stats["Total Affected"] >= total_affected_thresh
```

```python
df_stats["target_with_cerf"] = df_stats["target"] | df_stats["cerf"]
```

```python
df_stats[df_stats["target_with_cerf"]]
```

```python
df_plot = df_stats[df_stats["Total Affected"] > 0].sort_values(
    "Total Affected", ascending=True
)

cerf_color = "crimson"
noncerf_color = "dodgerblue"
precerf_color = "lightgrey"

# Set bar colors: red if cerf is True, else blue
colors = df_plot["cerf_str"].map(
    {"True": cerf_color, "False": noncerf_color, "nan": precerf_color}
)

# Plot
fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
ax.bar(df_plot["name_season"], df_plot["Total Affected"], color=colors)

# Formatting
ax.set_ylabel("Total Affected (EM-DAT)")
ax.set_xlabel("Name Season")
ax.set_title("Hurricane impact and CERF allocations")
ax.tick_params(axis="x", rotation=90)

ax.axhline(total_affected_thresh, linewidth=0.5, linestyle="--")

ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

ax.legend(
    handles=[
        mpatches.Patch(color=cerf_color, label="Yes"),
        mpatches.Patch(color=noncerf_color, label="No"),
        mpatches.Patch(color=precerf_color, label="Pre-CERF"),
    ],
    title="CERF allocation",
)

ax.spines.top.set_visible(False)
ax.spines.right.set_visible(False)

plt.tight_layout()
```

```python
cerf_year_count = df_stats[df_stats["cerf"] == True]["season"].nunique()
```

```python
cerf_total_years = 2024 - 2006 + 1
```

```python
cerf_rp = (cerf_total_years + 1) / cerf_year_count
```

```python
cerf_rp
```

```python
df_stats["Amount in US$"].mean()
```

```python
df_stats
```

```python
df_stats.columns
```

```python
dff
```

```python
dicts = []
for rain_col in tqdm([x for x in df_stats.columns if x.startswith("q")]):
    for rain_thresh in df_stats[rain_col].unique():
        for wind_speed_max in df_stats["wind_speed_max"].unique():
            for wind_speed_max_landfall in df_stats[
                "wind_speed_max_landfall"
            ].unique():
                dff = df_stats[
                    (df_stats[rain_col] >= rain_thresh)
                    & (
                        (df_stats["wind_speed_max"] >= wind_speed_max)
                        | (
                            df_stats["wind_speed_max_landfall"]
                            >= wind_speed_max_landfall
                        )
                    )
                ]
                n_years_triggered = dff["season"].nunique()
                if n_years_triggered == target_year_count:
                    dicts.append(
                        {
                            "rain_col": rain_col,
                            "rain_thresh": rain_thresh,
                            "wind_speed_max": wind_speed_max,
                            "wind_speed_max_landfall": wind_speed_max_landfall,
                            "target_sum": dff["target"].sum(),
                            "impact_sum": dff["Total Affected"].sum(),
                            "target_with_cerf_sum": dff[
                                "target_with_cerf"
                            ].sum(),
                            "cerf_sum": dff["cerf"].sum(),
                        }
                    )
```

```python
df_results = pd.DataFrame(dicts)
```

```python
df_results
```

```python
df_results["rain_col"][0][1:3]
```

```python
def get_rain_params(rain_col):
    agg_type = "mean_abv" if rain_col.endswith("mean_abv") else "quantile"
    q = int(rain_col[1:3])
    window = rain_col.split("_")[1]
    return agg_type, q, window


df_results[["rain_agg", "rain_q", "rain_window"]] = (
    df_results["rain_col"].apply(get_rain_params).apply(pd.Series)
)
```

```python
df_results
```

```python
fig, ax = plt.subplots(figsize=(7, 7))

# df_plot = df_results[df_results["cerf_sum"] == 3]
df_plot = df_results[df_results["target_sum"] == 5]

ax.scatter(
    df_plot["wind_speed_max"],
    df_plot["wind_speed_max_landfall"],
    color="k",
    edgecolors="none",
    alpha=0.05,
    marker=".",
)
ax.set_xlim(70, 150)
ax.set_ylim(70, 150)
```

```python
fig, ax = plt.subplots(figsize=(7, 7))

# df_plot = df_results[df_results["cerf_sum"] == 3]

ax.scatter(
    df_plot["wind_speed_max"],
    df_plot["rain_thresh"],
    color="k",
    edgecolors="none",
    alpha=0.05,
    marker=".",
)
# ax.set_xlim(70, 150)
# ax.set_ylim(70, 150)
```

```python
df_results.max()
```

```python
df_results.groupby(["rain_agg", "target_with_cerf_sum"]).size().rename(
    "count"
).reset_index().pivot(columns="rain_agg", index="target_with_cerf_sum").plot()
```

```python
fig, ax = plt.subplots()

for window, group in df_results.groupby("rain_agg"):
    group["impact_sum"].hist(ax=ax, alpha=0.5, label=window, bins=20)

ax.legend()
```

```python
fig, ax = plt.subplots()

for window, group in df_results.groupby("rain_window"):
    group["impact_sum"].hist(ax=ax, alpha=0.5, label=window, bins=20)

ax.legend()
```

```python
fig, ax = plt.subplots()

for q, group in df_results.groupby("rain_q"):
    group["impact_sum"].hist(ax=ax, alpha=0.5, label=q, bins=20)

ax.legend()
```

```python
df_results.groupby("rain_q")[
    ["impact_sum", "cerf_sum", "target_sum", "target_with_cerf_sum"]
].mean()
```

```python
df_results.groupby("rain_window")[
    ["impact_sum", "cerf_sum", "target_sum", "target_with_cerf_sum"]
].mean()
```

```python
df_results.sort_values(["cerf_sum", "impact_sum"], ascending=False).iloc[:20]
```

```python
df_results.sort_values(
    ["target_with_cerf_sum", "impact_sum"], ascending=False
).iloc[:20]
```

```python
# optimize_col = "target_with_cerf_sum"
optimize_col = "cerf_sum"
# optimize_col = "impact_sum"
df_results_top_single_col = df_results[
    df_results[optimize_col] == df_results[optimize_col].max()
]
df_results_top_duplicates = df_results_top_single_col[
    df_results_top_single_col["impact_sum"]
    == df_results_top_single_col["impact_sum"].max()
]
df_results_top = (
    df_results_top_duplicates.sort_values("rain_thresh")
    .drop_duplicates(
        subset=["wind_speed_max", "wind_speed_max_landfall", "rain_col"]
    )
    .sort_values("wind_speed_max")
    .drop_duplicates(
        subset=["rain_thresh", "wind_speed_max_landfall", "rain_col"]
    )
    .sort_values("wind_speed_max_landfall")
    .drop_duplicates(subset=["rain_thresh", "wind_speed_max", "rain_col"])
)
```

```python
df_results_top_single_col
```

```python
df_results_top_duplicates
```

```python
df_results_top
```

```python
# based on target_with_cerf_sum
selected_index = 11734
# based on cerf_sum
# selected_index = 11797
selected_trigger = df_results.loc[selected_index]
selected_trigger
```

```python
def plot_trigger_option(selected_index):
    selected_trigger = df_results.loc[selected_index]
    rain_col = selected_trigger["rain_col"]
    rain_thresh = selected_trigger["rain_thresh"]
    wind_speed_max = selected_trigger["wind_speed_max"]
    wind_speed_max_landfall = selected_trigger["wind_speed_max_landfall"]
    df_triggered = df_stats[
        (df_stats[rain_col] >= rain_thresh)
        & (
            (df_stats["wind_speed_max"] >= wind_speed_max)
            | (df_stats["wind_speed_max_landfall"] >= wind_speed_max_landfall)
        )
    ]

    ymax = df_stats[rain_col].max() * 1.1
    xmax = df_stats["wind_speed_max"].max() * 1.1

    fig, ax = plt.subplots(dpi=200, figsize=(7, 7))

    bubble_sizes = df_stats["Total Affected"].fillna(0)
    # Optional: scale for visual clarity
    bubble_sizes_scaled = (
        bubble_sizes / bubble_sizes.max() * 5000
    )  # Adjust 300 as needed

    # Plot bubbles
    ax.scatter(
        df_stats["wind_speed_max"],
        df_stats[rain_col],
        s=bubble_sizes_scaled,
        alpha=0.3,
        color="crimson",
        edgecolor="none",
        zorder=1,
    )

    for _, row in df_stats.iterrows():
        triggered = row["sid"] in df_triggered["sid"].to_list()
        ax.annotate(
            row["name"].capitalize() + "\n" + str(row["season"]),
            (row["wind_speed_max"], row[rain_col]),
            ha="center",
            va="center",
            fontsize=6,
            color="crimson" if row["cerf"] == True else "k",
            zorder=10 if row["cerf"] else 9,
            alpha=0.8,
            fontstyle="italic" if triggered else "normal",
            fontweight="bold" if triggered else "normal",
        )

    trig_color = "gold"
    ax.axvline(
        wind_speed_max,
        color=trig_color,
        linewidth=0.5,
        zorder=0,
    )
    ax.axvline(
        wind_speed_max_landfall,
        color=trig_color,
        linewidth=0.5,
        linestyle="--",
        zorder=0,
    )
    ax.axhline(
        rain_thresh,
        color=trig_color,
        linewidth=0.5,
        zorder=0,
    )
    ax.add_patch(
        mpatches.Rectangle(
            (wind_speed_max, rain_thresh),  # bottom left
            xmax - wind_speed_max,  # width
            ymax - rain_thresh,  # height
            facecolor=trig_color,
            alpha=0.1,
            zorder=0,
        )
    )

    for cat_value, cat_name in CAT_LIMITS:
        ax.annotate(
            cat_name + " -",
            (cat_value, 0),
            fontstyle="italic",
            color="grey",
            rotation=90,
            va="top",
            ha="center",
            fontsize=8,
        )

    ax.set_ylabel(rain_col)
    ax.set_xlabel("\nMax. wind speed while in ZMA (knots)")

    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(bottom=0, top=ymax)

    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
```

```python
# based on target_with_cerf_sum
selected_index = 11734
plot_trigger_option(selected_index)
```

```python
# based on cerf_sum
selected_index = 11797
plot_trigger_option(selected_index)
```

```python

```
