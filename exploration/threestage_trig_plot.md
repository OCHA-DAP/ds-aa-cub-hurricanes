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

# Action/observational trigger plotting
<!-- markdownlint-disable MD013 -->
Picking out combinations of action (forecast, limited to 3 days) and observational triggers. All trigger combination options have the same overall return period (3.7 years, which is 7 triggering years in the period 2000-2024).

We varied, each for the forecast and the observational:

- wind speed threshold (while storm is in, or is forecast to be in, the ZMA)
- rainfall aggregation (`mean`, or quantiles 50, 80, 90, 95)
- rainfall threshold (two-day sum per pixel during the period that the storm is in, or is forecast to be in, the ZMA, ±1 day)

And we are looking to optimize for (maximizing):

- Sum of `Total Affected` from EM-DAT for the triggered storms
- Sum of `Amount in US$` from CERF for the triggered storms

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

from src.datasources.ibtracs import knots2cat
from src.constants import *
```

## Load and process data

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/fcast_obsv_combined_trigger_metrics.parquet"
)
df_metrics = stratus.load_parquet_from_blob(blob_name)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fcast_obsv_combined_stats.parquet"
df_stats = stratus.load_parquet_from_blob(blob_name)
```

### Determine theoretical maximum values

```python
# note that we have to pick a differet number of triggered years for CERF since it only started in 2006
cerf_target_years = int((2024 - 2006 + 1 + 1) / 3.7)
```

```python
# we see that there were 6 CERF years, but we can only hope to target up to 5, because otherwise the RP is too low
df_stats[df_stats["cerf"]].groupby("year")[
    "Amount in US$"
].sum().reset_index().sort_values("Amount in US$", ascending=False)
```

```python
max_cerf_amount = (
    df_stats[df_stats["cerf"]]
    .groupby("year")["Amount in US$"]
    .sum()
    .reset_index()
    .sort_values("Amount in US$", ascending=False)
    .iloc[:cerf_target_years]
    .sum()["Amount in US$"]
)
```

```python
target_years = 7
```

```python
max_total_affected = (
    df_stats.groupby("year")["Total Affected"]
    .sum()
    .reset_index()
    .sort_values("Total Affected", ascending=False)
    .iloc[:target_years]
    .sum()["Total Affected"]
)
```

```python
max_total_affected
```

### Add metrics and filter redundant values

```python
df_metrics["n_years_diff"] = (
    df_metrics["n_years_fcast"] - df_metrics["n_years_obsv"]
)
df_metrics["n_years_diff_abs"] = df_metrics["n_years_diff"].abs()
df_metrics["n_years_total"] = (
    df_metrics["n_years_fcast"] + df_metrics["n_years_obsv"]
)
```

```python
def drop_redundant_thresholds(df, min_cols, id_cols, drop_high=True):
    for min_col in min_cols:
        unique_cols = [x for x in min_cols if x != min_col] + id_cols
        df = df.sort_values(min_col, ascending=drop_high).drop_duplicates(
            subset=unique_cols
        )
    return df
```

```python
thresh_cols = [
    "fcast_wind",
    "fcast_rain_thresh",
    "obsv_wind",
    "obsv_rain_thresh",
]
rain_agg_cols = ["fcast_rain_col", "obsv_rain_col"]
impact_cols = [
    "Total Affected",
    "Total Deaths",
    "Total Damage, Adjusted ('000 US$)",
    "Amount in US$",
]
```

```python
df_metrics_lowest = drop_redundant_thresholds(
    df_metrics, thresh_cols, impact_cols + rain_agg_cols
)
```

```python
df_metrics_lowest
```

## Plotting functions

```python
def plot_thresh_scatter(
    x="Total Affected",
    y="Amount in US$",
    color="n_years_diff_abs",
    zorder_rev=True,
    fcast_pref_only=False,
    same_wind=False,
    same_rain_col=False,
    zero_intercept=False,
    df=None,  # override default dataframe
):
    fig, ax = plt.subplots(figsize=(7, 7))

    if df is None:
        df_plot = df_metrics_lowest.copy()
    else:
        df_plot = df.copy()

    if same_wind:
        df_plot = df_plot[df_plot["fcast_wind"] == df_plot["obsv_wind"]]
    if same_rain_col:
        df_plot = df_plot[
            df_plot["fcast_rain_col"]
            == df_plot["obsv_rain_col"].str.removesuffix("_obsv")
        ]

    if fcast_pref_only:
        df_plot = df_plot[df_plot["n_years_diff"] >= 0]

    for n_years_diff_abs, group in df_plot.groupby(color):
        group.plot(
            x=x,
            y=y,
            marker=".",
            linewidth=0,
            alpha=1,
            label=n_years_diff_abs,
            ax=ax,
            zorder=-n_years_diff_abs if zorder_rev else n_years_diff_abs,
        )

    ax.axhline(max_cerf_amount, linestyle="--", color="dodgerblue")
    ax.axvline(max_total_affected, linestyle="--", color="dodgerblue")

    if zero_intercept:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    ax.legend(title=color)
    ax.set_ylabel(y)
    ax.set_xlabel(x)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    return fig, ax
```

```python
max_wind = df_stats[["wind", "wind_obsv"]].max().max()
```

```python
def get_triggered_storms(index):
    trig = df_metrics_lowest.loc[index]

    df_stats["fcast_trig"] = (df_stats["wind"] >= trig["fcast_wind"]) & (
        df_stats[trig["fcast_rain_col"]] >= trig["fcast_rain_thresh"]
    )

    df_stats["obsv_trig"] = (df_stats["wind_obsv"] >= trig["obsv_wind"]) & (
        df_stats[trig["obsv_rain_col"]] >= trig["obsv_rain_thresh"]
    )

    return df_stats
```

```python
def plot_selected_threshs(index, impact_col="Total Affected"):
    trig_color = "gold"
    cerf_color = "crimson"

    fig, axs = plt.subplots(1, 2, figsize=(14, 7), dpi=200)

    trig = df_metrics_lowest.loc[index]
    # print(trig)

    df_stats = get_triggered_storms(index)

    figs = []
    for stage, ax in zip(["fcast", "obsv"], axs):
        other_stage = "fcast" if stage == "obsv" else "obsv"
        wind_col = "wind" if stage == "fcast" else "wind_obsv"
        wind_thresh = trig[f"{stage}_wind"]
        rain_col = trig[f"{stage}_rain_col"]
        rain_thresh = trig[f"{stage}_rain_thresh"]

        ymax = df_stats[rain_col].max() * 1.1
        xmax = max_wind * 1.1

        # fig, ax = plt.subplots(dpi=200, figsize=(7, 7))

        bubble_sizes = df_stats[impact_col].fillna(0)
        bubble_sizes_scaled = bubble_sizes / bubble_sizes.max() * 5000

        ax.scatter(
            df_stats[wind_col],
            df_stats[rain_col],
            s=bubble_sizes_scaled,
            c=df_stats["cerf"].apply(lambda x: cerf_color if x else "grey"),
            alpha=0.3,
            edgecolor="none",
            zorder=1,
        )

        for _, row in df_stats.iterrows():
            triggered = row[f"{stage}_trig"]
            other_triggered = row[f"{other_stage}_trig"]
            ax.annotate(
                row["name"].capitalize() + "\n" + str(row["year"]),
                (row[wind_col], row[rain_col]),
                ha="center",
                va="center",
                fontsize=6,
                color=cerf_color if row["cerf"] == True else "k",
                zorder=10 if row["cerf"] else 9,
                alpha=0.8,
                fontstyle="italic" if other_triggered else "normal",
                fontweight="bold" if triggered else "normal",
            )

        ax.axvline(
            wind_thresh,
            color=trig_color,
            linewidth=0.5,
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
                (wind_thresh, rain_thresh),  # bottom left
                xmax - wind_thresh,  # width
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

        ax.annotate(
            f" {wind_thresh:.0f} ",
            (wind_thresh, 0),
            color=trig_color,
            rotation=90,
            fontsize=10,
            va="top",
            ha="center",
            fontweight="bold",
        )

        ax.annotate(
            f" {rain_thresh:.1f} ",
            (0, rain_thresh),
            color=trig_color,
            fontsize=10,
            va="center",
            ha="right",
            fontweight="bold",
        )

        if rain_col == "mean":
            rain_agg_str = "mean"
        else:
            q = rain_col.removeprefix("q").removesuffix("_obsv")
            if q == "50":
                rain_agg_str = "median"
            else:
                rain_agg_str = f"{q}th quantile"
        ax.set_ylabel(
            f"Two-day rainfall, {rain_agg_str} over whole country (mm)"
        )
        ax.set_xlabel("\nMax. wind speed while in ZMA (knots)")

        ax.set_xlim(left=0, right=xmax)
        ax.set_ylim(bottom=0, top=ymax)

        ax.set_title(
            "Action (forecast)" if stage == "fcast" else "Observational"
        )

        ax.spines.top.set_visible(False)
        ax.spines.right.set_visible(False)

        # figs.append((fig, ax))
    return fig, axs
```

```python
def set_cerf_str(row):
    if row["cerf"]:
        return "Yes"
    else:
        if row["year"] >= 2006:
            return "No"
        else:
            return "pre-"


def color_df(val):
    if val == "Yes":
        return "background-color: crimson"
    elif val == "No":
        return "background-color: dodgerblue"
    elif val == "Trig.":
        return "background-color: darkorange"
    else:
        return ""


def disp_selected_threshs(index, impact_col="Total Affected"):
    trig_color = "gold"
    cerf_color = "crimson"

    df_disp = get_triggered_storms(index).copy()

    df_disp["CERF"] = df_disp.apply(set_cerf_str, axis=1)

    df_disp["Storm"] = (
        df_disp["name"].str.capitalize() + " " + df_disp["year"].astype(str)
    )
    df_disp["Action"] = df_disp["fcast_trig"].apply(
        lambda x: "Trig." if x else "No trig."
    )
    df_disp["Obsv."] = df_disp["obsv_trig"].apply(
        lambda x: "Trig." if x else "No trig."
    )

    cols = ["Action", "Obsv.", "CERF", impact_col]
    display(
        df_disp.set_index("Storm")[cols]
        .sort_values(impact_col, ascending=False)
        .style.bar(
            subset=impact_col,
            color="mediumpurple",
            # vmax=500000,
            props="width: 150px;",
        )
        .map(color_df)
        .set_table_styles(
            {
                impact_col: [
                    {"selector": "th", "props": [("text-align", "left")]},
                    {"selector": "td", "props": [("text-align", "left")]},
                ]
            }
        )
        .format({"Total Affected": "{:,}"})
    )
```

## Plot possible combinations

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Balanced

The cluster (option 1 in slides) where the same number of years are triggered with forecast and observational.
<!-- #endregion -->

```python
fig, ax = plot_thresh_scatter(
    "Total Affected", "Amount in US$", "n_years_diff_abs"
)
ax.axhline(2.8e7)
```

```python
df_metrics_balanced = df_metrics_lowest[
    df_metrics_lowest["n_years_diff_abs"] == 0
]
```

```python
df_metrics_balanced_high = df_metrics_balanced[
    df_metrics_balanced["Amount in US$"] >= 2.8e7
]
```

```python
df_metrics_balanced_high
```

```python
plot_selected_threshs(60832)
```

```python
disp_selected_threshs(60832)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Best Total Affected

Cluster (option 2 in slides) maximizing impact
<!-- #endregion -->

```python
fig, ax = plot_thresh_scatter(
    "Total Affected", "Amount in US$", "n_years_diff_abs"
)
ax.axvline(2.7e7)
```

```python
df_metrics_high_impact = df_metrics_lowest[
    df_metrics_lowest["Total Affected"] >= 2.7e7
]
```

```python
df_metrics_high_impact
```

```python
plot_selected_threshs(321743)
```

```python
disp_selected_threshs(321743)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Best CERF amount

Cluster (option 3 in slides) maximizing CERF amount
<!-- #endregion -->

```python
fig, ax = plot_thresh_scatter(
    "Total Affected", "Amount in US$", "n_years_diff_abs"
)
ax.axhline(3.3e7)
```

```python
df_metrics_high_cerf = df_metrics_lowest[
    df_metrics_lowest["Amount in US$"] >= 3.3e7
]
```

```python
df_metrics_high_cerf
```

```python
plot_selected_threshs(91667)
```

```python
disp_selected_threshs(91667)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Two years diff

Looking at that little cluster with a two-year difference. Doesn't look that remarkable, so left out of slides
<!-- #endregion -->

```python
fig, ax = plot_thresh_scatter(
    "Total Affected", "Amount in US$", "n_years_diff_abs"
)
ax.axhline(3.0e7)
```

```python
df_twoyears = df_metrics_lowest[
    (df_metrics_lowest["n_years_diff_abs"] == 2)
    & (df_metrics_lowest["Amount in US$"] >= 3.0e7)
]
```

```python
df_twoyears
```

```python
plot_selected_threshs(312998)
```

```python
disp_selected_threshs(312998)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### One year diff

Small cluster with second-best total impact (option 4 in slides)
<!-- #endregion -->

```python
fig, ax = plot_thresh_scatter(
    "Total Affected", "Amount in US$", "n_years_diff_abs"
)
ax.axvline(2.55e7)
```

```python
df_oneyear = df_metrics_lowest[
    (df_metrics_lowest["Total Affected"] >= 2.55e7)
    & (df_metrics_lowest["n_years_diff_abs"] == 1)
]
```

```python
df_oneyear
```

```python
plot_selected_threshs(321972)
```

```python
disp_selected_threshs(321972)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Forecast preference

Looking at how we can tilt probability even more towards forecast - basically just variants of option 1
<!-- #endregion -->

```python
fig, ax = plot_thresh_scatter(
    "Total Affected",
    "Amount in US$",
    "n_years_diff",
    fcast_pref_only=True,
    zorder_rev=False,
)
ax.axhline(2.7e7)
ax.axvline(2.3e7)
```

```python
df_fcast_pref = df_metrics_lowest[
    (df_metrics_lowest["Total Affected"] >= 2.3e7)
    & (df_metrics_lowest["Amount in US$"] >= 2.7e7)
    & (df_metrics_lowest["n_years_diff"] >= 0)
]
```

```python
df_fcast_pref
```

```python
df_fcast_only = df_fcast_pref[df_fcast_pref["n_years_diff"] == 0]
```

```python
df_fcast_only
```

```python
plot_selected_threshs(25477)
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
### Simplified triggers

Keeping only options with same rainfall aggregation, and same windspeed across forecast and observational
<!-- #endregion -->

```python
df_simplified = df_metrics_lowest[
    (
        df_metrics_lowest["fcast_rain_col"]
        == df_metrics_lowest["obsv_rain_col"].str.removesuffix("_obsv")
    )
    & (df_metrics_lowest["fcast_wind"] == df_metrics_lowest["obsv_wind"])
]
```

```python
df_metrics_lowest[
    (df_metrics_lowest["fcast_wind"] == df_metrics_lowest["obsv_wind"])
].sort_values("Total Affected", ascending=False)
```

```python
plot_thresh_scatter(
    "Total Affected",
    "Amount in US$",
    color="n_years_diff_abs",
    zero_intercept=True,
)
```

```python
plot_thresh_scatter(
    "Total Affected",
    "Amount in US$",
    color="n_years_diff_abs",
    same_wind=True,
    zero_intercept=True,
)
```

```python
plot_thresh_scatter(
    "Total Affected",
    "Amount in US$",
    color="n_years_diff_abs",
    same_rain_col=True,
    zero_intercept=True,
)
```

```python
plot_thresh_scatter(
    "Total Affected",
    "Amount in US$",
    color="n_years_diff_abs",
    same_rain_col=True,
    same_wind=True,
    zero_intercept=True,
)
```

```python
df_metrics_lowest[df_metrics_lowest[]]
```

```python
df_simplified.sort_values(
    ["Amount in US$", "Total Affected"],
    ascending=False,
)
```

```python
plot_selected_threshs(321844)
```

```python
disp_selected_threshs(321844)
```

```python
df_metrics_lowest
```

### "Reasonable" triggers

Trigger combos that meet:

- Same rainfall aggregation, limited to `q50` and `q80`
- Windspeed thresholds within one category of each other
- Forecast wind speed is >= to observational wind speed
- Windspeed thresholds at least Cat. 1 (64 knots)

```python
for stage in ["fcast", "obsv"]:
    df_metrics_lowest[f"{stage}_cat"] = df_metrics_lowest[
        f"{stage}_wind"
    ].apply(knots2cat)

df_metrics_lowest["cat_diff"] = (
    df_metrics_lowest["fcast_cat"] - df_metrics_lowest["obsv_cat"]
)

df_metrics_lowest["min_cat"] = df_metrics_lowest[
    ["fcast_cat", "obsv_cat"]
].min(axis=1)
```

```python
df_metrics_reasonable = df_metrics_lowest[
    (df_metrics_lowest["cat_diff"].isin([0, 1]))
    & (df_metrics_lowest["min_cat"] >= 1)
    & (
        df_metrics_lowest["fcast_rain_col"]
        == df_metrics_lowest["obsv_rain_col"].str.removesuffix("_obsv")
    )
    & (df_metrics_lowest["fcast_rain_col"].isin(["q50", "q80"]))
].copy()
```

```python
len(df_metrics_reasonable)
```

```python
fig, ax = plot_thresh_scatter(
    df=df_metrics_reasonable,
    zero_intercept=True,
    # color="n_years_diff",
    # zorder_rev=False,
)
```

```python
fig, ax = plot_thresh_scatter(
    df=df_metrics_reasonable,
    zero_intercept=True,
    # color="n_years_diff",
    # zorder_rev=False,
)
ax.axhline(2.8e7)
ax.axvline(2.55e7)
```

```python
df_metrics_reasonable_bestcerf = df_metrics_reasonable[
    df_metrics_reasonable["Amount in US$"] >= 2.8e7
]
```

```python
df_metrics_reasonable_bestcerf[
    df_metrics_reasonable_bestcerf["n_years_diff"] == 1
]
```

#### Option 1b

Maximizing CERF, taking closest to balanced (`n_years_diff == 1`), and lowest obsv wind threshold.

```python
plot_selected_threshs(13136)
```

```python
disp_selected_threshs(13136)
```

#### Option 4

Taking the maximum impact just ends up with option 4 again

```python
df_metrics_reasonable_bestimpact = df_metrics_reasonable[
    df_metrics_reasonable["Total Affected"] >= 2.55e7
]
```

```python
df_metrics_reasonable_bestimpact
```

```python
plot_selected_threshs(321972)
```

```python
disp_selected_threshs(321972)
```

## Melissa

Plotting Melissa on the Option 4 plot to show how extreme the rainfall forecast is.

```python
CHD_GREEN = "#1bb580"
```

```python
current_rain = 142.38818
current_wind = 130
fig, axs = plot_selected_threshs(321972)
axs[0].scatter(
    [current_wind],
    [current_rain],
    marker="x",
    color=CHD_GREEN,
    linewidths=3,
    s=100,
)
axs[0].annotate(
    "   Melissa  ",
    (current_wind, current_rain),
    va="center",
    ha="right",
    color=CHD_GREEN,
    fontweight="bold",
)
axs[1].remove()
axs[0].set_title("Rainfall vs. wind speed forecast")
```
