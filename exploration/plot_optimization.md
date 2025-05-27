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

# Optimization - trigger selection and plotting

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

from src.constants import *
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/storm_stats/stats_with_targets2.parquet"
)
df_stats = stratus.load_parquet_from_blob(blob_name)
```

```python
df_stats
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/trigger_metrics_ibtracs_imerg.parquet"
df_results = stratus.load_parquet_from_blob(blob_name)
```

```python
df_results
```

```python
def get_optimal_triggers(optimize_col: str, years_triggered: int):
    df_results_rp = df_results[df_results["n_years"] == years_triggered]
    df_results_top_single_col = df_results_rp[
        df_results_rp[optimize_col] == df_results_rp[optimize_col].max()
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
    return df_results_top
```

```python
get_optimal_triggers("target_sum", 6)
```

```python
get_optimal_triggers("target_with_cerf_sum", 6)
```

```python
get_optimal_triggers("cerf_sum", 6)
```

```python
get_optimal_triggers("target_sum", 7)
```

```python
get_optimal_triggers("target_with_cerf_sum", 7)
```

```python
get_optimal_triggers("cerf_sum", 7)
```

```python
get_optimal_triggers("target_sum", 8)
```

```python
get_optimal_triggers("target_with_cerf_sum", 8)
```

```python
get_optimal_triggers("cerf_sum", 8)
```

```python
def get_triggered_storms(selected_index):
    selected_trigger = df_results.loc[selected_index]
    # print(selected_trigger)
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
    return df_triggered
```

```python
def plot_trigger_option(selected_index):
    selected_trigger = df_results.loc[selected_index]
    rain_col = selected_trigger["rain_col"]
    rain_thresh = selected_trigger["rain_thresh"]
    wind_speed_max = selected_trigger["wind_speed_max"]
    wind_speed_max_landfall = selected_trigger["wind_speed_max_landfall"]
    df_triggered = get_triggered_storms(selected_index)

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
# 6 triggered years, target_with_cerf_sum
plot_trigger_option(225077)
```

```python
# 6 triggered years, cerf_sum
plot_trigger_option(226352)
```

```python
# 7 triggered years
plot_trigger_option(72319)
```

```python
# 8 triggered years, target_with_cerf_sum
plot_trigger_option(225075)
```

```python
# 8 triggered years, target_sum
plot_trigger_option(73172)
```

```python
df_stats
```

```python
def color_df(val):
    if val == "Yes":
        return "background-color: crimson"
    elif val == "No":
        return "background-color: dodgerblue"
    elif val == "Trig.":
        return "background-color: darkorange"
    else:
        return ""
```

```python
plot_triggers = [
    # (689, "4.3-yr RP"),
    # (226352, "4.3-yr RP"),
    (225077, "4.3-yr RP<br>"),
    (72319, "3.7-yr RP<br>"),
    (225075, "3.3-yr RP"),
    # (73172, "3.3-yr RP<br>[B]"),
]
```

```python
df_disp = df_stats.copy()
df_disp = df_disp.rename(columns={"cerf_str": "CERF", "name_season": "Storm"})
df_disp = df_disp.set_index("Storm")
df_disp["CERF"] = df_disp["CERF"].replace(
    {"True": "Yes", "False": "No", "nan": "pre-CERF"}
)


for trig_index, trig_name in plot_triggers:
    df_triggered = get_triggered_storms(trig_index)
    df_disp[trig_name] = (
        df_disp["sid"]
        .isin(df_triggered["sid"].to_list())
        .apply(lambda x: "Trig." if x else "No trig.")
    )

trig_cols = [x[1] for x in plot_triggers]

df_disp = df_disp.sort_values(["Total Affected"] + trig_cols, ascending=False)

cols = trig_cols + ["CERF", "Total Affected"]

display(
    df_disp[cols]
    .style.bar(
        subset="Total Affected",
        color="mediumpurple",
        # vmax=500000,
        props="width: 400px;",
    )
    .map(color_df)
    .set_table_styles(
        {
            "Total Affected": [
                {"selector": "th", "props": [("text-align", "left")]},
                {"selector": "td", "props": [("text-align", "left")]},
            ]
        }
    )
    .format({"Total Affected": "{:,}"})
)
```

```python

```
