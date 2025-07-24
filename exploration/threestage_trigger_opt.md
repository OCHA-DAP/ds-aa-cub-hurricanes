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

# Three-stage trigger optimization
<!-- markdownlint-disable MD013 -->
Iterate over options to get trigger options that meet 3.7-yr RP. Variables iterated over:

- wind speed threshold (while storm is in, or is forecast to be in, the ZMA)
- rainfall aggregation (`mean`, or quantiles 50, 80, 90, 95)
- rainfall threshold (two-day sum per pixel during the period that the storm is in, or is forecast to be in, the ZMA, ±1 day)

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
import optuna
from tqdm.auto import tqdm

from src.datasources import ibtracs
from src.datasources.ibtracs import knots2cat
from src.constants import *
```

## Load and combine data

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
df_impact.loc[df_impact["sid"] == IKE, "Total Affected"] = 2.6e6
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/storm_stats/zma_stats_imerg_quantiles.parquet"
)

df_stats_obsv = stratus.load_parquet_from_blob(blob_name)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/storm_stats/zma_stats.parquet"

df_stats_obsv_meanonly = stratus.load_parquet_from_blob(blob_name)
```

```python
drop_cols = [
    x
    for x in df_stats_obsv
    if any([s in x for s in ["_total", "_abv", "_roll3", "_landfall"]])
]
df_stats_obsv = df_stats_obsv.drop(columns=drop_cols)
```

```python
df_stats_obsv = df_stats_obsv.merge(
    df_stats_obsv_meanonly[["sid", "max_roll2_mean"]]
).drop(columns=[x for x in df_stats_obsv if "valid" in x])
```

```python
blob_name = blob_name = (
    f"{PROJECT_PREFIX}/processed/nhc/monitors_nhc_chirpsgefs.parquet"
)
df_monitors_fcast = stratus.load_parquet_from_blob(blob_name)
```

```python
df_stats_fcast = (
    df_monitors_fcast[df_monitors_fcast["lt_name"] == "action"]
    .groupby("atcf_id")
    .max()
    .reset_index()
    .dropna()
    .drop(columns=["issue_time", "lt_name"])
)
```

```python
df_stats = (
    df_impact.merge(df_storms, how="left")
    .merge(df_stats_fcast, how="left")
    .merge(
        df_stats_obsv.rename(
            columns={x: x.replace("roll2", "obsv") for x in df_stats_obsv}
            | {"max_roll2_mean": "mean_obsv", "wind_speed_max": "wind_obsv"}
        ),
        how="left",
    )
)
df_stats["year"] = df_stats["sid"].str[:4]

int_cols = [
    "year",
    # "wind",
    # "wind_obsv",
    "Total Affected",
    "Total Deaths",
    "Total Damage, Adjusted ('000 US$)",
    "Amount in US$",
]
df_stats[int_cols] = df_stats[int_cols].fillna(0)
df_stats[int_cols] = df_stats[int_cols].astype(int)
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/fcast_obsv_combined_stats.parquet"
stratus.upload_parquet_to_blob(df_stats, blob_name)
```

## Optimization

```python
target_years = 7
```

```python
impact_cols = [
    "Total Affected",
    "Total Deaths",
    "Total Damage, Adjusted ('000 US$)",
    "Amount in US$",
    "cerf",
]
```

```python
# fcast_rain_cols = ["mean"]
# fcast_rain_cols = ["mean"] + [f"q{x}" for x in [50, 80, 90, 95, 99]]
fcast_rain_cols = ["mean"] + [f"q{x}" for x in [50, 80, 95]]
```

```python
obsv_rain_cols = [x + "_obsv" for x in fcast_rain_cols]
```

```python
fcast_rain_cols
```

```python
rows = []

count = 0
for fcast_wind_thresh in tqdm(
    df_stats["wind"].unique(), disable=tqdm_level < 1
):
    for obsv_wind_thresh in tqdm(
        df_stats["wind_obsv"].unique(), disable=tqdm_level < 2
    ):
        dff_wind = df_stats[
            (df_stats["wind"] >= fcast_wind_thresh)
            | (df_stats["wind_obsv"] >= obsv_wind_thresh)
        ]
        # if filtering already limits to too few years, skip
        # since we know that further filtering will only result in lower number
        if dff_wind["year"].nunique() < target_years:
            continue
        for fcast_rain_col in fcast_rain_cols:
            for fcast_rain_thresh in df_stats[fcast_rain_col].unique():
                for obsv_rain_col in obsv_rain_cols:
                    for obsv_rain_thresh in df_stats[obsv_rain_col].unique():
                        count += 1
                        # check years triggered with forecast
                        triggered_fcast = (
                            df_stats["wind"] >= fcast_wind_thresh
                        ) & (df_stats[fcast_rain_col] >= fcast_rain_thresh)
                        # check years triggered with forecast
                        triggered_obsv = (
                            df_stats["wind_obsv"] >= obsv_wind_thresh
                        ) & (df_stats[obsv_rain_col] >= obsv_rain_thresh)

                        # check years triggered with either
                        dff_triggered = df_stats[
                            triggered_fcast | triggered_obsv
                        ]
                        if dff_triggered["year"].nunique() == target_years:
                            row_out = dff_triggered[impact_cols].sum()
                            row_out["fcast_wind"] = fcast_wind_thresh
                            row_out["fcast_rain_col"] = fcast_rain_col
                            row_out["fcast_rain_thresh"] = fcast_rain_thresh
                            row_out["obsv_wind"] = obsv_wind_thresh
                            row_out["obsv_rain_col"] = obsv_rain_col
                            row_out["obsv_rain_thresh"] = obsv_rain_thresh
                            row_out["n_years_fcast"] = df_stats[
                                triggered_fcast
                            ]["year"].nunique()
                            row_out["n_years_obsv"] = df_stats[triggered_obsv][
                                "year"
                            ].nunique()
                            rows.append(row_out)
```

```python
count
```

```python
df_metrics = pd.concat(rows, axis=1).T
```

```python
df_metrics.sort_values("Total Affected", ascending=False)
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/fcast_obsv_combined_trigger_metrics.parquet"
)
stratus.upload_parquet_to_blob(df_metrics, blob_name)
```

## Using `optima`

Can ignore for now, didn't use, was just a way to try and speed things up. But I don't think the problem is well suited to this approach

```python
# --- Storage for all results ---
results_rows = []

# --- Loop over each impact column ---
for impact_col in impact_cols:
    print(f"\n🔍 Optimizing for impact column: {impact_col}")

    def objective(trial):
        # Sample threshold values
        fcast_wind_thresh = trial.suggest_int(
            "fcast_wind_thresh",
            int(df_stats["wind"].min()),
            int(df_stats["wind"].max()),
            step=5,
        )
        obsv_wind_thresh = trial.suggest_int(
            "obsv_wind_thresh",
            int(df_stats["wind_obsv"].min()),
            int(df_stats["wind_obsv"].max()),
            step=5,
        )
        fcast_rain_col = trial.suggest_categorical(
            "fcast_rain_col", fcast_rain_cols
        )
        obsv_rain_col = trial.suggest_categorical(
            "obsv_rain_col", obsv_rain_cols
        )
        fcast_rain_thresh = trial.suggest_int(
            "fcast_rain_thresh",
            int(df_stats[fcast_rain_col].min()),
            int(df_stats[fcast_rain_col].max()),
        )
        obsv_rain_thresh = trial.suggest_int(
            "obsv_rain_thresh",
            int(df_stats[obsv_rain_col].min()),
            int(df_stats[obsv_rain_col].max()),
        )

        # Filtering logic (trigger condition)
        dff_triggered = df_stats[
            (
                (df_stats["wind"] >= fcast_wind_thresh)
                & (df_stats[fcast_rain_col] >= fcast_rain_thresh)
            )
            | (
                (df_stats["wind_obsv"] >= obsv_wind_thresh)
                & (df_stats[obsv_rain_col] >= obsv_rain_thresh)
            )
        ]

        # Skip if not enough years are represented
        if dff_triggered["year"].nunique() != target_years:
            raise optuna.exceptions.TrialPruned()

        # Maximize the sum of the selected impact column
        score = dff_triggered[impact_col].sum()
        return -score  # Optuna minimizes, so negate

    # Create and run study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200, show_progress_bar=True)
    optuna.visualization.plot_optimization_history(study).show()
    # Store result as a row dict
    row = {
        "impact_col": impact_col,
        "best_score": -study.best_value,
        **study.best_params,
    }
    results_rows.append(row)

# --- Final results DataFrame ---
results_df = pd.DataFrame(results_rows)
print("\n📊 Optimization results:")
print(results_df)
```

```python
optuna.visualization.plot_optimization_history(study).show()
```

```python
for impact_col, row in results_df.set_index("impact_col").iterrows():
    df_disp = df_stats.copy()
    fcast_wind_thresh = row["fcast_wind_thresh"]
    obsv_wind_thresh = row["obsv_wind_thresh"]
    fcast_rain_col = row["fcast_rain_col"]
    fcast_rain_thresh = row["fcast_rain_thresh"]
    obsv_rain_col = row["obsv_rain_col"]
    obsv_rain_thresh = row["obsv_rain_thresh"]
    df_disp["trig"] = (
        (df_stats["wind"] >= fcast_wind_thresh)
        & (df_stats[fcast_rain_col] >= fcast_rain_thresh)
    ) | (
        (df_stats["wind_obsv"] >= obsv_wind_thresh)
        & (df_stats[obsv_rain_col] >= obsv_rain_thresh)
    )
    print(impact_col)
    print(row)
    display(df_disp.sort_values(impact_col, ascending=False))
```

```python
tqdm_level = 1
```
