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

# CHIRPS-GEFS

Process CHIRPS-GEFS

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import pandas as pd
import xarray as xr
import ocha_stratus as stratus
from sqlalchemy import text
from tqdm.auto import tqdm
from azure.core.exceptions import ResourceNotFoundError

from src.datasources.chirps_gefs import ChirpsGefsConfig, ChirpsGefsLoader
from src.datasources import codab
from src.constants import *
from src.utils.database import create_chirps_gefs_table
```

```python
TABLE_NAME = f'{PROJECT_PREFIX.replace("-", "_")}_chirps_gefs'
```

```python
TABLE_NAME
```

```python
# create projects schema - this should never need running again
# try:
#     with stratus.get_engine("dev", write=True).begin() as conn:
#         conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"))
# except Exception as e:
#     print("Schema creation failed:", e)
```

```python
# create table in projects schema
# create_chirps_gefs_table(TABLE_NAME, stratus.get_engine("dev", write=True))
```

```python
adm0 = codab.load_codab_from_blob()
```

```python
adm0.plot()
```

```python
config = ChirpsGefsConfig(geometry=adm0, region_name="databricks_run")
loader = ChirpsGefsLoader(config)
```

```python
# check which dates we already have processed
query = f"""
SELECT issued_date, COUNT(*) AS n_entries
FROM projects.{TABLE_NAME}
GROUP BY issued_date
ORDER BY issued_date;
"""

df_existing_dates = pd.read_sql(
    query, stratus.get_engine("dev"), parse_dates="issued_date"
)
```

```python
df_existing_dates
```

```python
# check complete dates - we have to do this because some dates might not have all the stats
df_complete_dates = df_existing_dates[
    df_existing_dates["n_entries"] == df_existing_dates["n_entries"].max()
]
```

```python
df_complete_dates
```

```python
def write_to_db(df_out):
    df_out.to_sql(
        name=TABLE_NAME,
        con=stratus.get_engine("dev", write=True),
        schema="projects",
        if_exists="append",
        method=stratus.postgres_upsert,
        index=False,
    )
```

```python
quantiles = [0.5, 0.8, 0.9, 0.95, 0.99]
```

```python
start_date = "2000-01-01"
end_date = "2024-12-31"

full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")

missing_date_range = full_date_range[
    ~full_date_range.isin(df_complete_dates["issued_date"].to_list())
]

print("processing for dates:")
display(missing_date_range)

verbose = False

# takes around 24 hours depending on internet connection
for issue_date in tqdm(missing_date_range):
    das_i = []
    for leadtime in range(16):
        valid_date = issue_date + pd.Timedelta(days=leadtime)
        try:
            da_in = loader.load_raster(issue_date, valid_date)
            da_in["valid_date"] = valid_date
            das_i.append(da_in)
        except ResourceNotFoundError as e:
            if verbose:
                print(f"{e} for {issue_date} {valid_date}")

    if das_i:
        da_i = xr.concat(das_i, dim="valid_date")
        da_i_clip = da_i.rio.clip(adm0.geometry, all_touched=True)
        da_rolling2 = da_i_clip.rolling(valid_date=2).sum()
        for quantile in quantiles:
            da_quantile_threshs = da_rolling2.quantile(
                quantile, dim=["x", "y"]
            )
            df_out = (
                da_quantile_threshs.to_dataframe("value")
                .reset_index()
                .drop(columns="quantile")
            )
            df_out["variable"] = f"q{quantile*100:.0f}"
            df_out["issued_date"] = issue_date
            write_to_db(df_out)
        da_means = da_rolling2.mean(dim=["x", "y"])
        df_out = (
            da_means.to_dataframe("value")
            .reset_index()
            .drop(columns="spatial_ref")
        )
        df_out["variable"] = "mean"
        df_out["issued_date"] = issue_date
        write_to_db(df_out)
    else:
        if verbose:
            print(f"no files for issue_date {issue_date}")
```

```python

```
