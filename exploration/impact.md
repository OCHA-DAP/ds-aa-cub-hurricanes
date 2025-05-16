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

# Impact

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import pandas as pd

from src.constants import *
```

## CERF

```python
blob_name = f"{PROJECT_PREFIX}/raw/impact/cerf-storms-with-sids-2024-02-27.csv"
df_cerf = stratus.load_csv_from_blob(
    blob_name, parse_dates=["Allocation date"]
)
```

```python
iso2 = "CU"
```

```python
df_cerf_cub_old = df_cerf[df_cerf["iso2"] == iso2]
```

```python
df_cerf_cub_old
```

```python
# add 2024 allocations
blob_name = f"{PROJECT_PREFIX}/raw/impact/cerf_cub_storms.csv"
df_cerf_cub_recent = stratus.load_csv_from_blob(
    blob_name, parse_dates=["Allocation date"]
)
df_cerf_cub_recent["Amount in US$"] = (
    df_cerf_cub_recent["Amount in US$"].str.replace(",", "").astype(int)
)
```

```python
df_cerf_cub_recent
```

```python
df_cerf_cub = df_cerf_cub_old.merge(df_cerf_cub_recent, how="outer")
```

```python
df_cerf_cub
```

```python
# Oscar and Rafael
df_cerf_cub.loc[5, "sid"] = "2024293N21294"
df_cerf_cub.loc[6, "sid"] = "2024309N13283"
# drop tornado
df_cerf_cub = df_cerf_cub[df_cerf_cub["Allocation date"] != "2019-03-05"]
```

```python
df_cerf_cub
```

## EM-DAT

```python
blob_name = f"{PROJECT_PREFIX}/raw/impact/emdat-tropicalcyclone-2000-2022-processed-sids.csv"
df_emdat_2000_2022 = stratus.load_csv_from_blob(blob_name)
```

```python
blob_name = f"{PROJECT_PREFIX}/raw/impact/cub_emdat_2023-2024.csv"
df_emdat_recent = stratus.load_csv_from_blob(blob_name)
df_emdat_recent["iso2"] = iso2
```

```python
df_emdat = pd.concat([df_emdat_2000_2022, df_emdat_recent], ignore_index=True)
```

```python
df_emdat_cub = df_emdat[df_emdat["iso2"] == iso2]
df_emdat_cub
```

```python
df_emdat_cub[df_emdat_cub["sid"].isnull()]
```

There are a few missing `sid`s.

```python
df_emdat_cub[df_emdat_cub["sid"].isnull()].iloc[0]
```

```python
# Tropical Storm Alex
df_emdat_cub.loc[973, "sid"] = "2022154N21273"
```

```python
df_emdat_cub[df_emdat_cub["sid"].isnull()].iloc[0]
```

```python
# Helene
df_emdat_cub.loc[1195, "sid"] = "2024268N17278"
```

```python
df_emdat_cub[df_emdat_cub["sid"].isnull()].iloc[0]
```

```python
# Rafael
df_emdat_cub.loc[1196, "sid"] = "2024309N13283"
```

```python
df_emdat_cub[df_emdat_cub["sid"].isnull()].iloc[0]
```

```python
# Oscar
df_emdat_cub.loc[1197, "sid"] = "2024293N21294"
```

```python
df_emdat_cub[df_emdat_cub["sid"].isnull()].iloc[0]
```

```python
# Idalia
df_emdat_cub.loc[1198, "sid"] = "2023239N21274"
```

```python
df_emdat_cub[df_emdat_cub["sid"].isnull()].iloc[0]
```

## Combined

```python
df_combined = df_emdat_cub.merge(df_cerf_cub, on="sid", how="outer")
```

```python
df_combined
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/impact/emdat_cerf_upto2024.parquet"
stratus.upload_parquet_to_blob(df_combined, blob_name)
```

```python

```
