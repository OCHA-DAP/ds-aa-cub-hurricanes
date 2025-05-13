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
```

```python
df_cerf = stratus.load_csv_from_blob(blob_name)
```

```python
iso2 = "CU"
```

```python
df_cerf_cub = df_cerf[df_cerf["iso2"] == iso2]
df_cerf_cub
```

Here are the CERF storms. There are more recent ones in 2024, but none in 2023. Since we only have tracks for up to 2023, we can stick with this record for now. And we can drop the one without a `sid` since this was a tornado.

```python
df_cerf_cub = df_cerf_cub.dropna()
```

## EM-DAT

```python
blob_name = f"{PROJECT_PREFIX}/raw/impact/emdat-tropicalcyclone-2000-2022-processed-sids.csv"
df_emdat_2000_2022 = stratus.load_csv_from_blob(blob_name)
```

```python
blob_name = f"{PROJECT_PREFIX}/raw/impact/cub_emdat_2023only.csv"
df_emdat_2023 = stratus.load_csv_from_blob(blob_name)
df_emdat_2023["iso2"] = iso2
```

```python
df_emdat = pd.concat([df_emdat_2000_2022, df_emdat_2023], ignore_index=True)
```

```python
df_emdat_cub = df_emdat[df_emdat["iso2"] == iso2]
df_emdat_cub
```

There are two missing `sid`s.

```python
df_emdat_cub[df_emdat_cub["sid"].isnull()].iloc[0]
```

Looks like it was Tropical Storm Alex

```python
df_emdat_cub.loc[973, "sid"] = "2022154N21273"
```

```python
df_emdat_cub[df_emdat_cub["sid"].isnull()].iloc[0]
```

And this one was Idalia, clearly.

```python
df_emdat_cub.loc[1195, "sid"] = "2023239N21274"
```

## Combined

```python
df_combined = df_emdat_cub.merge(df_cerf_cub, on="sid", how="outer")
```

```python
df_combined
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/impact/emdat_cerf_upto2023.parquet"
stratus.upload_parquet_to_blob(df_combined, blob_name)
```
