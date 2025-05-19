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

# IBTrACS
<!-- markdownlint-disable MD013 -->

Load IBTrACS from Postgres, and plot just to check the right stuff has been loaded.

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import matplotlib.pyplot as plt
import geopandas as gpd

from src.datasources import ibtracs, zma, codab
from src.constants import *
```

```python
adm0 = codab.load_codab_from_blob()
```

```python
adm0.plot()
```

```python
gdf_zma = zma.load_zma()
```

```python
total_bounds = gdf_zma.total_bounds
```

```python
total_bounds
```

```python
df_all = ibtracs.load_ibtracs_in_bounds(*total_bounds)
```

```python
df_all.sort_values("valid_time", ascending=False)
```

```python
df_all
```

```python
gdf_all = gpd.GeoDataFrame(
    data=df_all,
    geometry=gpd.points_from_xy(df_all["longitude"], df_all["latitude"]),
    crs=4326,
)
```

```python
gdf_filtered = gdf_all[gdf_all.within(gdf_zma.iloc[0].geometry)].copy()
```

```python
gdf_filtered["landfall"] = gdf_filtered.within(adm0.iloc[0].geometry)
```

```python
gdf_filtered_recent = gdf_filtered[gdf_filtered["valid_time"].dt.year >= 2000]
```

```python
gdf_filtered_recent
```

```python
fig, ax = plt.subplots(dpi=300)

gdf_filtered_recent.drop(columns="geometry")[
    ~gdf_filtered_recent["landfall"]
].plot(x="longitude", y="latitude", ax=ax, kind="scatter", color="dodgerblue")

gdf_filtered_recent.drop(columns="geometry")[
    gdf_filtered_recent["landfall"]
].plot(x="longitude", y="latitude", ax=ax, kind="scatter", color="crimson")


adm0.boundary.plot(linewidth=0.5, ax=ax, color="k")
gdf_zma.boundary.plot(linewidth=0.5, ax=ax, color="grey")
ax.axis("off")
```

Looks like all the tracks are inside the ZMA, so should be good

```python
gdf_filtered_recent.drop(columns="geometry")
```

```python
blob_name = f"{PROJECT_PREFIX}/processed/ibtracs/zma_tracks_2000-2024.parquet"
stratus.upload_parquet_to_blob(
    gdf_filtered_recent.drop(columns="geometry"), blob_name
)
```
