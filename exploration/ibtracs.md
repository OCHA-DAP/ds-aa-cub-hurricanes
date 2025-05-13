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
from src.datasources import ibtracs, zma
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
df_all
```

```python
df_all.dtypes
```

```python
def df_to_track_lines(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Convert a DataFrame of points to a GeoDataFrame of LineStrings,
    grouped by 'sid'.

    Parameters:
        df: DataFrame with 'sid', 'latitude', 'longitude' columns

    Returns:
        GeoDataFrame with one LineString per 'sid'
    """
    # Ensure sorted order if needed (e.g., by time)
    df = df.sort_values(["sid"])  # optionally add 'time' or similar

    # Group and build LineStrings
    lines = (
        df.groupby("sid")
        .apply(
            lambda group: LineString(
                zip(group["longitude"], group["latitude"])
            ),
            include_groups=False,
        )
        .reset_index(name="geometry")
    )

    return gpd.GeoDataFrame(lines, geometry="geometry", crs="EPSG:4326")
```

```python
gdf_lines = df_to_track_lines(df_all)
```

```python
gdf_lines
```

```python
fig, ax = plt.subplots()
gdf_zma.boundary.plot(ax=ax, color="k")
for sid, row in gdf_lines.iterrows():
    x, y = row.geometry.xy
    ax.plot(x, y, label=str(row["sid"]))  # matplotlib auto-assigns color

    ax.legend(title="sid", loc="upper right")
```

Looks like all the tracks are inside the ZMA, so should be good
