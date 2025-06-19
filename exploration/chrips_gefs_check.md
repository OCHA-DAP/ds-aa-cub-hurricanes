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

Checking that CHIRPS-GEFS outputs from DBX run are correct

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus

from src.datasources import codab
```

```python
adm0 = codab.load_codab_from_blob()
```

```python
adm0.plot()
```

```python
# test load
blob_name = "ds-aa-cub-hurricanes/raw/chirps_gefs/databricks_run/chirps-gefs-databricks_run_issued-2000-08-20_valid-2000-08-24.tif"
da_test = stratus.open_blob_cog(blob_name)
```

```python
da_test
```

```python
da_test.plot()
```

```python
ax = adm0.boundary.plot()
da_test.plot(ax=ax)
```

```python
# test download from browser
blob_name = "ds-aa-cub-hurricanes/raw/chirps_gefs/browser_download_test/data.2000.0824.tif"
da_test_browser = stratus.open_blob_cog(blob_name)
```

```python
da_test_browser_clip = da_test_browser.rio.clip(
    adm0.geometry, all_touched=True
)
```

```python
minx, miny, maxx, maxy = adm0.total_bounds
```

```python
da_test_browser
```

```python
da_test_browser_box = da_test_browser.sel(
    x=slice(minx, maxx), y=slice(maxy, miny)  # Note reversed lat if descending
)
```

```python
da_test_browser_box
```

```python
ax = adm0.boundary.plot()
da_test_browser_box.plot(ax=ax)
```

```python

```
