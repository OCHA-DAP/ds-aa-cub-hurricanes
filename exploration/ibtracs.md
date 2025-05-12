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

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
from src.datasources import ibtracs, zmi
```

```python
gdf_zmi = zmi.load_zmi()
```

```python
total_bounds = gdf_zmi.total_bounds
```

```python
total_bounds
```

```python
df_all = ibtracs.load_ibtracs_in_bounds(*total_bounds)
```

```python
df_all.dtypes
```

```python
df_all.groupby("sid").agg()
```
