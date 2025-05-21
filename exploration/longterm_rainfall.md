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

# Long-term rainfall

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import matplotlib.pyplot as plt
import pandas as pd

from src.datasources import imerg
```

```python
df_imerg = imerg.load_imerg(pcode="CU")
```

```python
df_imerg = df_imerg.sort_values("valid_date")
```

```python
windows = [15, 30, 60, 90, 120]

for window in windows:
    df_imerg[f"roll{window}"] = df_imerg["mean"].rolling(window).sum()
```

```python
df_imerg["year"] = df_imerg["valid_date"].dt.year
df_imerg["doy"] = df_imerg["valid_date"].dt.dayofyear
df_imerg["plot_date"] = pd.to_datetime(df_imerg["doy"], format="%j")
```

```python
df_imerg
```

```python
for window in windows:
    fig, ax = plt.subplots()
    df_plot = df_imerg.pivot(
        index="plot_date", columns="year", values=f"roll{window}"
    )
    df_plot.plot(ax=ax, alpha=0.5, linewidth=0.5)
    ax.get_legend().remove()
```

```python

```
