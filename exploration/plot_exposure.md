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

# Plot exposure

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

from src.utils import plotting
from src.datasources import codab
from src.constants import *
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/cub_melissa_adm2_wind_rain_exposure.parquet"
)
df_exp = stratus.load_parquet_from_blob(blob_name)
```

```python
adm0 = codab.load_codab_from_blob(admin_level=0)
```

```python
adm1 = codab.load_codab_from_blob(admin_level=1)
```

```python
adm2 = codab.load_codab_from_blob(admin_level=2)
```

```python
adm2_aoi = adm2[adm2["ADM1_PCODE"].isin(df_exp["ADM1_PCODE"].unique())]
```

```python
gdf_adm2 = adm2_aoi.merge(df_exp, how="outer")
```

```python
len(gdf_adm2)
```

```python
df_exp
```

```python
df_exp_wind = df_exp.melt(
    id_vars=["ADM2_PCODE"],
    value_vars=[f"exp_{x}_knots" for x in [34, 50, 64]],
    var_name="buffer_speed",
    value_name="pop_exposed",
)
df_exp_wind["buffer_speed"] = (
    df_exp_wind["buffer_speed"]
    .apply(lambda x: x.removeprefix("exp_").removesuffix("_knots"))
    .astype(int)
)
df_exp_wind
```

```python
gdf_template = plotting.build_circle_template(
    gdf_adm2, id_col="ADM2_PCODE", pop_col="total_pop", area_per_person=20_000
)
plotting.plot_template_circles(gdf_template)
```

```python
HOLGUIN2 = "CU0709"
```

```python
df_exp[df_exp["ADM2_PCODE"] == HOLGUIN2]
```

```python
df_exp_wind[df_exp_wind["ADM2_PCODE"] == HOLGUIN2]
```

```python
import re


def wrap_text(text, max_len=10):
    """
    Insert line breaks at spaces or dashes so each line
    is roughly limited to `max_len` characters.
    Does not add or remove spaces/dashes.
    """
    tokens = re.findall(r"\S+-|\S+|[-]", text)  # split into words and dashes
    lines, current = [], ""

    for token in tokens:
        if len(current) + len(token) + 1 > max_len:
            lines.append(current.rstrip())
            current = token
        else:
            if current:
                current += " " if not current.endswith("-") else ""
            current += token

    if current:
        lines.append(current.rstrip())

    return "\n".join(lines).removeprefix("\n")
```

```python
gdf_plot = gdf_template.merge(adm2_aoi[["ADM2_PCODE", "ADM2_ES"]])
gdf_plot["adm_label"] = gdf_plot["ADM2_ES"].apply(wrap_text)
```

```python
fig, ax = plotting.plot_bullseye_exposures(
    gdf_plot,
    df_exp_wind,
    id_col="ADM2_PCODE",
    label_col="adm_label",
    legend_title="Population exposed\nto wind speed (circle size\npropotional to population)",
)
ax.set_title(
    "Cuba: population exposed to Melissa wind speed per municipality\n"
)
```

```python
df_exp_rain = df_exp.melt(
    id_vars=["ADM2_PCODE"],
    value_vars=[f"exp_{x}_mm" for x in [100, 200, 300, 400, 500]],
    var_name="rain_mm",
    value_name="pop_exposed",
)
df_exp_rain["rain_mm"] = (
    df_exp_rain["rain_mm"]
    .apply(lambda x: x.removeprefix("exp_").removesuffix("_mm"))
    .astype(int)
)
df_exp_rain
```

```python
levels = [
    # 25,
    # 50,
    100,
    # 150,
    200,
    300,
    400,
    500,
    # 750,
]
colors = [
    # "lawngreen",
    # "limegreen",
    "yellow",
    # "gold",
    "darkorange",
    "red",
    "firebrick",
    "magenta",
    # "darkmagenta",
]
```

```python
colors_dict = {l: c for l, c in zip(levels, colors)}
```

```python
fig, ax = plotting.plot_bullseye_exposures(
    gdf_template.merge(adm2_aoi[["ADM2_PCODE", "ADM2_ES"]]),
    df_exp_rain,
    speed_col="rain_mm",
    speeds_order=levels,
    id_col="ADM2_PCODE",
    label_col="ADM2_ES",
    colors=colors_dict,
    legend_title="Population exposed\nto rainfall (circle size\npropotional to population)",
    legend_label_fmt="{spd} mm",
)
ax.set_title(
    "Cuba: population exposed to Melissa rainfall per municipality\n"
    "Total rainfall over 2025-10-28 to 2025-10-29"
)
```

```python
blob_name = "ds-flood-gfm/processed/CUB_adm2_pop_exposure.csv"
df_exp_flood = stratus.load_csv_from_blob(blob_name)
```

```python
df_exp_flood = df_exp_flood.rename(columns={"adm2_src": "ADM2_PCODE"})
```

```python
df_exp_flood
```

```python
df_exp_flood = df_exp_flood.merge(
    adm2_aoi[["ADM2_PCODE", "ADM2_ES", "ADM1_PCODE", "ADM1_ES"]]
)
```

```python
df_plot = df_exp_flood.copy()

df_plot["adm_label"] = df_plot["ADM2_ES"] + " (" + df_plot["ADM1_ES"] + ")"

cols = ["jrc_pop_exposed", "chd_gfm_pop_exposed"]

df_plot["max_cols"] = df_plot[cols].max(axis=1)

fig, ax = plt.subplots(figsize=(10, 7), dpi=200)

cutoff = 10

df_plot[df_plot["max_cols"] >= cutoff].sort_values("max_cols").plot.barh(
    x="adm_label",  # label (categorical axis)
    y=cols,  # numeric columns
    ax=ax,
    color=["chocolate", CHD_GREEN],
)

ax.legend(["JRC", "CHD"], title="Method")
ax.set_xlabel("Population exposed")
ax.set_ylabel("Municipality")

ax.set_title("Cuba: population exposed to flooding from hurricane Melissa\n")
ax.text(
    0.5,
    1.01,
    f"Only municipalities with exposure ≥ {cutoff} people shown",
    transform=ax.transAxes,
    ha="center",
    style="italic",
    color="grey",
)

ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
ax.yaxis.grid(False)

ax.xaxis.set_major_formatter(EngFormatter(unit=""))
[ax.spines[x].set_visible(False) for x in ["top", "right"]]
```

```python
df_out = (
    adm2_aoi[
        [
            "ADM1_PCODE",
            "ADM1_ES",
            "ADM2_PCODE",
            "ADM2_ES",
        ]
    ]
    .merge(df_exp_flood)
    .sort_values(["ADM1_ES", "ADM2_ES"])
)
```

```python
save_path = "temp/cub_melissa_adm2_flood_exposure.csv"
df_out.to_csv(save_path, index=False, encoding="utf-8-sig")

save_path = "temp/cub_melissa_adm2_flood_exposure.xlsx"
df_out.to_excel(save_path, index=False)
```
