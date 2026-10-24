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

# Wind history

```python
%load_ext jupyter_black
%load_ext autoreload
%autoreload 2
```

```python
import ocha_stratus as stratus
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from matplotlib.patches import Patch
from rioxarray.exceptions import NoDataInBounds
from tqdm.auto import tqdm
from matplotlib.ticker import EngFormatter
from dask.diagnostics import ProgressBar

from src.datasources import codab, chirps_gefs
from src.constants import *
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
adm2.plot()
```

```python
blob_name = "ghsl/pop/GHS_POP_E2025_GLOBE_R2023A_4326_3ss_V1_0.tif"
da_global = stratus.open_blob_cog(blob_name, container_name="raster").squeeze(
    drop=True
)
```

```python
# clip to box (need to do this first, otherwise Python crashes on normal .rio.clip)
minx, miny, maxx, maxy = adm0.total_bounds
da_clip_box = da_global.rio.clip_box(
    minx=minx, miny=miny, maxx=maxx, maxy=maxy
)
```

```python
da_ghsl = da_clip_box.rio.clip(adm0.geometry)
```

```python
da_ghsl.attrs["_FillValue"] = np.nan
```

```python
da_ghsl = da_ghsl.where(da_ghsl >= 0)
```

```python
da_ghsl = da_ghsl.compute()
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/raw/noaa/nhc/wind_history/al132025_best_track.zip"
)
gdf_wind = stratus.load_shp_from_blob(
    blob_name, shapefile="AL132025_windswath.shp"
)
```

```python
adm2_exp = adm2[adm2.intersects(gdf_wind.dissolve().iloc[0].geometry)]
```

```python
adm1_pcodes = adm2_exp["ADM1_PCODE"].unique()
```

```python
adm2_aoi = adm2[adm2["ADM1_PCODE"].isin(adm1_pcodes)]
```

```python
adm1_aoi = adm1[adm1["ADM1_PCODE"].isin(adm1_pcodes)]
```

```python
da_ghsl_aoi = da_ghsl.rio.clip(adm1_aoi.geometry)
```

```python
gdf_wind["Wind Speed (knots)"] = (
    gdf_wind["RADII"].astype(int).astype("category")
)
```

```python
minx, miny, maxx, maxy = adm1_aoi.total_bounds
```

```python
issued_date = datetime(2025, 10, 29).date()

das = []
for lt in tqdm(range(16)):
    valid_date = issued_date + pd.Timedelta(days=lt)
    da_in = chirps_gefs.open_chirps_gefs(issued_date, valid_date)
    da_in["valid_date"] = valid_date
    das.append(da_in)

da_gefs = xr.concat(das, dim="valid_date").squeeze(drop=True)
```

```python
total_bounds = adm0.total_bounds
```

```python
da_gefs_clip_box = da_gefs.rio.clip_box(*total_bounds)
```

```python
with ProgressBar():
    da_gefs_clip_box_computed = da_gefs_clip_box.compute()
```

```python
da_gefs_clip_box_computed.rio.clip(adm0.geometry).mean(dim=["x", "y"])
```

```python
gefs_dates = pd.date_range("2025-10-29", "2025-10-29")
```

```python
# get IMERG
query = """
SELECT *
FROM public.imerg
WHERE pcode = 'CU'
"""
with stratus.get_engine(stage="prod").connect() as con:
    df_imerg = pd.read_sql(query, con)
```

```python
df_imerg["valid_date"] = pd.to_datetime(df_imerg["valid_date"])
df_imerg = df_imerg.sort_values("valid_date")
```

```python
df_imerg.iloc[-20:]
```

```python
df_imerg.iloc[-20:].plot(x="valid_date", y="mean")
```

```python
imerg_dates = df_imerg.iloc[-2:]["valid_date"].dt.date
```

```python
imerg_dates
```

```python
IMERG_BLOB_NAME = (
    "imerg/daily/late/v7/processed/imerg-daily-late-{date_str}.tif"
)
```

```python
das = []
for d in imerg_dates:
    blob_name = IMERG_BLOB_NAME.format(date_str=d)
    da_in = stratus.open_blob_cog(
        blob_name, stage="prod", container_name="raster"
    )
    da_in["date"] = d
    das.append(da_in)
```

```python
da_imerg = xr.concat(das, dim="date").squeeze(drop=True)
```

```python
da_imerg_clip_box = da_imerg.rio.clip_box(*total_bounds)
```

```python
with ProgressBar():
    da_imerg_clip_box_computed = da_imerg_clip_box.compute()
```

```python
da_imerg_clip = da_imerg_clip_box_computed.rio.clip(
    adm0.geometry, all_touched=True
)
```

```python
da_imerg_clip_box_computed.isel(date=0).plot()
```

```python
da_imerg_clip.isel(date=0).plot()
```

```python
da_imerg_clip_box_computed.rio.clip(adm0.geometry, all_touched=False).isel(
    date=0
).plot()
```

```python
# for CHIRPS-GEFS:
# da_rainfall = (
#     da_gefs_clip_box_computed.where(da_gefs_clip_box_computed > 0)
#     .rio.clip(adm0.geometry)
#     .sel(valid_date=gefs_dates)
# )

# for IMERG:
da_rainfall = da_imerg_clip.where(da_imerg_clip > 0).sum(dim="date")
```

```python
fig, ax = plt.subplots(dpi=200, figsize=(12, 4))

minx, miny, maxx, maxy = adm2_exp.total_bounds

adm1.boundary.plot(ax=ax, linewidth=0.3, color="k")
adm2.boundary.plot(ax=ax, linewidth=0.1, color="k")

for _, row in adm2_exp.iterrows():
    c = row.geometry.centroid
    ax.annotate(
        row["ADM2_ES"],
        (c.x, c.y),
        ha="center",
        va="center",
        fontsize=4,
    )

da_rainfall.where(da_rainfall >= 0).plot(
    ax=ax,
    cmap="Greys",
    cbar_kwargs={"label": "Precipitation (mm)"},
    vmin=0,
)

alpha = 0.3
color_map = {34: "gold", 50: "crimson", 64: "indigo"}
for value, color in color_map.items():
    gdf_wind[gdf_wind["Wind Speed (knots)"] == value].plot(
        ax=ax, color=color, alpha=alpha, label=f"{value} kt"
    )

handles = [
    Patch(facecolor=color, label=f"{value} kt", alpha=alpha)
    for value, color in color_map.items()
]

ax.legend(
    handles=handles, title="Wind Speed (knots)", loc="lower left", frameon=True
)

ax.set_title(
    "Hurricane Melissa wind history\n"
    f"and rainfall for {imerg_dates.min()} to {imerg_dates.max()}"
)

# da_ghsl_aoi.plot(ax=ax, vmax=5, cmap="Greys")
margin = 0.1
ax.set_xlim(minx - margin, maxx + margin)
ax.set_ylim(miny - margin, maxy + margin)
ax.axis("off")
```

```python
levels = [25, 50, 100, 150, 200, 300, 400, 500, 750]
colors = [
    "lawngreen",
    "limegreen",
    "yellow",
    "gold",
    "darkorange",
    "red",
    "firebrick",
    "magenta",
    "darkmagenta",
]
cbar_kwargs = {
    "label": "Precipitation (mm)",  # Set label for the colorbar
    "shrink": 0.8,  # Shrink the colorbar to 80% of its default size
}
```

```python
gdf_wind
```

```python
fig, ax = plt.subplots(dpi=200, figsize=(12, 4))

minx, miny, maxx, maxy = adm2_exp.total_bounds

adm1.boundary.plot(ax=ax, linewidth=0.3, color="k")
adm2.boundary.plot(ax=ax, linewidth=0.1, color="k")

for _, row in adm2_exp.iterrows():
    c = row.geometry.centroid
    ax.annotate(
        row["ADM2_ES"],
        (c.x, c.y),
        ha="center",
        va="center",
        fontsize=4,
    )

da_rainfall.plot(
    ax=ax, levels=levels, colors=colors, extend="max", cbar_kwargs=cbar_kwargs
)

alpha = 0.4
for wind, color in color_map.items():
    gdf_wind[gdf_wind["RADII"] == wind].boundary.plot(
        ax=ax, color=color, alpha=alpha
    )

# Create legend using the same color map
legend_elements = [
    Patch(facecolor="white", edgecolor=color, label=f"{wind}", alpha=alpha)
    for wind, color in color_map.items()
]

ax.legend(
    handles=legend_elements,
    title="Wind Speed (knots)",
    loc="lower left",
    frameon=True,
)

ax.legend(
    handles=legend_elements, title="Wind Speed (knots)", loc="lower left"
)

ax.set_title(
    "Hurricane Melissa wind history\n"
    f"and rainfall for {imerg_dates.min()} to {imerg_dates.max()}"
)

# da_ghsl_aoi.plot(ax=ax, vmax=5, cmap="Greys")
margin = 0.1
ax.set_xlim(minx - margin, maxx + margin)
ax.set_ylim(miny - margin, maxy + margin)
ax.axis("off")
```

```python
dicts = []
for pcode, group in tqdm(adm2.groupby("ADM2_PCODE")):
    ghsl_adm2 = da_ghsl.rio.clip(group.geometry)
    dicts.append({"ADM2_PCODE": pcode, "total_pop": int(ghsl_adm2.sum())})
```

```python
df_adm2_pop = pd.DataFrame(dicts)
```

```python
df_adm2_pop_aoi = df_adm2_pop[
    df_adm2_pop["ADM2_PCODE"].isin(adm2_aoi["ADM2_PCODE"].unique())
]
```

```python
HOLGUIN2 = "CU0709"
```

```python
dicts = []
for pcode, group in tqdm(adm2_exp.groupby("ADM2_PCODE")):
    ghsl_adm2 = da_ghsl.rio.clip(group.geometry)
    for speed, row in gdf_wind.set_index("RADII").iterrows():
        try:
            ghsl_adm2_speed = ghsl_adm2.rio.clip([row.geometry])
            pop_exp = int(ghsl_adm2_speed.sum())
        except NoDataInBounds:
            pop_exp = 0
        dicts.append(
            {"ADM2_PCODE": pcode, "speed": int(speed), "pop_exp": pop_exp}
        )
```

```python
df_adm2_exp_raw = pd.DataFrame(dicts)
```

```python
df_adm2_exp = df_adm2_exp_raw.pivot(
    index="ADM2_PCODE", columns="speed", values="pop_exp"
)

df_adm2_exp = df_adm2_exp.rename(
    columns={x: f"exp_{x}_knots" for x in df_adm2_exp.columns}
)
```

```python
df_adm2_exp = df_adm2_exp.reset_index().merge(df_adm2_pop_aoi, how="right")
```

```python
df_adm2_exp["exp_34_knots"] = df_adm2_exp[
    ["exp_34_knots", "exp_50_knots", "exp_64_knots"]
].sum(axis=1)
df_adm2_exp["exp_50_knots"] = df_adm2_exp[
    ["exp_50_knots", "exp_64_knots"]
].sum(axis=1)
```

```python
df_adm2_exp = df_adm2_exp.fillna(0)
df_adm2_exp = df_adm2_exp.set_index("ADM2_PCODE")
df_adm2_exp = df_adm2_exp.astype(int).reset_index()
```

```python
df_adm2_exp
```

```python
df_out = adm2[["ADM1_PCODE", "ADM1_ES", "ADM2_PCODE", "ADM2_ES"]].merge(
    df_adm2_exp
)

save_path = "temp/cub_melissa_adm2_wind_exposure.csv"
df_out.to_csv(save_path, index=False, encoding="utf-8-sig")

save_path = "temp/cub_melissa_adm2_wind_exposure.xlsx"
df_out.to_excel(save_path, index=False)
```

```python
da_rainfall_interp = da_rainfall.interp_like(
    da_ghsl_aoi, method="nearest", kwargs={"fill_value": "extrapolate"}
).squeeze(drop=True)
```

```python
rain_threshs = [100, 200, 300, 400, 500]
```

```python
dicts = []
for pcode, group in tqdm(adm2_exp.groupby("ADM2_PCODE")):
    ghsl_adm2 = da_ghsl.rio.clip(group.geometry)
    for rain_thresh in rain_threshs:
        da_ghsl_rain_thresh = ghsl_adm2.where(
            da_rainfall_interp >= rain_thresh
        )
        dicts.append(
            {
                "ADM2_PCODE": pcode,
                "rain_thresh": rain_thresh,
                "pop_exp": int(da_ghsl_rain_thresh.sum()),
            }
        )
```

```python
df_adm2_exp_rain_raw = pd.DataFrame(dicts)
```

```python
df_adm2_exp_rain = df_adm2_exp_rain_raw.pivot(
    index="ADM2_PCODE", columns="rain_thresh", values="pop_exp"
)

df_adm2_exp_rain = df_adm2_exp_rain.rename(
    columns={x: f"exp_{x}_mm" for x in df_adm2_exp_rain.columns}
)

df_adm2_exp_rain = df_adm2_exp_rain.reset_index().merge(
    df_adm2_pop_aoi, how="right"
)
```

```python
df_adm2_exp_rain = df_adm2_exp_rain.fillna(0)
df_adm2_exp_rain = df_adm2_exp_rain.set_index("ADM2_PCODE")
df_adm2_exp_rain = df_adm2_exp_rain.astype(int).reset_index()
```

```python
df_exp = df_adm2_exp.merge(df_adm2_exp_rain)
```

```python
df_exp
```

```python
df_out = adm2[["ADM1_PCODE", "ADM1_ES", "ADM2_PCODE", "ADM2_ES"]].merge(df_exp)

df_out = df_out[
    [x for x in df_out.columns if x != "total_pop"] + ["total_pop"]
]

save_path = "temp/cub_melissa_adm2_wind_rain_exposure.csv"
df_out.to_csv(save_path, index=False, encoding="utf-8-sig")

save_path = "temp/cub_melissa_adm2_wind_rain_exposure.xlsx"
df_out.to_excel(save_path, index=False)
```

```python
blob_name = (
    f"{PROJECT_PREFIX}/processed/cub_melissa_adm2_wind_rain_exposure.parquet"
)
stratus.upload_parquet_to_blob(df_out, blob_name)
```

```python
df_exp.mean(numeric_only=True).plot.bar()
```

```python
df_plot = df_exp.merge(adm2)
```

```python
def plot_exp(df_plot, xcol, ycol):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=200)

    xmax = df_plot[xcol].max() * 1.1
    ymax = df_plot[ycol].max() * 1.1
    xymax = max(xmax, ymax)

    for _, row in df_plot.iterrows():
        ax.annotate(
            row["ADM2_ES"],
            row[[xcol, ycol]],
            va="center",
            ha="center",
            fontsize=6,
            rotation=-45,
        )

    ax.set_xlim((0, xymax))
    ax.set_ylim((0, xymax))

    ax.xaxis.set_major_formatter(EngFormatter(unit=""))
    ax.yaxis.set_major_formatter(EngFormatter(unit=""))

    [ax.spines[x].set_visible(False) for x in ["top", "right"]]
    return fig, ax
```

```python
speed = 50
rain = 100

xcol, ycol = f"exp_{speed}_knots", f"exp_{rain}_mm"

fig, ax = plot_exp(df_plot, xcol, ycol)

ax.set_xlabel(f"Population exposed to ≥ {speed} knot wind")
ax.set_ylabel(f"Population exposed to ≥ {rain} mm rainfall")
```

```python
df_plot["max_100_mm_50_knots"] = df_plot[["exp_100_mm", "exp_50_knots"]].max(
    axis=1
)
df_plot["min_100_mm_50_knots"] = df_plot[["exp_100_mm", "exp_50_knots"]].min(
    axis=1
)

df_plot["max_200_mm_64_knots"] = df_plot[["exp_200_mm", "exp_64_knots"]].max(
    axis=1
)
df_plot["min_200_mm_64_knots"] = df_plot[["exp_200_mm", "exp_64_knots"]].min(
    axis=1
)
```

```python
fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
df_plot[df_plot["min_100_mm_50_knots"] > 0].sort_values(
    "max_100_mm_50_knots"
).plot.bar(
    x="ADM2_ES",
    y=["exp_100_mm", "exp_50_knots"],
    ax=ax,
    color=["dodgerblue", "darkorange"],
)

ax.legend(
    ["≥ 100 mm rain", "≥ 50 knot wind"],
    title="Population exposed",
)

ax.set_ylabel("Population exposed")
ax.set_xlabel("Municipality")
ax.yaxis.set_major_formatter(EngFormatter(unit=""))

[ax.spines[x].set_visible(False) for x in ["top", "right"]]
```

```python
df_exp[df_exp["ADM2_PCODE"] == "CU0502"]
```

```python
fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
df_plot[df_plot["min_200_mm_64_knots"] > 0].sort_values(
    "max_200_mm_64_knots"
).plot.bar(
    x="ADM2_ES",
    y=["exp_200_mm", "exp_64_knots"],
    ax=ax,
    color=["dodgerblue", "darkorange"],
)

ax.legend(
    ["≥ 200 mm rain", "≥ 64 knot wind"],
    title="Population exposed",
)

ax.set_ylabel("Population exposed")
ax.set_xlabel("Municipality")

ax.yaxis.set_major_formatter(EngFormatter(unit=""))

[ax.spines[x].set_visible(False) for x in ["top", "right"]]
```

```python
speed = 64
rain = 200

xcol, ycol = f"exp_{speed}_knots", f"exp_{rain}_mm"

fig, ax = plot_exp(df_plot, xcol, ycol)

ax.set_xlabel(f"Population exposed to ≥ {speed} knot wind")
ax.set_ylabel(f"Population exposed to ≥ {rain} mm rainfall")
```

```python
for x in df_plot.columns:
    if "exp" in x and "frac" not in x:
        df_plot[f"frac_{x}"] = df_plot[x] / df_plot["total_pop"]
```

```python
speed = 64
rain = 200

xcol, ycol = f"frac_exp_{speed}_knots", f"frac_exp_{rain}_mm"

fig, ax = plot_exp(df_plot, xcol, ycol)

ax.set_xlabel(f"Fraction of population exposed to ≥ {speed} knot wind")
ax.set_ylabel(f"Fraction of population exposed to ≥ {rain} mm rainfall")
```

```python
da_rainfall_interp
```

```python
da_rainfall_interp.plot()
```

```python

```
