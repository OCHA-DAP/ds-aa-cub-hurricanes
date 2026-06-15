"""Methodology illustration map for the Cuba hurricane AA presentation.

Overlays IBTrACS wind-radii buffers (34 / 50 / 64 kt) for a selected storm on top
of the WorldPop population-count raster, clipped to Cuba. Produces a single PNG that
shows, at a glance, how the trigger methodology combines a storm's wind footprint with
where people actually live.

This is a STANDALONE script. It does not modify any existing project code. It only
reads from the same sources the pipeline already uses:
  - storms.ibtracs_wind_buffers   (PostGIS, via ocha_stratus.get_engine)
  - WorldPop population COG        (via ocha_stratus.open_blob_cog)
  - CODAB admin boundaries         (via src.datasources.codab)

Usage:
    python scripts/methodology_map.py                 # default storm (Ian 2022)
    python scripts/methodology_map.py ike             # by friendly name
    python scripts/methodology_map.py 2008245N17323   # by raw IBTrACS SID
    python scripts/methodology_map.py oscar --coarsen 3 --out docs/oscar.png
    python scripts/methodology_map.py melissa --upload  # also push PNG to blob

Notes:
  - The population raster is windowed to Cuba's bounding box (padded) before reading,
    so we never pull the full global grid.
  - --coarsen N downsamples the raster by NxN, summing population (each coarse cell =
    total people in it). Higher N = coarser/lower-resolution map.
  - --upload pushes the PNG to blob (projects/<BLOB_DIR>) so the Quarto book can load
    it at render time; the book's static images are gitignored so blob is the source.
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import ocha_stratus as stratus
from matplotlib.colors import AsinhNorm, Normalize
from matplotlib.lines import Line2D
from sqlalchemy import text

# Put the repo root on the path so `src` imports resolve when this script is run
# directly (python scripts/methodology_map.py). The pipelines/ scripts instead rely
# on PYTHONPATH=<repo root>, which is set for them in the GitHub workflows.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.constants import GUSTAV, IAN, IKE, IRMA, OSCAR, PROJECT_PREFIX  # noqa: E402
from src.datasources import codab  # noqa: E402
from src.utils.blob_utils import _upload_blob_data  # noqa: E402

# Friendly storm names -> IBTrACS SIDs. Most come from src/constants.py; Melissa 2025
# is not in constants yet, so it's defined here (resolved from the storms table).
MELISSA = "2025291N11319"
STORMS = {
    "ike": IKE,
    "gustav": GUSTAV,
    "irma": IRMA,
    "ian": IAN,
    "oscar": OSCAR,
    "melissa": MELISSA,
}

POP_BLOB = "worldpop/pop_count/global_pop_2026_CN_1km_R2025A_UA_v1.tif"
# Where --upload puts the PNG so the Quarto book can load it (container=projects).
BLOB_DIR = f"{PROJECT_PREFIX}/book_static"
WIND_SPEEDS = [34, 50, 64]  # knots; 34 is the widest buffer, 64 the tightest core

# Match the marimo storm-selector styling: YlOrRd ramp normalised 20-80 kt.
_CMAP = plt.get_cmap("YlOrRd")
_NORM = Normalize(vmin=20, vmax=80)


def resolve_sid(storm: str) -> str:
    """Accept either a friendly name (ike, ian, ...) or a raw IBTrACS SID."""
    key = storm.lower().strip()
    if key in STORMS:
        return STORMS[key]
    # Looks like a raw SID (e.g. 2008245N17323) -- pass it straight through.
    return storm


def load_buffers(sid: str) -> gpd.GeoDataFrame:
    """Wind-radii buffer polygons for one storm, from the PostGIS table."""
    engine = stratus.get_engine("dev")
    gdf = gpd.read_postgis(
        text(
            "SELECT sid, wind_speed_kt, geometry"
            " FROM storms.ibtracs_wind_buffers"
            " WHERE sid = :sid"
        ),
        engine,
        geom_col="geometry",
        params={"sid": sid},
    )
    if gdf.empty:
        raise SystemExit(
            f"No wind buffers found for sid={sid!r}. "
            f"Known names: {', '.join(STORMS)}."
        )
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf.to_crs(4326)


def storm_label(sid: str) -> str:
    """A human title like 'Ian (2022)' if the metadata table has it."""
    try:
        engine = stratus.get_engine("dev")
        with engine.connect() as con:
            row = con.execute(
                text(
                    "SELECT name, season FROM storms.ibtracs_storms"
                    " WHERE sid = :sid"
                ),
                {"sid": sid},
            ).fetchone()
        if row and row[0]:
            name = str(row[0]).title()
            return f"{name} ({row[1]})" if row[1] else name
    except Exception:
        pass
    return sid


def load_population(bounds, coarsen: int, adm0: gpd.GeoDataFrame):
    """Window the global WorldPop COG to `bounds`, coarsen, and clip to Cuba.

    Returns a 2-D DataArray of people per coarse cell that is a CONTINUOUS surface
    over land (empty land = 0, not NaN) with only the ocean masked out. Keeping land
    hole-free is what lets bilinear interpolation render a smooth density surface
    instead of scattered isolated squares.
    """
    minx, miny, maxx, maxy = bounds
    da = stratus.open_blob_cog(POP_BLOB, container_name="raster", stage="dev")
    da = da.rio.clip_box(minx, miny, maxx, maxy)  # windowed read only
    da = da.squeeze("band", drop=True)
    # Drop WorldPop's large-negative nodata BEFORE coarsening, or it would poison the
    # per-cell sums. coarsen().sum() then skips NaN and sums only the valid pixels.
    da = da.where(da >= 0)
    if coarsen > 1:
        da = da.coarsen(x=coarsen, y=coarsen, boundary="trim").sum()
    # Fill the remaining NaN (empty land + ocean) with 0 so land has no holes, then
    # clip to the coastline so only the ocean is masked back out (transparent).
    da = da.fillna(0)
    da = da.rio.clip(adm0.geometry, adm0.crs, all_touched=True, drop=False)
    return da


def make_map(sid: str, coarsen: int, pad: float, out: Path) -> Path:
    adm0 = codab.load_codab_from_blob(admin_level=0)
    adm1 = codab.load_codab_from_blob(admin_level=1)
    buffers = load_buffers(sid)

    # Plot window: the full bounding box of Cuba (padded), regardless of where the
    # storm's buffers fall. Buffers that extend past the coast simply run off-frame.
    minx, miny, maxx, maxy = adm0.total_bounds
    extent = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    pop = load_population(extent, coarsen, adm0)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=200)

    # Base layer: population on an asinh ("pseudo-log") scale -- linear through the
    # sparse low-population land and log-like across the dense cities, so the whole
    # dynamic range reads smoothly without a hard log floor. A magma ramp on a black
    # face gives a "population density" glow: empty land dark, cities bright.
    pop_cmap = plt.get_cmap("magma").copy()
    ax.set_facecolor(pop_cmap(0.0))
    # Cap at a high percentile, not the single max, so mid-size cities read bright
    # instead of being washed out by the one most-populous cell (Havana). The linear
    # width sets where asinh hands off from linear to log (roughly "village" scale).
    populated = pop.where(pop > 0)
    vmax = float(populated.quantile(0.99))
    linear_width = max(vmax / 50, 10.0)
    pop.plot.imshow(
        ax=ax,
        cmap=pop_cmap,
        norm=AsinhNorm(linear_width=linear_width, vmin=0, vmax=vmax),
        interpolation="gaussian",
        add_colorbar=True,
        cbar_kwargs={
            "label": f"Population (people per {coarsen} km cell)",
            "shrink": 0.7,
            "pad": 0.02,
        },
        zorder=1,
    )

    # Context: admin boundaries (light, so they read over the dark raster).
    adm1.boundary.plot(ax=ax, color="white", linewidth=0.3, alpha=0.35, zorder=2)
    adm0.boundary.plot(ax=ax, color="white", linewidth=1.0, alpha=0.8, zorder=3)

    # Overlay: wind buffers as bold rings with a very faint fill. The buffers are
    # nested disks (34 kt contains 50 contains 64), so the low-alpha fills stack
    # gently toward the core -- a subtle "more intense at the centre" wash -- while
    # staying light enough that the population underneath stays clearly visible.
    for wt in WIND_SPEEDS:
        buf = buffers[
            (buffers["wind_speed_kt"] == wt) & buffers.geometry.notna()
        ]
        if buf.empty:
            continue
        color = _CMAP(_NORM(wt))
        buf.plot(ax=ax, facecolor=color, edgecolor="none", alpha=0.14, zorder=4)
        buf.boundary.plot(ax=ax, color=color, linewidth=2.2, alpha=0.95, zorder=5)

    legend_handles = [
        Line2D([0], [0], color=_CMAP(_NORM(wt)), linewidth=2.2,
               label=f"{wt} kt wind")
        for wt in WIND_SPEEDS
    ]
    ax.legend(handles=legend_handles, loc="lower left", framealpha=0.9,
              title="Wind footprint")

    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        f"Wind footprint over population — {storm_label(sid)}",
        fontsize=13,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "storm",
        nargs="?",
        default="ian",
        help="Storm name (ike, gustav, irma, ian, oscar) or raw IBTrACS SID.",
    )
    p.add_argument(
        "--coarsen", type=int, default=2,
        help="Downsample factor for the raster (NxN, summed). Default 2.",
    )
    p.add_argument(
        "--pad", type=float, default=0.5,
        help="Degrees of padding around Cuba's bounding box. Default 0.5.",
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG path. Default docs/methodology_map_<storm>.png",
    )
    p.add_argument(
        "--upload", action="store_true",
        help=f"Also upload the PNG to blob (projects/{BLOB_DIR}) for the book.",
    )
    args = p.parse_args()

    sid = resolve_sid(args.storm)
    out = args.out or Path("docs") / f"methodology_map_{args.storm.lower()}.png"
    saved = make_map(sid, args.coarsen, args.pad, out)
    print(f"Saved {saved}")

    if args.upload:
        blob_name = f"{BLOB_DIR}/{saved.name}"
        _upload_blob_data(
            saved.read_bytes(),
            blob_name,
            stage="dev",
            container_name="projects",
            content_type="image/png",
        )
        print(f"Uploaded to blob (container=projects, stage=dev): {blob_name}")


if __name__ == "__main__":
    main()
