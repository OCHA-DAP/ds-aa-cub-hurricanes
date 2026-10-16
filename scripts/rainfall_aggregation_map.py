"""Rainfall aggregation illustration for the Cuba hurricane AA book.

Shows how one storm's two-day IMERG rainfall over Cuba turns into the single
numbers the trigger compares against thresholds: the national mean, or the 50th,
80th, 90th or 95th percentile across pixels. One map of the two-day rainfall, one
map per percentile marking the pixels at or above it, and a bar chart of the five
values side by side.

The aggregation follows the historical analysis (exploration/imerg_aggregation.md):
daily IMERG clipped to Cuba, two-day rolling sums over the days the storm was in the
ZMA plus one day either side, each statistic taken across pixels. The figure shows
the two-day window where the 80th percentile peaks, and the script checks that peak
against the stored q80_roll2 for the storm; a mismatch is an error.

This is a STANDALONE script. It only reads, from the sources the analysis already uses:
  - IMERG daily rasters              (via src.datasources.imerg)
  - ZMA storm tracks, storm stats    (blob parquet, via ocha_stratus)
  - CODAB admin boundaries           (via src.datasources.codab)

Usage:
    python scripts/rainfall_aggregation_map.py                 # default storm (Oscar 2024)
    python scripts/rainfall_aggregation_map.py ian
    python scripts/rainfall_aggregation_map.py oscar --out book_cub_hurricanes/figures/x.png
    python scripts/rainfall_aggregation_map.py oscar --upload  # also push PNG to blob
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ocha_stratus as stratus
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.constants import GUSTAV, IAN, IKE, IRMA, OSCAR, PROJECT_PREFIX  # noqa: E402
from src.datasources import codab, imerg  # noqa: E402
from src.utils.blob_utils import _upload_blob_data  # noqa: E402

STORMS = {"ike": IKE, "gustav": GUSTAV, "irma": IRMA, "ian": IAN, "oscar": OSCAR}
TRACKS_BLOB = f"{PROJECT_PREFIX}/processed/ibtracs/zma_tracks_2000-2024.parquet"
STATS_BLOB = f"{PROJECT_PREFIX}/processed/storm_stats/zma_stats_imerg_quantiles.parquet"
BLOB_DIR = f"{PROJECT_PREFIX}/book_static"
QUANTILES = [0.5, 0.8, 0.9, 0.95]

# Sequential blue ramp (light -> dark) and chart chrome from the team dataviz palette.
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ABOVE, BELOW = "#256abf", "#e1e0d9"
# Fixed rainfall bins: a few extreme pixels would otherwise stretch a linear scale.
RAIN_BINS = [0, 10, 25, 50, 100, 150, 200]
INK, MUTED, AXIS = "#0b0b0b", "#52514e", "#c3c2b7"


def resolve_sid(storm: str) -> str:
    key = storm.lower().strip()
    return STORMS.get(key, storm)


def storm_window(sid: str) -> pd.DatetimeIndex:
    """Days the storm was in the ZMA, plus one day either side (as in the analysis)."""
    tracks = stratus.load_parquet_from_blob(TRACKS_BLOB)
    tracks = tracks[tracks["sid"] == sid]
    if tracks.empty:
        raise ValueError(f"No ZMA track points for {sid} in {TRACKS_BLOB}")
    start = tracks["valid_time"].min().date() - pd.DateOffset(days=1)
    end = tracks["valid_time"].max().date() + pd.DateOffset(days=1)
    return pd.date_range(start, end)


def load_rainfall(dates, adm0):
    da = imerg.open_imerg_raster_dates(dates)
    # open_imerg_raster_dates prints and skips dates it can't load; a gap here would
    # silently change the two-day sums, so refuse instead.
    if da.sizes["date"] != len(dates):
        loaded = set(pd.to_datetime(da["date"].values))
        missing = [d.date() for d in dates if d not in loaded]
        raise RuntimeError(f"IMERG missing for {missing}; refusing to draw a partial window")
    return da.rio.clip(adm0.geometry).rolling(date=2).sum()


def check_against_stored(sid, q80_peak):
    stats = stratus.load_parquet_from_blob(STATS_BLOB)
    row = stats[stats["sid"] == sid]
    if len(row) != 1:
        raise ValueError(f"Expected one row for {sid} in {STATS_BLOB}, got {len(row)}")
    stored = float(row["q80_roll2"].iloc[0])
    if not np.isclose(q80_peak, stored, atol=0.01):
        raise RuntimeError(f"q80 peak {q80_peak:.2f} mm != stored q80_roll2 {stored:.2f} mm")
    print(f"q80 peak matches stored q80_roll2: {stored:.2f} mm")


def _style(ax, adm0, title):
    adm0.boundary.plot(ax=ax, color=AXIS, linewidth=0.6)
    ax.set_title("")  # drop xarray's automatic "date = ..., spatial_ref = 0" title
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_axis_off()
    ax.set_aspect("equal")


def make_figure(sid: str, storm_label: str, out: Path) -> Path:
    adm0 = codab.load_codab_from_blob()
    dates = storm_window(sid)
    roll2 = load_rainfall(dates, adm0)

    q80_by_date = roll2.quantile(0.8, dim=["x", "y"])
    peak = q80_by_date.idxmax("date").values
    check_against_stored(sid, float(q80_by_date.max()))

    da = roll2.sel(date=peak)
    valid = da.notnull()
    n_valid = int(valid.sum())
    stats = {"Mean": float(da.mean())}
    stats.update({f"{q * 100:.0f}th pct": float(da.quantile(q)) for q in QUANTILES})

    peak_end = pd.Timestamp(peak)
    window = f"{(peak_end - pd.Timedelta(days=1)):%d}-{peak_end:%d %b %Y}"

    fig, axes = plt.subplots(3, 2, figsize=(12, 9.5), dpi=200)
    fig.patch.set_facecolor("#fcfcfb")

    ax = axes[0, 0]
    cmap = ListedColormap(BLUES[: len(RAIN_BINS) - 1]).with_extremes(over=BLUES[-1])
    im = da.plot(ax=ax, cmap=cmap, norm=BoundaryNorm(RAIN_BINS, len(RAIN_BINS) - 1), add_colorbar=False)
    _style(ax, adm0, f"Two-day rainfall, {window}")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, extend="max")
    cb.set_label("mm over two days", color=MUTED, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)

    for ax, q in zip([axes[0, 1], axes[1, 0], axes[1, 1], axes[2, 0]], QUANTILES):
        thresh = float(da.quantile(q))
        above = (da >= thresh).where(valid)
        share = float((da >= thresh).where(valid).sum()) / n_valid * 100
        above.plot(ax=ax, cmap=ListedColormap([BELOW, ABOVE]), add_colorbar=False, vmin=0, vmax=1)
        _style(ax, adm0, f"{q * 100:.0f}th percentile = {thresh:.0f} mm\n"
                         f"{share:.0f}% of pixels at or above (blue)")

    ax = axes[2, 1]
    names, values = list(stats), list(stats.values())
    ypos = np.arange(len(names))[::-1]
    ax.barh(ypos, values, color=ABOVE, height=0.55)
    for y, v in zip(ypos, values):
        ax.text(v + max(values) * 0.02, y, f"{v:.0f} mm", va="center", fontsize=9, color=INK)
    ax.set_yticks(ypos, names, fontsize=9, color=INK)
    ax.set_xlim(0, max(values) * 1.18)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_title("The same rainfall, five ways", loc="left", fontsize=10, color=INK)

    fig.suptitle(f"{storm_label}: one storm's rainfall as each aggregation sees it",
                 x=0.02, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("storm", nargs="?", default="oscar",
                   help="Storm name (ike, gustav, irma, ian, oscar) or raw IBTrACS SID.")
    p.add_argument("--label", default=None, help="Storm label for the title. Default: the name given.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG. Default book_cub_hurricanes/figures/rainfall_aggregation_<storm>.png")
    p.add_argument("--upload", action="store_true",
                   help=f"Also upload the PNG to blob (projects/{BLOB_DIR}).")
    args = p.parse_args()

    sid = resolve_sid(args.storm)
    label = args.label or args.storm.capitalize()
    out = args.out or Path("book_cub_hurricanes/figures") / f"rainfall_aggregation_{args.storm.lower()}.png"
    saved = make_figure(sid, label, out)
    print(f"Saved {saved}")

    if args.upload:
        blob_name = f"{BLOB_DIR}/{saved.name}"
        _upload_blob_data(saved.read_bytes(), blob_name, stage="dev",
                          container_name="projects", content_type="image/png")
        print(f"Uploaded to blob (container=projects, stage=dev): {blob_name}")


if __name__ == "__main__":
    main()
