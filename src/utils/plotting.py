from __future__ import annotations

import math
import re
import typing

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Circle

# -------------------------------
# Helpers
# -------------------------------


def _to_equal_area(
    gdf: gpd.GeoDataFrame, crs_equal_area: str = "EPSG:6933"
) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError(
            "GeoDataFrame has no CRS; set it with gdf.set_crs(...)"
        )
    return gdf.to_crs(crs_equal_area)


def _radii_from_population(
    pop: np.ndarray, area_per_person: float
) -> np.ndarray:
    areas = np.maximum(pop, 0.0) * float(area_per_person)
    return np.sqrt(areas / math.pi)


def _resolve_overlaps(
    centers: np.ndarray,
    radii: np.ndarray,
    home: np.ndarray,
    *,
    k_spring: float = 1e-3,
    k_repulse: float = 1.0,
    friction: float = 0.85,
    max_step: float = 5_000.0,
    iters: int = 400,
) -> np.ndarray:
    N = centers.shape[0]
    pos = centers.astype(float).copy()
    vel = np.zeros_like(pos)
    for _ in range(iters):
        force = -k_spring * (pos - home)
        for i in range(N):
            for j in range(i + 1, N):
                dx, dy = pos[j] - pos[i]
                dist = math.hypot(dx, dy)
                min_d = radii[i] + radii[j]
                if dist < 1e-9:
                    dist = 1e-9
                    ux, uy = 1.0, 0.0
                else:
                    ux, uy = dx / dist, dy / dist
                overlap = min_d - dist
                if overlap > 0:
                    f = k_repulse * overlap
                    force[i] -= f * np.array([ux, uy])
                    force[j] += f * np.array([ux, uy])
        vel = vel + force
        vel *= friction
        step = np.linalg.norm(vel, axis=1, keepdims=True)
        clip = np.where(step > max_step, max_step / (step + 1e-12), 1.0)
        vel *= clip
        pos += vel
    return pos


# -------------------------------
# Public API
# -------------------------------


def build_circle_template(
    gdf_admin: gpd.GeoDataFrame,
    *,
    id_col: str = "ADM3_PCODE",
    pop_col: str = "pop_total",
    area_per_person: float = 2_000.0,
    crs_equal_area: str = "EPSG:6933",
    iters: int = 400,
    k_spring: float = 1e-3,
    k_repulse: float = 1.0,
) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
    [id_col, 'x','y','radius_total','pop_total']
    (+ any passthrough cols you merge later).
    Coordinates are in the equal-area CRS.
    """
    g = _to_equal_area(
        gdf_admin[[id_col, pop_col, "geometry"]], crs_equal_area=crs_equal_area
    ).copy()
    g = g.dropna(subset=[id_col, pop_col])

    home = np.vstack(
        g.geometry.centroid.apply(lambda p: (p.x, p.y)).to_numpy()
    )
    pop = g[pop_col].to_numpy(float)
    radii = _radii_from_population(pop, area_per_person=area_per_person)

    final = _resolve_overlaps(
        centers=home.copy(),
        radii=radii,
        home=home,
        k_spring=k_spring,
        k_repulse=k_repulse,
        iters=iters,
    )

    out = pd.DataFrame(
        {
            id_col: g[id_col].to_numpy(),
            "x": final[:, 0],
            "y": final[:, 1],
            "radius_total": radii,
            "pop_total": pop,
        }
    )
    return out


def plot_template_circles(
    template_df: pd.DataFrame,
    *,
    id_col: str = "ADM3_PCODE",
    label_col: str = None,  # e.g., 'name' if you've merged it into template_df
    min_font: float = 6,
    max_font: float = 16,
    fig_size: tuple = (10, 8),
    bounds: tuple | None = None,  # (xmin, xmax, ymin, ymax)
    outline_color: str = "black",
    outline_alpha: float = 1.0,
    outline_lw: float = 0.2,
    dpi: int = 200,
):
    """
    Plot ONLY the outer, unfilled circles from the template
    (for debugging sizing/layout).
    Labels (if label_col provided) are centered and scaled by total population.
    """
    t = template_df.copy()
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    # label font scaling (by total pop)
    p = t["pop_total"].to_numpy() if "pop_total" in t.columns else np.array([])
    if p.size:
        pmin, pmax = float(p.min()), float(p.max())
        denom = (pmax - pmin) if pmax > pmin else 1.0
        font_sizes = min_font + (p - pmin) / denom * (max_font - min_font)
    else:
        font_sizes = np.array([])

    for i, row in t.iterrows():
        x, y = float(row["x"]), float(row["y"])
        r_total = float(row["radius_total"])
        ax.add_patch(
            Circle(
                (x, y),
                r_total,
                fill=False,
                lw=outline_lw,
                alpha=outline_alpha,
                edgecolor=outline_color,
            )
        )

        if label_col and label_col in t.columns:
            fs = (
                float(font_sizes[i])
                if i in t.index and i < len(font_sizes)
                else min_font
            )
            ax.text(
                x,
                y,
                str(row[label_col]),
                ha="center",
                va="center",
                fontsize=fs,
            )

    ax.set_aspect("equal")

    # bounds
    if bounds is not None:
        xmin, xmax, ymin, ymax = bounds
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        cx, cy = t["x"].to_numpy(), t["y"].to_numpy()
        rt = t["radius_total"].to_numpy()
        if len(cx):
            xmin = np.min(cx - rt)
            xmax = np.max(cx + rt)
            ymin = np.min(cy - rt)
            ymax = np.max(cy + rt)
            pad = 0.03 * max(xmax - xmin, ymax - ymin)
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)

    return fig, ax


def lighten(color, amount=0.5):
    """Blend color toward white by `amount` (0=no change, 1=white)."""
    c = np.array(to_rgb(color))
    return to_hex(1 - amount * (1 - c))


def plot_bullseye_exposures(
    template_df: pd.DataFrame,
    exposures_df: pd.DataFrame,
    *,
    id_col: str = "ADM3_PCODE",
    pop_exposed_col: str = "pop_exposed",
    speed_col: str = "buffer_speed",
    # Draw order should be from largest expected radius → smallest
    # (e.g., lower thresholds first: 34, then 50, then 64)
    speeds_order=(34, 50, 64),
    colors: dict[int, str] = {34: "gold", 50: "crimson", 64: "indigo"},
    enforce_monotonic: bool = True,  # enforce non-increasing radii as speed
    # template-draw options:
    draw_template_first: bool = True,
    label_col: str | None = None,
    min_font: float = 6,
    max_font: float = 16,
    fig_size: tuple = (10, 8),
    bounds: tuple | None = None,
    outline_color: str = "black",
    outline_alpha: float = 1.0,
    outline_lw: float = 0.2,
    dpi: int = 200,
    # legend options:
    legend_title: str = "Population exposed\n(circle size ∝ population)",
    # Accept either a format string (uses {spd}) or a callable spd -> str
    legend_label_fmt: str | typing.Callable[[int], str] = "{spd} kt",
    legend_fontsize: float = 7,
    legend_title_fontsize: float = 8,
):
    """
    Draws (optionally) the template empty circles first,
    then concentric filled disks for exposures.
    Inner radii per admin:
        r_speed = r_total * sqrt(pop_exposed / pop_total), clipped to r_total.

    Parameters
    ----------
    speeds_order
        Iterable of wind thresholds in the order they should be drawn
        (largest radius first).
    legend_title
        Title for the legend. Set to None to omit legend entirely.
    legend_label_fmt
        Either a format string using "{spd}" (e.g., "{spd} kt" or "≥{spd} kt"),
        or a callable like `lambda s: f"{s} kt"`.
    """

    # --- helpers -------------------------------------------------------------
    def _label_for(spd: int) -> str:
        if callable(legend_label_fmt):
            return legend_label_fmt(spd)
        return str(legend_label_fmt).format(spd=spd)

    # Prep exposures wide table
    exp = exposures_df[[id_col, speed_col, pop_exposed_col]].copy()
    exp = exp.groupby([id_col, speed_col], as_index=False)[
        pop_exposed_col
    ].sum()
    exp_wide = exp.pivot(
        index=id_col, columns=speed_col, values=pop_exposed_col
    ).fillna(0.0)

    # Merge with template
    t = template_df.copy()
    t = t.merge(exp_wide, left_on=id_col, right_index=True, how="left").fillna(
        0.0
    )

    # Start figure (and optionally draw template)
    colors_pale = {
        s: lighten(colors.get(s, "gray"))
        for s in set(colors) | set(speeds_order)
    }
    if draw_template_first:
        fig, ax = plot_template_circles(
            t,
            id_col=id_col,
            label_col=label_col,
            min_font=min_font,
            max_font=max_font,
            fig_size=fig_size,
            bounds=bounds,
            outline_color=outline_color,
            outline_alpha=outline_alpha,
            outline_lw=outline_lw,
            dpi=dpi,
        )
    else:
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    # Draw exposures: largest radius first so smaller ones are visible on top
    for _, row in t.iterrows():
        x, y = float(row["x"]), float(row["y"])
        r_total = float(row["radius_total"])
        pop_total = max(float(row["pop_total"]), 1e-12)

        # raw radii by speed
        r = {}
        for spd in speeds_order:
            pop_exp = float(row.get(spd, 0.0))
            rs = r_total * math.sqrt(max(pop_exp, 0.0) / pop_total)
            r[spd] = min(rs, r_total)

        # enforce non-increasing radii as speed increases
        # (generic to any speeds_order)
        if enforce_monotonic and len(speeds_order) >= 2:
            prev = None
            for spd in speeds_order:
                if prev is None:
                    prev = r.get(spd, 0.0)
                else:
                    r[spd] = min(r.get(spd, 0.0), prev)
                    prev = r[spd]

        for spd in speeds_order:  # e.g., 34 first, then 50, then 64
            rs = r.get(spd, 0.0)
            if rs > 0:
                ax.add_patch(
                    Circle(
                        (x, y),
                        rs,
                        color=colors_pale.get(spd, "gray"),
                        lw=0,
                    )
                )

    # If we didn’t draw template first, set bounds now
    if not draw_template_first:
        if bounds is not None:
            xmin, xmax, ymin, ymax = bounds
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        else:
            cx, cy = t["x"].to_numpy(), t["y"].to_numpy()
            rt = t["radius_total"].to_numpy()
            if len(cx):
                xmin = np.min(cx - rt)
                xmax = np.max(cx + rt)
                ymin = np.min(cy - rt)
                ymax = np.max(cy + rt)
                pad = 0.03 * max(xmax - xmin, ymax - ymin)
                ax.set_xlim(xmin - pad, xmax + pad)
                ax.set_ylim(ymin - pad, ymax + pad)

    # --- legend (order and labels based on speeds_order) ---
    if legend_title is not None:
        legend_patches = [
            mpatches.Patch(
                facecolor=colors_pale.get(spd, "gray"), label=_label_for(spd)
            )
            for spd in speeds_order
        ]
        ax.legend(
            handles=legend_patches,
            title=legend_title,
            frameon=True,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


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
