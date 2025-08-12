"""
NOTE FOR ANALYSTS: Environment variable testing pattern
Uncomment the block below to test FORCE_ALERT functionality locally
This patches the environment to simulate alert conditions without modifying global env vars

Usage:
1. Uncomment the entire block below
2. Run in notebook/script to test alert functionality
3. Expected outputs are marked with comments

from unittest.mock import patch
import importlib
import os

with patch.dict(os.environ, {"FORCE_ALERT": "true"}):
    import src.constants
    importlib.reload(src.constants)

    print("env var:", os.getenv("FORCE_ALERT"))                  # Should print: true
    print("reloaded:", src.constants.FORCE_ALERT)                # Should print: True ✅

    from src.email import utils
    importlib.reload(utils)  # must reload downstream module that imported it
    from src.email.utils import (
        load_monitoring_data,
        open_static_image,
        create_dummy_storm_tracks)
    df_monitoring = utils.load_monitoring_data("obsv")
"""

import io
import json
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
from matplotlib import pyplot as plt
import ocha_stratus as stratus

from src.constants import (
    CERF_SIDS,
    CHD_GREEN,
    D_THRESH,
    SPANISH_MONTHS,
    LON_ZOOM_RANGE,
    MIN_EMAIL_DISTANCE,
    PROJECT_PREFIX,
    THRESHS,
    FORCE_ALERT,
)
from src.datasources import codab, nhc, zma
from src.email.utils import (
    load_monitoring_data,
    open_static_image,
    create_dummy_storm_tracks,
)


def get_plot_blob_name(monitor_id, plot_type: Literal["map", "scatter"]):
    fcast_obsv = "fcast" if "fcast" in monitor_id.lower() else "obsv"
    return (
        f"{PROJECT_PREFIX}/plots/{fcast_obsv}/" f"{monitor_id}_{plot_type}.png"
    )


def convert_datetime_to_es_str(x: pd.Timestamp) -> str:
    es_str = x.strftime("%Hh%M, %-d %b")
    for en_mo, es_mo in SPANISH_MONTHS.items():
        es_str = es_str.replace(en_mo, es_mo)
    return es_str


def update_plots(
    fcast_obsv: Literal["fcast", "obsv"],
    clobber: list = None,
    verbose: bool = False,
):
    if clobber is None:
        clobber = []
    df_monitoring = load_monitoring_data(fcast_obsv)
    existing_plot_blobs = stratus.list_container_blobs(
        name_starts_with=f"{PROJECT_PREFIX}/plots/{fcast_obsv}/"
    )

    # Log email eligibility summary
    # Info emails use MIN_EMAIL_DISTANCE criteria, not ZMA
    eligible_df = df_monitoring[
        df_monitoring["min_dist"] <= MIN_EMAIL_DISTANCE
    ]
    eligible_count = len(eligible_df)
    total_count = len(df_monitoring)
    skipped_count = total_count - eligible_count

    print(f"📧 Email eligibility for {fcast_obsv} monitoring:")
    if eligible_count > 0:
        # Show details about eligible storms and plot creation status
        plots_to_create = 0
        plots_already_exist = 0

        for _, row in eligible_df.iterrows():
            storm_date = row["issue_time"].strftime("%Y-%m-%d")
            monitor_id = str(row.name)  # Convert to string for consistency

            # Check if plots need to be created for this storm
            will_create_plots = False
            for plot_type in ["map", "scatter"]:
                blob_name = get_plot_blob_name(monitor_id, plot_type)
                if (
                    blob_name not in existing_plot_blobs
                    or plot_type in clobber
                ):
                    will_create_plots = True
                    break

            if will_create_plots:
                plots_to_create += 1
                status_icon = "🔄"
                status_text = "will create plots"
            else:
                plots_already_exist += 1
                status_icon = "✅"
                status_text = "plots exist"

            print(
                f"   {status_icon} {row['name']} ({storm_date}): "
                f"{row['min_dist']:.1f}km - {status_text}"
            )

        # Summary message
        if plots_to_create > 0 and plots_already_exist > 0:
            print(
                f"   📊 Total: {eligible_count} eligible storms "
                f"({plots_to_create} will create plots, "
                f"{plots_already_exist} plots exist)"
            )
        elif plots_to_create > 0:
            print(
                f"   📊 Total: {eligible_count} storms " f"(will create plots)"
            )
        else:
            print(
                f"   📊 Total: {eligible_count} storms "
                f"(all plots already exist)"
            )
    else:
        print(
            f"   ⚠️  No storms within {MIN_EMAIL_DISTANCE}km "
            f"(no plots to create)"
        )

    if skipped_count > 0:
        print(
            f"   ⏭️  {skipped_count} storms beyond distance threshold "
            f"(skipping plots)"
        )

    for monitor_id, row in df_monitoring.set_index("monitor_id").iterrows():
        # Skip plots for storms beyond distance threshold
        # (info emails use MIN_EMAIL_DISTANCE)
        if row["min_dist"] > MIN_EMAIL_DISTANCE:
            if verbose:
                print(
                    f"Skipping plots for {monitor_id}: "
                    f"storm beyond {MIN_EMAIL_DISTANCE}km threshold"
                )
            continue

        for plot_type in ["map", "scatter"]:
            blob_name = get_plot_blob_name(monitor_id, plot_type)
            if blob_name in existing_plot_blobs and plot_type not in clobber:
                if verbose:
                    print(f"Skipping {blob_name}, already exists")
                continue
            print(f"Creating {blob_name}")
            try:
                create_plot(monitor_id, plot_type, fcast_obsv)
            except Exception as e:
                print(
                    f"❌ Failed to create {plot_type} plot for "
                    f"{monitor_id}: {e}"
                )
                print("🔄 Continuing with next plot...")
                continue


def create_plot(
    monitor_id: str,
    plot_type: Literal["map", "scatter"],
    fcast_obsv: Literal["fcast", "obsv"],
):
    if plot_type == "map":
        create_map_plot(monitor_id, fcast_obsv)
    elif plot_type == "scatter":
        create_scatter_plot(monitor_id, fcast_obsv)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")


def create_1d_plot(stats, monitoring_point):
    """Create a 1D wind speed plot for forecast data."""
    # Hardcode forecast-specific parameters
    s_thresh = THRESHS["readiness"]["s"]
    fcast_obsv_es = "pronósticos"
    no_pass_text = "no está previsto que pase"

    # Extract needed values from monitoring_point
    cyclone_name = monitoring_point["name"]
    current_s = monitoring_point["readiness_s"]
    issue_time = monitoring_point["issue_time"]
    cuba_tz = pytz.timezone("America/Havana")
    issue_time_cuba = issue_time.astimezone(cuba_tz)
    issue_time_str_es = convert_datetime_to_es_str(issue_time_cuba)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Create 1D plot along x-axis (wind speed) with random jitter
    y_positions = np.random.normal(0.5, 0.1, len(stats))

    ax.scatter(
        stats["wind_plot"],
        y_positions,
        s=stats["marker_size"],
        c=stats["color"],
        alpha=0.6,
        edgecolors="none",
    )

    # Add storm name annotations
    for j, txt in enumerate(
        stats["name"].str.capitalize() + "\n" + stats["year"].astype(str)
    ):
        ax.annotate(
            txt.capitalize(),
            (stats["wind_plot"][j] + 0.5, y_positions[j]),
            ha="left",
            va="center",
            fontsize=7,
        )

    # Mark current storm
    current_y = 0.5  # Center position
    ax.scatter(
        [current_s],
        [current_y],
        marker="x",
        color=CHD_GREEN,
        linewidths=4,
        s=150,
        zorder=10,
    )
    ax.annotate(
        f"{cyclone_name}",
        (current_s, current_y + 0.15),
        va="center",
        ha="center",
        color=CHD_GREEN,
        fontweight="bold",
        fontsize=10,
    )
    ax.annotate(
        f"{fcast_obsv_es} emitidas {issue_time_str_es}",
        (current_s, current_y - 0.15),
        va="center",
        ha="center",
        color=CHD_GREEN,
        fontstyle="italic",
        fontsize=8,
    )

    # Add threshold line
    ax.axvline(
        x=s_thresh, color="orange", linewidth=2, linestyle="--", alpha=0.8
    )
    ax.fill_betweenx(
        [0, 1],
        s_thresh,
        155,
        color="gold",
        alpha=0.2,
        zorder=-1,
    )

    # Threshold annotation
    ax.annotate(
        "Umbral de activación",
        (s_thresh + 2, 0.9),
        ha="left",
        va="top",
        color="orange",
        fontweight="bold",
        fontsize=9,
    )

    # CERF legend
    ax.annotate(
        "Asignaciones CERF en rojo",
        (5, 0.9),
        ha="left",
        va="top",
        color="crimson",
        fontstyle="italic",
        fontsize=9,
    )

    ax.set_xlim(left=0, right=175)
    ax.set_ylim(bottom=0, top=1)

    # Hide y-axis as it's just for display purposes
    ax.set_yticks([])
    ax.set_xlabel("Velocidad máxima del viento (nudos)", fontsize=12)
    ax.set_ylabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_title(
        "Comparación de velocidad del viento e impacto\n"
        "Zona de Máxima Atención",
        fontsize=12,
        pad=20,
    )

    # Add overlay if storm is outside ZMA
    if not monitoring_point["in_zma"]:
        rect = plt.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            color="white",
            alpha=0.7,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            0.5,
            0.5,
            f"{cyclone_name} {no_pass_text}\n"
            "por la Zona de Máxima Atención",
            fontsize=30,
            color="grey",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    return fig, ax


def create_2d_plot(stats, monitoring_point):
    """Create a 2D scatter plot for observation data."""
    # Hardcode observation-specific parameters
    rain_col = "q80_roll2"
    s_thresh = THRESHS["obsv"]["s"]
    rain_thresh = THRESHS["obsv"]["p"]
    rain_ymax = 270
    rain_source_str = "IMERG"
    fcast_obsv_es = "observaciones"
    no_pass_text = "no ha pasado"

    # Extract needed values from monitoring_point
    cyclone_name = monitoring_point["name"]
    current_s = monitoring_point["obsv_s"]
    current_p = monitoring_point["obsv_p"]
    issue_time = monitoring_point["issue_time"]
    cuba_tz = pytz.timezone("America/Havana")
    issue_time_cuba = issue_time.astimezone(cuba_tz)
    issue_time_str_es = convert_datetime_to_es_str(issue_time_cuba)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    ax.scatter(
        stats["wind_plot"],
        stats[rain_col],
        s=stats["marker_size"],
        c=stats["color"],
        alpha=0.5,
        edgecolors="none",
    )

    for j, txt in enumerate(
        stats["name"].str.capitalize() + "\n" + stats["year"].astype(str)
    ):
        ax.annotate(
            txt.capitalize(),
            (stats["wind_plot"][j] + 0.5, stats[rain_col][j]),
            ha="left",
            va="center",
            fontsize=7,
        )

    ax.scatter(
        [current_s],
        [current_p],
        marker="x",
        color=CHD_GREEN,
        linewidths=3,
        s=100,
    )
    ax.annotate(
        f"   {cyclone_name}\n\n",
        (current_s, current_p),
        va="center",
        ha="left",
        color=CHD_GREEN,
        fontweight="bold",
    )
    ax.annotate(
        f"\n   {fcast_obsv_es} emitidas" f"\n   {issue_time_str_es}",
        (current_s, current_p),
        va="center",
        ha="left",
        color=CHD_GREEN,
        fontstyle="italic",
    )

    ax.axvline(x=s_thresh, color="lightgray", linewidth=0.5)
    ax.axhline(y=rain_thresh, color="lightgray", linewidth=0.5)
    ax.fill_between(
        np.arange(s_thresh, rain_ymax, 1),
        rain_thresh,
        rain_ymax,
        color="gold",
        alpha=0.2,
        zorder=-1,
    )

    ax.annotate(
        "\nZona de activación   ",
        (155, rain_ymax),
        ha="right",
        va="top",
        color="orange",
        fontweight="bold",
    )
    ax.annotate(
        "\n\nAsignaciones CERF en rojo   ",
        (155, rain_ymax),
        ha="right",
        va="top",
        color="crimson",
        fontstyle="italic",
    )

    ax.set_xlim(right=175, left=0)
    ax.set_ylim(top=rain_ymax, bottom=0)

    ax.set_xlabel("Velocidad máxima del viento (nudos)")
    ax.set_ylabel(
        "Precipitaciones durante dos días consecutivos máximo,\n"
        f"percentil 80 sobre toda la superficie (mm, {rain_source_str})"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        "Comparación de precipitaciones, viento, e impacto\n"
        "Zona de Máxima Atención",
    )

    # Add overlay if storm is outside ZMA
    if not monitoring_point["in_zma"]:
        rect = plt.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            color="white",
            alpha=0.7,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            0.5,
            0.5,
            f"{cyclone_name} {no_pass_text}\n"
            "por la Zona de Máxima Atención",
            fontsize=30,
            color="grey",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

    return fig, ax


def create_scatter_plot(monitor_id: str, fcast_obsv: Literal["fcast", "obsv"]):
    # Check if statistics file exists first, before loading data
    blob_name = (
        f"{PROJECT_PREFIX}/processed/storm_stats/stats_with_targets.parquet"
    )
    blob_name = f"{PROJECT_PREFIX}/processed/fcast_obsv_combined_stats.parquet"

    try:
        stats = stratus.load_parquet_from_blob(blob_name)
    except Exception as e:
        print(f"⚠️  Could not load statistics file {blob_name}: {e}")
        print(f"⚠️  Skipping scatter plot creation for {monitor_id}")
        return

    df_monitoring = load_monitoring_data(fcast_obsv)
    monitoring_point = df_monitoring.set_index("monitor_id").loc[monitor_id]

    stats["color"] = stats["cerf"].apply(lambda x: "crimson" if x else "grey")
    # stats["color"] = stats["cerf_str"].apply(
    #     lambda x: "red" if x == "True" else "blue"
    # )

    stats.rename(
        columns={"Total Affected": "affected_population"}, inplace=True
    )
    # stats.columns
    # # stats["year"] = stats["valid_time_min"].dt.year
    stats["marker_size"] = stats["affected_population"] / 6e2
    stats["marker_size"] = stats["marker_size"].fillna(1)

    # Create plot based on fcast_obsv type
    if fcast_obsv == "fcast":
        stats["wind_plot"] = stats["wind"]
        fig, ax = create_1d_plot(stats, monitoring_point)
    else:
        stats["wind_plot"] = stats["wind_obsv"]
        stats["q80_roll2"] = stats["q80_obsv"]
        fig, ax = create_2d_plot(stats, monitoring_point)

    # Common save logic
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    blob_name = get_plot_blob_name(monitor_id, "scatter")

    # Upload blob data using stratus container client
    try:
        container_client = stratus.get_container_client(write=True)
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(buffer.getvalue(), overwrite=True)
        print(f"✅ Successfully uploaded scatter plot: {blob_name}")
    except Exception as e:
        print(f"⚠️  Could not upload scatter plot {blob_name}: {e}")

    plt.close(fig)


def create_map_plot_figure(
    monitor_id: str, fcast_obsv: Literal["fcast", "obsv"]
) -> go.Figure:
    """Create a map plot figure object without saving to blob storage.

    Args:
        monitor_id: The monitoring ID to create plot for
        fcast_obsv: Whether this is forecast or observation data

    Returns:
        Plotly Figure object that can be displayed or saved
    """
    try:
        adm = codab.load_codab_from_blob(admin_level=0)
        # Drastically simplify Cuba outline to avoid Kaleido timeout
        # This reduces geometric complexity while preserving the basic shape
        adm = adm.to_crs(3857)  # Project to meters for better simplification
        adm_centroid = adm.to_crs(3857).centroid.to_crs(4326)[0]
        centroid_lat, centroid_lon = adm_centroid.y, adm_centroid.x
    except Exception as e:
        print(f"⚠️  Could not load CODAB administrative boundaries: {e}")
        print(f"⚠️  Skipping map plot creation for {monitor_id}")
        return None

    try:
        trig_zone = zma.load_zma()
    except Exception as e:
        print(f"⚠️  Could not load ZMA trigger zone data: {e}")
        print(f"⚠️  Skipping map plot creation for {monitor_id}")
        return None
    lts = {
        "action": {
            "color": "darkorange",
            "plot_color": "black",
            "dash": "solid",
            "label": "Action",
            "zorder": 2,
            "lt_max": pd.Timedelta(days=3),
            "lt_min": pd.Timedelta(days=-1),
            "threshs": {
                "wind_dist": THRESHS["action"]["s"],
                "dist_min": D_THRESH,
            },
        },
        "readiness": {
            "color": "dodgerblue",
            "plot_color": "grey",
            "dash": "dot",
            "label": "Movilización",
            "zorder": 1,
            "lt_max": pd.Timedelta(days=5),
            "lt_min": pd.Timedelta(days=2),
            "threshs": {
                "wind_dist": THRESHS["readiness"]["s"],
                "dist_min": D_THRESH,
            },
        },
        "obsv": {
            "color": "dodgerblue",
            "plot_color": "black",
            "dash": "dot",
            "label": "Observacional",
            "zorder": 1,
            "lt_max": pd.Timedelta(days=0),
            "lt_min": pd.Timedelta(days=0),
            "threshs": {
                "roll2_rain_dist": THRESHS["obsv"]["p"],
                "wind_dist": THRESHS["obsv"]["s"],
                "dist_min": D_THRESH,
            },
        },
    }
    df_monitoring = load_monitoring_data(fcast_obsv)
    monitoring_point = df_monitoring.set_index("monitor_id").loc[monitor_id]
    cuba_tz = pytz.timezone("America/Havana")
    cyclone_name = monitoring_point["name"]
    atcf_id = monitoring_point["atcf_id"]
    # if atcf_id == "TEST_ATCF_ID":
    #     atcf_id = "al182024"  # Use Hurricane Rafael for Cuba tests
    issue_time = monitoring_point["issue_time"]
    issue_time_cuba = issue_time.astimezone(cuba_tz)

    df_tracks = nhc.load_recent_glb_nhc(fcast_obsv=fcast_obsv)

    if FORCE_ALERT:
        df_dummy_tracks = create_dummy_storm_tracks(
            df_tracks, fcast_obsv=fcast_obsv
        )
        df_tracks = pd.concat([df_tracks, df_dummy_tracks], ignore_index=True)

    if fcast_obsv == "fcast":
        tracks_f = df_tracks[
            (df_tracks["id"] == atcf_id)
            & (df_tracks["issuance"] == issue_time)
        ].copy()

    elif fcast_obsv == "obsv":
        tracks_f = df_tracks[
            (df_tracks["id"] == atcf_id)
            & (df_tracks["lastUpdate"] <= issue_time)
        ].copy()
        tracks_f = tracks_f.rename(
            columns={"lastUpdate": "validTime", "intensity": "maxwind"}
        )
        tracks_f["issuance"] = tracks_f["validTime"]

    tracks_f["validTime_cuba"] = tracks_f["validTime"].apply(
        lambda x: x.astimezone(cuba_tz)
    )
    tracks_f["valid_time_str"] = tracks_f["validTime_cuba"].apply(
        convert_datetime_to_es_str
    )

    tracks_f["lt"] = tracks_f["validTime"] - tracks_f["issuance"]
    # No precipitation variables for forecast monitoring
    rain_plot_var = None if fcast_obsv == "fcast" else "obsv_p"
    rain_level = monitoring_point[rain_plot_var] if rain_plot_var else None
    fig = go.Figure()

    # buffer
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=json.loads(trig_zone.geometry.to_json()),
            locations=trig_zone.index,
            z=[1],
            colorscale="Reds",
            marker_opacity=0.2,
            showscale=False,
            marker_line_width=0,
            hoverinfo="none",
        )
    )

    relevant_lts = (
        ["readiness", "action"] if fcast_obsv == "fcast" else ["obsv"]
    )
    for lt_name in relevant_lts:
        lt_params = lts[lt_name]
        if lt_name == "obsv":
            dff = tracks_f.copy()
        else:
            dff = tracks_f[
                (tracks_f["lt"] <= lt_params["lt_max"])
                & (tracks_f["lt"] >= lt_params["lt_min"])
            ]
        # triggered points
        dff_trig = dff[
            (dff["maxwind"] >= lt_params["threshs"]["wind_dist"])
            & (dff["lt"] >= lt_params["lt_min"])
        ]
        fig.add_trace(
            go.Scattermapbox(
                lon=dff_trig["longitude"],
                lat=dff_trig["latitude"],
                mode="markers",
                marker=dict(size=50, color="red"),
            )
        )
        # all points
        fig.add_trace(
            go.Scattermapbox(
                lon=dff["longitude"],
                lat=dff["latitude"],
                mode="markers+text+lines",
                marker=dict(size=40, color=lt_params["plot_color"]),
                text=dff["maxwind"].astype(str),
                line=dict(width=2, color=lt_params["plot_color"]),
                textfont=dict(size=20, color="white"),
                customdata=dff["valid_time_str"],
                hovertemplate=("Hora válida: %{customdata}<extra></extra>"),
            )
        )

        # rainfall
        if lt_name in ["readiness", "obsv"] and rain_level is not None:
            # rain_level = dff["roll2_rain_dist"].max()
            if pd.isnull(rain_level):
                rain_level_str = ""
            else:
                rain_level_str = int(rain_level)
            if rain_level > lt_params["threshs"]["roll2_rain_dist"]:
                fig.add_trace(
                    go.Scattermapbox(
                        lon=[centroid_lon],
                        lat=[centroid_lat],
                        mode="markers",
                        marker=dict(size=50, color="red"),
                    )
                )
            fig.add_trace(
                go.Scattermapbox(
                    lon=[centroid_lon],
                    lat=[centroid_lat],
                    mode="text+markers",
                    text=[rain_level_str],
                    marker=dict(size=40, color="blue"),
                    textfont=dict(size=20, color="white"),
                    hoverinfo="none",
                )
            )

    if fcast_obsv == "fcast":
        lat_max = max(tracks_f["latitude"])
        lat_max = max(lat_max, centroid_lat)
        lat_min = min(tracks_f["latitude"])
        lat_min = min(lat_min, centroid_lat)
        lon_max = max(tracks_f["longitude"])
        lon_max = max(lon_max, centroid_lon)
        lon_min = min(tracks_f["longitude"])
        lon_min = min(lon_min, centroid_lon)
        width_to_height = 1
        margin = 1.7
        height = (lat_max - lat_min) * margin * width_to_height
        width = (lon_max - lon_min) * margin
        lon_zoom = np.interp(width, LON_ZOOM_RANGE, range(20, 0, -1))
        lat_zoom = np.interp(height, LON_ZOOM_RANGE, range(20, 0, -1))
        zoom = round(min(lon_zoom, lat_zoom), 2)
        center_lat = (lat_max + lat_min) / 2
        center_lon = (lon_max + lon_min) / 2
    else:
        # Static Cuba-centered view for observations with lower zoom
        zoom = 4.5  # Lowered from 5.8 to show more area
        center_lat = centroid_lat
        center_lon = centroid_lon

    issue_time_str_es = convert_datetime_to_es_str(issue_time_cuba)
    fcast_obsv_es = "Observaciones" if fcast_obsv == "obsv" else "Pronósticos"
    plot_title = (
        f"{fcast_obsv_es} NOAA para {cyclone_name}<br>"
        f"<sup>Emitidas {issue_time_str_es} (hora local Cuba)</sup>"
    )

    if fcast_obsv == "fcast":
        legend_filename = "map_legend.png"
        aspect = 1
    else:
        legend_filename = "map_legend_obsv.png"
        aspect = 1.3

    encoded_legend = open_static_image(legend_filename)

    fig.update_layout(
        title=plot_title,
        mapbox_style="open-street-map",
        mapbox_zoom=zoom,
        mapbox_center_lat=center_lat,
        mapbox_center_lon=center_lon,
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        height=850,
        width=800,
        showlegend=False,
        images=[
            dict(
                source=f"data:image/png;base64,{encoded_legend}",
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.01,
                sizex=0.3,
                sizey=0.3 / aspect,
                xanchor="left",
                yanchor="bottom",
                opacity=0.7,
            )
        ],
    )

    return fig


def save_plot_to_blob(
    fig: go.Figure, monitor_id: str, plot_type: Literal["map", "scatter"]
) -> None:
    """Save a plotly figure to blob storage.

    Args:
        fig: Plotly Figure object to save
        monitor_id: The monitoring ID for blob naming
        plot_type: Type of plot for blob naming
    """
    buffer = io.BytesIO()
    # scale corresponds to 150 dpi
    fig.write_image(buffer, format="png", scale=2.08)
    buffer.seek(0)

    blob_name = get_plot_blob_name(monitor_id, plot_type)

    # Upload blob data using stratus container client
    try:
        container_client = stratus.get_container_client(write=True)
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(buffer.getvalue(), overwrite=True)
        print(f"✅ Successfully uploaded {plot_type} plot: {blob_name}")
    except Exception as e:
        print(f"⚠️  Could not upload {plot_type} plot {blob_name}: {e}")


def create_map_plot(
    monitor_id: str, fcast_obsv: Literal["fcast", "obsv"]
) -> None:
    """Create a map plot and save it to blob storage (wrapper function).

    Args:
        monitor_id: The monitoring ID to create plot for
        fcast_obsv: Whether this is forecast or observation data
    """
    fig = create_map_plot_figure(monitor_id, fcast_obsv)
    if fig is not None:
        save_plot_to_blob(fig, monitor_id, "map")
    else:
        print(
            f"⚠️  Skipping map plot upload for {monitor_id} "
            f"due to data loading error"
        )
