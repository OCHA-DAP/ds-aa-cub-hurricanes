import io
import json
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
from matplotlib import pyplot as plt

from src.constants import (
    CERF_SIDS,
    CHD_GREEN,
    D_THRESH,
    MIN_EMAIL_DISTANCE,
    SPANISH_MONTHS,
    LON_ZOOM_RANGE,
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
import ocha_stratus as stratus


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
    eligible_count = len(
        df_monitoring[df_monitoring["min_dist"] <= MIN_EMAIL_DISTANCE]
    )
    total_count = len(df_monitoring)
    skipped_count = total_count - eligible_count

    print(f"📧 Email eligibility for {fcast_obsv} monitoring:")
    print(
        f"   ✅ {eligible_count} storms within {MIN_EMAIL_DISTANCE}km "
        f"(will create plots)"
    )
    if skipped_count > 0:
        print(
            f"   ⏭️  {skipped_count} storms beyond {MIN_EMAIL_DISTANCE}km "
            f"(skipping plots)"
        )

    for monitor_id, row in df_monitoring.set_index("monitor_id").iterrows():
        # Skip plots for storms beyond email distance threshold
        if row["min_dist"] > MIN_EMAIL_DISTANCE:
            if verbose:
                print(
                    f"Skipping plots for {monitor_id}: "
                    f"distance {row['min_dist']:.1f}km > "
                    f"{MIN_EMAIL_DISTANCE}km"
                )
            continue

        for plot_type in ["map", "scatter"]:
            blob_name = get_plot_blob_name(monitor_id, plot_type)
            if blob_name in existing_plot_blobs and plot_type not in clobber:
                if verbose:
                    print(f"Skipping {blob_name}, already exists")
                continue
            print(f"Creating {blob_name}")
            create_plot(monitor_id, plot_type, fcast_obsv)


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


def create_scatter_plot(monitor_id: str, fcast_obsv: Literal["fcast", "obsv"]):
    # Check if statistics file exists first, before loading data
    blob_name = f"{PROJECT_PREFIX}/processed/stats_{D_THRESH}km.csv"
    try:
        stats = stratus.load_csv_from_blob(blob_name)
    except Exception as e:
        print(f"⚠️  Could not load statistics file {blob_name}: {e}")
        print(f"⚠️  Skipping scatter plot creation for {monitor_id}")
        return

    df_monitoring = load_monitoring_data(fcast_obsv)
    monitoring_point = df_monitoring.set_index("monitor_id").loc[monitor_id]
    cuba_tz = pytz.timezone("America/Havana")
    cyclone_name = monitoring_point["name"]
    issue_time = monitoring_point["issue_time"]
    issue_time_cuba = issue_time.astimezone(cuba_tz)
    if fcast_obsv == "fcast":
        rain_plot_var = None  # No precipitation variables for forecast
        s_plot_var = "readiness_s"
        rain_col = "max_roll2_sum_rain"
        rain_source_str = "CHIRPS"
        rain_ymax = 100
        s_thresh = THRESHS["readiness"]["s"]
        rain_thresh = None  # No rain threshold for forecast
        fcast_obsv_es = "pronósticos"
        no_pass_text = "no está previsto que pase"
    else:
        rain_plot_var = "obsv_p"
        s_plot_var = "obsv_s"
        rain_col = "max_roll2_sum_rain_imerg"
        rain_source_str = "IMERG"
        rain_ymax = 170
        s_thresh = THRESHS["obsv"]["s"]
        rain_thresh = THRESHS["obsv"]["p"]
        fcast_obsv_es = "observaciones"
        no_pass_text = "no ha pasado"

    def sid_color(sid):
        color = "blue"
        if sid in CERF_SIDS:
            color = "red"
        return color

    stats["marker_size"] = stats["affected_population"] / 6e2
    stats["marker_size"] = stats["marker_size"].fillna(1)
    stats["color"] = stats["sid"].apply(sid_color)
    current_p = monitoring_point[rain_plot_var] if rain_plot_var else None
    current_s = monitoring_point[s_plot_var]
    issue_time_str_es = convert_datetime_to_es_str(issue_time_cuba)

    date_str = (
        f"Pronóstico "
        f'{monitoring_point["issue_time"].strftime("%Hh%M %d %b UTC")}'
    )

    for en_mo, es_mo in SPANISH_MONTHS.items():
        date_str = date_str.replace(en_mo, es_mo)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    ax.scatter(
        stats["max_wind"],
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
            (stats["max_wind"][j] + 0.5, stats[rain_col][j]),
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
        np.arange(s_thresh, 200, 1),
        rain_thresh,
        200,
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

    ax.set_xlim(right=155, left=0)
    ax.set_ylim(top=rain_ymax, bottom=0)

    ax.set_xlabel("Velocidad máxima del viento (nudos)")
    ax.set_ylabel(
        "Precipitaciones durante dos días consecutivos máximo,\n"
        f"promedio sobre toda la superficie (mm, {rain_source_str})"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Comparación de precipitaciones, viento, e impacto\n"
        f"Umbral de distancia = {D_THRESH} km"
    )

    if monitoring_point["min_dist"] >= D_THRESH:
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
            f"a menos de {D_THRESH} km de Cuba",
            fontsize=30,
            color="grey",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
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

    # adm0 outline
    for geom in adm.geometry[0].geoms:
        x, y = geom.exterior.coords.xy
        fig.add_trace(
            go.Scattermapbox(
                lon=list(x),
                lat=list(y),
                mode="lines",
                line_color="grey",
                showlegend=False,
            )
        )
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
                        lon=[-72.3],
                        lat=[19],
                        mode="markers",
                        marker=dict(size=50, color="red"),
                    )
                )
            fig.add_trace(
                go.Scattermapbox(
                    lon=[-72.3],
                    lat=[19],
                    mode="text+markers",
                    text=[rain_level_str],
                    marker=dict(size=40, color="blue"),
                    textfont=dict(size=20, color="white"),
                    hoverinfo="none",
                )
            )
    adm_centroid = adm.to_crs(3857).centroid.to_crs(4326)[0]
    centroid_lat, centroid_lon = adm_centroid.y, adm_centroid.x

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
        zoom = 5.8
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
