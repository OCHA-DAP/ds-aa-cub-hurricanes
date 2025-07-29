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
    FRENCH_MONTHS,
    LON_ZOOM_RANGE,
    PROJECT_PREFIX,
    THRESHS,
)
from src.datasources import codab, nhc
from src.email.utils import (
    TEST_FCAST_MONITOR_ID,
    TEST_OBSV_MONITOR_ID,
    TEST_STORM,
    add_test_row_to_monitoring,
    open_static_image,
)
from src.monitoring import monitoring_utils
import ocha_stratus as stratus


def get_plot_blob_name(monitor_id, plot_type: Literal["map", "scatter"]):
    fcast_obsv = "fcast" if "fcast" in monitor_id.lower() else "obsv"
    return (
        f"{PROJECT_PREFIX}/plots/{fcast_obsv}/" f"{monitor_id}_{plot_type}.png"
    )


def convert_datetime_to_fr_str(x: pd.Timestamp) -> str:
    fr_str = x.strftime("%Hh%M, %-d %b")
    for en_mo, fr_mo in FRENCH_MONTHS.items():
        fr_str = fr_str.replace(en_mo, fr_mo)
    return fr_str


def update_plots(
    fcast_obsv: Literal["fcast", "obsv"],
    clobber: list = None,
    verbose: bool = False,
):
    if clobber is None:
        clobber = []
    df_monitoring = monitoring_utils.load_existing_monitoring_points(
        fcast_obsv
    )
    if TEST_STORM:
        df_monitoring = add_test_row_to_monitoring(df_monitoring, fcast_obsv)
    existing_plot_blobs = stratus.list_container_blobs(
        name_starts_with=f"{PROJECT_PREFIX}/plots/{fcast_obsv}/"
    )

    for monitor_id, row in df_monitoring.set_index("monitor_id").iterrows():
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
    df_monitoring = monitoring_utils.load_existing_monitoring_points(
        fcast_obsv
    )
    if monitor_id in [TEST_FCAST_MONITOR_ID, TEST_OBSV_MONITOR_ID]:
        df_monitoring = add_test_row_to_monitoring(df_monitoring, fcast_obsv)
    monitoring_point = df_monitoring.set_index("monitor_id").loc[monitor_id]
    haiti_tz = pytz.timezone("America/Port-au-Prince")
    cyclone_name = monitoring_point["name"]
    issue_time = monitoring_point["issue_time"]
    issue_time_hti = issue_time.astimezone(haiti_tz)
    blob_name = f"{PROJECT_PREFIX}/processed/stats_{D_THRESH}km.csv"
    stats = stratus.load_csv_from_blob(blob_name)
    if fcast_obsv == "fcast":
        rain_plot_var = "readiness_p"
        s_plot_var = "readiness_s"
        rain_col = "max_roll2_sum_rain"
        rain_source_str = "CHIRPS"
        rain_ymax = 100
        s_thresh = THRESHS["readiness"]["s"]
        rain_thresh = THRESHS["readiness"]["p"]
        fcast_obsv_fr = "prévisions"
        no_pass_text = "pas prévu de passer"
    else:
        rain_plot_var = "obsv_p"
        s_plot_var = "obsv_s"
        rain_col = "max_roll2_sum_rain_imerg"
        rain_source_str = "IMERG"
        rain_ymax = 170
        s_thresh = THRESHS["obsv"]["s"]
        rain_thresh = THRESHS["obsv"]["p"]
        fcast_obsv_fr = "observations"
        no_pass_text = "n'a pas passé"

    def sid_color(sid):
        color = "blue"
        if sid in CERF_SIDS:
            color = "red"
        return color

    stats["marker_size"] = stats["affected_population"] / 6e2
    stats["marker_size"] = stats["marker_size"].fillna(1)
    stats["color"] = stats["sid"].apply(sid_color)
    current_p = monitoring_point[rain_plot_var]
    current_s = monitoring_point[s_plot_var]
    issue_time_str_fr = convert_datetime_to_fr_str(issue_time_hti)

    date_str = (
        f"Prévision "
        f'{monitoring_point["issue_time"].strftime("%Hh%M %d %b UTC")}'
    )

    for en_mo, fr_mo in FRENCH_MONTHS.items():
        date_str = date_str.replace(en_mo, fr_mo)

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
        f"\n   {fcast_obsv_fr} émises" f"\n   {issue_time_str_fr}",
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
        "\nZone de déclenchement   ",
        (155, rain_ymax),
        ha="right",
        va="top",
        color="orange",
        fontweight="bold",
    )
    ax.annotate(
        "\n\nAllocations CERF en rouge   ",
        (155, rain_ymax),
        ha="right",
        va="top",
        color="crimson",
        fontstyle="italic",
    )

    ax.set_xlim(right=155, left=0)
    ax.set_ylim(top=rain_ymax, bottom=0)

    ax.set_xlabel("Vitesse de vent maximum (noeuds)")
    ax.set_ylabel(
        "Précipitations pendant deux jours consécutifs maximum,\n"
        f"moyenne sur toute la superficie (mm, {rain_source_str})"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(
        f"Comparaison de précipitations, vent, et impact\n"
        f"Seuil de distance = {D_THRESH} km"
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
            f"à moins de {D_THRESH} km de Haïti",
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
    container_client = stratus.get_container_client(write=True)
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(buffer.getvalue(), overwrite=True)

    plt.close(fig)


def create_map_plot(monitor_id: str, fcast_obsv: Literal["fcast", "obsv"]):
    adm = codab.load_codab_from_blob(admin_level=0)
    trig_zone = codab.load_buffer()
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
                "roll2_rain_dist": THRESHS["action"]["p"],
                "wind_dist": THRESHS["action"]["s"],
                "dist_min": D_THRESH,
            },
        },
        "readiness": {
            "color": "dodgerblue",
            "plot_color": "grey",
            "dash": "dot",
            "label": "Mobilisation",
            "zorder": 1,
            "lt_max": pd.Timedelta(days=5),
            "lt_min": pd.Timedelta(days=2),
            "threshs": {
                "roll2_rain_dist": THRESHS["readiness"]["p"],
                "wind_dist": THRESHS["readiness"]["s"],
                "dist_min": D_THRESH,
            },
        },
        "obsv": {
            "color": "dodgerblue",
            "plot_color": "black",
            "dash": "dot",
            "label": "Observationnel",
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
    df_monitoring = monitoring_utils.load_existing_monitoring_points(
        fcast_obsv
    )
    if monitor_id in [TEST_FCAST_MONITOR_ID, TEST_OBSV_MONITOR_ID]:
        df_monitoring = add_test_row_to_monitoring(df_monitoring, fcast_obsv)
    monitoring_point = df_monitoring.set_index("monitor_id").loc[monitor_id]
    haiti_tz = pytz.timezone("America/Port-au-Prince")
    cyclone_name = monitoring_point["name"]
    atcf_id = monitoring_point["atcf_id"]
    if atcf_id == "TEST_ATCF_ID":
        atcf_id = "al022024"
    issue_time = monitoring_point["issue_time"]
    issue_time_hti = issue_time.astimezone(haiti_tz)

    if fcast_obsv == "fcast":
        df_tracks = nhc.load_recent_glb_forecasts()
        tracks_f = df_tracks[
            (df_tracks["id"] == atcf_id)
            & (df_tracks["issuance"] == issue_time)
        ].copy()
    else:
        df_tracks = nhc.load_recent_glb_obsv()
        tracks_f = df_tracks[
            (df_tracks["id"] == atcf_id)
            & (df_tracks["lastUpdate"] <= issue_time)
        ].copy()
        tracks_f = tracks_f.rename(
            columns={"lastUpdate": "validTime", "intensity": "maxwind"}
        )
        tracks_f["issuance"] = tracks_f["validTime"]

    tracks_f["validTime_hti"] = tracks_f["validTime"].apply(
        lambda x: x.astimezone(haiti_tz)
    )
    tracks_f["valid_time_str"] = tracks_f["validTime_hti"].apply(
        convert_datetime_to_fr_str
    )

    tracks_f["lt"] = tracks_f["validTime"] - tracks_f["issuance"]
    rain_plot_var = "readiness_p" if fcast_obsv == "fcast" else "obsv_p"
    rain_level = monitoring_point[rain_plot_var]
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
                hovertemplate=("Heure valide: %{customdata}<extra></extra>"),
            )
        )

        # rainfall
        if lt_name in ["readiness", "obsv"]:
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

    issue_time_str_fr = convert_datetime_to_fr_str(issue_time_hti)
    fcast_obsv_fr = "Observations" if fcast_obsv == "obsv" else "Prévisions"
    plot_title = (
        f"{fcast_obsv_fr} NOAA pour {cyclone_name}<br>"
        f"<sup>Émises {issue_time_str_fr} (heure locale Haïti)</sup>"
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

    buffer = io.BytesIO()
    # scale corresponds to 150 dpi
    fig.write_image(buffer, format="png", scale=2.08)
    buffer.seek(0)

    blob_name = get_plot_blob_name(monitor_id, "map")

    # Upload blob data using stratus container client
    container_client = stratus.get_container_client(write=True)
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(buffer.getvalue(), overwrite=True)
