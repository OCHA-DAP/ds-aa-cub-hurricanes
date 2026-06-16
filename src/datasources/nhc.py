import gzip
from datetime import datetime
from ftplib import FTP
from io import BytesIO
from typing import Optional

import geopandas as gpd
import ocha_stratus as stratus
import pandas as pd
from sqlalchemy import text
from tqdm.auto import tqdm

from src.constants import PROJECT_PREFIX, current_monitoring_season


def download_historical_forecasts(
    clobber: bool = False,
    include_archive: bool = True,
    include_recent: bool = True,
    start_year: int = 2000,
    end_year: int = 2022,
):
    # cols from
    # https://www.nrlmry.navy.mil/atcf_web/docs/database/new/abdeck.txt
    # tech list from https://ftp.nhc.noaa.gov/atcf/docs/nhc_techlist.dat
    nhc_cols_str = (
        "BASIN, CY, YYYYMMDDHH, TECHNUM/MIN, TECH, TAU, LatN/S, "
        "LonE/W, VMAX, MSLP, TY, RAD, WINDCODE, RAD1, RAD2, RAD3, "
        "RAD4, POUTER, ROUTER, RMW, GUSTS, EYE, SUBREGION, MAXSEAS, "
        "INITIALS, DIR, SPEED, STORMNAME, DEPTH, SEAS, SEASCODE, "
        "SEAS1, SEAS2, SEAS3, SEAS4"
    )
    nhc_cols = nhc_cols_str.split(", ")
    nhc_cols.extend(
        [y + str(x) for x in range(1, 21) for y in ["USERDEFINE", "userdata"]]
    )

    ftp_server = "ftp.nhc.noaa.gov"
    ftp = FTP(ftp_server)
    ftp.login("", "")
    recent_directory = "/atcf/aid_public"
    archive_directory = "/atcf/archive"

    existing_files = stratus.list_container_blobs(
        name_starts_with=f"{PROJECT_PREFIX}/raw/noaa/nhc"
    )
    if include_archive:
        ftp.cwd(archive_directory)
        for year in tqdm(range(start_year, end_year + 1)):
            if ftp.pwd() != archive_directory:
                ftp.cwd("..")
            ftp.cwd(str(year))
            filenames = [
                x
                for x in ftp.nlst()
                if x.endswith(".dat.gz") and x.startswith("aal")
            ]
            for filename in filenames:
                out_blob = (
                    f"{PROJECT_PREFIX}/raw/noaa/nhc/historical_forecasts/{year}/"  # noqa
                    f"{filename.removesuffix('.dat.gz')}.csv"
                )
                if out_blob in existing_files and not clobber:
                    continue
                with BytesIO() as buffer:
                    ftp.retrbinary("RETR " + filename, buffer.write)
                    buffer.seek(0)
                    with gzip.open(buffer, "rt") as file:
                        df = pd.read_csv(file, header=None, names=nhc_cols)
                stratus.upload_csv_to_blob(df, out_blob)
            ftp.cwd("..")
        ftp.cwd("..")

    if include_recent:
        ftp.cwd(recent_directory)
        filenames = [
            x
            for x in ftp.nlst()
            if x.endswith(".dat.gz") and x.startswith("aal")
        ]
        for filename in filenames:
            out_blob = (
                f"{PROJECT_PREFIX}/raw/noaa/nhc/historical_forecasts/recent/"
                f"{filename.removesuffix('.dat.gz')}.csv"
            )
            if out_blob in existing_files and not clobber:
                continue
            with BytesIO() as buffer:
                ftp.retrbinary("RETR " + filename, buffer.write)
                buffer.seek(0)
                with gzip.open(buffer, "rt") as file:
                    df = pd.read_csv(file, header=None, names=nhc_cols)
            stratus.upload_csv_to_blob(df, out_blob)

        ftp.cwd("..")


def process_historical_forecasts():
    blob_names = stratus.list_container_blobs(
        name_starts_with=f"{PROJECT_PREFIX}/raw/noaa/nhc/historical_forecasts/"
    )
    blob_names = [x for x in blob_names if x.endswith(".csv")]

    def proc_latlon(latlon):
        c = latlon[-1]
        if c in ["N", "E"]:
            return float(latlon[:-1]) / 10
        elif c in ["S", "W"]:
            return -float(latlon[:-1]) / 10

    dfs = []
    for blob_name in tqdm(blob_names):
        df_in = stratus.load_csv_from_blob(blob_name)
        atcf_id = blob_name.removesuffix(".csv")[-8:]

        cols = ["YYYYMMDDHH", "TAU", "LatN/S", "LonE/W", "MSLP", "VMAX"]
        dff = df_in[df_in["TECH"] == " OFCL"][cols]
        if dff.empty:
            continue

        dff["issue_time"] = dff["YYYYMMDDHH"].apply(
            lambda x: datetime.strptime(str(x), "%Y%m%d%H")
        )
        dff["valid_time"] = dff.apply(
            lambda row: row["issue_time"] + pd.Timedelta(hours=row["TAU"]),
            axis=1,
        )

        dff["lat"] = dff["LatN/S"].apply(proc_latlon)
        dff["lon"] = dff["LonE/W"].apply(proc_latlon)
        dff = dff.rename(
            columns={
                "TAU": "leadtime",
                "MSLP": "pressure",
                "VMAX": "windspeed",
            }
        )
        cols = [
            "issue_time",
            "valid_time",
            "lat",
            "lon",
            "windspeed",
            "pressure",
        ]
        dff = dff[cols]
        dff = dff.loc[~dff.duplicated()]
        dff["atcf_id"] = atcf_id
        dfs.append(dff)

    df = pd.concat(dfs, ignore_index=True)
    save_blob = f"{PROJECT_PREFIX}/processed/noaa/nhc/historical_forecasts/al_2000_2024.parquet"  # noqa
    stratus.upload_parquet_to_blob(df, save_blob)


def load_historical_forecasts(include_geometry: bool = False):
    blob_name = f"{PROJECT_PREFIX}/processed/noaa/nhc/historical_forecasts/al_2000_2024.parquet"  # noqa
    df = stratus.load_parquet_from_blob(blob_name)
    if include_geometry:
        return gpd.GeoDataFrame(
            data=df,
            geometry=gpd.points_from_xy(df["lon"], df["lat"]),
            crs=4326,
        )
    else:
        return df


# Atlantic NHC tracks now come from the dev database instead of the rolling
# global CSV snapshot (noaa/nhc/{forecasted,observed}_tracks.csv). The track
# geometries live in storms.nhc_tracks_geo and storm names in
# storms.nhc_storms. Because the DB is the full 1990-onward archive (the CSV
# only ever held recent storms), the loaders scope to a single Atlantic season;
# otherwise the monitor would reprocess decades of history into the monitoring
# parquet.
NHC_TRACKS_TABLE = "storms.nhc_tracks_geo"
NHC_STORMS_TABLE = "storms.nhc_storms"


def _as_utc(series: pd.Series) -> pd.Series:
    """NHC times are UTC; the DB stores them tz-naive. Return tz-aware UTC so
    the output matches the old CSV (downstream code calls .astimezone / splits
    the isoformat offset and needs aware timestamps)."""
    series = pd.to_datetime(series)
    if series.dt.tz is None:
        return series.dt.tz_localize("UTC")
    return series.dt.tz_convert("UTC")


def _load_nhc_tracks_from_db(
    select_cols: str, leadtime_clause: str, season: int
) -> pd.DataFrame:
    """Query one Atlantic season of NHC tracks, joined to storm names.

    `select_cols` and `leadtime_clause` are fixed internal SQL fragments (not
    user input); `season` is bound as a parameter.
    """
    # Plain .format() (not an f-string) so the SQLAlchemy ":season" bind param
    # isn't tokenised as code by the linter under Python 3.12.
    query = text(
        """
        SELECT {select_cols}
        FROM {tracks} t
        JOIN {storms} s ON s.atcf_id = t.atcf_id
        WHERE s.genesis_basin = 'NA'
          AND s.season = :season
          AND {leadtime_clause}
        """.format(
            select_cols=select_cols,
            tracks=NHC_TRACKS_TABLE,
            storms=NHC_STORMS_TABLE,
            leadtime_clause=leadtime_clause,
        )
    )
    return pd.read_sql(
        query, stratus.get_engine("dev"), params={"season": season}
    )


def load_recent_glb_nhc(
    fcast_obsv: str = "fcast", season: Optional[int] = None
):
    """
    Load the most recent NHC forecast or observed tracks.

    Parameters:
    fcast_obsv (str): "fcast" for forecasts, "obsv" for observations.
    season (int): Atlantic season (year) to load. Defaults to current year.

    Returns:
    pd.DataFrame: DataFrame containing the tracks.
    """
    if fcast_obsv == "fcast":
        return load_recent_glb_forecasts(season=season)
    elif fcast_obsv == "obsv":
        return load_recent_glb_obsv(season=season)
    else:
        raise ValueError("fcast_obsv must be 'fcast' or 'obsv'")


def load_recent_glb_forecasts(season: Optional[int] = None) -> pd.DataFrame:
    """Atlantic NHC forecast tracks for one season, from the dev database.

    Returns the same columns/dtypes the old global CSV did so the monitor and
    email code are unchanged: id (lowercase atcf), name, issuance, basin
    ('al'), latitude, longitude, maxwind, validTime (tz-aware UTC). Forecast
    points are the leadtime>0 rows of each issuance; the t=0 analysis point is
    served by the observed feed, matching how the CSV split the two products.
    """
    season = season or current_monitoring_season()
    select_cols = (
        "lower(t.atcf_id) AS id, "
        "s.name AS name, "
        "t.issued_time AS issuance, "
        "'al' AS basin, "
        "ST_Y(t.geometry) AS latitude, "
        "ST_X(t.geometry) AS longitude, "
        "t.wind_speed AS maxwind, "
        't.valid_time AS "validTime"'
    )
    df = _load_nhc_tracks_from_db(select_cols, "t.leadtime > 0", season)
    df["issuance"] = _as_utc(df["issuance"])
    df["validTime"] = _as_utc(df["validTime"])
    return df


def load_recent_glb_obsv(season: Optional[int] = None) -> pd.DataFrame:
    """Atlantic NHC observed (analysis) tracks for one season, from the dev
    database.

    Returns the old global-CSV columns/dtypes: id (lowercase atcf), name,
    basin ('al'), intensity, pressure, latitude, longitude, lastUpdate
    (tz-aware UTC). Observed points are the leadtime=0 analysis rows (synoptic
    6-hourly; pressure is not populated on these rows in the DB).
    """
    season = season or current_monitoring_season()
    select_cols = (
        "lower(t.atcf_id) AS id, "
        "s.name AS name, "
        "'al' AS basin, "
        "t.wind_speed AS intensity, "
        "t.pressure AS pressure, "
        "ST_Y(t.geometry) AS latitude, "
        "ST_X(t.geometry) AS longitude, "
        't.valid_time AS "lastUpdate"'
    )
    df = _load_nhc_tracks_from_db(select_cols, "t.leadtime = 0", season)
    df["lastUpdate"] = _as_utc(df["lastUpdate"])
    return df
