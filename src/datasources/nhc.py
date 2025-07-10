import gzip
from datetime import datetime
from ftplib import FTP
from io import BytesIO

import geopandas as gpd
import ocha_stratus as stratus
import pandas as pd
from tqdm.auto import tqdm

from src.constants import PROJECT_PREFIX


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


def load_recent_glb_forecasts():
    return stratus.load_csv_from_blob(
        "noaa/nhc/forecasted_tracks.csv",
        stage="dev",
        container_name="global",
        parse_dates=["issuance", "validTime"],
        sep=";",
    )


def load_recent_glb_obsv():
    return stratus.load_csv_from_blob(
        "noaa/nhc/observed_tracks.csv",
        stage="dev",
        container_name="global",
        parse_dates=["lastUpdate"],
        sep=";",
    )
