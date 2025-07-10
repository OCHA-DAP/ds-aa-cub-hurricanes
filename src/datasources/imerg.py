import numpy as np
import ocha_stratus as stratus
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm


def load_imerg(pcode: str):
    query = f"""
    SELECT * FROM public.imerg
    WHERE pcode = '{pcode}'
    """
    df = pd.read_sql(
        query, stratus.get_engine("prod"), parse_dates=["valid_date"]
    )
    return df


def get_blob_name(date: pd.Timestamp):
    return f"imerg/daily/late/v7/processed/imerg-daily-late-{date.date()}.tif"


def open_imerg_raster(date: pd.Timestamp):
    blob_name = get_blob_name(date)
    return stratus.open_blob_cog(
        blob_name, container_name="raster", stage="prod"
    )


def open_imerg_raster_dates(dates, disable_progress_bar: bool = True):
    das = []
    error_dates = []
    for date in tqdm(dates, disable=disable_progress_bar):
        try:
            da_in = open_imerg_raster(date)
        except Exception as e:
            print(date)
            print(e)
            error_dates.append(date)
            continue
        da_in.attrs["_FillValue"] = np.nan
        da_in = da_in.rio.write_crs(4326)
        da_in = da_in.where(da_in >= 0).squeeze(drop=True)
        da_in["date"] = date
        da_in = da_in.persist()
        das.append(da_in)
    da = xr.concat(das, dim="date")
    if len(error_dates) > 0:
        print(f"Error dates: {error_dates}")
    return da


def load_imerg_recent(recent: bool = False):
    query = """
    SELECT valid_date, mean
    FROM public.imerg
    WHERE pcode = 'HT'
    """
    df = pd.read_sql(
        query, stratus.get_engine(stage="prod"), parse_dates=["valid_date"]
    )
    df = df.rename(columns={"valid_date": "date"})
    if recent:
        df = df[df["date"] >= "2024-06-01"]
    return df
