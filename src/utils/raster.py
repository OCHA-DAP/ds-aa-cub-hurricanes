import numpy as np
import xarray as xr


def upsample_dataarray(
    da: xr.DataArray,
    resolution: float = 0.1,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
) -> xr.DataArray:
    new_lat = np.arange(
        da[lat_dim].min() - 1, da[lat_dim].max() + 1, resolution
    )
    new_lon = np.arange(
        da[lon_dim].min() - 1, da[lon_dim].max() + 1, resolution
    )
    return da.interp(
        coords={
            lat_dim: new_lat,
            lon_dim: new_lon,
        },
        method="nearest",
        kwargs={"fill_value": "extrapolate"},
    )
