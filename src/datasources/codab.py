import ocha_stratus as stratus
import requests

from src.constants import ISO3, PROJECT_PREFIX
from src.utils import blob_utils


def get_blob_name(iso3: str = ISO3):
    iso3 = iso3.lower()
    return f"{PROJECT_PREFIX}/raw/codab/{iso3}.shp.zip"


def download_codab_to_blob(iso3: str = ISO3):
    url = f"https://data.fieldmaps.io/cod/originals/{iso3.lower()}.shp.zip"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    blob_name = get_blob_name(iso3)
    blob_utils._upload_blob_data(response.content, blob_name, stage="dev")


def load_codab_from_blob(
    iso3: str = ISO3, admin_level: int = 0, aoi_only: bool = False
):
    iso3 = iso3.lower()
    shapefile = f"{iso3}_adm{admin_level}.shp"
    gdf = stratus.load_shp_from_blob(
        blob_name=get_blob_name(iso3),
        shapefile=shapefile,
        stage="dev",
    )
    return gdf
