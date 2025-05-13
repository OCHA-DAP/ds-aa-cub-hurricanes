import geopandas as gpd
from shapely.geometry import Polygon

ZMI_COORDS = [
    (-72.3, 18.5),
    (-78, 18.5),
    (-80, 20),
    (-85, 20),
    (-86.8, 21.2),
    (-86, 23),
    (-85, 24),
    (-81.6, 24),
    (-77, 23.5),
    (-72.3, 21),
]


def load_zma():
    """Create gdf of ZMI based on coords from Cuba Met map"""
    poly = Polygon(ZMI_COORDS)
    gdf = gpd.GeoDataFrame(index=[0], geometry=[poly], crs="EPSG:4326")
    return gdf
