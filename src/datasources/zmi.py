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


def load_zmi():
    # Create a Polygon object
    poly = Polygon(ZMI_COORDS)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(
        index=[0], geometry=[poly], crs="EPSG:4326"
    )  # WGS84

    return gdf
