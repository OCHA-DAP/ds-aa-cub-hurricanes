import ocha_stratus as stratus
import pandas as pd


def load_ibtracs_in_bounds(min_lon, min_lat, max_lon, max_lat):
    query = f"""
    SELECT * FROM storms.observed_tracks
    WHERE longitude BETWEEN {min_lon} AND {max_lon}
    AND latitude BETWEEN {min_lat} AND {max_lat}
    """
    df = pd.read_sql(query, stratus.get_engine("dev"))
    return df
