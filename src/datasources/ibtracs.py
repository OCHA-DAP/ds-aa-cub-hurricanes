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


def load_storms():
    query = """
    SELECT * FROM storms.storms
    """
    df = pd.read_sql(query, stratus.get_engine("dev"))
    return df


def knots2cat(knots):
    """Convert wind speed in knots to Saffir-Simpson hurricane category."""
    category = 0
    if knots >= 137:
        category = 5
    elif knots >= 113:
        category = 4
    elif knots >= 96:
        category = 3
    elif knots >= 83:
        category = 2
    elif knots >= 64:
        category = 1
    return category
