import pandas as pd
import geopandas as gpd
import datetime
import xarray as xr
import rioxarray as rxr
from pathlib import Path
import ocha_stratus as stratus

# Import our CHIRPS GEFS module components
from src.datasources.chirps_gefs import (
    ChirpsGefsManager,
    ChirpsGefsConfig,
    ChirpsGefsDownloader,
    ChirpsGefsLoader,
    ChirpsGefsProcessor,
)
from src.datasources import codab
from src.constants import *


cuba_gdf = codab.load_codab_from_blob()
cuba_gdf = cuba_gdf[cuba_gdf["ADM0_PCODE"] == "CU"]
cuba_gdf = cuba_gdf[["ADM0_ES", "ADM0_PCODE", "geometry"]].copy()

print(f"Loaded Cuba geometry with {len(cuba_gdf)} features")
print(f"Total bounds: {cuba_gdf.total_bounds}")
cuba_gdf.head()


config = ChirpsGefsConfig(
    geometry=cuba_gdf,
    region_name="cuba_databricks_test",
    start_date="2024-12-10",  # Single day to start
    end_date="2024-12-11",  # Just 2 days
    leadtime_days=5,  # First 5 leadtimes only
    verbose=True,
)

print(f"🌍 Region: {config.region_name}")
print(f"📅 Date range: {config.start_date} to {config.end_date}")
print(f"⏱️  Leadtime: {config.leadtime_days} days")

# Calculate total number of files: days × leadtimes
date_range = pd.date_range(config.start_date, config.end_date, freq="D")
total_files = len(date_range) * config.leadtime_days
print(f"📊 Total forecasts: {total_files} files (small test!)")

# Initialize downloader
downloader = ChirpsGefsDownloader(config)

# Download just our small test dataset
download_stats = downloader.download_date_range(
    config.start_date, config.end_date
)

print("Download Results:")
print(f"✅ Successful: {download_stats['success']}")
print(f"❌ Failed: {download_stats['failed']}")
print(f"📊 Total attempted: {download_stats['total']}")
