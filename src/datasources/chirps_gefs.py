"""
Generalized CHIRPS GEFS data processor

This module provides a flexible, configuration-driven approach to downloading,
processing, and analyzing CHIRPS GEFS precipitation forecast data for any
geographic region.

Key Features:
- Accept any GeoDataFrame for area of interest
- Configurable date ranges and parameters
- Modular design with separate concerns
- Robust error handling and logging
- Support for different storage backends

Storage Integration:
- Uses ocha-stratus for most blob operations (upload/download/load)
- Uses src.utils.blob.list_container_blobs for listing blobs
- Maintains compatibility with existing blob storage workflows
"""

import datetime
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import ocha_stratus as stratus
import pandas as pd
import rioxarray as rxr
import xarray as xr
from azure.core.exceptions import ResourceNotFoundError
from tqdm import tqdm

import logging

print("All imports successful!")

# Project constant
PROJECT_PREFIX = "ds-aa-cub-hurricanes"


@dataclass
class ChirpsGefsConfig:
    """Configuration class for CHIRPS GEFS processing."""

    # Area of interest
    geometry: gpd.GeoDataFrame
    region_name: str

    # Date range configuration
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None  # Defaults to yesterday

    # Forecast configuration
    leadtime_days: int = 16

    # Storage configuration
    blob_base_dir: str = "raw/chirps_gefs"
    processed_blob_dir: str = "processed/chirps_gefs"

    # URL template
    base_url: str = (
        "https://data.chc.ucsb.edu/products/EWX/data/forecasts/"
        "CHIRPS-GEFS_precip_v12/daily_16day/"
        "{iss_year}/{iss_month:02d}/{iss_day:02d}/"
        "data.{valid_year}.{valid_month:02d}{valid_day:02d}.tif"
    )

    # Processing options
    clobber: bool = False
    verbose: bool = False

    def __post_init__(self):
        """Set default end_date if not provided."""
        if self.end_date is None:
            self.end_date = (
                datetime.date.today() - datetime.timedelta(days=1)
            ).isoformat()

        # Validate geometry
        if not isinstance(self.geometry, gpd.GeoDataFrame):
            raise ValueError("geometry must be a GeoDataFrame")

        if self.geometry.empty:
            raise ValueError("geometry cannot be empty")

        # Ensure region_name is filesystem-safe
        self.region_name = "".join(
            c for c in self.region_name if c.isalnum() or c in "-_"
        ).lower()


class ChirpsGefsDownloader:
    """Handles downloading of CHIRPS GEFS data."""

    def __init__(self, config: ChirpsGefsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.total_bounds = self.config.geometry.total_bounds

    def get_existing_files(self, year: Optional[int] = None) -> List[str]:
        """Get list of existing files in blob storage."""
        search_prefix = (
            f"{PROJECT_PREFIX}/{self.config.blob_base_dir}/"
            f"{self.config.region_name}"
        )
        if year:
            search_prefix += f"_issued-{year}"

        # Use stratus for listing blobs instead of src.utils.blob
        try:
            blob_names = stratus.list_container_blobs(
                name_starts_with=search_prefix,
                stage="dev",
                container_name="projects",
            )
        except Exception as e:
            self.logger.warning(f"Error listing blobs: {e}")
            blob_names = []

        return blob_names

    def get_missing_dates(
        self, date_range: pd.DatetimeIndex
    ) -> List[pd.Timestamp]:
        """Identify dates that haven't been downloaded yet."""
        existing_files = self.get_existing_files()

        existing_issue_dates = []
        for file_path in existing_files:
            try:
                # Extract issue date from filename
                filename = Path(file_path).name
                if "issued-" in filename and "valid-" in filename:
                    issue_str = filename.split("issued-")[1].split("_valid-")[
                        0
                    ]
                    existing_issue_dates.append(pd.Timestamp(issue_str))
            except (IndexError, ValueError) as e:
                self.logger.warning(
                    f"Could not extract date from filename {filename}: {e}"
                )

        missing_dates = [
            d for d in date_range if d not in existing_issue_dates
        ]
        self.logger.info(
            f"Found {len(existing_issue_dates)} existing files, "
            f"{len(missing_dates)} dates to download"
        )

        return missing_dates

    def generate_url(
        self, issue_date: pd.Timestamp, valid_date: pd.Timestamp
    ) -> str:
        """Generate download URL for specific issue and valid dates."""
        return self.config.base_url.format(
            iss_year=issue_date.year,
            iss_month=issue_date.month,
            iss_day=issue_date.day,
            valid_year=valid_date.year,
            valid_month=valid_date.month,
            valid_day=valid_date.day,
        )

    def generate_filename(
        self, issue_date: pd.Timestamp, valid_date: pd.Timestamp
    ) -> str:
        """Generate standardized filename."""
        return (
            f"chirps-gefs-{self.config.region_name}_"
            f"issued-{issue_date.date()}_valid-{valid_date.date()}.tif"
        )

    def generate_blob_path(self, filename: str) -> str:
        """Generate full blob storage path."""
        return (
            f"{PROJECT_PREFIX}/{self.config.blob_base_dir}/"
            f"{self.config.region_name}/{filename}"
        )

    def download_single_file(
        self,
        issue_date: pd.Timestamp,
        valid_date: pd.Timestamp,
        existing_files: List[str],
    ) -> bool:
        """Download a single CHIRPS GEFS file."""
        filename = self.generate_filename(issue_date, valid_date)
        blob_path = self.generate_blob_path(filename)

        # Check if file already exists
        if blob_path in existing_files and not self.config.clobber:
            if self.config.verbose:
                self.logger.debug(f"File already exists: {filename}")
            return True

        url = self.generate_url(issue_date, valid_date)

        try:
            with rxr.open_rasterio(url) as da:
                # Clip to area of interest
                da_aoi = da.rio.clip_box(*self.total_bounds)

                # Upload directly to blob using COG upload function
                # Note: upload_cog_to_blob expects (DataArray, blob_name)
                stratus.upload_cog_to_blob(da_aoi, blob_path)

            if self.config.verbose:
                self.logger.debug(f"Successfully downloaded: {filename}")
            return True

        except Exception as e:
            self.logger.warning(f"Failed to download {filename}: {e}")
            return False

    def download_date_range(
        self, start_date: str, end_date: str
    ) -> Dict[str, int]:
        """Download CHIRPS GEFS data for a date range."""
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        missing_dates = self.get_missing_dates(date_range)

        if not missing_dates:
            self.logger.info("No new data to download")
            return {"total": 0, "success": 0, "failed": 0}

        self.logger.info(
            f"Downloading {len(missing_dates)} issue dates: "
            f"{[str(d.date()) for d in missing_dates[:5]]}"
            f"{'...' if len(missing_dates) > 5 else ''}"
        )

        existing_files = self.get_existing_files()
        total_downloads = 0
        successful_downloads = 0
        failed_downloads = 0

        for issue_date in tqdm(missing_dates, disable=not sys.stdout.isatty()):
            for leadtime in range(self.config.leadtime_days):
                valid_date = issue_date + pd.Timedelta(days=leadtime)
                total_downloads += 1

                if self.download_single_file(
                    issue_date, valid_date, existing_files
                ):
                    successful_downloads += 1
                else:
                    failed_downloads += 1

        stats = {
            "total": total_downloads,
            "success": successful_downloads,
            "failed": failed_downloads,
        }

        self.logger.info(f"Download complete: {stats}")
        return stats

    def download_recent(self, days_back: int = 30) -> Dict[str, int]:
        """Download recent CHIRPS GEFS data."""
        end_date = datetime.date.today() + datetime.timedelta(days=1)
        start_date = end_date - datetime.timedelta(days=days_back)

        return self.download_date_range(
            start_date.isoformat(), end_date.isoformat()
        )

    def download_all(self) -> Dict[str, int]:
        """Download all CHIRPS GEFS data in configured date range."""
        return self.download_date_range(
            self.config.start_date, self.config.end_date
        )


class ChirpsGefsLoader:
    """Handles loading of CHIRPS GEFS data."""

    def __init__(self, config: ChirpsGefsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def load_raster(
        self, issue_date: pd.Timestamp, valid_date: pd.Timestamp
    ) -> xr.DataArray:
        """Load a single CHIRPS GEFS raster."""
        filename = (
            f"chirps-gefs-{self.config.region_name}_"
            f"issued-{issue_date.date()}_valid-{valid_date.date()}.tif"
        )
        blob_path = (
            f"{PROJECT_PREFIX}/{self.config.blob_base_dir}/"
            f"{self.config.region_name}/{filename}"
        )

        try:
            # Use stratus.open_blob_cog for efficient COG loading
            da = stratus.open_blob_cog(blob_path)
            da = da.squeeze(drop=True)
            return da
        except Exception as e:
            raise ResourceNotFoundError(f"Could not load {filename}: {e}")

    def load_processed_data(self, dataset_name: str) -> pd.DataFrame:
        """Load processed data from blob storage."""
        blob_path = (
            f"{PROJECT_PREFIX}/{self.config.processed_blob_dir}/"
            f"{self.config.region_name}/{dataset_name}"
        )

        try:
            if dataset_name.endswith(".parquet"):
                return stratus.load_parquet_from_blob(blob_name=blob_path)
            elif dataset_name.endswith(".csv"):
                return stratus.load_csv_from_blob(blob_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset_name}")
        except Exception as e:
            raise ResourceNotFoundError(f"Could not load {dataset_name}: {e}")

    def load_raster_time_series(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        chunks: Optional[Dict[str, int]] = None,
        lazy: bool = True,
    ) -> xr.Dataset:
        """
        Load multiple CHIRPS GEFS rasters as a time series dataset.

        Uses lazy loading with dask for memory efficiency and speed.

        Args:
            start_date: Start date (YYYY-MM-DD). Uses config.start_date if None
            end_date: End date (YYYY-MM-DD). Uses config.end_date if None
            chunks: Chunking specification for dask arrays
            lazy: If True, use lazy loading. If False, load data immediately.

        Returns:
            xarray.Dataset with dimensions
            [issue_date, valid_date, leadtime, y, x]
        """
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date

        # Optimize chunking for regional COGs
        if chunks is None:
            # For small regions like Cuba (~220x100 pixels), use full
            # spatial chunks to avoid dask performance warnings
            chunks = {"x": -1, "y": -1}  # -1 means use full dimension

        self.logger.info(f"Loading time series: {start_date} to {end_date}")

        issue_date_range = pd.date_range(
            start=start_date, end=end_date, freq="D"
        )

        # Build list of all potential files
        lazy_arrays = []

        for issue_date in issue_date_range:
            for leadtime in range(self.config.leadtime_days):
                valid_date = issue_date + pd.Timedelta(days=leadtime)

                filename = (
                    f"chirps-gefs-{self.config.region_name}_"
                    f"issued-{issue_date.date()}_valid-{valid_date.date()}.tif"
                )
                blob_path = (
                    f"{PROJECT_PREFIX}/{self.config.blob_base_dir}/"
                    f"{self.config.region_name}/{filename}"
                )

                try:
                    # Create lazy DataArray - data not loaded until accessed
                    da = stratus.open_blob_cog(blob_path, chunks=chunks)
                    da = da.squeeze(drop=True)

                    # Add coordinate information
                    da = da.expand_dims(
                        {
                            "issue_date": [issue_date],
                            "valid_date": [valid_date],
                            "leadtime": [leadtime],
                        }
                    )

                    # CRITICAL: Rechunk after expand_dims to prevent
                    # chunk multiplication
                    if chunks:
                        da = da.chunk(chunks)

                    lazy_arrays.append(da)

                except Exception as e:
                    if self.config.verbose:
                        self.logger.debug(f"Skipping {blob_path}: {e}")
                    continue

        if not lazy_arrays:
            self.logger.warning(
                f"No files found for {start_date} to {end_date}"
            )
            return xr.Dataset()

        self.logger.info(f"Creating dataset from {len(lazy_arrays)} files")

        # OPTIMIZED: Single concat with chunk warning suppression
        # The "Increasing number of chunks" warning is cosmetic for our
        # use case - performance is actually good with our chunking strategy

        # Sort arrays by issue_date, then leadtime for consistent structure
        lazy_arrays.sort(
            key=lambda x: (x.issue_date.values[0], x.leadtime.values[0])
        )

        # Suppress the specific dask chunk multiplication warning
        # This warning appears because xarray creates coordinate chunks
        # for each new dimension added by expand_dims, but our actual
        # performance is optimal with the full spatial chunks we use
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*Increasing number of chunks.*",
                category=UserWarning,
            )

            # Single concat operation along forecast_time dimension
            combined_ds = xr.concat(lazy_arrays, dim="forecast_time")

        # Apply final consistent chunking
        if chunks:
            combined_ds = combined_ds.chunk(chunks)

        final_ds = combined_ds

        # Convert to Dataset if needed
        if isinstance(final_ds, xr.DataArray):
            final_ds = final_ds.to_dataset(name="precipitation")

        # Add metadata
        final_ds.attrs.update(
            {
                "title": f"CHIRPS GEFS forecasts for "
                f"{self.config.region_name}",
                "source": "CHIRPS-GEFS v12",
                "region": self.config.region_name,
                "date_range": f"{start_date} to {end_date}"[0:70],
                "loaded_files": len(lazy_arrays),
                "processing_mode": "lazy_loading" if lazy else "eager_loading",
            }
        )

        self.logger.info(
            f"Dataset created with dimensions: {dict(final_ds.dims)}"
        )
        self.logger.info("✨ Lazy loading enabled - data loads on-demand")

        return final_ds


def load_cog_stack_optimized(
    container_client, blob_paths_with_coords, chunks=None, logger=None
):
    """
    Load COG stack from Azure Blob Storage with optimized chunking.

    Avoids dask chunk multiplication warnings by:
    1. Loading all COGs with consistent chunking from the start
    2. Using xr.open_mfdataset for efficient multi-file handling
    3. Avoiding multiple expand_dims operations that fragment chunks

    Args:
        container_client: Azure ContainerClient from stratus.azure_blob
        blob_paths_with_coords: List of tuples
            (blob_path, issue_date, valid_date, leadtime)
        chunks: Chunk specification for dask arrays
        logger: Optional logger instance

    Returns:
        xarray.Dataset: Optimized dataset with proper coordinate structure
    """
    import tempfile
    import os
    from contextlib import contextmanager

    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    @contextmanager
    def temp_cog_files(container_client, blob_paths):
        """Context manager to create temporary COG files from blob."""
        temp_files = []
        temp_dir = tempfile.mkdtemp()

        try:
            for i, (blob_path, issue_date, valid_date, leadtime) in enumerate(
                blob_paths
            ):
                try:
                    # Download blob data
                    blob_client = container_client.get_blob_client(blob_path)
                    blob_data = blob_client.download_blob().readall()

                    # Create temporary file
                    temp_file = os.path.join(temp_dir, f"temp_cog_{i:03d}.tif")
                    with open(temp_file, "wb") as f:
                        f.write(blob_data)

                    temp_files.append(
                        (temp_file, issue_date, valid_date, leadtime)
                    )

                except Exception as e:
                    logger.debug(f"Could not download {blob_path}: {e}")
                    continue

            yield temp_files

        finally:
            # Clean up temporary files
            import shutil

            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    # Use temporary files approach for efficient multi-file loading
    with temp_cog_files(
        container_client, blob_paths_with_coords
    ) as temp_files:
        if not temp_files:
            return xr.Dataset()

        # Create coordinate arrays for all files at once
        issue_dates = []
        valid_dates = []
        leadtimes = []
        file_paths = []

        for temp_file, issue_date, valid_date, leadtime in temp_files:
            file_paths.append(temp_file)
            issue_dates.append(issue_date)
            valid_dates.append(valid_date)
            leadtimes.append(leadtime)

        # Use xr.open_mfdataset for optimized multi-file loading
        # This avoids the chunk multiplication issues
        try:
            ds = xr.open_mfdataset(
                file_paths,
                chunks=chunks,
                concat_dim="time",
                combine="nested",
                engine="rasterio",
                decode_coords="all",
            )

            # Add coordinate information efficiently
            ds = ds.assign_coords(
                {
                    "issue_date": ("time", issue_dates),
                    "valid_date": ("time", valid_dates),
                    "leadtime": ("time", leadtimes),
                }
            )

            # Rename time dimension for clarity
            ds = ds.rename({"time": "forecast_time"})

            # Convert to Dataset with proper variable name
            if hasattr(ds, "band_data"):
                ds = ds.rename({"band_data": "precipitation"})
            elif len(ds.data_vars) == 1:
                var_name = list(ds.data_vars)[0]
                ds = ds.rename({var_name: "precipitation"})

            return ds

        except Exception as e:
            logger.warning(
                f"xr.open_mfdataset failed: {e}, "
                f"falling back to individual loading"
            )

            # Fallback: individual file loading if multi-file fails
            arrays = []
            for temp_file, issue_date, valid_date, leadtime in temp_files:
                try:
                    da = rxr.open_rasterio(temp_file, chunks=chunks)
                    da = da.squeeze(drop=True)

                    # Add coordinates without expand_dims to avoid
                    # chunking issues
                    da = da.assign_coords(
                        {
                            "issue_date": issue_date,
                            "valid_date": valid_date,
                            "leadtime": leadtime,
                        }
                    )

                    arrays.append(da)

                except Exception as e:
                    logger.debug(f"Could not load {temp_file}: {e}")
                    continue

            if arrays:
                # Single concat operation with consistent chunking
                combined = xr.concat(arrays, dim="forecast_time")

                # Apply consistent chunking to final result
                if chunks:
                    combined = combined.chunk(chunks)

                # Convert to Dataset
                if isinstance(combined, xr.DataArray):
                    combined = combined.to_dataset(name="precipitation")

                return combined
            else:
                return xr.Dataset()


class ChirpsGefsProcessor:
    """Handles processing of CHIRPS GEFS data."""

    def __init__(self, config: ChirpsGefsConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.loader = ChirpsGefsLoader(config)

    def process_spatial_mean(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        output_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Calculate spatial mean from CHIRPS-GEFS forecasts."""
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date

        if output_name is None:
            start_year = pd.Timestamp(start_date).year
            end_year = pd.Timestamp(end_date).year
            output_name = (
                f"{self.config.region_name}_chirps_gefs_mean_daily_"
                f"{start_year}_{end_year}.parquet"
            )

        self.logger.info(
            f"Processing spatial mean for {start_date} to {end_date}"
        )

        issue_date_range = pd.date_range(
            start=start_date, end=end_date, freq="D"
        )

        dfs = []
        processed_count = 0
        failed_count = 0

        for issue_date in tqdm(
            issue_date_range, disable=not sys.stdout.isatty()
        ):
            das_i = []

            # Load all leadtimes for this issue date
            for leadtime in range(self.config.leadtime_days):
                valid_date = issue_date + pd.Timedelta(days=leadtime)
                try:
                    da_in = self.loader.load_raster(issue_date, valid_date)
                    da_in["valid_date"] = valid_date
                    das_i.append(da_in)
                except ResourceNotFoundError:
                    if self.config.verbose:
                        self.logger.debug(
                            (
                                f"No data for {issue_date.date()} -> "
                                f"{valid_date.date()}"
                            )
                        )

            if das_i:
                try:
                    # Combine all leadtimes
                    da_i = xr.concat(das_i, dim="valid_date")

                    # Clip to geometry and calculate spatial mean
                    da_i_clip = da_i.rio.clip(
                        self.config.geometry.geometry, all_touched=True
                    )
                    df_in = (
                        da_i_clip.mean(dim=["x", "y"])
                        .to_dataframe(name="mean")["mean"]
                        .reset_index()
                    )
                    df_in["issue_date"] = issue_date

                    # Convert datetime columns to date objects
                    df_in["issue_date"] = pd.to_datetime(
                        df_in["issue_date"]
                    ).dt.date
                    df_in["valid_date"] = pd.to_datetime(
                        df_in["valid_date"]
                    ).dt.date
                    dfs.append(df_in)
                    processed_count += 1

                except Exception as e:
                    self.logger.warning(
                        f"Failed to process {issue_date.date()}: {e}"
                    )
                    failed_count += 1
            else:
                if self.config.verbose:
                    self.logger.debug(
                        f"No files for issue_date {issue_date.date()}"
                    )
                failed_count += 1

        if not dfs:
            self.logger.warning(
                (
                    f"No data was successfully processed for {start_date} to "
                    f"{end_date}. "
                )
                + f"Processed: {processed_count}, Failed: {failed_count}"
            )
            # Return empty DataFrame instead of raising error
            return pd.DataFrame(columns=["issue_date", "valid_date", "mean"])

        # Combine all results
        df = pd.concat(dfs, ignore_index=True)

        # Save to blob storage
        blob_path = (
            f"{PROJECT_PREFIX}/{self.config.processed_blob_dir}/"
            f"{self.config.region_name}/{output_name}"
        )
        stratus.upload_parquet_to_blob(df, blob_path)

        self.logger.info(
            (
                f"Processing complete: {processed_count} successful, "
                f"{failed_count} failed"
            )
        )
        self.logger.info(f"Saved results to: {output_name}")

        return df

    def process_recent_updates(
        self, existing_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Process only recent data, updating existing processed dataset."""
        current_year = datetime.date.today().year

        # Load existing data if not provided
        if existing_df is None:
            recent_filename = (
                f"{self.config.region_name}_chirps_gefs_mean_daily_"
                f"since{current_year}.parquet"
            )
            try:
                existing_df = self.loader.load_processed_data(recent_filename)
                self.logger.info(
                    f"Loaded existing data with {len(existing_df)} records"
                )
            except ResourceNotFoundError:
                self.logger.warning(
                    "No existing recent data found, starting fresh"
                )
                existing_df = pd.DataFrame(
                    columns=["issue_date", "valid_date", "mean"]
                )

        # Determine date range to process
        start_date = f"{current_year}-01-01"
        end_date = (
            datetime.date.today() + datetime.timedelta(days=1)
        ).isoformat()

        issue_date_range = pd.date_range(
            start=start_date, end=end_date, freq="D"
        )

        # Filter to unprocessed dates
        existing_issue_dates = (
            pd.to_datetime(existing_df["issue_date"]).dt.date
            if not existing_df.empty
            else []
        )
        unprocessed_dates = [
            d for d in issue_date_range if d.date() not in existing_issue_dates
        ]

        if not unprocessed_dates:
            self.logger.info("No new dates to process")
            return existing_df

        self.logger.info(
            f"Processing {len(unprocessed_dates)} new issue dates"
        )

        # Process new data
        new_df = self.process_spatial_mean(
            start_date=unprocessed_dates[0].isoformat(),
            end_date=unprocessed_dates[-1].isoformat(),
            output_name=None,  # Don't save intermediate results
        )

        # Combine with existing data
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)

        # Save updated dataset
        recent_filename = (
            f"{self.config.region_name}_chirps_gefs_mean_daily_"
            f"since{current_year}.parquet"
        )
        blob_path = (
            f"{PROJECT_PREFIX}/{self.config.processed_blob_dir}/"
            f"{self.config.region_name}/{recent_filename}"
        )
        stratus.upload_parquet_to_blob(updated_df, blob_path)

        self.logger.info(f"Updated dataset saved: {recent_filename}")

        return updated_df


class ChirpsGefsManager:
    """Main interface for CHIRPS GEFS operations."""

    def __init__(
        self, geometry: gpd.GeoDataFrame, region_name: str, **config_kwargs
    ):
        """
        Initialize CHIRPS GEFS manager.

        Args:
            geometry: GeoDataFrame defining the area of interest
            region_name: Name identifier for the region
            **config_kwargs: Additional configuration parameters
        """
        self.config = ChirpsGefsConfig(
            geometry=geometry, region_name=region_name, **config_kwargs
        )

        self.downloader = ChirpsGefsDownloader(self.config)
        self.loader = ChirpsGefsLoader(self.config)
        self.processor = ChirpsGefsProcessor(self.config)
        self.logger = logging.getLogger(__name__)

    def run_full_pipeline(
        self, download_recent_only: bool = True, days_back: int = 30
    ) -> Dict[str, Union[Dict, pd.DataFrame]]:
        """
        Run the complete CHIRPS GEFS pipeline.

        Args:
            download_recent_only: If True, only download recent data
            days_back: Number of days back to download if
                download_recent_only=True

        Returns:
            Dictionary with download stats and processed DataFrame
        """
        self.logger.info(
            f"Starting CHIRPS GEFS pipeline for region: "
            f"{self.config.region_name}"
        )

        # Step 1: Download data
        if download_recent_only:
            # Use config date range if specified, otherwise download recent
            if (
                self.config.start_date != "2000-01-01"
                or self.config.end_date
                != (
                    datetime.date.today() - datetime.timedelta(days=1)
                ).isoformat()
            ):
                # User specified custom date range, use it
                download_stats = self.downloader.download_date_range(
                    self.config.start_date, self.config.end_date
                )
            else:
                # Use recent download
                download_stats = self.downloader.download_recent(
                    days_back=days_back
                )
        else:
            download_stats = self.downloader.download_all()

        # Step 2: Process data (only if some downloads were successful)
        if download_stats.get("success", 0) == 0:
            self.logger.warning(
                f"No files were successfully downloaded. "
                f"Download stats: {download_stats}"
            )
            # Return results with empty processed data but include
            # download stats
            return {
                "download_stats": download_stats,
                "processed_data": pd.DataFrame(),  # Empty DataFrame
                "config": self.config,
                "warning": (
                    "No data was downloaded - check Azure credentials "
                    "and connectivity"
                ),
            }

        self.logger.info(
            f"Processing {download_stats['success']} downloaded files..."
        )

        # Always use the configured date range for processing, not
        # the full year
        processed_df = self.processor.process_spatial_mean(
            start_date=self.config.start_date, end_date=self.config.end_date
        )

        self.logger.info("Pipeline completed successfully")

        return {
            "download_stats": download_stats,
            "processed_data": processed_df,
            "config": self.config,
        }

    def download_only(
        self, recent_only: bool = True, **kwargs
    ) -> Dict[str, int]:
        """Download CHIRPS GEFS data only."""
        if recent_only:
            return self.downloader.download_recent(**kwargs)
        else:
            return self.downloader.download_all()

    def process_only(self, recent_only: bool = True, **kwargs) -> pd.DataFrame:
        """Process CHIRPS GEFS data only."""
        if recent_only:
            return self.processor.process_recent_updates(**kwargs)
        else:
            return self.processor.process_spatial_mean(**kwargs)

    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """Load processed data."""
        return self.loader.load_processed_data(dataset_name)


def fix_date_formatting_in_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix date formatting in CHIRPS GEFS DataFrames.

    Converts datetime64[ns] columns to proper date objects to prevent
    nanosecond timestamp storage in parquet files.

    Args:
        df: DataFrame with potentially incorrectly formatted dates

    Returns:
        DataFrame with properly formatted date columns
    """
    df_fixed = df.copy()

    # Fix issue_date column if it exists
    if "issue_date" in df_fixed.columns:
        if df_fixed["issue_date"].dtype == "datetime64[ns]":
            df_fixed["issue_date"] = pd.to_datetime(
                df_fixed["issue_date"]
            ).dt.date
        elif df_fixed["issue_date"].dtype == "object":
            # Handle case where dates might be stored as large integers
            try:
                # Try to convert if they're nanosecond timestamps
                df_fixed["issue_date"] = pd.to_datetime(
                    df_fixed["issue_date"], unit="ns"
                ).dt.date
            except (ValueError, TypeError):
                # Try standard datetime parsing
                try:
                    df_fixed["issue_date"] = pd.to_datetime(
                        df_fixed["issue_date"]
                    ).dt.date
                except (ValueError, TypeError):
                    print("Warning: Could not parse issue_date column")

    # Fix valid_date column if it exists
    if "valid_date" in df_fixed.columns:
        if df_fixed["valid_date"].dtype == "datetime64[ns]":
            df_fixed["valid_date"] = pd.to_datetime(
                df_fixed["valid_date"]
            ).dt.date
        elif df_fixed["valid_date"].dtype == "object":
            # Handle case where dates might be stored as large integers
            try:
                # Try to convert if they're nanosecond timestamps
                df_fixed["valid_date"] = pd.to_datetime(
                    df_fixed["valid_date"], unit="ns"
                ).dt.date
            except (ValueError, TypeError):
                # Try standard datetime parsing
                try:
                    df_fixed["valid_date"] = pd.to_datetime(
                        df_fixed["valid_date"]
                    ).dt.date
                except (ValueError, TypeError):
                    print("Warning: Could not parse valid_date column")

    return df_fixed


def fix_existing_chirps_gefs_files(
    region_name: str, container_name: str = "projects"
) -> Dict[str, str]:
    """
    Fix date formatting in existing CHIRPS GEFS parquet files.

    This function will load, fix, and re-upload any existing CHIRPS GEFS
    parquet files that have incorrect date formatting (nanosecond timestamps).

    Args:
        region_name: The region name to fix files for
        container_name: Azure blob container name

    Returns:
        Dictionary with results of the fixing operation
    """
    import logging

    logger = logging.getLogger(__name__)

    results = {"files_checked": 0, "files_fixed": 0, "errors": []}

    # Search for existing parquet files for this region
    search_prefix = f"{PROJECT_PREFIX}/processed/chirps_gefs/{region_name}/"

    try:
        blob_names = stratus.list_container_blobs(
            name_starts_with=search_prefix,
            stage="dev",
            container_name=container_name,
        )

        parquet_files = [b for b in blob_names if b.endswith(".parquet")]
        results["files_checked"] = len(parquet_files)

        for blob_path in parquet_files:
            try:
                logger.info(f"Checking file: {blob_path}")

                # Load the file
                df = stratus.load_parquet_from_blob(blob_name=blob_path)

                # Check if it needs fixing
                needs_fixing = False
                if "issue_date" in df.columns:
                    if df["issue_date"].dtype == "datetime64[ns]" or (
                        df["issue_date"].dtype == "object"
                        and len(df) > 0
                        and isinstance(df["issue_date"].iloc[0], (int, str))
                        and str(df["issue_date"].iloc[0]).isdigit()
                        and len(str(df["issue_date"].iloc[0])) > 10
                    ):
                        needs_fixing = True

                if "valid_date" in df.columns and not needs_fixing:
                    if df["valid_date"].dtype == "datetime64[ns]" or (
                        df["valid_date"].dtype == "object"
                        and len(df) > 0
                        and isinstance(df["valid_date"].iloc[0], (int, str))
                        and str(df["valid_date"].iloc[0]).isdigit()
                        and len(str(df["valid_date"].iloc[0])) > 10
                    ):
                        needs_fixing = True

                if needs_fixing:
                    logger.info(f"Fixing date formatting in: {blob_path}")

                    # Fix the formatting
                    df_fixed = fix_date_formatting_in_dataframe(df)

                    # Re-upload the fixed file
                    stratus.upload_parquet_to_blob(df_fixed, blob_path)
                    results["files_fixed"] += 1

                    logger.info(f"Successfully fixed: {blob_path}")
                else:
                    logger.info(
                        f"File already has correct formatting: {blob_path}"
                    )

            except Exception as e:
                error_msg = f"Error processing {blob_path}: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)

    except Exception as e:
        error_msg = f"Error listing blob files: {str(e)}"
        logger.error(error_msg)
        results["errors"].append(error_msg)

    return results


# Convenience function for quick usage
def process_chirps_gefs_for_region(
    geometry: gpd.GeoDataFrame,
    region_name: str,
    recent_only: bool = True,
    **config_kwargs,
) -> Dict[str, Union[Dict, pd.DataFrame]]:
    """
    Convenience function to process CHIRPS GEFS data for any region.

    Args:
        geometry: GeoDataFrame defining the area of interest
        region_name: Name identifier for the region
        recent_only: If True, only process recent data
        **config_kwargs: Additional configuration parameters

    Returns:
        Dictionary with results from the full pipeline

    Example:
        >>> import geopandas as gpd
        >>> from cub_adapt.chirps_gefs import process_chirps_gefs_for_region
        >>>
        >>> # Load your area of interest
        >>> gdf = gpd.read_file("path/to/your/shapefile.shp")
        >>>
        >>> # Process CHIRPS GEFS data
        >>> results = process_chirps_gefs_for_region(
        ...     geometry=gdf,
        ...     region_name="my_region",
        ...     recent_only=True,
        ...     start_date="2020-01-01"
        ... )
        >>>
        >>> # Access results
        >>> download_stats = results["download_stats"]
        >>> processed_data = results["processed_data"]
    """
    manager = ChirpsGefsManager(geometry, region_name, **config_kwargs)
    return manager.run_full_pipeline(download_recent_only=recent_only)


def load_processed_chirps_gefs(variable_name: str = None):
    query = f"""
    SELECT *
    FROM projects.{PROJECT_PREFIX.replace('-', '_')}_chirps_gefs
    """
    if variable_name is not None:
        query += f"""
        WHERE variable = '{variable_name}'
        """
    return pd.read_sql(
        query, stratus.get_engine(), parse_dates=["valid_date", "issued_date"]
    )
