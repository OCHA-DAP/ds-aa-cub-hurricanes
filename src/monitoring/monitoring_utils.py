"""
Cuba-specific hurricane monitoring with clean, modular design.
Integrates NHC track data with optional rainfall processing for Cuba.
Each function has a single responsibility and is easy to test and understand.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, Literal, Optional

import geopandas as gpd
import ocha_stratus as stratus
import pandas as pd

from src.constants import D_THRESH, NUMERIC_NAME_REGEX, PROJECT_PREFIX, THRESHS
from src.datasources import codab, imerg, nhc
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RainfallProcessor(ABC):
    """Abstract base class for rainfall data processing."""

    @abstractmethod
    def load_recent_data(self) -> pd.DataFrame:
        """Load recent rainfall data."""
        pass

    @abstractmethod
    def get_issue_times(self, rain_df: pd.DataFrame) -> pd.Series:
        """Get issue times for monitoring from rainfall data."""
        pass

    @abstractmethod
    def filter_data_by_time(
        self, rain_df: pd.DataFrame, issue_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Filter rainfall data up to a specific issue time."""
        pass

    @abstractmethod
    def get_rainfall_for_period(
        self,
        rain_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        rain_col: str = "roll2_sum",
    ) -> float:
        """Get maximum rainfall for a specific time period."""
        pass

    @abstractmethod
    def is_storm_still_active(
        self,
        rain_df: pd.DataFrame,
        track_max_time: pd.Timestamp,
        tolerance_days: int = 1,
    ) -> bool:
        """Check if storm is still active based on rainfall availability."""
        pass


class IMERGProcessor(RainfallProcessor):
    """IMERG-specific rainfall data processor."""

    def load_recent_data(self) -> pd.DataFrame:
        """Load recent IMERG data with rolling sums."""
        logger.info("Loading recent IMERG data for Cuba.")
        obsv_rain = imerg.load_imerg_recent(recent=True)
        obsv_rain["roll2_sum"] = (
            obsv_rain["mean"]
            .rolling(window=2, center=True, min_periods=1)
            .sum()
        )
        obsv_rain["issue_time"] = obsv_rain["date"].apply(
            lambda x: x.tz_localize("UTC")
        ) + pd.Timedelta(hours=15, days=1)
        return obsv_rain

    def get_issue_times(self, rain_df: pd.DataFrame) -> pd.Series:
        """Get IMERG issue times for monitoring."""
        return rain_df["issue_time"]

    def filter_data_by_time(
        self, rain_df: pd.DataFrame, issue_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Filter IMERG data up to a specific issue time."""
        return rain_df[rain_df["issue_time"] <= issue_time]

    def get_rainfall_for_period(
        self,
        rain_df: pd.DataFrame,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        rain_col: str = "roll2_sum",
    ) -> float:
        """Get maximum rainfall for a specific time period."""
        filtered_rain = rain_df[
            (rain_df["date"] >= start_date) & (rain_df["date"] <= end_date)
        ]
        return filtered_rain[rain_col].max() if not filtered_rain.empty else 0

    def is_storm_still_active(
        self,
        rain_df: pd.DataFrame,
        track_max_time: pd.Timestamp,
        tolerance_days: int = 1,
    ) -> bool:
        """Check if storm is still active based on IMERG data availability."""
        rain_max = rain_df["date"].max().date()
        track_max = track_max_time.date()
        return rain_max - track_max <= pd.Timedelta(days=tolerance_days)


class CubaHurricaneMonitor:
    """Cuba-specific hurricane monitoring with optional rainfall."""

    def __init__(self, rainfall_processor: Optional[RainfallProcessor] = None):
        self.adm0 = codab.load_codab_from_blob().to_crs(3857)
        self.rainfall_processor = rainfall_processor

    # ============================================================================
    # UTILITY METHODS - Single responsibility, highly reusable
    # ============================================================================

    def _create_monitor_id(
        self, atcf_id: str, monitoring_type: str, issue_time: pd.Timestamp
    ) -> str:
        """Create standardized monitoring ID."""
        iso_time = issue_time.isoformat().split("+")[0]
        return f"{atcf_id}_{monitoring_type}_{iso_time}"

    def _should_skip_existing(
        self, monitor_id: str, existing_data: pd.DataFrame, clobber: bool
    ) -> bool:
        """Check if monitoring point already exists."""
        # Handle empty DataFrame (no existing data)
        if existing_data.empty or "monitor_id" not in existing_data.columns:
            logger.debug(
                f"Processing monitoring for {monitor_id} (no existing data)"
            )
            return False

        exists = monitor_id in existing_data["monitor_id"].unique()
        if exists and not clobber:
            logger.debug(f"Already monitored for {monitor_id}")
            return True
        elif not exists:
            logger.debug(f"Processing monitoring for {monitor_id}")
        return False

    def _remove_track_duplicates(
        self, df: pd.DataFrame, index_col: str, atcf_id: str = None
    ) -> pd.DataFrame:
        """Remove duplicate track entries based on specified index column."""
        df = df.copy()
        if atcf_id is None:
            atcf_id = df.iloc[0]["atcf_id"]

        # Check if name is numeric (before actual name assignment)
        df["numeric_name"] = df["name"].apply(
            lambda x: bool(re.compile(NUMERIC_NAME_REGEX).search(x))
        )

        df_duplicated = df[df.duplicated(subset=[index_col], keep=False)]
        if not df_duplicated.empty:
            drop_name, drop_lastupdate = df_duplicated[
                df_duplicated["numeric_name"]
            ].iloc[0][["name", index_col]]
            df = (
                df.sort_values("numeric_name", ascending=False)
                .drop_duplicates(subset=[index_col])
                .sort_values(index_col)
            )
            logger.warning(
                f"Dropping duplicate track entry for {atcf_id} "
                f"({drop_name} at {drop_lastupdate})"
            )
        return df

    def _load_existing_monitoring(
        self, monitoring_type: Literal["fcast", "obsv"]
    ) -> pd.DataFrame:
        """Load existing monitoring points from blob storage."""
        blob_name = (
            f"{PROJECT_PREFIX}/monitoring/"
            f"cub_{monitoring_type}_monitoring.parquet"
        )
        try:
            return stratus.load_parquet_from_blob(blob_name)
        except Exception:
            logger.info(
                f"No existing {monitoring_type} monitoring data found."
            )
            return pd.DataFrame()

    # ============================================================================
    # GEOMETRIC OPERATIONS
    # ============================================================================

    def _add_distance_to_cuba(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add distance column to GeoDataFrame."""
        gdf = gdf.copy()
        gdf["distance"] = (
            gdf.geometry.distance(self.adm0.iloc[0].geometry) / 1000
        )
        return gdf

    def _interpolate_track(
        self, group: pd.DataFrame, time_col: str, data_cols: list
    ) -> pd.DataFrame:
        """Interpolate track data to 30-minute intervals."""
        return (
            group.set_index(time_col)[data_cols]
            .resample("30min")
            .interpolate()
            .reset_index()
        )

    def _create_track_geodataframe(
        self, df_interp: pd.DataFrame
    ) -> gpd.GeoDataFrame:
        """Create GeoDataFrame from interpolated track data."""
        gdf = gpd.GeoDataFrame(
            df_interp,
            geometry=gpd.points_from_xy(
                df_interp["longitude"], df_interp["latitude"]
            ),
            crs="EPSG:4326",
        ).to_crs(3857)
        return self._add_distance_to_cuba(gdf)

    # ============================================================================
    # FILTERING OPERATIONS - Simple, reusable filters
    # ============================================================================

    def _filter_by_distance(
        self, gdf: gpd.GeoDataFrame, max_dist: float = D_THRESH
    ) -> gpd.GeoDataFrame:
        """Filter GeoDataFrame to points within distance threshold."""
        return gdf[gdf["distance"] < max_dist]

    def _filter_by_leadtime(
        self, gdf: gpd.GeoDataFrame, max_days: int
    ) -> gpd.GeoDataFrame:
        """Filter GeoDataFrame to points within leadtime threshold."""
        return gdf[gdf["leadtime"] <= pd.Timedelta(days=max_days)]

    def _filter_by_time(
        self, gdf: gpd.GeoDataFrame, time_col: str, max_time: pd.Timestamp
    ) -> gpd.GeoDataFrame:
        """Filter GeoDataFrame to points before specified time."""
        return gdf[gdf[time_col] <= max_time]

    # ============================================================================
    # WIND CALCULATIONS
    # ============================================================================

    def _get_max_wind_in_period(
        self, gdf: gpd.GeoDataFrame, wind_col: str = "maxwind"
    ) -> float:
        """Get maximum wind speed in a GeoDataFrame."""
        return gdf[wind_col].max() if not gdf.empty else 0

    def _check_wind_threshold(
        self, wind_speed: float, threshold_type: str
    ) -> bool:
        """Check if wind speed meets threshold for given trigger type."""
        return wind_speed >= THRESHS[threshold_type]["s"]

    def _get_closest_approach_stats(self, gdf: gpd.GeoDataFrame) -> Dict:
        """Get statistics for closest approach to Cuba."""
        if gdf.empty:
            return {"min_dist": float("inf"), "closest_row": None}

        closest_idx = gdf["distance"].idxmin()
        closest_row = gdf.loc[closest_idx]

        return {
            "min_dist": gdf["distance"].min(),
            "closest_row": closest_row,
        }

    # ============================================================================
    # TRIGGER CALCULATIONS - Modular trigger logic
    # ============================================================================

    def _get_wind_trigger_stats(
        self,
        gdf: gpd.GeoDataFrame,
        trigger_type: str,
        wind_col: str = "maxwind",
        use_leadtime: bool = True,
    ) -> Dict:
        """Get wind-based trigger statistics for a specific trigger type."""
        # Filter by distance first
        gdf_dist = self._filter_by_distance(gdf)

        # Apply leadtime filtering if needed (forecast data only)
        if (
            use_leadtime
            and "leadtime" in gdf.columns
            and trigger_type in ["action", "readiness"]
        ):
            gdf_filtered = self._filter_by_leadtime(
                gdf_dist, THRESHS[trigger_type]["lt_days"]
            )
        else:
            gdf_filtered = gdf_dist

        # Calculate wind statistics
        max_wind = self._get_max_wind_in_period(gdf_filtered, wind_col)
        trigger_met = self._check_wind_threshold(max_wind, trigger_type)

        return {
            f"{trigger_type}_s": max_wind,
            f"{trigger_type}_trigger": trigger_met,
        }

    # ============================================================================
    # MAIN PROCESSING METHODS -  orchestration
    # ============================================================================

    def _process_single_forecast(
        self, atcf_id: str, group: pd.DataFrame, issue_time: pd.Timestamp
    ) -> Optional[Dict]:
        """Process a single forecast track into monitoring record."""
        try:
            # Interpolate track
            df_interp = self._interpolate_track(
                group, "validTime", ["latitude", "longitude", "maxwind"]
            )

            # Create GeoDataFrame with distance
            gdf = self._create_track_geodataframe(df_interp)
            gdf["leadtime"] = gdf["validTime"] - issue_time

            # Calculate closest approach
            closest_stats = self._get_closest_approach_stats(gdf)
            if closest_stats["closest_row"] is None:
                return None

            closest_row = closest_stats["closest_row"]

            # Calculate trigger statistics
            action_stats = self._get_wind_trigger_stats(
                gdf, "action", use_leadtime=True
            )
            readiness_stats = self._get_wind_trigger_stats(
                gdf, "readiness", use_leadtime=True
            )

            # Build result
            return {
                "monitor_id": self._create_monitor_id(
                    atcf_id, "fcast", issue_time
                ),
                "atcf_id": atcf_id,
                "name": group["name"].iloc[0],
                "issue_time": issue_time,
                "time_to_closest": closest_row["leadtime"],
                "closest_s": closest_row["maxwind"],
                "past_cutoff": closest_row["leadtime"]
                < pd.Timedelta(hours=72),  # Using 72 hours as default
                "min_dist": closest_stats["min_dist"],
                **action_stats,
                **readiness_stats,
            }
        except Exception as e:
            logger.warning(f"Error processing forecast for {atcf_id}: {e}")
            return None

    def _process_single_observation(
        self, atcf_id: str, group: pd.DataFrame, issue_time: pd.Timestamp
    ) -> Optional[Dict]:
        """Process a single observational track into monitoring record."""
        try:
            # Filter to data available at issue time
            group_recent = self._filter_by_time(
                group, "lastUpdate", issue_time
            )
            if group_recent.empty:
                return None

            # Interpolate track
            df_interp = self._interpolate_track(
                group_recent,
                "lastUpdate",
                ["latitude", "longitude", "intensity"],
            )

            # Create GeoDataFrame with distance
            gdf = self._create_track_geodataframe(df_interp)

            # Calculate closest approach
            closest_stats = self._get_closest_approach_stats(gdf)
            if closest_stats["closest_row"] is None:
                return None

            closest_row = closest_stats["closest_row"]

            # Calculate observational trigger (no leadtime filtering)
            obsv_stats = self._get_wind_trigger_stats(
                gdf, "obsv", wind_col="intensity", use_leadtime=False
            )

            # Build result
            return {
                "monitor_id": self._create_monitor_id(
                    atcf_id, "obsv", issue_time
                ),
                "atcf_id": atcf_id,
                "name": group_recent["name"].iloc[-1],
                "issue_time": issue_time,
                "min_dist": closest_stats["min_dist"],
                "closest_s": closest_row["intensity"],
                **obsv_stats,
            }
        except Exception as e:
            logger.warning(f"Error processing observation for {atcf_id}: {e}")
            return None

    def _process_single_observation_with_rainfall(
        self,
        atcf_id: str,
        group: pd.DataFrame,
        gdf_recent: gpd.GeoDataFrame,
        rain_recent: pd.DataFrame,
        issue_time: pd.Timestamp,
    ) -> Optional[Dict]:
        """Process observational track with rainfall data integration."""
        try:
            # Get storm name
            name = group[group["lastUpdate"] <= issue_time].iloc[-1]["name"]

            # Calculate closest approach statistics
            closest_stats = self._get_closest_approach_stats(gdf_recent)
            if closest_stats["closest_row"] is None:
                return None

            landfall_row = closest_stats["closest_row"]
            closest_s = landfall_row["intensity"]

            # Calculate closest approach rainfall
            landfall_start_day = landfall_row["lastUpdate"].date()
            landfall_end_day_late = landfall_start_day + pd.Timedelta(days=1)
            closest_p = self.rainfall_processor.get_rainfall_for_period(
                rain_recent,
                pd.Timestamp(landfall_start_day),
                pd.Timestamp(landfall_end_day_late),
            )

            # Calculate observational trigger based on distance threshold
            gdf_dist = gdf_recent[gdf_recent["distance"] < D_THRESH]

            if gdf_dist.empty:
                logger.info(
                    f"{atcf_id}_{issue_time.isoformat()[:10]} did not pass "
                    f"distance threshold {D_THRESH} km."
                )
                # Still return data but with empty trigger values
                max_s = 0
                max_p = 0
                obsv_trigger = False
                rainfall_relevant = False
            else:
                max_s = gdf_dist["intensity"].max()

                # Define rainfall analysis period
                start_day = pd.Timestamp(gdf_dist["lastUpdate"].min().date())
                end_day_late = pd.Timestamp(
                    gdf_dist["lastUpdate"].max().date() + pd.Timedelta(days=1)
                )

                # Check if rainfall is relevant (within storm period)
                rainfall_relevant = (
                    self.rainfall_processor.is_storm_still_active(
                        rain_recent, end_day_late
                    )
                )

                # Calculate rainfall for trigger period
                max_p = self.rainfall_processor.get_rainfall_for_period(
                    rain_recent, start_day, end_day_late
                )

                # Calculate trigger based on both wind and rainfall thresholds
                obsv_trigger = (max_p >= THRESHS["obsv"]["p"]) & (
                    max_s >= THRESHS["obsv"]["s"]
                )

            return {
                "monitor_id": self._create_monitor_id(
                    atcf_id, "obsv", issue_time
                ),
                "atcf_id": atcf_id,
                "name": name,
                "issue_time": issue_time,
                "min_dist": closest_stats["min_dist"],
                "closest_s": closest_s,
                "closest_p": closest_p,
                "obsv_s": max_s,
                "obsv_p": max_p,
                "rainfall_relevant": rainfall_relevant,
                "obsv_trigger": obsv_trigger,
            }
        except Exception as e:
            logger.warning(f"Error processing observation for {atcf_id}: {e}")
            return None

    # ============================================================================
    # PUBLIC INTERFACE - Clean, simple API
    # ============================================================================

    def process_forecast_tracks(self, clobber: bool = False) -> pd.DataFrame:
        """Process NHC forecast tracks into monitoring records."""
        logger.info("Loading NHC forecast tracks for Atlantic basin.")
        df_tracks = nhc.load_recent_glb_forecasts()
        df_tracks = df_tracks[df_tracks["basin"] == "al"]

        df_existing = self._load_existing_monitoring("fcast")
        monitoring_records = []

        for issue_time, issue_group in df_tracks.groupby("issuance"):
            for atcf_id, group in issue_group.groupby("id"):
                monitor_id = self._create_monitor_id(
                    atcf_id, "fcast", issue_time
                )

                if self._should_skip_existing(
                    monitor_id, df_existing, clobber
                ):
                    continue

                group = self._remove_track_duplicates(
                    group, "validTime", atcf_id=atcf_id
                )
                result = self._process_single_forecast(
                    atcf_id, group, issue_time
                )

                if result:
                    monitoring_records.append(result)

        return pd.DataFrame(monitoring_records)

    def process_observational_tracks(
        self, clobber: bool = False
    ) -> pd.DataFrame:
        """Process NHC observational tracks into monitoring records."""
        logger.info("Loading NHC observational tracks for Atlantic basin.")
        obsv_tracks = nhc.load_recent_glb_obsv()
        obsv_tracks = obsv_tracks[obsv_tracks["basin"] == "al"]
        obsv_tracks = obsv_tracks.rename(columns={"id": "atcf_id"})
        obsv_tracks = obsv_tracks.sort_values("lastUpdate")

        df_existing = self._load_existing_monitoring("obsv")
        monitoring_records = []

        # Check if rainfall processor is available
        if self.rainfall_processor is None:
            logger.info("No rainfall processor provided. Using wind-only.")
            # Use the original wind-only processing for each track
            for atcf_id, group in obsv_tracks.groupby("atcf_id"):
                group = self._remove_track_duplicates(group, "lastUpdate")

                # Create synthetic issue times every 6 hours during lifecycle
                if group.empty:
                    continue

                issue_times = pd.date_range(
                    start=group["lastUpdate"].min(),
                    end=group["lastUpdate"].max(),
                    freq="6H",
                )

                for issue_time in issue_times:
                    monitor_id = self._create_monitor_id(
                        atcf_id, "obsv", issue_time
                    )

                    if self._should_skip_existing(
                        monitor_id, df_existing, clobber
                    ):
                        continue

                    result = self._process_single_observation(
                        atcf_id, group, issue_time
                    )

                    if result:
                        monitoring_records.append(result)
        else:
            # Use rainfall-enhanced processing
            obsv_rain = self.rainfall_processor.load_recent_data()

            for atcf_id, group in obsv_tracks.groupby("atcf_id"):
                group = self._remove_track_duplicates(group, "lastUpdate")

                # Interpolate track data
                try:
                    df_interp = self._interpolate_track(
                        group,
                        "lastUpdate",
                        ["latitude", "longitude", "intensity", "pressure"],
                    )
                except ValueError as e:
                    logger.warning(
                        f"Skipping {atcf_id} due to interpolation error: {e}"
                    )
                    continue

                # Create GeoDataFrame with distance
                gdf = self._create_track_geodataframe(df_interp)

                # Process each rainfall issue time
                issue_times = self.rainfall_processor.get_issue_times(
                    obsv_rain
                )
                for issue_time in issue_times:
                    monitor_id = self._create_monitor_id(
                        atcf_id, "obsv", issue_time
                    )

                    if self._should_skip_existing(
                        monitor_id, df_existing, clobber
                    ):
                        continue

                    rain_recent = self.rainfall_processor.filter_data_by_time(
                        obsv_rain, issue_time
                    )
                    gdf_recent = gdf[gdf["lastUpdate"] <= issue_time]

                    if gdf_recent.empty:
                        logger.debug(
                            f"Skipping {monitor_id} as storm not active yet."
                        )
                        continue

                    # Check if storm is still active
                    track_max = gdf_recent["lastUpdate"].max()
                    if not self.rainfall_processor.is_storm_still_active(
                        rain_recent, track_max
                    ):
                        logger.debug(
                            f"Skipping {monitor_id} as storm no longer active."
                        )
                        continue

                    result = self._process_single_observation_with_rainfall(
                        atcf_id, group, gdf_recent, rain_recent, issue_time
                    )

                    if result:
                        monitoring_records.append(result)

        return pd.DataFrame(monitoring_records)

    def prepare_monitoring_data(
        self, monitoring_type: Literal["fcast", "obsv"], clobber: bool = False
    ) -> pd.DataFrame:
        """Prepare monitoring data without saving."""
        if monitoring_type == "obsv":
            df_new = self.process_observational_tracks(clobber)
            data_type = "observational"
        else:
            df_new = self.process_forecast_tracks(clobber)
            data_type = "forecast"

        if df_new.empty:
            logger.info(f"No new {data_type} data found.")
            if clobber:
                return pd.DataFrame()
            else:
                return self._load_existing_monitoring(monitoring_type)

        logger.info(f"Found {len(df_new)} new {data_type} points.")

        if clobber:
            df_combined = df_new
        else:
            df_existing = self._load_existing_monitoring(monitoring_type)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        return df_combined.sort_values(["issue_time", "atcf_id"])

    def save_monitoring_data(
        self, df: pd.DataFrame, monitoring_type: Literal["fcast", "obsv"]
    ) -> None:
        """Save monitoring data to storage."""
        if df.empty:
            logger.warning(f"No {monitoring_type} data to save.")
            return

        blob_name = (
            f"{PROJECT_PREFIX}/monitoring/"
            f"cub_{monitoring_type}_monitoring.parquet"
        )
        stratus.upload_parquet_to_blob(df, blob_name, index=False)
        logger.info(
            f"Successfully saved {len(df)} {monitoring_type} "
            f"monitoring records."
        )

    def update_monitoring(
        self, monitoring_type: Literal["fcast", "obsv"], clobber: bool = False
    ) -> pd.DataFrame:
        """Convenience method that prepares and saves data."""
        df_combined = self.prepare_monitoring_data(monitoring_type, clobber)
        self.save_monitoring_data(df_combined, monitoring_type)
        return df_combined


def create_cuba_hurricane_monitor(
    rainfall_source: Optional[str] = "imerg",
) -> CubaHurricaneMonitor:
    """Factory function to create CubaHurricaneMonitor with rainfall processor.

    Args:
        rainfall_source: Type of rainfall processor ('imerg', None for wind)

    Returns:
        Configured CubaHurricaneMonitor instance
    """
    if rainfall_source == "imerg":
        rainfall_processor = IMERGProcessor()
    elif rainfall_source is None:
        rainfall_processor = None
    else:
        raise ValueError(f"Unsupported rainfall source: {rainfall_source}")

    return CubaHurricaneMonitor(rainfall_processor=rainfall_processor)


def main():
    """Main function to run Cuba hurricane monitoring updates."""
    # Create monitor with IMERG rainfall processing
    monitor = create_cuba_hurricane_monitor(rainfall_source="imerg")

    # Update both observational and forecast monitoring
    monitor.update_monitoring("obsv", clobber=False)
    monitor.update_monitoring("fcast", clobber=False)


if __name__ == "__main__":
    main()
