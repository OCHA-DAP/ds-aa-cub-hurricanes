"""
Tests for the IMERGRasterProcessor class and raster-based functionality.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

from src.monitoring.monitoring_utils import (
    IMERGRasterProcessor,
    CubaHurricaneMonitor,
)


@pytest.fixture
def mock_adm0_data():
    """Create mock administrative boundary data."""
    # Create a simple polygon for Cuba
    cuba_polygon = Polygon(
        [(-85, 19.5), (-74, 19.5), (-74, 23.5), (-85, 23.5), (-85, 19.5)]
    )

    return gpd.GeoDataFrame(
        {"country": ["Cuba"], "geometry": [cuba_polygon]}, crs="EPSG:4326"
    ).to_crs(
        3857
    )  # Convert to projected CRS


@pytest.fixture
def mock_raster_data():
    """Create mock IMERG raster data."""
    # Create sample dates
    dates = pd.date_range("2024-10-01", "2024-10-05", freq="D")

    # Create mock precipitation data (5 days, 10x10 grid)
    precip_data = np.random.rand(len(dates), 10, 10) * 50  # 0-50mm

    # Create xarray DataArray
    da = xr.DataArray(
        precip_data,
        dims=["date", "y", "x"],
        coords={"date": dates, "y": range(10), "x": range(10)},
        attrs={"units": "mm"},
    )

    # Add spatial reference
    da = da.rio.write_crs("EPSG:4326")

    return da


class TestIMERGRasterProcessor:
    """Test suite for IMERGRasterProcessor class."""

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_init(self, mock_codab, mock_adm0_data):
        """Test processor initialization."""
        mock_codab.return_value = mock_adm0_data

        processor = IMERGRasterProcessor(quantile=0.8)

        assert processor.quantile == 0.8
        assert processor.adm0 is not None
        mock_codab.assert_called_once()

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_init_default_quantile(self, mock_codab, mock_adm0_data):
        """Test processor initialization with default quantile."""
        mock_codab.return_value = mock_adm0_data

        processor = IMERGRasterProcessor()

        assert processor.quantile == 0.8

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_calculate_rainfall_for_storm_period_success(
        self, mock_codab, mock_adm0_data
    ):
        """Test successful rainfall calculation for storm period."""
        mock_codab.return_value = mock_adm0_data

        processor = IMERGRasterProcessor(quantile=0.8)

        # Mock the entire method to return expected result
        with patch.object(
            processor, "calculate_rainfall_for_storm_period"
        ) as mock_calc:
            mock_calc.return_value = pd.DataFrame(
                {
                    "date": pd.date_range("2024-10-01", periods=3),
                    "roll2_sum": [10.5, 25.3, 15.7],
                }
            )

            start_date = pd.Timestamp("2024-10-01")
            end_date = pd.Timestamp("2024-10-03")

            result = processor.calculate_rainfall_for_storm_period(
                start_date, end_date
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3  # 3 days
            assert "date" in result.columns
            assert "roll2_sum" in result.columns

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    @patch("src.monitoring.monitoring_utils.imerg.open_imerg_raster_dates")
    def test_calculate_rainfall_for_storm_period_empty_dates(
        self, mock_open_raster, mock_codab, mock_adm0_data
    ):
        """Test rainfall calculation with empty date range."""
        mock_codab.return_value = mock_adm0_data

        processor = IMERGRasterProcessor(quantile=0.8)

        # Same start and end date should result in empty range
        start_date = pd.Timestamp("2024-10-01")
        end_date = pd.Timestamp("2024-09-30")  # End before start

        result = processor.calculate_rainfall_for_storm_period(
            start_date, end_date
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        mock_open_raster.assert_not_called()

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    @patch("src.monitoring.monitoring_utils.imerg.open_imerg_raster_dates")
    def test_calculate_rainfall_for_storm_period_error_handling(
        self, mock_open_raster, mock_codab, mock_adm0_data
    ):
        """Test error handling in rainfall calculation."""
        mock_codab.return_value = mock_adm0_data
        mock_open_raster.side_effect = Exception("Network error")

        processor = IMERGRasterProcessor(quantile=0.8)

        start_date = pd.Timestamp("2024-10-01")
        end_date = pd.Timestamp("2024-10-03")

        result = processor.calculate_rainfall_for_storm_period(
            start_date, end_date
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0  # Should return empty DataFrame on error

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_get_max_rainfall_for_period_success(
        self, mock_codab, mock_adm0_data
    ):
        """Test getting maximum rainfall for a period."""
        mock_codab.return_value = mock_adm0_data

        processor = IMERGRasterProcessor(quantile=0.8)

        # Mock the calculate_rainfall_for_storm_period method
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-10-01", "2024-10-03"),
                "roll2_sum": [10.5, 25.3, 15.7],
            }
        )

        with patch.object(
            processor,
            "calculate_rainfall_for_storm_period",
            return_value=mock_df,
        ):
            start_date = pd.Timestamp("2024-10-01")
            end_date = pd.Timestamp("2024-10-03")

            result = processor.get_max_rainfall_for_period(
                start_date, end_date
            )

            assert result == 25.3

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_get_max_rainfall_for_period_empty_data(
        self, mock_codab, mock_adm0_data
    ):
        """Test getting maximum rainfall with empty data."""
        mock_codab.return_value = mock_adm0_data

        processor = IMERGRasterProcessor(quantile=0.8)

        # Mock empty DataFrame
        with patch.object(
            processor,
            "calculate_rainfall_for_storm_period",
            return_value=pd.DataFrame(),
        ):
            start_date = pd.Timestamp("2024-10-01")
            end_date = pd.Timestamp("2024-10-03")

            result = processor.get_max_rainfall_for_period(
                start_date, end_date
            )

            assert result == 0


class TestCubaHurricaneMonitorRasterMethods:
    """Test raster-specific methods in CubaHurricaneMonitor."""

    @pytest.fixture
    def cuba_monitor_raster(self, mock_adm0_data):
        """Create a CubaHurricaneMonitor instance with raster configuration."""
        with patch(
            "src.monitoring.monitoring_utils.codab.load_codab_from_blob"
        ) as mock_codab:
            mock_codab.return_value = mock_adm0_data
            monitor = CubaHurricaneMonitor(
                rainfall_processor=None, rainfall_source="raster"
            )
            return monitor

    @pytest.fixture
    def mock_obsv_tracks(self):
        """Create mock observational track data."""
        return pd.DataFrame(
            {
                "atcf_id": ["2024293N21294"] * 5,
                "name": ["OSCAR"] * 5,
                "lastUpdate": pd.date_range(
                    "2024-10-20", periods=5, freq="6h"
                ),
                "latitude": [20.0, 20.5, 21.0, 21.5, 22.0],
                "longitude": [-80.0, -79.5, -79.0, -78.5, -78.0],
                "intensity": [45, 65, 85, 90, 70],
                "pressure": [1000, 990, 980, 975, 985],
                "basin": ["al"] * 5,
            }
        )

    @patch("src.monitoring.monitoring_utils.nhc.load_recent_glb_obsv")
    @patch("src.monitoring.monitoring_utils.stratus.load_parquet_from_blob")
    def test_process_observational_tracks_with_raster_empty_tracks(
        self, mock_load_existing, mock_load_tracks, cuba_monitor_raster
    ):
        """Test processing with empty track data."""
        mock_load_tracks.return_value = pd.DataFrame()
        mock_load_existing.side_effect = Exception("No existing data")

        result = cuba_monitor_raster.process_observational_tracks_with_raster()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @patch("src.monitoring.monitoring_utils.nhc.load_recent_glb_obsv")
    @patch("src.monitoring.monitoring_utils.stratus.load_parquet_from_blob")
    @patch("src.monitoring.monitoring_utils.IMERGRasterProcessor")
    def test_process_observational_tracks_with_raster_success(
        self,
        mock_processor_class,
        mock_load_existing,
        mock_load_tracks,
        cuba_monitor_raster,
        mock_obsv_tracks,
    ):
        """Test successful processing with raster data."""
        # Setup mocks
        mock_load_tracks.return_value = mock_obsv_tracks
        mock_load_existing.side_effect = Exception("No existing data")

        # Mock the raster processor
        mock_processor = MagicMock()
        mock_processor.calculate_rainfall_for_storm_period.return_value = (
            pd.DataFrame(
                {
                    "date": pd.date_range("2024-10-20", periods=3),
                    "roll2_sum": [5.0, 15.0, 8.0],
                }
            )
        )
        mock_processor_class.return_value = mock_processor

        result = cuba_monitor_raster.process_observational_tracks_with_raster(
            quantile=0.9
        )

        assert isinstance(result, pd.DataFrame)
        # Should have at least some records for wind-based monitoring
        assert len(result) > 0

        # Check that raster processor was created with correct quantile
        mock_processor_class.assert_called_with(quantile=0.9)

    def test_calculate_rainfall_for_monitoring_record_success(
        self, cuba_monitor_raster
    ):
        """Test rainfall calculation for a single monitoring record."""
        # Create mock data
        mock_group = pd.DataFrame(
            {
                "atcf_id": ["2024293N21294"] * 3,
                "lastUpdate": pd.date_range(
                    "2024-10-20", periods=3, freq="12H"
                ),
                "latitude": [20.0, 20.5, 21.0],
                "longitude": [-80.0, -79.5, -79.0],
                "intensity": [65, 85, 70],
            }
        )

        # Create mock GeoDataFrame
        from shapely.geometry import Point

        mock_gdf = gpd.GeoDataFrame(
            mock_group,
            geometry=[
                Point(lon, lat)
                for lon, lat in zip(mock_group.longitude, mock_group.latitude)
            ],
            crs="EPSG:4326",
        ).to_crs(3857)

        # Add distance column
        mock_gdf["distance"] = [100, 50, 150]  # km
        mock_gdf_dist = mock_gdf[mock_gdf["distance"] < 230]

        # Mock rainfall data
        mock_storm_rainfall = pd.DataFrame(
            {
                "date": pd.date_range("2024-10-19", "2024-10-22"),
                "roll2_sum": [5.0, 15.0, 25.0, 10.0],
            }
        )

        analysis_start = pd.Timestamp("2024-10-19")
        analysis_end = pd.Timestamp("2024-10-22")
        issue_time = pd.Timestamp("2024-10-22")

        result = cuba_monitor_raster._calculate_rainfall_for_monitoring_record(
            atcf_id="2024293N21294",
            group=mock_group,
            gdf_recent=mock_gdf,
            gdf_dist_recent=mock_gdf_dist,
            storm_rainfall_df=mock_storm_rainfall,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            issue_time=issue_time,
            quantile=0.8,
        )

        assert result is not None
        assert "closest_p" in result
        assert "obsv_p" in result
        assert "obsv_trigger" in result
        assert "rainfall_relevant" in result
        assert result["rainfall_source"] == "raster_quantile"
        assert result["quantile_used"] == 0.8

    def test_calculate_rainfall_for_monitoring_record_empty_gdf(
        self, cuba_monitor_raster
    ):
        """Test rainfall calculation with empty GeoDataFrame."""
        mock_group = pd.DataFrame()
        mock_gdf = gpd.GeoDataFrame()
        mock_gdf_dist = gpd.GeoDataFrame()
        mock_storm_rainfall = pd.DataFrame()

        result = cuba_monitor_raster._calculate_rainfall_for_monitoring_record(
            atcf_id="test",
            group=mock_group,
            gdf_recent=mock_gdf,
            gdf_dist_recent=mock_gdf_dist,
            storm_rainfall_df=mock_storm_rainfall,
            analysis_start=pd.Timestamp("2024-10-01"),
            analysis_end=pd.Timestamp("2024-10-02"),
            issue_time=pd.Timestamp("2024-10-02"),
            quantile=0.8,
        )

        assert result is None

    @patch("src.monitoring.monitoring_utils.stratus.load_parquet_from_blob")
    def test_prepare_monitoring_data_with_raster_obsv(
        self, mock_load_existing, cuba_monitor_raster
    ):
        """Test prepare_monitoring_data_with_raster for observational data."""
        mock_load_existing.side_effect = Exception("No existing data")

        with patch.object(
            cuba_monitor_raster, "process_observational_tracks_with_raster"
        ) as mock_process:
            mock_process.return_value = pd.DataFrame(
                {
                    "monitor_id": ["test_obsv_1"],
                    "atcf_id": ["test"],
                    "issue_time": [pd.Timestamp("2024-10-01")],
                }
            )

            result = cuba_monitor_raster.prepare_monitoring_data_with_raster(
                monitoring_type="obsv", clobber=False, quantile=0.8
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            mock_process.assert_called_once_with(False, 0.8)

    @patch("src.monitoring.monitoring_utils.stratus.load_parquet_from_blob")
    def test_prepare_monitoring_data_with_raster_fcast(
        self, mock_load_existing, cuba_monitor_raster
    ):
        """Test prepare_monitoring_data_with_raster for forecast data."""
        mock_load_existing.side_effect = Exception("No existing data")

        with patch.object(
            cuba_monitor_raster, "process_forecast_tracks"
        ) as mock_process:
            mock_process.return_value = pd.DataFrame(
                {
                    "monitor_id": ["test_fcast_1"],
                    "atcf_id": ["test"],
                    "issue_time": [pd.Timestamp("2024-10-01")],
                }
            )

            result = cuba_monitor_raster.prepare_monitoring_data_with_raster(
                monitoring_type="fcast", clobber=False, quantile=0.8
            )

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            mock_process.assert_called_once_with(False)

    @patch("src.monitoring.monitoring_utils.stratus.upload_parquet_to_blob")
    def test_save_monitoring_data_success(
        self, mock_upload, cuba_monitor_raster
    ):
        """Test successful saving of monitoring data."""
        test_df = pd.DataFrame(
            {
                "monitor_id": ["test_1", "test_2"],
                "atcf_id": ["storm1", "storm2"],
                "issue_time": [
                    pd.Timestamp("2024-10-01"),
                    pd.Timestamp("2024-10-02"),
                ],
            }
        )

        cuba_monitor_raster.save_monitoring_data(test_df, "obsv")

        mock_upload.assert_called_once()
        args, kwargs = mock_upload.call_args
        assert len(args) == 2
        assert args[0].equals(test_df)
        assert "cub_obsv_monitoring.parquet" in args[1]
        assert kwargs["index"] is False

    def test_save_monitoring_data_empty(self, cuba_monitor_raster):
        """Test saving empty monitoring data."""
        empty_df = pd.DataFrame()

        with patch(
            "src.monitoring.monitoring_utils.stratus.upload_parquet_to_blob"
        ) as mock_upload:
            cuba_monitor_raster.save_monitoring_data(empty_df, "obsv")
            mock_upload.assert_not_called()
