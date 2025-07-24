"""
Simplified tests for the raster processor functionality.
"""

from unittest.mock import patch

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Polygon

from src.monitoring.monitoring_utils import (
    CubaHurricaneMonitor,
    IMERGRasterProcessor,
    create_cuba_hurricane_monitor,
)


@pytest.fixture
def mock_adm0_data():
    """Create mock administrative boundary data."""
    cuba_polygon = Polygon(
        [(-85, 19.5), (-74, 19.5), (-74, 23.5), (-85, 23.5), (-85, 19.5)]
    )

    return gpd.GeoDataFrame(
        {"country": ["Cuba"], "geometry": [cuba_polygon]}, crs="EPSG:4326"
    ).to_crs(3857)


class TestIMERGRasterProcessorBasics:
    """Basic tests for IMERGRasterProcessor class."""

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
    def test_get_max_rainfall_for_period_with_mock_data(
        self, mock_codab, mock_adm0_data
    ):
        """Test getting maximum rainfall with mocked calculation method."""
        mock_codab.return_value = mock_adm0_data

        processor = IMERGRasterProcessor(quantile=0.8)

        # Mock the calculation method
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


class TestRasterMonitorFactoryFunction:
    """Test the factory function for raster-based monitoring."""

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_create_monitor_with_raster_source(
        self, mock_codab, mock_adm0_data
    ):
        """Test creating monitor with raster rainfall source."""
        mock_codab.return_value = mock_adm0_data

        monitor = create_cuba_hurricane_monitor(rainfall_source="raster")

        assert isinstance(monitor, CubaHurricaneMonitor)
        assert monitor.rainfall_processor is None
        assert monitor.rainfall_source == "raster"


class TestCubaHurricaneMonitorRasterIntegration:
    """Integration tests for raster-based monitoring methods."""

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

    @patch("src.monitoring.monitoring_utils.nhc.load_recent_glb_obsv")
    @patch("src.monitoring.monitoring_utils.stratus.load_parquet_from_blob")
    def test_process_observational_tracks_with_raster_empty_tracks(
        self, mock_load_existing, mock_load_tracks, cuba_monitor_raster
    ):
        """Test processing with empty track data."""
        # Return empty DataFrame with proper columns
        empty_tracks = pd.DataFrame(
            columns=[
                "atcf_id",
                "name",
                "lastUpdate",
                "latitude",
                "longitude",
                "intensity",
                "basin",
            ]
        )
        mock_load_tracks.return_value = empty_tracks
        mock_load_existing.side_effect = Exception("No existing data")

        result = cuba_monitor_raster.process_observational_tracks_with_raster()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

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
            # Check that method was called with positional args
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
            # Check that method was called with positional args
            mock_process.assert_called_once_with(False)


class TestRainfallCalculationLogic:
    """Test the rainfall calculation logic specifically."""

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
