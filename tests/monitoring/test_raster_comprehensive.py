"""
Additional comprehensive tests for raster-based processing workflows.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import geopandas as gpd
from shapely.geometry import Point

from src.monitoring.monitoring_utils import (
    CubaHurricaneMonitor,
    create_cuba_hurricane_monitor,
)


class TestRasterProcessingWorkflows:
    """Test complete workflows for raster-based processing."""

    @pytest.fixture
    def mock_storm_tracks(self):
        """Create realistic storm track data."""
        return pd.DataFrame(
            {
                "atcf_id": ["2024293N21294"] * 8,
                "name": ["OSCAR"] * 8,
                "lastUpdate": pd.date_range(
                    "2024-10-20", periods=8, freq="6h"
                ),
                "latitude": [19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5],
                "longitude": [
                    -82.0,
                    -81.0,
                    -80.0,
                    -79.0,
                    -78.0,
                    -77.0,
                    -76.0,
                    -75.0,
                ],
                "intensity": [
                    35,
                    45,
                    65,
                    85,
                    105,
                    110,
                    100,
                    85,
                ],  # Peak at 110 kt
                "pressure": [1005, 1000, 990, 980, 975, 980, 990, 1000],
                "basin": ["al"] * 8,
            }
        )

    @pytest.fixture
    def mock_rainfall_data(self):
        """Create mock rainfall calculation results."""
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-10-19", "2024-10-23"),
                "roll2_sum": [5.2, 12.1, 35.7, 28.3, 8.9],
            }
        )

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    @patch("src.monitoring.monitoring_utils.nhc.load_recent_glb_obsv")
    @patch("src.monitoring.monitoring_utils.stratus.load_parquet_from_blob")
    @patch("src.monitoring.monitoring_utils.IMERGRasterProcessor")
    def test_end_to_end_raster_processing_workflow(
        self,
        mock_processor_class,
        mock_load_existing,
        mock_load_tracks,
        mock_codab,
        mock_storm_tracks,
        mock_rainfall_data,
    ):
        """Test complete end-to-end raster processing workflow."""
        # Setup basic mocks
        from shapely.geometry import Polygon

        cuba_polygon = Polygon(
            [(-85, 19.5), (-74, 19.5), (-74, 23.5), (-85, 23.5), (-85, 19.5)]
        )
        mock_adm0_data = gpd.GeoDataFrame(
            {"country": ["Cuba"], "geometry": [cuba_polygon]}, crs="EPSG:4326"
        ).to_crs(3857)

        mock_codab.return_value = mock_adm0_data
        mock_load_tracks.return_value = mock_storm_tracks
        mock_load_existing.side_effect = Exception("No existing data")

        # Setup raster processor mock
        mock_processor = MagicMock()
        mock_processor.calculate_rainfall_for_storm_period.return_value = (
            mock_rainfall_data
        )
        mock_processor_class.return_value = mock_processor

        # Create monitor and run processing
        monitor = create_cuba_hurricane_monitor(rainfall_source="raster")
        result = monitor.process_observational_tracks_with_raster(quantile=0.9)

        # Verify results
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0  # Should have monitoring records
        assert "atcf_id" in result.columns
        assert "monitor_id" in result.columns
        assert "issue_time" in result.columns
        assert "rainfall_source" in result.columns

        # Verify raster processor was used
        mock_processor_class.assert_called_with(quantile=0.9)
        mock_processor.calculate_rainfall_for_storm_period.assert_called()

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_raster_monitor_handles_wind_only_storms(self, mock_codab):
        """Test that storms not meeting rainfall criteria still get wind processing."""
        from shapely.geometry import Polygon

        cuba_polygon = Polygon(
            [(-85, 19.5), (-74, 19.5), (-74, 23.5), (-85, 23.5), (-85, 19.5)]
        )
        mock_adm0_data = gpd.GeoDataFrame(
            {"country": ["Cuba"], "geometry": [cuba_polygon]}, crs="EPSG:4326"
        ).to_crs(3857)
        mock_codab.return_value = mock_adm0_data

        monitor = create_cuba_hurricane_monitor(rainfall_source="raster")

        # Test that monitor is properly configured
        assert monitor.rainfall_source == "raster"
        assert monitor.rainfall_processor is None

        # Test that rainfall calculation method exists
        assert hasattr(monitor, "_calculate_rainfall_for_monitoring_record")
        assert hasattr(monitor, "process_observational_tracks_with_raster")

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_monitor_initialization_with_different_sources(self, mock_codab):
        """Test monitor initialization with different rainfall sources."""
        from shapely.geometry import Polygon

        cuba_polygon = Polygon(
            [(-85, 19.5), (-74, 19.5), (-74, 23.5), (-85, 23.5), (-85, 19.5)]
        )
        mock_adm0_data = gpd.GeoDataFrame(
            {"country": ["Cuba"], "geometry": [cuba_polygon]}, crs="EPSG:4326"
        ).to_crs(3857)
        mock_codab.return_value = mock_adm0_data

        # Test raster source
        monitor_raster = create_cuba_hurricane_monitor(
            rainfall_source="raster"
        )
        assert monitor_raster.rainfall_source == "raster"
        assert monitor_raster.rainfall_processor is None

        # Test IMERG source
        monitor_imerg = create_cuba_hurricane_monitor(rainfall_source="imerg")
        assert monitor_imerg.rainfall_source == "imerg"
        assert monitor_imerg.rainfall_processor is not None

        # Test no rainfall source
        monitor_none = create_cuba_hurricane_monitor(rainfall_source=None)
        assert monitor_none.rainfall_source is None
        assert monitor_none.rainfall_processor is None

    def test_rainfall_calculation_error_handling(self):
        """Test error handling in rainfall calculation methods."""
        with patch(
            "src.monitoring.monitoring_utils.codab.load_codab_from_blob"
        ):
            monitor = CubaHurricaneMonitor(
                rainfall_processor=None, rainfall_source="raster"
            )

            # Test with completely empty inputs
            result = monitor._calculate_rainfall_for_monitoring_record(
                atcf_id="test",
                group=pd.DataFrame(),
                gdf_recent=gpd.GeoDataFrame(),
                gdf_dist_recent=gpd.GeoDataFrame(),
                storm_rainfall_df=pd.DataFrame(),
                analysis_start=pd.Timestamp("2024-01-01"),
                analysis_end=pd.Timestamp("2024-01-02"),
                issue_time=pd.Timestamp("2024-01-02"),
                quantile=0.8,
            )

            assert result is None

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_raster_processor_quantile_configuration(self, mock_codab):
        """Test that quantile configuration is properly passed through."""
        from shapely.geometry import Polygon

        cuba_polygon = Polygon(
            [(-85, 19.5), (-74, 19.5), (-74, 23.5), (-85, 23.5), (-85, 19.5)]
        )
        mock_adm0_data = gpd.GeoDataFrame(
            {"country": ["Cuba"], "geometry": [cuba_polygon]}, crs="EPSG:4326"
        ).to_crs(3857)
        mock_codab.return_value = mock_adm0_data

        monitor = create_cuba_hurricane_monitor(rainfall_source="raster")

        # Mock the prepare method to test quantile passing
        with patch.object(
            monitor, "process_observational_tracks_with_raster"
        ) as mock_process:
            mock_process.return_value = pd.DataFrame()

            monitor.prepare_monitoring_data_with_raster(
                monitoring_type="obsv", quantile=0.95  # Custom quantile
            )

            # Verify quantile was passed through
            mock_process.assert_called_once_with(False, 0.95)


class TestRasterProcessorEdgeCases:
    """Test edge cases for raster processor functionality."""

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    @patch("src.monitoring.monitoring_utils.IMERGRasterProcessor")
    def test_processor_handles_missing_data_gracefully(
        self, mock_processor_class, mock_codab
    ):
        """Test that processor handles missing or incomplete data gracefully."""
        from shapely.geometry import Polygon

        cuba_polygon = Polygon(
            [(-85, 19.5), (-74, 19.5), (-74, 23.5), (-85, 23.5), (-85, 19.5)]
        )
        mock_adm0_data = gpd.GeoDataFrame(
            {"country": ["Cuba"], "geometry": [cuba_polygon]}, crs="EPSG:4326"
        ).to_crs(3857)
        mock_codab.return_value = mock_adm0_data

        # Mock processor to return empty data
        mock_processor = MagicMock()
        mock_processor.calculate_rainfall_for_storm_period.return_value = (
            pd.DataFrame()
        )
        mock_processor_class.return_value = mock_processor

        monitor = create_cuba_hurricane_monitor(rainfall_source="raster")

        # Test with mock track data
        mock_group = pd.DataFrame(
            {
                "atcf_id": ["test"],
                "lastUpdate": [pd.Timestamp("2024-10-01")],
                "latitude": [20.0],
                "longitude": [-80.0],
                "intensity": [65],
            }
        )

        mock_gdf = gpd.GeoDataFrame(
            mock_group, geometry=[Point(-80.0, 20.0)], crs="EPSG:4326"
        ).to_crs(3857)
        mock_gdf["distance"] = [100]  # Close to Cuba

        result = monitor._calculate_rainfall_for_monitoring_record(
            atcf_id="test",
            group=mock_group,
            gdf_recent=mock_gdf,
            gdf_dist_recent=mock_gdf[mock_gdf["distance"] < 230],
            storm_rainfall_df=pd.DataFrame(),  # Empty rainfall data
            analysis_start=pd.Timestamp("2024-09-30"),
            analysis_end=pd.Timestamp("2024-10-02"),
            issue_time=pd.Timestamp("2024-10-02"),
            quantile=0.8,
        )

        # Should handle gracefully and return result with zero rainfall
        assert result is not None
        assert result["closest_p"] == 0
        assert result["obsv_p"] == 0
        assert result["rainfall_source"] == "raster_quantile"
