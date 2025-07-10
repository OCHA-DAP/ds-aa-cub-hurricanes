"""
Integration tests for the monitoring system.
These tests require external dependencies and data sources.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from src.monitoring.monitoring_utils import create_cuba_hurricane_monitor


@pytest.mark.integration
class TestMonitoringIntegration:
    """Integration tests for the full monitoring workflow."""

    @patch("src.monitoring.monitoring_utils.stratus.upload_parquet_to_blob")
    @patch("src.monitoring.monitoring_utils.stratus.load_parquet_from_blob")
    @patch("src.monitoring.monitoring_utils.nhc.load_recent_glb_forecasts")
    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_full_forecast_workflow(
        self,
        mock_codab,
        mock_nhc,
        mock_load_blob,
        mock_upload_blob,
        mock_codab_data,
        mock_nhc_forecast_data,
    ):
        """Test the complete forecast monitoring workflow."""
        # Setup mocks
        mock_codab.return_value = mock_codab_data
        mock_nhc.return_value = mock_nhc_forecast_data
        mock_load_blob.side_effect = Exception("No existing data")
        mock_upload_blob.return_value = None

        # Create monitor and run full workflow
        monitor = create_cuba_hurricane_monitor(rainfall_source=None)

        # Test the complete update workflow
        result = monitor.update_monitoring("fcast", clobber=True)

        # Verify the workflow completed
        assert isinstance(result, pd.DataFrame)
        mock_codab.assert_called_once()
        mock_nhc.assert_called_once()
        mock_upload_blob.assert_called_once()

    @pytest.mark.slow
    @patch("src.monitoring.monitoring_utils.stratus.upload_parquet_to_blob")
    @patch("src.monitoring.monitoring_utils.stratus.load_parquet_from_blob")
    @patch("src.monitoring.monitoring_utils.nhc.load_recent_glb_obsv")
    @patch("src.monitoring.monitoring_utils.imerg.load_imerg_recent")
    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_full_observational_workflow_with_rainfall(
        self,
        mock_codab,
        mock_imerg,
        mock_nhc,
        mock_load_blob,
        mock_upload_blob,
        mock_codab_data,
        mock_nhc_obsv_data,
        mock_imerg_data,
    ):
        """Test the complete observational workflow with rainfall."""
        # Setup mocks
        mock_codab.return_value = mock_codab_data
        mock_nhc.return_value = mock_nhc_obsv_data
        mock_imerg.return_value = mock_imerg_data
        mock_load_blob.side_effect = Exception("No existing data")
        mock_upload_blob.return_value = None

        # Create monitor with rainfall processing
        monitor = create_cuba_hurricane_monitor(rainfall_source="imerg")

        # Test the complete update workflow
        result = monitor.update_monitoring("obsv", clobber=True)

        # Verify the workflow completed
        assert isinstance(result, pd.DataFrame)
        mock_codab.assert_called_once()
        mock_nhc.assert_called_once()
        mock_imerg.assert_called_once()
        mock_upload_blob.assert_called_once()

    @pytest.mark.integration
    def test_error_handling_invalid_data(self):
        """Test error handling with invalid input data."""
        # Test with invalid rainfall source
        with pytest.raises(ValueError):
            create_cuba_hurricane_monitor(rainfall_source="invalid_source")

    @pytest.mark.integration
    @patch("src.monitoring.monitoring_utils.nhc.load_recent_glb_forecasts")
    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_empty_data_handling(self, mock_codab, mock_nhc, mock_codab_data):
        """Test handling of empty data scenarios."""
        # Setup mocks with empty data that has correct structure
        mock_codab.return_value = mock_codab_data
        # Empty DataFrame with correct columns
        empty_forecast_data = pd.DataFrame(
            columns=[
                "id",
                "atcf_id",
                "name",
                "basin",
                "issuance",
                "issue_time",
                "validTime",
                "latitude",
                "longitude",
                "maxwind",
            ]
        )
        mock_nhc.return_value = empty_forecast_data

        monitor = create_cuba_hurricane_monitor(rainfall_source=None)
        result = monitor.prepare_monitoring_data("fcast", clobber=True)

        # Should handle empty data gracefully
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @pytest.mark.integration
    def test_configuration_validation(self):
        """Test that the monitoring system validates its configuration."""
        # Test default configuration
        monitor = create_cuba_hurricane_monitor()
        assert monitor.rainfall_processor is not None

        # Test without rainfall
        monitor_no_rain = create_cuba_hurricane_monitor(rainfall_source=None)
        assert monitor_no_rain.rainfall_processor is None

        # Verify both have proper Cuba boundaries loaded
        assert hasattr(monitor, "adm0")
        assert hasattr(monitor_no_rain, "adm0")


@pytest.mark.integration
class TestDataValidation:
    """Integration tests for data validation and consistency."""

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    @patch("src.monitoring.monitoring_utils.nhc.load_recent_glb_forecasts")
    def test_forecast_data_validation(
        self, mock_nhc, mock_codab, mock_codab_data, mock_nhc_forecast_data
    ):
        """Test validation of forecast data processing."""
        mock_codab.return_value = mock_codab_data
        mock_nhc.return_value = mock_nhc_forecast_data

        monitor = create_cuba_hurricane_monitor(rainfall_source=None)
        result = monitor.process_forecast_tracks(clobber=True)

        if not result.empty:
            # Validate expected columns exist
            expected_cols = [
                "monitor_id",
                "atcf_id",
                "name",
                "issue_time",
                "min_dist",
                "action_s",
                "action_trigger",
            ]
            for col in expected_cols:
                assert col in result.columns, f"Missing column: {col}"

            # Validate data types
            assert result["min_dist"].dtype in ["float64", "int64"]
            assert result["action_trigger"].dtype == "bool"

    @pytest.mark.slow
    def test_performance_benchmarks(self):
        """Test performance benchmarks for large datasets."""
        import time

        # This would be a placeholder for performance testing
        # In a real scenario, you'd test with larger datasets
        start_time = time.time()

        # Simulate some processing
        monitor = create_cuba_hurricane_monitor(rainfall_source=None)

        end_time = time.time()
        processing_time = end_time - start_time

        # Assert reasonable performance (adjust threshold as needed)
        assert (
            processing_time < 5.0
        ), f"Initialization took too long: {processing_time}s"
