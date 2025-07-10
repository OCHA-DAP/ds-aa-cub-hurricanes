"""
Unit tests for factory functions and module-level functionality.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.monitoring.monitoring_utils import (
    CubaHurricaneMonitor,
    IMERGProcessor,
    create_cuba_hurricane_monitor,
)


@pytest.mark.unit
class TestFactoryFunctions:
    """Test factory functions for creating monitors."""

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_create_cuba_hurricane_monitor_with_imerg(
        self, mock_codab, mock_codab_data
    ):
        """Test creating monitor with IMERG processor."""
        mock_codab.return_value = mock_codab_data

        monitor = create_cuba_hurricane_monitor(rainfall_source="imerg")

        assert isinstance(monitor, CubaHurricaneMonitor)
        assert isinstance(monitor.rainfall_processor, IMERGProcessor)

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_create_cuba_hurricane_monitor_without_rainfall(
        self, mock_codab, mock_codab_data
    ):
        """Test creating monitor without rainfall processor."""
        mock_codab.return_value = mock_codab_data

        monitor = create_cuba_hurricane_monitor(rainfall_source=None)

        assert isinstance(monitor, CubaHurricaneMonitor)
        assert monitor.rainfall_processor is None

    def test_create_cuba_hurricane_monitor_invalid_source(self):
        """Test creating monitor with invalid rainfall source."""
        with pytest.raises(ValueError, match="Unsupported rainfall source"):
            create_cuba_hurricane_monitor(rainfall_source="invalid")

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    def test_create_cuba_hurricane_monitor_default_imerg(
        self, mock_codab, mock_codab_data
    ):
        """Test creating monitor with default IMERG processor."""
        mock_codab.return_value = mock_codab_data

        monitor = create_cuba_hurricane_monitor()  # Default should be IMERG

        assert isinstance(monitor, CubaHurricaneMonitor)
        assert isinstance(monitor.rainfall_processor, IMERGProcessor)


@pytest.mark.integration
class TestModuleIntegration:
    """Integration tests for the monitoring module."""

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    @patch("src.monitoring.monitoring_utils.nhc.load_recent_glb_forecasts")
    @patch("src.monitoring.monitoring_utils.stratus.load_parquet_from_blob")
    def test_end_to_end_forecast_processing(
        self,
        mock_load_parquet,
        mock_load_forecasts,
        mock_codab,
        mock_codab_data,
        mock_nhc_forecast_data,
    ):
        """Test end-to-end forecast processing."""
        # Setup mocks
        mock_codab.return_value = mock_codab_data
        mock_load_forecasts.return_value = mock_nhc_forecast_data
        mock_load_parquet.side_effect = Exception("No existing data")

        # Create monitor and process data
        monitor = create_cuba_hurricane_monitor(rainfall_source=None)
        result = monitor.prepare_monitoring_data("fcast", clobber=True)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    @patch("src.monitoring.monitoring_utils.codab.load_codab_from_blob")
    @patch("src.monitoring.monitoring_utils.nhc.load_recent_glb_obsv")
    @patch("src.monitoring.monitoring_utils.stratus.load_parquet_from_blob")
    def test_end_to_end_observational_processing(
        self,
        mock_load_parquet,
        mock_load_obsv,
        mock_codab,
        mock_codab_data,
        mock_nhc_obsv_data,
    ):
        """Test end-to-end observational processing."""
        # Setup mocks
        mock_codab.return_value = mock_codab_data
        mock_load_obsv.return_value = mock_nhc_obsv_data
        mock_load_parquet.side_effect = Exception("No existing data")

        # Create monitor and process data
        monitor = create_cuba_hurricane_monitor(rainfall_source=None)
        result = monitor.prepare_monitoring_data("obsv", clobber=True)

        assert isinstance(result, pd.DataFrame)
        # May be empty if no storms pass the filtering criteria

    @patch.object(CubaHurricaneMonitor, "update_monitoring")
    @patch("src.monitoring.monitoring_utils.create_cuba_hurricane_monitor")
    def test_main_function(self, mock_create_monitor, mock_update):
        """Test the main function."""
        from src.monitoring.monitoring_utils import main

        mock_monitor = CubaHurricaneMonitor(rainfall_processor=None)
        mock_create_monitor.return_value = mock_monitor

        main()

        mock_create_monitor.assert_called_once_with(rainfall_source="imerg")
        assert mock_update.call_count == 2  # obsv and fcast
