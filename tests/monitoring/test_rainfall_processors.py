"""
Unit tests for rainfall processors.
"""

from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from src.monitoring.monitoring_utils import IMERGProcessor


@pytest.mark.unit
class TestIMERGProcessor:
    """Test IMERG rainfall processor."""

    @patch("src.monitoring.monitoring_utils.imerg.load_imerg_recent")
    def test_load_recent_data(self, mock_load_imerg):
        """Test loading recent IMERG data."""
        # Mock raw IMERG data
        raw_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-07-01", periods=5),
                "mean": [10, 15, 20, 25, 30],
            }
        )
        mock_load_imerg.return_value = raw_data

        processor = IMERGProcessor()
        result = processor.load_recent_data()

        assert "roll2_sum" in result.columns
        assert "issue_time" in result.columns
        assert len(result) == 5
        mock_load_imerg.assert_called_once_with(recent=True)

    def test_get_issue_times(self):
        """Test getting issue times from rainfall data."""
        rain_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-07-01", periods=3),
                "issue_time": pd.date_range("2024-07-02", periods=3),
            }
        )

        processor = IMERGProcessor()
        result = processor.get_issue_times(rain_df)

        assert len(result) == 3
        assert result.equals(rain_df["issue_time"])

    def test_filter_data_by_time(self):
        """Test filtering data by time."""
        rain_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-07-01", periods=5),
                "issue_time": pd.date_range("2024-07-02", periods=5),
                "mean": [10, 15, 20, 25, 30],
            }
        )

        cutoff_time = pd.Timestamp("2024-07-04")

        processor = IMERGProcessor()
        result = processor.filter_data_by_time(rain_df, cutoff_time)

        assert len(result) == 3  # First 3 records
        assert all(result["issue_time"] <= cutoff_time)

    def test_get_rainfall_for_period(self):
        """Test getting rainfall for a specific period."""
        rain_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-07-01", periods=5),
                "roll2_sum": [20, 25, 30, 35, 40],
            }
        )

        start_date = pd.Timestamp("2024-07-02")
        end_date = pd.Timestamp("2024-07-04")

        processor = IMERGProcessor()
        result = processor.get_rainfall_for_period(
            rain_df, start_date, end_date
        )

        assert result == 35  # Max in the period

    def test_get_rainfall_for_period_empty(self):
        """Test getting rainfall when no data in period."""
        rain_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-07-01", periods=3),
                "roll2_sum": [20, 25, 30],
            }
        )

        start_date = pd.Timestamp("2024-08-01")
        end_date = pd.Timestamp("2024-08-03")

        processor = IMERGProcessor()
        result = processor.get_rainfall_for_period(
            rain_df, start_date, end_date
        )

        assert result == 0

    def test_is_storm_still_active_true(self):
        """Test storm activity check when active."""
        rain_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-07-01", periods=5),
            }
        )

        track_max_time = pd.Timestamp("2024-07-04")  # Within tolerance

        processor = IMERGProcessor()
        result = processor.is_storm_still_active(rain_df, track_max_time)

        assert result is True

    def test_is_storm_still_active_false(self):
        """Test storm activity check when inactive."""
        rain_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-07-01", periods=3),  # July 1-3
            }
        )

        track_max_time = pd.Timestamp("2024-07-01")  # Rain is recent but old

        processor = IMERGProcessor()
        result = processor.is_storm_still_active(
            rain_df, track_max_time, tolerance_days=0
        )

        assert result is False


@pytest.mark.unit
class TestRainfallProcessorIntegration:
    """Integration tests for rainfall processors with monitoring."""

    @patch("src.monitoring.monitoring_utils.imerg.load_imerg_recent")
    def test_imerg_processor_integration(
        self, mock_load_imerg, mock_codab_data
    ):
        """Test IMERG processor integration with monitor."""
        from src.monitoring.monitoring_utils import CubaHurricaneMonitor

        # Mock IMERG data
        raw_data = pd.DataFrame(
            {
                "date": pd.date_range("2024-07-01", periods=5),
                "mean": [10, 15, 20, 25, 30],
            }
        )
        mock_load_imerg.return_value = raw_data

        # Mock codab loading
        with patch(
            "src.monitoring.monitoring_utils.codab.load_codab_from_blob"
        ) as mock_codab:
            mock_codab.return_value = mock_codab_data

            processor = IMERGProcessor()
            monitor = CubaHurricaneMonitor(rainfall_processor=processor)

            assert monitor.rainfall_processor is not None
            assert hasattr(monitor.rainfall_processor, "load_recent_data")
