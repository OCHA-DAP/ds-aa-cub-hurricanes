"""
Test configuration and fixtures for monitoring tests.
"""

import pytest
import pandas as pd
import geopandas as gpd
from unittest.mock import Mock, MagicMock
from datetime import datetime, timezone
import numpy as np

from src.monitoring.monitoring_utils import (
    CubaHurricaneMonitor,
    IMERGProcessor,
)


@pytest.fixture
def mock_codab_data():
    """Mock Cuba administrative boundary data."""
    # Create a simple polygon for Cuba
    from shapely.geometry import Point, Polygon

    # Approximate Cuba bounding box
    cuba_polygon = Polygon(
        [
            (-85, 19.8),  # SW corner
            (-74, 19.8),  # SE corner
            (-74, 23.3),  # NE corner
            (-85, 23.3),  # NW corner
            (-85, 19.8),  # Close polygon
        ]
    )

    gdf = gpd.GeoDataFrame(
        {"country": ["Cuba"]}, geometry=[cuba_polygon], crs="EPSG:4326"
    ).to_crs(
        3857
    )  # Convert to projected CRS

    return gdf


@pytest.fixture
def mock_nhc_forecast_data():
    """Mock NHC forecast track data."""
    base_time = datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    data = []
    for i in range(5):  # 5 time points
        data.append(
            {
                "id": "al012024",  # forecast data uses 'id'
                "atcf_id": "al012024",  # add for consistency
                "name": "Alberto",
                "basin": "al",
                "issuance": base_time,  # forecast data uses 'issuance'
                "issue_time": base_time,  # add for consistency
                "validTime": base_time + pd.Timedelta(hours=i * 6),
                "latitude": 20.0 + i * 0.5,  # Moving north
                "longitude": -80.0 + i * 0.3,  # Moving east
                "maxwind": 65 + i * 5,  # Strengthening
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def mock_nhc_obsv_data():
    """Mock NHC observational track data."""
    base_time = datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    data = []
    for i in range(5):  # 5 time points
        data.append(
            {
                # observational data uses 'id' (gets renamed to atcf_id)
                "id": "al012024",
                "name": "Alberto",
                "basin": "al",
                "issuance": base_time,  # add for consistency
                "issue_time": base_time,  # add for consistency
                # observational uses 'lastUpdate'
                "lastUpdate": base_time + pd.Timedelta(hours=i * 6),
                "latitude": 20.0 + i * 0.5,  # Moving north
                "longitude": -80.0 + i * 0.3,  # Moving east
                "intensity": 65 + i * 5,  # Strengthening
                "pressure": 1000 - i * 5,
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def mock_imerg_data():
    """Mock IMERG rainfall data."""
    base_date = pd.Timestamp("2024-07-01")

    data = []
    for i in range(10):  # 10 days of data
        data.append(
            {
                "date": base_date + pd.Timedelta(days=i),
                "mean": 15
                + np.random.normal(0, 5),  # Random rainfall around 15mm
            }
        )

    df = pd.DataFrame(data)
    df["roll2_sum"] = (
        df["mean"].rolling(window=2, center=True, min_periods=1).sum()
    )
    df["issue_time"] = df["date"].apply(
        lambda x: x.tz_localize("UTC")
    ) + pd.Timedelta(hours=15, days=1)

    return df


@pytest.fixture
def mock_rainfall_processor(mock_imerg_data):
    """Mock rainfall processor with test data."""
    processor = Mock(spec=IMERGProcessor)
    processor.load_recent_data.return_value = mock_imerg_data
    processor.get_issue_times.return_value = mock_imerg_data["issue_time"]

    def filter_by_time(rain_df, issue_time):
        return rain_df[rain_df["issue_time"] <= issue_time]

    def get_rainfall_for_period(
        rain_df, start_date, end_date, rain_col="roll2_sum"
    ):
        filtered = rain_df[
            (rain_df["date"] >= start_date) & (rain_df["date"] <= end_date)
        ]
        return filtered[rain_col].max() if not filtered.empty else 0

    def is_storm_still_active(rain_df, track_max_time, tolerance_days=1):
        rain_max = rain_df["date"].max().date()
        track_max = track_max_time.date()
        return rain_max - track_max <= pd.Timedelta(days=tolerance_days)

    processor.filter_data_by_time.side_effect = filter_by_time
    processor.get_rainfall_for_period.side_effect = get_rainfall_for_period
    processor.is_storm_still_active.side_effect = is_storm_still_active

    return processor


@pytest.fixture
def cuba_monitor_no_rainfall(mock_codab_data, monkeypatch):
    """Cuba hurricane monitor without rainfall processor."""
    # Mock the codab loading
    monkeypatch.setattr(
        "src.monitoring.monitoring_utils.codab.load_codab_from_blob",
        lambda: mock_codab_data,
    )

    return CubaHurricaneMonitor(rainfall_processor=None)


@pytest.fixture
def cuba_monitor_with_rainfall(
    mock_codab_data, mock_rainfall_processor, monkeypatch
):
    """Cuba hurricane monitor with mocked rainfall processor."""
    # Mock the codab loading
    monkeypatch.setattr(
        "src.monitoring.monitoring_utils.codab.load_codab_from_blob",
        lambda: mock_codab_data,
    )

    return CubaHurricaneMonitor(rainfall_processor=mock_rainfall_processor)


@pytest.fixture
def empty_existing_data():
    """Empty existing monitoring data."""
    return pd.DataFrame()


@pytest.fixture
def sample_existing_data():
    """Sample existing monitoring data."""
    return pd.DataFrame(
        {
            "monitor_id": ["al012024_fcast_2024-07-01T12:00:00"],
            "atcf_id": ["al012024"],
            "name": ["Alberto"],
            "issue_time": [
                datetime(2024, 7, 1, 12, 0, 0, tzinfo=timezone.utc)
            ],
            "min_dist": [150.0],
        }
    )


# Markers for different test types
pytestmark = [
    pytest.mark.unit,  # Mark all tests in this module as unit tests
]
