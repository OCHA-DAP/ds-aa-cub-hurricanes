"""Tests for the daily obsv issue-time grid.

obsv monitoring records (and the obsv emails they feed) are emitted once per
day at 15:00 UTC, the hour IMERG observational rainfall for the prior day
becomes available — matching the legacy IMERG cadence rather than the 6-hourly
track cadence. CubaHurricaneMonitor._daily_obsv_issue_times is the pure helper
that builds that grid; these tests pin its behaviour.
"""

import pandas as pd
import pytest

from src.monitoring.monitoring_utils import CubaHurricaneMonitor

_grid = CubaHurricaneMonitor._daily_obsv_issue_times


@pytest.mark.unit
class TestDailyObsvIssueTimes:
    def test_one_mark_per_day_at_1500z(self):
        """A multi-day storm yields one issue time per day, all at 15:00Z."""
        obs = pd.date_range(
            "2025-08-11T15:00Z", "2025-08-22T21:00Z", freq="6h"
        )
        out = _grid(pd.Series(obs))
        assert len(out) == 12  # Aug 11..22 inclusive
        assert (out.hour == 15).all()
        assert (out.minute == 0).all()

    def test_first_day_before_1500_is_kept(self):
        """First observation before 15:00Z keeps that day's mark."""
        obs = pd.date_range(
            "2025-09-01T09:00Z", "2025-09-03T03:00Z", freq="6h"
        )
        out = _grid(pd.Series(obs))
        assert [str(x) for x in out] == [
            "2025-09-01 15:00:00+00:00",
            "2025-09-02 15:00:00+00:00",
        ]

    def test_never_issues_past_latest_observation(self):
        """The grid is bounded by the latest observation (no future marks),
        so the Sep-03 15:00 mark is excluded until later obs arrive."""
        obs = pd.date_range(
            "2025-09-01T09:00Z", "2025-09-03T03:00Z", freq="6h"
        )
        out = _grid(pd.Series(obs))
        assert out.max() <= pd.Timestamp("2025-09-03T03:00Z")

    def test_single_1500_mark(self):
        obs = pd.to_datetime(["2025-10-10T03:00Z", "2025-10-10T21:00Z"])
        out = _grid(pd.Series(obs))
        assert [str(x) for x in out] == ["2025-10-10 15:00:00+00:00"]

    def test_subday_storm_after_1500_falls_back_to_last_obs(self):
        """A storm seen only within a sub-24h window that never spans a
        15:00Z mark still gets one record (at its latest observation)."""
        obs = pd.to_datetime(["2025-10-05T18:00Z", "2025-10-05T21:00Z"])
        out = _grid(pd.Series(obs))
        assert len(out) == 1
        assert str(out[0]) == "2025-10-05 21:00:00+00:00"
