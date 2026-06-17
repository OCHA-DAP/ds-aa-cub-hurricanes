"""Tests for issued-time scoping of monitoring data before email sending.

`filter_to_issued_time` is what makes each pipeline run email for exactly one
issuance (the ds-storms-alerts model) instead of looping over every un-emailed
record in the season. These tests pin that contract.
"""

import os

import pandas as pd
import pytest

# src.email.utils reads SMTP settings at import time (and int-casts the port),
# so provide dummy values before importing the module under test.
os.environ.setdefault("DSCI_AWS_EMAIL_HOST", "localhost")
os.environ.setdefault("DSCI_AWS_EMAIL_PORT", "587")
os.environ.setdefault("DSCI_AWS_EMAIL_PASSWORD", "x")
os.environ.setdefault("DSCI_AWS_EMAIL_USERNAME", "x")
os.environ.setdefault("DSCI_AWS_EMAIL_ADDRESS", "x@example.com")
os.environ["FORCE_ALERT"] = "false"

from src.email import utils  # noqa: E402
from src.email.utils import filter_to_issued_time  # noqa: E402


@pytest.fixture(autouse=True)
def _no_force_alert(monkeypatch):
    """Keep FORCE_ALERT off unless a test opts in."""
    monkeypatch.setattr(utils, "FORCE_ALERT", False)
    monkeypatch.delenv("ISSUED_TIME", raising=False)


@pytest.fixture
def df_three_issuances():
    """Three forecast issuances; two storms share the latest (15:00Z)."""
    return pd.DataFrame(
        {
            "monitor_id": [
                "al012026_fcast_2026-06-16T03:00:00",
                "al012026_fcast_2026-06-16T15:00:00",
                "al022026_fcast_2026-06-16T15:00:00",
                "al012026_fcast_2026-06-15T21:00:00",
            ],
            "atcf_id": ["al012026", "al012026", "al022026", "al012026"],
            "issue_time": pd.to_datetime(
                [
                    "2026-06-16T03:00:00Z",
                    "2026-06-16T15:00:00Z",
                    "2026-06-16T15:00:00Z",
                    "2026-06-15T21:00:00Z",
                ]
            ),
            "min_dist": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_defaults_to_latest_issuance(df_three_issuances):
    """With no override, only the most recent issuance is kept — across all
    storms present at that issuance."""
    out = filter_to_issued_time(df_three_issuances)
    assert sorted(out["monitor_id"]) == [
        "al012026_fcast_2026-06-16T15:00:00",
        "al022026_fcast_2026-06-16T15:00:00",
    ]


def test_explicit_issued_time_argument(df_three_issuances):
    out = filter_to_issued_time(
        df_three_issuances, issued_time="2026-06-16T03"
    )
    assert out["monitor_id"].tolist() == ["al012026_fcast_2026-06-16T03:00:00"]


def test_issued_time_env_var(df_three_issuances, monkeypatch):
    monkeypatch.setenv("ISSUED_TIME", "2026-06-15T21")
    out = filter_to_issued_time(df_three_issuances)
    assert out["monitor_id"].tolist() == ["al012026_fcast_2026-06-15T21:00:00"]


def test_explicit_arg_overrides_env(df_three_issuances, monkeypatch):
    monkeypatch.setenv("ISSUED_TIME", "2026-06-15T21")
    out = filter_to_issued_time(
        df_three_issuances, issued_time="2026-06-16T03"
    )
    assert out["monitor_id"].tolist() == ["al012026_fcast_2026-06-16T03:00:00"]


def test_tz_naive_issue_time_treated_as_utc(df_three_issuances):
    df = df_three_issuances.copy()
    df["issue_time"] = df["issue_time"].dt.tz_localize(None)
    out = filter_to_issued_time(df)
    assert len(out) == 2


def test_empty_frame_returns_empty(df_three_issuances):
    out = filter_to_issued_time(df_three_issuances.iloc[0:0])
    assert out.empty


def test_force_alert_bypasses_scoping(df_three_issuances, monkeypatch):
    """The FORCE_ALERT test row carries the season-start issue_time, so
    scoping must be bypassed or the injected row would be dropped."""
    monkeypatch.setattr(utils, "FORCE_ALERT", True)
    out = filter_to_issued_time(df_three_issuances)
    assert len(out) == len(df_three_issuances)
