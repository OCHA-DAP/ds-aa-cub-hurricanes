import base64
import os
import re
from pathlib import Path
from typing import Literal, Optional
import pandas as pd
import ocha_stratus as stratus
from src.datasources import nhc
from datetime import datetime, timezone

from src.constants import (
    PROJECT_PREFIX,
    DRY_RUN,
    TEST_EMAIL,
    FORCE_ALERT,
    MONITORING_START_DATE,
    TEST_ATCF_ID,
    TEST_MONITOR_ID,
    TEST_FCAST_MONITOR_ID,
    TEST_OBSV_MONITOR_ID,
    TEST_STORM_NAME,
    _parse_bool_env,
)

EMAIL_HOST = os.getenv("DSCI_AWS_EMAIL_HOST")
EMAIL_PORT = int(os.getenv("DSCI_AWS_EMAIL_PORT"))
EMAIL_PASSWORD = os.getenv("DSCI_AWS_EMAIL_PASSWORD")
EMAIL_USERNAME = os.getenv("DSCI_AWS_EMAIL_USERNAME")
EMAIL_ADDRESS = os.getenv("DSCI_AWS_EMAIL_ADDRESS")

# Legacy flags - will be deprecated
TEST_LIST = _parse_bool_env("TEST_LIST", default=False)
TEST_STORM = _parse_bool_env("TEST_STORM", default=False)
EMAIL_DISCLAIMER = _parse_bool_env("EMAIL_DISCLAIMER", default=False)


TEMPLATES_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "templates"
STATIC_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "static"


def create_dummy_storm_tracks(
    df_tracks: pd.DataFrame, fcast_obsv: str
) -> pd.DataFrame:
    """Create dummy storm tracks based on Hurricane Rafael data but with test IDs.

    Args:
        fcast_obsv: Whether to create forecast or observation tracks

    Returns:
        DataFrame with tracks data modified to match dummy storm monitoring data
    """
    # Use Hurricane Rafael as the base data
    dummy_id = "al182024"
    dummy_name = "Rafael"
    # al182024_fcast_2024-11-04T21:00:00

    if fcast_obsv == "obsv":
        from src.constants import THRESHS

        dummy_track = df_tracks[
            (df_tracks["id"] == dummy_id) & (df_tracks["name"] == dummy_name)
        ].copy()

        # Sort by time to ensure proper chronological order
        dummy_track = dummy_track.sort_values("lastUpdate")

        # Replace 100-knot values with 105 knots to create threshold crossing
        dummy_track.loc[dummy_track["intensity"] == 100, "intensity"] = 105

        # Find first point where wind crosses observation threshold (105 knots)
        obs_threshold = THRESHS["obsv"]["s"]  # 105 knots
        threshold_crossed_idx = dummy_track[
            dummy_track["intensity"] >= obs_threshold
        ].index

        if len(threshold_crossed_idx) > 0:
            # Include only points up to first threshold crossing
            first_crossing_idx = threshold_crossed_idx[0]
            dummy_track = dummy_track.loc[:first_crossing_idx]

        # Shift timestamps to start from MONITORING_START_DATE
        min_time = min(dummy_track["lastUpdate"])
        diff_from_min = dummy_track["lastUpdate"] - min_time
        dummy_track["lastUpdate"] = MONITORING_START_DATE + diff_from_min

    if fcast_obsv == "fcast":

        target_track_time = datetime(
            2024, 11, 4, 21, 0, 0, tzinfo=timezone.utc
        )
        dummy_track = df_tracks[
            (df_tracks["id"] == dummy_id)
            & (df_tracks["name"] == dummy_name)
            & (df_tracks["issuance"] == target_track_time)
        ].copy()
        # Calculate lead time between issuance and validTime
        lt = dummy_track["validTime"] - dummy_track["issuance"]
        dummy_track["issuance"] = MONITORING_START_DATE
        dummy_track["validTime"] = dummy_track["issuance"] + lt

        # Inject maxwind value of 125 where lead time = 2 days (action)
        dummy_track.loc[lt == pd.Timedelta(days=2, hours=9), "maxwind"] = 125
        # Inject readiness activation
        dummy_track.loc[lt == pd.Timedelta(days=4, hours=21), "maxwind"] = 125

    dummy_track["id"] = TEST_ATCF_ID
    return dummy_track


def create_dummy_storm_monitoring(fcast_obsv: str) -> pd.DataFrame:
    DUMMY_MONITOR_ID = (
        TEST_FCAST_MONITOR_ID
        if fcast_obsv == "fcast"
        else TEST_OBSV_MONITOR_ID
    )

    if fcast_obsv == "fcast":
        df = pd.DataFrame(
            [
                {
                    "monitor_id": DUMMY_MONITOR_ID,
                    "atcf_id": TEST_ATCF_ID,
                    "name": TEST_STORM_NAME,
                    "issue_time": MONITORING_START_DATE,
                    "time_to_closest": None,
                    "closest_s": 83.33,
                    "past_cutoff": False,
                    "min_dist": 83.0,
                    "action_s": 125,
                    "action_trigger": True,
                    "readiness_s": 125,
                    "readiness_trigger": True,
                }
            ]
        )
    else:
        # For observation data, set issue time to when threshold was crossed
        # This matches the realistic operational scenario

        # Calculate when the threshold would be crossed based on dummy data
        # In the dummy Hurricane Rafael data, threshold crossing happens
        # around day 3 of the storm track (specifically at 09:00 on day 4)
        threshold_crossing_date = MONITORING_START_DATE + pd.Timedelta(
            days=3, hours=12
        )

        df = pd.DataFrame(
            [
                {
                    "monitor_id": DUMMY_MONITOR_ID,
                    "atcf_id": TEST_ATCF_ID,
                    "name": TEST_STORM_NAME,
                    "issue_time": threshold_crossing_date,
                    "min_dist": 0.0,
                    "closest_s": 125,
                    "obsv_s": 110,
                    "obsv_trigger": True,
                    "closest_p": 100,
                    "obsv_p": 100,
                    "rainfall_relevant": True,
                    "rainfall_source": "raster_quantile",
                    "quantile_used": 0.8,
                    "analysis_start": None,
                    "analysis_end": None,
                }
            ]
        )
    return df


def add_test_row_to_monitoring(
    df_monitoring: pd.DataFrame, fcast_obsv: str
) -> pd.DataFrame:
    """Add test row to monitoring df to simulate new monitoring point.
    This new monitoring point will cause an activation of all triggers.
    Uses create_dummy_storm_monitoring to generate test data.
    """
    # Only print this once per fcast/obsv type per process
    if not hasattr(add_test_row_to_monitoring, f"_added_{fcast_obsv}"):
        if fcast_obsv == "fcast":
            print("🧪 Adding test forecast row for FORCE_ALERT testing")
        else:
            print("🧪 Adding test observation row for FORCE_ALERT testing")
        setattr(add_test_row_to_monitoring, f"_added_{fcast_obsv}", True)

    # Create dummy storm monitoring data
    df_monitoring_test = create_dummy_storm_monitoring(fcast_obsv)

    df_monitoring = pd.concat(
        [df_monitoring, df_monitoring_test], ignore_index=True
    )
    return df_monitoring


def open_static_image(filename: str) -> str:
    filepath = STATIC_DIR / filename
    with open(filepath, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image


def get_distribution_list() -> pd.DataFrame:
    """Load distribution list from blob storage."""
    if TEST_EMAIL:  # Use new flag instead of TEST_LIST
        blob_name = f"{PROJECT_PREFIX}/email/test_distribution_list.csv"
    else:
        blob_name = f"{PROJECT_PREFIX}/email/distribution_list.csv"
    return stratus.load_csv_from_blob(blob_name)


def load_email_record() -> pd.DataFrame:
    """Load record of emails that have been sent.

    Uses test_email_record.csv if TEST_EMAIL=True, otherwise email_record.csv.
    Returns empty DataFrame with correct columns if file doesn't exist.
    """
    if TEST_EMAIL:
        blob_name = f"{PROJECT_PREFIX}/email/test_email_record.csv"
    else:
        blob_name = f"{PROJECT_PREFIX}/email/email_record.csv"

    try:
        return stratus.load_csv_from_blob(blob_name)
    except Exception as e:
        if "BlobNotFound" in str(e) or "does not exist" in str(e):
            print(f"📝 Email record file not found: {blob_name}")
            print("📝 Creating empty email record with correct structure")
            # Return empty DataFrame with correct column structure
            return pd.DataFrame(
                columns=["monitor_id", "atcf_id", "email_type"]
            )
        else:
            # Re-raise other exceptions
            raise e


def load_monitoring_data(fcast_obsv: Literal["fcast", "obsv"]) -> pd.DataFrame:
    """Load monitoring data with optional test row injection.

    Args:
        fcast_obsv: Whether to load forecast or observation data

    Returns:
        DataFrame with monitoring data, optionally including test rows
    """
    from src.monitoring import monitoring_utils

    monitor = monitoring_utils.create_cuba_hurricane_monitor()
    df_monitoring = monitor._load_existing_monitoring(fcast_obsv)

    # Add test data if FORCE_ALERT is enabled - do this BEFORE filtering
    if FORCE_ALERT:
        df_monitoring = add_test_row_to_monitoring(df_monitoring, fcast_obsv)

    # No data yet (e.g. start of a season): nothing to filter. Return the
    # empty, schema-carrying frame so callers can filter/iterate safely.
    if df_monitoring.empty:
        return df_monitoring

    # Filter by MONITORING_START_DATE to limit processing scope and prevent timeouts
    df_monitoring["issue_time"] = pd.to_datetime(df_monitoring["issue_time"])
    df_monitoring = df_monitoring[
        df_monitoring["issue_time"] >= MONITORING_START_DATE
    ]
    print(
        f"📅 Filtered {fcast_obsv} monitoring data from "
        f"{MONITORING_START_DATE.strftime('%Y-%m-%d')}: "
        f"{len(df_monitoring)} records remaining"
    )

    return df_monitoring


def _issue_time_hours_utc(series: pd.Series) -> pd.Series:
    """issue_time values floored to the UTC hour.

    NHC advisories land on whole hours (03/09/15/21Z), so flooring lets us
    match records to a target issuance regardless of tz representation or any
    sub-hour drift in the stored timestamp.
    """
    series = pd.to_datetime(series)
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    else:
        series = series.dt.tz_convert("UTC")
    return series.dt.floor("h")


def filter_to_issued_time(
    df_monitoring: pd.DataFrame, issued_time: Optional[str] = None
) -> pd.DataFrame:
    """Scope monitoring records to a single issuance ("issued time").

    Each pipeline run emails for exactly ONE issued time, mirroring the
    once-per-issuance model of the ds-storms-alerts pipeline. This replaces the
    previous behaviour of looping over every un-emailed record in the season:
    when the email record was empty for those records (a fresh season, or a
    mid-season cut-over), the old loop would send the entire backlog at once,
    one email per past issuance. Scoping to a single issuance means a run can
    only ever email for the advisory it is processing; the email-record dedup
    remains as a second layer against re-sends.

    Target issued time, in priority order:
      1. the explicit ``issued_time`` argument,
      2. the ``ISSUED_TIME`` environment variable ("%Y-%m-%dT%H", UTC),
      3. the most recent ``issue_time`` present in ``df_monitoring``.

    We default to the latest issuance in the data (rather than a wall-clock
    advisory hour as ds-storms-alerts does) because the observed feed's
    issue_times are synthesised from observation timestamps, so the data is the
    reliable source of "which issuance do we actually have". The ISSUED_TIME
    override provides the same explicit, reproducible targeting for backfills
    or replays.

    Records are matched on the UTC hour of their ``issue_time``. The frame is
    returned unchanged when FORCE_ALERT is set, so the injected test row (whose
    issue_time is the season start, not the current advisory) still flows
    through to the senders.
    """
    if FORCE_ALERT or df_monitoring.empty:
        return df_monitoring

    issue_hours = _issue_time_hours_utc(df_monitoring["issue_time"])

    if issued_time is None:
        issued_time = os.getenv("ISSUED_TIME")
    if issued_time:
        target = pd.Timestamp(issued_time)
        target = (
            target.tz_localize("UTC")
            if target.tzinfo is None
            else target.tz_convert("UTC")
        ).floor("h")
    else:
        target = issue_hours.max()

    df_at_issuance = df_monitoring[issue_hours == target]
    print(
        f"📌 Scoped to issued time {target:%Y-%m-%dT%H}Z: "
        f"{len(df_at_issuance)}/{len(df_monitoring)} records"
    )
    return df_at_issuance


def load_email_record_with_test_filtering(
    email_types: list = None,
) -> pd.DataFrame:
    """Load email record with optional test data filtering.

    Args:
        email_types: List of email types to filter out for test data

    Returns:
        DataFrame with email records, filtered if FORCE_ALERT is enabled
    """
    df_existing_email_record = load_email_record()

    if FORCE_ALERT and email_types:
        df_existing_email_record = df_existing_email_record[
            ~(
                (df_existing_email_record["atcf_id"] == TEST_ATCF_ID)
                & (df_existing_email_record["email_type"].isin(email_types))
            )
        ]

    return df_existing_email_record


def save_email_record(df_existing: pd.DataFrame, new_records: list) -> None:
    """Combine existing email records with new ones and save to blob.

    Args:
        df_existing: Existing email record DataFrame
        new_records: List of dictionaries representing new email records
    """
    if DRY_RUN:
        print(f"DRY_RUN: Would save {len(new_records)} new email records")
        return

    if len(new_records) == 0:
        print("No new email records to save")
        return

    df_new_email_record = pd.DataFrame(new_records)
    df_combined_email_record = pd.concat(
        [df_existing, df_new_email_record], ignore_index=True
    )

    # Use appropriate file based on TEST_EMAIL setting
    if TEST_EMAIL:
        blob_name = f"{PROJECT_PREFIX}/email/test_email_record.csv"
    else:
        blob_name = f"{PROJECT_PREFIX}/email/email_record.csv"

    print(f"💾 Saving {len(new_records)} email records to {blob_name}")
    stratus.upload_csv_to_blob(df_combined_email_record, blob_name)


def is_valid_email(email):
    # Define a regex pattern for validating an email
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    # Use the re.match() method to check if the email matches the pattern
    if re.match(email_regex, email):
        return True
    else:
        return False
