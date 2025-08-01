import base64
import os
import re
from pathlib import Path
from typing import Literal
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
        # around day 3 of the storm track
        threshold_crossing_date = MONITORING_START_DATE + pd.Timedelta(days=3)

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
