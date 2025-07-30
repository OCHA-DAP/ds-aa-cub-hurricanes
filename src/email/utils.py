import base64
import os
import re
from pathlib import Path
from typing import Literal
import pandas as pd
import ocha_stratus as stratus

from src.constants import (
    PROJECT_PREFIX,
    DRY_RUN,
    TEST_EMAIL,
    FORCE_ALERT,
    MONITORING_START_DATE,
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

TEST_ATCF_ID = "TEST_ATCF_ID"
TEST_MONITOR_ID = "TEST_MONITOR_ID"
TEST_FCAST_MONITOR_ID = "TEST_FCAST_MONITOR_ID"
TEST_OBSV_MONITOR_ID = "TEST_OBSV_MONITOR_ID"
TEST_STORM_NAME = "TEST_STORM_NAME"

TEMPLATES_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "templates"
STATIC_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "static"


def add_test_row_to_monitoring(
    df_monitoring: pd.DataFrame, fcast_obsv: str
) -> pd.DataFrame:
    """Add test row to monitoring df to simulate new monitoring point.
    This new monitoring point will cause an activation of all triggers.
    Uses Hurricane Rafael data as a template but creates proper test IDs.
    """
    # Only print this once per fcast/obsv type per process
    if not hasattr(add_test_row_to_monitoring, f"_added_{fcast_obsv}"):
        if fcast_obsv == "fcast":
            print("🧪 Adding test forecast row for FORCE_ALERT testing")
        else:
            print("🧪 Adding test observation row for FORCE_ALERT testing")
        setattr(add_test_row_to_monitoring, f"_added_{fcast_obsv}", True)
    if fcast_obsv == "fcast":
        # Use Hurricane Rafael as template but create test row
        df_monitoring_test = df_monitoring[
            df_monitoring["monitor_id"] == "al182024_fcast_2024-11-04T21:00:00"
        ].copy()
        df_monitoring_test[
            [
                "monitor_id",
                "name",
                "atcf_id",
                "readiness_trigger",
                "action_trigger",
            ]
        ] = (
            TEST_FCAST_MONITOR_ID,
            TEST_STORM_NAME,
            TEST_ATCF_ID,
            True,
            True,
        )
        # Set issue_time to MONITORING_START_DATE for test row
        df_monitoring_test["issue_time"] = MONITORING_START_DATE
        # Ensure test row always triggers by setting past_cutoff to False
        if "past_cutoff" in df_monitoring_test.columns:
            df_monitoring_test["past_cutoff"] = False
        df_monitoring = pd.concat(
            [df_monitoring, df_monitoring_test], ignore_index=True
        )
    else:
        # Use Hurricane Rafael as template but create test row
        df_monitoring_test = df_monitoring[
            df_monitoring["monitor_id"] == "al182024_obsv_2024-11-06T21:00:00"
        ].copy()
        df_monitoring_test[
            [
                "monitor_id",
                "name",
                "atcf_id",
                "obsv_trigger",
            ]
        ] = (
            TEST_OBSV_MONITOR_ID,
            TEST_STORM_NAME,
            TEST_ATCF_ID,
            True,
        )
        # Set issue_time to MONITORING_START_DATE for test row
        df_monitoring_test["issue_time"] = MONITORING_START_DATE
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
