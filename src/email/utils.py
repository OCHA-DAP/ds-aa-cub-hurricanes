import base64
import os
import re
from pathlib import Path
from typing import Literal
import pandas as pd
import ocha_stratus as stratus

from src.constants import PROJECT_PREFIX

EMAIL_HOST = os.getenv("CHD_DS_HOST")
EMAIL_PORT = int(os.getenv("CHD_DS_PORT"))
EMAIL_PASSWORD = os.getenv("CHD_DS_EMAIL_PASSWORD")
EMAIL_USERNAME = os.getenv("CHD_DS_EMAIL_USERNAME")
EMAIL_ADDRESS = os.getenv("CHD_DS_EMAIL_ADDRESS")

TEST_LIST = os.getenv("TEST_LIST")
if TEST_LIST == "False":
    TEST_LIST = False
else:
    TEST_LIST = True

TEST_STORM = os.getenv("TEST_STORM")
if TEST_STORM == "False":
    TEST_STORM = False
else:
    TEST_STORM = True

EMAIL_DISCLAIMER = os.getenv("EMAIL_DISCLAIMER")
if EMAIL_DISCLAIMER == "True":
    EMAIL_DISCLAIMER = True
else:
    EMAIL_DISCLAIMER = False

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
    print("adding test row to monitoring data")
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
    if TEST_LIST:
        blob_name = f"{PROJECT_PREFIX}/email/test_distribution_list.csv"
    else:
        blob_name = f"{PROJECT_PREFIX}/email/distribution_list.csv"
    return stratus.load_csv_from_blob(blob_name)


def load_email_record() -> pd.DataFrame:
    """Load record of emails that have been sent."""
    blob_name = f"{PROJECT_PREFIX}/email/email_record.csv"
    return stratus.load_csv_from_blob(blob_name)


def load_monitoring_data(
    fcast_obsv: Literal["fcast", "obsv"], with_tests: bool = True
) -> pd.DataFrame:
    """Load monitoring data with optional test row injection.

    Args:
        fcast_obsv: Whether to load forecast or observation data
        with_tests: Whether to add test rows when TEST_STORM is enabled

    Returns:
        DataFrame with monitoring data, optionally including test rows
    """
    from src.monitoring import monitoring_utils

    monitor = monitoring_utils.create_cuba_hurricane_monitor()
    df_monitoring = monitor._load_existing_monitoring(fcast_obsv)

    if with_tests and TEST_STORM:
        df_monitoring = add_test_row_to_monitoring(df_monitoring, fcast_obsv)

    return df_monitoring


def load_email_record_with_test_filtering(
    email_types: list = None,
) -> pd.DataFrame:
    """Load email record with optional test data filtering.

    Args:
        email_types: List of email types to filter out for test data

    Returns:
        DataFrame with email records, filtered if TEST_STORM is enabled
    """
    df_existing_email_record = load_email_record()

    if TEST_STORM and email_types:
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
    df_new_email_record = pd.DataFrame(new_records)
    df_combined_email_record = pd.concat(
        [df_existing, df_new_email_record], ignore_index=True
    )
    blob_name = f"{PROJECT_PREFIX}/email/email_record.csv"
    stratus.upload_csv_to_blob(df_combined_email_record, blob_name)


def is_valid_email(email):
    # Define a regex pattern for validating an email
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"

    # Use the re.match() method to check if the email matches the pattern
    if re.match(email_regex, email):
        return True
    else:
        return False
