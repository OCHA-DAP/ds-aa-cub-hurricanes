import traceback

import pandas as pd

from src.constants import MIN_EMAIL_DISTANCE, PROJECT_PREFIX
from src.email.send_emails import send_info_email, send_trigger_email
from src.email.utils import (
    TEST_ATCF_ID,
    TEST_STORM,
    add_test_row_to_monitoring,
    load_email_record,
)
from src.monitoring import monitoring_utils
import ocha_stratus as stratus


def update_obsv_info_emails(verbose: bool = False):
    df_monitoring = monitoring_utils.load_existing_monitoring_points("obsv")
    df_existing_email_record = load_email_record()
    if TEST_STORM:
        df_monitoring = add_test_row_to_monitoring(df_monitoring, "obsv")
        df_existing_email_record = df_existing_email_record[
            ~(
                (df_existing_email_record["atcf_id"] == TEST_ATCF_ID)
                & (df_existing_email_record["email_type"] == "info")
            )
        ]

    dicts = []
    for monitor_id, row in df_monitoring.set_index("monitor_id").iterrows():
        if row["min_dist"] > MIN_EMAIL_DISTANCE:
            if verbose:
                print(
                    f"min_dist is {row['min_dist']}, "
                    f"skipping info email for {monitor_id}"
                )
            continue
        if (
            monitor_id
            in df_existing_email_record[
                df_existing_email_record["email_type"] == "info"
            ]["monitor_id"].unique()
        ):
            if verbose:
                print(f"already sent info email for {monitor_id}")
            continue
        if not row["rainfall_relevant"]:
            if verbose:
                print(f"rainfall not relevant for {monitor_id}")
            continue
        try:
            print(f"sending info email for {monitor_id}")
            send_info_email(monitor_id=monitor_id, fcast_obsv="obsv")
            dicts.append(
                {
                    "monitor_id": monitor_id,
                    "atcf_id": row["atcf_id"],
                    "email_type": "info",
                }
            )
        except Exception as e:
            print(f"could not send info email for {monitor_id}: {e}")
            traceback.print_exc()

    df_new_email_record = pd.DataFrame(dicts)
    df_combined_email_record = pd.concat(
        [df_existing_email_record, df_new_email_record], ignore_index=True
    )
    blob_name = f"{PROJECT_PREFIX}/email/email_record.csv"
    stratus.upload_csv_to_blob(df_combined_email_record, blob_name)


def update_fcast_info_emails(verbose: bool = False):
    df_monitoring = monitoring_utils.load_existing_monitoring_points("fcast")
    df_existing_email_record = load_email_record()
    if TEST_STORM:
        df_monitoring = add_test_row_to_monitoring(df_monitoring, "fcast")
        df_existing_email_record = df_existing_email_record[
            ~(
                (df_existing_email_record["atcf_id"] == TEST_ATCF_ID)
                & (df_existing_email_record["email_type"] == "info")
            )
        ]

    dicts = []
    for monitor_id, row in df_monitoring.set_index("monitor_id").iterrows():
        if row["min_dist"] > MIN_EMAIL_DISTANCE:
            if verbose:
                print(
                    f"min_dist is {row['min_dist']}, "
                    f"skipping info email for {monitor_id}"
                )
            continue
        if (
            monitor_id
            in df_existing_email_record[
                df_existing_email_record["email_type"] == "info"
            ]["monitor_id"].unique()
        ):
            if verbose:
                print(f"already sent info email for {monitor_id}")
        else:
            try:
                print(f"sending info email for {monitor_id}")
                send_info_email(monitor_id=monitor_id, fcast_obsv="fcast")
                dicts.append(
                    {
                        "monitor_id": monitor_id,
                        "atcf_id": row["atcf_id"],
                        "email_type": "info",
                    }
                )
            except Exception as e:
                print(f"could not send info email for {monitor_id}: {e}")
                traceback.print_exc()

    df_new_email_record = pd.DataFrame(dicts)
    df_combined_email_record = pd.concat(
        [df_existing_email_record, df_new_email_record], ignore_index=True
    )
    blob_name = f"{PROJECT_PREFIX}/email/email_record.csv"
    stratus.upload_csv_to_blob(df_combined_email_record, blob_name)


def update_obsv_trigger_emails():
    df_monitoring = monitoring_utils.load_existing_monitoring_points("obsv")
    df_existing_email_record = load_email_record()
    if TEST_STORM:
        df_monitoring = add_test_row_to_monitoring(df_monitoring, "obsv")
        df_existing_email_record = df_existing_email_record[
            ~(
                (df_existing_email_record["atcf_id"] == TEST_ATCF_ID)
                & (df_existing_email_record["email_type"] == "obsv")
            )
        ]
    dicts = []
    for atcf_id, group in df_monitoring.groupby("atcf_id"):
        if (
            atcf_id
            in df_existing_email_record[
                df_existing_email_record["email_type"] == "obsv"
            ]["atcf_id"].unique()
        ):
            print(f"already sent obsv email for {atcf_id}")
        elif (
            atcf_id
            in df_existing_email_record[
                df_existing_email_record["email_type"] == "action"
            ]["atcf_id"].unique()
            and not TEST_STORM
        ):
            print(f"already sent action email for {atcf_id}")
        else:
            for monitor_id, row in group.set_index("monitor_id").iterrows():
                if row["obsv_trigger"]:
                    try:
                        print(f"sending obsv email for {monitor_id}")
                        send_trigger_email(
                            monitor_id=monitor_id, trigger_name="obsv"
                        )
                        dicts.append(
                            {
                                "monitor_id": monitor_id,
                                "atcf_id": atcf_id,
                                "email_type": "obsv",
                            }
                        )
                    except Exception as e:
                        print(
                            f"could not send trigger email for {monitor_id}: "
                            f"{e}"
                        )
                        traceback.print_exc()

    df_new_email_record = pd.DataFrame(dicts)
    df_combined_email_record = pd.concat(
        [df_existing_email_record, df_new_email_record], ignore_index=True
    )
    blob_name = f"{PROJECT_PREFIX}/email/email_record.csv"
    stratus.upload_csv_to_blob(df_combined_email_record, blob_name)


def update_fcast_trigger_emails():
    """Cycle through all historical monitoring points to see if we should have
    sent a trigger email for any of them. If we need to send any emails,
    send them.
    """
    df_monitoring = monitoring_utils.load_existing_monitoring_points("fcast")
    df_existing_email_record = load_email_record()
    if TEST_STORM:
        df_monitoring = add_test_row_to_monitoring(df_monitoring, "fcast")
        df_existing_email_record = df_existing_email_record[
            ~(
                (df_existing_email_record["atcf_id"] == "TEST_ATCF_ID")
                & (
                    df_existing_email_record["email_type"].isin(
                        ["readiness", "action"]
                    )
                )
            )
        ]
    dicts = []
    for atcf_id, group in df_monitoring.groupby("atcf_id"):
        for trigger_name in ["readiness", "action"]:
            if (
                atcf_id
                in df_existing_email_record[
                    df_existing_email_record["email_type"] == trigger_name
                ]["atcf_id"].unique()
            ):
                print(f"already sent {trigger_name} email for {atcf_id}")
            else:
                for (
                    monitor_id,
                    row,
                ) in group.set_index("monitor_id").iterrows():
                    if (
                        row[f"{trigger_name}_trigger"]
                        and not row["past_cutoff"]
                    ):
                        try:
                            print(
                                f"sending {trigger_name} email for "
                                f"{monitor_id}"
                            )
                            send_trigger_email(
                                monitor_id=monitor_id,
                                trigger_name=trigger_name,
                            )
                            dicts.append(
                                {
                                    "monitor_id": monitor_id,
                                    "atcf_id": atcf_id,
                                    "email_type": trigger_name,
                                }
                            )
                        except Exception as e:
                            print(
                                f"could not send trigger email for "
                                f"{monitor_id}: {e}"
                            )
                            traceback.print_exc()

    df_new_email_record = pd.DataFrame(dicts)
    df_combined_email_record = pd.concat(
        [df_existing_email_record, df_new_email_record], ignore_index=True
    )
    blob_name = f"{PROJECT_PREFIX}/email/email_record.csv"
    stratus.upload_csv_to_blob(df_combined_email_record, blob_name)
