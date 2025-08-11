import traceback

from src.constants import MIN_EMAIL_DISTANCE
from src.email.send_emails import send_info_email, send_trigger_email
from src.email.utils import (
    FORCE_ALERT,
    load_monitoring_data,
    load_email_record_with_test_filtering,
    save_email_record,
)


def update_obsv_info_emails(verbose: bool = False):
    """Check observational monitoring data and coordinate info email sending.

    Iterates through observational monitoring points and calls
    send_info_email() for storms meeting criteria (within distance, no prior
    email, relevant rainfall). Updates the email record to track what was sent.

    Args:
        verbose: Print detailed progress messages
    """
    df_monitoring = load_monitoring_data("obsv")
    df_existing_email_record = load_email_record_with_test_filtering(["info"])

    # Log email eligibility summary with rainfall criteria
    within_distance = df_monitoring[
        df_monitoring["min_dist"] <= MIN_EMAIL_DISTANCE
    ]
    within_distance_and_rain = within_distance[
        within_distance["rainfall_relevant"]
    ]

    print("📧 Observational email eligibility check:")
    if len(within_distance) > 0:
        print(
            f"   ✅ {len(within_distance)} storms within "
            f"{MIN_EMAIL_DISTANCE}km:"
        )
        for _, row in within_distance.iterrows():
            storm_date = row["issue_time"].strftime("%Y-%m-%d")
            print(
                f"      • {row['name']} ({storm_date}): "
                f"{row['min_dist']:.1f}km"
            )
    else:
        print(f"   ⚠️  No storms within {MIN_EMAIL_DISTANCE}km threshold")

    print(
        f"   🌧️  {len(within_distance_and_rain)}/{len(within_distance)} "
        f"have relevant rainfall"
    )

    if len(within_distance_and_rain) == 0:
        print("   ⚠️  No storms meet both distance AND rainfall criteria")

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

    save_email_record(df_existing_email_record, dicts)


def update_fcast_info_emails(verbose: bool = False):
    """Check forecast monitoring data and coordinate info email sending.

    Iterates through forecast monitoring points and calls send_info_email()
    for storms meeting criteria (within distance, no prior email).
    Updates the email record to track what was sent.

    Args:
        verbose: Print detailed progress messages
    """
    df_monitoring = load_monitoring_data("fcast")
    df_existing_email_record = load_email_record_with_test_filtering(["info"])

    # Log email eligibility summary
    eligible_storms = df_monitoring[
        df_monitoring["min_dist"] <= MIN_EMAIL_DISTANCE
    ]
    total_count = len(df_monitoring)
    eligible_count = len(eligible_storms)
    skipped_count = total_count - eligible_count

    print("📧 Forecast email eligibility check:")
    if eligible_count > 0:
        print(f"   ✅ {eligible_count} storms within {MIN_EMAIL_DISTANCE}km:")
        for _, row in eligible_storms.iterrows():
            storm_date = row["issue_time"].strftime("%Y-%m-%d")
            print(
                f"      • {row['name']} ({storm_date}): "
                f"{row['min_dist']:.1f}km"
            )
    else:
        print(f"   ⚠️  No storms within {MIN_EMAIL_DISTANCE}km threshold")
    if skipped_count > 0:
        print(f"   ⏭️  {skipped_count} storms beyond distance threshold")

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

    save_email_record(df_existing_email_record, dicts)


def update_obsv_trigger_emails():
    """Check observational data and coordinate trigger email sending.

    Iterates through observational monitoring data grouped by storm (atcf_id)
    and calls send_trigger_email() for storms with obsv_trigger=True.
    Avoids duplicates by checking if obsv or action emails already sent.
    Updates the email record to track what was sent.
    """
    df_monitoring = load_monitoring_data("obsv")
    df_existing_email_record = load_email_record_with_test_filtering(["obsv"])
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
            and not FORCE_ALERT
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

    save_email_record(df_existing_email_record, dicts)


def update_fcast_trigger_emails():
    """Check forecast data and coordinate trigger email sending.

    Iterates through forecast monitoring data grouped by storm (atcf_id)
    and calls send_trigger_email() for storms meeting readiness or action
    trigger criteria (not past cutoff). Checks each trigger type separately.
    Updates the email record to track what was sent.
    """
    df_monitoring = load_monitoring_data("fcast")
    df_existing_email_record = load_email_record_with_test_filtering(
        ["readiness", "action"]
    )
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

    save_email_record(df_existing_email_record, dicts)
