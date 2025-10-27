import io
import smtplib
import ssl
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Literal

import ocha_stratus as stratus
import pytz
from html2text import html2text
from jinja2 import Environment, FileSystemLoader

from src.constants import DRY_RUN, FORCE_ALERT, SPANISH_MONTHS
from src.email.plotting import get_plot_blob_name
from src.email.utils import (
    EMAIL_ADDRESS,
    EMAIL_DISCLAIMER,
    EMAIL_HOST,
    EMAIL_PASSWORD,
    EMAIL_PORT,
    EMAIL_USERNAME,
    STATIC_DIR,
    TEMPLATES_DIR,
    get_distribution_list,
    is_valid_email,
    load_monitoring_data,
)


def prepare_email_data(
    monitor_id: str,
    fcast_obsv: Literal["fcast", "obsv"],
):
    """Prepare data needed for email (shared between send/preview)."""
    # Load monitoring data (FORCE_ALERT controlled internally)
    df_monitoring = load_monitoring_data(fcast_obsv)

    monitoring_point = df_monitoring.set_index("monitor_id").loc[monitor_id]

    # No need for DataFrame check - test IDs are unique
    cuba_tz = pytz.timezone("America/Havana")
    cyclone_name = monitoring_point["name"]
    issue_time = monitoring_point["issue_time"]
    issue_time_cuba = issue_time.astimezone(cuba_tz)
    pub_time = issue_time_cuba.strftime("%Hh%M")
    pub_date = issue_time_cuba.strftime("%-d %b %Y")

    day = issue_time_cuba.strftime("%-d")
    month = issue_time_cuba.strftime("%B")
    year = issue_time_cuba.strftime("%Y")
    time = issue_time_cuba.strftime("%Hh%M")

    for en_mo, es_mo in SPANISH_MONTHS.items():
        pub_date = pub_date.replace(en_mo, es_mo)
    fcast_obsv_es = "observación" if fcast_obsv == "obsv" else "pronóstico"
    fcast_obsv_en = "observation" if fcast_obsv == "obsv" else "forecast"
    activation_subject = "(SIN ACTIVACIÓN)"

    pub_datetime_txt = f"{day} de {month} {year} a las {time}"

    # English date formatting
    pub_date_en = issue_time_cuba.strftime("%-d %B %Y")

    if fcast_obsv == "fcast":
        readiness = (
            "ALCANZADO"
            if monitoring_point["readiness_trigger"]
            else "NO ALCANZADO"
        )
        action = (
            "ALCANZADO"
            if monitoring_point["action_trigger"]
            else "NO ALCANZADO"
        )
        obsv = ""

        # English versions
        readiness_en = (
            "REACHED"
            if monitoring_point["readiness_trigger"]
            else "NOT REACHED"
        )
        action_en = (
            "REACHED" if monitoring_point["action_trigger"] else "NOT REACHED"
        )
        obsv_en = ""
    else:
        readiness = ""  # noqa
        action = ""  # noqa
        obsv = (
            "ALCANZADO" if monitoring_point["obsv_trigger"] else "NO ALCANZADO"
        )

        # English versions
        readiness_en = ""  # noqa
        action_en = ""  # noqa
        obsv_en = (
            "REACHED" if monitoring_point["obsv_trigger"] else "NOT REACHED"
        )

    return {
        "cyclone_name": cyclone_name,
        "pub_time": pub_time,
        "pub_date": pub_date,
        "pub_date_en": pub_date_en,
        "pub_datetime_txt": pub_datetime_txt,
        "fcast_obsv_es": fcast_obsv_es,
        "fcast_obsv_en": fcast_obsv_en,
        "activation_subject": activation_subject,
        "readiness": "ALCANZADO",
        "readiness_en": "REACHED",
        "action": "ALCANZADO",
        "action_en": "REACHED",
        "obsv": obsv,
        "obsv_en": obsv_en,
        "show_scatter_plot": False,  # Set to True to show the scatter plot
    }


def create_info_email_content(
    monitor_id: str,
    fcast_obsv: Literal["fcast", "obsv"],
    for_preview: bool = False,
):
    """Create the HTML and text content for an info email."""
    # Email data loading respects FORCE_ALERT internally
    email_data = prepare_email_data(monitor_id, fcast_obsv)

    if not for_preview:
        distribution_list = get_distribution_list()
        valid_distribution_list = distribution_list[
            distribution_list["email"].apply(is_valid_email)
        ]
        invalid_distribution_list = distribution_list[
            ~distribution_list["email"].apply(is_valid_email)
        ]
        if not invalid_distribution_list.empty:
            print(
                f"Invalid emails found in distribution list: "
                f"{invalid_distribution_list['email'].tolist()}"
            )
        to_list = valid_distribution_list[
            valid_distribution_list["info"] == "to"
        ]
        cc_list = valid_distribution_list[
            valid_distribution_list["info"] == "cc"
        ]
    else:
        to_list = cc_list = None

    test_subject = "PRUEBA : " if FORCE_ALERT else ""

    environment = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    template_name = "informational"
    template = environment.get_template(f"{template_name}.html")

    subject = (
        f"{test_subject}Acción anticipatoria Cuba – información sobre "
        f"{email_data['fcast_obsv_es']} {email_data['cyclone_name']} "
        f"{email_data['pub_time']}, {email_data['pub_date']} "
        # f"{email_data['activation_subject']}"
    )

    if for_preview:
        # For preview, use placeholder CIDs that will be replaced later
        map_cid = "PREVIEW_MAP_PLACEHOLDER"
        scatter_cid = "PREVIEW_SCATTER_PLACEHOLDER"
        ocha_logo_cid = "PREVIEW_LOGO_PLACEHOLDER"
    else:
        # For real emails, generate proper CIDs
        map_cid = make_msgid(domain="humdata.org")[1:-1]
        scatter_cid = make_msgid(domain="humdata.org")[1:-1]
        ocha_logo_cid = make_msgid(domain="humdata.org")[1:-1]

    html_str = template.render(
        name=email_data["cyclone_name"],
        pub_time=email_data["pub_time"],
        pub_date=email_data["pub_date"],
        pub_date_en=email_data["pub_date_en"],
        fcast_obsv=email_data["fcast_obsv_es"],
        fcast_obsv_en=email_data["fcast_obsv_en"],
        readiness=email_data["readiness"],
        readiness_en=email_data["readiness_en"],
        action=email_data["action"],
        action_en=email_data["action_en"],
        obsv=email_data["obsv"],
        obsv_en=email_data["obsv_en"],
        test_email=FORCE_ALERT,
        email_disclaimer=EMAIL_DISCLAIMER,
        map_cid=map_cid,
        scatter_cid=scatter_cid,
        ocha_logo_cid=ocha_logo_cid,
    )
    text_str = html2text(html_str)

    return {
        "subject": subject,
        "html": html_str,
        "text": text_str,
        "to_list": to_list,
        "cc_list": cc_list,
        "email_data": email_data,
        "cids": {
            "map": map_cid,
            "scatter": scatter_cid,
            "ocha_logo": ocha_logo_cid,
        },
    }


def send_info_email(monitor_id: str, fcast_obsv: Literal["fcast", "obsv"]):
    """Send info email using shared email content creation."""
    if DRY_RUN:
        print(f"DRY_RUN: Would send info email for {monitor_id}")
        return

    email_content = create_info_email_content(
        monitor_id, fcast_obsv, for_preview=False
    )

    msg = EmailMessage()
    msg.set_charset("utf-8")
    msg["Subject"] = email_content["subject"]
    msg["From"] = Address(
        "Centro de Datos Humanitarios OCHA",
        EMAIL_ADDRESS.split("@")[0],
        EMAIL_ADDRESS.split("@")[1],
    )
    msg["To"] = [
        Address(
            row["name"], row["email"].split("@")[0], row["email"].split("@")[1]
        )
        for _, row in email_content["to_list"].iterrows()
    ]
    msg["Cc"] = [
        Address(
            row["name"], row["email"].split("@")[0], row["email"].split("@")[1]
        )
        for _, row in email_content["cc_list"].iterrows()
    ]

    msg.set_content(email_content["text"])
    msg.add_alternative(email_content["html"], subtype="html")

    # Add plot images from blob storage using the CIDs from email content
    for plot_type in ["map", "scatter"]:
        blob_name = get_plot_blob_name(monitor_id, plot_type)
        image_data = io.BytesIO()
        container_client = stratus.get_container_client()
        blob_client = container_client.get_blob_client(blob_name)

        try:
            blob_client.download_blob().download_to_stream(image_data)
            image_data.seek(0)
            # Use the CID that was already embedded in the email content
            cid = email_content["cids"][plot_type]
            msg.get_payload()[1].add_related(
                image_data.read(), "image", "png", cid=cid
            )
            print(f"✅ Added {plot_type} plot to email")
        except Exception as e:
            print(f"⚠️  Could not attach {plot_type} plot: {e}")
            # Continue without this plot - the email will show broken image

    # Add static images using the CIDs from email content
    for filename, cid_key in zip(
        ["ocha_logo_wide.png"],
        ["ocha_logo"],
    ):
        img_path = STATIC_DIR / filename
        with open(img_path, "rb") as img:
            # Use the CID that was already embedded in the email content
            cid = email_content["cids"][cid_key]
            msg.get_payload()[1].add_related(
                img.read(), "image", "png", cid=cid
            )

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as server:
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.sendmail(
            EMAIL_ADDRESS,
            email_content["to_list"]["email"].tolist()
            + email_content["cc_list"]["email"].tolist(),
            msg.as_string(),
        )


def send_trigger_email(monitor_id: str, trigger_name: str):
    """Send trigger email to distribution list."""
    if DRY_RUN:
        print(
            f"DRY_RUN: Would send {trigger_name} trigger email for "
            f"{monitor_id}"
        )
        return

    fcast_obsv = "fcast" if trigger_name in ["readiness", "action"] else "obsv"
    df_monitoring = load_monitoring_data(fcast_obsv)
    monitoring_point = df_monitoring.set_index("monitor_id").loc[monitor_id]
    cuba_tz = pytz.timezone("America/Havana")
    cyclone_name = monitoring_point["name"]
    issue_time = monitoring_point["issue_time"]
    issue_time_cuba = issue_time.astimezone(cuba_tz)
    pub_time = issue_time_cuba.strftime("%Hh%M")
    pub_date = issue_time_cuba.strftime("%-d %b %Y")
    pub_date_en = issue_time_cuba.strftime("%-d %B %Y")
    for en_mo, es_mo in SPANISH_MONTHS.items():
        pub_date = pub_date.replace(en_mo, es_mo)
    if trigger_name == "readiness":
        trigger_name_es = "preparación"
        trigger_name_en = "readiness"
        trigger_type_subj = "de ALISTAMIENTO"
    elif trigger_name == "action":
        trigger_name_es = "acción"
        trigger_name_en = "action"
        trigger_type_subj = "de ACCIÓN"
    else:
        trigger_name_es = "observacional"
        trigger_name_en = "observational"
        trigger_type_subj = trigger_name_es.upper()

    fcast_obsv_es = "observación" if fcast_obsv == "obsv" else "pronóstico"
    fcast_obsv_en = "observation" if fcast_obsv == "obsv" else "forecast"

    distribution_list = get_distribution_list()
    valid_distribution_list = distribution_list[
        distribution_list["email"].apply(is_valid_email)
    ]
    invalid_distribution_list = distribution_list[
        ~distribution_list["email"].apply(is_valid_email)
    ]
    if not invalid_distribution_list.empty:
        print(
            f"Invalid emails found in distribution list: "
            f"{invalid_distribution_list['email'].tolist()}"
        )
    to_list = valid_distribution_list[
        valid_distribution_list["trigger"] == "to"
    ]
    cc_list = valid_distribution_list[
        valid_distribution_list["trigger"] == "cc"
    ]

    test_subject = "PRUEBA : " if FORCE_ALERT else ""

    environment = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

    template_name = "observational" if trigger_name == "obsv" else trigger_name
    template = environment.get_template(f"{template_name}.html")
    msg = EmailMessage()
    msg.set_charset("utf-8")
    msg["Subject"] = (
        f"{test_subject}Acción anticipatoria Cuba – Disparador "
        f"{trigger_type_subj} alcanzado"
    )
    msg["From"] = Address(
        "Centro de Datos Humanitarios OCHA",
        EMAIL_ADDRESS.split("@")[0],
        EMAIL_ADDRESS.split("@")[1],
    )
    msg["To"] = [
        Address(
            row["name"],
            row["email"].split("@")[0],
            row["email"].split("@")[1],
        )
        for _, row in to_list.iterrows()
    ]
    msg["Cc"] = [
        Address(
            row["name"],
            row["email"].split("@")[0],
            row["email"].split("@")[1],
        )
        for _, row in cc_list.iterrows()
    ]

    ocha_logo_cid = make_msgid(domain="humdata.org")

    html_str = template.render(
        name=cyclone_name,
        pub_time=pub_time,
        pub_date=pub_date,
        pub_date_en=pub_date_en,
        fcast_obsv=fcast_obsv_es,
        fcast_obsv_en=fcast_obsv_en,
        trigger_name_en=trigger_name_en,
        test_email=FORCE_ALERT,
        email_disclaimer=EMAIL_DISCLAIMER,
        ocha_logo_cid=ocha_logo_cid[1:-1],
    )
    text_str = html2text(html_str)
    msg.set_content(text_str)
    msg.add_alternative(html_str, subtype="html")

    for filename, cid in zip(["ocha_logo_wide.png"], [ocha_logo_cid]):
        img_path = STATIC_DIR / filename
        with open(img_path, "rb") as img:
            msg.get_payload()[1].add_related(
                img.read(), "image", "png", cid=cid
            )

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT, context=context) as server:
        server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
        server.sendmail(
            EMAIL_ADDRESS,
            to_list["email"].tolist() + cc_list["email"].tolist(),
            msg.as_string(),
        )
