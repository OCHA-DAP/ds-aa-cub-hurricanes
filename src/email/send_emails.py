import io
import smtplib
import ssl
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Literal

import pytz
from html2text import html2text
from jinja2 import Environment, FileSystemLoader

from src.constants import SPANISH_MONTHS
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
    TEST_FCAST_MONITOR_ID,
    TEST_OBSV_MONITOR_ID,
    TEST_STORM,
    add_test_row_to_monitoring,
    get_distribution_list,
    is_valid_email,
)
from src.monitoring import monitoring_utils
import ocha_stratus as stratus


def send_info_email(monitor_id: str, fcast_obsv: Literal["fcast", "obsv"]):
    df_monitoring = monitoring_utils.load_existing_monitoring_points(
        fcast_obsv
    )
    if monitor_id in [TEST_FCAST_MONITOR_ID, TEST_OBSV_MONITOR_ID]:
        df_monitoring = add_test_row_to_monitoring(df_monitoring, fcast_obsv)
    monitoring_point = df_monitoring.set_index("monitor_id").loc[monitor_id]
    cuba_tz = pytz.timezone("America/Havana")
    cyclone_name = monitoring_point["name"]
    issue_time = monitoring_point["issue_time"]
    issue_time_cuba = issue_time.astimezone(cuba_tz)
    pub_time = issue_time_cuba.strftime("%Hh%M")
    pub_date = issue_time_cuba.strftime("%-d %b %Y")
    for en_mo, es_mo in SPANISH_MONTHS.items():
        pub_date = pub_date.replace(en_mo, es_mo)
    fcast_obsv_es = "observación" if fcast_obsv == "obsv" else "pronóstico"
    activation_subject = "(SIN ACTIVACIÓN)"
    if fcast_obsv == "fcast":
        readiness = (
            "ACTIVADO"
            if monitoring_point["readiness_trigger"]
            else "NO ACTIVADO"
        )
        action = (
            "ACTIVADO" if monitoring_point["action_trigger"] else "NO ACTIVADO"
        )
        obsv = ""
    else:
        readiness = ""
        action = ""
        obsv = (
            "ACTIVADO" if monitoring_point["obsv_trigger"] else "NO ACTIVADO"
        )

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
    to_list = valid_distribution_list[valid_distribution_list["info"] == "to"]
    cc_list = valid_distribution_list[valid_distribution_list["info"] == "cc"]

    test_subject = "PRUEBA : " if TEST_STORM else ""

    environment = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

    template_name = "informational"
    template = environment.get_template(f"{template_name}.html")
    msg = EmailMessage()
    msg.set_charset("utf-8")
    msg["Subject"] = (
        f"{test_subject}Acción anticipatoria Cuba – información sobre "
        f"pronóstico {cyclone_name} {pub_time}, {pub_date} {activation_subject}"
    )
    msg["From"] = Address(
        "Centro de Datos Humanitarios OCHA",
        EMAIL_ADDRESS.split("@")[0],
        EMAIL_ADDRESS.split("@")[1],
    )
    msg["To"] = [
        Address(
            row["name"], row["email"].split("@")[0], row["email"].split("@")[1]
        )
        for _, row in to_list.iterrows()
    ]
    msg["Cc"] = [
        Address(
            row["name"], row["email"].split("@")[0], row["email"].split("@")[1]
        )
        for _, row in cc_list.iterrows()
    ]

    map_cid = make_msgid(domain="humdata.org")
    scatter_cid = make_msgid(domain="humdata.org")
    chd_banner_cid = make_msgid(domain="humdata.org")
    ocha_logo_cid = make_msgid(domain="humdata.org")

    html_str = template.render(
        name=cyclone_name,
        pub_time=pub_time,
        pub_date=pub_date,
        fcast_obsv=fcast_obsv_es,
        readiness=readiness,
        action=action,
        obsv=obsv,
        test_email=TEST_STORM,
        email_disclaimer=EMAIL_DISCLAIMER,
        map_cid=map_cid[1:-1],
        scatter_cid=scatter_cid[1:-1],
        chd_banner_cid=chd_banner_cid[1:-1],
        ocha_logo_cid=ocha_logo_cid[1:-1],
    )
    text_str = html2text(html_str)
    msg.set_content(text_str)
    msg.add_alternative(html_str, subtype="html")

    for plot_type, cid in zip(["map", "scatter"], [map_cid, scatter_cid]):
        blob_name = get_plot_blob_name(monitor_id, plot_type)
        image_data = io.BytesIO()
        container_client = stratus.get_container_client()
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.download_blob().download_to_stream(image_data)
        image_data.seek(0)
        msg.get_payload()[1].add_related(
            image_data.read(), "image", "png", cid=cid
        )

    for filename, cid in zip(
        ["centre_banner.png", "ocha_logo_wide.png"],
        [chd_banner_cid, ocha_logo_cid],
    ):
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


def send_trigger_email(monitor_id: str, trigger_name: str):
    """Send trigger email to distribution list."""
    fcast_obsv = "fcast" if trigger_name in ["readiness", "action"] else "obsv"
    df_monitoring = monitoring_utils.load_existing_monitoring_points(
        fcast_obsv
    )
    if monitor_id in [TEST_FCAST_MONITOR_ID, TEST_OBSV_MONITOR_ID]:
        df_monitoring = add_test_row_to_monitoring(df_monitoring, fcast_obsv)
    monitoring_point = df_monitoring.set_index("monitor_id").loc[monitor_id]
    cuba_tz = pytz.timezone("America/Havana")
    cyclone_name = monitoring_point["name"]
    issue_time = monitoring_point["issue_time"]
    issue_time_cuba = issue_time.astimezone(cuba_tz)
    pub_time = issue_time_cuba.strftime("%Hh%M")
    pub_date = issue_time_cuba.strftime("%-d %b %Y")
    for en_mo, es_mo in SPANISH_MONTHS.items():
        pub_date = pub_date.replace(en_mo, es_mo)
    if trigger_name == "readiness":
        trigger_name_es = "preparación"
    elif trigger_name == "action":
        trigger_name_es = "acción"
    else:
        trigger_name_es = "observacional"
    fcast_obsv_es = "observación" if fcast_obsv == "obsv" else "pronóstico"

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

    test_subject = "PRUEBA : " if TEST_STORM else ""

    environment = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

    template_name = "observational" if trigger_name == "obsv" else trigger_name
    template = environment.get_template(f"{template_name}.html")
    msg = EmailMessage()
    msg.set_charset("utf-8")
    msg["Subject"] = (
        f"{test_subject}Acción anticipatoria Cuba – "
        f"activador {trigger_name_es} alcanzado para "
        f"{cyclone_name}"
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

    chd_banner_cid = make_msgid(domain="humdata.org")
    ocha_logo_cid = make_msgid(domain="humdata.org")

    html_str = template.render(
        name=cyclone_name,
        pub_time=pub_time,
        pub_date=pub_date,
        fcast_obsv=fcast_obsv_es,
        test_email=TEST_STORM,
        email_disclaimer=EMAIL_DISCLAIMER,
        chd_banner_cid=chd_banner_cid[1:-1],
        ocha_logo_cid=ocha_logo_cid[1:-1],
    )
    text_str = html2text(html_str)
    msg.set_content(text_str)
    msg.add_alternative(html_str, subtype="html")

    for filename, cid in zip(
        ["centre_banner.png", "ocha_logo_wide.png"],
        [chd_banner_cid, ocha_logo_cid],
    ):
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
