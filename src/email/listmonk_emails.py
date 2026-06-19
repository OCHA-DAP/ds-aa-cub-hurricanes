"""Listmonk dispatch for Cuba hurricane emails (ocha_relay campaigns).

Mirrors the public send functions of src.email.send_emails (the humdata_email
SMTP fallback) so src.email.backends can route to either based on
EMAIL_BACKEND. There is no automatic failover; the backend is chosen manually.

Content is kept identical to the SMTP emails by reusing the SAME content
templates (informational / observational / action / readiness) and the SAME
context builder (send_emails.prepare_email_data). Only the Jinja base differs:
the content templates are rendered against the bilingual colour-card base in
templates/listmonk/ instead of the full SMTP base.html, and the inline-image
cid references are swapped for hosted upload_media URLs. The listmonk template
(LISTMONK_CAMPAIGN_TEMPLATE_NAME) supplies the document, OCHA logo, contact and
footer.
"""

import os
from pathlib import Path

import ocha_stratus as stratus
import requests
from jinja2 import ChoiceLoader, Environment, FileSystemLoader
from ocha_relay.listmonk import ListmonkClient

from src.constants import (
    DRY_RUN,
    FORCE_ALERT,
    LISTMONK_CAMPAIGN_TEMPLATE_NAME,
    LISTMONK_LISTS,
    LISTMONK_PROJECT_TAG,
    TEST_EMAIL,
    _parse_bool_env,
)
from src.email.plotting import get_plot_blob_name
from src.email.send_emails import prepare_email_data
from src.email.utils import TEMPLATES_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

_LISTMONK_TEMPLATES_DIR = (
    Path(__file__).resolve().parent / "templates" / "listmonk"
)
# Placeholder cids rendered into the content blocks, swapped for hosted
# upload_media URLs after rendering (the content templates use cid: refs).
_MAP_TOKEN = "LISTMONK_MAP"
_SCATTER_TOKEN = "LISTMONK_SCATTER"

# trigger_name -> (es label, en label, subject fragment). Mirrors the mapping
# in send_emails.send_trigger_email so subjects match the SMTP path.
_TRIGGER_LABELS = {
    "readiness": ("preparación", "readiness", "de ALISTAMIENTO"),
    "action": ("acción", "action", "de ACCIÓN"),
    "obsv": ("observacional", "observational", "OBSERVACIONAL"),
}


def _env() -> Environment:
    """Jinja env that renders the existing content templates against the
    bilingual listmonk base (``base.html`` resolves to templates/listmonk/)."""
    return Environment(
        loader=ChoiceLoader(
            [
                FileSystemLoader(str(_LISTMONK_TEMPLATES_DIR)),
                FileSystemLoader(str(TEMPLATES_DIR)),
            ]
        )
    )


def _resolve_list_id(client: ListmonkClient, list_type: str) -> int:
    # In test mode every send goes to the test list instead of the real
    # info/trigger audience.
    effective_type = "test" if TEST_EMAIL else list_type
    tag = LISTMONK_LISTS[effective_type]["tag"]
    for lst in client.fetch_all_lists(tag=LISTMONK_PROJECT_TAG):
        if tag in lst.get("tags", []):
            return lst["id"]
    raise RuntimeError(
        f"No listmonk list tagged {tag!r}. Run "
        f"pipelines/setup_cub_listmonk_lists.py first."
    )


def _resolve_template_id() -> int:
    """Resolve the campaign template id by name (ocha_relay has no list-
    templates method, so query the API directly with the send credentials)."""
    base = os.environ["DSCI_LISTMONK_BASE_URL"].rstrip("/")
    auth = (
        os.environ["DSCI_LISTMONK_API_USERNAME"],
        os.environ["DSCI_LISTMONK_API_KEY"],
    )
    resp = requests.get(f"{base}/templates", auth=auth)
    resp.raise_for_status()
    for t in resp.json()["data"]:
        if t["name"] == LISTMONK_CAMPAIGN_TEMPLATE_NAME:
            return t["id"]
    raise RuntimeError(
        f"Listmonk template {LISTMONK_CAMPAIGN_TEMPLATE_NAME!r} not found."
    )


def _campaign_name(kind: str, monitor_id: str) -> str:
    name = f"{LISTMONK_PROJECT_TAG} {kind} {monitor_id}"
    # [test] makes the listmonk template show its test banner — applied for a
    # forced-alert (dummy data) or any test-list send.
    return f"{name} [test]" if (FORCE_ALERT or TEST_EMAIL) else name


def _read_plot_bytes(monitor_id: str, plot_type: str) -> bytes:
    blob_name = get_plot_blob_name(monitor_id, plot_type)
    container_client = stratus.get_container_client()
    blob_client = container_client.get_blob_client(blob_name)
    return blob_client.download_blob().readall()


def _upload_and_swap(
    client: ListmonkClient, html: str, monitor_id: str
) -> str:
    """Upload referenced plots to the listmonk media library and replace the
    placeholder cid refs with the returned hosted URLs."""
    for token, plot_type in ((_MAP_TOKEN, "map"), (_SCATTER_TOKEN, "scatter")):
        marker = f"cid:{token}"
        if marker not in html:
            continue
        url = client.upload_media(
            _read_plot_bytes(monitor_id, plot_type), f"{plot_type}.png"
        )
        html = html.replace(marker, url)
    return html


def _send_campaign(
    subject: str,
    body: str,
    list_type: str,
    kind: str,
    monitor_id: str,
    with_images: bool,
) -> None:
    if DRY_RUN:
        logger.info(
            f"DRY_RUN: would send listmonk {kind} campaign for {monitor_id}"
        )
        return
    client = ListmonkClient.from_env()
    if with_images:
        body = _upload_and_swap(client, body, monitor_id)
    cid = client.create_campaign(
        name=_campaign_name(kind, monitor_id),
        subject=subject,
        body=body,
        list_ids=[_resolve_list_id(client, list_type)],
        template_id=_resolve_template_id(),
    )
    # Confirmation mode: by default ocha_relay's safe-send prompts the caller to
    # type the campaign name before sending (good for local/manual runs). A
    # headless run has no stdin and would hit EOFError on that prompt, so the
    # automated path (the Databricks wrapper sets LISTMONK_SKIP_CONFIRMATION=true)
    # skips it. Unset (local/manual) keeps the interactive confirmation.
    skip_confirmation = _parse_bool_env(
        "LISTMONK_SKIP_CONFIRMATION", default=False
    )
    client.send_campaign(cid, skip_confirmation=skip_confirmation)
    logger.info(f"Sent listmonk {kind} campaign {cid} for {monitor_id}")


def send_info_email(monitor_id: str, fcast_obsv: str) -> None:
    """Listmonk equivalent of send_emails.send_info_email."""
    d = prepare_email_data(monitor_id, fcast_obsv)
    test = "[TEST] " if (FORCE_ALERT or TEST_EMAIL) else ""
    subject = (
        f"{test}Acción anticipatoria Cuba – información sobre "
        f"{d['fcast_obsv_es']} {d['cyclone_name']} {d['pub_time']}, "
        f"{d['pub_date']}"
    )
    body = _env().get_template("informational.html").render(
        name=d["cyclone_name"],
        fcast_obsv=d["fcast_obsv_es"],
        fcast_obsv_en=d["fcast_obsv_en"],
        pub_time=d["pub_time"],
        pub_date=d["pub_date"],
        pub_date_en=d["pub_date_en"],
        pub_datetime_txt=d["pub_datetime_txt"],
        readiness=d["readiness"],
        action=d["action"],
        obsv=d["obsv"],
        readiness_en=d["readiness_en"],
        action_en=d["action_en"],
        obsv_en=d["obsv_en"],
        map_cid=_MAP_TOKEN,
        scatter_cid=_SCATTER_TOKEN,
        show_scatter_plot=d["show_scatter_plot"],
    )
    _send_campaign(subject, body, "info", "info", monitor_id, with_images=True)


def send_trigger_email(monitor_id: str, trigger_name: str) -> None:
    """Listmonk equivalent of send_emails.send_trigger_email."""
    fcast_obsv = "fcast" if trigger_name in ("readiness", "action") else "obsv"
    d = prepare_email_data(monitor_id, fcast_obsv)
    _, trigger_name_en, trigger_type_subj = _TRIGGER_LABELS[trigger_name]
    test = "[TEST] " if (FORCE_ALERT or TEST_EMAIL) else ""
    subject = (
        f"{test}Acción anticipatoria Cuba – Disparador "
        f"{trigger_type_subj} alcanzado"
    )
    template_name = "observational" if trigger_name == "obsv" else trigger_name
    body = _env().get_template(f"{template_name}.html").render(
        name=d["cyclone_name"],
        fcast_obsv=d["fcast_obsv_es"],
        fcast_obsv_en=d["fcast_obsv_en"],
        pub_time=d["pub_time"],
        pub_date=d["pub_date"],
        pub_date_en=d["pub_date_en"],
        trigger_name_en=trigger_name_en,
        test_email=FORCE_ALERT,
    )
    _send_campaign(
        subject, body, "trigger", trigger_name, monitor_id, with_images=False
    )
