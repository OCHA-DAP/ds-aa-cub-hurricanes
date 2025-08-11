"""
Email preview functions for testing email templates without sending them.
"""

import base64
import io
from typing import Literal
from email.utils import make_msgid

import pandas as pd
import pytz
from html2text import html2text
from jinja2 import Environment, FileSystemLoader

from src.constants import SPANISH_MONTHS
from src.email.plotting import get_plot_blob_name
from src.email.send_emails import create_info_email_content
from src.email.utils import (
    EMAIL_DISCLAIMER,
    STATIC_DIR,
    TEMPLATES_DIR,
    FORCE_ALERT,
    DRY_RUN,
    TEST_EMAIL,
    load_monitoring_data,
)
from src.monitoring import monitoring_utils
import ocha_stratus as stratus


def show_environment_status():
    """Display current environment variable settings for debugging."""
    print("🔧 Environment Variables:")
    print(f"   DRY_RUN: {DRY_RUN}")
    print(f"   TEST_EMAIL: {TEST_EMAIL}")
    print(f"   FORCE_ALERT: {FORCE_ALERT}")


def get_image_as_base64(
    image_path_or_blob_name: str, is_blob: bool = False
) -> str:
    """
    Load an image and convert it to base64 data URL for embedding in HTML.

    Parameters:
    -----------
    image_path_or_blob_name : str
        Path to local image file or blob name for cloud storage
    is_blob : bool
        If True, load from blob storage; if False, load from local file

    Returns:
    --------
    str: Base64 data URL string (e.g., "data:image/png;base64,...")
    """
    try:
        if is_blob:
            # Load from blob storage
            container_client = stratus.get_container_client()
            blob_client = container_client.get_blob_client(
                image_path_or_blob_name
            )
            image_data = io.BytesIO()
            blob_client.download_blob().download_to_stream(image_data)
            image_data.seek(0)
            image_bytes = image_data.read()
        else:
            # Load from local file
            with open(image_path_or_blob_name, "rb") as f:
                image_bytes = f.read()

        # Convert to base64
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/png;base64,{base64_str}"
    except Exception as e:
        print(f"Warning: Could not load image {image_path_or_blob_name}: {e}")
        # Return a placeholder data URL for a small transparent PNG
        placeholder = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        return f"data:image/png;base64,{placeholder}"


def preview_info_email(
    monitor_id: str,
    fcast_obsv: Literal["fcast", "obsv"],
    save_to_file: bool = True,
):
    """
    Preview an info email without sending it.
    Uses the actual email creation logic from send_emails.py.

    Parameters:
    -----------
    monitor_id : str
        The monitor ID for the email
    fcast_obsv : Literal["fcast", "obsv"]
        Whether it's a forecast or observation email
    save_to_file : bool
        If True, saves HTML to a file you can open in a browser

    Returns:
    --------
    dict: Contains 'html', 'text', 'subject', and other email details
    """
    # Show environment status for debugging
    show_environment_status()

    # Use the simplified email creation logic
    email_content = create_info_email_content(
        monitor_id, fcast_obsv, for_preview=True
    )

    # Get the HTML and replace placeholder CIDs with base64 data URLs for browser viewing
    html_str = email_content["html"]

    try:
        # Load plot images from blob storage
        map_blob_name = get_plot_blob_name(monitor_id, "map")
        scatter_blob_name = get_plot_blob_name(monitor_id, "scatter")

        map_data_url = get_image_as_base64(map_blob_name, is_blob=True)
        scatter_data_url = get_image_as_base64(scatter_blob_name, is_blob=True)

        # Load static images from local files
        chd_banner_path = STATIC_DIR / "centre_banner.png"
        ocha_logo_path = STATIC_DIR / "ocha_logo_wide.png"

        chd_banner_data_url = get_image_as_base64(
            str(chd_banner_path), is_blob=False
        )
        ocha_logo_data_url = get_image_as_base64(
            str(ocha_logo_path), is_blob=False
        )

    except Exception as e:
        print(f"Warning: Could not load images: {e}")
        # Use placeholder for plot images but load actual static images
        placeholder_url = get_image_as_base64("placeholder", is_blob=False)
        map_data_url = scatter_data_url = placeholder_url

        # Load actual static images even if plot images failed
        chd_banner_path = STATIC_DIR / "centre_banner.png"
        ocha_logo_path = STATIC_DIR / "ocha_logo_wide.png"
        chd_banner_data_url = get_image_as_base64(
            str(chd_banner_path), is_blob=False
        )
        ocha_logo_data_url = get_image_as_base64(
            str(ocha_logo_path), is_blob=False
        )

    # Replace placeholder CID references with direct data URLs for browser viewing
    html_str = html_str.replace(
        'src="cid:PREVIEW_MAP_PLACEHOLDER"', f'src="{map_data_url}"'
    )
    html_str = html_str.replace(
        'src="cid:PREVIEW_SCATTER_PLACEHOLDER"', f'src="{scatter_data_url}"'
    )
    html_str = html_str.replace(
        'src="cid:PREVIEW_BANNER_PLACEHOLDER"', f'src="{chd_banner_data_url}"'
    )
    html_str = html_str.replace(
        'src="cid:PREVIEW_LOGO_PLACEHOLDER"', f'src="{ocha_logo_data_url}"'
    )

    # Save to file if requested
    if save_to_file:
        filename = f"email_preview_{monitor_id}_{fcast_obsv}_info.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"Email preview saved to: {filename}")
        print(
            f"Open this file in your browser to see how the email will look!"
        )

    # Print summary using the email data
    email_data = email_content["email_data"]
    print(f"\n📧 EMAIL PREVIEW SUMMARY:")
    print(f"Subject: {email_content['subject']}")
    print(f"Storm: {email_data['cyclone_name']}")
    print(f"Time: {email_data['pub_time']}, {email_data['pub_date']}")
    print(f"Readiness: {email_data['readiness']}")
    print(f"Action: {email_data['action']}")
    print(f"Observation: {email_data['obsv']}")
    print(f"Recipients (TO): Preview mode - distribution list not loaded")
    print(f"Recipients (CC): Preview mode - distribution list not loaded")

    # Return preview data
    return {
        "subject": email_content["subject"],
        "html": html_str,
        "text": email_content["text"],
        "to_emails": [],
        "cc_emails": [],
        "cyclone_name": email_data["cyclone_name"],
        "pub_time": email_data["pub_time"],
        "pub_date": email_data["pub_date"],
        "readiness": email_data["readiness"],
        "action": email_data["action"],
        "obsv": email_data["obsv"],
    }


def preview_trigger_email(
    monitor_id: str,
    trigger_name: str,
    save_to_file: bool = True,
):
    """
    Preview a trigger email without sending it.

    Parameters:
    -----------
    monitor_id : str
        The monitor ID for the email
    trigger_name : str
        The trigger type ("readiness", "action", or "obsv")
    save_to_file : bool
        If True, saves HTML to a file you can open in a browser

    Returns:
    --------
    dict: Contains 'html', 'text', 'subject', and other email details
    """
    # Show environment status for debugging
    show_environment_status()

    fcast_obsv = "fcast" if trigger_name in ["readiness", "action"] else "obsv"

    # Load monitoring data using the simplified API
    df_monitoring = load_monitoring_data(fcast_obsv)
    monitoring_point = df_monitoring.set_index("monitor_id").loc[monitor_id]
    cuba_tz = pytz.timezone("America/Havana")
    cyclone_name = monitoring_point["name"]
    # Convert to scalar datetime by accessing the underlying value
    issue_time_raw = monitoring_point["issue_time"]
    if hasattr(issue_time_raw, "to_pydatetime"):
        issue_time = issue_time_raw.to_pydatetime()
    else:
        issue_time = pd.to_datetime(issue_time_raw).to_pydatetime()
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

    # For preview, skip distribution list entirely - we'll show generic counts
    to_list = None
    cc_list = None

    test_subject = "PREVIEW: " if not FORCE_ALERT else "TEST PREVIEW: "

    environment = Environment(loader=FileSystemLoader(TEMPLATES_DIR))

    template_name = "observational" if trigger_name == "obsv" else trigger_name
    template = environment.get_template(f"{template_name}.html")

    subject = (
        f"{test_subject}Acción anticipatoria Cuba – "
        f"activador {trigger_name_es} alcanzado para "
        f"{cyclone_name}"
    )

    # Load images and convert to base64 data URLs for browser viewing
    try:
        # Load static images from local files
        chd_banner_path = STATIC_DIR / "centre_banner.png"
        ocha_logo_path = STATIC_DIR / "ocha_logo_wide.png"

        chd_banner_data_url = get_image_as_base64(
            str(chd_banner_path), is_blob=False
        )
        ocha_logo_data_url = get_image_as_base64(
            str(ocha_logo_path), is_blob=False
        )

    except Exception as e:
        print(f"Warning: Could not load images: {e}")
        # Still try to load actual static images even if there's an error
        chd_banner_path = STATIC_DIR / "centre_banner.png"
        ocha_logo_path = STATIC_DIR / "ocha_logo_wide.png"

        try:
            chd_banner_data_url = get_image_as_base64(
                str(chd_banner_path), is_blob=False
            )
            ocha_logo_data_url = get_image_as_base64(
                str(ocha_logo_path), is_blob=False
            )
        except Exception:
            # Only use placeholder if static images also fail
            placeholder_url = get_image_as_base64("placeholder", is_blob=False)
            chd_banner_data_url = ocha_logo_data_url = placeholder_url

    html_str = template.render(
        name=cyclone_name,
        pub_time=pub_time,
        pub_date=pub_date,
        fcast_obsv=fcast_obsv_es,
        test_email=True,  # Always show as test for preview
        email_disclaimer=EMAIL_DISCLAIMER,
        chd_banner_cid="PREVIEW_BANNER_PLACEHOLDER",
        ocha_logo_cid="PREVIEW_LOGO_PLACEHOLDER",
    )

    # Replace placeholder CID references with direct data URLs for browser viewing
    html_str = html_str.replace(
        'src="cid:PREVIEW_BANNER_PLACEHOLDER"', f'src="{chd_banner_data_url}"'
    )
    html_str = html_str.replace(
        'src="cid:PREVIEW_LOGO_PLACEHOLDER"', f'src="{ocha_logo_data_url}"'
    )

    # Convert HTML to text
    text_str = html2text(html_str)

    # Save to file if requested
    if save_to_file:
        filename = f"email_preview_{monitor_id}_{trigger_name}.html"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html_str)
        print(f"Email preview saved to: {filename}")
        print(
            f"Open this file in your browser to see how the email will look!"
        )

    # Print summary
    print(f"\n🚨 TRIGGER EMAIL PREVIEW SUMMARY:")
    print(f"Subject: {subject}")
    print(f"Storm: {cyclone_name}")
    print(f"Trigger: {trigger_name_es}")
    print(f"Time: {pub_time}, {pub_date}")
    print(f"Recipients (TO): Preview mode - distribution list not loaded")
    print(f"Recipients (CC): Preview mode - distribution list not loaded")

    # Return preview data
    return {
        "subject": subject,
        "html": html_str,
        "text": text_str,
        "to_emails": [],
        "cc_emails": [],
        "cyclone_name": cyclone_name,
        "pub_time": pub_time,
        "pub_date": pub_date,
        "trigger_name": trigger_name_es,
    }


def list_available_monitors():
    """List available monitoring points for testing."""
    try:
        # Create monitor instance to access data
        monitor = monitoring_utils.create_cuba_hurricane_monitor()

        # Load existing monitoring data to see what's available
        obsv_data = monitor._load_existing_monitoring("obsv")
        fcast_data = monitor._load_existing_monitoring("fcast")

        print("Available observation monitoring points:")
        if not obsv_data.empty:
            print(f"  {len(obsv_data)} points available")
            print(
                f"  Latest date: {obsv_data['date'].max() if 'date' in obsv_data else 'No date column'}"
            )
        else:
            print("  No observation data found")

        print("\nAvailable forecast monitoring points:")
        if not fcast_data.empty:
            print(f"  {len(fcast_data)} points available")
            print(
                f"  Latest date: {fcast_data['date'].max() if 'date' in fcast_data else 'No date column'}"
            )
        else:
            print("  No forecast data found")

        return obsv_data, fcast_data

    except Exception as e:
        print(f"Error loading monitoring data: {e}")
        return None, None


if __name__ == "__main__":
    print("📧 Email Preview Tool")
    print("====================")
    print("This tool helps you preview emails without sending them.")
    print("\nExample usage:")
    print(
        "from src.email.preview import preview_info_email, preview_trigger_email, list_available_monitors"
    )
    print()
    print("# List available monitor IDs")
    print("list_available_monitors('fcast')")
    print()
    print("# Preview an info email")
    print("preview_info_email('some_monitor_id', 'fcast')")
    print()
    print("# Preview a trigger email")
    print("preview_trigger_email('some_monitor_id', 'readiness')")
