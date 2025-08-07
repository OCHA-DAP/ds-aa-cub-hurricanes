#!/usr/bin/env python3
"""
Enhanced email sender that embeds images to avoid "download external images" prompts.
This script provides multiple approaches to handle images in HTML emails.
"""

import os
import sys
import base64
import mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
import ssl
from pathlib import Path


def embed_images_as_base64(html_content, image_mappings):
    """
    Embed images as base64 data URLs in HTML content.

    Args:
        html_content (str): HTML content with placeholder variables
        image_mappings (dict): Map of placeholder -> image file path

    Returns:
        str: HTML with embedded base64 images
    """
    for placeholder, image_path in image_mappings.items():
        try:
            if not os.path.exists(image_path):
                print(f"⚠️ Image not found: {image_path}")
                continue

            with open(image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("utf-8")

            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith("image/"):
                mime_type = "image/png"  # default

            # Create data URL
            data_url = f"data:{mime_type};base64,{img_data}"

            # Replace placeholder with embedded image
            html_content = html_content.replace(f"${placeholder}$", data_url)
            print(f"✅ Embedded {placeholder} from {image_path}")

        except Exception as e:
            print(f"⚠️ Could not embed {placeholder}: {e}")

    return html_content


def create_email_with_embedded_images(
    html_content, subject, sender, recipients, image_attachments=None
):
    """
    Create email with images as CID attachments (alternative approach).

    Args:
        html_content (str): HTML content
        subject (str): Email subject
        sender (str): Sender email
        recipients (str): Comma-separated recipients
        image_attachments (dict): Map of cid -> image file path

    Returns:
        MIMEMultipart: Email message with embedded images
    """
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipients

    # Create alternative container for text/html
    msg_alternative = MIMEMultipart("alternative")
    msg.attach(msg_alternative)

    # Add HTML part
    html_part = MIMEText(html_content, "html")
    msg_alternative.attach(html_part)

    # Add image attachments with CID
    if image_attachments:
        for cid, image_path in image_attachments.items():
            try:
                with open(image_path, "rb") as img_file:
                    img_data = img_file.read()

                # Determine image type
                mime_type, _ = mimetypes.guess_type(image_path)
                if mime_type and mime_type.startswith("image/"):
                    img_mime = MIMEImage(img_data)
                    img_mime.add_header("Content-ID", f"<{cid}>")
                    img_mime.add_header("Content-Disposition", "inline")
                    msg.attach(img_mime)
                    print(f"✅ Attached {cid} from {image_path}")

            except Exception as e:
                print(f"⚠️ Could not attach {cid}: {e}")

    return msg


def send_email_ssl(msg, smtp_server, smtp_port, username, password):
    """Send email using SSL/TLS."""
    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL(
            smtp_server, smtp_port, context=context
        ) as server:
            server.login(username, password)
            text = msg.as_string()
            server.sendmail(msg["From"], msg["To"].split(","), text)
        print("✅ Email sent successfully")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        raise


# Example usage function
def send_hurricane_report():
    """Render Quarto document and send hurricane report with images."""
    import subprocess

    # Email configuration from existing email utilities
    from src.email.utils import (
        EMAIL_HOST,
        EMAIL_PORT,
        EMAIL_USERNAME,
        EMAIL_PASSWORD,
        EMAIL_ADDRESS,
    )
    from src.constants import PROJECT_PREFIX
    import ocha_stratus as stratus

    SMTP_SERVER = EMAIL_HOST
    SMTP_PORT = EMAIL_PORT
    EMAIL_USER = EMAIL_USERNAME
    EMAIL_PASS = EMAIL_PASSWORD

    # Load distribution list from blob storage
    print("📧 Loading email distribution list...")
    try:
        blob_name = f"{PROJECT_PREFIX}/email/test_distribution_list.csv"
        df_distribution = stratus.load_csv_from_blob(blob_name)
        df_distribution = df_distribution[
            df_distribution["daily_summary"].notna()
        ]
        email_list = df_distribution["email"].tolist()
        EMAIL_TO = ", ".join(email_list)
        print(f"✅ Loaded {len(email_list)} emails from distribution list")
    except Exception as e:
        print(f"⚠️ Could not load distribution list: {e}")
        print("📧 Falling back to default test email")
        EMAIL_TO = "zachary.arno@un.org"

    EMAIL_FROM = EMAIL_ADDRESS
    SUBJECT = "Daily Hurricane Summary - Atlantic Basin"

    # Step 1: Render Quarto document
    print("📊 Rendering Quarto document...")
    try:
        subprocess.run(
            ["quarto", "render", "Report.qmd", "--to", "html"],
            capture_output=True,
            text=True,
            check=True,
        )
        print("✅ Quarto document rendered successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to render Quarto document: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return
    except FileNotFoundError:
        print("❌ Quarto not found. Please install Quarto CLI.")
        return

    # Step 2: Load HTML report
    html_file = "Report.html"
    if not os.path.exists(html_file):
        print(f"❌ HTML file not found: {html_file}")
        return

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Method 1: Embed as base64 (recommended for most email clients)
    # Handle the CHD logo embedding
    logo_path = "src/email/static/centre_banner_white.png"
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("utf-8")

            # Create data URL
            data_url = f"data:image/png;base64,{img_data}"

            # Replace the logo reference with base64 embedded version
            html_content = html_content.replace(
                f'src="{logo_path}"', f'src="{data_url}"'
            )
            print(f"✅ Embedded CHD logo from {logo_path}")
        except Exception as e:
            print(f"⚠️ Could not embed CHD logo: {e}")
    else:
        print(f"⚠️ CHD logo not found: {logo_path}")

    # Also embed the scraped NHC outlook image if it exists
    nhc_image_path = "temp_nhc_outlook.png"
    if os.path.exists(nhc_image_path):
        try:
            with open(nhc_image_path, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode("utf-8")

            # Replace the image reference with base64 embedded version
            data_url = f"data:image/png;base64,{img_data}"
            html_content = html_content.replace(
                f'src="{nhc_image_path}"', f'src="{data_url}"'
            )
            html_content = html_content.replace(
                'src="temp_nhc_outlook.png"', f'src="{data_url}"'
            )
            print(f"✅ Embedded NHC outlook image from {nhc_image_path}")
        except Exception as e:
            print(f"⚠️ Could not embed NHC outlook image: {e}")
    else:
        print(f"⚠️ NHC outlook image not found: {nhc_image_path}")

    # Create and send email
    msg = MIMEMultipart("alternative")
    msg["Subject"] = SUBJECT
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg.attach(MIMEText(html_content, "html"))

    send_email_ssl(msg, SMTP_SERVER, SMTP_PORT, EMAIL_USER, EMAIL_PASS)


if __name__ == "__main__":
    send_hurricane_report()
