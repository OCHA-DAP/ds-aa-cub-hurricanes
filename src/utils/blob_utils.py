from typing import Literal

import ocha_stratus as stratus
from azure.storage.blob import ContentSettings


def _upload_blob_data(
    data,
    blob_name,
    stage: Literal["prod", "dev"] = "dev",
    container_name: str = "projects",
    content_type: str = None,
):
    """
    Internal function to upload raw data to Azure Blob Storage.

    Parameters
    ----------
    data : bytes or BinaryIO
        Data to upload
    blob_name : str
        Name of the blob to create/update
    stage : Literal["prod", "dev"], optional
        Environment stage to upload to, by default "dev"
    container_name : str, optional
        Name of the container to upload to, by default "projects"
    content_type : str, optional
        MIME type of the content, by default None
    """
    container_client = stratus.get_container_client(
        stage=stage, container_name=container_name, write=True
    )

    if content_type is None:
        content_settings = ContentSettings(
            content_type="application/octet-stream"
        )
    else:
        content_settings = ContentSettings(content_type=content_type)

    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(
        data, overwrite=True, content_settings=content_settings
    )
