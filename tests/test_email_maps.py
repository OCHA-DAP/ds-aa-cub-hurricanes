"""Tests for the per-language maps in the info email.

Each language block of the email embeds its own map. These tests pin the
plumbing between rendering, blob naming and the two send backends, since a
mix-up there shows the wrong-language map (or no map) to recipients.
"""

import os

import pytest
from azure.core.exceptions import ResourceNotFoundError
from jinja2 import Environment, FileSystemLoader

# src.email.utils reads SMTP settings at import time (and int-casts the port),
# so provide dummy values before importing the modules under test.
os.environ.setdefault("DSCI_AWS_EMAIL_HOST", "localhost")
os.environ.setdefault("DSCI_AWS_EMAIL_PORT", "587")
os.environ.setdefault("DSCI_AWS_EMAIL_PASSWORD", "x")
os.environ.setdefault("DSCI_AWS_EMAIL_USERNAME", "x")
os.environ.setdefault("DSCI_AWS_EMAIL_ADDRESS", "x@example.com")
os.environ["FORCE_ALERT"] = "false"

from src.email import listmonk_emails, plotting  # noqa: E402
from src.email.utils import TEMPLATES_DIR  # noqa: E402

MONITOR_ID = "al092026_fcast_2026-10-08T21:00:00"
ES_BLOB = (
    "ds-aa-cub-hurricanes/plots/2026/fcast/"
    "al092026_fcast_2026-10-08T21:00:00_map.png"
)
EN_BLOB = ES_BLOB.replace("_map.png", "_map_en.png")

INFO_CONTEXT = dict(
    name="ISAIAS",
    fcast_obsv="pronóstico",
    fcast_obsv_en="forecast",
    pub_time="17h00",
    pub_date="8 oct. 2026",
    pub_date_en="8 October 2026",
    pub_datetime_txt="8 de October 2026 a las 17h00",
    readiness="NO ALCANZADO",
    action="NO ALCANZADO",
    obsv="",
    readiness_en="NOT REACHED",
    action_en="NOT REACHED",
    obsv_en="",
    show_scatter_plot=False,
)


class FakeContainerClient:
    """Stands in for the blob container: serves `blobs`, 404s otherwise."""

    def __init__(self, blobs: dict):
        self.blobs = blobs
        self.requested = []

    def get_blob_client(self, blob_name):
        self.requested.append(blob_name)
        container = self

        class _Blob:
            def download_blob(self):
                if blob_name not in container.blobs:
                    raise ResourceNotFoundError(blob_name)
                data = container.blobs[blob_name]

                class _Download:
                    def readall(self):
                        return data

                return _Download()

        return _Blob()


@pytest.fixture
def fake_blobs(monkeypatch):
    def _make(blobs: dict) -> FakeContainerClient:
        client = FakeContainerClient(blobs)
        monkeypatch.setattr(
            plotting.stratus, "get_container_client", lambda: client
        )
        return client

    return _make


def test_spanish_map_keeps_original_blob_name():
    """The Spanish map name must not change: it is what update_plots checks
    to skip storms already plotted, so renaming it would re-render them all."""
    assert plotting.get_plot_blob_name(MONITOR_ID, "map") == ES_BLOB
    assert plotting.get_plot_blob_name(MONITOR_ID, "map", "es") == ES_BLOB


def test_english_map_has_its_own_blob_name():
    assert plotting.get_plot_blob_name(MONITOR_ID, "map", "en") == EN_BLOB


def test_load_plot_bytes_returns_requested_language(fake_blobs):
    fake_blobs({ES_BLOB: b"es", EN_BLOB: b"en"})
    assert plotting.load_plot_bytes(MONITOR_ID, "map", "es") == b"es"
    assert plotting.load_plot_bytes(MONITOR_ID, "map", "en") == b"en"


def test_load_plot_bytes_falls_back_to_spanish(fake_blobs):
    """A missing English map must not leave the email without a map."""
    fake_blobs({ES_BLOB: b"es"})
    assert plotting.load_plot_bytes(MONITOR_ID, "map", "en") == b"es"


def test_load_plot_bytes_raises_when_spanish_missing(fake_blobs):
    fake_blobs({})
    with pytest.raises(ResourceNotFoundError):
        plotting.load_plot_bytes(MONITOR_ID, "map", "es")
    with pytest.raises(ResourceNotFoundError):
        plotting.load_plot_bytes(MONITOR_ID, "map", "en")


def test_create_map_plot_saves_every_language(monkeypatch):
    rendered, saved = [], []

    def fake_figure(monitor_id, fcast_obsv, lang="es"):
        rendered.append(lang)
        return f"fig-{lang}"

    def fake_save(fig, monitor_id, plot_type, lang="es"):
        saved.append((fig, plot_type, lang))

    monkeypatch.setattr(plotting, "create_map_plot_figure", fake_figure)
    monkeypatch.setattr(plotting, "save_plot_to_blob", fake_save)

    plotting.create_map_plot(MONITOR_ID, "fcast")

    # Spanish first: its blob is the existence marker for update_plots.
    assert rendered == ["es", "en"]
    assert saved == [("fig-es", "map", "es"), ("fig-en", "map", "en")]


def test_map_text_has_same_structure_in_every_language():
    es, en = plotting.MAP_TEXT["es"], plotting.MAP_TEXT["en"]
    assert set(plotting.MAP_TEXT) == set(plotting.MAP_LANGS)
    assert es.keys() == en.keys()
    for fcast_obsv in ["fcast", "obsv"]:
        assert es[fcast_obsv].keys() == en[fcast_obsv].keys()
        es_legend = es[fcast_obsv]["legend"]
        en_legend = en[fcast_obsv]["legend"]
        # same symbols (fill, outline) in the same order
        assert [e[:2] for e in es_legend] == [e[:2] for e in en_legend]


@pytest.mark.parametrize("lang", plotting.MAP_LANGS)
@pytest.mark.parametrize("fcast_obsv", ["fcast", "obsv"])
def test_legend_states_the_wind_unit(lang, fcast_obsv):
    legend = plotting.MAP_TEXT[lang][fcast_obsv]["legend"]
    unit = {"es": "nudos (kt)", "en": "knots (kt)"}[lang]
    assert any(unit in text for _, _, text in legend)


def _split_language_blocks(html: str):
    """Split a rendered listmonk body into its Spanish and English cards."""
    es, en = html.split("English version")
    assert "Versión en español" in es
    return es, en


def test_each_language_block_embeds_its_own_map_listmonk():
    html = (
        listmonk_emails._env()
        .get_template("informational.html")
        .render(
            map_cid="MAP_ES",
            map_cid_en="MAP_EN",
            scatter_cid="SCATTER",
            **INFO_CONTEXT,
        )
    )
    es, en = _split_language_blocks(html)
    assert "cid:MAP_ES" in es and "cid:MAP_EN" not in es
    assert "cid:MAP_EN" in en and "cid:MAP_ES" not in en


def test_each_language_block_embeds_its_own_map_smtp():
    environment = Environment(loader=FileSystemLoader(TEMPLATES_DIR))
    html = environment.get_template("informational.html").render(
        map_cid="MAP_ES",
        map_cid_en="MAP_EN",
        scatter_cid="SCATTER",
        ocha_logo_cid="LOGO",
        **INFO_CONTEXT,
    )
    assert html.count("cid:MAP_ES") == 1
    assert html.count("cid:MAP_EN") == 1
    assert html.index("cid:MAP_ES") < html.index("cid:MAP_EN")


def test_listmonk_swap_gives_each_block_its_own_image(fake_blobs):
    fake_blobs({ES_BLOB: b"es", EN_BLOB: b"en"})

    class FakeListmonk:
        def __init__(self):
            self.uploads = []

        def upload_media(self, data, filename):
            self.uploads.append((data, filename))
            return f"https://media.example/{filename}"

    client = FakeListmonk()
    html = (
        f'<img src="cid:{listmonk_emails._MAP_ES_TOKEN}">'
        f'<img src="cid:{listmonk_emails._MAP_EN_TOKEN}">'
    )

    swapped = listmonk_emails._upload_and_swap(client, html, MONITOR_ID)

    assert client.uploads == [(b"es", "map_es.png"), (b"en", "map_en.png")]
    assert swapped == (
        '<img src="https://media.example/map_es.png">'
        '<img src="https://media.example/map_en.png">'
    )
