import pytest

from co_extract import ingest


@pytest.fixture(autouse=True)
def no_real_ocr(request, monkeypatch):
    """Keep offline tests fast and deterministic whether or not docling is installed.

    Live tests are left alone so they exercise whatever the environment provides.
    """
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.setattr(ingest, "_ocr_pages", lambda path, pages: {})
