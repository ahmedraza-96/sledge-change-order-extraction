import pytest


@pytest.fixture(autouse=True)
def no_hosted_ocr(request, monkeypatch):
    """Offline tests must never reach IBM's Docling service, even when .env configures it.

    Tests that exercise OCR monkeypatch ``docling_ocr`` themselves; live tests are left alone.
    """
    if request.node.get_closest_marker("live"):
        return
    monkeypatch.delenv("DOCLING_SERVICE_URL", raising=False)
    monkeypatch.delenv("DOCLING_SERVICE_API_KEY", raising=False)
    monkeypatch.delenv("DOCLING_OCR_ENABLED", raising=False)
