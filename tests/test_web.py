from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from co_extract import web

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def client():
    return TestClient(web.app)


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Change Order Extractor" in r.text


def test_providers_list_includes_demo(client):
    ids = {p["id"]: p for p in client.get("/api/providers").json()}
    assert set(ids) == {"claude", "gemini", "openai", "openrouter", "demo"}
    assert ids["demo"]["available"]


def test_upload_demo(client):
    doc = (FIX / "docs" / "simple_co.txt").read_bytes()
    r = client.post("/api/extract", data={"provider": "demo"}, files={"file": ("simple_co.txt", doc, "text/plain")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "simple_co.txt"
    assert body["fields"]["total_change_amount"]["value"] == "3850.00"
    assert 0 <= body["fields"]["total_change_amount"]["confidence"] <= 1


def test_pasted_sample_runs_repair(client):
    text = client.get("/api/samples/misread_co").text
    r = client.post("/api/extract", data={"provider": "demo", "text": text, "name": "misread_co"})
    assert r.status_code == 200, r.text
    assert r.json()["meta"]["llm_calls"] == 2


def test_empty_request_rejected(client):
    assert client.post("/api/extract", data={"provider": "demo"}).status_code == 400


def test_unsupported_suffix_rejected(client):
    r = client.post("/api/extract", data={"provider": "demo"}, files={"file": ("co.docx", b"x", "application/octet-stream")})
    assert r.status_code == 400


def test_demo_unknown_document(client):
    r = client.post("/api/extract", data={"provider": "demo", "text": "CHANGE ORDER No. 9"})
    assert r.status_code == 502
    assert "Demo mode only has recorded output" in r.json()["detail"]


def test_provider_without_key_rejected(client, monkeypatch):
    monkeypatch.setattr(web, "has_key", lambda p: False)
    r = client.post("/api/extract", data={"provider": "openai", "text": "CHANGE ORDER No. 9"})
    assert r.status_code == 400
    assert "No API key" in r.json()["detail"]


def test_image_upload_accepted(client):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (200, 100), "white").save(buf, format="PNG")
    r = client.post("/api/extract", data={"provider": "demo"}, files={"file": ("photo.png", buf.getvalue(), "image/png")})
    # Not rejected as an unsupported type; demo mode just has no recording for it.
    assert r.status_code == 502
    assert "Demo mode only has recorded output" in r.json()["detail"]
