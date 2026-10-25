import json
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
    assert "Change order review" in r.text


def test_providers_list_includes_demo(client):
    ids = {p["id"]: p for p in client.get("/api/providers").json()}
    assert set(ids) == {"claude", "gemini", "openai", "openrouter", "luna", "demo"}
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




def test_provider_is_picked_automatically(client, monkeypatch):
    picked = []
    monkeypatch.setattr(web, "has_key", lambda p: p is web.Provider.gemini)
    monkeypatch.setattr(web, "make_extractor", lambda p, model=None: picked.append(p) or web.FixtureExtractor(FIX / "llm"))
    doc = (FIX / "docs" / "simple_co.txt").read_bytes()
    r = client.post("/api/extract", files={"file": ("simple_co.txt", doc, "text/plain")})
    assert r.status_code == 200 and picked == [web.Provider.gemini]


def test_luna_is_the_default_and_nemotron_the_menu_alternative(client, monkeypatch):
    picked = []
    monkeypatch.setattr(web, "has_key", lambda p: True)
    monkeypatch.setattr(web, "make_extractor", lambda p, model=None: picked.append(p) or web.FixtureExtractor(FIX / "llm"))
    doc = (FIX / "docs" / "simple_co.txt").read_bytes()
    r = client.post("/api/extract", files={"file": ("simple_co.txt", doc, "text/plain")})
    assert r.status_code == 200 and picked == [web.Provider.luna]
    menu = [p["id"] for p in client.get("/api/providers").json() if p["in_menu"]]
    assert menu == ["luna", "openrouter"]


def test_no_configured_provider_explains_how_to_fix(client, monkeypatch):
    monkeypatch.setattr(web, "has_key", lambda p: False)
    r = client.post("/api/extract", data={"text": "CHANGE ORDER No. 9"})
    assert r.status_code == 400 and "OPENROUTER_API_KEY" in r.json()["detail"]


def test_stream_reports_real_steps_then_the_result(client):
    import json

    doc = (FIX / "docs" / "misread_co.txt").read_bytes()
    r = client.post("/api/extract/stream", data={"provider": "demo"},
                    files={"file": ("misread_co.txt", doc, "text/plain")})
    events = [json.loads(line) for line in r.text.splitlines()]
    steps = [e["step"] for e in events if "step" in e]
    # misread_co fails its checks on the first read, so the repair step appears.
    # Demo mode replays recorded verifier answers too.
    assert steps == ["read", "extract", "validate", "repair", "score", "verify"]
    assert events[[e.get("step") for e in events].index("repair")]["checks"]
    assert "result" in events[-1] and events[-1]["result"]["fields"]["total_change_amount"]["value"]


def test_stream_reports_extraction_errors(client):
    import json

    r = client.post("/api/extract/stream", data={"provider": "demo", "text": "CHANGE ORDER No. 9"})
    events = [json.loads(line) for line in r.text.splitlines()]
    assert "Demo mode only has recorded output" in events[-1]["error"]


# ---------- history ----------

def test_extraction_is_recorded_in_history(client):
    doc = (FIX / "docs" / "simple_co.txt").read_bytes()
    body = client.post("/api/extract", data={"provider": "demo"},
                       files={"file": ("simple_co.txt", doc, "text/plain")}).json()
    run_id = body["history_id"]
    [summary] = client.get("/api/history").json()
    assert summary["id"] == run_id and summary["filename"] == "simple_co.txt"
    assert summary["source"] == "file" and summary["status"] == "ok"
    run = client.get(f"/api/history/{run_id}").json()
    assert run["result"]["fields"]["total_change_amount"]["value"] == "3850.00"
    stored = client.get(f"/api/history/{run_id}/document")
    assert stored.status_code == 200 and stored.content == doc
    assert stored.headers["content-type"].startswith("text/plain")
    assert stored.headers["x-content-type-options"] == "nosniff"


def test_streamed_extraction_is_recorded(client):
    text = client.get("/api/samples/simple_co").text
    r = client.post("/api/extract/stream", data={"provider": "demo", "text": text, "name": "simple_co"})
    final = [json.loads(line) for line in r.text.splitlines() if line.strip()][-1]
    run = client.get(f"/api/history/{final['result']['history_id']}").json()
    assert run["source"] == "text" and run["status"] == "ok"


def test_failed_extraction_is_recorded(client):
    r = client.post("/api/extract", data={"provider": "demo", "text": "CHANGE ORDER No. 9"})
    assert r.status_code == 502
    [summary] = client.get("/api/history").json()
    assert summary["status"] == "error"
    assert "Demo mode only has recorded output" in summary["error"]
    assert client.get(f"/api/history/{summary['id']}/document").content == b"CHANGE ORDER No. 9"


def test_unknown_run_is_404(client):
    assert client.get("/api/history/999").status_code == 404
    assert client.get("/api/history/999/document").status_code == 404


def test_samples_are_gone_from_the_page(client):
    assert 'id="samples"' not in client.get("/").text


def _one_run(client) -> int:
    text = client.get("/api/samples/simple_co").text
    return client.post("/api/extract", data={"provider": "demo", "text": text, "name": "simple_co"}).json()["history_id"]


def test_delete_needs_the_admin_key(client, monkeypatch):
    monkeypatch.setenv("HISTORY_ADMIN_KEY", "s3cret")
    run_id = _one_run(client)
    assert client.delete(f"/api/history/{run_id}").status_code == 403
    assert client.delete(f"/api/history/{run_id}", headers={"X-Admin-Key": "wrong"}).status_code == 403
    assert client.get(f"/api/history/{run_id}").status_code == 200
    r = client.delete(f"/api/history/{run_id}", headers={"X-Admin-Key": "s3cret"})
    assert r.status_code == 200 and r.json() == {"deleted": run_id}
    assert client.get(f"/api/history/{run_id}").status_code == 404
    assert client.delete(f"/api/history/{run_id}", headers={"X-Admin-Key": "s3cret"}).status_code == 404


def test_delete_is_off_without_a_configured_key(client, monkeypatch):
    monkeypatch.setattr(web, "load_dotenv", lambda: None)  # ignore a developer's real .env
    monkeypatch.delenv("HISTORY_ADMIN_KEY", raising=False)
    run_id = _one_run(client)
    assert client.delete(f"/api/history/{run_id}", headers={"X-Admin-Key": ""}).status_code == 403
    assert client.get(f"/api/history/{run_id}").status_code == 200


def test_pdf_from_history_is_served_as_a_pdf(client):
    pdf = (Path(__file__).parents[1] / "test_samples" / "1_original_samples" / "01_aia_clean.pdf").read_bytes()
    run_id = history_record_pdf(pdf)
    r = client.get(f"/api/history/{run_id}/document")
    assert r.headers["content-type"] == "application/pdf" and r.content == pdf


def history_record_pdf(pdf: bytes) -> int:
    from co_extract import history
    return history.record(filename="01_aia_clean.pdf", source="file", document=pdf, provider="demo",
                          result={"document_confidence": 1.0, "needs_review": [], "issues": [], "meta": {}})
