import json
from pathlib import Path
from types import SimpleNamespace

import pymupdf

from co_extract import docling_ocr, pipeline
from co_extract.extract import ClaudeExtractor, FixtureExtractor, document_prompt, wants_pdf
from co_extract.ingest import load

FIX = Path(__file__).parent / "fixtures"
OCR_TEXT = ("CHANGE ORDER No. CO-007\nProject: Lakeview Medical Office Building\n"
            "Total this Change Order $3,850.00\nThe Contract Sum including this Change Order will be $1,296,150.00")


def _blank_pdf(tmp_path, name="scan.pdf") -> Path:
    pdf = pymupdf.open()
    pdf.new_page()  # no text layer
    path = tmp_path / name
    pdf.save(path)
    return path


def _configure(monkeypatch, result=None, error=None, calls=None):
    monkeypatch.setenv("DOCLING_SERVICE_URL", "https://example.test/instance")
    monkeypatch.setenv("DOCLING_SERVICE_API_KEY", "test-key")

    def fake(pdf_bytes, page_nos, name="document.pdf"):
        if calls is not None:
            calls.append(page_nos)
        return ({p: OCR_TEXT for p in page_nos} if result is None else result), error
    monkeypatch.setattr(docling_ocr, "read_pages", fake)


def test_scanned_page_gets_ocr_text(tmp_path, monkeypatch):
    _configure(monkeypatch)
    doc = load(_blank_pdf(tmp_path))
    assert doc.ocr_pages == [1] and doc.scanned_pages == [] and doc.ocr_error is None
    assert "$3,850.00" in doc.text
    assert "were OCR'd" in document_prompt(doc) and "no text layer" not in document_prompt(doc)
    assert wants_pdf(doc, send_pdf=False)  # the model still gets the image alongside the text


def test_ocr_fields_are_grounded_and_score_above_a_plain_scan(tmp_path, monkeypatch):
    path = _blank_pdf(tmp_path, "simple_co.pdf")  # the fixture extractor is keyed by stem
    as_scan = pipeline.run(path, FixtureExtractor(FIX / "llm"))
    _configure(monkeypatch)
    as_ocr = pipeline.run(path, FixtureExtractor(FIX / "llm"))
    scan, ocr = as_scan["fields"]["total_change_amount"], as_ocr["fields"]["total_change_amount"]
    assert any("scanned page" in r for r in scan["reasons"])
    assert "page text from OCR" in ocr["reasons"] and ocr["confidence"] > scan["confidence"]
    assert as_ocr["meta"]["ocr_pages"] == [1]
    assert as_ocr["meta"]["ocr"] == {"engine": "docling-watsonx", "error": None}


def test_ocr_failure_falls_back_to_the_scan_path(tmp_path, monkeypatch):
    _configure(monkeypatch, result={}, error="UsageLimitExceededError: trial quota used")
    path = _blank_pdf(tmp_path, "simple_co.pdf")
    r = pipeline.run(path, FixtureExtractor(FIX / "llm"))
    assert r["meta"]["scanned_pages"] == [1] and r["meta"]["ocr_pages"] == []
    assert r["meta"]["ocr"]["error"].startswith("UsageLimitExceededError")
    assert any("scanned page" in x for x in r["fields"]["total_change_amount"]["reasons"])


def test_not_configured_never_calls_the_service(tmp_path, monkeypatch):
    calls = []
    _configure(monkeypatch, calls=calls)
    monkeypatch.delenv("DOCLING_SERVICE_URL")
    doc = load(_blank_pdf(tmp_path))
    assert calls == [] and doc.scanned_pages == [1] and doc.ocr_pages == []


def test_text_documents_never_call_the_service(tmp_path, monkeypatch):
    calls = []
    _configure(monkeypatch, calls=calls)
    load(FIX / "docs" / "simple_co.txt")
    text_pdf = pymupdf.open()
    text_pdf.new_page().insert_text((72, 72), "CHANGE ORDER No. CO-007 Total this Change Order $3,850.00")
    text_pdf.save(tmp_path / "text.pdf")
    load(tmp_path / "text.pdf")
    assert calls == []


def test_pipeline_reports_ocr_as_part_of_reading(tmp_path, monkeypatch):
    _configure(monkeypatch)
    steps = []
    pipeline.run(_blank_pdf(tmp_path, "simple_co.pdf"), FixtureExtractor(FIX / "llm"),
                 on_step=lambda name, **info: steps.append((name, info.get("ocr"))))
    assert steps[:3] == [("read", None), ("read", True), ("extract", None)]


def test_page_texts_keeps_reading_order_and_table_rows():
    class Table:
        def __init__(self, page):
            self.prov = [SimpleNamespace(page_no=page)]

        def export_to_markdown(self, doc=None):
            return "| Relocate 4\" sanitary line | 1 | LS | $2,480.00 |"

    para = lambda text, page: SimpleNamespace(text=text, prov=[SimpleNamespace(page_no=page)])  # noqa: E731
    doc = SimpleNamespace(iterate_items=lambda: iter([(para("CHANGE ORDER No. 7", 1), 0), (Table(1), 0),
                                                       (para("Page two", 2), 0), (para("skipped", 3), 0)]))
    out = docling_ocr.page_texts(doc, [1, 2])
    assert out == {1: 'CHANGE ORDER No. 7\n| Relocate 4" sanitary line | 1 | LS | $2,480.00 |', 2: "Page two"}


def test_read_pages_reports_errors_instead_of_raising(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("job took too long")
    monkeypatch.setattr(docling_ocr, "_convert", boom)
    assert docling_ocr.read_pages(b"%PDF", [1]) == ({}, "TimeoutError: job took too long")


def test_claude_receives_image_and_ocr_text(tmp_path, monkeypatch):
    import httpx2 as httpx
    import anthropic

    _configure(monkeypatch)
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "m", "type": "message", "role": "assistant", "model": "claude-test",
                                         "content": [{"type": "text", "text": (FIX / "llm" / "simple_co.json").read_text()}],
                                         "stop_reason": "end_turn", "stop_sequence": None,
                                         "usage": {"input_tokens": 1, "output_tokens": 1}})
    client = anthropic.Anthropic(api_key="t", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)
    ClaudeExtractor(client=client, send_pdf=False).extract(load(_blank_pdf(tmp_path)))
    blocks = captured[0]["messages"][0]["content"]
    assert blocks[0]["type"] == "document" and "$3,850.00" in blocks[-1]["text"]
