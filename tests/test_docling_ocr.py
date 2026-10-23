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
    monkeypatch.setenv("DOCLING_OCR_ENABLED", "true")

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
    assert "read by OCR" in document_prompt(doc) and "no text layer" not in document_prompt(doc)
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


def _text_pdf(tmp_path) -> Path:
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "CHANGE ORDER No. CO-007 Total this Change Order $3,850.00")
    path = tmp_path / "text.pdf"
    pdf.save(path)
    return path


def test_every_kind_of_document_goes_through_docling(tmp_path, monkeypatch):
    calls = []
    _configure(monkeypatch, calls=calls)
    eml = tmp_path / "co.eml"
    eml.write_text("From: a@b.com\nTo: c@d.com\nSubject: CO 9\n\nChange order 9, total add of $8,740.50.\n")
    for path in (_text_pdf(tmp_path), FIX / "docs" / "simple_co.txt", eml):
        doc = load(path)
        assert doc.ocr_pages == [1] and "$3,850.00" in doc.text  # the OCR text replaced the native text
    assert calls == [[1], [1], [1]]


def test_docling_failure_keeps_the_native_text(tmp_path, monkeypatch):
    _configure(monkeypatch, result={}, error="ServiceUnavailableError: 503")
    pdf = load(_text_pdf(tmp_path))
    assert pdf.ocr_pages == [] and pdf.scanned_pages == [] and "CO-007" in pdf.text
    assert pdf.ocr_error.startswith("ServiceUnavailableError")
    txt = load(FIX / "docs" / "simple_co.txt")
    assert txt.ocr_pages == [] and "Northgate" in txt.text.title() and txt.ocr_error


def test_text_documents_keep_their_text_unless_every_page_is_read(tmp_path, monkeypatch):
    long_text = tmp_path / "long.txt"
    long_text.write_text("\n".join(f"line {i} of a long change order" for i in range(200)))  # several pages
    _configure(monkeypatch, result={1: OCR_TEXT})  # only page 1 comes back
    doc = load(long_text)
    assert doc.ocr_pages == [] and "line 199" in doc.text and "1 of" in doc.ocr_error


def test_text_to_pdf_round_trip():
    from co_extract.ingest import text_to_pdf

    pdf = pymupdf.open(stream=text_to_pdf(["CHANGE ORDER No. 7\nTotal $3,850.00"]), filetype="pdf")
    assert len(pdf) == 1 and "Total $3,850.00" in pdf[0].get_text()


def test_empty_docling_result_is_reported(monkeypatch):
    monkeypatch.setattr(docling_ocr, "_convert", lambda *a, **k: SimpleNamespace(iterate_items=lambda: iter([])))
    assert docling_ocr.read_pages(b"%PDF", [1]) == ({}, "Docling returned no text for these pages")


def test_convert_builds_a_real_request(monkeypatch):
    """Runs the real _convert (imports, options, client call) against a stand-in client, no network."""
    import docling.service_client as service_client

    seen = {}

    class FakeClient:
        def __init__(self, url, api_key, job_timeout):
            seen.update(url=url, has_key=bool(api_key), job_timeout=job_timeout)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def convert(self, source, options):
            seen.update(suffix=source.suffix, exists=source.exists(), options=options)
            return SimpleNamespace(document="the-document")

    monkeypatch.setenv("DOCLING_SERVICE_URL", "https://example.test/instance")
    monkeypatch.setenv("DOCLING_SERVICE_API_KEY", "k")
    monkeypatch.setattr(service_client, "DoclingServiceClient", FakeClient)
    assert docling_ocr._convert(b"%PDF-1.4", (1, 2), "photo.jpg") == "the-document"
    opts = seen["options"]
    assert seen["suffix"] == ".pdf" and seen["exists"] and seen["has_key"]
    assert opts.force_ocr and opts.do_ocr and opts.do_table_structure and not opts.include_images
    assert tuple(opts.page_range) == (1, 2) and [str(getattr(f, "value", f)) for f in opts.to_formats] == ["json"]
