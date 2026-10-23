import copy
import json
from email.message import EmailMessage
from pathlib import Path

import httpx2 as httpx  # anthropic's http client
import anthropic
import httpx as genai_httpx  # google-genai's http client (a different package)
import openai
import pytest
from google import genai
from google.genai import types

from co_extract import pipeline
from co_extract.extract import ClaudeExtractor, ExtractionError, FixtureExtractor, LLMResult
from co_extract.gemini import GeminiExtractor
from co_extract.ingest import load
from co_extract.openai_extractor import OpenAIExtractor
from co_extract.schema import RawExtraction
from co_extract.validate import check_rules, normalize_extraction

FIX = Path(__file__).parent / "fixtures"
GOOD = json.loads((FIX / "llm" / "simple_co.json").read_text())


def raw(**overrides) -> RawExtraction:
    d = copy.deepcopy(GOOD)
    for k, v in overrides.items():
        if isinstance(d.get(k), dict) and not isinstance(v, dict):
            d[k]["value"] = v
        else:
            d[k] = v
    return RawExtraction.model_validate(d)


def codes(r: RawExtraction) -> set[str]:
    typed, issues = normalize_extraction(r)
    return {i.code for i in issues + check_rules(typed, r)}


def test_clean_extraction_has_no_issues():
    assert codes(raw()) == set()


def test_rules_fire():
    assert "line_items_vs_subtotal" in codes(raw(subtotal="3050.00"))
    assert "subtotal_to_total" in codes(raw(markup="300.00"))
    assert "contract_sum_rollforward" in codes(raw(revised_contract_sum="1300000.00"))
    assert "date_order" in codes(raw(revised_completion_date="2025-01-01"))
    items = copy.deepcopy(GOOD["line_items"])
    items[0]["amount"] = "1200.00"
    assert "line_item_math" in codes(raw(line_items=items))
    assert "missing_critical" in codes(raw(change_order_number=None))
    assert "unparseable" in codes(raw(total_change_amount="about three grand"))


def test_credit_sign_rule():
    total = {"value": "3200.00", "evidence": "Total this Change Order ($3,200.00)", "page": 1, "confidence": 0.9}
    assert "credit_sign" in codes(raw(total_change_amount=total))


def test_pipeline_clean_doc():
    r = pipeline.run(FIX / "docs" / "simple_co.txt", FixtureExtractor(FIX / "llm"))
    assert r["change_order"]["total_change_amount"] == "3850.00"
    assert r["change_order"]["date_issued"] == "2026-03-04"
    assert r["issues"] == []
    assert r["needs_review"] == []
    assert r["document_confidence"] > 0.9
    assert r["meta"]["llm_calls"] == 1


def test_pipeline_repairs_misread():
    r = pipeline.run(FIX / "docs" / "misread_co.txt", FixtureExtractor(FIX / "llm"))
    assert r["meta"]["llm_calls"] == 2
    assert "unparseable" in r["meta"]["repair"]["triggered_by"]
    assert r["change_order"]["total_change_amount"] == "3850.00"
    assert r["issues"] == []


def test_misread_without_repair_is_flagged():
    r = pipeline.run(FIX / "docs" / "misread_co.txt", FixtureExtractor(FIX / "llm"), allow_repair=False)
    # The misread total doesn't appear in its own evidence and breaks two arithmetic checks.
    assert "total_change_amount" in r["needs_review"]
    assert r["fields"]["total_change_amount"]["confidence"] <= 0.3
    assert "date_issued" in r["needs_review"]
    assert r["document_confidence"] <= 0.5


def test_hallucinated_value_capped():
    r = raw(project_number={"value": "99-999", "evidence": "Project No. 24-118", "page": 1, "confidence": 0.99})
    doc = load(FIX / "docs" / "simple_co.txt")
    typed, issues = normalize_extraction(r)
    result = pipeline.build_result(doc, r, typed, issues, [LLMResult(r, "fixture")], None, 0.0)
    assert result["fields"]["project_number"]["confidence"] <= 0.3


def test_eml_ingest_drops_quoted_chain(tmp_path):
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = "pm@gc.com", "owner@client.com", "CO #9"
    msg.set_content("Total add is $8,740.50.\n\nOn Mon, Mar 2, 2026 Owner wrote:\n> earlier estimate $9,100.00\n")
    p = tmp_path / "co.eml"
    p.write_bytes(bytes(msg))
    doc = load(p)
    assert "$8,740.50" in doc.text
    assert "9,100" not in doc.text


def _mock_client(captured: list, payload: dict, stop_reason="end_turn") -> anthropic.Anthropic:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "stop_reason": stop_reason, "stop_sequence": None,
            "usage": {"input_tokens": 1234, "output_tokens": 567},
        })
    return anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)


def test_claude_extractor_request_and_parse():
    captured: list = []
    ex = ClaudeExtractor(client=_mock_client(captured, GOOD))
    r = pipeline.run(FIX / "docs" / "simple_co.txt", ex)
    assert r["change_order"]["total_change_amount"] == "3850.00"
    assert r["meta"]["input_tokens"] == 1234
    body = captured[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"]["effort"] == "medium"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert "<document" in body["messages"][0]["content"][-1]["text"]


def test_claude_extractor_sends_pdf_for_scanned(tmp_path):
    import pymupdf

    pdf = pymupdf.open()
    pdf.new_page()  # blank page = no text layer
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    captured: list = []
    ClaudeExtractor(client=_mock_client(captured, GOOD), send_pdf=False).extract(load(path))
    blocks = captured[0]["messages"][0]["content"]
    assert blocks[0]["type"] == "document"
    assert "no text layer" in blocks[-1]["text"]


def test_claude_extractor_refusal():
    ex = ClaudeExtractor(client=_mock_client([], GOOD, stop_reason="refusal"))
    with pytest.raises(ExtractionError, match="refused"):
        ex.extract(load(FIX / "docs" / "simple_co.txt"))


def _mock_gemini_client(captured: list, payload: dict | None, finish_reason="STOP") -> genai.Client:
    def handler(request: genai_httpx.Request) -> genai_httpx.Response:
        captured.append(json.loads(request.content))
        candidate = {"content": {"role": "model", "parts": []}, "finishReason": finish_reason}
        if payload is not None:
            candidate["content"]["parts"] = [{"text": json.dumps(payload)}]
        return genai_httpx.Response(200, json={
            "candidates": [candidate],
            "usageMetadata": {"promptTokenCount": 1234, "candidatesTokenCount": 567},
            "modelVersion": "gemini-3.5-flash-test",
        })
    return genai.Client(api_key="test", http_options=types.HttpOptions(
        client_args={"transport": genai_httpx.MockTransport(handler)}))


def test_gemini_extractor_request_and_parse():
    captured: list = []
    ex = GeminiExtractor(client=_mock_gemini_client(captured, GOOD))
    r = pipeline.run(FIX / "docs" / "simple_co.txt", ex)
    assert r["change_order"]["total_change_amount"] == "3850.00"
    assert r["meta"]["provider"] == "gemini"
    assert r["meta"]["model"] == "gemini-3.5-flash-test"
    assert r["meta"]["input_tokens"] == 1234
    assert r["meta"]["output_tokens"] == 567
    body = captured[0]
    cfg = body["generationConfig"]
    assert cfg["responseMimeType"] == "application/json"
    # The 21 repeated ExtractedField $refs must arrive inlined, not as $ref.
    assert "$ref" not in json.dumps(cfg["responseSchema"])
    assert set(cfg["responseSchema"]["properties"]) >= {"change_order_number", "total_change_amount", "line_items"}
    assert "You extract structured data" in body["systemInstruction"]["parts"][0]["text"]
    assert "<document" in body["contents"][0]["parts"][-1]["text"]


def test_all_providers_send_identical_prompt_text():
    """The eval comparison is only meaningful if every provider sees the same words."""
    doc = load(FIX / "docs" / "simple_co.txt")
    claude, gemini, oai = [], [], []
    ClaudeExtractor(client=_mock_client(claude, GOOD)).extract(doc)
    GeminiExtractor(client=_mock_gemini_client(gemini, GOOD)).extract(doc)
    OpenAIExtractor(client=_mock_openai_client(oai, GOOD)).extract(doc)
    assert claude[0]["system"] == gemini[0]["systemInstruction"]["parts"][0]["text"] == oai[0]["instructions"]
    assert (claude[0]["messages"][0]["content"][-1]["text"] == gemini[0]["contents"][0]["parts"][-1]["text"]
            == oai[0]["input"][0]["content"][-1]["text"])


def test_gemini_extractor_sends_pdf_for_scanned(tmp_path):
    import pymupdf

    pdf = pymupdf.open()
    pdf.new_page()  # blank page = no text layer
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    captured: list = []
    GeminiExtractor(client=_mock_gemini_client(captured, GOOD), send_pdf=False).extract(load(path))
    parts = captured[0]["contents"][0]["parts"]
    inline = parts[0]["inlineData"]  # the SDK serialises the inner key as mime_type
    assert inline.get("mime_type", inline.get("mimeType")) == "application/pdf"
    assert inline["data"]
    assert "no text layer" in parts[-1]["text"]


def test_gemini_extractor_repair_pass_sends_failed_checks():
    # The misread fixture has an unparseable date (a hard issue), so validation
    # demands a re-read; the mock replays it both times, so the issue persists.
    misread = json.loads((FIX / "llm" / "misread_co.json").read_text())
    captured: list = []
    ex = GeminiExtractor(client=_mock_gemini_client(captured, misread))
    r = pipeline.run(FIX / "docs" / "misread_co.txt", ex)
    assert r["meta"]["llm_calls"] == 2
    assert "unparseable" in r["meta"]["repair"]["triggered_by"]
    assert "A previous extraction of this document failed validation" in captured[1]["contents"][0]["parts"][-1]["text"]


@pytest.mark.parametrize("finish,match", [("SAFETY", "declined"), ("MAX_TOKENS", "truncated")])
def test_gemini_extractor_bad_finish_reasons(finish, match):
    ex = GeminiExtractor(client=_mock_gemini_client([], GOOD, finish_reason=finish))
    with pytest.raises(ExtractionError, match=match):
        ex.extract(load(FIX / "docs" / "simple_co.txt"))


def test_gemini_extractor_no_structured_output():
    ex = GeminiExtractor(client=_mock_gemini_client([], None))
    with pytest.raises(ExtractionError, match="no structured output"):
        ex.extract(load(FIX / "docs" / "simple_co.txt"))


def _mock_openai_client(captured: list, payload: dict | None, status: int = 200, status_field="completed",
                        error: dict | None = None) -> openai.OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        if error is not None:
            return httpx.Response(status, json={"error": error})
        content = [{"type": "output_text", "text": json.dumps(payload), "annotations": []}] if payload is not None else []
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0, "model": "gpt-5-mini-test",
            "status": status_field, "incomplete_details": {"reason": "max_output_tokens"} if status_field == "incomplete" else None,
            "output": [{"type": "message", "id": "msg_test", "role": "assistant", "status": "completed", "content": content}],
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": 1234, "output_tokens": 567, "total_tokens": 1801,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
        })
    return openai.OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)


def test_openai_extractor_request_and_parse():
    captured: list = []
    ex = OpenAIExtractor(client=_mock_openai_client(captured, GOOD))
    r = pipeline.run(FIX / "docs" / "simple_co.txt", ex)
    assert r["change_order"]["total_change_amount"] == "3850.00"
    assert r["meta"]["provider"] == "openai"
    assert r["meta"]["model"] == "gpt-5-mini-test"
    assert r["meta"]["input_tokens"] == 1234
    body = captured[0]
    assert body["model"] == "gpt-5-mini"
    assert body["text"]["format"]["type"] == "json_schema"
    assert body["text"]["format"]["strict"] is True
    assert "You extract structured data" in body["instructions"]
    assert "<document" in body["input"][0]["content"][-1]["text"]


def test_openai_extractor_sends_pdf_for_scanned(tmp_path):
    import pymupdf

    pdf = pymupdf.open()
    pdf.new_page()  # blank page = no text layer
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    captured: list = []
    OpenAIExtractor(client=_mock_openai_client(captured, GOOD), send_pdf=False).extract(load(path))
    parts = captured[0]["input"][0]["content"]
    assert parts[0]["type"] == "input_file"
    assert parts[0]["file_data"].startswith("data:application/pdf;base64,")
    assert "no text layer" in parts[-1]["text"]


def test_openai_extractor_out_of_credit():
    err = {"message": "You have no credits remaining.", "type": "insufficient_quota", "code": "credit_balance_exhausted"}
    ex = OpenAIExtractor(client=_mock_openai_client([], None, status=429, error=err))
    with pytest.raises(ExtractionError, match="quota exhausted: .*no credits remaining"):
        ex.extract(load(FIX / "docs" / "simple_co.txt"))


def test_openai_extractor_incomplete():
    ex = OpenAIExtractor(client=_mock_openai_client([], GOOD, status_field="incomplete"))
    with pytest.raises(ExtractionError, match="incomplete: max_output_tokens"):
        ex.extract(load(FIX / "docs" / "simple_co.txt"))


def _blank_pdf(tmp_path) -> Path:
    import pymupdf

    pdf = pymupdf.open()
    pdf.new_page()  # blank page = no text layer
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    return path


OCR_TEXT = ("CHANGE ORDER No. CO-007\nProject: Lakeview Medical Office Building\n"
            "Total this Change Order $3,850.00\nThe Contract Sum including this Change Order will be $1,296,150.00")


def test_ocr_recovers_scanned_page(tmp_path, monkeypatch):
    from co_extract import ingest

    monkeypatch.setattr(ingest, "_ocr_pages", lambda path, pages: {p: OCR_TEXT for p in pages})
    doc = load(_blank_pdf(tmp_path))
    assert doc.ocr_pages == [1] and doc.scanned_pages == []
    assert "$3,850.00" in doc.text


def test_ocr_pages_are_grounded_with_a_milder_penalty(tmp_path, monkeypatch):
    from co_extract import ingest

    # The fixture is keyed by stem, so give the blank PDF the simple_co stem.
    path = _blank_pdf(tmp_path).rename(tmp_path / "simple_co.pdf")
    as_scan = pipeline.run(path, FixtureExtractor(FIX / "llm"))
    monkeypatch.setattr(ingest, "_ocr_pages", lambda p, pages: {i: OCR_TEXT for i in pages})
    as_ocr = pipeline.run(path, FixtureExtractor(FIX / "llm"))
    f_scan, f_ocr = as_scan["fields"]["total_change_amount"], as_ocr["fields"]["total_change_amount"]
    assert any("scanned page" in r for r in f_scan["reasons"])
    assert "page text from OCR" in f_ocr["reasons"]
    assert f_ocr["confidence"] > f_scan["confidence"]
    assert as_ocr["meta"]["ocr_pages"] == [1] and as_ocr["meta"]["scanned_pages"] == []


def test_ocr_note_in_prompt_and_pdf_attached(tmp_path, monkeypatch):
    from co_extract import ingest

    monkeypatch.setattr(ingest, "_ocr_pages", lambda path, pages: {p: OCR_TEXT for p in pages})
    captured: list = []
    ClaudeExtractor(client=_mock_client(captured, GOOD), send_pdf=False).extract(load(_blank_pdf(tmp_path)))
    blocks = captured[0]["messages"][0]["content"]
    assert blocks[0]["type"] == "document"  # the image is still sent alongside the OCR text
    assert "were OCR'd" in blocks[-1]["text"]
    assert "no text layer" not in blocks[-1]["text"]


def test_ocr_disabled_keeps_page_scanned(tmp_path, monkeypatch):
    from co_extract import ingest

    monkeypatch.setattr(ingest, "_ocr_pages", lambda path, pages: {p: OCR_TEXT for p in pages})
    doc = load(_blank_pdf(tmp_path), ocr=False)
    assert doc.scanned_pages == [1] and doc.ocr_pages == []


def test_ocr_forced_without_docling_fails_loudly(tmp_path, monkeypatch):
    from co_extract import ingest

    monkeypatch.setattr(ingest, "ocr_available", lambda: False)
    with pytest.raises(RuntimeError, match="docling is not installed"):
        load(_blank_pdf(tmp_path), ocr=True)


def _image(tmp_path, name="photo.png", size=(400, 300), exif_orientation=None, frames=1) -> Path:
    from PIL import Image

    imgs = [Image.new("RGB", size, "white") for _ in range(frames)]
    path = tmp_path / name
    kwargs = {}
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        kwargs["exif"] = exif
    if frames > 1:
        kwargs.update(save_all=True, append_images=imgs[1:])
    imgs[0].save(path, **kwargs)
    return path


def _page_sizes(pdf_bytes: bytes) -> list[tuple[float, float]]:
    import io

    import pdfplumber

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return [(p.width, p.height) for p in pdf.pages]


def test_image_is_loaded_as_a_scanned_pdf_page(tmp_path):
    doc = load(_image(tmp_path))
    assert doc.kind == "image"
    assert doc.pdf_bytes.startswith(b"%PDF")
    assert doc.scanned_pages == [1]


def test_image_exif_rotation_is_applied(tmp_path):
    # Stored landscape, EXIF says "rotate 90°": the page must come out portrait.
    doc = load(_image(tmp_path, "rotated.jpg", size=(400, 200), exif_orientation=6))
    (w, h), = _page_sizes(doc.pdf_bytes)
    assert h > w


def test_large_image_is_downscaled(tmp_path):
    from co_extract.ingest import IMAGE_MAX_SIDE

    doc = load(_image(tmp_path, "big.png", size=(5000, 3000)))
    (w, h), = _page_sizes(doc.pdf_bytes)
    assert max(w, h) <= IMAGE_MAX_SIDE / 200 * 72 + 1  # points at the 200 dpi the page is written at


def test_multipage_tiff_becomes_multipage_pdf(tmp_path):
    doc = load(_image(tmp_path, "scan.tiff", frames=2))
    assert len(doc.pages) == 2 and doc.scanned_pages == [1, 2]


def test_image_goes_through_ocr_and_is_sent_to_the_model(tmp_path, monkeypatch):
    from co_extract import ingest

    seen = []
    monkeypatch.setattr(ingest, "_ocr_pages", lambda source, pages: seen.append(source) or {p: OCR_TEXT for p in pages})
    path = _image(tmp_path)
    doc = load(path)
    assert doc.ocr_pages == [1] and "$3,850.00" in doc.text
    assert seen and seen[0] != path  # an in-memory PDF, not the image file itself
    captured: list = []
    ClaudeExtractor(client=_mock_client(captured, GOOD), send_pdf=False).extract(doc)
    assert captured[0]["messages"][0]["content"][0]["type"] == "document"
