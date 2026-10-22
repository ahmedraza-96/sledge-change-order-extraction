import copy
import json
from email.message import EmailMessage
from pathlib import Path

import httpx2 as httpx
import anthropic
import pytest

from co_extract import pipeline
from co_extract.extract import ClaudeExtractor, ExtractionError, FixtureExtractor, LLMResult
from co_extract.ingest import load
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
