import copy
import json
from pathlib import Path

import httpx  # the verifier's own http client (not anthropic's httpx2)
import pymupdf
import pytest

from co_extract import evaluate, pipeline
from co_extract.confidence import REVIEW_THRESHOLD
from co_extract.extract import FixtureExtractor, LLMResult
from co_extract.ingest import load
from co_extract.schema import FIELD_SPECS, RawExtraction
from co_extract.verify import (
    DECISIONS_URL,
    DecisionsVerifier,
    FixtureVerifier,
    VerifyError,
    VerifyResult,
    claim,
    line_item_claim,
    make_verifier,
    questions,
)

FIX = Path(__file__).parent / "fixtures"
LLM = FixtureExtractor(FIX / "llm")
VERIFY = FixtureVerifier(FIX / "verify")


def test_claims_say_which_field_and_the_value():
    assert claim("owner", "Lakeview Health Partners LLC") == \
        'The owner or client the change order is addressed to is "Lakeview Health Partners LLC".'
    assert claim("total_change_amount", "3850.00") == "The total amount of this change order is 3,850.00."
    assert claim("total_change_amount", "-12387.60") == \
        "The total amount of this change order is a credit (deduct) of 12,387.60."
    assert claim("date_issued", "2026-03-04") == "The date the change order was issued is March 4, 2026."
    assert claim("schedule_impact_days", 2) == "The contract time is increased by 2 days."
    assert claim("schedule_impact_days", 0) == "The contract time does not change (0 days)."
    assert claim("owner_signed", False) == "The owner signature block is blank (not signed)."
    assert claim("reason", "owner_request") == "The change was made because of the owner requested it."
    # A null is a claim too: a wrongly missed value gets checked.
    assert claim("tax", None) == "The document does not state the sales tax amount."


def test_line_item_claims():
    assert line_item_claim({"description": "Relocate sanitary line", "amount": "2480.00"}) == \
        'The document lists a line item "Relocate sanitary line" with an amount of 2,480.00.'
    assert line_item_claim({"description": "Allowance", "amount": None}).endswith("with no amount stated.")


def test_one_question_per_field_and_line_item():
    r = pipeline.run(FIX / "docs" / "simple_co.txt", LLM)
    qs = questions(r["change_order"])
    names = [q["name"] for q in qs]
    assert names == [*FIELD_SPECS, "line_item_0", "line_item_1"]
    assert all(q["type"] == "predicate" for q in qs)


def _mock(captured: list, status=200, payload: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append({"url": str(request.url), "auth": request.headers["authorization"],
                         "body": json.loads(request.content)})
        body = payload if payload is not None else {
            "model": "gpt-6-luna",
            "answers": [{"type": "predicate", "name": q["name"], "probability": 0.9}
                        for q in json.loads(request.content)["questions"]],
            "usage": {"input_tokens": 777, "output_tokens": 0},
        }
        return httpx.Response(status, json=body)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_decisions_request_and_parse():
    captured: list = []
    v = DecisionsVerifier(api_key="test", client=_mock(captured))
    r = pipeline.run(FIX / "docs" / "simple_co.txt", LLM, verifier=v)
    req = captured[0]
    assert req["url"] == DECISIONS_URL and req["auth"] == "Bearer test"
    body = req["body"]
    assert body["model"] == "gpt-6-luna"
    developer, user = body["input"]
    assert developer["role"] == "developer" and "not instructions" in developer["content"][0]["text"]
    assert [p["type"] for p in user["content"]] == ["input_text"]  # text document: no images
    assert user["content"][0]["text"].startswith("<document")
    assert len(body["questions"]) == len(FIELD_SPECS) + 2
    assert r["fields"]["owner"]["verifier_probability"] == 0.9
    assert r["fields"]["line_items"]["verifier_probability"] == 0.9
    assert r["meta"]["verify"] == {"model": "gpt-6-luna", "input_tokens": 777, "weight": 0.35}


def test_decisions_sends_page_images_for_scans(tmp_path):
    pdf = pymupdf.open()
    pdf.new_page()  # blank page = no text layer
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    captured: list = []
    DecisionsVerifier(api_key="test", client=_mock(captured)).verify(load(path), {})
    parts = captured[0]["body"]["input"][1]["content"]
    assert parts[1]["type"] == "input_image" and parts[1]["image_url"].startswith("data:image/png;base64,")


def test_decisions_refusal_is_none():
    payload = {"model": "gpt-6-luna", "answers": [{"type": "refusal", "name": "owner"},
                                                  {"type": "predicate", "name": "tax", "probability": 0.0}]}
    v = DecisionsVerifier(api_key="test", client=_mock([], payload=payload))
    res = v.verify(load(FIX / "docs" / "simple_co.txt"), {})
    assert res.probabilities == {"owner": None, "tax": 0.0}


def test_decisions_api_error_raises():
    v = DecisionsVerifier(api_key="test", client=_mock([], status=401, payload={"error": {"message": "bad key"}}))
    with pytest.raises(VerifyError, match="401: bad key"):
        v.verify(load(FIX / "docs" / "simple_co.txt"), {})


def test_no_key_is_an_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("co_extract.verify.load_dotenv", lambda: None)
    with pytest.raises(VerifyError, match="OPENAI_API_KEY"):
        make_verifier("decisions")
    assert make_verifier("off") is None and make_verifier(None) is None


def test_verifier_is_blended_into_confidence():
    plain = pipeline.run(FIX / "docs" / "misread_co.txt", LLM)
    checked = pipeline.run(FIX / "docs" / "misread_co.txt", LLM, verifier=VERIFY)
    # owner_signed: the verifier says 0.08, so 35% of its score now comes from that "no".
    base = plain["fields"]["owner_signed"]["confidence"]
    f = checked["fields"]["owner_signed"]
    assert f["base_confidence"] == base
    assert f["confidence"] == round(0.65 * base + 0.35 * 0.08, 3)
    assert f["confidence"] < plain["fields"]["owner_signed"]["confidence"]
    assert ("owner_signed" in checked["needs_review"]) == (f["confidence"] < REVIEW_THRESHOLD)
    assert any(r.startswith("second-model check doubts") for r in f["reasons"])
    # description: the verifier declined, so its score is untouched.
    assert checked["fields"]["description"]["confidence"] == plain["fields"]["description"]["confidence"]
    assert "base_confidence" not in checked["fields"]["description"]
    assert checked["meta"]["verify"]["weight"] == 0.35


def test_verifier_yes_cannot_clear_a_failed_hard_check():
    # A value missing from its own evidence is capped at 0.3; a verifier "yes" lifts it to 0.545, still in review.
    r = pipeline.run(FIX / "docs" / "simple_co.txt", LLM)
    r["fields"]["subtotal"].update(confidence=0.3, verifier_probability=1.0)
    pipeline.apply_verifier(r)
    assert r["fields"]["subtotal"]["confidence"] == 0.545 and "subtotal" in r["needs_review"]


def test_apply_verifier_is_idempotent_and_redoes_document_score():
    r = pipeline.run(FIX / "docs" / "simple_co.txt", LLM, verifier=VERIFY)
    once = json.loads(json.dumps(r))
    pipeline.apply_verifier(r)
    assert r == once
    r["fields"]["total_change_amount"]["verifier_probability"] = 0.0  # a critical field
    pipeline.apply_verifier(r)
    assert r["document_confidence"] < once["document_confidence"]
    assert "total_change_amount" in r["needs_review"]


class _Failing:
    def verify(self, doc, values) -> VerifyResult:
        raise VerifyError("rate limited")


def test_verifier_failure_never_fails_the_run():
    r = pipeline.run(FIX / "docs" / "simple_co.txt", LLM, verifier=_Failing())
    assert r["meta"]["verify"] == {"error": "rate limited"}
    assert "verifier_probability" not in r["fields"]["owner"]


class _Counting:
    calls = 0

    def verify(self, doc, values) -> VerifyResult:
        self.calls += 1
        return VerifyResult({}, "x")


def test_not_a_change_order_is_not_verified():
    d = json.loads((FIX / "llm" / "simple_co.json").read_text())
    d["is_change_order"], d["change_order_number"]["value"] = False, None
    extraction = RawExtraction.model_validate(copy.deepcopy(d))

    class Replay:
        def extract(self, doc, repair=None, progress=None):
            return LLMResult(extraction=extraction, model="replay")

    v = _Counting()
    r = pipeline.run(FIX / "docs" / "simple_co.txt", Replay(), verifier=v)
    assert v.calls == 0 and "verify" not in r["meta"]


def test_cached_result_can_be_verified_later():
    r = pipeline.run(FIX / "docs" / "simple_co.txt", LLM)
    cached = json.loads(json.dumps(r))  # what the eval cache holds
    pipeline.attach_verification(cached, load(FIX / "docs" / "simple_co.txt"), VERIFY)
    assert cached["fields"]["total_change_amount"]["verifier_probability"] == 1.0


def _result(fields: dict[str, tuple], flagged: list[str]) -> dict:
    """fields: name -> (value, confidence, verifier probability)."""
    return {
        "change_order": {k: v[0] for k, v in fields.items()},
        "fields": {k: {"confidence": v[1], "verifier_probability": v[2]} for k, v in fields.items()},
        "needs_review": flagged, "issues": [], "document_confidence": 0.9,
        "meta": {"llm_calls": 1, "input_tokens": 0, "output_tokens": 0, "seconds": 0},
    }


def _truth() -> dict:
    return {name: [] if name == "line_items" else "x" for name in evaluate.FIELDS}


def test_eval_reports_verifier_signal():
    truth = _truth()
    fields = {name: (truth[name], 0.9, 1.0) for name in evaluate.FIELDS}
    fields["owner"] = ("wrong", 0.9, 0.1)       # wrong, missed by review, caught by verifier
    fields["contractor"] = ("wrong", 0.5, 0.95)  # wrong, caught by review, missed by verifier
    fields["tax"] = ("x", 0.9, 0.2)              # right, but the verifier doubts it
    s = evaluate.score({"d": _result(fields, ["contractor"])}, {"d": truth})
    v = s["verifier"]
    assert v["n"] == len(evaluate.FIELDS) and v["n_wrong"] == 2
    assert v["review_recall"] == 0.5 and v["verifier_recall"] == 0.5 and v["combined_recall"] == 1.0
    assert v["extra_flags"] == 1
    assert v["mean_p_wrong"] == pytest.approx(0.525)
    assert "## Verifier" in evaluate.render_markdown(s, "m")


def test_eval_without_verifier_has_no_section():
    truth = _truth()
    res = _result({name: (truth[name], 0.9, None) for name in evaluate.FIELDS}, [])
    s = evaluate.score({"d": res}, {"d": truth})
    assert s["verifier"] is None
    assert "## Verifier" not in evaluate.render_markdown(s, "m")
