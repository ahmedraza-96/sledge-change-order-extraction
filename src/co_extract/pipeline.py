"""ingest -> extract -> validate -> (repair once) -> score -> JSON envelope."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from . import docling_ocr, heuristics
from .confidence import (
    REVIEW_THRESHOLD,
    VERIFIER_DOUBT,
    W_VERIFIER,
    blend_verifier,
    document_confidence,
    document_score,
    field_confidence,
    line_items_confidence,
)
from .extract import Extractor, LLMResult
from .ingest import Document, load
from .normalize import to_json
from .schema import FIELD_SPECS, Issue, RawExtraction
from .validate import ARITHMETIC_CODES, Typed, check_rules, normalize_extraction, overrule_not_a_change_order
from .verify import Verifier, VerifyError


def _validate(raw: RawExtraction, doc: Document) -> tuple[RawExtraction, Typed, list[Issue]]:
    typed, issues = normalize_extraction(raw)
    issues += check_rules(typed, raw)
    # The "not a change order" verdict zeroes every score, so the document gets a say too.
    raw, issues = overrule_not_a_change_order(raw, typed, issues, doc.pages[0] if doc.pages else "")
    return raw, typed, issues


def _needs_repair(raw: RawExtraction, issues: list[Issue]) -> bool:
    # Hard errors are model mistakes. Arithmetic mismatches are often a misread digit,
    # so they get one re-read too; if the second read agrees, the document itself is wrong.
    # Not for a document that isn't a change order: re-reading an invoice can't produce a
    # CO number, it only spends a second model call.
    if not raw.is_change_order:
        return False
    return any(i.severity == "hard" or i.code in ARITHMETIC_CODES for i in issues)


def run(path: str | Path, extractor: Extractor, allow_repair: bool = True,
        on_step: Callable[..., None] | None = None, verifier: Verifier | None = None) -> dict:
    """``on_step(name, **info)`` is called as each stage starts (read, extract, validate,
    repair, score, verify), so a caller such as the web UI can show progress."""
    step = on_step or (lambda name, **info: None)
    t0 = time.perf_counter()
    step("read")
    doc = load(path, on_ocr=lambda: step("read", ocr=True))
    step("extract")
    calls: list[LLMResult] = [extractor.extract(doc, progress=lambda **info: step("extract", **info))]
    raw = calls[0].extraction
    step("validate")
    raw, typed, issues = _validate(raw, doc)

    repair = None
    if allow_repair and _needs_repair(raw, issues):
        first_codes = sorted(i.code for i in issues)
        step("repair", checks=first_codes)
        calls.append(extractor.extract(doc, repair=(raw, issues), progress=lambda **info: step("repair", **info)))
        raw2 = calls[-1].extraction
        raw2, typed2, issues2 = _validate(raw2, doc)
        n_hard = lambda xs: sum(i.severity == "hard" for i in xs)  # noqa: E731
        # Keep whichever attempt is better: fewer hard errors, then fewer issues overall.
        if (n_hard(issues2), len(issues2)) <= (n_hard(issues), len(issues)):
            raw, typed, issues = raw2, typed2, issues2
        repair = {"triggered_by": first_codes, "remaining": sorted(i.code for i in issues)}
        for i in issues:
            if i.code in ARITHMETIC_CODES:
                i.message += " (persisted after re-read: likely an error in the source document)"

    step("score")
    result = build_result(doc, raw, typed, issues, calls, repair, time.perf_counter() - t0)
    if verifier is not None:
        step("verify")
        attach_verification(result, doc, verifier)
        result["meta"]["seconds"] = round(time.perf_counter() - t0, 2)
    return result


def attach_verification(result: dict, doc: Document, verifier: Verifier) -> dict:
    """Ask the verifier about every field, store its probability and fold it into the confidence.

    Works on a cached result too. A verifier failure never fails the run: it is noted in
    ``meta.verify`` and the scores stay as they were. Skipped for a document that isn't a
    change order, whose scores are already 0.
    """
    if any(i["code"] == "not_a_change_order" for i in result["issues"]):
        return result
    try:
        v = verifier.verify(doc, result["change_order"])
    except VerifyError as e:
        result["meta"]["verify"] = {"error": str(e)}
        return result
    for name, f in result["fields"].items():
        if name == "line_items":
            rows = [v.probabilities.get(f"line_item_{i}") for i in range(len(f["value"]))]
            rows = [p for p in rows if p is not None]
            f["verifier_probability"] = min(rows) if rows else None
        else:
            f["verifier_probability"] = v.probabilities.get(name)
    result["meta"]["verify"] = {"model": v.model, "input_tokens": v.input_tokens}
    return apply_verifier(result)


def apply_verifier(result: dict) -> dict:
    """Blend each field's stored ``verifier_probability`` into its confidence, then redo the
    document score and the review list.

    Always blends from ``base_confidence`` (the score before the verifier), so applying it twice,
    or re-scoring a cached run after ``W_VERIFIER`` changes, gives the same answer.
    """
    if any(i["code"] == "not_a_change_order" for i in result["issues"]):
        return result
    for f in result["fields"].values():
        if f.get("verifier_probability") is None:
            continue
        p = f["verifier_probability"]
        f.setdefault("base_confidence", f["confidence"])
        f["confidence"] = blend_verifier(f["base_confidence"], p)
        f["reasons"] = [r for r in f["reasons"] if not r.startswith("second-model check")]
        if p < VERIFIER_DOUBT:
            f["reasons"].append(f"second-model check doubts this value ({p:.2f})")
    scores = {k: f["confidence"] for k, f in result["fields"].items()}
    codes = {i["code"] for i in result["issues"]}
    hard = any(i["severity"] == "hard" for i in result["issues"])
    result["document_confidence"] = document_score(scores, hard, "multiple_change_orders" in codes)
    result["needs_review"] = sorted(k for k, s in scores.items() if s < REVIEW_THRESHOLD)
    result["meta"].setdefault("verify", {})["weight"] = W_VERIFIER
    return result


def build_result(doc: Document, raw: RawExtraction, typed: Typed, issues: list[Issue], calls: list[LLMResult],
                 repair: dict | None, seconds: float) -> dict:
    candidates = heuristics.extract(doc.text)
    fields: dict[str, dict] = {}
    for name in FIELD_SPECS:
        f = getattr(raw, name)
        value = typed.values[name]
        conf, reasons = field_confidence(name, f, value, name in typed.parse_failed, doc, candidates, issues)
        fields[name] = {
            "value": to_json(value),
            "confidence": conf,
            "evidence": f.evidence,
            "page": f.page,
            "model_confidence": f.confidence,
            "reasons": reasons,
        }
    li_conf, li_reasons = line_items_confidence(raw, typed, doc, issues)
    line_items = [
        {"description": li.description, "quantity": to_json(li.quantity) if li.quantity is not None else None,
         "unit": li.unit, "unit_price": to_json(li.unit_price), "amount": to_json(li.amount)}
        for li in typed.line_items
    ]
    fields["line_items"] = {"value": line_items, "confidence": li_conf, "reasons": li_reasons}

    scores = {k: v["confidence"] for k, v in fields.items()}
    hard = any(i.severity == "hard" for i in issues)
    doc_conf = document_confidence(scores, raw, hard)
    if not raw.is_change_order:
        # Values read from something that isn't a change order (an invoice, a letter) are forced
        # into change-order slots, so no field may look more trustworthy than the document.
        for f in fields.values():
            if f["confidence"] > doc_conf:
                f["confidence"] = doc_conf
                f["reasons"].append("capped: the document is not a change order")
        scores = {k: v["confidence"] for k, v in fields.items()}
    return {
        "source": doc.source,
        "change_order": {k: v["value"] for k, v in fields.items()},
        "fields": fields,
        "document_confidence": doc_conf,
        "needs_review": sorted(k for k, s in scores.items() if s < REVIEW_THRESHOLD),
        "issues": [i.model_dump() for i in issues],
        "notes": raw.notes,
        "meta": {
            "input_kind": doc.kind,
            "scanned_pages": doc.scanned_pages,
            "ocr_pages": doc.ocr_pages,
            # Present only when hosted OCR was tried; "error" says why it gave no text.
            **({"ocr": {"engine": docling_ocr.ENGINE, "error": doc.ocr_error}}
               if doc.ocr_pages or doc.ocr_error else {}),
            "provider": calls[-1].provider,
            "model": calls[-1].model,
            "llm_calls": len(calls),
            "repair": repair,
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "seconds": round(seconds, 2),
        },
    }
