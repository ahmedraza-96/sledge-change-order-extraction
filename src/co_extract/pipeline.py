"""ingest -> extract -> validate -> (repair once) -> score -> JSON envelope."""

from __future__ import annotations

import time
from pathlib import Path

from . import heuristics
from .confidence import REVIEW_THRESHOLD, document_confidence, field_confidence, line_items_confidence
from .extract import Extractor, LLMResult
from .ingest import Document, load
from .normalize import to_json
from .schema import FIELD_SPECS, Issue, RawExtraction
from .validate import ARITHMETIC_CODES, Typed, check_rules, normalize_extraction


def _validate(raw: RawExtraction) -> tuple[Typed, list[Issue]]:
    typed, issues = normalize_extraction(raw)
    return typed, issues + check_rules(typed, raw)


def _needs_repair(issues: list[Issue]) -> bool:
    # Hard errors are model mistakes. Arithmetic mismatches are often a misread digit,
    # so they get one re-read too; if the second read agrees, the document itself is wrong.
    return any(i.severity == "hard" or i.code in ARITHMETIC_CODES for i in issues)


def run(path: str | Path, extractor: Extractor, allow_repair: bool = True) -> dict:
    t0 = time.perf_counter()
    doc = load(path)
    calls: list[LLMResult] = [extractor.extract(doc)]
    raw = calls[0].extraction
    typed, issues = _validate(raw)

    repair = None
    if allow_repair and _needs_repair(issues):
        first_codes = sorted(i.code for i in issues)
        calls.append(extractor.extract(doc, repair=(raw, issues)))
        raw2 = calls[-1].extraction
        typed2, issues2 = _validate(raw2)
        n_hard = lambda xs: sum(i.severity == "hard" for i in xs)  # noqa: E731
        # Keep whichever attempt is better: fewer hard errors, then fewer issues overall.
        if (n_hard(issues2), len(issues2)) <= (n_hard(issues), len(issues)):
            raw, typed, issues = raw2, typed2, issues2
        repair = {"triggered_by": first_codes, "remaining": sorted(i.code for i in issues)}
        for i in issues:
            if i.code in ARITHMETIC_CODES:
                i.message += " (persisted after re-read: likely an error in the source document)"

    return build_result(doc, raw, typed, issues, calls, repair, time.perf_counter() - t0)


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
    return {
        "source": doc.source,
        "change_order": {k: v["value"] for k, v in fields.items()},
        "fields": fields,
        "document_confidence": document_confidence(scores, raw, hard),
        "needs_review": sorted(k for k, s in scores.items() if s < REVIEW_THRESHOLD),
        "issues": [i.model_dump() for i in issues],
        "notes": raw.notes,
        "meta": {
            "input_kind": doc.kind,
            "scanned_pages": doc.scanned_pages,
            "model": calls[-1].model,
            "llm_calls": len(calls),
            "repair": repair,
            "input_tokens": sum(c.input_tokens for c in calls),
            "output_tokens": sum(c.output_tokens for c in calls),
            "seconds": round(seconds, 2),
        },
    }
