"""Per-field confidence scores.

LLM self-reported confidence is poorly calibrated on its own, so it gets a
low weight. Most of the signal comes from things we can check:

- grounding: does the evidence quote actually occur in the document text?
- value-in-evidence: does the quote actually contain the value we extracted?
- validation: is the field part of a failed arithmetic / consistency check?
- agreement: does an independent regex extractor find the same value?

Scanned pages cannot be grounded against a text layer, so they get a neutral
grounding score and a small penalty.
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from rapidfuzz import fuzz

from . import heuristics
from .ingest import Document
from .schema import FIELD_SPECS, ExtractedField, FieldKind, Issue, RawExtraction
from .validate import Typed

REVIEW_THRESHOLD = 0.7

W_GROUNDING, W_MODEL, W_CHECKS = 0.5, 0.2, 0.3
SCANNED_GROUNDING = 0.7  # neutral grounding when there is no text layer to check against
SCANNED_PENALTY = 0.9
UNSUPPORTED_CAP = 0.3  # value not found in its own evidence quote: likely hallucinated
ISSUE_PENALTY = 0.6  # per failed validation rule involving the field
DISAGREE_PENALTY = 0.5
AGREE_BONUS = 0.1

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def grounding(evidence: str | None, page: int | None, doc: Document) -> float | None:
    """Fuzzy containment of the evidence in the source text. None = cannot check."""
    if page is not None and page in doc.scanned_pages:
        return None
    if not doc.text.strip():
        return None
    if not evidence:
        return 0.0
    ev = _norm(evidence)
    hay = _norm(doc.pages[page - 1]) if page and 1 <= page <= len(doc.pages) else _norm(doc.text)
    score = fuzz.partial_ratio(ev, hay) / 100
    if score < 0.9 and hay != _norm(doc.text):  # wrong page cited; check the whole document
        score = max(score, fuzz.partial_ratio(ev, _norm(doc.text)) / 100)
    return score


def _numbers(text: str) -> set[Decimal]:
    out = set()
    for m in _NUM.findall(text):
        try:
            out.add(Decimal(m.replace(",", "")))
        except InvalidOperation:
            pass
    return out


def value_in_evidence(name: str, value, evidence: str | None) -> bool | None:
    """Does the evidence contain the extracted value? None = not checkable for this field."""
    if evidence is None or value is None:
        return None
    kind = FIELD_SPECS.get(name)
    if kind == FieldKind.MONEY:
        return abs(value) in _numbers(evidence)
    if kind == FieldKind.INT:
        return value == 0 or Decimal(abs(value)) in _numbers(evidence)
    if kind == FieldKind.DATE and isinstance(value, date):
        nums = _numbers(evidence)
        month_named = value.strftime("%b").lower() in evidence.lower()
        return Decimal(value.day) in nums and (month_named or Decimal(value.month) in nums)
    if kind == FieldKind.STR and name != "description":
        return fuzz.partial_ratio(_norm(str(value)), _norm(evidence)) >= 80
    return None  # interpretive fields (description, reason, status, signatures)


def field_confidence(name: str, f: ExtractedField, value, parse_failed: bool, doc: Document,
                     candidates: dict[str, set], issues: list[Issue]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if parse_failed:
        return 0.1, ["value could not be parsed"]
    model_conf = min(max(f.confidence, 0.0), 1.0)
    agreement = heuristics.agrees(name, value, candidates)

    if value is None:
        if candidates.get(name):
            reasons.append("returned null but a labelled value was found by the regex cross-check")
            return 0.3, reasons
        return round(0.55 + 0.4 * model_conf, 3), reasons

    g = grounding(f.evidence, f.page, doc)
    scanned = g is None
    if scanned:
        g = SCANNED_GROUNDING
        reasons.append("scanned page: evidence not verifiable against a text layer")
    elif g < 0.8:
        reasons.append(f"evidence quote only weakly matches the document ({g:.2f})")

    score = W_GROUNDING * g + W_MODEL * model_conf + W_CHECKS
    if agreement is True:
        score = min(1.0, score + AGREE_BONUS)
    elif agreement is False:
        score *= DISAGREE_PENALTY
        reasons.append(f"regex cross-check found a different value: {sorted(map(str, candidates[name]))}")
    for issue in issues:
        if name in issue.fields:
            score *= ISSUE_PENALTY
            reasons.append(f"involved in failed check: {issue.code}")
    if scanned:
        score *= SCANNED_PENALTY
    if value_in_evidence(name, value, f.evidence) is False:
        score = min(score, UNSUPPORTED_CAP)
        reasons.append("value does not appear in its own evidence quote")
    return round(score, 3), reasons


def line_items_confidence(raw: RawExtraction, typed: Typed, doc: Document, issues: list[Issue]) -> tuple[float, list[str]]:
    if "line_items" in typed.parse_failed:
        return 0.1, ["a line item could not be parsed"]
    if not raw.line_items:
        return 0.85, []
    scores, reasons = [], []
    for li in raw.line_items:
        g = grounding(li.evidence, li.page, doc)
        s = SCANNED_GROUNDING * SCANNED_PENALTY if g is None else g
        if li.amount and li.evidence and g is not None:
            try:
                if abs(Decimal(li.amount.replace(",", "").lstrip("-$"))) not in _numbers(li.evidence):
                    s = min(s, UNSUPPORTED_CAP)
            except InvalidOperation:
                pass
        scores.append(s)
    score = 0.7 * min(scores) + 0.3 * (sum(scores) / len(scores))
    if min(scores) < 0.8:
        reasons.append("at least one row is weakly grounded")
    for issue in issues:
        if "line_items" in issue.fields:
            score *= ISSUE_PENALTY
            reasons.append(f"involved in failed check: {issue.code}")
    return round(score, 3), reasons


def document_confidence(field_scores: dict[str, float], raw: RawExtraction, hard_issues: bool) -> float:
    from .schema import CRITICAL_FIELDS

    if not raw.is_change_order:
        return 0.0
    critical = [field_scores[f] for f in CRITICAL_FIELDS if f in field_scores]
    score = 0.6 * min(critical) + 0.4 * (sum(field_scores.values()) / len(field_scores))
    if hard_issues:
        score = min(score, 0.5)
    if raw.change_order_count > 1:
        score = min(score, 0.5)
    return round(score, 3)
