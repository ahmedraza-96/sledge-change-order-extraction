"""Independent regex extractor used only as a cross-check on the LLM.

It is deliberately simple and high-precision: it finds labelled values
("Original Contract Sum ... $1,200,000.00") and abstains otherwise. Agreement
with the LLM raises confidence; disagreement lowers it.
"""

from __future__ import annotations

import re
from decimal import Decimal

from .normalize import parse_date, parse_money

_MONEY = r"(\(?-?\s*\$?\s*[\d,]+\.\d{2}\)?(?:\s*(?:CR|credit))?)"
_GAP = r"[^\n$\d(]{0,60}?"

_MONEY_LABELS: dict[str, str] = {
    "original_contract_sum": r"original\s+contract\s+sum",
    "previous_change_orders_total": r"(?:net\s+change\s+by\s+previously\s+authori[sz]ed|previous(?:ly)?\s+(?:approved\s+)?change\s+orders?)",
    "revised_contract_sum": r"(?:(?:revised|new)\s+contract\s+sum|contract\s+sum\s+including\s+this\s+change\s+order)",
    "subtotal": r"sub-?\s?total",
    "tax": r"(?:sales\s+)?tax",
}

_CO_NUMBER = re.compile(
    r"(?:change\s+order|chnage\s+ordr|\bCO\b|\bPCO\b)\s*(?:no\.?|number|#)?\s*[:#]?\s*([A-Z]{0,4}-?\s?\d+[A-Z0-9-]*)",
    re.I,
)
_DATE_LABEL = re.compile(r"\bdate(?:d)?\s*[:\-]?\s*([A-Za-z]{3,9}\.? \d{1,2}(?:st|nd|rd|th)?,? \d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})", re.I)


def extract(text: str) -> dict[str, set]:
    """Return field -> set of candidate typed values (empty dict entries omitted)."""
    out: dict[str, set] = {}
    for fld, label in _MONEY_LABELS.items():
        vals: set[Decimal] = set()
        for m in re.finditer(label + _GAP + _MONEY, text, re.I):
            try:
                vals.add(parse_money(m.group(1)))
            except ValueError:
                pass
        if vals:
            out[fld] = vals
    nums = {m.group(1).strip().upper().replace(" ", "") for m in _CO_NUMBER.finditer(text)}
    if nums:
        out["change_order_number"] = nums
    dates = set()
    for m in _DATE_LABEL.finditer(text):
        try:
            dates.add(parse_date(m.group(1)))
        except ValueError:
            pass
    if dates:
        out["date_issued"] = dates
    return out


def agrees(field: str, value, candidates: dict[str, set]) -> bool | None:
    """True/False if the heuristic has an opinion on this field, else None."""
    cands = candidates.get(field)
    if not cands or value is None:
        return None
    if field == "change_order_number":
        v = str(value).upper().replace(" ", "")
        return any(v == c or v.endswith(c) or c.endswith(v) for c in cands)
    if field == "date_issued":
        # Many dates appear in a CO; only count it as agreement, never as conflict.
        return True if value in cands else None
    return value in cands
