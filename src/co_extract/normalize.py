"""Deterministic parsing of raw strings into typed values.

Each parser returns the typed value or raises ValueError; callers turn that
into an Issue rather than crashing.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from .schema import REASONS, STATUSES, FieldKind

_CREDIT_WORDS = re.compile(r"\b(credit|deduct(?:ion)?|cr)\b", re.I)


def parse_money(raw: str) -> Decimal:
    s = raw.strip()
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")")) or bool(_CREDIT_WORDS.search(s))
    digits = re.sub(r"[^\d.]", "", s)
    if not digits or digits.count(".") > 1:
        raise ValueError(f"not a money amount: {raw!r}")
    try:
        value = Decimal(digits).quantize(Decimal("0.01"))
    except InvalidOperation as e:
        raise ValueError(f"not a money amount: {raw!r}") from e
    return -value if negative else value


_DATE_FORMATS = (
    "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%m-%d-%y", "%B %d, %Y", "%b %d, %Y",
    "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y", "%b. %d, %Y",
)


def parse_date(raw: str) -> date:
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", raw.strip())
    s = re.sub(r"\s+", " ", s)
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {raw!r}")


def parse_int(raw: str) -> int:
    s = raw.strip().lower()
    if s in {"none", "no change", "n/a", "zero"}:
        return 0
    m = re.search(r"[-+]?\d+", s.replace(",", ""))
    if not m:
        raise ValueError(f"not an integer: {raw!r}")
    return int(m.group())


def parse_bool(raw: str) -> bool:
    s = raw.strip().lower()
    if s in {"true", "yes", "y", "signed", "1"}:
        return True
    if s in {"false", "no", "n", "unsigned", "blank", "0"}:
        return False
    raise ValueError(f"not a boolean: {raw!r}")


def _parse_enum(raw: str, allowed: tuple[str, ...]) -> str:
    s = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if s not in allowed:
        raise ValueError(f"{raw!r} not in {allowed}")
    return s


def parse_currency(raw: str) -> str:
    s = raw.strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", s):
        raise ValueError(f"not an ISO currency code: {raw!r}")
    return s


def parse_value(kind: FieldKind, raw: str):
    match kind:
        case FieldKind.STR:
            s = raw.strip()
            if not s:
                raise ValueError("empty string")
            return s
        case FieldKind.MONEY:
            return parse_money(raw)
        case FieldKind.DATE:
            return parse_date(raw)
        case FieldKind.INT:
            return parse_int(raw)
        case FieldKind.BOOL:
            return parse_bool(raw)
        case FieldKind.REASON:
            return _parse_enum(raw, REASONS)
        case FieldKind.STATUS:
            return _parse_enum(raw, STATUSES)
        case FieldKind.CURRENCY:
            return parse_currency(raw)
    raise AssertionError(kind)


def to_json(value):
    """Typed value -> JSON-safe value (money as 2-dp string, dates ISO)."""
    if isinstance(value, Decimal):
        return f"{value:.2f}"
    if isinstance(value, date):
        return value.isoformat()
    return value
