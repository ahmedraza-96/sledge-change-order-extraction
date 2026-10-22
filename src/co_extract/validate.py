"""Type coercion + business-rule checks.

``normalize_extraction`` turns RawExtraction into typed values, emitting
issues for anything that does not parse. ``check_rules`` runs the arithmetic
and consistency checks a construction accountant would run by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from .normalize import parse_money, parse_value
from .schema import FIELD_SPECS, Issue, RawExtraction

# Rounding slack: tax and markup are often rounded per line.
TOLERANCE = Decimal("0.02")

ARITHMETIC_CODES = {"line_item_math", "line_items_vs_subtotal", "subtotal_to_total", "contract_sum_rollforward"}

# Signals that an amount is a credit. Checked against the total's evidence quote.
_CREDIT_HINT = re.compile(r"\b(credit|deduct|deduction|cr)\b|\(\s*\$?[\d,]+\.\d{2}\s*\)", re.I)


@dataclass
class TypedLineItem:
    description: str
    quantity: Decimal | None
    unit: str | None
    unit_price: Decimal | None
    amount: Decimal | None


@dataclass
class Typed:
    values: dict[str, object]
    line_items: list[TypedLineItem]
    parse_failed: set[str] = field(default_factory=set)


def _num(raw: str | None) -> Decimal | None:
    if raw is None or not raw.strip():
        return None
    return parse_money(raw)


def normalize_extraction(raw: RawExtraction) -> tuple[Typed, list[Issue]]:
    issues: list[Issue] = []
    values: dict[str, object] = {}
    failed: set[str] = set()
    for name, kind in FIELD_SPECS.items():
        f = getattr(raw, name)
        if f.value is None or not f.value.strip():
            values[name] = None
            continue
        try:
            values[name] = parse_value(kind, f.value)
        except ValueError as e:
            values[name] = None
            failed.add(name)
            issues.append(Issue(code="unparseable", severity="hard", message=f"{name}: {e}", fields=[name]))

    items: list[TypedLineItem] = []
    for i, li in enumerate(raw.line_items):
        try:
            items.append(TypedLineItem(li.description, _num(li.quantity), li.unit, _num(li.unit_price), _num(li.amount)))
        except ValueError as e:
            failed.add("line_items")
            issues.append(Issue(code="unparseable", severity="hard", message=f"line_items[{i}]: {e}", fields=["line_items"]))

    if not raw.is_change_order:
        issues.append(Issue(code="not_a_change_order", severity="soft", message="Model judged this is not a change order."))
    if raw.change_order_count > 1:
        issues.append(Issue(code="multiple_change_orders", severity="soft",
                            message=f"Document contains {raw.change_order_count} change orders; only one was extracted."))
    if values["change_order_number"] is None and "change_order_number" not in failed:
        issues.append(Issue(code="missing_critical", severity="hard", message="No change order number found.",
                            fields=["change_order_number"]))
    if values["total_change_amount"] is None and "total_change_amount" not in failed:
        issues.append(Issue(code="missing_critical", severity="soft", message="No change order amount found.",
                            fields=["total_change_amount"]))
    return Typed(values, items, failed), issues


def _close(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= TOLERANCE


def check_rules(t: Typed, raw: RawExtraction, today: date | None = None) -> list[Issue]:
    v = t.values
    issues: list[Issue] = []
    today = today or date.today()

    for i, li in enumerate(t.line_items):
        if li.quantity is not None and li.unit_price is not None and li.amount is not None:
            expected = (li.quantity * li.unit_price).quantize(Decimal("0.01"))
            if not _close(expected, li.amount) and abs(expected - li.amount) > abs(li.amount) * Decimal("0.005"):
                issues.append(Issue(code="line_item_math", severity="soft", fields=["line_items"],
                                    message=f"line {i + 1}: {li.quantity} x {li.unit_price} = {expected}, printed {li.amount}"))

    amounts = [li.amount for li in t.line_items if li.amount is not None]
    subtotal, markup, tax, total = v["subtotal"], v["markup"], v["tax"], v["total_change_amount"]
    if amounts and len(amounts) == len(t.line_items):
        s = sum(amounts, Decimal(0))
        if subtotal is not None and not _close(s, subtotal):
            issues.append(Issue(code="line_items_vs_subtotal", severity="soft", fields=["line_items", "subtotal"],
                                message=f"line items sum to {s}, subtotal is {subtotal}"))
        elif subtotal is None and markup is None and tax is None and total is not None and not _close(s, total):
            issues.append(Issue(code="line_items_vs_subtotal", severity="soft", fields=["line_items", "total_change_amount"],
                                message=f"line items sum to {s}, total is {total}"))

    if subtotal is not None and total is not None:
        expected = subtotal + (markup or 0) + (tax or 0)
        if not _close(expected, total):
            issues.append(Issue(code="subtotal_to_total", severity="soft",
                                fields=["subtotal", "markup", "tax", "total_change_amount"],
                                message=f"subtotal {subtotal} + markup {markup or 0} + tax {tax or 0} = {expected}, total is {total}"))

    orig, prev, revised = v["original_contract_sum"], v["previous_change_orders_total"], v["revised_contract_sum"]
    if orig is not None and total is not None and revised is not None:
        expected = orig + (prev or 0) + total
        if not _close(expected, revised):
            issues.append(Issue(code="contract_sum_rollforward", severity="soft",
                                fields=["original_contract_sum", "previous_change_orders_total", "total_change_amount", "revised_contract_sum"],
                                message=f"original {orig} + previous {prev or 0} + this CO {total} = {expected}, revised sum is {revised}"))

    issued, completion = v["date_issued"], v["revised_completion_date"]
    if isinstance(issued, date) and not (date(2000, 1, 1) <= issued <= date(today.year + 1, 12, 31)):
        issues.append(Issue(code="date_order", severity="soft", fields=["date_issued"], message=f"implausible issue date {issued}"))
    if isinstance(issued, date) and isinstance(completion, date) and completion < issued:
        issues.append(Issue(code="date_order", severity="soft", fields=["date_issued", "revised_completion_date"],
                            message=f"revised completion {completion} is before issue date {issued}"))

    days = v["schedule_impact_days"]
    if isinstance(days, int) and not -365 <= days <= 1000:
        issues.append(Issue(code="implausible_value", severity="soft", fields=["schedule_impact_days"],
                            message=f"schedule impact of {days} days"))

    if total is not None and total > 0:
        if _CREDIT_HINT.search(raw.total_change_amount.evidence or ""):
            issues.append(Issue(code="credit_sign", severity="soft", fields=["total_change_amount"],
                                message="total is positive but the document describes a credit/deduct"))
    return issues
