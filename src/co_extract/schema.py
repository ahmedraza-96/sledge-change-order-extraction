"""Data contracts.

Two layers:
- ``RawExtraction`` is what the LLM returns. Every value is a string as it
  appears in the document (lightly normalised), plus a verbatim evidence quote,
  the page it came from and a self-reported confidence. Keeping the LLM-facing
  schema stringly-typed makes it robust: the model never has to decide how to
  coerce "($3,200.00)" into a number, we do that deterministically.
- ``FIELD_SPECS`` declares the real type of each field; ``normalize.py`` turns
  the raw strings into typed values and records parse failures as issues.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class FieldKind(StrEnum):
    STR = "str"
    MONEY = "money"
    DATE = "date"
    INT = "int"
    BOOL = "bool"
    REASON = "reason"
    STATUS = "status"
    CURRENCY = "currency"


REASONS = ("owner_request", "unforeseen_condition", "design_change", "code_compliance", "other")
STATUSES = ("pending", "approved", "rejected")

# Ordered: this is also the order fields appear in the prompt and output.
FIELD_SPECS: dict[str, FieldKind] = {
    "change_order_number": FieldKind.STR,
    "project_name": FieldKind.STR,
    "project_number": FieldKind.STR,
    "contractor": FieldKind.STR,
    "owner": FieldKind.STR,
    "date_issued": FieldKind.DATE,
    "description": FieldKind.STR,
    "reason": FieldKind.REASON,
    "subtotal": FieldKind.MONEY,
    "markup": FieldKind.MONEY,
    "tax": FieldKind.MONEY,
    "total_change_amount": FieldKind.MONEY,
    "currency": FieldKind.CURRENCY,
    "original_contract_sum": FieldKind.MONEY,
    "previous_change_orders_total": FieldKind.MONEY,
    "revised_contract_sum": FieldKind.MONEY,
    "schedule_impact_days": FieldKind.INT,
    "revised_completion_date": FieldKind.DATE,
    "status": FieldKind.STATUS,
    "contractor_signed": FieldKind.BOOL,
    "owner_signed": FieldKind.BOOL,
}

# Fields whose errors are costly downstream (money moves, contract value changes).
CRITICAL_FIELDS = (
    "change_order_number",
    "total_change_amount",
    "original_contract_sum",
    "previous_change_orders_total",
    "revised_contract_sum",
)

FIELD_GUIDE: dict[str, str] = {
    "change_order_number": "CO / PCO identifier exactly as printed (e.g. 'CO-014', '7', 'PCO 22-B').",
    "project_name": "Project name.",
    "project_number": "Project / job number, if printed.",
    "contractor": "Company performing the work / issuing the CO (usually the GC).",
    "owner": "Owner / client the CO is addressed to.",
    "date_issued": "Date the CO was issued or dated. Return as YYYY-MM-DD.",
    "description": "One-sentence summary of the change in scope.",
    "reason": f"Why the change happened, one of: {', '.join(REASONS)}. Only if the document states or clearly implies it.",
    "subtotal": "Sum of line items / direct cost before markup and tax.",
    "markup": "Overhead & profit / fee amount (a dollar amount, not a percentage).",
    "tax": "Sales tax amount.",
    "total_change_amount": "The amount of THIS change order. Negative for credits / deducts (write e.g. '-3200.00').",
    "currency": "ISO currency code (USD unless the document says otherwise).",
    "original_contract_sum": "Original contract sum.",
    "previous_change_orders_total": "Net change by previously authorized change orders.",
    "revised_contract_sum": "Contract sum including this change order. NOT the change order amount.",
    "schedule_impact_days": "Change in contract time, in days (0 if the document explicitly says no change).",
    "revised_completion_date": "New substantial completion date. Return as YYYY-MM-DD.",
    "status": f"One of: {', '.join(STATUSES)}.",
    "contractor_signed": "'true' if the contractor signature block is signed, 'false' if blank.",
    "owner_signed": "'true' if the owner signature block is signed, 'false' if blank.",
}


class ExtractedField(BaseModel):
    value: str | None = Field(description="Value, or null if the document does not state it. Never guess.")
    evidence: str | None = Field(description="Shortest verbatim quote from the document that supports the value.")
    page: int | None = Field(description="1-based page number of the evidence.")
    confidence: float = Field(description="Your confidence from 0.0 to 1.0 that the value is correct.")


class RawLineItem(BaseModel):
    description: str
    quantity: str | None
    unit: str | None
    unit_price: str | None
    amount: str | None
    evidence: str | None = Field(description="Verbatim row text from the document.")
    page: int | None


class RawExtraction(BaseModel):
    """LLM output. One ExtractedField per entry in FIELD_SPECS."""

    is_change_order: bool = Field(description="False if the document is not a change order at all.")
    change_order_count: int = Field(description="How many distinct change orders the document contains.")
    change_order_number: ExtractedField
    project_name: ExtractedField
    project_number: ExtractedField
    contractor: ExtractedField
    owner: ExtractedField
    date_issued: ExtractedField
    description: ExtractedField
    reason: ExtractedField
    subtotal: ExtractedField
    markup: ExtractedField
    tax: ExtractedField
    total_change_amount: ExtractedField
    currency: ExtractedField
    original_contract_sum: ExtractedField
    previous_change_orders_total: ExtractedField
    revised_contract_sum: ExtractedField
    schedule_impact_days: ExtractedField
    revised_completion_date: ExtractedField
    status: ExtractedField
    contractor_signed: ExtractedField
    owner_signed: ExtractedField
    line_items: list[RawLineItem]
    notes: list[str] = Field(
        description="Ambiguities worth a human's attention: conflicting values, handwritten edits, superseded numbers."
    )


assert set(FIELD_SPECS) <= set(RawExtraction.model_fields), "schema drift"


class Issue(BaseModel):
    code: str
    severity: str  # "hard" -> triggers repair pass; "soft" -> lowers confidence
    message: str
    fields: list[str] = []
