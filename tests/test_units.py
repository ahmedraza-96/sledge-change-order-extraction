from datetime import date
from decimal import Decimal

import pytest

from co_extract import heuristics
from co_extract.confidence import grounding, value_in_evidence
from co_extract.ingest import Document
from co_extract.normalize import parse_bool, parse_date, parse_int, parse_money


@pytest.mark.parametrize("raw,expected", [
    ("$12,450.00", "12450.00"),
    ("12450", "12450.00"),
    ("($3,200.00)", "-3200.00"),
    ("-3200.00", "-3200.00"),
    ("CREDIT $3,200.00", "-3200.00"),
    ("$1,200.00 CR", "-1200.00"),
])
def test_parse_money(raw, expected):
    assert parse_money(raw) == Decimal(expected)


@pytest.mark.parametrize("raw", ["", "N/A", "1.2.3"])
def test_parse_money_rejects(raw):
    with pytest.raises(ValueError):
        parse_money(raw)


@pytest.mark.parametrize("raw", ["2026-03-04", "3/4/26", "03/04/2026", "March 4, 2026", "Mar 4 2026", "March 4th, 2026"])
def test_parse_date(raw):
    assert parse_date(raw) == date(2026, 3, 4)


def test_parse_int_and_bool():
    assert parse_int("14 calendar days") == 14
    assert parse_int("no change") == 0
    assert parse_int("-3") == -3
    assert parse_bool("true") is True and parse_bool("blank") is False
    with pytest.raises(ValueError):
        parse_bool("maybe")


def test_heuristics_find_labelled_amounts():
    text = ("CHANGE ORDER No. CO-007\nThe original Contract Sum was $1,250,000.00\n"
            "Net change by previously authorized Change Orders $42,300.00\n"
            "The Contract Sum including this Change Order will be $1,296,150.00\nSubtotal $3,500.00")
    c = heuristics.extract(text)
    assert c["original_contract_sum"] == {Decimal("1250000.00")}
    assert c["previous_change_orders_total"] == {Decimal("42300.00")}
    assert c["revised_contract_sum"] == {Decimal("1296150.00")}
    assert c["subtotal"] == {Decimal("3500.00")}
    assert heuristics.agrees("change_order_number", "CO-007", c)
    assert heuristics.agrees("subtotal", Decimal("3500.00"), c) is True
    assert heuristics.agrees("subtotal", Decimal("3050.00"), c) is False
    assert heuristics.agrees("tax", Decimal("1.00"), c) is None


def test_grounding():
    doc = Document(source="x.txt", kind="text", pages=["Total this Change Order      $3,850.00"])
    assert grounding("Total this Change Order $3,850.00", 1, doc) == 1.0
    assert grounding("Total this Change Order $9,999.00", 1, doc) < 1.0
    assert grounding(None, 1, doc) == 0.0
    scanned = Document(source="x.pdf", kind="pdf", pages=[""], scanned_pages=[1])
    assert grounding("anything", 1, scanned) is None


def test_value_in_evidence():
    assert value_in_evidence("total_change_amount", Decimal("3850.00"), "Total $3,850.00") is True
    assert value_in_evidence("total_change_amount", Decimal("3580.00"), "Total $3,850.00") is False
    assert value_in_evidence("total_change_amount", Decimal("-3200.00"), "($3,200.00)") is True
    assert value_in_evidence("date_issued", date(2026, 3, 4), "Date: March 4, 2026") is True
    assert value_in_evidence("date_issued", date(2026, 4, 3), "Date: March 4, 2026") is False
    assert value_in_evidence("contractor", "Northgate Builders, Inc.", "NORTHGATE BUILDERS, INC.") is True
    assert value_in_evidence("description", "anything", "whatever") is None


def test_heuristics_do_not_cross_lines():
    # Reflowed or scanned text often puts the amount before its label; the label must not grab the next line's number.
    text = ("$7,950,000.00 The original Contract Sum was\n"
            "$212,400.00 Net change by previously authorized Change Orders\n")
    c = heuristics.extract(text)
    assert "original_contract_sum" not in c  # abstain, rather than guess 212,400.00
    assert heuristics.agrees("original_contract_sum", Decimal("7950000.00"), c) is None


def test_line_items_with_a_missing_amount_score_as_wrong_not_crash():
    from co_extract.evaluate import field_correct

    truth = [{"amount": "1020.00"}, {"amount": "2480.00"}]
    assert field_correct("line_items", [{"amount": "2480.00"}, {"amount": "1020.00"}], truth)
    assert not field_correct("line_items", [{"amount": None}, {"amount": "2480.00"}], truth)
    assert not field_correct("line_items", [{"amount": None}, {"amount": None}], truth)
