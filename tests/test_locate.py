import json
from pathlib import Path

import pymupdf

from co_extract import locate
from co_extract.ingest import load

FIX = Path(__file__).parent / "fixtures"


def _pdf(tmp_path, lines) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    for i, line in enumerate(lines):
        page.insert_text((72, 100 + i * 20), line, fontname="helv", fontsize=11)
    path = tmp_path / "co.pdf"
    doc.save(path)
    return path


def _field(value, evidence, page=1):
    return {"value": value, "evidence": evidence, "page": page, "confidence": 0.9}


def test_value_is_circled_not_its_label(tmp_path):
    lay = locate.layout(load(_pdf(tmp_path, ["CHANGE ORDER No. CO-007", "Total this Change Order $3,850.00"])))
    marks, _ = locate.marks(lay, {"total_change_amount": _field("3850.00", "Total this Change Order $3,850.00")})
    m = marks["total_change_amount"]
    assert m["scope"] == "value"
    words = {w.text: w for w in lay.words[0]}
    x0, _, x1, _ = m["box"]
    # The box spans the amount only, starting right of the label's last word.
    assert x0 * 612 >= words["Order"].x1 and abs(x1 * 612 - words["$3,850.00"].x1) < 1


def test_credit_in_parentheses_matches_negative_value(tmp_path):
    lay = locate.layout(load(_pdf(tmp_path, ["TOTAL CREDIT THIS CHANGE ORDER ($3,200.00)"])))
    marks, _ = locate.marks(lay, {"total_change_amount": _field("-3200.00", "TOTAL CREDIT THIS CHANGE ORDER ($3,200.00)")})
    assert marks["total_change_amount"]["scope"] == "value"


def test_text_documents_are_typeset_and_mapped():
    doc = load(FIX / "docs" / "simple_co.txt")
    fields = json.loads((FIX / "llm" / "simple_co.json").read_text())
    lay = locate.layout(doc)
    assert lay.typeset and lay.positions == {1: "text"}
    marks, _ = locate.marks(lay, {k: v for k, v in fields.items() if isinstance(v, dict)})
    assert marks["change_order_number"]["scope"] == "value"
    assert marks["revised_contract_sum"]["scope"] == "value"


def test_scanned_page_uses_ocr_positions(tmp_path, monkeypatch):
    blank = pymupdf.open()
    blank.new_page(width=612, height=792)
    path = tmp_path / "scan.pdf"
    blank.save(path)
    # One OCR line at 150 dpi: x 150-750 px, y 300-330 px.
    monkeypatch.setattr(locate, "_ocr_lines", lambda page: [([[150, 300], [750, 300], [750, 330], [150, 330]],
                                                             "Total Change Order Amount $16,934.40")])
    lay = locate.layout(load(path))
    assert lay.positions == {1: "ocr"}
    marks, _ = locate.marks(lay, {"total_change_amount": _field("16934.40", "Total Change Order Amount $16,934.40")})
    m = marks["total_change_amount"]
    assert m["scope"] == "value"
    # Inside the OCR line, toward its right end where the amount is.
    x0, y0, x1, y1 = (v * s for v, s in zip(m["box"], (612, 792, 612, 792)))
    assert 300 * 72 / 150 < x0 < x1 <= 750 * 72 / 150 + 0.5
    assert abs(y0 - 300 * 72 / 150) < 0.5 and abs(y1 - 330 * 72 / 150) < 0.5


def test_ocr_glued_token_still_matches(tmp_path, monkeypatch):
    blank = pymupdf.open()
    blank.new_page()
    path = tmp_path / "scan.pdf"
    blank.save(path)
    monkeypatch.setattr(locate, "_ocr_lines", lambda page: [([[100, 100], [600, 100], [600, 130], [100, 130]],
                                                             "Project No:ESP-07 Amount.$16,934.40")])
    lay = locate.layout(load(path))
    marks, _ = locate.marks(lay, {"project_number": _field("ESP-07", "ESP-07"),
                                  "total_change_amount": _field("16934.40", "16,934.40")})
    assert set(marks) == {"project_number", "total_change_amount"}


def test_unmapped_fields_say_why(tmp_path):
    lay = locate.layout(load(_pdf(tmp_path, ["CHANGE ORDER No. CO-007"])))
    _, missing = locate.marks(lay, {
        "tax": _field(None, None),
        "currency": _field("USD", "$"),
        "owner": _field("Someone Else LLC", "Owner: Someone Else LLC"),
    })
    assert missing == {"tax": "no_quote", "currency": "symbol_only", "owner": "not_found"}


def test_preview_renders_pages():
    lay = locate.layout(load(FIX / "docs" / "simple_co.txt"))
    pv = locate.preview(lay, {})
    assert pv["pages"][0]["image"].startswith("data:image/png;base64,")
    assert pv["pages"][0]["width"] > 0 and pv["positions"] == {"1": "text"}
