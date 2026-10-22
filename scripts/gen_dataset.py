"""Generate 10 synthetic change orders + ground truth into data/samples and data/truth.

Documents are rendered from the truth dicts so the two can't drift, except where a
sample deliberately differs (08: the document's own math is wrong; 10: handwritten override).
Deterministic; re-run to overwrite.
"""

from __future__ import annotations

import io
import json
import random
from decimal import Decimal
from email.message import EmailMessage
from pathlib import Path

import pymupdf
from PIL import Image, ImageFilter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent.parent
SAMPLES, TRUTH = ROOT / "data" / "samples", ROOT / "data" / "truth"
random.seed(7)

FIELDS = ["change_order_number", "project_name", "project_number", "contractor", "owner", "date_issued",
          "description", "reason", "subtotal", "markup", "tax", "total_change_amount", "currency",
          "original_contract_sum", "previous_change_orders_total", "revised_contract_sum",
          "schedule_impact_days", "revised_completion_date", "status", "contractor_signed", "owner_signed"]


def D(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"))


def money(x, parens=False) -> str:
    x = D(x)
    s = f"${abs(x):,.2f}"
    return f"({s})" if x < 0 and parens else (f"-{s}" if x < 0 else s)


def item(desc, qty, unit, price):
    return {"description": desc, "quantity": qty, "unit": unit, "unit_price": f"{D(price):.2f}",
            "amount": f"{D(qty) * D(price):.2f}"}


def truth(**kw) -> dict:
    t = {f: None for f in FIELDS}
    t.update(currency="USD", line_items=[], _expected_issues=[])
    t.update(kw)
    for f in ("subtotal", "markup", "tax", "total_change_amount", "original_contract_sum",
              "previous_change_orders_total", "revised_contract_sum"):
        if t[f] is not None:
            t[f] = f"{D(t[f]):.2f}"
    return t


def finish_money(t: dict, markup_pct=None, tax_pct=None, contract=None, previous=None):
    """Fill subtotal/markup/tax/total/contract sums from line items."""
    sub = sum(D(li["amount"]) for li in t["line_items"])
    mk = (sub * Decimal(markup_pct) / 100).quantize(Decimal("0.01")) if markup_pct else None
    tx = ((sub + (mk or 0)) * Decimal(tax_pct) / 100).quantize(Decimal("0.01")) if tax_pct else None
    total = sub + (mk or 0) + (tx or 0)
    t.update(subtotal=f"{sub:.2f}", markup=None if mk is None else f"{mk:.2f}",
             tax=None if tx is None else f"{tx:.2f}", total_change_amount=f"{total:.2f}")
    if contract is not None:
        t.update(original_contract_sum=f"{D(contract):.2f}", previous_change_orders_total=f"{D(previous):.2f}",
                 revised_contract_sum=f"{D(contract) + D(previous) + total:.2f}")
    return t


def item_rows(t):
    rows = [f"{'Description':<44}{'Qty':>6} {'Unit':<5}{'Unit Price':>12}{'Amount':>14}"]
    for li in t["line_items"]:
        q = li["quantity"]
        rows.append(f"{li['description']:<44}{q:>6} {li['unit']:<5}{money(li['unit_price']):>12}{money(li['amount']):>14}")
    return rows


# ---------------------------------------------------------------- rendering

def render_pdf(path: Path, pages: list[list[str]], font_size=9, extra=None):
    c = canvas.Canvas(str(path), pagesize=letter)
    for n, lines in enumerate(pages, 1):
        y = 750
        for line in lines:
            bold = line.startswith("**")
            c.setFont("Courier-Bold" if bold else "Courier", font_size + (2 if bold else 0))
            c.drawString(40, y, line.strip("*"))
            y -= font_size + 5
        if extra:
            extra(c, n)
        c.showPage()
    c.save()


def scan(src: Path, dst: Path):
    """Rasterize and degrade a PDF into an image-only PDF."""
    doc = pymupdf.open(src)
    pix = doc[0].get_pixmap(dpi=130)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    img = img.rotate(1.1, expand=True, fillcolor=255).filter(ImageFilter.GaussianBlur(0.6))
    px = img.load()
    for _ in range(img.width * img.height // 60):
        x, y = random.randrange(img.width), random.randrange(img.height)
        px[x, y] = random.choice((0, 90, 200))
    img.save(dst, "PDF", resolution=130)
    src.unlink()


def sig_lines(t, contractor_name="", owner_name=""):
    c = f"/s/ {contractor_name}" if t["contractor_signed"] else "______________________"
    o = f"/s/ {owner_name}" if t["owner_signed"] else "______________________"
    return ["", f"CONTRACTOR: {t['contractor']}   By: {c}",
            f"OWNER: {t['owner'] or ''}   By: {o}"]


BOILER = ["", "This Change Order is not valid until signed by both the Owner and Contractor.",
          "All work shall be performed in accordance with the Contract Documents.",
          "Contractor waives claims for additional cost or time arising from this change except as stated."]


def contract_block(t, credit_parens=False):
    return ["", f"The original Contract Sum was ................................ {money(t['original_contract_sum'])}",
            f"Net change by previously authorized Change Orders ............ {money(t['previous_change_orders_total'])}",
            f"The Contract Sum prior to this Change Order was .............. "
            f"{money(D(t['original_contract_sum']) + D(t['previous_change_orders_total']))}",
            f"The Contract Sum will be {'decreased' if D(t['total_change_amount']) < 0 else 'increased'} by this Change Order "
            f"in the amount of {money(abs(D(t['total_change_amount'])))}",
            f"The new Contract Sum including this Change Order will be ..... {money(t['revised_contract_sum'])}"]


# ---------------------------------------------------------------- samples

def s01():
    t = truth(change_order_number="CO-003", project_name="Riverside Elementary School Addition", project_number="2025-041",
              contractor="Harmon & Pike Construction, Inc.", owner="Riverside Unified School District",
              date_issued="2026-02-12", description="Add fire-rated door assemblies at new corridor per Addendum 4",
              reason="code_compliance", schedule_impact_days=5, revised_completion_date="2026-08-21", status="approved",
              contractor_signed=True, owner_signed=True, _variant="clean AIA G701-style form",
              line_items=[item("Furnish & install 90-min HM door assembly", 4, "EA", "2850.00"),
                          item("Door hardware sets, Group 3", 4, "EA", "1120.00"),
                          item("Patch & paint adjacent walls", 1, "LS", "1460.00")])
    finish_money(t, contract=4_812_000, previous=86_450)
    t["subtotal"] = None  # the AIA form only prints the CO amount, not a subtotal
    lines = ["**CHANGE ORDER                                  AIA Document G701 - style",
             f"PROJECT: {t['project_name']}            CHANGE ORDER NUMBER: 003",
             f"PROJECT NO.: {t['project_number']}                             DATE: February 12, 2026",
             f"TO CONTRACTOR: {t['contractor']}, 455 Mill St, Sacramento CA",
             f"OWNER: {t['owner']}", "ARCHITECT: Okafor Lindqvist Architects",
             "", "THE CONTRACT IS CHANGED AS FOLLOWS:",
             f"  {t['description']} (required by fire marshal review).", "",
             *item_rows(t), *contract_block(t),
             f"The Contract Time will be increased by Five (5) days.",
             "The date of Substantial Completion as of the date of this Change Order therefore is August 21, 2026.",
             "", "STATUS: APPROVED", *sig_lines(t, "D. Harmon 2/12/26", "L. Ortiz 2/14/26"), *BOILER]
    t["change_order_number"] = "003"
    render_pdf(SAMPLES / "01_aia_clean.pdf", [lines])
    return "01_aia_clean", t


def s02():
    t = truth(change_order_number="PCO 22-B", project_name="Harbor Point Apartments - Building C", project_number="HP-C-118",
              contractor="Northgate Builders, Inc.", owner="Harbor Point Housing LP", date_issued="2026-03-04",
              description="Relocate sanitary line and add blocking for revised kitchen layout in units C201-C208",
              reason="design_change", schedule_impact_days=3, status="pending", contractor_signed=True, owner_signed=False,
              _variant="contractor letterhead, line-item table, OH&P and tax",
              line_items=[item("Relocate 4\" sanitary line", 8, "EA", "640.00"),
                          item("Furnish & install 2x6 blocking", 320, "LF", "8.75"),
                          item("Saw-cut & patch slab", 96, "SF", "18.50"),
                          item("Plumbing labor, overtime", 24, "HR", "142.00")])
    finish_money(t, markup_pct=10, tax_pct="8.25")
    lines = ["**NORTHGATE BUILDERS, INC.", "1200 Harbor Way, Oakland, CA 94607  |  Lic. #884112  |  (510) 555-0142", "",
             "**PROPOSED CHANGE ORDER", f"PCO No.: 22-B                     Date: 03/04/2026",
             f"Project: {t['project_name']}   Job #: {t['project_number']}",
             f"To: {t['owner']}, Attn: R. Castillo, Owner's Rep", "",
             f"Scope: {t['description']}, per Architect's revised sheet A-402 (design change).", "",
             *item_rows(t), "",
             f"{'Subtotal':<64}{money(t['subtotal']):>14}",
             f"{'Overhead & Profit (10%)':<64}{money(t['markup']):>14}",
             f"{'Sales Tax (8.25%)':<64}{money(t['tax']):>14}",
             f"{'TOTAL PCO AMOUNT':<64}{money(t['total_change_amount']):>14}", "",
             "Schedule impact: 3 working days.", "Pricing valid for 30 days. Status: PENDING OWNER APPROVAL",
             *sig_lines(t, "J. Whitaker, PM"), *BOILER]
    t["change_order_number"] = "PCO 22-B"
    render_pdf(SAMPLES / "02_contractor_table.pdf", [lines])
    return "02_contractor_table", t


def s03():
    t = truth(change_order_number="CO-005", project_name="Mesa Verde Community Library", project_number="MV-2291",
              contractor="Copperline General Contractors", owner="City of Mesa Verde", date_issued="2026-04-20",
              description="Delete exterior trellis at north entry per Owner direction",
              reason="owner_request", schedule_impact_days=0, status="approved", contractor_signed=True, owner_signed=True,
              _variant="credit / deduct change order",
              line_items=[item("Delete steel trellis fabrication & install", 1, "LS", "-2650.00"),
                          item("Delete trellis footings", 4, "EA", "-137.50")])
    finish_money(t, contract=2_340_000, previous=31_200)
    t["subtotal"] = None
    lines = ["**COPPERLINE GENERAL CONTRACTORS", "88 Foundry Rd, Tempe AZ 85281", "",
             "**CHANGE ORDER #CO-005  (DEDUCT)", f"Project: {t['project_name']} ({t['project_number']})",
             f"Owner: {t['owner']}", "Date: April 20, 2026", "",
             f"Description: {t['description']}.", "",
             *item_rows(t), "",
             f"TOTAL CREDIT THIS CHANGE ORDER ............................ {money(t['total_change_amount'], parens=True)}",
             *contract_block(t), "Contract Time: No change.", "", "Status: APPROVED",
             *sig_lines(t, "M. Reyes", "K. Patel, City Engineer"), *BOILER]
    render_pdf(SAMPLES / "03_credit.pdf", [lines])
    return "03_credit", t


def s04():
    t = truth(change_order_number="CO-11", project_name="Elm Street Parking Structure", project_number="ESP-07",
              contractor="Ridgeback Concrete & Construction", owner="Elm Street Partners LLC", date_issued="2026-05-07",
              description="Additional rebar and epoxy injection at cracked level 2 slab discovered during demolition",
              reason="unforeseen_condition", schedule_impact_days=7, revised_completion_date="2026-11-13", status="approved",
              contractor_signed=True, owner_signed=True, _variant="scanned image-only PDF (no text layer)",
              line_items=[item("Epoxy crack injection", 180, "LF", "42.00"),
                          item("Additional #5 rebar, furnish & place", 2.4, "TON", "3150.00")])
    finish_money(t, markup_pct=12, contract=7_950_000, previous=212_400)
    lines = ["**RIDGEBACK CONCRETE & CONSTRUCTION", "", "**CHANGE ORDER No. CO-11",
             f"Project: {t['project_name']}   Project No: {t['project_number']}", f"Owner: {t['owner']}",
             "Date: 05/07/2026", "", "Reason: Unforeseen condition.", f"{t['description']}.", "",
             *item_rows(t), "", f"Subtotal {money(t['subtotal'])}", f"Markup 12% {money(t['markup'])}",
             f"Total Change Order Amount {money(t['total_change_amount'])}", *contract_block(t),
             "Time extension: 7 calendar days. New completion: November 13, 2026", "Status: APPROVED",
             *sig_lines(t, "T. Okonkwo", "A. Brandt")]
    tmp = SAMPLES / "_tmp04.pdf"
    render_pdf(tmp, [lines], font_size=10)
    scan(tmp, SAMPLES / "04_scanned.pdf")
    return "04_scanned", t


def s05():
    t = truth(change_order_number="9", project_name="Grandview Office Tower Lobby Renovation", project_number=None,
              contractor="Summit Interiors Group", owner="Grandview Properties", date_issued="2026-03-16",
              description="Replace lobby ceiling grid with acoustic wood slat ceiling", reason="owner_request",
              total_change_amount="8740.50", schedule_impact_days=3, status="pending", _variant="email body with superseded quoted estimate")
    msg = EmailMessage()
    msg["From"] = "Dana Kim <dkim@summitinteriors.com>"
    msg["To"] = "Paul Ferris <pferris@grandviewprop.com>"
    msg["Date"] = "Mon, 16 Mar 2026 09:12:00 -0700"
    msg["Subject"] = "RE: Lobby ceiling - CO #9 pricing"
    msg.set_content(
        "Hi Paul,\n\nFollowing up on your request to swap the lobby ceiling grid for the acoustic wood slat ceiling "
        "(Grandview Office Tower Lobby Renovation). We re-ran numbers with the supplier's final quote.\n\n"
        "CO #9 - total add of $8,740.50, and we'll need 3 days added to the schedule for the slat lead time.\n"
        "Breakdown is attached to the formal CO; let me know if you're good to approve and I'll send it for signature.\n\n"
        "Thanks,\nDana Kim\nProject Manager, Summit Interiors Group\n\n"
        "On Thu, Mar 12, 2026 at 4:40 PM Paul Ferris <pferris@grandviewprop.com> wrote:\n"
        "> Dana - can you price the wood slat ceiling? Your earlier estimate was $9,100.00 and 5 days.\n"
        "> That seems high.\n")
    (SAMPLES / "05_email.eml").write_bytes(bytes(msg))
    return "05_email", t


def s06():
    t = truth(change_order_number="CO-14", project_name="Cedar Ridge Medical Clinic", project_number="CR-3307",
              contractor="Bluestone Builders", owner="Cedar Ridge Health", date_issued="2026-03-04",
              description="Add 2 dedicated 20A circuits and data drops in exam rooms 4 and 5",
              reason="owner_request", schedule_impact_days=2, revised_completion_date="2026-06-18", status="approved",
              contractor_signed=True, owner_signed=True, _variant="messy plain text with typos and mixed date formats",
              line_items=[item("New 20A circuit incl. wire and conduit", 2, "EA", "1180.00"),
                          item("Cat6 data drop", 4, "EA", "265.00")])
    finish_money(t, markup_pct=15, contract=1_480_000, previous=18_900)
    txt = f"""BLUESTONE BUILDERS -- FIELD OFFICE
CHNAGE ORDR # CO-14
PROJ: cedar ridge medical clinic   job no CR-3307
TO: Cedar Ridge Health (owner)
dated 3/4/26

WORK: owner asked us to add 2 dedicated 20A circuits + data drops in exam rms 4 and 5

  new 20A circuit incl wire/conduit   2 ea @ 1,180.00 = 2360.00
  cat6 data drop                      4 ea @ 265.00   = $1,060.00

subtotal 3,420.00
markup 15%  513.00
TOTAL ADD THIS CO: $3,933.00

orig contract sum 1,480,000.00 / prev CO's net 18,900.00 / new contract sum $1,502,833.00
time: +2 days, substantial completion now March 18th... CORRECTION june 18th, 2026

APPROVED -- signed B. Stone (contractor) 3/5/26, signed M. Huang (owner) 3/6/26
"""
    assert D(t["revised_contract_sum"]) == D("1502833.00"), t["revised_contract_sum"]
    (SAMPLES / "06_messy.txt").write_text(txt)
    return "06_messy", t


def s07():
    t = truth(change_order_number="CO-2", project_name="Lakeside Fire Station 6 Renovation", project_number=None,
              contractor="Pinecrest Builders Ltd.", owner="Town of Lakeside", date_issued="2026-06-02",
              description="Replace rotted sill plates at apparatus bay discovered during siding removal",
              reason="unforeseen_condition", currency="CAD", status=None, contractor_signed=False, owner_signed=False,
              _variant="sparse: missing project no, schedule, contract sums, unsigned; CAD",
              line_items=[item("Replace PT sill plate", 64, "LF", "38.00"),
                          item("Carpentry labour", 16, "HR", "95.00")])
    finish_money(t)
    t["subtotal"] = None
    lines = ["**PINECREST BUILDERS LTD.", "Kelowna, BC", "", "**CHANGE ORDER CO-2",
             f"Project: {t['project_name']}", f"Client: {t['owner']}", "Date: June 2, 2026  |  Currency: CAD", "",
             f"{t['description']}.", "", *item_rows(t), "",
             f"Total: {money(t['total_change_amount'])} CAD", *sig_lines(t)]
    render_pdf(SAMPLES / "07_missing_fields.pdf", [lines])
    return "07_missing_fields", t


def s08():
    t = truth(change_order_number="CO-021", project_name="Westfield Distribution Center", project_number="WDC-552",
              contractor="Atlas Industrial Builders", owner="Westfield Logistics Inc.", date_issued="2026-07-09",
              description="Add four dock levelers and bumpers at doors 12-15", reason="owner_request",
              schedule_impact_days=10, status="pending", contractor_signed=True, owner_signed=False,
              _variant="document's own arithmetic is wrong (subtotal transposed, revised sum off)",
              _expected_issues=["line_items_vs_subtotal", "subtotal_to_total", "contract_sum_rollforward"],
              line_items=[item("Hydraulic dock leveler 7x8", 4, "EA", "2950.00"),
                          item("Dock bumpers", 8, "EA", "212.50"),
                          item("Electrical connection", 4, "EA", "387.50")])
    # Items sum to 15,050.00; the printed subtotal transposes to 15,500.00 and the total is printed from the real sum.
    t.update(subtotal="15500.00", markup="1505.00", total_change_amount="16555.00",
             original_contract_sum="12400000.00", previous_change_orders_total="-48200.00",
             revised_contract_sum="12386355.00")  # printed with a transposition
    t["_expected_issues"] = ["line_items_vs_subtotal", "subtotal_to_total", "contract_sum_rollforward"]
    lines = ["**ATLAS INDUSTRIAL BUILDERS", "", "**CHANGE ORDER NO. CO-021",
             f"Project: {t['project_name']}  #{t['project_number']}", f"Owner: {t['owner']}", "Date: July 9, 2026", "",
             f"Scope: {t['description']} (owner request).", "", *item_rows(t), "",
             f"{'Subtotal':<64}{money(t['subtotal']):>14}", f"{'Fee 10%':<64}{money(t['markup']):>14}",
             f"{'Change Order Total':<64}{money(t['total_change_amount']):>14}", "",
             f"Original Contract Sum: {money(t['original_contract_sum'])}",
             f"Net change by previously authorized Change Orders: {money(t['previous_change_orders_total'], parens=True)}",
             f"Revised Contract Sum: {money(t['revised_contract_sum'])}",
             "Contract time: increased by 10 days.", "Status: Pending", *sig_lines(t, "S. Grant")]
    render_pdf(SAMPLES / "08_bad_math.pdf", [lines])
    return "08_bad_math", t


def s09():
    descs = ["Demo existing VCT flooring", "Floor prep & self-leveler", "LVT plank flooring, furnish & install",
             "Rubber base 4\"", "Transition strips", "Move & reset casework", "Furniture moving",
             "Night shift premium", "Floor protection", "Final clean", "Dumpster & haul-off", "Moisture testing"]
    qtys = [4200, 4200, 4200, 980, 22, 1, 1, 1, 4200, 4200, 2, 12]
    units = ["SF", "SF", "SF", "LF", "EA", "LS", "LS", "LS", "SF", "SF", "EA", "EA"]
    prices = ["1.15", "2.40", "6.85", "3.10", "48.00", "3800.00", "2200.00", "4100.00", "0.35", "0.22", "685.00", "95.00"]
    t = truth(change_order_number="CO-008", project_name="St. Brendan Hospital - 3rd Floor Admin Refresh", project_number="SBH-24-310",
              contractor="Keystone Healthcare Builders", owner="St. Brendan Health System", date_issued="2026-08-17",
              description="Replace flooring throughout 3rd floor administrative wing per revised finish schedule",
              reason="design_change", schedule_impact_days=12, revised_completion_date="2027-01-29", status="approved",
              contractor_signed=True, owner_signed=True, _variant="multi-page, line items continue onto page 2",
              line_items=[item(d, q, u, p) for d, q, u, p in zip(descs, qtys, units, prices)])
    finish_money(t, markup_pct=8, contract=9_870_000, previous=145_800)
    rows = item_rows(t)
    head = lambda n: [f"Keystone Healthcare Builders        CO-008        Page {n} of 2", ""]  # noqa: E731
    p1 = [*head(1), "**CHANGE ORDER No. CO-008", f"Project: {t['project_name']}", f"Project No.: {t['project_number']}",
          f"Owner: {t['owner']}", "Date: August 17, 2026", "", f"Description: {t['description']}.", "",
          "Reason: design change (Architect's ASI-31).", "", *rows[:8], "", "(continued on next page)", *BOILER]
    p2 = [*head(2), "LINE ITEMS (continued)", rows[0], *rows[8:], "",
          f"{'Subtotal':<64}{money(t['subtotal']):>14}", f"{'GC Fee 8%':<64}{money(t['markup']):>14}",
          f"{'TOTAL THIS CHANGE ORDER':<64}{money(t['total_change_amount']):>14}", *contract_block(t),
          "Contract Time increased by 12 days; Substantial Completion: January 29, 2027", "Status: Approved",
          *sig_lines(t, "R. Iyer", "C. Doyle")]
    render_pdf(SAMPLES / "09_multipage.pdf", [p1, p2])
    return "09_multipage", t


def s10():
    t = truth(change_order_number="CO-017", project_name="Oakmont Brewery Taproom Buildout", project_number="OB-1902",
              contractor="Foxglove Construction Co.", owner="Oakmont Brewing Company", date_issued="2026-03-27",
              description="Upgrade bar top to quartz and add under-bar refrigeration rough-ins", reason="owner_request",
              schedule_impact_days=4, status="approved", contractor_signed=True, owner_signed=True,
              _variant="handwritten override of the typed total",
              _expected_issues=["subtotal_to_total", "contract_sum_rollforward"])
    typed_total = D("12600.00")
    t.update(subtotal="11454.55", markup="1145.45", total_change_amount="11850.00",
             original_contract_sum="685000.00", previous_change_orders_total="22150.00",
             revised_contract_sum=f"{D('685000') + D('22150') + typed_total:.2f}")
    lines = ["**FOXGLOVE CONSTRUCTION CO.", "", "**CHANGE ORDER CO-017", f"Project: {t['project_name']} ({t['project_number']})",
             f"Owner: {t['owner']}", "Date: 03/27/2026", "", f"Scope: {t['description']} (owner request).", "",
             f"Subtotal ........................................ {money(t['subtotal'])}",
             f"Contractor fee 10% .............................. {money(t['markup'])}",
             f"TOTAL CHANGE ORDER .............................. {money(typed_total)}", "", "", "",
             f"Original Contract Sum ........................... {money(t['original_contract_sum'])}",
             f"Net change by previously authorized Change Orders  {money(t['previous_change_orders_total'])}",
             f"Revised Contract Sum ............................ {money(t['revised_contract_sum'])}",
             "Contract time: +4 days", "Status: APPROVED", *sig_lines(t, "N. Fox", "G. Albers")]

    def handwriting(c, page):
        y_total = 750 - 11 * 14 - 2  # line index 11 ("TOTAL CHANGE ORDER")
        c.setStrokeColorRGB(0.1, 0.2, 0.8)
        c.setLineWidth(1.4)
        c.line(300, y_total + 3, 372, y_total + 1)
        c.saveState()
        c.setFillColorRGB(0.1, 0.2, 0.8)
        c.setFont("Times-Italic", 13)
        c.translate(300, y_total - 22)
        c.rotate(4)
        c.drawString(0, 0, "Revised per OAC mtg 4/2 - $11,850.00  JK")
        c.restoreState()

    render_pdf(SAMPLES / "10_handwritten_override.pdf", [lines], extra=handwriting)
    return "10_handwritten_override", t


def main():
    SAMPLES.mkdir(parents=True, exist_ok=True)
    TRUTH.mkdir(parents=True, exist_ok=True)
    for gen in (s01, s02, s03, s04, s05, s06, s07, s08, s09, s10):
        stem, t = gen()
        for li in t["line_items"]:
            li["quantity"] = float(li["quantity"]) if li["quantity"] is not None else None
        (TRUTH / f"{stem}.json").write_text(json.dumps(t, indent=2) + "\n")
        print(stem)


if __name__ == "__main__":
    main()
