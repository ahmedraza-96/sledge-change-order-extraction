"""Where on the page is each extracted value? Feeds the web UI's field map.

For every field with an evidence quote, find the quote among the page's words and box
the words that carry the value itself (falling back to the whole quote). Word positions
come from the PDF text layer, from our own typesetting for text and email, or from
RapidOCR for pages with no text layer.

This is presentation only: nothing here feeds extraction, validation or confidence.
"""

from __future__ import annotations

import base64
import re
import threading
from dataclasses import dataclass
from datetime import date

import pymupdf
from rapidfuzz import fuzz

from .ingest import Document

PREVIEW_DPI = 110
OCR_DPI = 150
MAX_PAGES = 10
FUZZY_MIN = 80

# Text and email are typeset onto Letter pages so they can be shown and boxed like a PDF.
_PAGE_W, _PAGE_H, _MARGIN, _FONT_SIZE = 612, 792, 54, 9.5
_LINE_H = _FONT_SIZE * 1.35
_CHARS = int((_PAGE_W - 2 * _MARGIN) / (_FONT_SIZE * 0.6))
_LINES = int((_PAGE_H - 2 * _MARGIN) / _LINE_H)

_EDGE = "$,()[]:;.\"'*"
_MONTHS = [date(2000, m, 1).strftime(f).lower() for m in range(1, 13) for f in ("%B", "%b")]


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def norm(self) -> str:
        return _norm(self.text)


@dataclass
class Layout:
    """The pages to show, and the words on each, in PDF points."""
    pdf: pymupdf.Document
    words: list[list[Word]]
    positions: dict[int, str]  # page -> "text" | "ocr" | "none"
    typeset: bool  # True for text/email: their page numbers don't follow the PDF's


def _norm(token: str) -> str:
    return token.lower().strip(_EDGE)


# ---------- layout ----------

def text_to_pdf(pages: list[str]) -> pymupdf.Document:
    """Typeset plain text in Courier on Letter pages, wrapping long lines."""
    lines: list[str] = []
    for page in pages:
        for raw in page.splitlines() or [""]:
            raw = raw.expandtabs(4)
            while len(raw) > _CHARS:
                cut = raw.rfind(" ", 0, _CHARS)
                cut = cut if cut > 0 else _CHARS
                lines.append(raw[:cut])
                raw = raw[cut:].lstrip()
            lines.append(raw)
    pdf = pymupdf.open()
    for start in range(0, max(len(lines), 1), _LINES):
        page = pdf.new_page(width=_PAGE_W, height=_PAGE_H)
        for i, line in enumerate(lines[start:start + _LINES]):
            if line.strip():
                page.insert_text((_MARGIN, _MARGIN + _FONT_SIZE + i * _LINE_H), line, fontname="cour", fontsize=_FONT_SIZE)
    return pdf


def layout(doc: Document) -> Layout:
    """Pages and word positions for a document. Runs OCR on pages with no text layer (slow)."""
    typeset = not doc.pdf_bytes
    pdf = text_to_pdf(doc.pages) if typeset else pymupdf.open(stream=doc.pdf_bytes, filetype="pdf")
    words, positions = [], {}
    for i, page in enumerate(list(pdf)[:MAX_PAGES], 1):
        found = [Word(*w[:5]) for w in page.get_text("words")]
        if found:
            words.append(found); positions[i] = "text"
            continue
        found = _ocr_words(page)
        words.append(found); positions[i] = "ocr" if found else "none"
    return Layout(pdf, words, positions, typeset)


_engine = None
_engine_lock = threading.Lock()


def _ocr_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr import RapidOCR

            # No orientation pass (images are already upright from EXIF) and no upscaling:
            # measured ~16 s/page on CPU against ~21 s with the defaults.
            _engine = RapidOCR(params={"Global.use_cls": False, "Det.limit_type": "max", "Det.limit_side_len": 1280})
        return _engine


def _ocr_lines(page: pymupdf.Page) -> list[tuple[list[list[float]], str]]:
    """OCR one page: (quad in pixels at OCR_DPI, text) per detected line."""
    import numpy as np

    pix = page.get_pixmap(dpi=OCR_DPI)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[:, :, :3]
    result = _ocr_engine()(img)
    if result.boxes is None:
        return []
    return [(box.tolist(), text) for box, text in zip(result.boxes, result.txts)]


def _ocr_words(page: pymupdf.Page) -> list[Word]:
    """Split OCR lines into word boxes, proportional to character position, in PDF points."""
    scale = 72 / OCR_DPI
    words = []
    for quad, text in _ocr_lines(page):
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        per_char = (x1 - x0) / max(len(text), 1)
        for m in re.finditer(r"\S+", text):
            words.append(Word((x0 + m.start() * per_char) * scale, y0 * scale,
                              (x0 + m.end() * per_char) * scale, y1 * scale, m.group()))
    return words


# ---------- matching ----------

def _find(words: list[Word], target: str) -> tuple[int, int] | None:
    """Word range [i, j) best matching ``target``: exact token run first, then fuzzy.

    The fuzzy pass compares with spaces removed, so OCR that merges words
    ("CHANGEORDER") still matches.
    """
    want = [t for t in map(_norm, target.split()) if t]
    if not want or not words:
        return None
    toks = [w.norm for w in words]
    n = len(want)
    for i in range(len(toks) - n + 1):
        if toks[i:i + n] == want:
            return i, i + n
    flat = "".join(want)
    if n == 1 and len(flat) >= 4:
        # A short quote glued to its label by OCR ("No:ESP-07"): take the token containing it.
        for i, t in enumerate(toks):
            if flat in t:
                return i, i + 1
    best, best_score = None, FUZZY_MIN
    for size in range(max(1, n - 2), n + 2):
        for i in range(len(toks) - size + 1):
            score = fuzz.ratio(flat, "".join(toks[i:i + size]))
            if score > best_score:
                best, best_score = (i, i + size), score
    return best


def _value_span(words: list[Word], value) -> tuple[int, int] | None:
    """The tokens inside a quote that carry the value, if they can be told apart."""
    if value is None or isinstance(value, bool):
        return None
    toks = [w.norm for w in words]
    s = str(value)
    if re.fullmatch(r"-?\d+(\.\d+)?", s):  # money or a count
        digits = s.lstrip("-")
        whole = digits.split(".")[0]
        # strip("."): OCR glues amounts to their label ("Amount.$16,934.40").
        hits = [i for i, t in enumerate(toks) if re.sub(r"[^\d.]", "", t).strip(".") in (digits, whole) and re.search(r"\d", t)]
        return (hits[0], hits[0] + 1) if hits else None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        y, m, d = s.split("-")
        keys = {y, str(int(d)), d, str(int(m)), m, *_MONTHS}
        hits = [i for i, t in enumerate(toks)
                if t in keys or y in t or re.sub(r"(st|nd|rd|th)$", "", t) in keys]
        return (hits[0], hits[-1] + 1) if hits and hits[-1] - hits[0] < 4 else None
    return _find(words, s)


def _box(words: list[Word]) -> list[float]:
    return [min(w.x0 for w in words), min(w.y0 for w in words), max(w.x1 for w in words), max(w.y1 for w in words)]


def marks(lay: Layout, fields: dict) -> tuple[dict, dict]:
    """field -> mark ({page, box as page fractions, scope}); field -> reason it has no mark."""
    out, missing = {}, {}
    for name, f in fields.items():
        if name == "line_items":
            continue
        if not f.get("evidence"):
            missing[name] = "no_quote"
            continue
        if not any(_norm(t) for t in f["evidence"].split()):
            missing[name] = "symbol_only"  # e.g. currency quoted as just "$"
            continue
        pages = list(range(1, len(lay.words) + 1))
        if not lay.typeset and f.get("page") in pages:
            pages.remove(f["page"]); pages.insert(0, f["page"])  # the cited page first
        hit = None
        for p in pages:
            span = _find(lay.words[p - 1], f["evidence"])
            if span:
                hit = (p, span)
                break
        if not hit:
            readable = any(lay.words[p - 1] for p in pages)
            missing[name] = "not_found" if readable else "no_text"
            continue
        p, (i, j) = hit
        quote = lay.words[p - 1][i:j]
        inner = _value_span(quote, f.get("value"))
        chosen, scope = (quote[inner[0]:inner[1]], "value") if inner else (quote, "quote")
        rect = lay.pdf[p - 1].rect
        x0, y0, x1, y1 = _box(chosen)
        out[name] = {"page": p, "box": [round(x0 / rect.width, 4), round(y0 / rect.height, 4),
                                        round(x1 / rect.width, 4), round(y1 / rect.height, 4)], "scope": scope}
    return out, missing


def render(lay: Layout) -> list[dict]:
    pages = []
    for page in list(lay.pdf)[:MAX_PAGES]:
        pix = page.get_pixmap(dpi=PREVIEW_DPI)
        pages.append({"image": "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode(),
                      "width": pix.width, "height": pix.height})
    return pages


def preview(lay: Layout, fields: dict) -> dict:
    found, missing = marks(lay, fields)
    return {"pages": render(lay), "marks": found, "missing": missing,
            "positions": {str(k): v for k, v in lay.positions.items()}}
