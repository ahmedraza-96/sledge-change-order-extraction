"""Load a change order from PDF, image, .eml or plain text into page-tagged text."""

from __future__ import annotations

import email
import email.policy
import io
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import pymupdf
from PIL import Image, ImageOps

from . import docling_ocr

# A page with less extractable text than this is treated as a scan.
SCANNED_PAGE_MIN_CHARS = 30
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
# Phone photos are often 4000px+; this keeps the attached PDF well under provider size limits
# while leaving small print legible.
IMAGE_MAX_SIDE = 2400


@dataclass
class Document:
    source: str
    kind: str  # "pdf" | "image" | "email" | "text"
    pages: list[str]
    scanned_pages: list[int] = field(default_factory=list)  # 1-based; no text at all
    ocr_pages: list[int] = field(default_factory=list)  # 1-based; text read by Docling OCR
    ocr_error: str | None = None  # why OCR was attempted and failed
    pdf_bytes: bytes | None = None

    @property
    def is_scanned(self) -> bool:
        return bool(self.scanned_pages)

    @property
    def text(self) -> str:
        return "\n".join(self.pages)

    def tagged_text(self) -> str:
        return "\n\n".join(f"[[page {i}]]\n{p}" for i, p in enumerate(self.pages, 1))


def load(path: str | Path, on_ocr: Callable[[], None] | None = None) -> Document:
    """Read a document. With Docling configured, every page of every kind is OCR'd.

    ``on_ocr`` is called just before pages are sent to Docling, so a UI can say so.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        doc = _load_pdf(path)
    elif suffix in IMAGE_SUFFIXES:
        # A photo or scan is just a PDF page with no text layer: convert it and reuse that path.
        doc = _load_pdf(path, data=image_to_pdf(path), kind="image")
    elif suffix == ".eml":
        doc = _load_eml(path)
    else:
        doc = Document(source=str(path), kind="text", pages=[path.read_text(encoding="utf-8", errors="replace")])
    if docling_ocr.available():
        _ocr(doc, path.name, on_ocr)
    return doc


def _ocr(doc: Document, name: str, on_ocr: Callable[[], None] | None) -> None:
    """One path for every document: render it as page images and read them with Docling.

    The text the model sees is then OCR text whatever the input was. If Docling fails, the
    native text stays (pdfplumber or the file itself) and ``ocr_error`` says why; pages with
    no native text remain scanned.
    """
    pdf = doc.pdf_bytes or text_to_pdf(doc.pages)
    with pymupdf.open(stream=pdf, filetype="pdf") as rendered:
        n = len(rendered)
    if on_ocr:
        on_ocr()
    recovered, doc.ocr_error = docling_ocr.read_pages(pdf, list(range(1, n + 1)), name)
    read = {i: t for i, t in recovered.items() if len(t.strip()) >= SCANNED_PAGE_MIN_CHARS}
    if doc.pdf_bytes is None:
        # Text and email were typeset onto pages for OCR; swap their text only if every page came back.
        if len(read) == n:
            doc.pages = [read[i] for i in range(1, n + 1)]
            doc.ocr_pages = list(range(1, n + 1))
        elif not doc.ocr_error:
            doc.ocr_error = f"OCR read {len(read)} of {n} pages; kept the original text"
        return
    for i, text in read.items():
        doc.pages[i - 1] = text
        doc.ocr_pages.append(i)
    doc.scanned_pages = [i for i in doc.scanned_pages if i not in read]


# Text and email are typeset onto Letter pages so Docling can read them as page images.
_PAGE_W, _PAGE_H, _MARGIN, _FONT_SIZE = 612, 792, 54, 10
_LINE_H = _FONT_SIZE * 1.4
_CHARS = int((_PAGE_W - 2 * _MARGIN) / (_FONT_SIZE * 0.6))
_LINES = int((_PAGE_H - 2 * _MARGIN) / _LINE_H)


def text_to_pdf(pages: list[str]) -> bytes:
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
    return pdf.tobytes()


def image_to_pdf(path: Path) -> bytes:
    """Render an image (every frame of a multi-page TIFF) as PDF pages, upright and size-capped."""
    frames = []
    with Image.open(path) as img:
        for i in range(getattr(img, "n_frames", 1)):
            img.seek(i)
            frame = ImageOps.exif_transpose(img).convert("RGB")  # phone photos carry rotation in EXIF
            frame.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
            frames.append(frame)
    buf = io.BytesIO()
    frames[0].save(buf, format="PDF", save_all=True, append_images=frames[1:], resolution=200)
    return buf.getvalue()


def _load_pdf(path: Path, data: bytes | None = None, kind: str = "pdf") -> Document:
    data = data if data is not None else path.read_bytes()
    pages, scanned = [], []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append(text)
            if len(text.strip()) < SCANNED_PAGE_MIN_CHARS:
                scanned.append(i)
    return Document(source=str(path), kind=kind, pages=pages, scanned_pages=scanned, pdf_bytes=data)


_QUOTE_HEADER = re.compile(r"^On .+wrote:\s*$", re.M)


def _load_eml(path: Path) -> Document:
    msg = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    body = msg.get_body(preferencelist=("plain", "html"))
    text = body.get_content() if body else ""
    # Drop the quoted reply chain: it often holds superseded numbers.
    m = _QUOTE_HEADER.search(text)
    if m:
        text = text[: m.start()]
    text = "\n".join(line for line in text.splitlines() if not line.startswith(">"))
    header = "\n".join(f"{h}: {msg[h]}" for h in ("From", "To", "Date", "Subject") if msg[h])
    return Document(source=str(path), kind="email", pages=[f"{header}\n\n{text.strip()}"])
