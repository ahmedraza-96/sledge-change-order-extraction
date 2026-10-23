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
    ocr_pages: list[int] = field(default_factory=list)  # 1-based; text read by hosted Docling OCR
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
    """``on_ocr`` is called just before pages are sent to Docling, so a UI can say so."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, on_ocr=on_ocr)
    if suffix in IMAGE_SUFFIXES:
        # A photo or scan is just a PDF page with no text layer: convert it and reuse that path
        # (scan detection, OCR, and attaching the page image so the model reads it).
        return _load_pdf(path, data=image_to_pdf(path), kind="image", on_ocr=on_ocr)
    if suffix == ".eml":
        return _load_eml(path)
    return Document(source=str(path), kind="text", pages=[path.read_text(encoding="utf-8", errors="replace")])


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


def _load_pdf(path: Path, data: bytes | None = None, kind: str = "pdf",
              on_ocr: Callable[[], None] | None = None) -> Document:
    data = data if data is not None else path.read_bytes()
    pages, scanned = [], []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append(text)
            if len(text.strip()) < SCANNED_PAGE_MIN_CHARS:
                scanned.append(i)
    doc = Document(source=str(path), kind=kind, pages=pages, scanned_pages=scanned, pdf_bytes=data)
    if scanned and docling_ocr.available():
        if on_ocr:
            on_ocr()
        recovered, doc.ocr_error = docling_ocr.read_pages(data, scanned, path.name)
        # Text pages keep pdfplumber's exact text; only pages without one take OCR text.
        for i, text in recovered.items():
            if len(text.strip()) >= SCANNED_PAGE_MIN_CHARS:
                pages[i - 1] = text
                doc.ocr_pages.append(i)
        doc.scanned_pages = [i for i in scanned if i not in doc.ocr_pages]
    return doc


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
