"""Load a change order from PDF, image, .eml or plain text into page-tagged text."""

from __future__ import annotations

import email
import email.policy
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from PIL import Image, ImageOps

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
    ocr_pages: list[int] = field(default_factory=list)  # 1-based; text recovered by OCR
    pdf_bytes: bytes | None = None

    @property
    def is_scanned(self) -> bool:
        return bool(self.scanned_pages)

    @property
    def text(self) -> str:
        return "\n".join(self.pages)

    def tagged_text(self) -> str:
        return "\n\n".join(f"[[page {i}]]\n{p}" for i, p in enumerate(self.pages, 1))


def load(path: str | Path, ocr: bool | None = None) -> Document:
    """``ocr``: True/False forces OCR of scanned PDF pages on/off; None uses it when docling is installed."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, ocr)
    if suffix in IMAGE_SUFFIXES:
        # A photo or scan is just a PDF page with no text layer: convert it and reuse that path
        # (scan detection, OCR, and attaching the page image for the model).
        return _load_pdf(path, ocr, data=image_to_pdf(path), kind="image")
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


def _load_pdf(path: Path, ocr: bool | None = None, data: bytes | None = None, kind: str = "pdf") -> Document:
    data = data if data is not None else path.read_bytes()
    pages, scanned = [], []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append(text)
            if len(text.strip()) < SCANNED_PAGE_MIN_CHARS:
                scanned.append(i)
    if ocr and not ocr_available():
        raise RuntimeError("OCR requested but docling is not installed (uv sync --extra ocr)")
    # Text pages keep pdfplumber's exact text layer; only pages without one are OCR'd.
    source = path if kind == "pdf" else _stream(path.stem + ".pdf", data)
    recovered = _ocr_pages(source, scanned) if scanned and ocr is not False else {}
    ocr_done = []
    for i, text in recovered.items():
        if len(text.strip()) >= SCANNED_PAGE_MIN_CHARS:
            pages[i - 1] = text
            ocr_done.append(i)
    return Document(source=str(path), kind=kind, pages=pages, pdf_bytes=data,
                    scanned_pages=[i for i in scanned if i not in ocr_done], ocr_pages=sorted(ocr_done))


def ocr_available() -> bool:
    try:
        import docling  # noqa: F401
    except ImportError:
        return False
    return True


def _stream(name: str, data: bytes):
    """In-memory PDF for Docling; None when it is not installed (then no OCR happens anyway)."""
    if not ocr_available():
        return None
    from docling.datamodel.base_models import DocumentStream

    return DocumentStream(name=name, stream=io.BytesIO(data))


def _ocr_pages(source, page_nos: list[int]) -> dict[int, str]:
    """OCR the given 1-based pages of ``source`` (a path or Docling DocumentStream) with Docling.

    Docling is an optional dependency; returns {} when it is not installed.
    """
    if not ocr_available():
        return {}
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=True, do_table_structure=True)
    opts.ocr_options.force_full_page_ocr = True
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    doc = converter.convert(source).document
    out: dict[int, list[str]] = {i: [] for i in page_nos}
    for item, _level in doc.iterate_items():
        if not getattr(item, "prov", None) or item.prov[0].page_no not in out:
            continue
        # Tables carry no .text; render them as markdown rows so amounts stay on their row.
        text = item.export_to_markdown(doc=doc) if hasattr(item, "export_to_markdown") else getattr(item, "text", "")
        if text:
            out[item.prov[0].page_no].append(text)
    return {i: "\n".join(parts) for i, parts in out.items()}


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
