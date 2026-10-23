"""Read documents with Docling for IBM watsonx (hosted OCR). Off unless switched on.

Opt in with DOCLING_OCR_ENABLED=true (plus the service URL and key). A benchmark on 10
scanned-looking samples (scripts/benchmark_ocr.py) found the vision model reading the image
directly was more accurate (95.5% against 91.9%), so it is not the default.

When it is on, ``ingest`` renders every document as page images (text and email typeset
first) and sends them here; they come back as text, tables kept as rows, and that text is
what the model reads and what quotes are checked against.

This is best effort: any failure (network, timeout, usage limit, bad key) returns no
text plus the reason, and the document keeps its own text (a photo stays a scan).
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger("uvicorn.error")  # shows in the `co-extract serve` console

ENGINE = "docling-watsonx"
JOB_TIMEOUT = 90.0  # seconds for the whole conversion, including the queue


def available() -> bool:
    """Switched on and configured. Credentials alone don't turn it on."""
    return (os.environ.get("DOCLING_OCR_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
            and bool(os.environ.get("DOCLING_SERVICE_URL") and os.environ.get("DOCLING_SERVICE_API_KEY")))


def read_pages(pdf_bytes: bytes, page_nos: list[int], name: str = "document.pdf") -> tuple[dict[int, str], str | None]:
    """OCR ``page_nos`` (1-based) of a PDF with the hosted service.

    Returns (page -> text, error). On any failure the dict is empty and ``error`` says why.
    """
    if not page_nos:
        return {}, None
    try:
        doc = _convert(pdf_bytes, (min(page_nos), max(page_nos)), name)
    except Exception as e:  # never let OCR break extraction
        error = f"{type(e).__name__}: {e}"
        log.warning("Docling OCR failed for %s: %s", name, error)
        return {}, error
    texts = page_texts(doc, page_nos)
    if not texts:
        log.warning("Docling OCR returned no text for %s (pages %s)", name, page_nos)
        return {}, "Docling returned no text for these pages"
    return texts, None


def _convert(pdf_bytes: bytes, page_range: tuple[int, int], name: str):
    from docling.datamodel.service.options import ConvertDocumentsOptions
    from docling.service_client import DoclingServiceClient

    options = ConvertDocumentsOptions(
        to_formats=["json"],      # the structured document, so text can be split by page
        do_ocr=True,
        force_ocr=True,           # these pages have no usable text layer by definition
        do_table_structure=True,  # keep each line item's amount on its row
        include_images=False,     # we only need text; skipping pictures keeps the job small
        page_range=page_range,
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / (Path(name).stem + ".pdf")
        path.write_bytes(pdf_bytes)
        with DoclingServiceClient(url=os.environ["DOCLING_SERVICE_URL"], api_key=os.environ["DOCLING_SERVICE_API_KEY"],
                                  job_timeout=JOB_TIMEOUT) as client:
            return client.convert(source=path, options=options).document


def page_texts(doc, page_nos: list[int]) -> dict[int, str]:
    """Text of each requested page, in reading order. Tables become markdown rows."""
    out: dict[int, list[str]] = {p: [] for p in page_nos}
    for item, _level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov or prov[0].page_no not in out:
            continue
        # Tables carry no .text; markdown keeps each amount on the same line as its description.
        text = item.export_to_markdown(doc=doc) if hasattr(item, "export_to_markdown") else getattr(item, "text", "")
        if text:
            out[prov[0].page_no].append(text)
    return {p: "\n".join(parts) for p, parts in out.items() if parts}
