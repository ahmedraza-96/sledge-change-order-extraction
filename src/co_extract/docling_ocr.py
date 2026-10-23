"""Read scanned pages and photos with Docling for IBM watsonx (hosted OCR).

Pages with no text layer can't be grounded or cross-checked, and a text-only model can't
read them at all. When ``DOCLING_SERVICE_URL`` and ``DOCLING_SERVICE_API_KEY`` are set,
those pages are sent to IBM's hosted Docling and come back as text, tables kept as rows.

This is best effort: any failure (network, timeout, usage limit, bad key) returns no
text plus the reason, and the page is handled as a scan exactly as before. Text PDFs,
emails and .txt never reach this module.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ENGINE = "docling-watsonx"
JOB_TIMEOUT = 90.0  # seconds for the whole conversion, including the queue


def available() -> bool:
    return bool(os.environ.get("DOCLING_SERVICE_URL") and os.environ.get("DOCLING_SERVICE_API_KEY"))


def read_pages(pdf_bytes: bytes, page_nos: list[int], name: str = "document.pdf") -> tuple[dict[int, str], str | None]:
    """OCR ``page_nos`` (1-based) of a PDF with the hosted service.

    Returns (page -> text, error). On any failure the dict is empty and ``error`` says why.
    """
    if not page_nos:
        return {}, None
    try:
        doc = _convert(pdf_bytes, (min(page_nos), max(page_nos)), name)
    except Exception as e:  # never let OCR break extraction
        return {}, f"{type(e).__name__}: {e}"
    return page_texts(doc, page_nos), None


def _convert(pdf_bytes: bytes, page_range: tuple[int, int], name: str):
    from docling.datamodel.service.options import ConvertDocumentsRequestOptions
    from docling.service_client import DoclingServiceClient

    options = ConvertDocumentsRequestOptions(
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
