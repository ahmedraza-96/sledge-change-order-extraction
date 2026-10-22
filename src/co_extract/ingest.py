"""Load a change order from PDF, .eml or plain text into page-tagged text."""

from __future__ import annotations

import email
import email.policy
import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

# A page with less extractable text than this is treated as a scan.
SCANNED_PAGE_MIN_CHARS = 30


@dataclass
class Document:
    source: str
    kind: str  # "pdf" | "email" | "text"
    pages: list[str]
    scanned_pages: list[int] = field(default_factory=list)  # 1-based
    pdf_bytes: bytes | None = None

    @property
    def is_scanned(self) -> bool:
        return bool(self.scanned_pages)

    @property
    def text(self) -> str:
        return "\n".join(self.pages)

    def tagged_text(self) -> str:
        return "\n\n".join(f"[[page {i}]]\n{p}" for i, p in enumerate(self.pages, 1))


def load(path: str | Path) -> Document:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".eml":
        return _load_eml(path)
    return Document(source=str(path), kind="text", pages=[path.read_text(errors="replace")])


def _load_pdf(path: Path) -> Document:
    data = path.read_bytes()
    pages, scanned = [], []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            pages.append(text)
            if len(text.strip()) < SCANNED_PAGE_MIN_CHARS:
                scanned.append(i)
    return Document(source=str(path), kind="pdf", pages=pages, scanned_pages=scanned, pdf_bytes=data)


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
