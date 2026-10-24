"""A record of every extraction the web app runs: the document as uploaded and the full result.

SQLite, so it needs no server. The path comes from ``HISTORY_DB`` (default ``data/history.db``);
``HISTORY_DB=off`` turns recording off. Recording never breaks an extraction: a database error is
logged and the result is returned as usual.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger("uvicorn.error")  # shows in the `co-extract serve` console

DEFAULT_PATH = "data/history.db"
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024  # larger files keep their result but not the file

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    filename TEXT NOT NULL,
    source TEXT NOT NULL,             -- 'file' or 'text'
    provider TEXT,
    model TEXT,
    status TEXT NOT NULL,             -- 'ok' or 'error'
    error TEXT,
    document_confidence REAL,
    review_count INTEGER,
    is_change_order INTEGER,
    duration_ms INTEGER,
    document BLOB,
    result TEXT
)
"""
SUMMARY = ("id, created_at, filename, source, provider, model, status, error, document_confidence, "
           "review_count, is_change_order, duration_ms, document IS NOT NULL AS has_document")


def db_path() -> Path | None:
    value = os.environ.get("HISTORY_DB", "").strip() or DEFAULT_PATH
    return None if value.lower() in ("off", "false", "0", "none") else Path(value)


def _connect(path: Path) -> sqlite3.Connection:
    # A connection per call: the web app runs each extraction in its own thread.
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(SCHEMA)
    return conn


def record(*, filename: str, source: str, document: bytes, provider: str, result: dict | None = None,
           error: str | None = None, duration_ms: int | None = None) -> int | None:
    """Save one run; returns its id, or None when recording is off or failed."""
    path = db_path()
    if path is None:
        return None
    meta = (result or {}).get("meta", {})
    row = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "filename": filename,
        "source": source,
        "provider": meta.get("provider") or provider,
        "model": meta.get("model"),
        "status": "error" if error else "ok",
        "error": error,
        "document_confidence": result.get("document_confidence") if result else None,
        "review_count": len(result.get("needs_review", [])) if result else None,
        "is_change_order": (None if not result else
                            int(not any(i.get("code") == "not_a_change_order" for i in result.get("issues", [])))),
        "duration_ms": duration_ms,
        "document": document if len(document) <= MAX_DOCUMENT_BYTES else None,
        "result": json.dumps(result) if result else None,
    }
    try:
        with closing(_connect(path)) as conn, conn:
            cur = conn.execute(f"INSERT INTO runs ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                               tuple(row.values()))
            return cur.lastrowid
    except (sqlite3.Error, OSError) as e:
        log.warning("history: could not record %s: %s", filename, e)
        return None


def list_runs(limit: int = 50, before: int | None = None) -> list[dict]:
    """Newest first; ``before`` pages back past a run id."""
    path = db_path()
    if path is None or not path.exists():
        return []
    sql, args = f"SELECT {SUMMARY} FROM runs", []
    if before is not None:
        sql, args = sql + " WHERE id < ?", [before]
    with closing(_connect(path)) as conn:
        rows = conn.execute(sql + " ORDER BY id DESC LIMIT ?", [*args, limit]).fetchall()
    return [_summary(r) for r in rows]


def get_run(run_id: int) -> dict | None:
    path = db_path()
    if path is None or not path.exists():
        return None
    with closing(_connect(path)) as conn:
        row = conn.execute(f"SELECT {SUMMARY}, result FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    out = _summary(row)
    out["result"] = json.loads(row["result"]) if row["result"] else None
    return out


def get_document(run_id: int) -> tuple[str, bytes] | None:
    path = db_path()
    if path is None or not path.exists():
        return None
    with closing(_connect(path)) as conn:
        row = conn.execute("SELECT filename, document FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None or row["document"] is None:
        return None
    return row["filename"], bytes(row["document"])


def _summary(row: sqlite3.Row) -> dict:
    out = {k: row[k] for k in row.keys() if k != "result"}
    out["has_document"] = bool(out["has_document"])
    if out["is_change_order"] is not None:
        out["is_change_order"] = bool(out["is_change_order"])
    return out
