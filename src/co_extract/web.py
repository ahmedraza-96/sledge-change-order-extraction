"""Small web UI over ``pipeline.run``: upload or paste a change order, get fields with confidence."""

from __future__ import annotations

import json
import os
import queue
import re
import secrets
import tempfile
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse

from . import history, pipeline
from .cli import SUFFIXES
from .extract import ExtractionError, FixtureExtractor
from .providers import DEFAULT_MODELS, Provider, has_key, make_extractor

STATIC = Path(__file__).parent / "static"
# Demo mode replays the recorded LLM output the tests use, so it needs no key —
# but it only has recordings for these sample documents.
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
SAMPLES = {p.stem: p for p in sorted((FIXTURES / "docs").glob("*.txt"))}
DEMO = "demo"
AUTO_ORDER = (Provider.openrouter, Provider.gemini, Provider.claude, Provider.openai)
LABELS = {Provider.claude: "Claude", Provider.gemini: "Gemini", Provider.openai: "OpenAI",
          Provider.openrouter: "OpenRouter (free models)"}

app = FastAPI(title="co-extract")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/providers")
def providers() -> list[dict]:
    out = [{"id": p.value, "label": LABELS[p], "default_model": DEFAULT_MODELS[p], "available": has_key(p)}
           for p in Provider]
    return out + [{"id": DEMO, "label": "Demo (recorded output, no API call)", "default_model": "fixture",
                   "available": bool(SAMPLES)}]


@app.get("/api/samples")
def samples() -> list[str]:
    return list(SAMPLES)


@app.get("/api/samples/{name}", response_class=PlainTextResponse)
def sample(name: str) -> str:
    if name not in SAMPLES:
        raise HTTPException(404, f"no sample named {name!r}")
    return SAMPLES[name].read_text(encoding="utf-8")


@app.get("/api/history")
def history_list(limit: int = Query(50, ge=1, le=200), before: int | None = None) -> list[dict]:
    """Past runs, newest first. Public by design: anyone who can open the app can see them."""
    return history.list_runs(limit=limit, before=before)


@app.get("/api/history/{run_id}")
def history_run(run_id: int) -> dict:
    run = history.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"no run {run_id}")
    return run


@app.delete("/api/history/{run_id}")
def history_delete(run_id: int, x_admin_key: str | None = Header(None)) -> dict:
    """Delete a run. Needs HISTORY_ADMIN_KEY: the list is public, deleting from it isn't.
    With no key configured, nobody can delete."""
    load_dotenv()
    expected = os.environ.get("HISTORY_ADMIN_KEY", "").strip()
    if not expected or not secrets.compare_digest((x_admin_key or "").encode(), expected.encode()):
        raise HTTPException(403, "Wrong or missing admin key.")
    if not history.delete_run(run_id):
        raise HTTPException(404, f"no run {run_id}")
    return {"deleted": run_id}


# Uploads are limited to these suffixes (cli.SUFFIXES), none of which a browser runs as a page.
MEDIA_TYPES = {".pdf": "application/pdf", ".txt": "text/plain; charset=utf-8", ".eml": "text/plain; charset=utf-8",
               ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
               ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff"}


@app.get("/api/history/{run_id}/document")
def history_document(run_id: int) -> Response:
    found = history.get_document(run_id)
    if found is None:
        raise HTTPException(404, f"no stored document for run {run_id}")
    filename, data = found
    media = MEDIA_TYPES.get(Path(filename).suffix.lower(), "application/octet-stream")
    safe = re.sub(r'[^\w.-]', "_", filename)
    return Response(data, media_type=media, headers={"X-Content-Type-Options": "nosniff",
                                                      "Content-Disposition": f'inline; filename="{safe}"'})


class _Job:
    """A validated request: what to read and which extractor reads it."""

    def __init__(self, filename: str, data: bytes, extractor, provider: str, source: str = "file"):
        self.filename, self.data, self.extractor, self.provider = filename, data, extractor, provider
        self.source = source  # "file" or "text", for the history

    def run(self, on_step=None) -> dict:
        started = time.monotonic()
        try:
            result = self._run(on_step)
        except Exception as e:
            # Failed runs go in the history too: what the client tried matters as much as what worked.
            msg = str(e) if isinstance(e, ExtractionError) else f"Unexpected error: {e}"
            self._record(error=msg, started=started)
            raise
        result["history_id"] = self._record(result=result, started=started)
        return result

    def _run(self, on_step) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / self.filename
            path.write_bytes(self.data)
            try:
                result = pipeline.run(path, self.extractor, on_step=on_step)
            except ExtractionError as e:
                if self.provider == DEMO:
                    raise ExtractionError(f"Demo mode only has recorded output for the sample documents "
                                          f"({', '.join(SAMPLES)}). {e}") from e
                raise
        result["source"] = self.filename  # not the temp path
        return result

    def _record(self, started: float, result: dict | None = None, error: str | None = None) -> int | None:
        return history.record(filename=self.filename, source=self.source, document=self.data,
                              provider=self.provider, result=result, error=error,
                              duration_ms=round((time.monotonic() - started) * 1000))


def _job(provider: str | None, model: str | None, text: str | None, name: str | None,
         file: UploadFile | None) -> _Job:
    """Check the request before any work starts; problems become a 400."""
    if file is not None and file.filename:
        filename = Path(file.filename).name
        data = file.file.read()
        source = "file"
    elif text and text.strip():
        # Keep a sample's name so demo mode can find its recording.
        filename = (re.sub(r"[^\w-]", "", name or "") or "pasted") + ".txt"
        data = text.encode()
        source = "text"
    else:
        raise HTTPException(400, "Upload a file or paste the change order text.")
    if Path(filename).suffix.lower() not in SUFFIXES:
        raise HTTPException(400, f"Unsupported file type; use one of {', '.join(sorted(SUFFIXES))}.")

    if provider is None:
        # The UI doesn't ask: use the first provider that has a key, free OpenRouter first.
        provider = next((p.value for p in AUTO_ORDER if has_key(p)), None)
        if provider is None:
            raise HTTPException(400, "No model is configured. Add OPENROUTER_API_KEY to the .env file and restart.")
    if provider == DEMO:
        return _Job(filename, data, FixtureExtractor(FIXTURES / "llm"), provider, source)
    try:
        p = Provider(provider)
    except ValueError:
        raise HTTPException(400, f"Unknown provider {provider!r}.") from None
    if not has_key(p):
        raise HTTPException(400, f"No API key for {LABELS[p]} — set it in .env.")
    return _Job(filename, data, make_extractor(p, model=(model or "").strip() or None), provider, source)


@app.post("/api/extract")
def extract(
    provider: str | None = Form(None),
    model: str | None = Form(None),
    text: str | None = Form(None),
    name: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> dict:
    # A sync handler, so FastAPI runs it in a threadpool: the pipeline blocks on the LLM call.
    job = _job(provider, model, text, name, file)
    try:
        return job.run()
    except ExtractionError as e:
        raise HTTPException(502, str(e)) from e


@app.post("/api/extract/stream")
def extract_stream(
    provider: str | None = Form(None),
    model: str | None = Form(None),
    text: str | None = Form(None),
    name: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> StreamingResponse:
    """Same as /api/extract, but streams progress as newline-delimited JSON:
    ``{"step": "extract"}`` … then ``{"result": {...}}`` or ``{"error": "..."}``."""
    job = _job(provider, model, text, name, file)
    events: queue.Queue = queue.Queue()

    def work():
        try:
            events.put({"result": job.run(on_step=lambda name, **info: events.put({"step": name, **info}))})
        except ExtractionError as e:
            events.put({"error": str(e)})
        except Exception as e:  # report it to the page instead of dropping the stream
            events.put({"error": f"Unexpected error: {e}"})
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (event := events.get()) is not None:
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
