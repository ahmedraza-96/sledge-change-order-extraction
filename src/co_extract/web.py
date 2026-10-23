"""Small web UI over ``pipeline.run``: upload or paste a change order, get fields with confidence."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from . import pipeline
from .cli import SUFFIXES
from .extract import ExtractionError, FixtureExtractor
from .providers import DEFAULT_MODELS, Provider, has_key, make_extractor

STATIC = Path(__file__).parent / "static"
# Demo mode replays the recorded LLM output the tests use, so it needs no key —
# but it only has recordings for these sample documents.
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
SAMPLES = {p.stem: p for p in sorted((FIXTURES / "docs").glob("*.txt"))}
DEMO = "demo"
LABELS = {Provider.claude: "Claude", Provider.gemini: "Gemini", Provider.openai: "OpenAI"}

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


@app.post("/api/extract")
def extract(
    provider: str = Form(...),
    model: str | None = Form(None),
    text: str | None = Form(None),
    name: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> dict:
    # A sync handler, so FastAPI runs it in a threadpool: the pipeline blocks on the LLM call.
    if file is not None and file.filename:
        filename = Path(file.filename).name
        data = file.file.read()
    elif text and text.strip():
        # Keep a sample's name so demo mode can find its recording.
        filename = (re.sub(r"[^\w-]", "", name or "") or "pasted") + ".txt"
        data = text.encode()
    else:
        raise HTTPException(400, "Upload a file or paste the change order text.")
    if Path(filename).suffix.lower() not in SUFFIXES:
        raise HTTPException(400, f"Unsupported file type; use one of {', '.join(sorted(SUFFIXES))}.")

    if provider == DEMO:
        ex = FixtureExtractor(FIXTURES / "llm")
    else:
        try:
            p = Provider(provider)
        except ValueError:
            raise HTTPException(400, f"Unknown provider {provider!r}.") from None
        if not has_key(p):
            raise HTTPException(400, f"No API key for {LABELS[p]} — set it in .env.")
        ex = make_extractor(p, model=(model or "").strip() or None)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        path.write_bytes(data)
        try:
            result = pipeline.run(path, ex)
        except ExtractionError as e:
            msg = str(e)
            if provider == DEMO:
                msg = (f"Demo mode only has recorded output for the sample documents "
                       f"({', '.join(SAMPLES)}). {msg}")
            raise HTTPException(502, msg) from e
    result["source"] = filename  # not the temp path
    return result
