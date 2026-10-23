"""Benchmark: how should a scanned change order be read?

Two pipelines, same images, same ground truth (data/truth):

  vision   the page image goes straight to OpenRouter's vision model
           (nvidia/nemotron-3.5-super-vl:free), which reads it itself
  docling  Docling for IBM watsonx OCRs the image, then OpenRouter's text model
           (nvidia/nemotron-3-super-120b-a12b:free) writes the JSON from that text

The 10 generated samples are rendered as scanned-looking images first (slight tilt, blur,
JPEG), so both pipelines actually have to *read*: a digital PDF would hand them exact text.
The repair pass is off, so each pipeline makes exactly one model call per document and the
comparison is first-read accuracy. Results are cached per document; rerun to resume.

    uv run python scripts/benchmark_ocr.py                  # -> reports/benchmark-ocr.md
    uv run python scripts/benchmark_ocr.py --retry-failed   # also re-run documents that errored
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
import time
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from PIL import Image, ImageFilter

from co_extract import pipeline
from co_extract.evaluate import field_correct, load_truths
from co_extract.extract import ExtractionError
from co_extract.ingest import load, text_to_pdf
from co_extract.openrouter import OpenRouterExtractor
from co_extract.schema import FIELD_SPECS

ROOT = Path(__file__).resolve().parents[1]
SAMPLES, TRUTH = ROOT / "data" / "samples", ROOT / "data" / "truth"
IMAGES = ROOT / "test_samples" / "benchmark"
RUNS = ROOT / "reports" / "runs" / "benchmark"
REPORT = ROOT / "reports" / "benchmark-ocr.md"
FIELDS = [*FIELD_SPECS, "line_items"]
PIPELINES = {"vision": "OpenRouter vision model reads the image",
             "docling": "IBM Docling OCR, then OpenRouter text model"}
DOCLING_ENV = ("DOCLING_SERVICE_URL", "DOCLING_SERVICE_API_KEY")
SWITCH = "DOCLING_OCR_ENABLED"


# ---------- test images ----------

def _scan_look(page: Image.Image, rng: random.Random) -> Image.Image:
    """A flatbed-scan / careful-photo look: small tilt, slight blur and grain. Still legible."""
    img = page.convert("RGB").rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor="white", resample=Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    noise = Image.effect_noise(img.size, 10).convert("RGB")
    return Image.blend(img, noise, 0.04)


def build_images() -> dict[str, Path]:
    """Render every ground-truth sample as an image (multi-page ones as a multi-page TIFF)."""
    IMAGES.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    out = {}
    for src in sorted(SAMPLES.iterdir()):
        if not (TRUTH / f"{src.stem}.json").exists():
            continue
        if src.suffix.lower() == ".pdf":
            pdf_bytes = src.read_bytes()
        else:  # email and text: typeset onto a page first, as the app does
            pdf_bytes = text_to_pdf(load(src).pages)
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            pages = [_scan_look(Image.open(io.BytesIO(p.get_pixmap(dpi=150).tobytes("png"))), rng) for p in pdf]
        if len(pages) == 1:
            path = IMAGES / f"{src.stem}.jpg"
            pages[0].save(path, quality=85)
        else:
            path = IMAGES / f"{src.stem}.tiff"
            pages[0].save(path, save_all=True, append_images=pages[1:], compression="jpeg", quality=85)
        out[src.stem] = path
    return out


# ---------- running ----------

def run_one(name: str, image: Path, saved: dict) -> dict:
    """One document through one pipeline, with Docling switched on only for 'docling'."""
    if name == "vision":
        os.environ.pop(SWITCH, None)
    else:
        os.environ.update(saved)
        os.environ[SWITCH] = "true"
    t0 = time.perf_counter()
    try:
        result = pipeline.run(image, OpenRouterExtractor(), allow_repair=False)
        result["benchmark"] = {"seconds": round(time.perf_counter() - t0, 1), "error": None}
    except ExtractionError as e:
        result = {"benchmark": {"seconds": round(time.perf_counter() - t0, 1), "error": str(e)}}
    return result


def run_all(images: dict[str, Path]) -> dict[str, dict[str, dict]]:
    load_dotenv(ROOT / ".env")
    saved = {k: os.environ[k] for k in DOCLING_ENV if os.environ.get(k)}
    if len(saved) < 2:
        sys.exit("Docling isn't configured: set DOCLING_SERVICE_URL and DOCLING_SERVICE_API_KEY in .env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY isn't set in .env")
    results: dict[str, dict[str, dict]] = {p: {} for p in PIPELINES}
    for stem, image in images.items():
        for name in PIPELINES:
            cache = RUNS / name / f"{stem}.json"
            # Cached runs are reused, failed ones too: retrying spends quota, so only --retry-failed does it.
            if cache.exists() and ("--retry-failed" not in sys.argv
                                   or not json.loads(cache.read_text()).get("benchmark", {}).get("error")):
                results[name][stem] = json.loads(cache.read_text())
                print(f"{stem:26} {name:8} cached", flush=True)
                continue
            r = run_one(name, image, saved)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(r, indent=2))
            results[name][stem] = r
            b = r["benchmark"]
            print(f"{stem:26} {name:8} {b['seconds']:6.1f}s  " + (f"ERROR {b['error'][:90]}" if b["error"]
                  else f"model={r['meta']['model']}  ocr={r['meta'].get('ocr')}"), flush=True)
    return results


# ---------- scoring ----------

def score(results: dict[str, dict[str, dict]], truths: dict[str, dict]) -> dict:
    out = {}
    for name, docs in results.items():
        per_doc, per_field = {}, {f: [0, 0] for f in FIELDS}
        right = total = confident_wrong = 0
        for stem, r in docs.items():
            if r["benchmark"]["error"]:
                per_doc[stem] = {"error": r["benchmark"]["error"], "seconds": r["benchmark"]["seconds"]}
                continue
            truth, wrong = truths[stem], []
            for f in FIELDS:
                ok = field_correct(f, r["change_order"].get(f), truth.get(f))
                per_field[f][0] += ok
                per_field[f][1] += 1
                right += ok
                total += 1
                if not ok:
                    conf = r["fields"][f]["confidence"]
                    wrong.append(f"{f} ({conf:.2f})")
                    confident_wrong += conf >= 0.7
            per_doc[stem] = {"correct": len(FIELDS) - len(wrong), "wrong": wrong, "seconds": r["benchmark"]["seconds"],
                             "doc_conf": r["document_confidence"], "model": r["meta"]["model"],
                             "ocr_error": (r["meta"].get("ocr") or {}).get("error")}
        done = [d for d in per_doc.values() if "error" not in d]
        out[name] = {"accuracy": right / total if total else 0, "right": right, "total": total,
                     "confident_wrong": confident_wrong, "docs_ok": len(done), "docs": per_doc,
                     "per_field": per_field,
                     "mean_seconds": sum(d["seconds"] for d in done) / len(done) if done else 0}
    return out


def render(s: dict, images: dict[str, Path]) -> str:
    a, b = s["vision"], s["docling"]
    lines = ["# OCR benchmark: OpenRouter vision vs IBM Docling", "",
             "Same 10 scanned-looking images of the ground-truth samples, one model call each (no repair pass).", "",
             "| | Vision (`nemotron-3.5-super-vl`) | Docling OCR → text model (`nemotron-3-super`) |", "|---|---|---|",
             f"| Field accuracy | **{a['accuracy']:.1%}** ({a['right']}/{a['total']}) | **{b['accuracy']:.1%}** ({b['right']}/{b['total']}) |",
             f"| Wrong but confident (≥0.70) | {a['confident_wrong']} | {b['confident_wrong']} |",
             f"| Documents completed | {a['docs_ok']}/{len(images)} | {b['docs_ok']}/{len(images)} |",
             f"| Mean time per document | {a['mean_seconds']:.0f} s | {b['mean_seconds']:.0f} s |", "",
             "## Per document", "", "| Document | Vision | Docling |", "|---|---|---|"]

    def cell(d):
        if not d:
            return "not run"
        if "error" in d:
            return f"error: {d['error'][:60]}"
        wrong = ", ".join(d["wrong"]) or "all correct"
        note = f" · OCR failed: {d['ocr_error'][:40]}" if d.get("ocr_error") else ""
        return f"{d['correct']}/{len(FIELDS)} · {d['seconds']:.0f}s · wrong: {wrong}{note}"

    for stem in images:
        lines.append(f"| {stem} | {cell(a['docs'].get(stem))} | {cell(b['docs'].get(stem))} |")
    lines += ["", "## Per field", "", "| Field | Vision | Docling |", "|---|---|---|"]
    for f in FIELDS:
        va, vb = a["per_field"][f], b["per_field"][f]
        if va[1] or vb[1]:
            lines.append(f"| {f} | {va[0]}/{va[1]} | {vb[0]}/{vb[1]} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    images = build_images()
    print(f"{len(images)} test images in {IMAGES}", flush=True)
    results = run_all(images)
    s = score(results, load_truths(TRUTH))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(s, images), encoding="utf-8")
    for name in PIPELINES:
        print(f"{name:8} accuracy {s[name]['accuracy']:.1%}  confident-wrong {s[name]['confident_wrong']}  "
              f"mean {s[name]['mean_seconds']:.0f}s  docs {s[name]['docs_ok']}/{len(images)}")
    print(f"-> {REPORT}")


if __name__ == "__main__":
    main()
