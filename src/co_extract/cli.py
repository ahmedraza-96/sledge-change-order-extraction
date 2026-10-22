from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from . import evaluate, pipeline
from .extract import MODEL, ClaudeExtractor, ExtractionError, Extractor, FixtureExtractor

app = typer.Typer(add_completion=False, help="Extract structured fields from construction change orders.")

SUFFIXES = {".pdf", ".txt", ".eml"}


def _extractor(fixtures: Path | None, text_only: bool, model: str) -> Extractor:
    if fixtures:
        return FixtureExtractor(fixtures)
    load_dotenv()
    return ClaudeExtractor(model=model, send_pdf=not text_only)


def _inputs(paths: list[Path]) -> list[Path]:
    out = []
    for p in paths:
        out += sorted(q for q in p.iterdir() if q.suffix.lower() in SUFFIXES) if p.is_dir() else [p]
    return out


@app.command()
def extract(
    paths: Annotated[list[Path], typer.Argument(help="Files or directories")],
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Write JSON here (dir if several inputs)")] = None,
    fixtures: Annotated[Path | None, typer.Option(help="Replay recorded LLM outputs instead of calling the API")] = None,
    no_repair: Annotated[bool, typer.Option("--no-repair", help="Skip the validation-driven repair pass")] = False,
    text_only: Annotated[bool, typer.Option("--text-only", help="Send extracted text only, not the PDF")] = False,
    model: str = MODEL,
):
    """Extract one or more change orders to JSON."""
    ex = _extractor(fixtures, text_only, model)
    files = _inputs(paths)
    failed = 0
    for f in files:
        try:
            result = pipeline.run(f, ex, allow_repair=not no_repair)
        except ExtractionError as e:
            failed += 1
            typer.echo(f"{f}: extraction failed: {e}", err=True)
            continue
        text = json.dumps(result, indent=2)
        if out is None:
            typer.echo(text)
        else:
            target = out / f"{f.stem}.json" if len(files) > 1 or out.is_dir() else out
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
            typer.echo(f"{f.name}: confidence {result['document_confidence']:.2f}, "
                       f"review {result['needs_review'] or 'none'} -> {target}", err=True)
    raise typer.Exit(1 if failed else 0)


@app.command("eval")
def eval_cmd(
    samples: Path = Path("data/samples"),
    truth: Path = Path("data/truth"),
    runs: Annotated[Path, typer.Option(help="Where per-document results are cached")] = Path("reports/runs"),
    report: Path = Path("reports/eval.md"),
    cached: Annotated[bool, typer.Option("--cached", help="Re-score cached results without calling the API")] = False,
    fixtures: Annotated[Path | None, typer.Option(help="Replay recorded LLM outputs")] = None,
    model: str = MODEL,
):
    """Run the pipeline over the labelled samples and write a markdown report."""
    truths = evaluate.load_truths(truth)
    runs.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    ex = None if cached else _extractor(fixtures, False, model)
    for f in _inputs([samples]):
        if f.stem not in truths:
            continue
        cache = runs / f"{f.stem}.json"
        if cached:
            if not cache.exists():
                typer.echo(f"{f.stem}: no cached run, skipping", err=True)
                continue
            results[f.stem] = json.loads(cache.read_text())
            continue
        try:
            results[f.stem] = pipeline.run(f, ex)
        except ExtractionError as e:
            typer.echo(f"{f.stem}: extraction failed: {e}", err=True)
            continue
        cache.write_text(json.dumps(results[f.stem], indent=2))
        typer.echo(f"{f.stem}: done ({results[f.stem]['meta']['seconds']}s)", err=True)
    if not results:
        typer.echo("nothing to score", err=True)
        raise typer.Exit(1)
    s = evaluate.score(results, truths)
    used_model = next(iter(results.values()))["meta"]["model"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(evaluate.render_markdown(s, used_model))
    typer.echo(f"field accuracy {s['field_accuracy']:.1%}, error recall {evaluate._pct(s['error_recall'])}, "
               f"ECE {s['ece']:.3f} -> {report}", err=True)


def main() -> None:
    app()
