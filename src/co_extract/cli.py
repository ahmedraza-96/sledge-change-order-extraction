from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from . import evaluate, pipeline
from .extract import ExtractionError
from .ingest import IMAGE_SUFFIXES, load
from .providers import Provider, make_extractor
from .verify import VerifyError, make_verifier

app = typer.Typer(add_completion=False, help="Extract structured fields from construction change orders.")

SUFFIXES = {".pdf", ".txt", ".eml", *IMAGE_SUFFIXES}

VerifyOpt = Annotated[str, typer.Option("--verify", help="Check each field with a second model: off | decisions "
                                                         "(OpenAI Decisions API, needs OPENAI_API_KEY)")]
VerifyFixturesOpt = Annotated[Path | None, typer.Option(help="Replay recorded verifier answers")]


def _verifier(name: str, fixtures: Path | None):
    try:
        return make_verifier(name, fixtures)
    except VerifyError as e:
        typer.echo(f"verifier: {e}", err=True)
        raise typer.Exit(1) from e


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
    provider: Annotated[Provider, typer.Option(help="Which model backend to call")] = Provider.claude,
    model: Annotated[str | None, typer.Option(help="Override the provider's default model")] = None,
    verify: VerifyOpt = "off",
    verify_fixtures: VerifyFixturesOpt = None,
):
    """Extract one or more change orders to JSON."""
    ex = make_extractor(provider, model, fixtures, text_only)
    verifier = _verifier(verify, verify_fixtures)
    files = _inputs(paths)
    failed = 0
    for f in files:
        try:
            result = pipeline.run(f, ex, allow_repair=not no_repair, verifier=verifier)
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
    runs: Annotated[Path | None, typer.Option(help="Where per-document results are cached "
                                                   "(default: reports/runs/<provider>)")] = None,
    report: Annotated[Path | None, typer.Option(help="Default: reports/eval-<provider>.md")] = None,
    cached: Annotated[bool, typer.Option("--cached", help="Re-score cached results without calling the API")] = False,
    resume: Annotated[bool, typer.Option("--resume", help="Reuse documents already cached and only call "
                                                          "the API for the rest")] = False,
    fixtures: Annotated[Path | None, typer.Option(help="Replay recorded LLM outputs")] = None,
    provider: Annotated[Provider, typer.Option(help="Which model backend to call")] = Provider.claude,
    model: Annotated[str | None, typer.Option(help="Override the provider's default model")] = None,
    verify: VerifyOpt = "off",
    verify_fixtures: VerifyFixturesOpt = None,
):
    """Run the pipeline over the labelled samples and write a markdown report.

    With ``--verify``, cached runs that have no verifier answers yet are verified too and written
    back, so existing extractions can be checked without calling the extractor again."""
    # Scoped per provider: a shared cache would let one provider's runs overwrite the
    # other's, and --cached would then silently score a mix of the two.
    runs = runs or Path("reports/runs") / provider.value
    report = report or Path(f"reports/eval-{provider.value}.md")
    truths = evaluate.load_truths(truth)
    runs.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    ex = None if cached else make_extractor(provider, model, fixtures)
    verifier = _verifier(verify, verify_fixtures)
    for f in _inputs([samples]):
        if f.stem not in truths:
            continue
        cache = runs / f"{f.stem}.json"
        if cached or (resume and cache.exists()):
            if not cache.exists():
                typer.echo(f"{f.stem}: no cached run, skipping", err=True)
                continue
            results[f.stem] = json.loads(cache.read_text())
            if resume:
                typer.echo(f"{f.stem}: cached", err=True)
            # Not yet verified, or the last attempt failed.
            if verifier and "model" not in results[f.stem]["meta"].get("verify", {}):
                pipeline.attach_verification(results[f.stem], load(f), verifier)
                typer.echo(f"{f.stem}: verify {results[f.stem]['meta'].get('verify')}", err=True)
            elif "model" in results[f.stem]["meta"].get("verify", {}):
                # Re-score with the current W_VERIFIER, free: the probabilities are in the cache.
                pipeline.apply_verifier(results[f.stem])
            cache.write_text(json.dumps(results[f.stem], indent=2))
            continue
        try:
            results[f.stem] = pipeline.run(f, ex, verifier=verifier)
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
    report.write_text(evaluate.render_markdown(s, used_model), encoding="utf-8")
    typer.echo(f"field accuracy {s['field_accuracy']:.1%}, error recall {evaluate._pct(s['error_recall'])}, "
               f"ECE {s['ece']:.3f} -> {report}", err=True)
    # A partial run's headline numbers are not comparable to a full one, and read far
    # better than they deserve: documents that failed are usually the awkward ones.
    if (missing := len(truths) - len(results)) > 0:
        typer.echo(f"WARNING: scored {len(results)}/{len(truths)} documents ({missing} missing). "
                   f"These numbers are not a usable baseline — rerun with --resume to fill the gaps.", err=True)
        raise typer.Exit(2)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on")] = 8000,
):
    """Run the web UI: upload or paste a change order, see fields with confidence."""
    import uvicorn

    typer.echo(f"co-extract UI on http://{host}:{port}", err=True)
    uvicorn.run("co_extract.web:app", host=host, port=port)


def main() -> None:
    app()
