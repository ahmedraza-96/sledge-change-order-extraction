# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                    # install (Python 3.12+, uv_build backend)
uv run pytest                              # offline tests; `-m 'not live'` is the default in pyproject
uv run pytest tests/test_units.py::test_parse_money     # single test
uv run pytest -m live                      # 2 real API calls, needs ANTHROPIC_API_KEY in .env

uv run python scripts/gen_dataset.py       # (re)generate data/samples + data/truth — required before `eval`
uv run co-extract extract <file|dir> [-o out/] [--provider claude|gemini|openai] [--model ...]
uv run co-extract eval --provider gemini   # -> reports/eval-gemini.md, caches runs in reports/runs/gemini/
uv run co-extract eval --cached            # re-score cached runs without calling the API
uv run co-extract serve                    # web UI (FastAPI) on http://127.0.0.1:8000
```

`--provider` defaults to `claude` (`ANTHROPIC_API_KEY`); `gemini` needs `GEMINI_API_KEY` and defaults to
`gemini-3.5-flash`; `openai` needs `OPENAI_API_KEY` and defaults to `gpt-5-mini`. Gemini Pro models need billing enabled — on the free tier they report quota `limit: 0`.

The free Gemini tier caps flash at ~5 requests/min **and ~20/day**, which is less than one full eval, so use
`--resume` to spread a run across quota windows (it reuses cached documents and only calls for the rest).
`eval` exits 2 with a warning when it scores fewer documents than it has truth files for — a partial run's
headline numbers read better than a full one's, because the documents that fail are the awkward ones.

Work offline without a key by replaying recorded model output:

```bash
uv run co-extract extract tests/fixtures/docs/misread_co.txt --fixtures tests/fixtures/llm
```

`data/samples/`, `data/truth/` and `uv.lock` are gitignored build output — regenerate, don't commit.
There is no linter or formatter configured.

## Architecture

One pass per document: `ingest -> extract (Claude) -> normalize -> validate -> repair once? -> confidence -> JSON`,
orchestrated by `pipeline.run`. The design goal is not just accuracy but *knowing which fields are wrong*,
so every field carries a confidence score and fields below `REVIEW_THRESHOLD` (0.7) land in `needs_review`.
`WRITEUP.md` is the rationale document — read it before changing confidence weights or validation rules.

**Two schema layers (`schema.py`).** `RawExtraction` is the LLM contract and is deliberately **all strings** —
the model never coerces types. `FIELD_SPECS` maps each field name to a `FieldKind`, and `normalize.py` does the
coercion deterministically so it can be unit-tested. `FIELD_SPECS` is the single source of truth: it drives the
prompt field list (via `FIELD_GUIDE`), normalization, the output envelope and the eval field list. Adding a field
means touching `FIELD_SPECS`, `FIELD_GUIDE` and the `RawExtraction` body — a module-level `assert` catches drift.
`line_items` is handled separately everywhere (it is not in `FIELD_SPECS`).

**Extraction is behind a Protocol.** `extract.Extractor` has four implementations: `ClaudeExtractor`
(`client.beta.messages.parse` with `output_format=RawExtraction`), `GeminiExtractor` in `gemini.py`
(`client.models.generate_content` with `response_schema=RawExtraction`), `OpenAIExtractor` in
`openai_extractor.py` (`client.responses.parse` with `text_format=RawExtraction`, strict mode), and `FixtureExtractor`, which replays
`tests/fixtures/llm/<stem>.json` (and `<stem>.repair.json` for the repair pass). `providers.make_extractor` builds one by name for both
the CLI and the web app (`web.py` + `static/index.html`, whose Demo option is the fixture path). Every test and offline demo goes
through the fixture path; never add a code path that only works with a live client. Scanned PDFs (or `send_pdf=True`)
attach the PDF bytes alongside the page-tagged text.

**The prompt is provider-neutral and must stay that way.** `SYSTEM_PROMPT`, `document_prompt()` and
`repair_prompt()` live in `extract.py`; each provider only wraps that text in its own content format
(`build_content` for Anthropic blocks, `build_parts` for Gemini parts, `build_input` for OpenAI input items). The eval is a provider comparison, so any
prompt change must go in the shared helpers — `test_all_providers_send_identical_prompt_text` pins this.
`LLMResult.provider` flows into `meta.provider` so a cached run self-identifies, and eval caches and reports are
scoped per provider (`reports/runs/<provider>/`, `reports/eval-<provider>.md`) — a shared cache would let one
provider overwrite the other and make `--cached` score a mix.

Gemini's `response_schema` takes the `RawExtraction` pydantic model directly; the SDK inlines the `$ref`s pydantic
emits for the 21 repeated `ExtractedField` properties. Note Gemini Flash self-reports `confidence` near 1.0 on
nearly every field, so `W_MODEL` carries almost no discriminative signal there — grounding and the rules do the work.

**The repair pass** fires once when validation produces a `hard` issue or anything in `ARITHMETIC_CODES`. The model
re-reads the document with the failed checks in the prompt; `pipeline.run` keeps whichever attempt has fewer hard
issues, then fewer issues overall. If an arithmetic mismatch survives, the source document is presumed wrong: values
are kept **as printed** and the issue message says so. Do not "fix" documents.

**Issue severity is load-bearing.** `hard` (unparseable value, missing CO number) triggers repair and caps document
confidence at 0.5; `soft` only lowers field confidence. Each issue lists the `fields` it involves, and
`confidence.py` multiplies those fields' scores by `ISSUE_PENALTY` — so `Issue.fields` is how a rule reaches
confidence scoring.

**Confidence (`confidence.py`)** weights verifiable signals over the model's self-report: grounding 0.5
(rapidfuzz match of the evidence quote against the page text), model confidence 0.2, checks 0.3. Two hard overrides:
a value absent from its own evidence quote is capped at `UNSUPPORTED_CAP` (0.3), and disagreement with `heuristics.py`
halves the score. Scanned pages can't be grounded, so `grounding()` returns `None` and they get a neutral score plus a
penalty — deliberately pushing scans toward review. Images (`ingest.IMAGE_SUFFIXES`) are converted to PDF pages by `image_to_pdf` and then follow the scanned-PDF path
exactly (`Document.kind == "image"`). If docling is installed (`uv sync --extra ocr`), `ingest._ocr_pages`
OCRs pages with no text layer; those move from `scanned_pages` to `ocr_pages`, are grounded normally and take the milder
`OCR_PENALTY`. `tests/conftest.py` stubs `_ocr_pages` for every non-live test, so the suite never runs real OCR. Document confidence is driven by the weakest `CRITICAL_FIELDS`.

**`heuristics.py` is a cross-check, not a fallback.** A high-precision labelled-value regex extractor that abstains
(`agrees()` returns `None`) rather than guessing. Keep it independent of the LLM path; its value is disagreeing.

**`evaluate.py`** scores cached runs against `data/truth`. The headline metric is **error recall** — the share of
wrong fields that were flagged — not raw accuracy. It also reports calibration buckets/ECE, review precision, and
whether each truth file's `_expected_issues` actually fired. `field_correct` is field-type aware (fuzzy for strings,
normalized CO numbers, line items compared by amount).

**`scripts/gen_dataset.py`** renders documents *from* the truth dicts (seeded, `random.seed(7)`), so samples and
ground truth can't drift — except where a variant deliberately lies (08: the document's own math is wrong;
10: handwritten override). Each truth file carries `_variant` and `_expected_issues`, which the eval report surfaces.

## Conventions

- Money is a signed 2-decimal string in JSON (`"-3200.00"`), `Decimal` in code; credits are negative. Dates are ISO.
  `null` means the document does not state the value — never a guess.
- Parsers in `normalize.py` raise `ValueError`; callers turn that into an `Issue` rather than crashing. Extraction
  failures raise `ExtractionError`, which the CLI reports per-file and exits 1 on.
- Document text is fenced in `<document>` tags and declared to be data, not instructions. Keep it that way.
- Tests use `httpx2`'s `MockTransport` against a real `anthropic.Anthropic` client to assert on the request body —
  that is the way to test prompt/request changes.
