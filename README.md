# co-extract — change-order extraction pipeline

Turns messy construction change orders (born-digital PDFs, scans, emails, plain text) into
**validated JSON with per-field confidence scores**, and flags the fields a human should check.

See **[WRITEUP.md](WRITEUP.md)** for the approach, how confidence is computed, and failure modes.

## Quickstart

```bash
uv sync
cp .env.example .env            # add OPENROUTER_API_KEY (free), ANTHROPIC_API_KEY, GEMINI_API_KEY and/or OPENAI_API_KEY
uv run python scripts/gen_dataset.py                      # generates data/samples + data/truth (not committed)
uv run co-extract extract data/samples/03_credit.pdf      # JSON to stdout
uv run co-extract extract data/samples -o out/            # a whole folder
uv run co-extract eval                                    # accuracy + calibration -> reports/eval-claude.md
uv run co-extract serve                                   # web UI on http://127.0.0.1:8000
uv run pytest                                             # offline tests (no API key needed)
uv run pytest -m live                                     # real API calls, per configured key
```

### Providers

Extraction runs against Claude (default), Gemini, OpenAI or OpenRouter's free models; everything after the model call —
normalization, validation, the repair pass, confidence scoring — is shared, so the two are directly
comparable on the same eval. OpenAI defaults to `gpt-5-mini` (`--provider openai`).

**OpenRouter (free).** `--provider openrouter` uses free models only, chosen from the free models
that accept images *and* enforce a JSON schema: `nvidia/nemotron-3.5-super-vl:free` (the only one
with published document/OCR results), falling back to `dots-studio/dots-3-note-preview:free` and
then the text-only `nvidia/nemotron-3-super-120b-a12b:free` when a model is rate-limited or down.
`meta.model` records which one answered. PDF pages are sent as rendered images, so no paid
PDF-parsing engine is involved. Free limits: 20 requests/min and 50/day (1000/day once $10 of
credit has been bought). If every request fails with a 404, allow free-model providers in
https://openrouter.ai/settings/privacy. Free endpoints may log prompts, so don't send confidential
documents through them.

```bash
uv run co-extract extract data/samples/03_credit.pdf --provider gemini
uv run co-extract eval --provider gemini                  # -> reports/eval-gemini.md
uv run co-extract eval --provider claude                  # -> reports/eval-claude.md
uv run co-extract extract <file> --provider gemini --model gemini-3-flash-preview
```

Cached runs and reports are scoped per provider (`reports/runs/<provider>/`), so the two never
overwrite each other and `--cached` always re-scores one provider's results. `eval` warns and exits
non-zero if it scored fewer documents than it has ground truth for — a partial run's headline
numbers are not a usable baseline.

Gemini defaults to `gemini-3.5-flash`. Pro models need billing enabled on the Google Cloud project
behind the key — on the free tier they return a quota error with `limit: 0`.

The free Gemini tier also caps **flash** usage at roughly 5 requests/minute and 20 per day, which is
less than one full eval (10 documents, more when the repair pass fires). Spread it across quota
windows with:

```bash
uv run co-extract eval --provider gemini --resume   # keeps documents already cached
```

Without an API key you can still run the full pipeline on recorded model outputs:

```bash
uv run co-extract extract tests/fixtures/docs/misread_co.txt --fixtures tests/fixtures/llm
```

That fixture has a misread total (`3580.00` vs printed `$3,850.00`) and an unparseable date on the
first pass. Validation catches both and the repair pass fixes them (`meta.repair` shows it). With
`--no-repair` the total gets confidence 0.3 and is listed in `needs_review`.

### Web UI

```bash
uv run co-extract serve          # then open http://127.0.0.1:8000
```

Upload a PDF, an image (PNG/JPG/TIFF/WebP — e.g. a phone photo), .txt or .eml, or paste the text, pick a provider and click **Extract**. The page shows
document confidence, the fields that need review, and a table of every field with its value,
confidence (red below 0.70), evidence quote and the reasons behind the score, plus line items,
validation issues and a JSON download. Providers without a key in `.env` are greyed out; **Demo**
replays the recorded fixtures for the bundled samples and needs no key.

## Output (abridged; numbers illustrative)

```jsonc
{
  "source": "data/samples/03_credit.pdf",
  "change_order": { "change_order_number": "CO-005", "total_change_amount": "-3200.00", ... , "line_items": [...] },
  "fields": {
    "total_change_amount": {
      "value": "-3200.00", "confidence": 0.98,
      "evidence": "TOTAL CREDIT THIS CHANGE ORDER ($3,200.00)", "page": 1,
      "model_confidence": 0.95, "reasons": []
    },
    ...
  },
  "document_confidence": 0.95,
  "needs_review": [],                      // fields with confidence < 0.7
  "issues": [],                            // failed validation rules: {code, severity, message, fields}
  "notes": [],                             // ambiguities the model noticed (handwritten edits, etc.)
  "meta": { "model": "claude-opus-5-5", "llm_calls": 1, "repair": null, "input_tokens": ..., "output_tokens": ..., "seconds": ... }
}
```

Money is a signed 2-decimal string (credits are negative), dates are ISO, and `null` means the document does not state the value.

## Layout

```
src/co_extract/
  ingest.py      PDF (pdfplumber) / .eml / .txt -> page-tagged text; detects scanned pages
  schema.py      LLM output schema (value + evidence + page + confidence per field), field types
  extract.py     Extractor protocol, the shared prompt, Claude call; FixtureExtractor for offline replay
  gemini.py      Gemini call against the same prompt and schema
  openai_extractor.py  OpenAI (Responses API) call against the same prompt and schema
  providers.py   provider enum + extractor factory shared by the CLI and web app
  normalize.py   deterministic parsing of money / dates / ints / enums
  validate.py    type coercion issues + arithmetic & consistency rules
  heuristics.py  independent regex extractor used as a cross-check
  confidence.py  per-field and document confidence
  pipeline.py    ingest -> extract -> validate -> repair once -> score
  evaluate.py    accuracy, calibration (ECE), review precision / recall
  cli.py         `co-extract extract | eval | serve`, `--provider claude|gemini|openai`
  web.py, static/index.html  FastAPI web UI over pipeline.run
scripts/gen_dataset.py   synthetic dataset generator (seeded)
data/samples, data/truth  10 change orders + ground truth
```
