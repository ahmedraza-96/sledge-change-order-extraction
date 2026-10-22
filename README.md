# co-extract — change-order extraction pipeline

Turns messy construction change orders (born-digital PDFs, scans, emails, plain text) into
**validated JSON with per-field confidence scores**, and flags the fields a human should check.

See **[WRITEUP.md](WRITEUP.md)** for the approach, how confidence is computed, and failure modes.

## Quickstart

```bash
uv sync
cp .env.example .env            # add ANTHROPIC_API_KEY
uv run python scripts/gen_dataset.py                      # generates data/samples + data/truth (not committed)
uv run co-extract extract data/samples/03_credit.pdf      # JSON to stdout
uv run co-extract extract data/samples -o out/            # a whole folder
uv run co-extract eval                                    # accuracy + calibration -> reports/eval.md
uv run pytest                                             # offline tests (no API key needed)
uv run pytest -m live                                     # 2 real API calls
```

Without an API key you can still run the full pipeline on recorded model outputs:

```bash
uv run co-extract extract tests/fixtures/docs/misread_co.txt --fixtures tests/fixtures/llm
```

That fixture has a misread total (`3580.00` vs printed `$3,850.00`) and an unparseable date on the
first pass. Validation catches both and the repair pass fixes them (`meta.repair` shows it). With
`--no-repair` the total gets confidence 0.3 and is listed in `needs_review`.

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
  extract.py     Claude call via structured outputs; FixtureExtractor for offline replay
  normalize.py   deterministic parsing of money / dates / ints / enums
  validate.py    type coercion issues + arithmetic & consistency rules
  heuristics.py  independent regex extractor used as a cross-check
  confidence.py  per-field and document confidence
  pipeline.py    ingest -> extract -> validate -> repair once -> score
  evaluate.py    accuracy, calibration (ECE), review precision / recall
  cli.py         `co-extract extract | eval`
scripts/gen_dataset.py   synthetic dataset generator (seeded)
data/samples, data/truth  10 change orders + ground truth
```
