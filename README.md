# co-extract — change-order extraction pipeline

Turns messy construction change orders (born-digital PDFs, scans, emails, plain text) into
**validated JSON with per-field confidence scores**, and flags the fields a human should check.

See **[WRITEUP.md](WRITEUP.md)** for the approach, how confidence is computed, and failure modes.

## Quickstart

```bash
uv sync
cp .env.example .env            # add OPENROUTER_API_KEY (free); other providers and Docling are optional
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

**OpenRouter (free).** `--provider openrouter` uses free models only. A vision model reads every
document, `nvidia/nemotron-3.5-super-vl:free` with `dots-studio/dots-3-note-preview:free` as the
fallback: PDFs and photos go as page images alongside their text, and text and email as text.
`meta.model` records which model answered. Free limits: 20 requests/min and 50/day (1000/day once
$10 of credit has been bought). If every request fails with a 404, allow free-model providers in
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

Upload a PDF, an image (PNG/JPG/TIFF/WebP, e.g. a phone photo), .txt or .eml, or paste the text, and
click **Extract**. The model is chosen automatically: OpenRouter when `OPENROUTER_API_KEY` is set,
otherwise the next provider with a key. While it runs, a timeline shows each real step of the pipeline,
with a live count of fields as the model writes them (OpenRouter streams). The result shows the original
document on the left and, on the right, the extracted JSON: each field's value and confidence
(colour-coded, red below 0.70), the document confidence, and a warning if the document isn't a change
order. Copy and Download give exactly that JSON; the CLI gives the full output with evidence, reasons
and validation issues.

**History.** Every extraction, including failed ones, is saved to SQLite (`co_extract/history.py`): the
document exactly as uploaded or pasted, the full result, the model, the time taken and any error. The
**History** button lists the runs newest first; opening one shows the document beside its JSON, exactly
as it was first shown. The database is `data/history.db` (`HISTORY_DB` changes the path; `HISTORY_DB=off`
turns recording off). Files over 20 MB keep their result but not the file. The history is visible to
anyone who can open the app and has no delete button, so don't upload confidential documents to a public
deployment. To clear it, delete the database file on the server.

**Optional: Docling for IBM watsonx OCR (off by default).** With `DOCLING_OCR_ENABLED=true` plus
`DOCLING_SERVICE_URL` and `DOCLING_SERVICE_API_KEY`, every document is rendered as page images and OCR'd
by that hosted service, and the OCR text is what the model reads. It is off by default because the
vision model reading the image directly did better in a benchmark on 10 scanned-looking samples
(`uv run python scripts/benchmark_ocr.py`, report in `reports/benchmark-ocr.md`):

| | Vision model reads the image | Docling OCR, then a text model |
|---|---|---|
| Field accuracy | **95.5%** (210/220) | 91.9% (182/198) |
| Wrong but confident (at least 0.70) | **5** | 11 |
| Documents completed | **10/10** | 9/10 |
| Mean time per document | **38 s** | 64 s |

OCR misreads were copied into the answer and then "verified" against the same misread text, which is
why Docling had twice as many confidently-wrong fields.

## How confidence is calculated

Every score is computed by our code (`confidence.py`), not taken from the model. The model's own
self-rating is just one input, worth 20%, because it is poorly calibrated: some models report about
1.0 on nearly everything.

**Each field (0 to 1):**

```
score = 0.5 × grounding + 0.2 × model_confidence + 0.3
```

- **grounding**: how closely the model's evidence quote matches the document text, a fuzzy match
  from 0 to 1. A quote the model made up doesn't match.
- **model_confidence**: the model's self-rating for that field.
- **0.3**: the share for passing the checks. It is then reduced by the adjustments below.

Then, in order:

| Signal | Effect |
|---|---|
| Independent regex extractor finds the **same** labelled value | +0.1 (max 1.0) |
| Regex extractor finds a **different** value | × 0.5 |
| Each failed validation rule that involves the field (e.g. subtotal + markup + tax ≠ total) | × 0.6 |
| Page text came from OCR (only when Docling is switched on) | × 0.95 |
| Page has no text at all (unread scan), so grounding is a neutral 0.7 | × 0.9 |
| The value doesn't appear inside its own quote (e.g. a misread digit) | capped at 0.3 |
| The value couldn't be parsed (not a valid amount or date) | 0.1 |
| `null` (document doesn't state it) | 0.55 + 0.4 × model_confidence, or 0.3 if the regex *did* find a labelled value |

**Line items** share one score. Each row is grounded the same way, and a row whose amount isn't in its
own quote is capped at 0.3. The score is 0.7 × the weakest row + 0.3 × the average, × 0.6 per failed
rule involving the line items.

**Document confidence** = 0.6 × the weakest *critical* field (CO number, CO amount, the three contract
sums) + 0.4 × the average of all fields. It is capped at 0.5 if there is a hard error (e.g. no CO number)
or several change orders in one file. It is 0 if the document isn't a change order at all, and then
every field is capped at 0 too.

**Review:** any field below **0.70** is listed in `needs_review`.

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
  history.py     SQLite record of every web run (document + result), behind the History view
  evaluate.py    accuracy, calibration (ECE), review precision / recall
  cli.py         `co-extract extract | eval | serve`, `--provider claude|gemini|openai`
  web.py, static/index.html  FastAPI web UI over pipeline.run
scripts/gen_dataset.py   synthetic dataset generator (seeded)
data/samples, data/truth  10 change orders + ground truth
```
