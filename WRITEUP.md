# Change-order extraction: approach and failure modes

## Goal

A change order is the contract amendment behind a payment. The fields that matter most are the CO
amount, the contract-sum roll-forward and the schedule impact, and a wrong number there costs money.
So the design goal is not just high accuracy. It is **knowing which fields are wrong**: auto-accept
what we can verify and route the rest to a person.

## Pipeline

```
ingest ──► extract (LLM)    ──► normalize + validate ──► repair once? ──► confidence ──► JSON
            │                        │                                       ▲
            └ PDF pages + tagged text└ arithmetic / consistency rules        └ grounding, cross-check, rules
```

1. **Ingest.** pdfplumber pulls per-page text, tagged with `[[page N]]`. A page with almost no text
   layer is treated as scanned. Images (PNG/JPG/TIFF/WebP — phone photos, fax scans) are converted to
   PDF pages first, upright per their EXIF rotation and capped at 2400 px, so they take exactly the
   scanned-PDF path from here on: the page image is sent to the model, which reads it directly.
   Docling for IBM watsonx OCR is available as an opt-in (`DOCLING_OCR_ENABLED`) but off by default; see
   *OCR or vision?* below for the benchmark that decided it. `.eml` files drop the quoted reply chain, because earlier messages often hold
   superseded numbers.
2. **Extract.** One LLM call with structured outputs against a Pydantic schema, so the response
   always matches the schema. The default is `claude-opus-5-5`; Gemini and OpenAI sit behind the same
   interface (see *Swapping the model*). The PDF itself is attached, so the
   model sees the layout, tables, strike-throughs and handwriting, alongside the text layer we ground
   against. For each field the model returns:
   - the **value**,
   - a **verbatim evidence quote**,
   - the **page**,
   - a **self-rated confidence**.

   The prompt is short and aimed at the known traps:
   - the CO amount vs. the revised contract sum,
   - credits written in parentheses,
   - handwritten overrides,
   - superseded numbers in email threads,
   - returning `null` instead of guessing.

   Document text is wrapped in tags and declared to be data, not instructions.
3. **Normalize.** The LLM schema is deliberately all strings. Turning `"($3,200.00)"`, `"CR"` or
   `"3/4/26"` into typed values is done in code, deterministically, so it can be tested.
4. **Validate.** Rules an accountant would check by hand:
   - qty × unit price = amount,
   - the line items sum to the subtotal,
   - subtotal + markup + tax = total,
   - original + previous COs + this CO = revised contract sum,
   - the dates are plausible and in order,
   - a positive total is not described as a credit.

   *Hard* issues (unparseable value, no CO number) and arithmetic failures trigger **one repair
   pass**: the model re-reads the document with the failed checks in front of it. If the mismatch
   survives the re-read, the most likely explanation is that the source document is wrong. We then
   keep the printed values and say so, because silently "fixing" a contract document is worse than
   flagging it.
5. **Confidence.** Self-reported LLM confidence is poorly calibrated, so it carries only 20% of the
   weight. The rest comes from signals we can check:
   - **Grounding (50%):** a fuzzy match of the evidence quote against the page text. A hallucinated
     quote doesn't match.
   - **Value-in-evidence cap:** if the quote doesn't contain the extracted number or date, the
     field is capped at 0.3. This catches a misread digit even when the quote is real.
   - **Rules:** each failed rule involving the field multiplies its score by 0.6.
   - **Independent cross-check:** a labelled-value regex extractor. Agreement adds 0.1;
     disagreement halves the score.
   - **OCR'd pages** (only with Docling switched on): grounded against the OCR text with a milder
     penalty, because OCR can misread a digit too. Without it, or if OCR fails, there is no
     text to check a quote against: they get a neutral score and a small penalty, which
     deliberately pushes them toward review.

   Fields below 0.7 go to `needs_review`. Document confidence is driven by the weakest *critical*
   field (CO number, amount, contract sums).

## Evaluation

The repo includes a seeded generator for 10 synthetic COs with ground truth. They cover:
- a clean AIA-style form,
- a contractor table layout,
- a credit,
- an image-only scan,
- an email with a superseded quoted number,
- messy text with typos,
- missing fields,
- a document whose own arithmetic is wrong,
- a multi-page continued table,
- a handwritten override.

`co-extract eval` reports:
- per-field accuracy,
- calibration (accuracy by confidence bucket, and ECE),
- **error recall**: the share of wrong fields that were flagged for review,
- **review precision**,
- whether the expected validation issues fired.

The key question is error recall: an error that ships unflagged is the expensive failure.

> **Results:** see `reports/eval-<provider>.md` (generated by `uv run co-extract eval
> [--provider gemini]`; requires an API key). The Gemini run so far covers only 1 of the 10
> documents (the clean AIA form, 22/22 fields correct) because of the free-tier daily quota. That
> is not a usable baseline. The harness says so and exits non-zero, and `--resume` completes it.

Unit tests cover the parsers, every rule, grounding, the repair path and a mocked API round-trip.
They run offline.

## Failure modes and mitigations

| Failure | Mitigation | Residual risk |
|---|---|---|
| CO amount confused with revised contract sum | Explicit field guide; roll-forward rule; regex cross-check on labelled sums | Forms that don't label the sums |
| Credit sign lost (`($3,200)` → `3200`) | Deterministic parser handles parens/CR; `credit_sign` rule | Credit implied only by prose |
| Misread digit / transposition | Value-in-evidence cap; arithmetic rules; repair pass | A misread in a field no rule touches (e.g. project number) is caught only if the quote is exact |
| Hallucinated value for a missing field | "null over guess" prompt; evidence must ground; null-vs-regex check | Plausible value with a plausible quote on a scan |
| Scans / phone photos | Photos are EXIF-rotated and size-capped, then a vision model reads the page image; confidence is discounted because there is no text layer to check quotes against | Scans lean on review. Hosted OCR (Docling) was tried and benchmarked; it made errors look verified (see *OCR or vision?*), so it is opt-in only |
| Handwritten overrides / strike-throughs | Model sees the page image; prompt says latest edit wins; surfaced in `notes` | Ambiguous markups. These *should* go to a human |
| Superseded numbers in email threads | Quoted chain stripped at ingest; prompt warning | Forwarded threads without `>` quoting |
| Document's own math is wrong | Rules fire, re-read confirms, values kept as printed and flagged | — (correct behaviour is to flag) |
| Not a change order at all (an invoice, a resume) | Model reports `is_change_order`; a `not_a_change_order` issue fires, document confidence is forced to 0 and **every field is capped at 0** too, so no value looks trustworthy on its own (an invoice's supplier otherwise came back as `contractor` at 0.98). The repair pass is skipped: re-reading an invoice can't produce a CO number, it only spends a second call | Relies on the model's own judgement of the document type; a near-miss (a quote or a proposal phrased like a CO) may pass as a change order |
| Several COs in one PDF | Model reports `change_order_count`; doc confidence capped | Only the first is extracted. Next step: split, then extract each |
| Tables split across pages | Prompt asks for continued rows; line items must sum to the subtotal | Repeated header rows parsed as items |
| Prompt injection inside the document | Document fenced and declared as data; output is schema-constrained; numbers are re-verified against the source | Cannot fully rule out. Nothing downstream executes model output |
| Confidence miscalibrated on new layouts | Mostly verification signals, not self-report; eval measures ECE | Weights were hand-set on 10 synthetic docs. They should be fit on real labelled data |

## Cost and latency

There is one model call per document, and a second only when validation fails. Claude Opus 5.5 is
$4 / $20 per million input / output tokens. A 1–2 page CO uses a few thousand input tokens (page
images plus text) and about 2–4k output tokens, because the evidence quotes make the output verbose.
That puts it at **roughly $0.05–0.15 per document** (an estimate; `reports/eval-claude.md` has the
measured token counts). For backlog processing the Batch API halves this.

## Web UI

`uv run co-extract serve` starts a small FastAPI app at http://127.0.0.1:8000. Upload a document
or paste text and click Extract; the model is picked automatically. Progress is streamed from the
server as the pipeline actually advances (read, extract, validate, repair, score), and with OpenRouter
the model's own output is streamed too, so the page counts fields as they are written instead of
showing a frozen spinner. The result is the original document beside a simplified JSON: value and
confidence per field, the document confidence, and a warning when the document is not a change order.

It calls the same `pipeline.run` as the CLI. It is a thin view, not a second code path. A **Demo**
provider replays the recorded fixtures so it works without any API key.

## OCR or vision?

Scans and photos have no text layer, so two designs were compared on the same 10 ground-truth samples,
rendered as scanned-looking images (`scripts/benchmark_ocr.py`, one model call per document, no repair):
the free vision model reading the image, and Docling for IBM watsonx OCR followed by a free text model.

| | Vision model reads the image | Docling OCR, then a text model |
|---|---|---|
| Field accuracy | **95.5%** (210/220) | 91.9% (182/198) |
| Wrong but confident (at least 0.70) | **5** | 11 |
| Documents completed | **10/10** | 9/10 |
| Mean time per document | **38 s** | 64 s |

The surprise was the confidence column. OCR misreads (a `$17,340` read as `$17,040`) were copied into
the answer and then checked against the same misread text, so they looked *grounded*: OCR turns a
reading error into a verified one. The vision model has no intermediate text to agree with, so its
mistakes are less often confident. The text model was also unreliable on the free tier: its primary
never answered, and the fallback ran out of output tokens on the multi-page document. Docling stays
in the code as an opt-in.

## Swapping the model

Extraction sits behind an `Extractor` protocol, and everything after it — normalization, the
validation rules, the repair pass, confidence scoring — is provider-independent, because it all
operates on the `RawExtraction` schema rather than on a provider's response. So testing a cheaper
model against this same eval is a matter of adding one module. There are four today: Claude
(`extract.py`), Gemini (`gemini.py`), OpenAI (`openai_extractor.py`, Responses API in strict
structured-output mode) and OpenRouter (`openrouter.py`). The OpenRouter backend uses free models
only: `nvidia/nemotron-3.5-super-vl:free`, picked because it is the free model that both reads images
and enforces a JSON schema *and* has published document-reading results, with two free fallbacks
that OpenRouter switches to when it is rate-limited. PDF pages go to it as rendered images rather
than as a PDF, because OpenRouter's PDF handling can route through a paid OCR engine.

```bash
uv run co-extract eval --provider gemini     # -> reports/eval-gemini.md
uv run co-extract eval --provider openai     # -> reports/eval-openai.md
uv run co-extract eval --provider openrouter # -> reports/eval-openrouter.md
```

All providers send byte-identical prompt text (the shared `document_prompt` / `repair_prompt`
helpers, pinned by a test), which is what makes the two reports comparable. Gemini 3 Flash measures
at roughly $0.30 / $2.50 per million tokens, an order of magnitude below Opus.

Two things surfaced immediately that are worth recording:

- **Self-reported confidence is near-useless on Flash.** It returns `confidence` at or just below
  1.0 on almost every field, including fields it got wrong. The design already discounts self-report
  to 20% of the score, which is why this degrades calibration rather than accuracy — but it means on
  Flash essentially all of the discriminative signal comes from grounding and the rules. This is the
  clearest vindication of weighting verification over self-report.
- **Free-tier quota, not cost, is the binding constraint on evaluating it.** There are two separate
  ceilings — roughly 5 requests/minute and roughly 20 per *day* — plus 503 load-shedding under
  contention. A 10-document eval needs 10–20 calls (the repair pass doubles some), so it does not
  fit in a day's free quota. `GeminiExtractor` budgets its backoff for the per-minute window and
  fails fast past it, because the daily 429 reports a reset *15 hours* out and retrying it is dead
  wall-clock. `co-extract eval --resume` reuses already-cached documents so the run can be spread
  across quota windows.

  This also exposed a sharp edge in the harness that was worth fixing on its own account: a run
  that loses documents still printed a headline accuracy, and a partial sample reads far *better*
  than a full one, because the documents that fail are disproportionately the awkward ones. A first
  Gemini run scored 2 of 10 documents and reported "100% field accuracy". `eval` now warns and exits
  non-zero whenever it scores fewer documents than it has ground truth for.

## What I'd do next

- **Turn the UI into a review tool:** today it shows each field with its evidence and confidence.
  Next is highlighting the evidence on the rendered page and letting a reviewer correct a value.
  Those corrections become labelled data.
- **Fit the confidence weights** (logistic regression on the signals) on real reviewed documents,
  instead of hand-setting them. The weights are currently shared across providers, which the Gemini
  numbers show is wrong: a model whose self-report is flat needs `W_MODEL` near zero and the freed
  weight moved to grounding. Fitting per provider falls out of doing this properly.
- **OCR as a second opinion, not as the input.** The benchmark showed OCR text makes a bad *input*
  (its misreads look verified). It could still be a useful independent check: let the vision model
  read the image, then compare each value with Docling's text and lower confidence where they
  disagree, the same way the regex cross-check works today.
- **Split multi-CO packets** and run extraction on each.
- **Per-customer layout hints** (few-shot from their past COs), with prompt caching for the static
  prefix.
