# Change-order extraction: how it works and how well

## The goal

A change order changes a construction contract, and its numbers turn into payments. A wrong amount
costs money. So the aim is not only to read the fields correctly but to **know which fields might be
wrong**, so they can go to a person instead of straight into the books.

Every field in the output has a value and a **confidence** from 0 to 1. Anything below **0.65** goes to
human review.

## How it works

```
read the file -> AI model extracts fields -> check the math -> re-read once if it fails -> score each field -> JSON
```

1. **Read.** PDFs, photos, scans, faxes, emails and plain text. Photos are turned upright and sent to
   the model as images. Emails drop the quoted reply chain, which often holds old numbers.
2. **Extract.** One call to the AI model. For every field it returns the value, a word-for-word quote
   from the document as evidence, and the page. The default model is **GPT-6 Luna** (OpenAI); the free
   **NVIDIA Nemotron** model on OpenRouter is the other choice in the menu. The rest of the pipeline is
   the same for both.
3. **Check.** The rules an accountant would apply: line items add up to the subtotal, subtotal + markup
   + tax = total, original contract + earlier changes + this change = new contract sum, dates in order.
4. **Re-read once.** If a check fails, the model reads the document again with the failed check in front
   of it. If it still fails, the document itself is probably wrong, so the printed values are kept and
   flagged. The system never "fixes" a contract document.
5. **Score.** Each field gets a confidence (below).

## How confidence is calculated

Models are bad at rating themselves: Luna rates 97% of its fields 0.95 or higher and Nemotron 80%,
right or wrong. So most of the score comes from things the system can check:

| Part | Share | What it checks |
|---|---|---|
| Second-model check | 35% | A separate AI (OpenAI's Decisions API) is asked a yes/no question per field, e.g. "The total of this change order is $3,850.00", and returns how likely that is true |
| Evidence check | 32.5% | Does the model's quote really appear in the document? An invented quote doesn't |
| Rules | 19.5% | A starting score, cut when the field is part of a failed math check |
| Model's own rating | 13% | Kept small, because it is rarely informative |

Two hard limits sit on top. If the quote doesn't contain the value (say the quote says $11,400 but the
value is $17,340), the field is capped at 0.3. And a simple pattern-matcher reads labelled values on its
own; if it disagrees with the model, the score is halved.

**The 0.65 review line was picked by a test, not by hand** (`scripts/find_threshold.py`). It tries every
line from 0.50 to 1.00 on 20 labelled documents and picks the one that best balances errors caught
against fields sent to review, counting a missed error twice as bad as an extra review. It is then
checked on documents it wasn't picked from.

## Results

Tested on 10 sample change orders with known answers (clean form, table, credit, scan, email, messy
text, missing fields, wrong math, multi-page, handwritten change), plus 10 harder ones: phone photos,
a screenshot, a fax, a multi-page scan and 3 documents the system had never seen.

| | GPT-6 Luna (default) | NVIDIA Nemotron (free) |
|---|---|---|
| Fields read correctly, 10 samples | **98.2%** (4 wrong of 220) | 97.3% (6 wrong of 220) |
| Fields read correctly, 10 photos and new documents | 97.7% | not measured |
| Time per document | **~16 s** | ~60 s |
| Cost per document | ~$0.002 | free (50 a day) |
| Errors sent to review, 10 samples | 0 of 4 | **2 of 6** |
| Errors sent to review, photos and new documents | 3 of 5 | not measured |

The billing numbers (change order amount, contract sums) were right on every sample for both models.

**The honest weak spot:** some errors still get a high score. Luna's mistakes come with real quotes, so
every check passes. Two kinds cause most of them:

- **A value worked out, not read.** On forms with no subtotal line, the model adds up the line items
  itself. The number is arithmetically right, but the document never states it. This is half of all
  the errors.
- **The right text in the wrong field**, mostly on emails: the owner taken from an email address, or
  "not signed" on an email that has no signature block.

## What else can go wrong, and what handles it

| Problem | What handles it |
|---|---|
| Change order amount confused with the new contract sum | Clear field instructions, the contract-sum math check, the pattern-matcher |
| A credit's minus sign lost, e.g. "($3,200)" | Parsed in code, not by the model; a rule checks the sign |
| A misread digit | The value must appear in its own quote; the math checks; the re-read |
| Photos and scans (no text to check quotes against) | The verifier can see the page images; scores are lowered so these lean toward review |
| The document's own math is wrong | Checks fail, the re-read confirms, the printed values are kept and flagged |
| Not a change order at all (an invoice) | Detected, and every field is set to 0 so nothing looks trustworthy |
| Text in the document trying to give the AI instructions | The document is marked as data, the output format is fixed, and numbers are re-checked against the source |

## Decisions worth noting

- **A vision model, not OCR, for scans.** Hosted OCR (IBM Docling) was tested: 91.9% accuracy against
  95.5% for the model reading the image. Worse, an OCR misread was then "confirmed" by checking it
  against the same misread text. OCR is in the code but off by default.
- **The model is swappable.** Luna, Nemotron, Claude, Gemini and GPT-5 mini all sit behind one interface
  and get exactly the same prompt, so the test results compare like with like.
- **The test refuses partial runs.** A run that loses documents reads better than it should (the lost
  ones are the hard ones), so the evaluation fails loudly instead of reporting a headline number.

## What I'd do next

1. **Ask the verifier whether the document actually states each value**, not just whether the value is
   right. That targets the "worked out, not read" errors, half of all mistakes.
2. **Turn the page into a review tool:** highlight the evidence on the document and let the reviewer
   correct a value. Each correction becomes labelled data.
3. **Fit the score weights and the review line on real reviewed documents.** They are set on 20 test
   documents today, which is too few to be sure.
