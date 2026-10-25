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

**The honest weak spot:** some errors still get a high score, mostly the first three failure modes below.

## Failure modes

Each row says what goes wrong, what catches it, and what can still slip through. The first five were
found while testing; the rest are designed for.

**Found in testing**

| Failure | What catches it | What can still slip through |
|---|---|---|
| **A value worked out, not read.** On a form with no subtotal line, the model adds up the line items and fills in a subtotal | The value must appear in its own quote, or the field is capped at 0.3 | Half of all errors in testing. If the quote happens to contain the same number (the total), it looks backed up; the verifier also says "yes", because the sum is right |
| **The right text in the wrong field.** On the email: the owner taken from an email address, "not signed" on an email with no signature block | Field instructions; the verifier is asked about each field by name | The quote is real, so the evidence check passes |
| **Confident mistakes.** Luna's errors come with real quotes | The math checks, the verifier | Luna's 4 errors on the samples scored 0.92-0.96, among its correct fields. No review line catches them without reviewing about half of all fields |
| **A blank that should have a value.** A phone photo's "+5 days" left empty | The verifier is asked "the document does not state the change in contract time" | Before the verifier, a blank had nothing to check against and scored 0.95 |
| **Verifier false alarms.** The second model says "no" to a correct value | Its answer is only 35% of the score, so a weak "no" lowers a field without sending it to review | In testing, 8 of its 9 "no" answers were wrong. It is a useful second opinion, not a judge |

**Designed for**

| Failure | What catches it | What can still slip through |
|---|---|---|
| Change order amount confused with the new contract sum | Field instructions, the contract-sum math check, the pattern-matcher | Forms that don't label the sums |
| A credit's minus sign lost, e.g. "($3,200)" | Parsed in code, not by the model; a rule checks the sign | A credit described only in words |
| A misread digit | The value must appear in its own quote; the math checks; the re-read | A field no rule touches (e.g. the project number), if the quote has the same misread |
| An invented value for a missing field | "Blank, never guess" instruction; the quote must appear in the document | A believable value with a believable quote on a scan |
| Photos and scans (no text to check quotes against) | The model and the verifier both see the page images; scores are lowered | Photo fields sit near 0.84, so a review line above 0.80 would send almost every photo to a person |
| Handwritten changes and strike-throughs | The model sees the image; the latest edit wins; noted in `notes` | Unclear markups. These should go to a person |
| Old numbers in an email thread | The quoted reply chain is removed | Forwarded threads without quote markers |
| The document's own math is wrong | Checks fail, the re-read confirms, the printed values are kept and flagged | Nothing: flagging it is the right outcome |
| Not a change order at all (an invoice) | Detected; every field is set to 0 and the re-read is skipped | A quote or proposal written like a change order |
| A real change order judged "not a change order" (seen live) | Overruled when page 1 shows a matching change-order number and the math passes | On a scan with no text, the model's verdict stands |
| Several change orders in one file | Counted; the document score is capped at 0.5 | Only the first one is extracted |
| A table continued onto the next page | Continued rows are asked for; line items must add up to the subtotal | Repeated header rows read as items |
| Text in the document trying to instruct the AI | Marked as data; fixed output format; numbers re-checked against the source | Can't be fully ruled out; nothing acts on the model's output automatically |
| A model is down or out of credit | The other model is one click away in the menu; the failed run is saved in History | No automatic switch to the other model yet |
| Scores tuned on few documents | The review line is picked by a test and checked on documents it wasn't picked from | 20 documents is too few; the weights and the line should be refit on real reviewed documents |

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
