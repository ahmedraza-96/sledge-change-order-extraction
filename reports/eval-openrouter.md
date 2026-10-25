# Evaluation report

Model: `nvidia/nemotron-3.5-super-vl:free` · documents: 10 · fields scored: 220

## Headline

| metric | value |
|---|---|
| field accuracy | 97% |
| errors caught by review flags (recall) | 33% |
| flagged fields that were actually wrong (precision) | 29% |
| share of fields sent to review | 3% |
| expected calibration error | 0.059 |

## Calibration (all fields)

| confidence | n | accuracy | mean confidence |
|---|---|---|---|
| 0.0-0.5 | 3 | 67% | 0.41 |
| 0.5-0.7 | 6 | 83% | 0.59 |
| 0.7-0.9 | 60 | 95% | 0.77 |
| 0.9-1.0 | 151 | 99% | 0.99 |

## Per document

| doc | variant | accuracy | doc conf | wrong (unflagged in **bold**) | expected issues caught / missed | calls | tokens | s |
|---|---|---|---|---|---|---|---|---|
| 01_aia_clean | clean AIA G701-style form | 95% | 0.97 | subtotal | — / — | 1 | 6256 | 53.83 |
| 02_contractor_table | contractor letterhead, line-item table, OH&P and tax | 100% | 0.79 | — | — / — | 1 | 10605 | 84.9 |
| 03_credit | credit / deduct change order | 95% | 0.99 | **subtotal** | — / — | 1 | 6339 | 21.02 |
| 04_scanned | scanned image-only PDF (no text layer) | 100% | 0.84 | — | — / — | 1 | 6857 | 34.56 |
| 05_email | email body with superseded quoted estimate | 86% | 0.75 | owner, **contractor_signed**, **owner_signed** | — / — | 1 | 8020 | 102.69 |
| 06_messy | messy plain text with typos and mixed date formats | 100% | 0.99 | — | — / — | 1 | 5510 | 43.45 |
| 07_missing_fields | sparse: missing project no, schedule, contract sums, unsigned; CAD | 95% | 0.78 | **subtotal** | — / — | 1 | 8123 | 37.95 |
| 08_bad_math | document's own arithmetic is wrong (subtotal transposed, revised sum off) | 100% | 0.71 | — | contract_sum_rollforward, line_items_vs_subtotal, subtotal_to_total / — | 2 | 17996 | 83.75 |
| 09_multipage | multi-page, line items continue onto page 2 | 100% | 1.00 | — | — / — | 1 | 10625 | 45.07 |
| 10_handwritten_override | handwritten override of the typed total | 100% | 0.64 | — | contract_sum_rollforward, subtotal_to_total / — | 2 | 19455 | 93.69 |

## Verifier signal (its raw probability; it is blended into the confidence above)

Fields with a verifier probability: 218 (6 wrong). A probability below 0.5 counts as a flag.

| metric | value |
|---|---|
| mean probability, correct fields | 0.98 |
| mean probability, wrong fields | 0.63 |
| errors caught by needs_review (verifier included) | 33% |
| errors caught by the verifier | 17% |
| errors caught by either | 33% |
| correct fields the verifier would add to review | 2 |
| verifier calibration error (ECE) | 0.024 |

| probability | n | accuracy | mean probability |
|---|---|---|---|
| 0.0-0.5 | 5 | 80% | 0.20 |
| 0.5-0.7 | 4 | 50% | 0.56 |
| 0.7-0.9 | 4 | 50% | 0.88 |
| 0.9-1.0 | 205 | 100% | 1.00 |

## Per field

| field | accuracy |
|---|---|
| change_order_number | 100% |
| project_name | 100% |
| project_number | 100% |
| contractor | 100% |
| owner | 90% |
| date_issued | 100% |
| description | 100% |
| reason | 100% |
| subtotal | 70% |
| markup | 100% |
| tax | 100% |
| total_change_amount | 100% |
| currency | 100% |
| original_contract_sum | 100% |
| previous_change_orders_total | 100% |
| revised_contract_sum | 100% |
| schedule_impact_days | 100% |
| revised_completion_date | 100% |
| status | 100% |
| contractor_signed | 90% |
| owner_signed | 90% |
| line_items | 100% |
