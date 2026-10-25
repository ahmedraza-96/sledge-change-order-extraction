# Evaluation report

Model: `gpt-6-luna` · documents: 10 · fields scored: 220

## Headline

| metric | value |
|---|---|
| field accuracy | 98% |
| errors caught by review flags (recall) | 0% |
| flagged fields that were actually wrong (precision) | 0% |
| share of fields sent to review | 2% |
| expected calibration error | 0.059 |

## Calibration (all fields)

| confidence | n | accuracy | mean confidence |
|---|---|---|---|
| 0.0-0.5 | 1 | 100% | 0.47 |
| 0.5-0.7 | 5 | 100% | 0.60 |
| 0.7-0.9 | 39 | 100% | 0.80 |
| 0.9-1.0 | 175 | 98% | 0.99 |

## Per document

| doc | variant | accuracy | doc conf | wrong (unflagged in **bold**) | expected issues caught / missed | calls | tokens | s |
|---|---|---|---|---|---|---|---|---|
| 01_aia_clean | clean AIA G701-style form | 95% | 0.99 | **subtotal** | — / — | 1 | 5902 | 16.36 |
| 02_contractor_table | contractor letterhead, line-item table, OH&P and tax | 100% | 0.79 | — | — / — | 1 | 5753 | 11.79 |
| 03_credit | credit / deduct change order | 95% | 1.00 | **subtotal** | — / — | 1 | 5564 | 10.76 |
| 04_scanned | scanned image-only PDF (no text layer) | 100% | 0.84 | — | — / — | 1 | 5609 | 19.78 |
| 05_email | email body with superseded quoted estimate | 91% | 0.96 | **owner**, **date_issued** | — / — | 1 | 2802 | 9.98 |
| 06_messy | messy plain text with typos and mixed date formats | 100% | 1.00 | — | — / — | 1 | 3163 | 11.36 |
| 07_missing_fields | sparse: missing project no, schedule, contract sums, unsigned; CAD | 100% | 0.97 | — | — / — | 1 | 5418 | 11.25 |
| 08_bad_math | document's own arithmetic is wrong (subtotal transposed, revised sum off) | 100% | 0.71 | — | contract_sum_rollforward, line_items_vs_subtotal, subtotal_to_total / — | 2 | 12796 | 26.39 |
| 09_multipage | multi-page, line items continue onto page 2 | 100% | 1.00 | — | — / — | 1 | 9174 | 16.5 |
| 10_handwritten_override | handwritten override of the typed total | 100% | 0.65 | — | contract_sum_rollforward, subtotal_to_total / — | 2 | 12062 | 22.39 |

## Verifier signal (its raw probability; it is blended into the confidence above)

Fields with a verifier probability: 218 (4 wrong). A probability below 0.5 counts as a flag.

| metric | value |
|---|---|
| mean probability, correct fields | 0.98 |
| mean probability, wrong fields | 0.87 |
| errors caught by needs_review (verifier included) | 0% |
| errors caught by the verifier | 0% |
| errors caught by either | 0% |
| correct fields the verifier would add to review | 3 |
| verifier calibration error (ECE) | 0.033 |

| probability | n | accuracy | mean probability |
|---|---|---|---|
| 0.0-0.5 | 4 | 100% | 0.24 |
| 0.5-0.7 | 1 | 100% | 0.68 |
| 0.7-0.9 | 6 | 33% | 0.87 |
| 0.9-1.0 | 207 | 100% | 1.00 |

## Per field

| field | accuracy |
|---|---|
| change_order_number | 100% |
| project_name | 100% |
| project_number | 100% |
| contractor | 100% |
| owner | 90% |
| date_issued | 90% |
| description | 100% |
| reason | 100% |
| subtotal | 80% |
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
| contractor_signed | 100% |
| owner_signed | 100% |
| line_items | 100% |
