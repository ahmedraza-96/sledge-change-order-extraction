# Evaluation report

Model: `gemini-3.6-flash` · documents: 7 · fields scored: 154

## Headline

| metric | value |
|---|---|
| field accuracy | 100% |
| errors caught by review flags (recall) | n/a |
| flagged fields that were actually wrong (precision) | 0% |
| share of fields sent to review | 21% |
| expected calibration error | 0.129 |

## Calibration (all fields)

| confidence | n | accuracy | mean confidence |
|---|---|---|---|
| 0.0-0.5 | 4 | 100% | 0.39 |
| 0.5-0.7 | 28 | 100% | 0.57 |
| 0.7-0.9 | 21 | 100% | 0.77 |
| 0.9-1.0 | 101 | 100% | 1.00 |

## Per document

| doc | variant | accuracy | doc conf | wrong (unflagged in **bold**) | expected issues caught / missed | calls | tokens | s |
|---|---|---|---|---|---|---|---|---|
| 01_aia_clean | clean AIA G701-style form | 100% | 1.00 | — | — / — | 1 | 2770 | 16.03 |
| 02_contractor_table | contractor letterhead, line-item table, OH&P and tax | 100% | 0.69 | — | — / — | 1 | 2794 | 17.97 |
| 03_credit | credit / deduct change order | 100% | 1.00 | — | — / — | 1 | 2505 | 32.83 |
| 04_scanned | scanned image-only PDF (no text layer) | 100% | 0.76 | — | — / — | 1 | 2789 | 66.07 |
| 07_missing_fields | sparse: missing project no, schedule, contract sums, unsigned; CAD | 100% | 0.65 | — | — / — | 1 | 2739 | 14.02 |
| 08_bad_math | document's own arithmetic is wrong (subtotal transposed, revised sum off) | 100% | 0.54 | — | contract_sum_rollforward, line_items_vs_subtotal, subtotal_to_total / — | 2 | 7322 | 77.92 |
| 10_handwritten_override | handwritten override of the typed total | 100% | 0.56 | — | contract_sum_rollforward, subtotal_to_total / — | 2 | 5818 | 40.32 |

## Per field

| field | accuracy |
|---|---|
| change_order_number | 100% |
| project_name | 100% |
| project_number | 100% |
| contractor | 100% |
| owner | 100% |
| date_issued | 100% |
| description | 100% |
| reason | 100% |
| subtotal | 100% |
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
