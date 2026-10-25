# Evaluation report

Model: `gpt-6-luna` · documents: 10 · fields scored: 220

## Headline

| metric | value |
|---|---|
| field accuracy | 98% |
| errors caught by review flags (recall) | 60% |
| flagged fields that were actually wrong (precision) | 15% |
| share of fields sent to review | 9% |
| expected calibration error | 0.124 |

## Calibration (all fields)

| confidence | n | accuracy | mean confidence |
|---|---|---|---|
| 0.0-0.5 | 4 | 75% | 0.40 |
| 0.5-0.7 | 16 | 88% | 0.59 |
| 0.7-0.9 | 136 | 99% | 0.84 |
| 0.9-1.0 | 64 | 100% | 0.99 |

## Per document

| doc | variant | accuracy | doc conf | wrong (unflagged in **bold**) | expected issues caught / missed | calls | tokens | s |
|---|---|---|---|---|---|---|---|---|
| 01_aia_clean_phone_photo_sideways | photo/scan of 01_aia_clean | 95% | 0.83 | subtotal | — / — | 1 | 5613 | 20.22 |
| 02_contractor_table_screenshot | photo/scan of 02_contractor_table | 100% | 0.84 | — | — / — | 1 | 3974 | 15.04 |
| 03_credit_phone_photo | photo/scan of 03_credit | 95% | 0.84 | **subtotal** | — / — | 1 | 5353 | 14.74 |
| 07_missing_fields_lowcontrast | photo/scan of 07_missing_fields | 100% | 0.86 | — | — / — | 1 | 3872 | 13.53 |
| 08_bad_math_phone_photo | photo/scan of 08_bad_math | 100% | 0.61 | — | contract_sum_rollforward, line_items_vs_subtotal, subtotal_to_total / — | 2 | 12085 | 28.75 |
| 09_multipage_scan | photo/scan of 09_multipage | 100% | 0.84 | — | — / — | 1 | 6831 | 21.67 |
| 10_handwritten_override_fax | photo/scan of 10_handwritten_override | 86% | 0.53 | **project_number**, previous_change_orders_total, revised_contract_sum | contract_sum_rollforward, subtotal_to_total / — | 2 | 8654 | 29.79 |
| A_new_deduct_change_order | new: deduct change order (PDF) | 100% | 1.00 | — | — / — | 1 | 5605 | 11.02 |
| A_new_deduct_change_order_photo | new: deduct change order (phone photo) | 100% | 0.85 | — | — / — | 1 | 5215 | 12.84 |
| C_new_text_change_order | new: plain-text change order | 100% | 1.00 | — | — / — | 1 | 3188 | 11.83 |

## Verifier signal (its raw probability; it is blended into the confidence above)

Fields with a verifier probability: 219 (5 wrong). A probability below 0.5 counts as a flag.

| metric | value |
|---|---|
| mean probability, correct fields | 0.96 |
| mean probability, wrong fields | 0.84 |
| errors caught by needs_review (verifier included) | 60% |
| errors caught by the verifier | 20% |
| errors caught by either | 60% |
| correct fields the verifier would add to review | 0 |
| verifier calibration error (ECE) | 0.043 |

| probability | n | accuracy | mean probability |
|---|---|---|---|
| 0.0-0.5 | 8 | 88% | 0.15 |
| 0.5-0.7 | 0 | n/a | n/a |
| 0.7-0.9 | 4 | 100% | 0.84 |
| 0.9-1.0 | 207 | 98% | 0.99 |

## Per field

| field | accuracy |
|---|---|
| change_order_number | 100% |
| project_name | 100% |
| project_number | 90% |
| contractor | 100% |
| owner | 100% |
| date_issued | 100% |
| description | 100% |
| reason | 100% |
| subtotal | 80% |
| markup | 100% |
| tax | 100% |
| total_change_amount | 100% |
| currency | 100% |
| original_contract_sum | 100% |
| previous_change_orders_total | 90% |
| revised_contract_sum | 90% |
| schedule_impact_days | 100% |
| revised_completion_date | 100% |
| status | 100% |
| contractor_signed | 100% |
| owner_signed | 100% |
| line_items | 100% |
