# OCR benchmark: OpenRouter vision vs IBM Docling

Same 10 scanned-looking images of the ground-truth samples, one model call each (no repair pass).

| | Vision (`nemotron-3.5-super-vl`) | Docling OCR → text model (`nemotron-3-super`) |
|---|---|---|
| Field accuracy | **95.5%** (210/220) | **91.9%** (182/198) |
| Wrong but confident (≥0.70) | 5 | 11 |
| Documents completed | 10/10 | 9/10 |
| Mean time per document | 38 s | 64 s |

## Per document

| Document | Vision | Docling |
|---|---|---|
| 01_aia_clean | 21/22 · 69s · wrong: subtotal (0.30) | 21/22 · 78s · wrong: subtotal (0.94) |
| 02_contractor_table | 22/22 · 34s · wrong: all correct | 21/22 · 58s · wrong: line_items (1.00) |
| 03_credit | 21/22 · 36s · wrong: subtotal (0.30) | 21/22 · 48s · wrong: subtotal (0.83) |
| 04_scanned | 22/22 · 57s · wrong: all correct | 22/22 · 74s · wrong: all correct |
| 05_email | 19/22 · 24s · wrong: owner (0.55), contractor_signed (0.75), owner_signed (0.75) | 18/22 · 48s · wrong: owner (0.91), total_change_amount (0.30), contractor_signed (0.93), owner_signed (0.93) |
| 06_messy | 22/22 · 19s · wrong: all correct | 21/22 · 69s · wrong: line_items (0.85) |
| 07_missing_fields | 20/22 · 22s · wrong: subtotal (0.75), status (0.73) | 20/22 · 45s · wrong: subtotal (0.30), status (0.93) |
| 08_bad_math | 21/22 · 29s · wrong: line_items (0.10) | 18/22 · 77s · wrong: subtotal (0.30), previous_change_orders_total (0.95), contractor_signed (0.95), owner_signed (0.86) |
| 09_multipage | 20/22 · 34s · wrong: schedule_impact_days (0.30), line_items (0.85) | error: output truncated at max_tokens |
| 10_handwritten_override | 22/22 · 52s · wrong: all correct | 20/22 · 77s · wrong: subtotal (0.56), tax (0.56) |

## Per field

| Field | Vision | Docling |
|---|---|---|
| change_order_number | 10/10 | 9/9 |
| project_name | 10/10 | 9/9 |
| project_number | 10/10 | 9/9 |
| contractor | 10/10 | 9/9 |
| owner | 9/10 | 8/9 |
| date_issued | 10/10 | 9/9 |
| description | 10/10 | 9/9 |
| reason | 10/10 | 9/9 |
| subtotal | 7/10 | 4/9 |
| markup | 10/10 | 9/9 |
| tax | 10/10 | 8/9 |
| total_change_amount | 10/10 | 8/9 |
| currency | 10/10 | 9/9 |
| original_contract_sum | 10/10 | 9/9 |
| previous_change_orders_total | 10/10 | 8/9 |
| revised_contract_sum | 10/10 | 9/9 |
| schedule_impact_days | 9/10 | 9/9 |
| revised_completion_date | 10/10 | 9/9 |
| status | 9/10 | 8/9 |
| contractor_signed | 9/10 | 7/9 |
| owner_signed | 9/10 | 7/9 |
| line_items | 8/10 | 7/9 |
