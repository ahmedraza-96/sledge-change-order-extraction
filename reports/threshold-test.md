# Review threshold test

20 documents, 440 fields, 9 wrong. Rule: highest F2 (recall weighted 2x), checked leaving one document out.

**Chosen threshold: 0.65**: catches 3/9 errors (33%), sends 5% of fields to review.

On documents it was not picked from: catches 2/9 (22%), 5% reviewed (threshold picked per fold: 0.55-0.65).

| below | errors caught | recall | fields reviewed | review rate | flagged that are wrong | F2 |
|---|---|---|---|---|---|---|
| 0.50 | 1/9 | 11% | 5 | 1% | 20% | 0.122 |
| 0.55 | 2/9 | 22% | 12 | 3% | 17% | 0.208 |
| 0.60 | 2/9 | 22% | 16 | 4% | 12% | 0.192 |
| 0.65 <-- chosen | 3/9 | 33% | 24 | 5% | 12% | 0.250 |
| 0.70 | 3/9 | 33% | 26 | 6% | 12% | 0.242 |
| 0.75 | 3/9 | 33% | 47 | 11% | 6% | 0.181 |
| 0.80 | 3/9 | 33% | 53 | 12% | 6% | 0.169 |
| 0.85 | 5/9 | 56% | 197 | 45% | 3% | 0.107 |
| 0.90 | 5/9 | 56% | 201 | 46% | 2% | 0.105 |
| 0.95 | 7/9 | 78% | 205 | 47% | 3% | 0.145 |
| 1.00 | 9/9 | 100% | 307 | 70% | 3% | 0.131 |

Wrong fields, lowest confidence first:

- 0.374 threshold-luna/10_handwritten_override_fax `previous_change_orders_total`
- 0.545 threshold-luna/01_aia_clean_phone_photo_sideways `subtotal`
- 0.646 threshold-luna/10_handwritten_override_fax `revised_contract_sum`
- 0.837 threshold-luna/10_handwritten_override_fax `project_number`
- 0.842 threshold-luna/03_credit_phone_photo `subtotal`
- 0.917 luna/05_email `owner`
- 0.929 luna/05_email `date_issued`
- 0.951 luna/03_credit `subtotal`
- 0.960 luna/01_aia_clean `subtotal`
