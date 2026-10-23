Change-order test pack
======================
Upload any of these at http://127.0.0.1:8000 (provider Gemini; type gemini-2.5-flash in the Model box
while the default model's daily quota is used up).

1_original_samples/   The 10 generated documents. Ground truth is in the repo's data/truth/.
  01 clean form, 02 contractor table, 03 credit (negative total), 04 image-only scan,
  05 email with a superseded quote below it, 06 messy text, 07 missing fields (CAD),
  08 document's own math is wrong, 09 table continues on page 2, 10 handwritten override.

2_photos_and_scans/   The same documents, as real-world images. OCR runs on these (about 1.5-2 min each).
  01_..._phone_photo_sideways.jpg  phone photo saved sideways with an EXIF rotation tag
  03_credit_phone_photo.jpg        tilted, uneven lighting, slight blur and noise
  08_bad_math_phone_photo.jpg      photo of the document whose arithmetic is wrong
  02_..._screenshot.png            clean screenshot
  10_..._fax.png                   black-and-white, low resolution, skewed
  09_multipage_scan.tiff           2-page TIFF
  07_..._lowcontrast.webp          faded, low-quality WebP

3_new_documents/      Documents the pipeline has never seen. Expected answers:
  A_new_deduct_change_order.pdf / _photo.jpg
      CO-031, total -12387.60 (a deduct), subtotal -11470.00, markup -917.60,
      original 4310000.00, previous 96250.00, revised 4393862.40, schedule 0 days,
      status approved, both signed.  All the arithmetic checks pass.
  B_not_a_change_order_invoice.pdf
      An invoice. Expect a "not_a_change_order" issue and document confidence 0.
  C_new_text_change_order.txt
      CO 14, total 21321.00, subtotal 18540.00, markup 2781.00, original 2875000.00,
      previous 143600.00, revised 3039921.00, +6 days, completion 2026-12-11,
      status pending, contractor signed, owner not signed.
