# Phase 2 - Cheque understanding: complete code explanation

Version **0.3.0** · Course outcomes **CO-3** (feature detection and matching, the core of this
phase), **CO-2** (OCR), **CO-1** (template subtraction).

Phase 2 turns a flattened cheque photo into validated fields and rule checks:

```
... enhance -> align -> fields -> rules -> (ocr only for non-cheques)
                |         |         |
       which bank? where  read each  amount words = figures?
       warp onto its      field,     stale / post-dated?
       template           parse it   IFSC, account, MICR valid? signed?
```

The key design decision, from the SRS: **cheques are fixed-layout**, so we don't learn where
fields are. We identify the bank's template, compute a homography from the photo to that
template, and every field box is then known exactly. Invoices vary in layout and get a learned
model in Phase 6.

---

## 1. Big picture

| Step | Technique | File |
| --- | --- | --- |
| Get test data | Render synthetic CTS-2010-style cheques for 4 fictional banks with known ground truth and fraud conditions | `findoc/synth/cheque.py`, `dataset.py` |
| Which bank? | SIFT keypoints + Lowe ratio test + RANSAC homography against every template; most inliers wins | `findoc/localize/align.py` |
| Where is each field? | Warp the photo into template coordinates (registration); field boxes come from the template | `align.py`, `templates.py` |
| Isolate the writing | Template subtraction: ink = pixels darker than the (eroded) blank template | `findoc/recognize/fields.py` |
| Read | OCR on each clean crop (recognition only, with digit allowlists) | `fields.py`, `ocr.py` |
| Make sense of it | Parse words (lakh/crore), figures (Indian grouping), date, IFSC, account, MICR; repair OCR typos | `findoc/recognize/parsers.py` |
| Judge it | Rules: amount mismatch, stale, post-dated, formats, missing signature | `findoc/fraud/rules.py` |
| Measure | Geometry accuracy and robustness to OCR errors | `experiments/phase2_cheque_eval.py` |

---

## 2. How a run flows

`findoc process data/samples/synthetic_cheque_mismatch.jpg -s outputs`:

1. Phase 1 stages run as before: the cheque is detected, flattened and its lighting normalised.
2. `align`: the flat image is resized to the template width (1624 px), SIFT features are
   computed once, and matched against each of the 4 templates. The template with the most
   RANSAC inliers wins if it has at least 40 and a sane homography. `doc.doc_type` becomes
   `cheque`, `doc.template_id` is set, the photo is warped into template coordinates
   (`doc.registered`), and `template_inliers`, `template_inlier_ratio` and `template_margin`
   are recorded.
3. `fields`: the OCR engine is loaded (load time reported separately), each field box is
   cropped from the registered image after template subtraction, read, and parsed into
   `doc.fields` (`value` as read, `normalized` validated or `None`).
4. `rules`: every rule runs and is added to `doc.checks`, fired or not.
5. `ocr`: skipped for cheques (the fields were already read).
6. The CLI prints the field table and the checks table; `--save-dir` also writes
   `<name>_fields.png`, the registered cheque with coloured field boxes.

Without OCR installed (or with `--no-ocr`), only the signature check runs, since it needs no
text.

---

## 3. File-by-file walkthrough

### 3.1 `findoc/synth/cheque.py`: synthetic cheques

Real cheque images are scarce, unannotated and contain personal financial data, so we render
our own. All banks, names and numbers are fictional.

**Layout.** Indian cheques follow the RBI's **CTS-2010** standard (Cheque Truncation System):
a fixed leaf size (about 203 x 92 mm) with the date boxes top-right, "Pay" line, two "Rupees"
lines, the figures box, account number, signature area bottom-right and a **MICR** band at the
bottom. We render at 8 px/mm: **1624 x 736 px**.

**`BankStyle`** gives each fictional bank its own tint, logo, security-pattern frequency and a
small layout offset (`dx`, `dy`). Real banks also share the CTS layout and differ in exactly
these ways, which is what makes template identification a real task.

**`field_specs(style)`** returns the `FieldSpec` for each field: its box in template pixels,
its kind (`handwritten`, `printed`, `signature`) and an OCR **allowlist** (e.g. digits only for
the date), shifted by the bank's offsets.

**`render_blank(style)`** draws the pre-printed leaf: tint, the **security pattern** (faint
sine-wave lines like the guilloche printed on real cheques to defeat photocopying), logo, bank
name, labels, date boxes, guide lines, figures box and a plain white MICR band. This blank leaf
**is** the template's reference image.

**`render_cheque(rng, ...)`** fills it in: a random amount (`random_amount`, mostly round
business amounts from Rs 100 to 50 lakh), the amount in words split over two lines, the figures
with Indian grouping (`8,40,515/-`), a date (one digit per box), payee, account number, IFSC,
MICR line and a signature scribble (a smooth curve through random control points). Handwriting
uses a TrueType handwriting font if you put one in `data/fonts/` or set
`FINDOC_HANDWRITING_FONT`, otherwise OpenCV's script font.

**Fraud flags** create the conditions the rules must catch:

| Flag | What the generator does |
| --- | --- |
| `mismatch` | writes words for a different amount (x2, x10 or +1000) than the figures |
| `stale` | dates the cheque 100-300 days before today |
| `postdated` | dates it 5-60 days after today |
| `unsigned` | leaves the signature area empty |

**`ChequeTruth`** records everything that was written, including `fields`: the exact text in
each box, which is what a perfect OCR would return.

### 3.2 `findoc/synth/dataset.py`: cheque photos

`cheque_photo(seed, fraud)` renders a cheque, then photographs it with the Phase 1 generator
(landscape 1600 x 1200, cheque covering 35-60% of the frame), and also returns the **ground-
truth homography** from cheque pixels to photo pixels. That matrix lets the evaluation check
exactly where each field really is in the photo. `generate_cheques` writes a dataset with
`labels.json` (used by `findoc synth --kind cheque`).

`photograph()` in `photo.py` was generalised to keep any page's aspect ratio (cheques are 2.2:1,
pages 0.77:1), shrinking both sides together if the page wouldn't fit the frame.

### 3.3 `findoc/localize/templates.py`: the template store

A `ChequeTemplate` is `id`, `name`, `reference` (the blank leaf image) and `fields`
(`FieldSpec` list). Templates can be saved to and loaded from disk:

```
data/templates/ARVL/reference.png
data/templates/ARVL/template.json    {"id", "name", "fields": [{name, box, kind, allowlist}]}
```

`TemplateStore.from_dir()` loads a folder of them, so supporting a real bank means scanning a
blank leaf and writing its field boxes (the SRS's FR-APP-08), with no code change.
`builtin_store()` renders the four fictional banks once and caches them (`@lru_cache`);
`default_store()` uses `FINDOC_TEMPLATES_DIR` when set. `findoc templates export` writes the
built-ins to disk as a starting point.

`draw_fields()` overlays the field boxes (orange handwritten, blue printed, green signature).

### 3.4 `findoc/localize/align.py`: which bank, and where is everything?

This is the CO-3 heart of the phase.

**Step 1: keypoints and descriptors (`FeatureMatcher.detect`).**
**SIFT** (Scale-Invariant Feature Transform):
- builds a scale space (the image blurred at increasing sigma) and finds extrema of the
  **Difference of Gaussians** across position and scale: blob-like points that are stable when
  the image is resized;
- assigns each point a dominant gradient orientation, so the descriptor can be rotated to a
  canonical direction (rotation invariance);
- describes the 16x16 neighbourhood as 4x4 cells x 8 orientation bins = a **128-dimensional
  histogram of gradients**, normalised for brightness changes.

Before detection, **CLAHE** (Contrast-Limited Adaptive Histogram Equalisation) boosts local
contrast so faint print still produces keypoints. `method="orb"` switches to **ORB** (FAST
corners + binary BRIEF descriptors compared by Hamming distance): faster, less accurate.

**Step 2: matching with Lowe's ratio test (`FeatureMatcher.match`).** For each photo descriptor,
`knnMatch(k=2)` finds the two nearest template descriptors. A match is kept only if

```latex
d_1 < 0.75 \, d_2
```

i.e. the best match is clearly better than the runner-up. Repetitive structures (the wavy
security pattern, the eight identical date boxes) produce near-ties and are discarded here.

**Step 3: RANSAC homography (`_align`).** Some kept matches are still wrong. **RANSAC**
(RANdom SAmple Consensus) repeatedly:
1. picks 4 random matches and computes the homography they define;
2. counts **inliers**: matches whose photo point, mapped by that homography, lands within 5 px
   of its template point;

and keeps the model with the most inliers (then refits it on all of them).
`cv2.findHomography(src, dst, cv2.RANSAC, 5.0)` does this and returns the inlier mask. The
number of inliers is the match score: a correct template typically gets 300-400; a wrong bank
still gets ~140, because all banks share the CTS layout (same labels, lines, boxes).

**Step 4: sanity check (`homography_is_sane`).** A homography can be numerically valid but
physically absurd (folded inside out, collapsed to a sliver). We map the template's corners
back into the photo and require a **convex** quadrilateral with a reasonable area.

**Step 5: identification (`identify`).** Photo features are computed **once**; template
features are cached (`template_features`). The template with most inliers wins if it has at
least `MIN_INLIERS = 40`; otherwise the document is not a known cheque (a Phase 1 page gets
under 40). `AlignResult.scores` keeps every template's inlier count.

**Registration (`register`).** `warpPerspective` with the winning homography produces an image
the same size as the template, in which every field sits exactly in its template box.

**Scale bookkeeping.** The flat image is first resized to the template width
(`resize_to_width`, returning the 3x3 scale matrix S). The homography stored in the document
maps **flat-page pixels to template pixels**: `H_template @ S`. Chaining with Phase 1's
homography (photo to flat) lets us map any template box all the way back to the original photo,
which is exactly how the evaluation measures field-box accuracy.

### 3.5 `findoc/localize/stages.py`: the align and fields stages

`align_stage` runs identification and stores `doc_type = CHEQUE`, `template_id`,
`template_homography`, `registered` and three quality numbers:

| Metric | Meaning |
| --- | --- |
| `template_inliers` | RANSAC inliers for the winning template |
| `template_inlier_ratio` | inliers / ratio-test matches |
| `template_margin` | winner's inliers / runner-up's inliers (~2.2-2.6 on synthetic data) |

If the margin is below 1.3 the stage warns that the bank is ambiguous, rather than silently
trusting a near-tie.

`fields_stage(doc, use_ocr)` loads the OCR engine (or notes that it's unavailable) and calls
`read_cheque_fields`. `make_fields_stage` uses `functools.partial` to bake `use_ocr` into a
one-argument stage, so `--no-ocr` works without changing the stage signature.

`registered_box_to_flat` maps a template box back onto the flat page (for drawing on the
user's image later).

### 3.6 `findoc/recognize/fields.py`: template subtraction and reading

**Template subtraction (`ink_mask`).** After registration, the photo and the blank reference
are pixel-aligned. Pre-printed content is in both; what was written exists only in the photo:
1. Normalise lighting in both (Phase 1's `normalize_illumination`), so paper is ~255 in both.
2. **Erode** the reference with a 5x5 kernel (minimum filter): its dark lines get a little
   thicker, so a 1-2 px misregistration doesn't leave thin ghost edges of template lines.
3. Ink = pixels **at least 45 grey levels darker** than the eroded reference.
4. A small morphological opening removes isolated speckle.

`crop_fields` then builds each field crop as **ink pixels on a white background**, and records
the fraction of the box that is ink (`ink_ratio`). The README figure shows the result: date
boxes, guide lines, labels and the security pattern are all gone.

**Reading.** Each crop is trimmed to its ink (`_prepare`), scaled to 64 px height and read with
`engine.read_line(img, allowlist)`. `read_line` uses EasyOCR's `recognize()`, which **skips
text detection** and reads the whole crop as one line: we already know where the text is. The
allowlist (e.g. digits for the date and account number) stops the recogniser from producing
letters in numeric fields. Empty boxes (`ink_ratio < 0.1%`) are not sent to OCR at all.

`_read` also supports engines with a `read_field(name, ...)` method: the simulated OCR used in
tests and experiments needs to know which field it's "reading".

**Normalising.** Each field gets `value` (text as read) and `normalized` (validated canonical
value or `None`), via the parsers. The amount in words joins both lines; its confidence is the
OCR confidence times the parser's confidence (lower when typos had to be repaired).

**Boxed fields (`compact_cells`, v0.3.1).** A `FieldSpec` with `cells > 1` (the date: 8 cells)
holds one character per equal-width box. Before reading, the crop is split into its cells, each
character is trimmed to its ink, and the characters are re-joined with a gap of 15% of the crop
height, like normal letter spacing. Empty cells are skipped (a missing digit then fails date
validation instead of being invented). Without this, the recogniser saw "1   8   0   9 ..." with
huge gaps and sometimes dropped digits: date accuracy was 94% in the real-OCR evaluation.

**Signature.** No OCR: present if at least 0.6% of the signature box is ink after subtraction.

### 3.7 `findoc/recognize/parsers.py`: from noisy text to validated values

**Indian numbering.** 1,00,000 = one **lakh**; 1,00,00,000 = one **crore**.

- `number_to_words(n)` splits n into crore / lakh / thousand / hundred groups recursively
  (crores can exceed 99, so the crore count is itself converted recursively).
- `amount_in_words` produces cheque style: "Twenty Five Thousand and Fifty Paise Only".
- `indian_grouping(n)`: last three digits, then groups of two: `12,34,56,789`.

**`words_to_amount(text)`**, the important one:
1. Tokenise into lowercase words (hyphens become spaces).
2. **Repair typos** (`_closest`): unknown tokens are snapped to the nearest number word using
   `difflib` similarity, only if the lengths are close. Two-letter tokens are never repaired,
   and three-letter tokens only to words at most one letter longer, so "fve" becomes "five" but
   "the" never becomes "three" (a test caught exactly that bug).
3. If unknown words outnumber number words, the text isn't an amount: return `None`.
4. Separate paise ("... and fifty paise").
5. Accumulate: small numbers add to `current`; "hundred" multiplies `current`; thousand/lakh add
   `current x multiplier` to `total`; crore multiplies everything so far.
6. Return `AmountParse(value, corrections, unknown)`; its `confidence` drops with each repair.

Round trip `number_to_words` then `words_to_amount` is exact for 2,000 random amounts up to
Rs 1,000 crore (tested).

**Other parsers.** Each applies **type-specific OCR repairs**: in numeric fields
`O, D, Q -> 0`, `I, l, | -> 1`, `S -> 5`, `B -> 8`, and so on (`DIGIT_FIXES`); in the IFSC's
letter part the reverse (`LETTER_FIXES`).

| Function | Accepts | Validates |
| --- | --- | --- |
| `parse_amount_figures` | `Rs. 2,50,000.00/-`, `25,OOO` | strips symbols, keeps only the last dot as decimal |
| `parse_cheque_date` | `15112026`, `1 5 1 1 2 0 2 6`, `151126` | a real calendar date (31 Nov is rejected) |
| `normalize_ifsc` | `arvl O001234` | 4 letters + `0` + 6 alphanumerics |
| `normalize_account` | any digits | 9-18 digits |
| `parse_micr` | `123456 302912282 834713 10` | cheque no. (6) + MICR code (9: city, bank, branch) + account id (6) + transaction code (2) |

`add_months` handles month ends (30 Nov + 3 months = 28 Feb).

### 3.8 `findoc/fraud/rules.py` and `findoc/fraud/stages.py`: the checks

Every rule returns a `CheckResult` **whether or not it fired**, so the report shows what passed
too. A rule never guesses: an unreadable field yields an `*_UNREADABLE` result instead of a
pass or a fail.

| Check | Fires when | Severity |
| --- | --- | --- |
| `AMOUNT_MISMATCH` | words and figures differ by more than Rs 0.50 | 1.0 |
| `AMOUNT_UNREADABLE` | either amount can't be parsed | 0.4 |
| `DATE_POSTDATED` | date is after today | 0.6 |
| `DATE_STALE` | today is more than 3 months after the date (RBI validity) | 0.9 |
| `DATE_UNREADABLE` | not a valid date | 0.5 |
| `IFSC_INVALID`, `ACCOUNT_INVALID` | format check fails | 0.4 |
| `MICR_INVALID` | not 6+9+6+2 digits | 0.3 |
| `SIGNATURE_MISSING` | signature box has no ink | 0.8 |

Format rules have low severity because a single OCR error trips them; they are sanity checks,
not evidence of fraud. `confidence` comes from the fields involved. Severity and confidence are
the `s_i` and `c_i` of the Phase 5 risk score.

`rules_stage` runs the rules for cheques only, and only the signature rule when OCR didn't run.

### 3.9 `findoc/recognize/ocr.py`: what changed

- `OcrEngine` gains `read_line(img, allowlist)` and a `device` attribute.
- `load_engine()` returns the engine and, if this call created it, the load time.
  `note_engine_load` records `ocr_model_load_ms` and a message such as
  "OCR model loaded on gpu in 4.9 s". Stage timings then reflect recognition, not model
  loading (your 6.7 s "OCR" in Phase 1 was mostly loading).
- `ocr_stage` skips cheques (their fields are read by the `fields` stage).

### 3.10 `findoc/synth/ocr_noise.py`: simulated OCR

To measure robustness independently of any OCR model, `OracleOcr` returns each field's true
text with **controlled character errors** (`corrupt`, at a Character Error Rate `cer`):
60% substitutions with realistic confusions (`0/O`, `rn/m`, `e/c`; with a digits-only allowlist,
digit/digit confusions like `1/7`, `3/8`), 25% deletions, 15% spurious spaces. Each cheque uses
its own random seed (an early version reused one seed for every cheque, which made errors hit
the same positions every time and gave non-monotonic results).

### 3.11 CLI and demo

- `findoc synth --kind cheque --n 20 [--fraud-rate 0.4]`: cheque photos + labels.
- `findoc process <cheque>`: field table (read / validated / confidence) and checks table.
- `findoc templates list` / `findoc templates export --out data/templates`.
- Streamlit: choose "Cheque" and a fraud condition, then see the registered cheque with field
  boxes, the template-subtracted crops ("what OCR sees"), the field table, ground truth, and
  each check with its reason.

### 3.12 `experiments/phase2_cheque_eval.py`: results

120 synthetic cheque photos: half clean, half spread over the four fraud types.

**Geometry (no OCR needed):**

| Metric | Result |
| --- | --- |
| Template identification | 100% |
| Field-box IoU vs. ground truth, mean / worst | 0.995 / 0.982 (SRS target >= 0.80) |
| Signature presence | 100% |
| Median alignment time (4 templates, CPU) | ~0.4 s |

Field-box IoU compares each template box mapped into the photo through our estimated chain
(template to flat to photo) with the same box mapped through the ground-truth homography.

**Robustness to OCR errors (simulated):**

| Metric | CER 0% | CER 2% | CER 5% | CER 10% |
| --- | --- | --- | --- | --- |
| Amount mismatch recall | 100% | 100% | 100% | 93% |
| Stale recall | 100% | 87% | 87% | 73% |
| Post-dated recall | 100% | 100% | 93% | 80% |
| Unsigned recall | 100% | 100% | 100% | 100% |
| Clean cheques with any flag | 0% | 45% | 72% | 93% |
| ... flagged by AMOUNT_MISMATCH | 0% | 20% | 40% | 57% |

**Real OCR** (EasyOCR on an RTX 3050 laptop GPU, 100 cheques, v0.3.0):

| Metric | Result |
| --- | --- |
| Payee, amount in words, account, IFSC, MICR accuracy | 100% each |
| Amount in figures accuracy | 99% |
| Date accuracy | 94% |
| Recall: mismatch / stale / unsigned / post-dated | 100% / 100% / 100% / 83% |
| Clean cheques with any flag | 8% (date unreadable 6%, amount mismatch 2%) |
| Median field-reading time | 0.49 s per cheque |

Real OCR on template-subtracted crops behaves better than the 2% simulation: the clean crops
are close to what OCR models are trained on. The **date** was the weak field: each digit sits
in its own box, so after subtraction the digits are far apart and the recogniser sometimes
dropped one. That single issue caused both missed post-dated cheques and most false flags.
**v0.3.1** fixes it with `compact_cells` (section 3.6): the date crop is cut into its 8 cells,
each digit trimmed, and the digits re-joined with normal spacing before OCR.

**What this means (put it in the paper):** detection of real fraud is robust to OCR errors,
because a misread rarely turns a bad cheque into a consistent one. The cost is **false flags**:
one misread digit in the figures looks exactly like an altered amount. Since altering a single
digit is also what fraudsters do, the rule can't simply ignore one-digit differences. The fix
is to weigh each flag by OCR confidence and to route flagged cheques to review rather than
reject them, which is the Phase 5 risk score's job. Real-OCR numbers come from
`python experiments/phase2_cheque_eval.py --real-ocr` on a machine with EasyOCR.

### 3.13 Tests (111 in total, 95% coverage)

| File | What it checks |
| --- | --- |
| `test_parsers.py` | Indian-system words both ways, 2,000-amount round trip, typo repair, rejection of non-amounts, figures, grouping, dates, IFSC/account/MICR |
| `test_rules.py` | each rule fires exactly when it should, tolerance, unreadable handling, no-OCR mode |
| `test_cheques.py` | generator consistency and fraud flags; template save/load; every bank identified from a photo; ORB alternative; a non-cheque is not matched; field boxes within IoU 0.85 of ground truth; template subtraction removes pre-printed content; reading with simulated OCR; full pipeline flags each fraud type and nothing on a clean cheque |
| `test_pipeline.py`, `test_cli.py` | new stage order; cheque synth/process/templates commands |

---

## 4. Concepts used in this phase

| Concept | Explanation |
| --- | --- |
| Template matching vs. learned localisation | Fixed layouts are solved exactly by registration to a known template; variable layouts need a learned model. |
| Keypoint detection | Finding repeatable points (blobs, corners) that can be re-found in another view. |
| SIFT | Difference-of-Gaussians keypoints + 128-D gradient-orientation descriptors; invariant to scale, rotation, brightness. |
| ORB | FAST corners + binary BRIEF descriptors; fast, Hamming distance. |
| Lowe's ratio test | Keep a match only if it's clearly better than the second-best. |
| RANSAC | Robust model fitting: random minimal samples, count agreeing inliers, keep the best. |
| Homography (again) | Here estimated from hundreds of noisy matches instead of 4 exact corners. |
| Image registration | Warping one image into another's coordinate frame so pixels correspond. |
| Image differencing | Subtracting an aligned reference to isolate what changed (here: the writing). |
| Erosion as a tolerance | Thickening reference strokes so small misalignments don't create false differences. |
| OCR allowlists | Constraining recogniser output to a field's alphabet. |
| Fuzzy string matching | `difflib` similarity to snap OCR typos to a known vocabulary. |
| Validation with explicit unknowns | Parsers return `None` rather than a guess; rules report "unreadable". |
| Precision vs. recall | Catching all fraud (recall) vs. not flagging clean cheques (false positives). |

---

## 5. Try it yourself

1. `findoc process data/samples/synthetic_cheque_mismatch.jpg -s outputs` (with OCR
   installed). Which check fires? Open `outputs/synthetic_cheque_mismatch_fields.png`.
2. `streamlit run demo/streamlit_app.py`: generate a cheque for each fraud condition and look
   at "What OCR sees". Which field is hardest for OCR?
3. In `align.py`, change `RATIO` to 0.95 and run a cheque. What happens to `template_margin`?
   Why? Change it back.
4. Run `findoc templates export`, open `data/templates/ARVL/template.json`, move the payee box
   20 px up, set `$env:FINDOC_TEMPLATES_DIR="data/templates"`, and process an Aravalli cheque.
   What does the payee crop look like now?
5. `python -c "from findoc.recognize.parsers import words_to_amount as w; print(w('Eigth Lakh Fourty Thousnd'))"`.
   Which corrections were made?
6. With OCR on your GPU: `python experiments/phase2_cheque_eval.py --n 100 --real-ocr`. Compare
   the real-OCR column with the simulated ones. Which CER does EasyOCR behave like?

---

## 6. Viva and interview questions

**Q: Why templates for cheques and a learned model for invoices?**
A: Cheques follow the fixed CTS-2010 layout, so registering a photo to its bank's template gives
every field's position exactly, with no training data (a public cheque dataset of 112 images
gives almost nothing to train on anyway). Invoice layouts vary per vendor, so they need a
layout-aware model (LayoutLMv3, Phase 6).

**Q: How do you identify the bank?**
A: SIFT features in the photo are matched to each bank's blank template with Lowe's ratio test,
a homography is fitted with RANSAC, and the template with the most inliers wins. The correct
bank gets 300-400 inliers versus about 140 for others; we also record the margin over the
runner-up and warn below 1.3x.

**Q: Why does a wrong bank still get 140 inliers?**
A: All banks share the CTS layout: the same labels, lines and boxes match geometrically.
Discrimination comes from the logo, name and security pattern, hence the margin check.

**Q: What is template subtraction and why does it help?**
A: After registration the photo and the blank leaf are aligned, so pixels darker than the
template are the writing. Removing guide lines, box borders and the security pattern gives OCR
clean ink on white. Eroding the template first tolerates 1-2 px misalignment.

**Q: How accurate is field localisation?**
A: Mean IoU 0.995 (worst 0.982) between estimated and true field boxes in the photo, on 120
synthetic photos, against an SRS target of 0.80.

**Q: How do you parse "Eight Lakh Forty Thousand" when OCR makes mistakes?**
A: Tokens are snapped to the closest number word only when spelling and length are close (so
"thousnd" becomes "thousand" but "the" never becomes "three"), then accumulated with
hundred/thousand/lakh/crore multipliers. If unknown words dominate, it returns "unreadable".

**Q: What is a stale cheque and how do you detect it?**
A: In India a cheque is valid for three months from its date (RBI). We parse the date boxes and
flag `DATE_STALE` if today is more than three months later, `DATE_POSTDATED` if the date is in
the future.

**Q: Your false-positive rate is high under OCR noise. Is the system useless?**
A: No. Recall of real fraud stays high (100% for amount mismatch at 2-5% character error), but
a single misread digit mimics an altered amount, so clean cheques get flagged for review. The
Phase 5 risk score weights each check by OCR confidence, and the system's role is to route
cheques to a reviewer, not reject them. We report both numbers honestly.

**Q: How accurate is it with real OCR?**
A: On 100 synthetic cheques with EasyOCR on a laptop GPU: 100% on payee, amount in words,
account, IFSC and MICR; 99% on figures; 94% on the date. Amount mismatches, stale and unsigned
cheques were all caught; 8% of clean cheques got a flag. The date errors came from widely spaced
boxed digits, which we fixed by re-spacing them before OCR.

**Q: Why ORB as an option?**
A: ORB is several times faster and patent-free; SIFT is more accurate under blur and scale
change. Both are behind one interface, and the SuperPoint + LightGlue learned matcher used in
Phase 4 can be added the same way.
