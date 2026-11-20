# Phase 6 - Fraud engine: complete code explanation

Version **0.7.0** · Course outcomes **CO-2** (a learned classifier on hand-designed features),
**CO-3** (feature matching and homographies for duplicates), **CO-1** (image forensics, JPEG
compression), and the payoff of **CO-5**: the 3D correspondence map carries forensic evidence.

## Why this phase exists

Phases 1-5 turn a phone capture into a clean, flat, read cheque, and the Phase 2 rules catch
*inconsistent* cheques (words vs. figures, stale dates). Three common frauds look perfectly
consistent:

| Fraud | What the fraudster does | What catches it |
| --- | --- | --- |
| Altered amount, payee or date | Edits the photo before uploading it | `TAMPER_REGION`: the edited area has a different compression history |
| Duplicate presentation | Presents the same cheque twice (two banks, or a photo after depositing the paper) | `DUPLICATE`: the handwriting matches a cheque already seen |
| Forged signature | Signs in the account holder's name | `SIGNATURE_MISMATCH`: shape differs from the account's references |

Every check, old and new, then feeds one **explainable risk score** with a band (low, medium,
high) that a reviewer can recompute by hand.

The novel part is **where** tamper detection runs. Dewarping (2D or 3D) resamples every pixel,
which smears the tiny traces forensic detectors look for. So the detector runs on the
**original** capture and its heatmap, not the image, is carried to the flat page through the
correspondence map. This was the SRS hypothesis; section 3.8 measures it.

```
original capture --[tamper model]--> heatmap --[homography or 3D map]--> heatmap on flat page
                                                                       --[template homography]--> per-field scores
registered cheque --[template subtraction]--> handwriting --[SIFT + RANSAC vs. index]--> DUPLICATE
                                           \-> signature crop --[HOG + layout vs. references]--> SIGNATURE_MISMATCH
all checks --> risk score = sum of points x severity x confidence (capped at 100) --> band
```

---

## 1. Big picture

| Step | Technique | File |
| --- | --- | --- |
| Synthetic forgeries | Retype (inpaint + new text), splice (from another capture), copy-move (same image, resampled); exact masks; re-saved JPEG | `findoc/synth/tamper.py` |
| Synthetic signers | One movement per person, natural variation; random and skilled forgeries | `findoc/synth/signature.py` |
| Forensic features | Per 16x16 block: JPEG error curve at 10 qualities, its shape, texture, brightness, ink, noise, and all of it relative to this capture | `findoc/fraud/tamper.py` |
| Tamper model | Gradient-boosted trees on block features, trained on synthetic clean/forged cheques, threshold from validation | `findoc/fraud/tamper_train.py`, `findoc tamper train` |
| Mapping | Heatmap -> flat page (homography or 3D map) -> template -> field boxes | `findoc/fraud/tamper.py`, `findoc/fraud/stages.py` |
| Duplicates | Template-subtracted handwriting, SIFT + RANSAC against an index | `findoc/fraud/duplicates.py`, `findoc index`, `--register` |
| Signatures | Normalised crop, HOG + ink-layout descriptor, nearest reference | `findoc/fraud/signature.py`, `findoc signatures` |
| Risk score | Points x severity x confidence, bands, critical overrides | `findoc/fraud/risk.py` |
| Evidence image | Heatmap overlay on the flat page | `findoc/fraud/viz.py` |
| Measure | Tamper mapping experiment; duplicates and signatures | `experiments/phase6_tamper_eval.py`, `experiments/phase6_fraud_eval.py` |

New dependency: scikit-learn (the gradient-boosted trees). No new heavy frameworks.

---

## 2. How a run flows

One-time setup:

```
findoc tamper train                    # ~5 min on a laptop CPU -> models/tamper.pkl (git-ignored)
findoc signatures add 123456789012 ref1.jpg ref2.jpg   # reference signatures of an account
```

`findoc process cheque.jpg` runs the earlier stages, then:

1. **tamper**: block features of the original capture -> P(edited) per block -> heatmap ->
   flat page -> template frame -> score per field. `TAMPER_REGION` fires if the most suspicious
   field (weighted: amount, payee, date 1.0; signature 0.8; others less) passes the model's
   threshold. Without a model, the classical JPEG-ghost heatmap is computed and saved, and the
   check says the model is not trained instead of guessing.
2. **duplicate**: the handwriting layer is compared with every indexed cheque of the same bank.
   With `--register`, the cheque is added to the index afterwards (unless it *is* a duplicate).
   With an empty index, no check is reported.
3. **signature**: if the account is known (OCR's account number, or `--account`) and has
   references, the signature is compared with them.
4. **risk** (last stage): the score, band and every contribution, in the console and the JSON
   (`risk_score`, `risk`).

`--save-dir` also writes `<name>_tamper.jpg`: the flat page with the heatmap overlaid where warm.

Settings: `FINDOC_TAMPER` (`auto`/`off`), `FINDOC_TAMPER_MODEL`, `FINDOC_DUPLICATE_INDEX`
(default `data/index`), `FINDOC_SIGNATURES_DIR` (default `data/signatures`), `FINDOC_RISK_POINTS`
(JSON, per-check points), `FINDOC_RISK_MEDIUM`, `FINDOC_RISK_HIGH`. The index and the
signatures hold data derived from real cheques when used with them: both folders are
git-ignored and must stay private.

---

## 3. File-by-file walkthrough

### 3.1 `findoc/fraud/risk.py`: the score

`risk_score(checks)` adds `points x severity x confidence` for every **fired** check, caps at
100 and sorts the contributions. Points say how much a problem alone may add: 100 for
`AMOUNT_MISMATCH` and `DUPLICATE`, 90 `TAMPER_REGION`, 80 `SIGNATURE_MISMATCH`, 75
`SIGNATURE_MISSING`, 70 `DATE_STALE`, 60 `DATE_POSTDATED`, 40 for unreadable fields, 20-25 for
format errors (often one OCR mistake). Bands: below 30 low, below 70 medium, otherwise high.
`CRITICAL` forces high when the check is confident enough (`AMOUNT_MISMATCH` at 0.9,
`DUPLICATE` 0.8, `TAMPER_REGION` 0.8), even if the points alone stay lower.

The SRS first wrote weights that sum to 1. With about a dozen checks, one proven fraud would
then add about 10 points and land in "low". Points per check state the policy directly, and
remain recomputable: the report lists each contribution.

### 3.2 `findoc/synth/tamper.py`: forgeries with exact masks

The threat model is the common one for remote deposit: the fraudster **edits the photo**. Each
manipulation takes the field's quadrilateral in the image (`box_quad` maps a template box
through the page -> photo homography):

- `retype`: dark ink in the field is found (darker than the paper by 40 levels), dilated, and
  removed with `cv2.inpaint` (Telea). New digits are drawn with a script font into a box-shaped
  alpha mask, warped onto the quadrilateral, slightly blurred and blended in ink colour. The
  new pixels never went through the camera.
- `paste` / `splice`: the same field from **another** capture (another cheque, different light,
  blur, JPEG quality) is warped onto the quad, brightness-matched, and blended with a feathered
  edge (Gaussian-blurred mask).
- `copymove`: a region of the **same** image (handwriting from the payee line), rescaled by
  0.85-1.15 and rotated up to 3 degrees, pasted over the field.
- `tamper` re-saves the result as JPEG at quality 85-95 (an edited upload is re-compressed).
  `changed_mask` (pixels changed by more than 4 levels before that compression, closed, inside
  the field) is the ground truth.
- `capture` renders a cheque photo with a random camera JPEG quality (80-95);
  `forged_sample(seed, forged)` returns everything an evaluation needs. Half the clean samples
  are re-saved too, like an innocent re-upload, so "was re-compressed" alone is not evidence.

### 3.3 `findoc/fraud/tamper.py`: features, model, mapping

**JPEG background.** JPEG cuts the image into 8x8 blocks, transforms each with the DCT and
rounds (quantises) the coefficients; the quality setting decides how coarsely. Compressing an
image again at the quality it was compressed with changes it very little: the coefficients are
already on the quantisation grid. That is the **JPEG ghost** (Farid 2009): re-compress at many
qualities and look where the error dips. A pasted or drawn region has a different history from
the rest, so its curve differs.

- `block_features(img, page)`: per 16x16 block (two JPEG blocks), 35 numbers: the log mean
  squared change after re-compression at qualities 50, 55 ... 95; the same curve minus its mean
  (shape); texture (log mean squared Laplacian), brightness, ink fraction, high-pass noise; and
  the error curve and noise **minus the median of blocks of the same kind on this capture**
  (ink blocks compared with ink, paper with paper). The last group matters because each phone
  compresses and denoises differently: what is normal is set per capture.
- `TamperModel`: wraps the classifier; `block_probability` gives P(edited) per block on the
  page; `save`/`load` pickle it with a version and the feature names, refusing a model made for
  other features. (Pickle executes code when loading: only load model files you trained.)
- `classical_heatmap`: the untrained baseline: each block's worst error ratio (vs. the page
  median, over qualities) in robust standard deviations (median and MAD).
- `upsample`: block grid -> pixels. `page_mask`: the page in the original, from the corners.
- `to_flat`: the core of the "map, don't resample" idea. A photo's flat page was made with a
  homography (original -> flat), so the heatmap is warped with the same homography. A video's
  flat page was made by `cv2.remap` with the Phase 5 maps (`map_x[y, x], map_y[y, x]` = where
  flat pixel (x, y) came from in the best frame), so the heatmap is remapped with the same maps.
  Interpolating a smooth heatmap is harmless; interpolating the pixels is what destroys traces.
- `field_scores`: the 90th percentile of the heat inside each field box: robust to one hot
  block, sensitive to two changed digits.

### 3.4 `findoc/fraud/tamper_train.py`: training and the threshold

`train()` renders 80 clean + 80 forged training cheques in parallel (one process per CPU core
minus one), turns every page block into a training row (about 520,000 blocks, 2% edited) and
fits `HistGradientBoostingClassifier` (300 trees of up to 31 leaves, learning rate 0.08,
`class_weight="balanced"` so the 2% edited blocks are not ignored). Gradient boosting adds small
decision trees one at a time, each fitted to the errors of the ones before; on tabular features
like these it is accurate, fast on a CPU and needs no feature scaling.

Validation uses 40 + 40 other cheques, each scored like the pipeline does (most suspicious
weighted field). The **threshold** is just above the highest clean score (a target of at most
2% false alarms on 40 clean cheques allows none), never below 0.5. Its validation metrics are
stored in the model file and printed: detection 77.5%, false alarms 0%, cheque-level AUC 0.986,
pixel AUC 0.964, edited field named correctly 100%, threshold 0.771, about 5 minutes on two
cores. `view="flat"` trains the same model on dewarped pages (`dewarp_sample`), only for the
experiment's fair baseline.

### 3.5 `findoc/fraud/duplicates.py`: has this cheque been seen?

Two photos of one cheque differ in angle, light and blur; two cheques of one bank share their
whole printed template. What is unique to a leaf is its **handwriting**, and Phase 2 already
isolates it: `ink_image` = `ink_mask` (registered cheque vs. blank template) as dark-on-white.

- `ink_features`: up to 600 SIFT keypoints on the handwriting.
- `verify`: ratio-test matches, one homography by RANSAC (6 px); returns inliers and
  inliers / smaller keypoint count.
- `DuplicateIndex`: `index.json` + one `.npz` of points and descriptors per cheque. `add` appends;
  `search` verifies the query against every entry of the same bank and returns the best first.
  `Match.duplicate`: at least 150 inliers and a ratio of at least 0.2, or an identical MICR line
  when OCR read it (same leaf by definition).

A perceptual hash pre-filter was implemented and **removed**: all cheques have their writing
in the same boxes, so hashes of different cheques were as close as those of one cheque
photographed twice (Hamming 4-10 vs. 2-8 of 64 bits). A filter that cannot separate them can
only hide duplicates.

### 3.6 `findoc/fraud/signature.py` and `findoc/synth/signature.py`

The SRS planned a Siamese CNN; that needs thousands of real signatures from many writers, which
the project cannot use. The classical version:

- `signature_image`: signature box of the registered cheque, template subtracted, cut to the
  ink's bounding box and scaled into a 192x64 canvas (`normalise`). Position, size and pen colour
  no longer matter.
- `describe`: a **HOG** (histogram of oriented gradients: per 16x16 cell, how much stroke runs in
  each of 9 directions; OpenCV 5 dropped its HOG class, so `hog` is 15 lines of NumPy) plus a
  24x8 ink-density layout grid; each L2-normalised.
- `distance`: (1 - cosine) for each, summed. The questioned signature's distance is the
  smallest over the account's references, because people sign with variation.
- `SignatureStore`: `<root>/<account>/<n>.png`.

`Signer` (synthetic) fixes a person's movement: control points of the main stroke, letter loops,
a wave, underline, pen width. `draw` varies it slightly each time; `skilled=True` copies it with
three times the variation, tremor and a different pen width; another `Signer` is a random
forgery. `render_cheque(..., signer=, skilled_forgery=)` signs a cheque with it.

`SIGNATURE_THRESHOLD = 0.39` is the 95th percentile of genuine distances on the calibration
signers (section 3.8).

### 3.7 `findoc/fraud/stages.py`, `pipeline.py`, `cli.py`, `config.py`, `types.py`

- Stages, in order after `rules`: `tamper_stage`, `duplicate_stage(register=)`,
  `signature_stage(account=)`, then `ocr`, then `risk_stage`.
- `default_pipeline(ocr, register, account)` and `process(...)` pass the options through
  (`functools.partial`).
- `Document`: `risk` (band, contributions, forced_by), `tamper_map` (excluded from the JSON).
- CLI: `findoc tamper train|check`, `findoc index list|clear`, `findoc signatures add|list`,
  `findoc process --register --account`, a "Risk score" line, `<name>_tamper.jpg`.
- `tests/conftest.py` points the model, index and signature folders at empty temporary
  folders in every test, so a model or index on your laptop never changes test results.

### 3.8 Results

**Tamper detection: original + map vs. on the flat page** (`experiments/results/phase6.json`;
40 clean + 40 forged synthetic photos, forged field and manipulation cycled; flat page from the
pipeline's homography). Pixel AUC = edited vs. untouched pixels on the flat page; cheque AUC =
most suspicious handwritten field, forged vs. clean cheques.

| Detector | Where it runs | Pixel AUC | Cheque AUC | Edited field named |
| --- | --- | --- | --- | --- |
| Tamper model | **original + map** | **0.953** | **0.956** | **97.5%** |
| Tamper model trained on flat pages | flat page | 0.858 | 0.923 | 90.0% |
| Tamper model | flat page | 0.826 | 0.784 | 80.0% |
| JPEG ghost (untrained) | original + map | 0.707 | 0.446 | 20.0% |
| JPEG ghost (untrained) | flat page | 0.602 | 0.398 | 7.5% |

Per manipulation, pixel AUC for the model on original + map: retype 0.971, splice 0.943,
copy-move 0.945 (flat-trained model on flat: 0.873, 0.868, 0.831).

Reading it: for every detector, running on the original and mapping beats running on the
dewarped page, including against a model trained specifically on dewarped pages (the fair
comparison). These photos are dewarped by a gentle homography; the hypothesis predicts larger
gaps under the non-uniform 3D resampling.

**Videos (3D)**: 6 bent-cheque videos (fold, curl, curl at both ends), the best frame kept as a
JPEG still and edited (3 flattened in 3D, 3 kept the 2D homography). Pixel AUC, original
+ map vs. flat: model 0.769 vs. 0.724, JPEG ghost 0.756 vs. 0.614, and the flat-trained model
drops to 0.554. The cheque-level AUC is mixed (0.917 vs. 0.972 for the model) and 6 clean +
6 forged clips are too few to conclude anything at that level. Stated plainly in the results:
supportive on pixels, inconclusive per cheque, more clips needed.

**End to end** (the same 80 photos through `findoc process`): forged cheques flagged **77.5%**
(retype 100%, splice 69%, copy-move 62%); clean cheques flagged **0%**; when flagged, the edited
field was named correctly **100%** of the time. Risk bands: forged 29 high, 2 medium, 9 low;
clean 40 low. Median processing time 1.5 s per photo (laptop CPU, no OCR).

**Duplicates** (`experiments/results/phase6_fraud.json`, 20 cheques photographed twice):
**20/20** second photos matched their first; **0** false matches in 82 comparisons with other
cheques of the same bank. Same cheque: 232-305 inliers (median 262); different cheques: at most
90 (median 43); threshold 150.

**Signatures** (30 synthetic signers; 15 choose the threshold, the other 15 test it): random
forgeries detected **100%** (AUC 1.00); skilled forgeries **87%** (AUC 0.92); genuine
signatures sent to review **13%** (2 of 15).

#### How we got here: lessons worth telling in a viva

1. **A rule that is right on average fires on every cheque.** The first tamper detector
   flagged blocks whose JPEG error was unusual for the page. Handwriting compresses very
   differently from paper, so every clean cheque scored 1.0 in some field. The question is
   "unusual *for this content on this capture*", which needs a model of normal content: hence
   the learned classifier and the per-capture relative features (cheque AUC 0.95 -> 0.99 when
   those were added).
2. **Make the baseline fair before claiming a win.** "Detect on flat" was also given a model
   trained on flat pages. It is better than the original-trained model on flat pages, and still
   worse than original + map.
3. **Remove what does not separate.** The perceptual hash looked like the textbook duplicate
   filter but could not tell cheques apart here.
4. **Calibrate on one set, report on another.** Thresholds come from validation captures
   (tamper) or calibration signers (signatures), never from the numbers reported.

### 3.9 Tests (`tests/test_fraud.py`)

16 tests: risk arithmetic, bands, points override and critical forcing; each manipulation stays
inside its field and invalid arguments are refused; feature shapes and the classical map; mapping
through a homography and through a remap; a tiny model trained, saved, loaded (and an old
version refused); the tamper stage without a model, with one, and switched off; the duplicate
index on disk (other banks never compared, MICR match); `--register` then a second photo of the
same cheque flagged (and not re-added), an unrelated cheque not flagged; signature descriptors;
`findoc signatures add`, a genuine and a forged signature through `--account`, an unknown
account; and the CLI commands.

---

## 4. Concepts used in this phase

- **JPEG compression**: 8x8 blocks, DCT, quantisation by a quality-dependent table; why
  re-compression at the same quality changes little (idempotence on the quantisation grid).
- **Image forensics cues**: JPEG ghosts / double compression, noise inconsistency, resampling.
  Resampling destroys them, which is why detection must precede dewarping.
- **Gradient-boosted decision trees**: an additive model of small trees, each fitted to the
  remaining errors (gradient of the loss); histogram binning makes it fast.
- **Class imbalance**: 2% positives; `class_weight="balanced"` reweights them.
- **ROC AUC**: the probability that a random positive scores higher than a random negative;
  threshold-free. Pixel-level vs. image-level AUC answer different questions.
- **Operating point**: a threshold chosen for a target false-alarm rate on held-out data.
- **Correspondence maps / warping a map instead of an image**: `warpPerspective` and `remap`
  move values; applied to a heatmap they carry evidence without touching pixels.
- **SIFT + RANSAC verification**: robust matching with a geometric model; inlier count as
  evidence of the same physical object.
- **HOG**: oriented-gradient histograms per cell, a classic shape descriptor (Dalal & Triggs).
- **Cosine distance**, nearest-reference matching.
- **Explainable scoring**: additive contributions, bands, overrides.

---

## 5. Try it yourself

1. Run `findoc tamper train`, then `findoc synth --kind cheque --n 5` and `findoc tamper check`
   on a photo. Open one in an image editor, change one digit of the amount, save as JPEG, check
   again. Where is it warm?
2. Process your edited photo with `--save-dir outputs` and open `<name>_tamper.jpg`. Which
   field does `TAMPER_REGION` name, and what is the risk band?
3. `findoc process a.jpg --register`, then take a second photo of the same printed cheque and
   process it. What does the duplicate check report? What about a photo of a different cheque?
4. In `risk.py`, set `DATE_POSTDATED` to 20 points with `FINDOC_RISK_POINTS='{"DATE_POSTDATED": 20}'`.
   How does a post-dated cheque's band change? Why might a bank want that?
5. Train with `--samples 20` and compare the printed detection rate and AUC with the default
   80. What does that say about how much synthetic data the model needs?
6. Change `SIGNATURE_THRESHOLD` to 0.45 and re-run `experiments/phase6_fraud_eval.py`: which
   errors trade against which?

---

## 6. Viva and interview questions

**Q: Why not run the tamper detector on the clean flat page? It looks better.**
Dewarping interpolates every pixel at a varying scale. That is invisible to the eye but erases
the JPEG block statistics and noise the detector relies on. Running on the original and mapping
the heatmap keeps the evidence intact; our experiment shows it beats detecting on the flat page,
even with a model trained on flat pages (pixel AUC 0.95 vs. 0.86 on photos).

**Q: What is a JPEG ghost?**
Re-compressing an image at the quality it was last compressed with changes it very little. A
region with a different history (pasted, drawn) responds differently across qualities, so its
re-compression error curve stands out.

**Q: Why a learned model and not a rule?**
The rule flagged every handwritten field on clean cheques: ink naturally compresses differently
from paper. Whether an error level is suspicious depends on the content and on the phone; the
model learns that from labelled synthetic forgeries.

**Q: Isn't training only on synthetic forgeries a weakness?**
Yes, and it is stated. The manipulations are the standard ones in the forensics literature and
the captures vary camera quality, re-saving, blur and lighting; but real phones apply their own
processing (sharpening, HEIC, denoising) and real forgers use other tools. The next step is a
set of real cheque photos edited by people, the same way Phase 4 was checked on real videos.

**Q: How do you avoid false alarms costing reviewers time?**
Thresholds are set on held-out clean captures (0 of 40 flagged, then 0 of 40 in the test), the
check names the field and shows the heatmap, and the risk band rather than one check decides
what a reviewer sees.

**Q: How does duplicate detection work, and why not compare whole images?**
Whole images of one cheque differ (angle, light), and different cheques of one bank share the
template. The template-subtracted handwriting is unique to the leaf; SIFT matches verified by
one homography separate a re-photographed cheque (232+ inliers) from another cheque (at most 90).

**Q: Why did you drop the perceptual hash?**
It did not separate the classes on cheques (Hamming 2-8 for the same cheque, 4-10 for different
ones), so as a pre-filter it could only discard real duplicates.

**Q: How reliable is the signature check?**
On synthetic signers: random forgeries 100%, skilled 87%, with 13% of genuine signatures sent
to review. It compares shape only, not pen dynamics; a careful tracing can pass. It is one input
to the risk score, not a verdict.

**Q: How is the risk score computed, and why points rather than weights summing to 1?**
Sum of points x severity x confidence over fired checks, capped at 100, with bands and critical
overrides. Weights summing to 1 would let one proven fraud contribute only ~10 points.

**Q: What would you do next?**
Real edited photos for evaluation; forgeries applied to every video frame and re-encoded (H.264
needs other features than JPEG); more videos for the 3D mapping result; a learned signature
model if a consented signature dataset is available.
