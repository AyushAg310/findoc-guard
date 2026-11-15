# Phase 1 - Flat-photo pipeline: complete code explanation

Version **0.2.0** · Course outcome **CO-1** (image acquisition and processing), first use of
**CO-2** (deep-learning OCR).

Phase 1 turns a phone photo of a flat document into a clean, front-on scan and its text. It
adds five pipeline stages:

```
load -> quality -> detect -> rectify -> enhance -> ocr
         |           |         |          |          |
     blurry?     where is   make it    remove     read the
     too dark?   the page?  front-on   shadows    text
```

Everything here is classical computer vision except OCR. Each step is explained below with the
maths, then the code.

---

## 1. Big picture

A phone photo of a document has four problems:

| Problem | Example | Stage that fixes it |
| --- | --- | --- |
| The photo may be unusable | camera shake, too dark | `quality`: reject early with a reason |
| The page is only part of the frame | desk around the page | `detect`: find the 4 page corners |
| The page is seen at an angle | top edge shorter than bottom | `rectify`: homography to a front-on view |
| Lighting is uneven | shadow of a hand, lamp on one side | `enhance`: estimate and divide out the lighting |

Then `ocr` reads the cleaned page.

Why reject bad photos instead of trying anyway? Every later stage (field extraction, fraud
checks) gets worse silently on a bad input. A clear "retake" message at capture time is
cheaper than a wrong extraction later. This is requirement FR-CAP-03.

---

## 2. How a run flows

`findoc process data/samples/synthetic_shadow.jpg --save-dir outputs`:

1. `load`: reads the JPEG into a BGR NumPy array (`doc.original`), shape `(1200, 900, 3)`.
2. `quality`: computes sharpness, dark/bright fractions and glare on a 1000 px copy. If any
   blocking issue exists, sets `status = rejected_quality`, and the pipeline stops (Phase 0's
   early-stop rule).
3. `detect`: finds the page's 4 corners, stores them in `doc.corners`, the method used in
   `doc.detection_method`, and the page/frame area ratio in `doc.quality["coverage"]`. Rejects
   if the page is too small (< 25% of the frame; 40% before v0.3.0).
4. `rectify`: computes the homography from those corners to a rectangle and warps the photo
   into `doc.flat`. Stores the 3x3 matrix in `doc.homography`.
5. `enhance`: removes uneven lighting from `doc.flat` into `doc.enhanced`.
6. `ocr`: if EasyOCR is installed, reads `doc.enhanced` into `doc.words` and `doc.full_text`;
   otherwise records "OCR skipped" and continues.
7. The CLI prints the stage table, quality metrics, messages and text, saves
   `*_detected.jpg`, `*_flat.png`, `*_enhanced.png`, and writes the JSON.

---

## 3. File-by-file walkthrough

### 3.1 `findoc/preprocess/quality.py`: is this photo usable?

**Resolution independence.** `resize_longest(img, 1000)` shrinks the photo so its longest side
is 1000 px before measuring. A 12-megapixel photo and a 2-megapixel photo of the same scene
then give similar numbers, so one threshold works for every phone.

**Sharpness: variance of the Laplacian.**

The Laplacian is the sum of second derivatives:

```latex
\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}
```

OpenCV approximates it with the 3x3 kernel `[[0,1,0],[1,-4,1],[0,1,0]]`. On a flat region the
response is ~0; on an edge it swings strongly positive and negative. A sharp photo of text has
many crisp edges, so the response has a **large variance**. Blur spreads each edge over many
pixels, the second derivative becomes small everywhere, and the variance collapses.

```python
def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
```

`CV_64F` (64-bit float output) matters: with 8-bit output, negative responses would be clipped
to 0 and half the edge information lost.

Measured on the synthetic set: sharp photos score about 700-1000, a 1.5 px blur lands around
the threshold, and a 4 px blur scores about 4. The default threshold is 100
(`FINDOC_BLUR_THRESHOLD`).

**Exposure.** `dark = mean(gray < 20)`, `bright = mean(gray > 250)`: the fraction of pixels
that are nearly black or clipped white. More than 50% of either is blocking.

**Glare.** Specular highlights are pixels saturated in **all three** channels (>= 250) with
**low colour saturation** (HSV S < 30). This is only a *warning*: a white scanned page can
legitimately contain pure-white pixels, so rejecting on it would cause false rejections.

**Score.** A 0-1 summary for the UI: 60% sharpness (capped at twice the threshold), 40%
exposure, minus a glare penalty. Decisions use the explicit issues, not the score.

**`QualityReport`** is a `dataclass` (a lightweight class for holding data) with an
`acceptable` property (no blocking issues) and `as_dict()` for the JSON output.

### 3.2 `findoc/preprocess/detect.py`: where is the page?

Detection runs on an 800 px copy (fast), then scales corners back to full resolution.

**Step 1: edges (`_auto_canny`).** Canny edge detection:
1. Gaussian smoothing (done before, with a 5x5 kernel) to suppress noise.
2. Sobel gradients to get edge strength and direction.
3. Non-maximum suppression: keep only the ridge of each edge, 1 px wide.
4. Hysteresis: pixels above the upper threshold are edges; pixels between the thresholds are
   edges only if connected to a strong edge.

Canny needs two thresholds. `_auto_canny` sets them to 0.67x and 1.33x the image's median
intensity, so bright and dark photos both work without tuning.

**Step 2: close the gaps.** `morphologyEx(..., MORPH_CLOSE)` is dilation followed by erosion:
it joins edge fragments broken by noise or text touching the page border, so the page outline
becomes one closed contour.

**Step 3: contours and quadrilateral fitting (`_best_quad`).**
- `findContours(RETR_EXTERNAL)` traces the outer boundary of each connected edge region.
- The 5 largest contours above 15% of the frame are tried, biggest first.
- `approxPolyDP` is the **Douglas-Peucker** algorithm: it simplifies a curve by recursively
  keeping only points further than `epsilon` from the straight line between endpoints. With
  `epsilon` = 2-5% of the perimeter, a slightly noisy page outline collapses to exactly 4
  points.
- The result must have 4 points and be **convex** (`isContourConvex`); a page seen in
  perspective is always a convex quadrilateral.

**Fallbacks.**
- *threshold*: Otsu's method picks the gray level that best separates two populations of
  pixels (it maximises the between-class variance). The bright page becomes white, the darker
  desk black, and the same quadrilateral fitting runs on that mask. `MORPH_OPEN` (erosion then
  dilation) first removes small specks.
- *min_area_rect*: if no clean 4-point shape exists (a folded corner), the smallest rotated
  rectangle around the biggest blob.
- *full_frame*: nothing found, or the "page" covers more than 97% of the image (a flatbed scan
  or screenshot): use the whole image.

**`order_corners`.** Algorithms return corners in arbitrary order, but the homography needs a
fixed order (top-left, top-right, bottom-right, bottom-left). Trick: top-left has the smallest
`x + y`, bottom-right the largest; top-right has the smallest `y - x`, bottom-left the largest.

**Coverage.** `contourArea(corners) / (h * w)`. In `stages.py`, a page covering less than
`FINDOC_MIN_COVERAGE` (25% since v0.3.0, 40% in v0.2.0) is rejected with "Move closer", because the page would have too few
pixels for reliable OCR.

`draw_detection` draws the quadrilateral and labelled corners on a **copy** (never modify
inputs in place: other stages still need the original).

### 3.3 `findoc/preprocess/rectify.py`: make it front-on

**The homography.** Any flat surface photographed by a pinhole camera is related to its
front-on view by a projective transformation, a 3x3 matrix H with 8 degrees of freedom (9
entries, defined up to scale):

```latex
\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = H \begin{bmatrix} x \\ y \\ 1 \end{bmatrix},
\qquad (u, v) = \left(\frac{x'}{w'}, \frac{y'}{w'}\right)
```

Each point correspondence gives 2 equations, so 4 corners give exactly the 8 needed.
`cv2.getPerspectiveTransform(src, dst)` solves that 8x8 linear system. (With more than 4
noisy points you'd use `findHomography` with RANSAC; Phase 2 does that for cheque templates.)

**Output size.** `output_size` uses the longer of each pair of opposite edges as the flat
page's width and height, so no detail is shrunk away.

**Warping.** `warpPerspective` computes, for every output pixel, where it came from in the
photo (using H inverted) and samples there. `INTER_CUBIC` interpolation (4x4 neighbourhood)
keeps text edges sharper than bilinear. `BORDER_REPLICATE` avoids black borders if a corner
lands slightly outside the image.

**Why keep H?** `map_points(points, H)` moves any coordinates between photo and flat page.
Phase 2 maps field boxes back onto the original photo for display; Phase 5 maps tamper
evidence. Phase 4's 3D correspondence map is the generalisation of this matrix to curved pages.

### 3.4 `findoc/preprocess/illumination.py`: remove shadows

**Model.** What the camera sees is roughly *reflectance x illumination*. Paper has constant
high reflectance, ink low reflectance; illumination varies slowly across the page. If we can
estimate the illumination (what the blank paper would look like), dividing by it leaves only
reflectance:

```latex
I_{\text{out}}(x, y) = \frac{I(x, y)}{B(x, y)} \times 255
```

**Estimating the background B** (`estimate_background`):
1. `dilate` with a kernel slightly wider than a pen stroke (image side / 150). Dilation takes
   the **maximum** in a neighbourhood, so bright paper spreads over thin dark text and the
   text disappears.
2. `medianBlur` with a large kernel (side / 30): replaces each pixel by the median of its
   neighbourhood, smoothing what is left into a slowly varying lighting map. The median, unlike
   a mean, ignores leftover dark specks.

Each colour channel is processed separately, then merged. Result on the synthetic set: the
background's brightness standard deviation drops from 24 to 0.7 (97% more even), while text
stays dark (checked in `tests/test_preprocess.py`).

`_odd(n)`: median and morphology kernels must have odd sizes so they have a centre pixel.

### 3.5 `findoc/preprocess/stages.py`: wiring into the pipeline

Four functions with the Phase 0 stage signature `(Document) -> Document`:

| Stage | Reads | Writes | Can reject? |
| --- | --- | --- | --- |
| `quality_stage` | `original` | `quality` metrics, warnings/issues in `messages` | Yes: blurry, dark, overexposed |
| `detect_stage` | `original` | `corners`, `detection_method`, `quality["coverage"]` | Yes: page too small |
| `rectify_stage` | `original`, `corners` | `flat`, `homography` | No |
| `enhance_stage` | `flat` | `enhanced` | No |

Every stage first checks its input exists (`_require_image`): a video input (Phase 3) has no
`original` yet, so these stages pass it through untouched instead of crashing.

Thresholds come from `get_settings()`, so they can be changed with `FINDOC_*` variables.

### 3.6 `findoc/recognize/ocr.py`: read the text

**Engine interface.** `OcrEngine` is a `typing.Protocol`: any class with a
`read(img) -> list[OcrWord]` method counts as an engine (structural typing, "duck typing with
type checking"). The pipeline never imports EasyOCR directly, so swapping in PaddleOCR or
TrOCR later is a one-class change.

**EasyOCR** works in two deep-learning steps:
1. **CRAFT** (Character Region Awareness for Text detection): a CNN that predicts, per pixel,
   how likely it is to be the centre of a character and the gap between characters; grouping
   these gives word boxes.
2. **CRNN** recogniser: CNN features read left to right by a bidirectional LSTM, trained with
   **CTC loss** (Connectionist Temporal Classification), which lets the network output a
   character sequence without knowing where each character starts and ends.

**Optional dependency.** EasyOCR needs PyTorch (~2 GB), so it's in the `[ocr]` extra. If it's
missing, `EasyOcrEngine.__init__` raises `OcrUnavailableError` with the install command, and
`ocr_stage` records "OCR skipped" and returns normally: the rest of the pipeline is still
useful. `raise ... from exc` keeps the original ImportError attached for debugging.

**GPU.** `_use_gpu` returns True when `FINDOC_DEVICE` isn't `cpu` and PyTorch sees a CUDA
device. On your NVIDIA laptop that's automatic once the CUDA build of PyTorch is installed.

**`@lru_cache(maxsize=1)` on `get_engine`.** Loading the OCR models takes seconds; the cache
creates the engine once per process and reuses it (essential in the Streamlit demo and the
Phase 7 API).

**`words_to_text`.** OCR returns boxes in arbitrary order. To produce readable text: sort boxes
by vertical centre; start a new line when a box's centre is more than 0.6 x its height away
from the previous box's centre; sort each line left to right; join.

### 3.7 `findoc/synth/photo.py`: synthetic photos with ground truth

You can't measure detection accuracy without knowing the true corners, and labelling real
photos by hand is slow. So we **simulate** the photo:

1. `render_page`: draws an invoice-like page (header box, random text lines, a 6x4 table, a
   total) with OpenCV's Hershey fonts. Random content comes from `rng` (a seeded
   `numpy.random.Generator`), so the same seed always gives the same page: experiments are
   reproducible.
2. `_random_quad`: picks where the page lands in the photo: size (45-75% of the frame), small
   rotation (±12°), offset, and per-corner jitter (±4%) that creates perspective.
3. `photograph`: warps the page onto a noisy coloured "desk" with
   `getPerspectiveTransform` (the inverse of what `rectify` does), blends with a warped mask,
   multiplies by a lighting gradient and an optional Gaussian shadow, adds sensor noise,
   optional blur (camera shake) and JPEG compression.
4. `generate`: writes N photos and `labels.json` with ground-truth corners for each.

`findoc synth --n 100` exposes it from the CLI.

### 3.8 `findoc/cli.py`: new options and command

- `--save-dir/-s`: `save_artifacts` writes `<name>_detected.jpg` (overlay), `_flat.png`,
  `_enhanced.png`. PNG for the processed pages (lossless, keeps text crisp for later stages).
- `--ocr/--no-ocr`: a Typer boolean flag pair; `--no-ocr` skips the OCR stage entirely.
- Output now shows quality metrics, a coloured status, and recognised text.
- `findoc synth`: generate synthetic photos.
- A quality rejection exits with code **0**: rejecting a bad photo is a correct outcome, not a
  program failure. Only `failed` (an exception) exits with 1.

### 3.9 `findoc/types.py` and `findoc/pipeline.py`: what changed

`Document` gained `enhanced` (image, excluded from JSON), `corners`, `homography`,
`detection_method`, `words` (list of the new `OcrWord` model) and `full_text`. Images are
stored as NumPy arrays but corners and matrices as plain lists, so they serialise to JSON.

`default_pipeline(ocr=True)` now registers the five Phase 1 stages. The stage imports happen
**inside** the function, so `findoc --help` starts instantly (no OpenCV-heavy imports) and no
circular import can arise between `pipeline.py` and the stage modules.

### 3.10 `demo/streamlit_app.py`: the demo UI

Streamlit turns a Python script into a web page; the script re-runs top to bottom on every
interaction. The sidebar takes an upload or generates a synthetic photo (seed and blur
sliders). The app runs `process()`, shows a coloured status, the three images side by side
(converted BGR to RGB, since OpenCV and browsers disagree on channel order), quality metrics,
a stage-timing bar chart, the recognised text, and the full JSON.

Run: `pip install -e ".[ui]"` then `streamlit run demo/streamlit_app.py`.

### 3.11 `experiments/phase1_detection_eval.py`: measuring the pipeline

Generates 200 photos with known corners and reports:

| Metric | Result | Meaning |
| --- | --- | --- |
| Detection success (error < 2% of diagonal) | 100% | page found correctly every time |
| Mean corner error | 0.10% of diagonal | about 2 px on a 1200x1600 photo |
| Acceptance at blur sigma 0 / 1 / 1.5 / 2 / 3 / 5 | 100 / 100 / 18 / 0 / 0 / 0 % | sharp cut-off around 1.5 px |
| Background unevenness | 24.1 -> 0.7 | 97% more even lighting |
| Median latency (CPU) | quality 24, detect 4, rectify 16, enhance 96 ms | real-time capable |

Corner error is normalised by the image diagonal so results compare across resolutions.

**Honest limitation**: synthetic photos have clean page edges and a contrasting background.
Real photos (white page on a white table, cluttered desk, fingers on the page) are harder.
Phase 8 adds results on real captures; mention this in the term paper.

### 3.12 Tests

| File | What it checks |
| --- | --- |
| `tests/conftest.py` | Fixtures: synthetic photo + ground truth, a blurry photo, and an **autouse** fixture that disables the real OCR model in every test (fast, no 2 GB download in CI) |
| `tests/test_preprocess.py` | Sharp passes / blurred fails; dark and overexposed fail; glare only warns; corner ordering; detection within 1% of ground truth across seeds; threshold and full-frame fallbacks; rectified aspect ratio and exact corner mapping; illumination evenness and that text stays dark |
| `tests/test_ocr.py` | Reading-order assembly; OCR skipped cleanly when not installed; a **fake engine** injected with `monkeypatch` to test the stage without a model; clear error message when EasyOCR is missing |
| `tests/test_synth.py` | Page content, determinism (same seed, same photo), corners inside the frame, files and labels written |
| `tests/test_pipeline.py` | Stage order; end-to-end success; blurry photo stops after `quality`; small page rejected; video passes through untouched |
| `tests/test_cli.py` | JSON + saved images; `--no-ocr`; rejection exits 0; `synth` command |

42 tests, 96% coverage. `monkeypatch` (a pytest fixture) temporarily replaces a function or
attribute for one test and restores it afterwards: here, `ocr.get_engine`.

---

## 4. Concepts used in this phase

| Concept | Explanation |
| --- | --- |
| Grayscale conversion | `0.299 R + 0.587 G + 0.114 B` (weights follow human brightness sensitivity). Most measurements need only intensity. |
| Convolution / filtering | Sliding a small kernel over the image and summing weighted neighbours: blur (Gaussian), edges (Sobel), second derivative (Laplacian). |
| Morphology | Dilation = max over a neighbourhood (grows bright regions), erosion = min (shrinks them). Closing = dilate then erode (fills gaps); opening = erode then dilate (removes specks). |
| Canny | Smoothing, gradients, non-maximum suppression, hysteresis thresholding: thin, connected edges. |
| Contours | Boundaries of connected regions in a binary image (`findContours`). |
| Douglas-Peucker | Polyline simplification by recursive farthest-point splitting (`approxPolyDP`). |
| Otsu thresholding | Automatic threshold maximising the variance between foreground and background classes. |
| Homography | 3x3 projective transform between two views of a plane; 4 point pairs determine it. |
| Interpolation | Estimating pixel values at non-integer positions: nearest, bilinear (2x2), bicubic (4x4). |
| Median filter | Replace each pixel by its neighbourhood median: removes outliers, keeps edges. |
| Synthetic data with ground truth | Simulating the capture process so accuracy can be measured exactly and reproducibly. |
| Test doubles | Fake objects (the fake OCR engine) that stand in for slow or external dependencies. |

---

## 5. Try it yourself

1. `findoc process data/samples/synthetic_tilted.jpg -s outputs` and open the three images.
   Then try `synthetic_blurry.jpg`. What does the CLI say?
2. Photograph a printed page on your desk with your phone, copy it into `data/raw/`, and run
   the CLI on it. Which detection method was used? Try a white page on a white table.
3. In PowerShell set `$env:FINDOC_BLUR_THRESHOLD="2"` and re-run the blurry sample (sharpness 3). Why does
   it pass now? Remove the variable afterwards (`Remove-Item Env:FINDOC_BLUR_THRESHOLD`).
4. `findoc synth --n 5 --blur 1.5 --out data/synthetic/blur15` then process each file. Compare
   their sharpness values with the 18% acceptance figure in the evaluation.
5. Install OCR (`pip install -e ".[ocr]"`, ideally after the CUDA build of PyTorch) and run
   `streamlit run demo/streamlit_app.py`. Compare OCR confidence on the flat vs. enhanced page
   by temporarily changing `ocr_stage` to read `doc.flat`.
6. Run `python experiments/phase1_detection_eval.py --n 50` and check you get similar numbers.

---

## 6. Viva and interview questions

**Q: How do you detect blur, and why does it work?**
A: Variance of the Laplacian on a resolution-normalised grayscale image. The Laplacian
responds to edges; blur weakens edges so the response variance drops. Threshold 100: sharp
photos score 700+, 1.5 px blur is borderline, 2 px is always rejected.

**Q: Why not just run OCR on the original photo?**
A: OCR accuracy drops with perspective distortion (letters are skewed and differently sized
across the page) and uneven lighting (text in shadows has low contrast). Rectification and
illumination normalisation remove both. Phase 8 quantifies the gain with OCR error rates.

**Q: Explain the homography and why 4 points are enough.**
A: A plane seen by a pinhole camera maps to the image by a 3x3 projective matrix with 8 degrees
of freedom. Each correspondence gives two equations, so 4 corners determine it exactly.

**Q: What if the page is curved or folded?**
A: A homography only models flat planes, so a folded page stays distorted. That is precisely the
motivation for Phase 4, which reconstructs the 3D surface from video and unrolls it.

**Q: How did you evaluate detection without labelled data?**
A: A synthetic generator renders pages and photographs them with known perspective, lighting,
noise and blur, recording the true corners. On 200 photos: 100% success and 0.10% of the
diagonal mean corner error. Real photos are harder, which we state as a limitation.

**Q: How does shadow removal work?**
A: Estimate the paper's background brightness by dilating (erasing thin dark text) and median
filtering (smoothing), then divide the image by it. This follows a reflectance x illumination
model; it reduced background unevenness by 97%.

**Q: Why is the OCR engine behind an interface?**
A: So the pipeline doesn't depend on a specific library. EasyOCR can be replaced with PaddleOCR
or a fine-tuned TrOCR for handwriting (Phase 2) by writing one class; tests use a fake engine.

**Q: Why does a rejected photo exit with code 0?**
A: Rejection is a valid business outcome (ask the user to retake), not a crash. Exit code 1 is
reserved for unexpected failures so scripts and CI can tell them apart.
