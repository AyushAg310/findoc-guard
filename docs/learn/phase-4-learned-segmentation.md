# Phase 4 - Learned page detection: complete code explanation

Version **0.5.0** (patch 0.5.1) · Course outcomes **CO-2** (deep learning: a convolutional segmentation
network), **CO-1** (edge-based refinement of its output), and support for **CO-4/CO-5** (tracking
and 3D both start from the page outline).

## Why this phase exists

This phase was not in the original plan. It was added because of three real phone videos:

| Video | What happened | Root cause |
| --- | --- | --- |
| Blank notebook page, larger than the frame | Rejected (correctly) | Nothing to track; page never fully in view |
| Real cheque, portrait, on red patterned cloth | Tracked half the cheque | Page larger than the frame; classical detector closed the outline on a shading line |
| Real cheque, landscape, on a pale wooden board | Rejected: "never fits in the frame" | Classical detector outlined the **whole board**: white paper on pale wood has almost no brightness contrast |

The third video was filmed correctly, so the system was at fault. The Phase 1 detector finds a
page with **hand-written rules** (Canny edges, Otsu brightness threshold). Rules about brightness
break when the background is as bright as paper, and rules about edges break when the
background has strong edges of its own (patterns, posters, other papers). As you put it: it
should not matter where the cheque is kept.

The fix is to **learn** what a document looks like instead of describing it with rules: a small
neural network labels every pixel "document" or "background". Trained on thousands of
synthetic images with deliberately hard backgrounds, it learns cues no single rule captures
(paper texture, printed content, straight edges, shape) together.

```
frame -> [U-Net, 256x256] -> P(document) per pixel -> largest region -> 4-corner polygon
      -> refine each side on the real image edges -> corners
      -> if mask and polygon agree (IoU >= 0.9): use them; else classical detector (Phase 1)
```

Everything downstream (tracking, keyframes, cheque templates) is unchanged: it simply receives
better corners.

---

## 1. Big picture

| Step | Technique | File |
| --- | --- | --- |
| Hard backgrounds | Procedural pale wood, near-white tables, bright boards, patterned cloth, white-text posters, clutter, noise | `findoc/synth/backgrounds.py` |
| Training samples | Random document in perspective on a hard background, thumb occlusion, lighting, blur, noise, JPEG; exact mask from the quadrilateral | `findoc/synth/segdata.py` |
| Network | Small U-Net (1.2 M parameters, 256x256 input) | `findoc/segment/model.py` |
| Training | On-the-fly synthetic data, BCE + Dice loss, AdamW, one-cycle learning rate, validation IoU, ONNX export | `findoc/segment/train.py` |
| Inference | ONNX model run by OpenCV DNN (no PyTorch at runtime); mask -> quadrilateral; sub-pixel edge refinement; agreement score | `findoc/segment/infer.py` |
| Integration | Learned first, classical fallback; settings `FINDOC_SEGMENTATION`, `FINDOC_SEGMENTER_MODEL` | `findoc/preprocess/detect.py`, `config.py` |
| Commands | `findoc segment train`, `findoc segment check` | `findoc/cli.py` |
| Measure | Learned vs. classical vs. learned-without-refinement on 320 hard synthetic images; real videos | `experiments/phase4_segmentation_eval.py` |

---

## 2. How a run flows

**Training** (once, on the GPU laptop): `findoc segment train` builds a pool of rendered
documents, then for every training step generates fresh images on the fly (new background,
pose, lighting each time), trains the U-Net, measures IoU on a fixed validation set made from
*different* documents, and writes the best model to `models/docseg.onnx` plus a
`models/docseg.json` with the training history. `models/` is git-ignored: weights are never
committed.

**Using it**: nothing to change. `find_document` (used by photos, video tracking start,
drift correction and re-localisation) now:

1. loads the model once (`load_segmenter`, cached) if `FINDOC_SEGMENTATION` is `auto` (default)
   and the file exists;
2. runs it (`Segmenter.segment`) and accepts the outline if the mask agrees with the fitted
   quadrilateral (IoU at least 0.9) and it covers 4-97% of the image; `doc.detection_method`
   becomes `learned`;
3. otherwise falls back to the Phase 1 classical detector, exactly as before.

Without a model file, behaviour is identical to v0.4.2. Set `FINDOC_SEGMENTATION=off` to force
the classical detector (for comparisons, or if the model misbehaves on some input).

`findoc segment check photo.jpg` writes `outputs/segment_check.jpg`: the photo with the
probability map as a heat map and the fitted outline, to see what the network sees.

---

## 3. File-by-file walkthrough

### 3.1 `findoc/synth/backgrounds.py`: hard backgrounds

Each generator draws one kind of surface at any size, deterministically from the random
generator:

- `desk`: ordinary wood-grain desk (the Phase 3 kind).
- `pale_wood`: the same grain but beige and bright (150-215 grey levels, paper is 200-250).
- `light_table`: near-white surfaces with a gentle gradient: white desks, bed sheets.
- `board`: a bright board covering most of the frame (sometimes running off it) on a dark,
  textured surround. This reproduces the third real video exactly: the board is the biggest
  bright quadrilateral in view, so a brightness rule picks it instead of the document.
- `cloth`: saturated fabric with a repeating pattern of ellipses and dots (second real video).
- `poster`: dark graphic with large white letters and strokes, which touch and merge with paper
  edges (seen behind the cheque in the second and third videos).
- `clutter`: other rectangles (papers, phones, books) partly in view, bright or dark.
- `noise`: multi-scale random texture, to teach "not a document" for arbitrary content.

### 3.2 `findoc/synth/segdata.py`: training samples

- `document_pool`: renders documents once (synthetic cheques from Phase 2, invoice-like pages
  from Phase 1, and `plain_sheet`s with a few printed lines or none), tinted white, cream,
  pink, blue or green. Re-rendering a cheque for every sample would make data generation the
  bottleneck; warping a stored one costs a few milliseconds.
- `_quad`: a random page outline: 15-75% of the frame, rotated by up to 25 degrees (20% of the
  time by 65-115 degrees, i.e. sideways), with perspective jitter, scaled so the *rotated*
  shape fits inside the frame. (A first version scaled before rotating, so sideways pages could
  stick out of the frame; a test caught it.)
- `_partial_quad`: 20% of samples show a document that **runs off the frame** (bigger than the
  frame and/or pushed past one edge); its mask then reaches the image border. The first trained
  model never saw this: on the two real videos where the cheque was cut off by the frame, it
  outlined the visible part as if it were a whole cheque, its mask stopping a few pixels short
  of the border, so the "clear of the border" test let it through (on synthetic cut-off pages it
  claimed "whole document" 30% of the time). Teaching the network to mark paper right up to the
  image edge lets the pipeline tell "cut off" from "whole".
- `make_sample`: landscape or portrait frame; background; document warped in with
  `warpPerspective`; 25% of samples get a thumb near a corner; then lighting gradient, soft
  shadow, white-balance shift, sensor noise, optional blur and JPEG compression at quality
  55-95. The **label is the full quadrilateral**, also under the thumb: what the pipeline needs
  is the page outline, and the network learns to complete it.

### 3.3 `findoc/segment/model.py`: the U-Net

A **convolutional neural network** applies learned 3x3 filters layer after layer; early layers
respond to edges and textures, deeper layers to larger patterns. For segmentation we need an
answer for every pixel, so the network must both understand the whole image and keep exact
positions. A **U-Net** does this with:

- an **encoder**: 4 levels of two 3x3 convolutions (each followed by batch normalisation and
  ReLU), then 2x2 max-pooling, so resolution halves (256, 128, 64, 32, 16 px) while channels
  grow (16, 32, 64, 128): more abstract features over a larger area;
- a **bottleneck** at 16x16, where each feature sees most of the image ("is this region the
  document?");
- a **decoder**: 4 levels of bilinear upsampling, **concatenation with the encoder features of
  the same size** (skip connections), and two convolutions. The skips bring back the fine
  spatial detail (exactly where the paper edge is) that pooling threw away;
- a 1x1 convolution head giving one **logit** per pixel; sigmoid(logit) = P(document).

Sizes: 1.2 M parameters (EasyOCR's models are ~10x larger). Bilinear upsampling plus
convolution avoids the checkerboard artefacts of transposed convolutions and exports cleanly to
ONNX.

**Batch normalisation** rescales each channel to zero mean and unit variance over the batch,
which makes training faster and less sensitive to the learning rate. **ReLU** (max(0, x)) is the
non-linearity that lets stacked layers represent more than a linear filter.

### 3.4 `findoc/segment/train.py`: training

- `SyntheticDocs`: a PyTorch `Dataset` that generates sample *i* of epoch *e* from the random
  seed (seed, e, i), so every epoch is new data and every run is reproducible. Images are
  resized to 256x256 and normalised to [-0.5, 0.5] (`to_input`, identical to inference).
- **Loss = binary cross-entropy + Dice**. BCE scores each pixel separately; on its own it is
  dominated by the (usually larger) background. Dice loss, 1 - 2|P∩T| / (|P| + |T|), scores the
  overlap of the whole shape, so a small document matters as much as a large one.
- **AdamW** (Adam with decoupled weight decay) and a **one-cycle** learning-rate schedule
  (warm up, then anneal): fast and stable for small networks.
- **Validation** on a fixed set built from a *different* document pool (unseen cheques and
  pages) reports mean IoU and its 5th percentile (the hard cases). The best epoch is exported.
- `export_onnx`: **ONNX** is an open format for trained networks. The legacy TorchScript
  exporter (opset 17) is used because its output is the most compatible with OpenCV's DNN
  module.

### 3.5 `findoc/segment/infer.py`: from pixels to corners

- `Segmenter.probability`: `cv2.dnn.readNetFromONNX` + `forward`: the same numbers PyTorch
  gives (tested to 1e-3; measured 6e-8), with only OpenCV installed.
- `mask_to_quad`: threshold at 0.5, keep the largest connected component (other papers in view
  are smaller), take its convex hull, and simplify with Douglas-Peucker (`approxPolyDP`) at
  increasing tolerance until it has 4 vertices; else the minimum-area rectangle.
- `refine_quad`: the network works at 256 px, so on a 1280 px frame one mask pixel is 5 image
  pixels. Each side is refined on the real image: at 40 points along the middle 80% of the side,
  look across it (±6 px) for the strongest brightness step (Sobel gradient projected on the
  side's normal), fit a line to those points with **Huber** loss (`cv2.fitLine(DIST_HUBER)`,
  robust to a thumb or text near the edge), and intersect adjacent lines for the corners. A
  corner moves at most 15 px, and only where both of its sides were found. Sides lying along
  the image border are not refined: the page continues past the edge there, and the nearest
  strong edge would be a printed line *inside* the page, which would make a cut-off page look
  whole (measured: refinement had raised false "whole" outlines from 20% to 30%).
- `solid_region`: the largest connected region of the thresholded mask with its holes filled.
  On the real pale-wood video the network left a hole where the cheque has grey shading; a
  document is solid, so holes are filled before fitting and before the agreement score.
- **Agreement** = IoU between the mask and the final quadrilateral. A real page gives a
  near-quadrilateral mask (around 0.95+); two merged papers or a failed prediction do not, and
  `find_document` then falls back to the classical detector. The network's own confidence is
  therefore checked by geometry.
- `load_segmenter`: cached loader; returns None for a missing or unreadable file, so the app
  works without a model.

### 3.6 `findoc/preprocess/detect.py`, `config.py`, `cli.py`: integration

- `find_document` = learned (if available and agreeing) else `find_document_classical` (the
  Phase 1 code, unchanged). New settings `segmentation` ("auto"/"off") and `segmenter_model`.
- Tests always run with `FINDOC_SEGMENTATION=off` (an autouse fixture in `tests/conftest.py`), so
  results don't depend on whether a trained model happens to exist on the machine.
- CLI: `findoc segment train` (needs `pip install -e ".[train]"`, i.e. PyTorch and onnx) and
  `findoc segment check`.

### 3.7 `findoc/track/klt.py`: the border margin now scales

"Clear of the image border" (used to decide whether the whole document is in view) required
4 px. That suited the classical detector, whose outlines are clipped *exactly* at the border. The
learned detector predicts at 256 px, so on a 960 px frame one mask cell is ~4 px, and its outline
of a page running off the frame can stop a cell or two short of the edge: on the real sideways
cheque video the bottom edge sat 6-7 px from the border and tracking started on a partial cheque.
The margin is now 1% of the image's longer side (at least 4 px).

### 3.8 Results (`experiments/phase4_segmentation_eval.py`)

320 synthetic images (40 per background kind, 1280x720, documents never used in training) for
accuracy, plus 160 where the document runs off the frame. Model: `findoc segment train
--workers 8 --epochs 10` on the RTX 3050 laptop (20,000 fresh images per epoch, 55 min,
validation IoU 0.993, 5th percentile 0.988):

| Detector | Page found | Corner error | Cut-off page called whole | Time per image (laptop CPU) |
| --- | --- | --- | --- | --- |
| Classical (Phase 1) | 45.0% | 0.556% | 0.0% | 4.0 ms |
| Learned | 95.9% | 0.136% | 0.0% | 33.5 ms |
| Learned, no edge refinement | 95.9% | 0.262% | 0.0% | 27.8 ms |

| Background | Classical | Learned |
| --- | --- | --- |
| desk | 98% | 100% |
| pale_wood | 57% | 80% |
| light_table | 5% | 100% |
| board | 8% | 88% |
| cloth | 68% | 100% |
| poster | 25% | 100% |
| clutter | 48% | 100% |
| noise | 52% | 100% |

Real videos (no ground truth):

| Video | Classical | Learned | Correct outcome |
| --- | --- | --- | --- |
| Whole cheque, landscape, pale wooden board | rejected | ok: 102/137 usable frames, 7 keyframes, flat page 2.28:1 (a CTS cheque is 2.21:1) | ok |
| Sideways cheque running off the frame | rejected | rejected ("never fits") | rejected |
| Portrait cheque on red cloth, running off the frame | rejected | rejected | rejected |

**What the numbers say:**

1. The learned detector more than doubles the success rate on hard backgrounds (45% -> 96%) and
   is 4x more precise (0.56% -> 0.14% of the diagonal, about 2 px on a 1280x720 frame).
   Near-white tables go from 5% to 100%.
2. Edge refinement halves the corner error (0.26% -> 0.14%) without changing the success rate:
   the network finds the page, the classical edge fit places its sides precisely. Neither part
   alone is as good: a hybrid of learned and classical vision.
3. The weakest cases are pale wood (80%) and bright boards (88%): paper on a surface almost as
   light as paper. These are the next things to improve (more such samples, a larger input
   size).
4. It is about 8x slower than the classical detector (33 ms vs 4 ms on CPU), but it only runs
   when the tracker needs a detection (start, drift checks, re-localisation), not every frame:
   the whole tracked video still ran at 8.5 ms per frame.
5. Training mattered: the CPU development model (2,400 images, 7 epochs, with a data-replay bug)
   reached 90% success and 0.38% error; the GPU model with 20,000 fresh images per epoch reached
   96% and 0.14%.

**Why training took long.** Each training image is rendered on the CPU (background, warp,
lighting, JPEG: ~50-70 ms). With 2 worker processes an epoch of 20,000 images took ~12 minutes
while the GPU mostly waited; with 8 workers ~5-6 minutes. Since v0.5.1 the default is "CPU cores
minus 2, at most 8" (`default_workers`). This is a common situation in deep learning: the data
pipeline, not the model, is the bottleneck.

### 3.9 Tests (`tests/test_segment.py`)

Backgrounds of every kind and their determinism; sample masks match their corners and stay in
the frame; `mask_to_quad` on a perspective quad, an empty mask and a circle; `refine_quad` moves
corners that were off by 4-5 px to within 1.5 px and leaves them alone where there are no
edges; `find_document` prefers a confident learned outline and falls back when agreement is low
or nothing is found (with a fake segmenter, so no model is needed); missing or corrupt model
files; `segment check` without a model. With PyTorch installed (skipped in CI): the ONNX export
run by OpenCV equals PyTorch, and a one-epoch training run writes the model and its metadata.

---

## 4. Concepts used in this phase

| Concept | In one paragraph |
| --- | --- |
| Semantic segmentation | Classify every pixel (here: document or not), instead of the whole image. |
| Convolution | A small learned filter slid over the image; stacking them builds features from edges up to objects. |
| Encoder-decoder, U-Net | Downsample to understand context, upsample to localise; skip connections carry fine detail across. |
| Receptive field | The image area that influences one output; pooling grows it so a pixel "knows" whether it is inside a page. |
| Batch normalisation, ReLU | Normalise activations per channel for stable training; ReLU provides non-linearity. |
| BCE and Dice loss | Per-pixel log loss plus a whole-shape overlap loss that is insensitive to class imbalance. |
| IoU | Intersection over union of predicted and true regions; 1 = perfect. |
| Synthetic training data and domain randomisation | Generate unlimited labelled images; vary everything irrelevant (background, light, colour, blur) so the network learns only what is relevant, which helps transfer to real photos. |
| Train / validation split | Validate on documents never used in training, to detect memorisation. |
| ONNX, OpenCV DNN | Train in one framework, run anywhere; keeps PyTorch out of the deployed app. |
| Douglas-Peucker | Simplify a polygon by dropping points closer than a tolerance to the simplified line. |
| Robust line fitting (Huber) | Squared error for small residuals, linear for large ones, so outliers pull less. |
| Learned + classical hybrid | The network proposes, geometry checks (agreement) and refines (edges), and the old method remains a fallback. |

---

## 5. Try it yourself

1. Train on the GPU laptop: `findoc segment train` (about 55 min with the default workers), then
   `findoc segment check` on a photo from your phone.
2. Look at `models/docseg.json`: how did validation IoU change per epoch? Where did it level off?
3. Generate training samples and look at them (`make_sample` with each background kind). Which
   ones would the classical detector get wrong?
4. Run `python experiments/phase4_segmentation_eval.py --video myvideo3.mp4` and compare the
   classical and learned rows for your pale-wood video.
5. The evaluation's "learned, no edge refinement" row runs the same model without
   `refine_quad`. How much does refinement change the corner error? Why does it not change
   the success rate much?
6. Set `FINDOC_SEGMENTATION=off` and process the same video: the old behaviour.

---

## 6. Viva and interview questions

**Q: Why add a learned detector? Wasn't the classical one 100% on the Phase 1 test set?**
A: On synthetic photos with dark, plain desks, yes. Real phone videos showed it depends on the
background: on a pale wooden board it outlined the whole board, and patterned cloth and white
posters confused it. Rules about brightness and edges can't separate "paper" from "anything
bright or edgy"; a network trained on varied backgrounds learns the combination of cues.

**Q: What architecture and why?**
A: A small U-Net: a 4-level encoder (16-128 channels), a bottleneck, a decoder with skip
connections, 1.2 M parameters, 256x256 input. Documents are big, simple shapes, so a small
network suffices, and it must run per video frame on a CPU.

**Q: Where does the training data come from?**
A: It's synthetic: our cheque and page renderers, warped in perspective onto procedurally drawn
hard backgrounds (pale wood, white tables, bright boards, patterned cloth, white-text posters,
clutter), with thumbs, lighting, blur, noise and JPEG. Masks are exact and free. Validation
uses documents never seen in training.

**Q: Isn't there a gap between synthetic and real images?**
A: Yes, the domain gap. We reduce it by randomising everything irrelevant (domain
randomisation), and the pipeline guards against failures: the mask must agree with a
quadrilateral (IoU at least 0.9) or the classical detector is used. Real videos are checked in
the evaluation; a larger real test set is planned for Phase 9.

**Q: Why BCE plus Dice?**
A: BCE alone is dominated by background pixels when the document is small; Dice measures the
overlap of the whole shape and is insensitive to that imbalance. Together they train faster and
give cleaner edges.

**Q: The network works at 256x256. How are corners accurate on a 1280x720 frame?**
A: The mask gives a rough quadrilateral; each side is then refined on the full-resolution image
by finding the strongest brightness step across it at 40 points and fitting a robust line
(Huber); adjacent lines intersect at the corners. The evaluation measures the error with and
without this step.

**Q: Why ONNX and OpenCV DNN instead of PyTorch at runtime?**
A: PyTorch is a 2+ GB dependency. The app, CI and Docker image only need OpenCV, which can run
ONNX models. Training is the only step that needs PyTorch. OpenCV's output matches PyTorch's to
within 1e-3.

**Q: How good is it?**
A: On 320 hard synthetic images: page found 96% (classical 45%), corner error 0.14% of the
diagonal (classical 0.56%), and never claims a cut-off page is whole. On a real video of a
cheque on a pale wooden board, which the classical detector rejected, it gave 102 usable frames
and a flattened cheque of the right shape.

**Q: Why did training take an hour on a GPU?**
A: The bottleneck was generating synthetic images on the CPU, not the network: with 2 workers
the GPU mostly waited (12 min per epoch); 8 workers halved it. The default now scales with the
number of CPU cores.

**Q: What if the model is wrong?**
A: Three guards: the agreement check (mask vs. quadrilateral), the classical fallback, and the
rest of the pipeline's own checks (tracking only starts on an outline clear of the border; the
coverage gate; template registration corrects the outline for cheques). `FINDOC_SEGMENTATION=off`
disables it entirely.

**Q: Why are weights not in the repository?**
A: Project rule: never commit datasets or weights. Anyone can reproduce the model with one
command, because the data is generated.
