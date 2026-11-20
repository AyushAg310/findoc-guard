# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/);
each build phase is released as a minor version.

## [0.6.0] - Phase 5: 3D reconstruction and flattening

### Added
- **Bent synthetic documents** (`findoc/synth/surface.py`): developable surfaces (wave, curl,
  curl at both ends, fold) defined by a curvature profile, rendered with exact ground truth;
  `render_video(..., bend=...)` and `findoc synth --kind video --bend fold`.
- **Structure from motion** (`findoc/recon3d/sfm.py`): SIFT, essential-matrix and
  homography-decomposition initialisation, track building, PnP registration, triangulation and
  sparse bundle adjustment (SciPy); focal-length search when the camera is not calibrated.
- **Flattening** (`findoc/recon3d/flatten.py`, `pipeline.py`): planarity test (flat pages keep
  the exact 2D homography), robust thin-plate-spline surface, ray-cast mesh, ARAP
  parameterisation to convergence, resampling with a flat-page -> photo correspondence map.
  Safety checks fall back to 2D with a reason.
- **`recon3d` pipeline stage** for video: `doc.flattening` (`homography` or `3d`) and
  `doc.recon3d` (statistics, reason) in the JSON; settings `FINDOC_RECON3D` (`auto`/`off`) and
  `FINDOC_FOCAL_PX`.
- **Camera calibration** (`findoc calibrate SOURCE --board 9x6 --save`): Zhang's method on a
  printed checkerboard, writes the focal length to `.env`.
- 3D outputs with `--save-dir`: point cloud and mesh (PLY), 3D preview, flat map (`.npz`).
- `experiments/phase5_recon_eval.py`: 3D vs. 2D on 15 synthetic hand-held videos, focal
  estimated or known. Fold error 0.179% -> 0.072% of cheque width; every bend improves; flat
  pages correctly kept 2D.
- Learning guide `docs/learn/phase-5-3d-reconstruction.md`.

### Changed
- SciPy is a core dependency; the unused `recon3d` extra (pycolmap, Open3D, trimesh) is removed.
- `rectify` is skipped when the page was already flattened from 3D.

## [0.5.1] - GPU-trained results; automatic training workers

### Changed
- `findoc segment train` uses CPU cores minus 2 (at most 8) data workers by default
  (`--workers -1`). Image generation on the CPU was the bottleneck: on a 12-core laptop with an
  RTX 3050, 2 workers gave ~12 min per epoch, 8 workers ~5-6 min.
- Published Phase 4 results now come from the GPU-trained model (20,000 images x 10 epochs,
  55 min, validation IoU 0.993): page found on hard backgrounds 95.9% (classical 45.0%), corner
  error 0.136% (0.556%), cut-off pages never called whole; edge refinement halves the error. Real
  pale-wood video: 102/137 usable frames, flattened cheque 2.28:1.

## [0.5.0] - Phase 4: learned page detection

Added after real phone videos showed the classical detector depends on the background (it
outlined a whole pale wooden board instead of the cheque on it). 3D reconstruction moves to
Phase 5; later phases shift by one.

### Added
- **Hard synthetic backgrounds** (`findoc/synth/backgrounds.py`): pale wood, near-white tables,
  bright boards on dark surrounds, patterned cloth, white-text posters, clutter, noise.
- **Segmentation training samples** (`findoc/synth/segdata.py`): cheques, pages and plain sheets
  in several paper tints, in perspective (including sideways), thumbs, lighting, blur, noise,
  JPEG; exact full-document masks.
- **DocSegNet** (`findoc/segment/model.py`): small U-Net, 1.2 M parameters, 256x256 input.
- **Training** (`findoc/segment/train.py`, `findoc segment train`): on-the-fly synthetic data,
  BCE + Dice loss, AdamW, one-cycle schedule, validation IoU on unseen documents, ONNX export
  plus a JSON training record. New `[train]` extra (PyTorch, onnx); weights stay out of git.
- **Inference without PyTorch** (`findoc/segment/infer.py`): ONNX through OpenCV DNN, largest
  region to quadrilateral, sub-pixel edge refinement with robust (Huber) line fits, and a
  mask/quadrilateral agreement score.
- `findoc segment check IMAGE`: heat map and fitted outline.
- `experiments/phase4_segmentation_eval.py`: learned vs. classical vs. learned-without-refinement
  on hard synthetic backgrounds, plus optional real videos.
- Learning guide `docs/learn/phase-4-learned-segmentation.md`.

### Changed
- `find_document` uses the learned detector when a model exists (`FINDOC_SEGMENTER_MODEL`,
  default `models/docseg.onnx`) and its outline agrees with its mask (IoU >= 0.9); otherwise the
  classical detector (`find_document_classical`), unchanged. `FINDOC_SEGMENTATION=off` disables
  it. Tracking start, drift correction and re-localisation use it automatically.
- "Clear of the image border" margin is now 1% of the longer image side (at least 4 px): the
  learned outline of a page running off a 960 px frame can stop 6-7 px short of the edge.
- Roadmap renumbered: 3D reconstruction is Phase 5, fraud engine 6, invoices 7, product 8,
  evaluation 9.

### Results (development model, CPU-trained)
- Hard synthetic backgrounds: page found 90% (classical 45%); near-white table 78% (5%), bright
  board 88% (8%), white-text poster 92% (25%); corner error 0.38% (0.56%); cut-off pages never
  called whole.
- Real videos: the whole cheque on a pale wooden board is now processed (classical rejected it);
  the two videos with the cheque running off the frame are still rejected.

### Notes
- Mask holes are filled before fitting (`solid_region`); sides along the image border are not
  edge-refined; training includes 20% documents running off the frame. Each came from a failure
  on real or synthetic data, described in the learning guide.

## [0.4.2] - Track only the whole document

### Fixed
- Tracking now starts only at a frame where the page detector finds the whole document, with an
  outline clear of the image border. Found with a real portrait video of a cheque wider than the
  frame: the detector returned the visible part clipped at the border, the tracker followed it,
  and the pipeline reported "ok" on a partial, square "page". Such videos are now rejected with
  "never fits fully inside the frame" and, for cheques, advice to film in landscape.
- The tracker no longer falls back to tracking the whole frame when no page is found.
- Frame sharpness is measured after a light Gaussian smoothing (sigma 1 px): Laplacian variance
  also rises with pixel noise, so a noisy blurred frame could look sharp. Under OpenCV 5.1 the
  test clip picked a blurred best frame; rank correlation with true blur on that clip improved
  from -0.15 to -0.82. The absolute sharpness floor is now 15 (new scale); the relative test
  (75% of the clip median) does the selection. Tests use 960x540 clips (robust on OpenCV 4.10,
  5.0 and 5.1).

## [0.4.1] - Clear reasons when a video is rejected

### Fixed
- A rejected video now says why (`findoc.track.stages.rejection_reasons`): the document has too
  little detail to track (e.g. a blank page), it never fits fully inside the frame, no outline
  was found, or the frames are blurred. Previously every case got "hold the phone steadier".
  Found with the first real phone video (4K portrait, a blank notebook page larger than the
  frame).
- A document outline touching the image border no longer counts as "fully in view": the page
  detector clips off-image outlines to the border, so such a page usually continues beyond it.

## [0.4.0] - Phase 3: video input, KLT tracking, keyframe selection

### Added
- **Video input** (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`): two-pass reading. Frames are
  streamed, subsampled to at most 180 and downscaled to 960 px for tracking
  (`findoc.io.read_video`); only the chosen frames are decoded again at full resolution
  (`read_frames`).
- **KLT document tracking** (`findoc/track/klt.py`): Shi-Tomasi corners inside the document,
  pyramidal Lucas-Kanade, forward-backward check (1 px), reference-anchored RANSAC homography,
  re-detection below 150 points with drift correction (snap to the page detector when it
  agrees within 1% of the diagonal, else verify against a SIFT anchor view), loss detection and
  SIFT re-localisation, and sharpness measured on the rectified document.
- **Keyframe and best-frame selection** (`findoc/track/keyframes.py`): usable = tracked, fully in
  view, at least 75% of the median sharpness; keyframes by projective viewpoint change (corner
  residual after the best similarity); best frame by sharpness x frontalness x coverage.
- Pipeline stages `video` and `track` (`findoc/track/stages.py`): the best frame and its tracked
  corners feed the unchanged Phase 1-2 stages. New `Document` fields: `track`, `tracking`,
  `keyframes`, `best_frame`, `video`, `frame_numbers` (frames and keyframe images are kept out
  of the JSON). New settings: `video_max_frames`, `video_track_side`,
  `keyframe_min_view_change`, `keyframe_max`.
- **Synthetic hand-held videos** with exact per-frame ground truth: pinhole camera model
  (`findoc/synth/camera.py`, plane-induced homography), camera sweep, hand shake, wrist jerks,
  shutter-integrated motion blur, optional thumb occlusion and pan-away
  (`findoc/synth/video.py`). `findoc synth --kind video`.
- CLI: tracking summary; `--save-dir` writes the tracking video, keyframe contact sheet and best
  frame. Streamlit: video upload and "Cheque video" mode.
- `experiments/phase3_tracking_eval.py` and results (`experiments/results/phase3.json`), with
  `--real-ocr`.
- Learning guide `docs/learn/phase-3-video-tracking.md`. 139 tests, 95% coverage.

### Changed
- `detect_stage` keeps tracked corners for video input.
- An unreadable video now fails in the `video` stage (previously video was only registered).

## [0.3.2] - Validated date reading, figures parser fix

### Fixed
- **Regression from 0.3.1:** reading the date only in re-spaced form dropped real-OCR date
  accuracy from 94% to 49%. Boxed fields are now read with validated fallbacks
  (`read_boxed`): as written, then re-spaced, then digit by digit, keeping the first reading
  that is a valid calendar date. The method used is recorded in `ExtractedField.method`.
- Amount in figures: a dot followed by three digits is a misread comma, not a decimal point
  (`30,00.189` was parsed as 3,000.19; now 30,00,189).

### Added
- `experiments/date_reading_ablation.py`: accuracy of each date-reading method on real OCR,
  with images of failures. Results (RTX 3050, 100 cheques): as written 96%, re-spaced 45%,
  digit by digit 100%, validated pipeline 100%.
- Real-OCR results for 0.3.2 (`experiments/results/phase2_real_ocr.json`): 100% on every
  field, 100% recall for all four fraud types, 0% false alarms on clean cheques.

## [0.3.1] - Date-box re-spacing (superseded), real-OCR results

### Added
- Real-OCR evaluation results (EasyOCR on an RTX 3050 laptop GPU, 100 cheques):
  `experiments/results/phase2_real_ocr_v0.3.0.json`.

### Fixed
- Date boxes are now read as a compact digit string: the crop is cut into its 8 cells, each
  digit trimmed and re-joined with normal spacing (`compact_cells`). With v0.3.0 the date was
  the weakest field (94%) and caused most false flags and missed post-dated cheques.
- `FieldSpec` gains `cells` (saved in template JSON; defaults to 1 for existing templates).

## [0.3.0] - Phase 2: Cheque understanding

### Added
- Synthetic CTS-2010-style cheque generator with four fictional banks, handwriting-style
  fields, signature scribbles, MICR line and fraud conditions (amount mismatch, stale,
  post-dated, unsigned) (`findoc/synth/cheque.py`, `findoc/synth/dataset.py`).
- Cheque template store: reference image + field boxes, loadable from a folder
  (`findoc/localize/templates.py`); `findoc templates list|export`.
- Template identification and registration with SIFT (or ORB) features, Lowe ratio test and
  RANSAC homography, with sanity checks and an ambiguity margin (`findoc/localize/align.py`).
- Field extraction with template subtraction: pre-printed lines, labels and security pattern
  are removed before OCR (`findoc/recognize/fields.py`).
- Parsers for amount in words (Indian lakh/crore system, OCR typo repair), amount in figures
  (Indian digit grouping), cheque dates, IFSC, account numbers and MICR lines
  (`findoc/recognize/parsers.py`).
- Rule checks: amount mismatch/unreadable, stale and post-dated cheques (RBI 3-month
  validity), IFSC/account/MICR format, missing signature (`findoc/fraud/rules.py`).
- OCR engine `read_line` (recognition-only on known regions); model-load time reported
  separately from recognition time.
- Simulated OCR with realistic character errors for robustness experiments
  (`findoc/synth/ocr_noise.py`).
- `findoc synth --kind cheque`; cheque fields and checks tables in `findoc process`; registered
  cheque with field boxes saved by `--save-dir`.
- Streamlit demo: synthetic cheques with fraud conditions, field table, checks, field crops.
- Evaluation (`experiments/phase2_cheque_eval.py`): geometry, simulated-OCR robustness and a
  `--real-ocr` mode.
- Learning guide `docs/learn/phase-2-cheques.md`.

### Changed
- Pipeline: load -> quality -> detect -> rectify -> enhance -> align -> fields -> rules -> ocr.
  Full-page OCR is skipped for recognised cheques.
- `ExtractedField` gains `raw` and `normalized`; `Document` gains `template_id`,
  `template_homography`, `registered`.
- Default `FINDOC_MIN_COVERAGE` lowered from 0.40 to 0.25 (cheques are long and thin).
- Synthetic photo generator preserves any page aspect ratio.

## [0.2.0] - Phase 1: Flat-photo pipeline

### Added
- Quality gate (`findoc/preprocess/quality.py`): Laplacian-variance sharpness, exposure and
  glare metrics; blurry, dark or overexposed photos are rejected with a retake message.
- Page detection (`findoc/preprocess/detect.py`): auto-Canny edges and Otsu threshold with
  Douglas-Peucker quadrilateral fitting, rotated-rectangle and full-frame fallbacks; rejects
  photos where the page fills less than 40% of the frame.
- Perspective correction (`findoc/preprocess/rectify.py`): four-point homography; the matrix
  is stored in the result for later box mapping.
- Shadow removal (`findoc/preprocess/illumination.py`): background estimation by dilation and
  median filtering, then division.
- OCR stage (`findoc/recognize/ocr.py`): pluggable engine interface, EasyOCR backend
  (optional `[ocr]` extra, GPU when available), reading-order text assembly.
- Synthetic photo generator with ground-truth corners (`findoc/synth/photo.py`) and the
  `findoc synth` command.
- CLI: `--save-dir` (detected/flat/enhanced images), `--ocr/--no-ocr`.
- Streamlit demo (`demo/streamlit_app.py`).
- Evaluation script (`experiments/phase1_detection_eval.py`) and results.
- Synthetic sample images in `data/samples/`.
- Learning guide `docs/learn/phase-1-flat-photo.md`.

### Changed
- `Document` gains `enhanced`, `corners`, `homography`, `detection_method`, `words`,
  `full_text`; new `OcrWord` model.
- `default_pipeline()` runs load -> quality -> detect -> rectify -> enhance -> ocr.
- CI also tests Python 3.12.

## [0.1.0] - Phase 0: Scaffold

### Added
- `findoc` package with stage-based `Pipeline` and `Document` result model.
- `findoc` CLI (`version`, `process`) with JSON output.
- Settings via environment variables / `.env` (`FINDOC_*`).
- Empty stage packages for every planned phase.
- Tests (pytest + coverage), linting (ruff), pre-commit hooks.
- GitHub Actions CI (Python 3.10 and 3.11, Docker build + smoke test).
- CPU Dockerfile.
- Project documentation: SRS & design document, architecture diagram, flowcharts.
