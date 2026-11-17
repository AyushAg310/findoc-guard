# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/);
each build phase is released as a minor version.

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
  with images of failures.

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
