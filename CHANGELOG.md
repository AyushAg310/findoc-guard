# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/);
each build phase is released as a minor version.

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
