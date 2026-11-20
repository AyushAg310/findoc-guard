# Learning guide: understand FinDoc Guard file by file

This folder explains **every file in the repository**, one guide per build phase. Each guide
is written so you can defend the code in a viva or an interview without having written every
line yourself.

| Guide | Covers | Version |
| --- | --- | --- |
| [Phase 0 - Scaffold](phase-0-scaffold.md) | Packaging, CLI, config, pipeline design, data models, tests, CI, Docker, git hygiene | 0.1.0 |
| [Phase 1 - Flat-photo pipeline](phase-1-flat-photo.md) | Quality gate, page detection, homography, shadow removal, OCR, synthetic data, evaluation | 0.2.0 |
| [Phase 2 - Cheque understanding](phase-2-cheques.md) | Synthetic cheques, SIFT + RANSAC template registration, template subtraction, parsers, rules, robustness evaluation | 0.3.0 |
| [Phase 3 - Video and tracking](phase-3-video-tracking.md) | Pinhole camera and synthetic video, KLT optical flow, drift correction, re-localisation, keyframe and best-frame selection, tracking evaluation | 0.4.0 |
| [Phase 4 - Learned page detection](phase-4-learned-segmentation.md) | Hard synthetic backgrounds, U-Net segmentation, BCE + Dice training, ONNX export and OpenCV DNN inference, mask-to-quad, edge refinement, evaluation | 0.5.0 |
| [Phase 5 - 3D reconstruction](phase-5-3d-reconstruction.md) | Developable surfaces, incremental SfM, bundle adjustment, focal search, planarity test, surface fit, ARAP flattening, correspondence map, camera calibration, evaluation | 0.6.0 |
| Phase 6 - Fraud engine | Tamper heatmaps, duplicate search, signature check, risk score | coming next |
| Phase 7 - Invoices | LayoutLMv3, GSTIN checksum, invoice arithmetic | |
| Phase 8 - Product | FastAPI, PostgreSQL, Celery, React UI | |
| Phase 9 - Evaluation | Metrics, ablations, results | |

> **Phase numbers changed in v0.5.0.** Learned page detection was inserted as Phase 4, so 3D
> reconstruction is now Phase 5, the fraud engine Phase 6, invoices Phase 7, the product Phase 8
> and evaluation Phase 9. The guides for Phases 0-2 were written before that and use the old
> numbers when they mention future phases (add one to any phase number of 4 or more).

## How each guide is organised

1. **Big picture**: what the phase adds and why, in a few sentences.
2. **How a run flows**: what happens, in order, when you run the command.
3. **File-by-file walkthrough**: every file, what it does, and the important lines explained.
4. **Concepts used**: the Python, CV or engineering ideas behind the code, explained from scratch.
5. **Try it yourself**: small exercises that prove you understand it.
6. **Viva questions**: questions an examiner or interviewer is likely to ask, with answers.

## How to study a guide

Open the guide on one side and the code on the other (VS Code split view: `Ctrl+\`). Read the
walkthrough for a file, then read the file itself top to bottom, then do the exercise. Run the
tests (`pytest`) after any change you make while experimenting.
