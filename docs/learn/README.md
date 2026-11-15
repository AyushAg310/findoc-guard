# Learning guide: understand FinDoc Guard file by file

This folder explains **every file in the repository**, one guide per build phase. Each guide
is written so you can defend the code in a viva or an interview without having written every
line yourself.

| Guide | Covers | Version |
| --- | --- | --- |
| [Phase 0 - Scaffold](phase-0-scaffold.md) | Packaging, CLI, config, pipeline design, data models, tests, CI, Docker, git hygiene | 0.1.0 |
| [Phase 1 - Flat-photo pipeline](phase-1-flat-photo.md) | Quality gate, page detection, homography, shadow removal, OCR, synthetic data, evaluation | 0.2.0 |
| Phase 2 - Cheque understanding | Template alignment, field recognition, amount-in-words, rules | coming next |
| Phase 3 - Video and tracking | KLT optical flow, keyframe selection | |
| Phase 4 - 3D reconstruction | SfM, MVS, meshing, ARAP flattening, correspondence maps | |
| Phase 5 - Fraud engine | Tamper heatmaps, duplicate search, signature check, risk score | |
| Phase 6 - Invoices | LayoutLMv3, GSTIN checksum, invoice arithmetic | |
| Phase 7 - Product | FastAPI, PostgreSQL, Celery, React UI | |
| Phase 8 - Evaluation | Metrics, ablations, results | |

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
