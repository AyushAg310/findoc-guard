# Repository conventions

- Build proceeds in phases (see README roadmap). Each phase is one minor version.
- Every phase must ship with a learning guide at `docs/learn/phase-N-<name>.md` that explains
  every new or changed file, following the structure in `docs/learn/README.md`
  (big picture, run flow, file-by-file walkthrough, concepts, exercises, viva questions).
  Update the index table in `docs/learn/README.md` and the roadmap status in `README.md`.
- New functionality is added as pipeline stages (`Callable[[Document], Document]`) registered in
  `findoc/pipeline.py`; don't change the CLI/JSON contract without updating the guide.
- Keep `ruff check .`, `ruff format --check .` and `pytest` passing; coverage at least 75%.
- Update `CHANGELOG.md` and bump the version in `pyproject.toml` and `findoc/__init__.py`.
- Only synthetic or consented documents. Never commit datasets, weights or real documents.
