# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/);
each build phase is released as a minor version.

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
