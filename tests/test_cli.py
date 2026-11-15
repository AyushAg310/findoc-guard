from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from findoc import __version__
from findoc.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_process_writes_json(sample_image: Path, tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    result = runner.invoke(app, ["process", str(sample_image), "--out", str(out)])
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert data["stages"][0]["stage"] == "load"
