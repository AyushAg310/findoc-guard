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


def test_process_writes_json_and_images(synth_photo_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    save = tmp_path / "imgs"
    result = runner.invoke(
        app, ["process", str(synth_photo_path), "--out", str(out), "--save-dir", str(save)]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    assert data["status"] == "ok"
    assert [s["stage"] for s in data["stages"]][:5] == [
        "load",
        "quality",
        "detect",
        "rectify",
        "enhance",
    ]
    names = sorted(p.name for p in save.iterdir())
    assert names == ["synth_detected.jpg", "synth_enhanced.png", "synth_flat.png"]


def test_process_no_ocr_flag(synth_photo_path: Path) -> None:
    result = runner.invoke(app, ["process", str(synth_photo_path), "--no-ocr"])
    assert result.exit_code == 0
    assert "OCR skipped" not in result.output


def test_blurry_photo_exit_code_zero_with_warning(blurry_photo_path: Path) -> None:
    result = runner.invoke(app, ["process", str(blurry_photo_path)])
    assert result.exit_code == 0  # rejection is a valid outcome, not a crash
    assert "rejected_quality" in result.output


def test_synth_command(tmp_path: Path) -> None:
    result = runner.invoke(app, ["synth", "--n", "2", "--out", str(tmp_path), "--seed", "4"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "labels.json").exists()
    assert len(list(tmp_path.glob("*.jpg"))) == 2


def test_synth_cheques_and_process(tmp_path: Path) -> None:
    out = tmp_path / "cheques"
    result = runner.invoke(app, ["synth", "--kind", "cheque", "--n", "1", "--out", str(out)])
    assert result.exit_code == 0, result.output
    photo = next(out.glob("*.jpg"))
    save = tmp_path / "imgs"
    result = runner.invoke(app, ["process", str(photo), "--no-ocr", "--save-dir", str(save)])
    assert result.exit_code == 0, result.output
    assert "Cheque template" in result.output
    assert "SIGNATURE_MISSING" in result.output
    assert (save / f"{photo.stem}_fields.png").exists()


def test_synth_bad_kind(tmp_path: Path) -> None:
    result = runner.invoke(app, ["synth", "--kind", "invoice", "--out", str(tmp_path)])
    assert result.exit_code != 0


def test_templates_commands(tmp_path: Path) -> None:
    result = runner.invoke(app, ["templates", "list"])
    assert result.exit_code == 0 and "ARAVALLI BANK" in result.output
    result = runner.invoke(app, ["templates", "export", "--out", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "ARVL" / "template.json").exists()
