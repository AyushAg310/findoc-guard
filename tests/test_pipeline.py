from __future__ import annotations

from pathlib import Path

import pytest

from findoc.io import detect_kind, load_image
from findoc.pipeline import Pipeline, process
from findoc.types import Document, InputKind, Status


def test_detect_kind() -> None:
    assert detect_kind(Path("a.JPG")) is InputKind.IMAGE
    assert detect_kind(Path("clip.mp4")) is InputKind.VIDEO
    with pytest.raises(ValueError):
        detect_kind(Path("notes.txt"))


def test_load_image(sample_image: Path) -> None:
    img = load_image(sample_image)
    assert img.shape == (600, 800, 3)


def test_process_image(sample_image: Path) -> None:
    doc = process(sample_image)
    assert doc.status is Status.OK
    assert [s.stage for s in doc.stages] == ["load"]
    result = doc.result()
    assert "original" not in result  # images never leak into JSON
    assert result["kind"] == "image"


def test_failing_stage_is_recorded(sample_image: Path) -> None:
    def boom(doc: Document) -> Document:
        raise RuntimeError("stage exploded")

    doc = Pipeline().add("boom", boom).run(sample_image)
    assert doc.status is Status.FAILED
    assert doc.stages[-1].ok is False
    assert "exploded" in doc.stages[-1].note


def test_pipeline_stops_after_rejection(sample_image: Path) -> None:
    calls: list[str] = []

    def reject(doc: Document) -> Document:
        doc.status = Status.REJECTED_QUALITY
        return doc

    def never(doc: Document) -> Document:
        calls.append("never")
        return doc

    Pipeline().add("reject", reject).add("never", never).run(sample_image)
    assert calls == []
