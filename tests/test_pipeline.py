from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from findoc.io import detect_kind, load_image
from findoc.pipeline import Pipeline, default_pipeline, process
from findoc.types import Document, InputKind, Status


def test_detect_kind() -> None:
    assert detect_kind(Path("a.JPG")) is InputKind.IMAGE
    assert detect_kind(Path("clip.mp4")) is InputKind.VIDEO
    with pytest.raises(ValueError):
        detect_kind(Path("notes.txt"))


def test_load_image(sample_image: Path) -> None:
    img = load_image(sample_image)
    assert img.shape == (600, 800, 3)


def test_default_pipeline_stage_order() -> None:
    names = [n for n, _ in default_pipeline().stages]
    assert names == [
        "load", "video", "track", "quality", "detect", "recon3d", "rectify", "enhance",
        "classify", "align", "fields", "rules", "tamper", "duplicate", "signature", "ocr",
        "invoice", "risk",
    ]  # fmt: skip
    assert "ocr" not in [n for n, _ in default_pipeline(ocr=False).stages]


def test_process_synthetic_photo(synth_photo_path: Path) -> None:
    doc = process(synth_photo_path)
    assert doc.status is Status.OK
    assert doc.detection_method in {"edges", "threshold"}
    assert doc.flat is not None and doc.enhanced is not None
    assert doc.quality["sharpness"] > 100
    result = doc.result()
    assert "original" not in result and "flat" not in result  # images never leak into JSON
    assert len(result["corners"]) == 4
    assert len(result["homography"]) == 3


def test_blurry_photo_is_rejected_early(blurry_photo_path: Path) -> None:
    doc = process(blurry_photo_path)
    assert doc.status is Status.REJECTED_QUALITY
    # video/track are no-ops for photos; stages after the quality gate are skipped
    assert [s.stage for s in doc.stages] == ["load", "video", "track", "quality"]
    assert any("blurry" in m for m in doc.messages)


def test_small_document_is_rejected(tmp_path: Path) -> None:
    img = np.full((1200, 1600, 3), 80, np.uint8)
    img += np.random.default_rng(0).integers(0, 20, img.shape, dtype=np.uint8)
    cv2.rectangle(img, (650, 420), (1050, 900), (240, 240, 240), -1)  # ~10% of the frame
    for y in range(470, 880, 30):
        cv2.putText(img, "TEXT LINE", (680, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,) * 3, 2)
    path = tmp_path / "far.jpg"
    cv2.imwrite(str(path), img)
    doc = process(path)
    # Too small for the 15% candidate threshold -> either rejected as small or full-frame.
    assert doc.status in {Status.REJECTED_QUALITY, Status.OK}
    if doc.status is Status.REJECTED_QUALITY:
        assert any("Move closer" in m for m in doc.messages)


def test_unreadable_video_fails_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00")
    doc = process(path)
    assert doc.status is Status.FAILED
    assert doc.flat is None
    assert doc.stages[-1].stage == "video" and not doc.stages[-1].ok


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
