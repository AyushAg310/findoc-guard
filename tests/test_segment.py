"""Phase 4: hard backgrounds, segmentation samples, mask -> quad, edge refinement, integration.

Tests that need PyTorch (model, training, ONNX export) are skipped when it isn't installed (CI).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from findoc.cli import app
from findoc.preprocess import detect
from findoc.segment.infer import Segmentation, load_segmenter, mask_to_quad, refine_quad, to_blob
from findoc.synth.backgrounds import KINDS, background
from findoc.synth.segdata import document_pool, make_sample, plain_sheet


@pytest.fixture(scope="module")
def pool() -> list[np.ndarray]:
    return document_pool(seed=1, n=6)


def test_every_background_kind() -> None:
    for kind in KINDS:
        img, k = background(np.random.default_rng(3), 160, 90, kind)
        assert k == kind and img.shape == (90, 160, 3) and img.dtype == np.uint8
        again, _ = background(np.random.default_rng(3), 160, 90, kind)
        assert np.array_equal(img, again)  # deterministic


def test_sample_mask_matches_corners(pool) -> None:  # type: ignore[no-untyped-def]
    for seed in range(6):
        s = make_sample(np.random.default_rng(seed), pool, size=(320, 240), partial_rate=0.0)
        h, w = s.mask.shape
        assert s.image.shape[:2] == (h, w) and {h, w} == {240, 320}
        cx, cy = s.corners.mean(axis=0).round().astype(int)
        assert s.mask[cy, cx] == 255
        assert np.all(s.corners >= 0) and np.all(s.corners <= [w, h])
        area = cv2.contourArea(s.corners.astype(np.float32))
        assert abs(area - np.count_nonzero(s.mask)) / area < 0.05


def test_partial_samples_reach_the_border(pool) -> None:  # type: ignore[no-untyped-def]
    for seed in range(8):
        s = make_sample(np.random.default_rng(seed), pool, size=(320, 240), partial_rate=1.0)
        h, w = s.mask.shape
        assert s.partial and not s.occluded
        outside = np.any((s.corners < 0) | (s.corners > [w - 1, h - 1]))
        on_edge = s.mask[0].any() or s.mask[-1].any() or s.mask[:, 0].any() or s.mask[:, -1].any()
        assert outside and on_edge


def test_plain_sheet_and_pool(pool) -> None:  # type: ignore[no-untyped-def]
    assert len(pool) == 6
    assert plain_sheet(np.random.default_rng(0)).dtype == np.uint8


def _quad_mask(shape: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, quad.round().astype(np.int32), 255)
    return m


def test_mask_to_quad() -> None:
    truth = np.array([[60, 40], [300, 55], [290, 200], [50, 180]], np.float32)
    quad = mask_to_quad(_quad_mask((240, 360), truth))
    assert quad is not None
    err = np.linalg.norm(detect.order_corners(quad) - truth, axis=1).max()
    assert err < 3
    assert mask_to_quad(np.zeros((50, 50), np.uint8)) is None
    blob = np.zeros((100, 100), np.uint8)
    cv2.circle(blob, (50, 50), 30, 255, -1)
    assert mask_to_quad(blob).shape == (4, 2)  # min-area-rect fallback for a non-quad shape


def test_solid_region_fills_holes_and_keeps_largest() -> None:
    from findoc.segment.infer import solid_region

    m = np.zeros((100, 160), np.uint8)
    cv2.rectangle(m, (20, 20), (120, 80), 255, -1)
    cv2.circle(m, (70, 50), 15, 0, -1)  # hole
    cv2.rectangle(m, (135, 5), (155, 20), 255, -1)  # small second region
    out = solid_region(m)
    assert out is not None and out[50, 70] == 255 and out[10, 145] == 0
    assert solid_region(np.zeros((10, 10), np.uint8)) is None


def test_refine_quad_snaps_to_edges() -> None:
    truth = np.array([[80, 60], [520, 80], [500, 340], [70, 320]], np.float32)
    img = np.full((400, 600), 70, np.uint8)
    cv2.fillConvexPoly(img, truth.astype(np.int32), 230)
    rough = truth + np.array([[4, -3], [-3, 4], [3, 3], [-4, -2]], np.float32)
    refined = refine_quad(img, rough)
    assert np.linalg.norm(refined - truth, axis=1).max() < 1.5
    assert np.linalg.norm(rough - truth, axis=1).max() > 4
    cut = np.full((400, 600), 70, np.uint8)  # page running off the right edge, printed line inside
    cv2.rectangle(cut, (100, 60), (599, 340), 230, -1)
    cv2.line(cut, (590, 60), (590, 340), 40, 2)
    on_edge = np.array([[100, 60], [599, 60], [599, 340], [100, 340]], np.float32)
    assert np.allclose(refine_quad(cut, on_edge)[1:3, 0], 599, atol=1)  # right side not pulled in
    flat = np.full((400, 600), 100, np.uint8)  # no edges: corners stay where they were
    assert np.allclose(refine_quad(flat, rough), rough)


def test_to_blob() -> None:
    blob = to_blob(np.full((90, 160, 3), 255, np.uint8))
    assert blob.shape == (1, 3, 256, 256) and np.isclose(blob.max(), 0.5)


class FakeSegmenter:
    def __init__(self, corners: np.ndarray | None, agreement: float) -> None:
        self.corners, self.agreement = corners, agreement

    def segment(self, img: np.ndarray) -> Segmentation:
        return Segmentation(self.corners, np.zeros(img.shape[:2], np.float32), self.agreement)


def test_find_document_prefers_confident_learned(
    monkeypatch: pytest.MonkeyPatch, synth_photo
) -> None:  # type: ignore[no-untyped-def]
    photo, truth = synth_photo
    monkeypatch.setattr(detect, "_segmenter", lambda: FakeSegmenter(truth, 0.97))
    det = detect.find_document(photo)
    assert det.method == "learned" and np.allclose(det.corners, detect.order_corners(truth))
    monkeypatch.setattr(detect, "_segmenter", lambda: FakeSegmenter(truth, 0.5))  # unsure
    assert detect.find_document(photo).method != "learned"
    monkeypatch.setattr(detect, "_segmenter", lambda: FakeSegmenter(None, 0.0))  # nothing
    assert detect.find_document(photo).method != "learned"


def test_segmentation_setting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from findoc.config import get_settings

    assert detect._segmenter() is None  # tests run with FINDOC_SEGMENTATION=off
    monkeypatch.setenv("FINDOC_SEGMENTATION", "auto")
    monkeypatch.setenv("FINDOC_SEGMENTER_MODEL", str(tmp_path / "missing.onnx"))
    get_settings.cache_clear()
    assert detect._segmenter() is None  # no model file -> classical only


def test_load_segmenter_bad_files(tmp_path: Path) -> None:
    assert load_segmenter(str(tmp_path / "nope.onnx")) is None
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"not a model")
    assert load_segmenter(str(bad)) is None


def test_cli_segment_check_without_model(
    tmp_path: Path, sample_image: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from findoc.config import get_settings

    monkeypatch.setenv("FINDOC_SEGMENTER_MODEL", str(tmp_path / "none.onnx"))
    get_settings.cache_clear()
    res = CliRunner().invoke(app, ["segment", "check", str(sample_image)])
    assert res.exit_code != 0 and "train one first" in res.output


# ------------------------------------------------------------------ PyTorch (skipped in CI)


def test_model_export_and_opencv_inference(tmp_path: Path, synth_photo) -> None:  # type: ignore[no-untyped-def]
    torch = pytest.importorskip("torch")
    from findoc.segment.infer import Segmenter
    from findoc.segment.model import DocSegNet
    from findoc.segment.train import export_onnx

    model = DocSegNet()
    path = export_onnx(model, tmp_path / "m.onnx")
    seg = Segmenter(path)
    x = to_blob(synth_photo[0])
    with torch.no_grad():
        ref = model.eval()(torch.from_numpy(x)).numpy()
    seg.net.setInput(x)
    assert np.abs(seg.net.forward() - ref).max() < 1e-3  # OpenCV DNN == PyTorch
    out = seg.segment(synth_photo[0])
    assert out.mask.shape == synth_photo[0].shape[:2]


def test_training_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from findoc.segment.train import train

    res = train(
        tmp_path / "m.onnx",
        samples=8,
        val_samples=4,
        epochs=1,
        batch_size=4,
        workers=0,
        log=lambda *_: None,
    )
    assert res.onnx_path.exists() and (tmp_path / "m.json").exists()
    assert 0.0 <= res.best_val_iou <= 1.0
