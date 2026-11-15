from __future__ import annotations

import cv2
import numpy as np
import pytest

from findoc.preprocess.detect import draw_detection, find_document, order_corners
from findoc.preprocess.illumination import background_unevenness, normalize_illumination
from findoc.preprocess.quality import assess, resize_longest
from findoc.preprocess.rectify import map_points, output_size, warp_document

# ---------- quality ----------


def test_sharp_photo_passes(synth_photo: tuple[np.ndarray, np.ndarray]) -> None:
    report = assess(synth_photo[0])
    assert report.acceptable
    assert report.sharpness > 100
    assert 0 < report.score <= 1


def test_blurred_photo_rejected(synth_photo: tuple[np.ndarray, np.ndarray]) -> None:
    blurred = cv2.GaussianBlur(synth_photo[0], (0, 0), 5)
    report = assess(blurred)
    assert not report.acceptable
    assert "blurry" in report.issues[0]


def test_dark_and_overexposed_rejected() -> None:
    noise = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    dark = (noise * 0.05).astype(np.uint8)
    bright = np.full((400, 400, 3), 255, np.uint8)
    assert any("dark" in i for i in assess(dark, blur_threshold=0).issues)
    assert any("overexposed" in i for i in assess(bright, blur_threshold=0).issues)


def test_glare_is_warning_not_issue(synth_photo: tuple[np.ndarray, np.ndarray]) -> None:
    img = synth_photo[0].copy()
    cv2.circle(img, (600, 800), 180, (255, 255, 255), -1)
    report = assess(img)
    assert report.acceptable
    assert report.warnings and "glare" in report.warnings[0]


def test_resize_longest_never_upsamples() -> None:
    img = np.zeros((100, 50, 3), np.uint8)
    assert resize_longest(img, 1000).shape == img.shape
    assert max(resize_longest(np.zeros((2000, 1000, 3), np.uint8), 500).shape[:2]) == 500


# ---------- detection ----------


def test_order_corners_any_input_order() -> None:
    pts = np.array([[100, 90], [10, 5], [5, 95], [90, 10]], np.float32)
    ordered = order_corners(pts)
    np.testing.assert_array_equal(ordered, [[10, 5], [90, 10], [100, 90], [5, 95]])


def test_detection_matches_ground_truth(synth_photo: tuple[np.ndarray, np.ndarray]) -> None:
    photo, gt = synth_photo
    det = find_document(photo)
    diag = np.hypot(*photo.shape[:2])
    error = np.linalg.norm(det.corners - gt, axis=1).mean()
    assert det.method in {"edges", "threshold"}
    assert error < 0.01 * diag  # within 1% of the image diagonal
    assert 0.3 < det.coverage < 0.9


@pytest.mark.parametrize("seed", range(5))
def test_detection_robust_across_seeds(seed: int) -> None:
    from findoc.synth.photo import photograph, render_page

    rng = np.random.default_rng(1000 + seed)
    photo, gt = photograph(render_page(rng), rng)
    det = find_document(photo)
    assert np.linalg.norm(det.corners - gt, axis=1).mean() < 0.02 * np.hypot(*photo.shape[:2])


def test_threshold_fallback_when_edges_are_weak() -> None:
    # Soft-edged bright page: blurred so Canny finds no clean boundary.
    img = np.full((600, 800, 3), 70, np.uint8)
    cv2.fillPoly(img, [np.array([[200, 100], [600, 120], [580, 500], [220, 480]])], (235,) * 3)
    img = cv2.GaussianBlur(img, (0, 0), 12)
    det = find_document(img)
    assert det.method in {"edges", "threshold", "min_area_rect"}
    assert det.coverage > 0.2


def test_blank_image_falls_back_to_full_frame() -> None:
    det = find_document(np.full((300, 400, 3), 128, np.uint8))
    assert det.method == "full_frame"
    assert det.coverage == pytest.approx(1.0, abs=0.02)


def test_draw_detection_does_not_modify_input(synth_photo: tuple[np.ndarray, np.ndarray]) -> None:
    photo, gt = synth_photo
    before = photo.copy()
    out = draw_detection(photo, gt)
    assert np.array_equal(photo, before)
    assert not np.array_equal(out, photo)


# ---------- rectification ----------


def test_rectify_recovers_page_aspect(synth_photo: tuple[np.ndarray, np.ndarray]) -> None:
    photo, gt = synth_photo
    flat, homography = warp_document(photo, gt)
    h, w = flat.shape[:2]
    assert abs((w / h) - 850 / 1100) < 0.08  # rendered page is 850x1100
    corners_on_flat = map_points(gt, homography)
    np.testing.assert_allclose(
        corners_on_flat, [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], atol=1e-2
    )


def test_output_size_axis_aligned() -> None:
    corners = np.array([[0, 0], [199, 0], [199, 99], [0, 99]], np.float32)
    assert output_size(corners) == (199, 99)


# ---------- illumination ----------


def test_illumination_flattens_uneven_lighting() -> None:
    page = np.full((800, 600, 3), 240, np.uint8)
    for y in range(80, 760, 40):
        cv2.putText(
            page, "SAMPLE TEXT LINE 12345", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20,) * 3, 2
        )
    ramp = np.linspace(0.45, 1.0, 600, dtype=np.float32)[None, :, None]
    shaded = (page.astype(np.float32) * ramp).astype(np.uint8)
    fixed = normalize_illumination(shaded)
    assert background_unevenness(fixed) < 0.3 * background_unevenness(shaded)
    # Text stays dark relative to paper
    gray = cv2.cvtColor(fixed, cv2.COLOR_BGR2GRAY)
    assert np.percentile(gray, 2) < 120 and np.median(gray) > 230
