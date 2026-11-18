"""Document boundary detection (CO-1, FR-CAP-04).

Finds the four corners of the page in a photo.

0. **learned** (Phase 4): if a trained segmentation model is available
   (``FINDOC_SEGMENTER_MODEL``, default ``models/docseg.onnx``), a small U-Net predicts which
   pixels are document; its mask is turned into a quadrilateral and refined on the image edges
   (:mod:`findoc.segment.infer`). Accepted when the mask agrees with the quadrilateral
   (IoU >= ``LEARNED_MIN_AGREEMENT``). Works on backgrounds as bright or busy as the paper.

Otherwise (no model, ``FINDOC_SEGMENTATION=off``, or low agreement), the classical methods are
tried in order, then two fallbacks:

1. **edges**: Canny edge map -> close small gaps -> external contours -> the largest contour
   that simplifies (Douglas-Peucker, ``approxPolyDP``) to a convex quadrilateral.
2. **threshold**: Otsu threshold separates the bright page from a darker background; the
   largest blob is simplified the same way. Works when the page edge is low-contrast but the
   page is brighter than the desk.
3. **min_area_rect**: the minimum-area rotated rectangle around the largest blob, when no
   clean quadrilateral exists (e.g. a corner is folded or hidden).
4. **full_frame**: assume the whole image is the document (flatbed scans, screenshots).

Detection runs on a downscaled copy (``WORK_SIDE`` px) for speed; corners are scaled back.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

WORK_SIDE = 800
MIN_AREA_RATIO = 0.15  # a candidate must cover at least 15% of the frame
FULL_FRAME_RATIO = 0.97  # a quad covering more than this is the frame itself
LEARNED_MIN_AGREEMENT = 0.9  # mask vs. quadrilateral IoU needed to trust the learned outline
LEARNED_MIN_AREA = 0.04  # the learned detector may find smaller documents than the classical


@dataclass
class Detection:
    corners: np.ndarray  # (4, 2) float32, order tl, tr, br, bl, original-image pixels
    method: str
    coverage: float  # quadrilateral area / image area


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return 4 points ordered top-left, top-right, bottom-right, bottom-left.

    Top-left has the smallest x+y, bottom-right the largest; top-right has the smallest
    y-x, bottom-left the largest.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()  # y - x
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def quad_area(corners: np.ndarray) -> float:
    return float(cv2.contourArea(corners.astype(np.float32)))


def _auto_canny(gray: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """Canny with thresholds set around the median intensity (no manual tuning)."""
    v = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(gray, lower, upper)


def _best_quad(mask: np.ndarray, min_area: float) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(c) < min_area:
            break
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.03, 0.05):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype(np.float32)
    return None


def _largest_blob(mask: np.ndarray, min_area: float) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None
    return cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float32)


def _segmenter():  # noqa: ANN202
    from findoc.config import get_settings
    from findoc.segment.infer import load_segmenter

    s = get_settings()
    if s.segmentation == "off":
        return None
    return load_segmenter(str(s.segmenter_model))


def find_document(img: np.ndarray) -> Detection:
    """Learned segmentation when a model is available and confident, else classical."""
    seg_model = _segmenter()
    if seg_model is not None:
        seg = seg_model.segment(img)
        if seg.corners is not None and seg.agreement >= LEARNED_MIN_AGREEMENT:
            h, w = img.shape[:2]
            corners = order_corners(seg.corners)
            coverage = quad_area(corners) / float(h * w)
            if LEARNED_MIN_AREA <= coverage <= FULL_FRAME_RATIO:
                return Detection(corners=corners, method="learned", coverage=coverage)
    return find_document_classical(img)


def find_document_classical(img: np.ndarray) -> Detection:
    h, w = img.shape[:2]
    scale = min(1.0, WORK_SIDE / max(h, w))
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    min_area = MIN_AREA_RATIO * sh * sw

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 1. Edge-based
    edges = _auto_canny(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    quad, method = _best_quad(edges, min_area), "edges"

    # 2. Threshold-based (bright page on darker background)
    if quad is None:
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        quad, method = _best_quad(mask, min_area), "threshold"
        # 3. Rotated rectangle around the largest blob
        if quad is None:
            quad, method = _largest_blob(mask, min_area), "min_area_rect"

    # 4. Whole frame
    if quad is None:
        quad = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], np.float32)
        method = "full_frame"

    corners = order_corners(quad / scale)
    coverage = quad_area(corners) / float(h * w)
    if method != "full_frame" and coverage > FULL_FRAME_RATIO:
        method = "full_frame"  # the "page" is the whole image: a scan or a uniform frame
    return Detection(corners=corners, method=method, coverage=min(coverage, 1.0))


def draw_detection(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Return a copy of ``img`` with the detected quadrilateral and corner labels drawn."""
    out = img.copy()
    thickness = max(2, round(max(img.shape[:2]) / 400))
    pts = corners.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(out, [pts], True, (0, 200, 0), thickness)
    for label, (x, y) in zip(("TL", "TR", "BR", "BL"), corners, strict=True):
        cv2.circle(out, (int(x), int(y)), thickness * 4, (0, 0, 255), -1)
        cv2.putText(
            out,
            label,
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            thickness * 0.5,
            (0, 0, 255),
            thickness,
        )
    return out
