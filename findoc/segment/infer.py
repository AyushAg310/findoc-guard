"""Run the document segmenter (ONNX through OpenCV DNN) and turn its mask into four corners.

No PyTorch here: the trained network is exported to ONNX and executed by ``cv2.dnn``, which
ships with OpenCV. The app stays light, and CI and Docker run without PyTorch.

From mask to corners:

1. **Predict** P(document) for each pixel of a 256x256 copy of the image, upsample to the
   working size and threshold at 0.5.
2. **Largest region**: keep the biggest connected component (other papers in view are smaller)
   and fill its holes (dark print or shading can leave holes; a document is solid).
3. **Quadrilateral**: approximate the region's convex hull with Douglas-Peucker, increasing the
   tolerance until 4 vertices remain; fall back to the minimum-area rectangle.
4. **Edge refinement**: the network sees a 256 px image, so its outline can be a few pixels
   off at full resolution. Each side is refined on the real image: at points along the side, we
   search across it (within a few pixels) for the strongest brightness step, fit a line to those
   points robustly (``cv2.fitLine`` with the Huber loss, which down-weights outliers such as a
   thumb or text near the edge), and intersect neighbouring lines to get the corners. Sides
   lying along the image border are left alone: the page continues past the edge there, so the
   nearest "edge" would be a printed line inside it.
5. **Agreement score**: IoU between the mask and the final quadrilateral. A real document mask is
   close to a quadrilateral (score near 1); a blob that isn't (two merged papers, a failure) is
   not, and the caller falls back to the classical detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

INPUT_SIZE = 256


@dataclass
class Segmentation:
    corners: np.ndarray | None  # (4, 2) tl, tr, br, bl in the input image's pixels
    mask: np.ndarray  # float32 P(document) at the input image's size
    agreement: float  # IoU between the thresholded mask and the quadrilateral (0 if none)


def to_blob(img: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """BGR uint8 -> (1, 3, size, size) float32 in [-0.5, 0.5] (same as training)."""
    x = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 - 0.5
    return x.transpose(2, 0, 1)[None]


def solid_region(mask: np.ndarray) -> np.ndarray | None:
    """The largest connected region of a binary mask with its holes filled (a document is
    solid: holes come from dark print or shading the network was unsure about)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == best).astype(np.uint8) * 255
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(region)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def mask_to_quad(mask: np.ndarray) -> np.ndarray | None:
    """Binary mask (uint8 0/255) -> 4 corners of its largest region, or None."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    region = (labels == best).astype(np.uint8) * 255
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    peri = cv2.arcLength(hull, True)
    for eps in (0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)


def _line_intersection(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    (vx1, vy1, x1, y1), (vx2, vy2, x2, y2) = l1, l2
    a = np.array([[vx1, -vx2], [vy1, -vy2]], np.float64)
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    t = np.linalg.solve(a, np.array([x2 - x1, y2 - y1], np.float64))
    return np.array([x1 + t[0] * vx1, y1 + t[0] * vy1], np.float32)


def _on_border(p: np.ndarray, q: np.ndarray, w: int, h: int, tol: float) -> bool:
    """Both ends of side p-q lie along the same image border."""
    for axis, limit in ((0, 0), (0, w - 1), (1, 0), (1, h - 1)):
        if abs(p[axis] - limit) <= tol and abs(q[axis] - limit) <= tol:
            return True
    return False


def refine_quad(gray: np.ndarray, quad: np.ndarray, band: int = 6, samples: int = 40) -> np.ndarray:
    """Snap each side of ``quad`` (ordered tl, tr, br, bl) to the strongest nearby edge."""
    g = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    h, w = gray.shape[:2]
    centre = quad.mean(axis=0)
    lines = []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        if _on_border(p, q, w, h, band + 2):
            lines.append(None)  # the page continues past the image edge: nothing to snap to
            continue
        d = q - p
        length = float(np.linalg.norm(d))
        if length < 10:
            return quad
        d /= length
        nrm = np.array([-d[1], d[0]], np.float32)
        if np.dot(centre - p, nrm) < 0:
            nrm = -nrm  # point inward
        pts = []
        for t in np.linspace(0.1, 0.9, samples):  # avoid the corners themselves
            base = p + d * t * length
            offs = np.arange(-band, band + 1, dtype=np.float32)
            cand = base[None] + offs[:, None] * nrm[None]
            xi = np.clip(cand[:, 0].round().astype(int), 0, w - 1)
            yi = np.clip(cand[:, 1].round().astype(int), 0, h - 1)
            strength = np.abs(gx[yi, xi] * nrm[0] + gy[yi, xi] * nrm[1])  # gradient across side
            k = int(np.argmax(strength))
            if strength[k] > 8.0:
                pts.append(cand[k])
        if len(pts) < samples // 3:
            lines.append(None)
            continue
        lines.append(cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
    out = quad.copy()
    for i in range(4):  # corner i is where side i-1 (into it) meets side i (out of it)
        a, b = lines[i - 1], lines[i]
        if a is None or b is None:
            continue
        x = _line_intersection(a, b)
        if x is not None and np.linalg.norm(x - quad[i]) <= 2.5 * band:
            out[i] = x
    return out


class Segmenter:
    """Document segmenter running an ONNX model with OpenCV DNN."""

    def __init__(self, model_path: Path, work_side: int = 800) -> None:
        self.path = Path(model_path)
        self.net = cv2.dnn.readNetFromONNX(str(self.path))
        self.work_side = work_side

    def probability(self, img: np.ndarray) -> np.ndarray:
        """P(document) for every pixel, at the image's size."""
        self.net.setInput(to_blob(img))
        logits = self.net.forward()[0, 0]
        prob = 1.0 / (1.0 + np.exp(-logits))
        h, w = img.shape[:2]
        return cv2.resize(prob.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    def segment(self, img: np.ndarray, refine: bool = True) -> Segmentation:
        from findoc.preprocess.detect import order_corners

        h, w = img.shape[:2]
        scale = min(1.0, self.work_side / max(h, w))
        small = (
            img
            if scale == 1.0
            else cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        )
        prob = self.probability(small)
        binary = solid_region((prob > 0.5).astype(np.uint8) * 255)
        quad = None if binary is None else mask_to_quad(binary)
        if binary is None or quad is None:
            return Segmentation(None, cv2.resize(prob, (w, h)), 0.0)
        quad = order_corners(quad)
        if refine:
            quad = refine_quad(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), quad)
        poly = np.zeros_like(binary)
        cv2.fillConvexPoly(poly, quad.round().astype(np.int32), 255)
        inter = np.count_nonzero(poly & binary)
        union = np.count_nonzero(poly | binary)
        agreement = inter / union if union else 0.0
        return Segmentation(quad / scale, cv2.resize(prob, (w, h)), float(agreement))


@lru_cache(maxsize=2)
def load_segmenter(model_path: str) -> Segmenter | None:
    """The segmenter for ``model_path``, or None if the file doesn't exist or can't be read."""
    path = Path(model_path)
    if not path.is_file():
        return None
    try:
        return Segmenter(path)
    except cv2.error:
        return None
