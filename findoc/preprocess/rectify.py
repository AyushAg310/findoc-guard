"""Perspective correction (CO-1, FR-CAP-04).

A flat page photographed at an angle is related to its front-on view by a **homography**:
a 3x3 matrix H that maps each image point (x, y, 1) to page coordinates up to scale.
Four point correspondences (the page corners) determine H exactly
(``cv2.getPerspectiveTransform``), and ``cv2.warpPerspective`` resamples the image.

The homography is kept in the result: later phases use it to map boxes and evidence
between the original photo and the flat page (the 2D version of the Phase 4
correspondence map).
"""

from __future__ import annotations

import cv2
import numpy as np


def output_size(corners: np.ndarray) -> tuple[int, int]:
    """Width and height of the flat page, from the longer of each pair of opposite edges."""
    tl, tr, br, bl = corners
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    return max(1, round(width)), max(1, round(height))


def warp_document(img: np.ndarray, corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the flattened page and the homography (original -> flat)."""
    w, h = output_size(corners)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    flat = cv2.warpPerspective(
        img, homography, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return flat, homography


def map_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Apply a homography to an (N, 2) array of points."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)
