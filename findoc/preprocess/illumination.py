"""Shadow removal and illumination normalisation (CO-1, FR-CAP-05).

Phone photos of paper have uneven lighting: a hand's shadow, a lamp on one side. We model
each colour channel as ``observed = reflectance x illumination`` and estimate the slowly
varying illumination (the paper's background brightness) by removing the text:

1. **Dilate** with a kernel larger than a stroke: bright paper spreads over thin dark text.
2. **Median blur** with a large kernel: smooths what is left into a background estimate.
3. **Divide** the channel by the background and rescale: shadows (low background) are lifted,
   bright areas are left alone, and the paper becomes uniformly white.

Kernel sizes scale with the image so the method works at any resolution.
"""

from __future__ import annotations

import cv2
import numpy as np


def _odd(n: float, minimum: int = 3) -> int:
    k = max(minimum, int(n))
    return k if k % 2 == 1 else k + 1


def estimate_background(channel: np.ndarray) -> np.ndarray:
    side = max(channel.shape[:2])
    dil = cv2.dilate(channel, np.ones((_odd(side / 150), _odd(side / 150)), np.uint8))
    return cv2.medianBlur(dil, min(_odd(side / 30), 255))


def normalize_illumination(img: np.ndarray) -> np.ndarray:
    """Return a copy of a BGR image with uneven lighting and shadows removed."""
    channels = []
    for ch in cv2.split(img):
        bg = estimate_background(ch).astype(np.float32)
        norm = ch.astype(np.float32) / np.maximum(bg, 1.0) * 255.0
        channels.append(np.clip(norm, 0, 255).astype(np.uint8))
    return cv2.merge(channels)


def background_unevenness(img: np.ndarray) -> float:
    """Std-dev of the estimated background brightness (lower = more even lighting)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(np.std(estimate_background(gray)))
