"""Pictures of fraud evidence for reviewers."""

from __future__ import annotations

import cv2
import numpy as np


def tamper_overlay(img: np.ndarray, heat: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """``img`` with the tamper heatmap ([0, 1], same size) as a colour overlay where it is warm.

    Cool (low) areas are left untouched so the document stays readable; warm areas are tinted
    yellow to red in proportion to the heat."""
    h = np.clip(heat, 0, 1)
    colour = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    a = (alpha * np.clip((h - 0.2) / 0.6, 0, 1))[..., None]
    out = img.astype(np.float32) * (1 - a) + colour.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)
