"""Capture quality assessment (CO-1, FR-CAP-03).

Decides whether a photo is good enough to process, and if not, tells the user why.

Metrics (all computed on a copy resized so its longest side is ``ANALYSIS_SIDE`` pixels,
so thresholds don't depend on the camera's resolution):

* **sharpness**: variance of the Laplacian. The Laplacian is a second-derivative filter that
  responds to edges; a sharp image has strong edges, so its response varies a lot. Blur
  smooths edges away and the variance drops.
* **dark_fraction / bright_fraction**: share of pixels that are nearly black (< 20) or fully
  saturated (> 250) in grayscale. Large values mean under- or over-exposure.
* **glare_fraction**: share of pixels saturated in *all three* colour channels and low in
  colour saturation, i.e. specular highlights from a lamp or flash.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

ANALYSIS_SIDE = 1000


@dataclass
class QualityReport:
    sharpness: float
    dark_fraction: float
    bright_fraction: float
    glare_fraction: float
    score: float
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def acceptable(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, float]:
        return {
            "sharpness": round(self.sharpness, 2),
            "dark_fraction": round(self.dark_fraction, 4),
            "bright_fraction": round(self.bright_fraction, 4),
            "glare_fraction": round(self.glare_fraction, 4),
            "score": round(self.score, 3),
        }


def resize_longest(img: np.ndarray, side: int) -> np.ndarray:
    """Resize so the longest side equals ``side`` (never upsamples)."""
    h, w = img.shape[:2]
    scale = side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess(
    img: np.ndarray,
    blur_threshold: float = 100.0,
    max_dark: float = 0.5,
    max_bright: float = 0.5,
    glare_warn: float = 0.02,
) -> QualityReport:
    """Score a BGR photo and list blocking issues and non-blocking warnings."""
    small = resize_longest(img, ANALYSIS_SIDE)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    sharp = sharpness(gray)
    dark = float(np.mean(gray < 20))
    bright = float(np.mean(gray > 250))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    all_saturated = np.all(small >= 250, axis=2)
    glare = float(np.mean(all_saturated & (hsv[..., 1] < 30)))

    issues: list[str] = []
    warnings: list[str] = []
    if sharp < blur_threshold:
        issues.append(
            f"Image is blurry (sharpness {sharp:.0f} < {blur_threshold:.0f}). "
            "Hold the phone steady and retake."
        )
    if dark > max_dark:
        issues.append(f"Image is too dark ({dark:.0%} of pixels nearly black). Add light.")
    if bright > max_bright:
        issues.append(f"Image is overexposed ({bright:.0%} of pixels saturated).")
    if glare > glare_warn:
        warnings.append(f"Possible glare on {glare:.1%} of the image; tilt to avoid reflections.")

    # 0..1 summary score: sharpness dominates, exposure and glare subtract.
    s_sharp = min(1.0, sharp / (2 * max(blur_threshold, 1e-6)))
    s_exposure = max(0.0, 1.0 - dark - bright)
    score = max(0.0, 0.6 * s_sharp + 0.4 * s_exposure - min(glare * 5, 0.3))

    return QualityReport(sharp, dark, bright, glare, score, issues, warnings)
