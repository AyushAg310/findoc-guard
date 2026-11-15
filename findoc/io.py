"""Input loading helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from findoc.types import InputKind

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def detect_kind(path: Path) -> InputKind:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return InputKind.IMAGE
    if ext in VIDEO_EXTS:
        return InputKind.VIDEO
    raise ValueError(f"Unsupported file type: {ext}")


def load_image(path: Path) -> np.ndarray:
    """Read an image as BGR uint8. Works with non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    buf.tofile(str(path))
