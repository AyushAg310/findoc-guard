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


def video_info(path: Path) -> dict[str, float]:
    """Frame count, fps and size of a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def read_video(
    path: Path, max_frames: int = 180, max_side: int = 960
) -> tuple[list[np.ndarray], list[int], float]:
    """First pass: read frames for tracking, streaming so memory stays bounded.

    Every ``stride``-th frame is kept (stride chosen so at most ``max_frames`` are kept) and
    downscaled so the longest side is at most ``max_side``. Returns (frames, original frame
    numbers, scale), where scale = tracking size / original size.
    """
    info = video_info(path)
    total = info["frames"] if info["frames"] > 0 else max_frames
    stride = max(1, int(np.ceil(total / max_frames)))
    scale = min(1.0, max_side / max(info["width"], info["height"], 1))
    cap = cv2.VideoCapture(str(path))
    frames, numbers, i = [], [], 0
    while len(frames) < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if i % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(frame)
            numbers.append(i)
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"No frames could be read from {path}")
    return frames, numbers, scale


def read_frames(path: Path, numbers: list[int]) -> dict[int, np.ndarray]:
    """Second pass: read selected frames at full resolution (sequential grab, no seeking,
    because seeking is unreliable for many phone codecs)."""
    wanted = set(numbers)
    out: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    i = 0
    while wanted - out.keys():
        if not cap.grab():
            break
        if i in wanted:
            ok, frame = cap.retrieve()
            if ok:
                out[i] = frame
        i += 1
    cap.release()
    return out
