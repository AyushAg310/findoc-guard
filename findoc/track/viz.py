"""Visual outputs for video input: keyframe contact sheet and annotated tracking video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from findoc.types import Document


def contact_sheet(
    images: list[np.ndarray], labels: list[str], cols: int = 4, width: int = 360
) -> np.ndarray:
    """Grid of thumbnails with a label on each."""
    if not images:
        return np.full((60, width, 3), 255, np.uint8)
    thumbs = []
    for img, label in zip(images, labels, strict=True):
        h, w = img.shape[:2]
        t = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
        cv2.putText(t, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(t, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        thumbs.append(t)
    th = max(t.shape[0] for t in thumbs)
    thumbs = [
        cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for t in thumbs
    ]
    while len(thumbs) % cols:
        thumbs.append(np.full_like(thumbs[0], 255))
    rows = [np.hstack(thumbs[i : i + cols]) for i in range(0, len(thumbs), cols)]
    return np.vstack(rows)


def keyframe_sheet(doc: Document) -> np.ndarray:
    images = []
    for kf, img in zip(doc.keyframes, doc.keyframe_images, strict=False):
        out = img.copy()
        cv2.polylines(out, [np.array(kf.corners, np.int32).reshape(-1, 1, 2)], True, (0, 220, 0), 4)
        images.append(out)
    labels = [f"#{k.frame}  sharp {k.sharpness:.0f}  dv {k.view_change:.3f}" for k in doc.keyframes]
    return contact_sheet(images, labels[: len(images)])


def write_tracking_video(doc: Document, path: Path) -> Path:
    """The tracking-size frames with the tracked outline; keyframes green, best frame blue,
    other frames yellow, lost frames marked."""
    if not doc.frames:
        raise ValueError("No video frames in this document")
    scale = doc.video.get("scale", 1.0)
    keys = {k.frame for k in doc.keyframes}
    fps = doc.video.get("fps", 30.0) / max(doc.video.get("stride", 1), 1)
    h, w = doc.frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame, tf in zip(doc.frames, doc.track, strict=False):
        out = frame.copy()
        if tf.corners is not None:
            if tf.frame == doc.best_frame:
                colour = (255, 120, 0)
            elif tf.frame in keys:
                colour = (0, 220, 0)
            else:
                colour = (0, 220, 255)
            pts = (np.array(tf.corners) * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, colour, 3)
        tag = "BEST" if tf.frame == doc.best_frame else "KEY" if tf.frame in keys else ""
        label = f"#{tf.frame} sharp {tf.sharpness:.0f} tracks {tf.tracks} {tag}"
        if tf.corners is None:
            label += " LOST"
        cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(out)
    writer.release()
    return path
