"""Keyframe selection from a tracked video (CO-4, FR-TRK-02/03).

Two consumers need different frames:

* the **2D pipeline** (Phases 1-2) wants the single **best frame**: sharp, document fully in
  view and large, viewed as head-on as possible;
* the **3D reconstruction** (Phase 4) wants a set of **keyframes**: all sharp, and seen from
  well-spread viewpoints, because depth comes from parallax between views.

**Viewpoint change without 3D.** Image displacement alone is a poor proxy: panning the camera
moves everything in the image but adds no parallax. What does change with viewpoint is the
document's *projective shape* (a rectangle seen from the side becomes a trapezoid). So we
remove in-plane motion by aligning the two sets of document corners with the best similarity
transform (translation + rotation + uniform scale), and measure what is left over:

    viewpoint_change = mean residual corner distance / document diagonal

Zero for pure panning, zooming or rolling; grows as the viewing angle changes.

**Selection** (greedy, in time order):

1. Usable frames: document tracked, fully inside the frame, and sharp: sharpness at least
   ``min_sharpness`` *and* at least ``rel_sharpness`` x the clip's median (blur is relative:
   a frame that is half as sharp as the typical frame is smeared by camera motion).
2. The first keyframe is the sharpest usable frame in the first part of the clip.
3. Scan forward: once a frame's viewpoint has changed by at least ``min_view_change`` from
   the last keyframe, choose the sharpest usable frame in a short window starting there.
4. Stop at ``max_keyframes``; if fewer than ``min_keyframes`` were found, relax the viewpoint
   threshold and repeat.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from findoc.track.klt import FrameTrack, TrackResult


@dataclass
class Keyframe:
    index: int
    sharpness: float
    view_change: float  # from the previous keyframe
    coverage: float


@dataclass
class KeyframeSelection:
    keyframes: list[Keyframe]
    best: int | None  # frame index for the 2D pipeline
    usable: int  # frames that passed the usability test
    min_view_change: float  # threshold actually used


def viewpoint_change(a: np.ndarray, b: np.ndarray) -> float:
    """Projective shape difference between two document outlines (similarity removed)."""
    m, _ = cv2.estimateAffinePartial2D(a.astype(np.float32), b.astype(np.float32), method=cv2.LMEDS)
    if m is None:
        return 0.0
    aligned = cv2.transform(a.reshape(-1, 1, 2).astype(np.float32), m).reshape(-1, 2)
    diag = np.linalg.norm(b[2] - b[0])
    return float(np.mean(np.linalg.norm(aligned - b, axis=1)) / max(diag, 1e-6))


def frontalness(corners: np.ndarray) -> float:
    """1.0 for a document seen head-on (opposite sides equal), lower when foreshortened."""
    tl, tr, br, bl = corners
    w_ratio = min(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)) / max(
        np.linalg.norm(tr - tl), np.linalg.norm(br - bl), 1e-6
    )
    h_ratio = min(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)) / max(
        np.linalg.norm(bl - tl), np.linalg.norm(br - tr), 1e-6
    )
    return float(w_ratio * h_ratio)


def usable_frames(
    track: TrackResult, min_sharpness: float, rel_sharpness: float = 0.75
) -> list[FrameTrack]:
    tracked = [f for f in track.frames if f.corners is not None and f.in_frame]
    if not tracked:
        return []
    cut = max(min_sharpness, rel_sharpness * float(np.median([f.sharpness for f in tracked])))
    return [f for f in tracked if f.sharpness >= cut]


def best_frame(frames: list[FrameTrack]) -> int | None:
    """Score = relative sharpness x frontalness x relative coverage."""
    if not frames:
        return None
    max_s = max(f.sharpness for f in frames)
    max_c = max(f.coverage for f in frames)
    scores = [
        (f.sharpness / max_s) * frontalness(f.corners) * (f.coverage / max_c)  # type: ignore[arg-type]
        for f in frames
    ]
    return frames[int(np.argmax(scores))].index


def select_keyframes(
    track: TrackResult,
    min_sharpness: float = 15.0,
    rel_sharpness: float = 0.75,
    min_view_change: float = 0.008,
    window: int = 6,
    min_keyframes: int = 8,
    max_keyframes: int = 25,
) -> KeyframeSelection:
    usable = usable_frames(track, min_sharpness, rel_sharpness)
    if not usable:
        return KeyframeSelection([], None, 0, min_view_change)
    by_index = {f.index: f for f in usable}
    order = [f.index for f in usable]

    threshold = min_view_change
    for _ in range(4):
        first_window = order[: max(window, len(order) // 6)]
        first = max(first_window, key=lambda i: by_index[i].sharpness)
        keys = [Keyframe(first, by_index[first].sharpness, 0.0, by_index[first].coverage)]
        pos = order.index(first) + 1
        while pos < len(order) and len(keys) < max_keyframes:
            last = by_index[keys[-1].index]
            idx = order[pos]
            change = viewpoint_change(last.corners, by_index[idx].corners)  # type: ignore[arg-type]
            if change >= threshold:
                candidates = [i for i in order[pos:] if i <= idx + window]
                pick = max(candidates, key=lambda i: by_index[i].sharpness)
                pick_change = viewpoint_change(last.corners, by_index[pick].corners)  # type: ignore[arg-type]
                keys.append(
                    Keyframe(pick, by_index[pick].sharpness, pick_change, by_index[pick].coverage)
                )
                pos = order.index(pick) + 1
            else:
                pos += 1
        if len(keys) >= min_keyframes or threshold < min_view_change / 6:
            break
        threshold /= 2
    return KeyframeSelection(keys, best_frame(usable), len(usable), threshold)
