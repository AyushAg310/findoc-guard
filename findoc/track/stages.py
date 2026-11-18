"""Phase 3 pipeline stages: read video, track the document, choose keyframes and best frame.

After :func:`track_stage`, ``doc.original`` is the best frame at full resolution and
``doc.corners`` its tracked document outline, so the Phase 1-2 stages (quality, rectify,
align, fields, rules) run on video exactly as they do on a photo.
"""

from __future__ import annotations

import time

import numpy as np

from findoc.config import get_settings
from findoc.io import read_frames, read_video, video_info
from findoc.track.keyframes import select_keyframes
from findoc.track.klt import MIN_FOR_HOMOGRAPHY, TrackResult, track_document
from findoc.types import Document, InputKind, Keyframe, Status, TrackedFrame


def video_load_stage(doc: Document) -> Document:
    if doc.kind is not InputKind.VIDEO:
        return doc
    s = get_settings()
    info = video_info(doc.source)
    frames, numbers, scale = read_video(doc.source, s.video_max_frames, s.video_track_side)
    doc.frames, doc.frame_numbers = frames, numbers
    stride = numbers[1] - numbers[0] if len(numbers) > 1 else 1
    doc.video = info | {"used": len(frames), "stride": stride, "scale": round(scale, 6)}
    doc.messages.append(
        f"Loaded video {info['width']}x{info['height']} @ {info['fps']:.0f} fps, "
        f"{len(frames)} frames for tracking (every {stride}), scale {scale:.2f}"
    )
    return doc


def rejection_reasons(result: TrackResult) -> list[str]:
    """Explain why no frame was usable, most fundamental cause first."""
    frames = result.frames
    found = [f for f in frames if f.corners is not None]
    reasons = ["No usable frame: none was sharp with the whole document in view."]
    if found and np.mean([f.n_tracks for f in frames]) < MIN_FOR_HOMOGRAPHY:
        reasons.append(
            "The document has too little detail to track (a blank page has nothing for optical "
            "flow to follow). Film a cheque or a page with printed text."
        )
    if result.start_problem == "partial" or (found and not any(f.in_frame for f in found)):
        reasons.append(
            "The document never fits fully inside the frame. Hold the phone farther away so all "
            "four corners and a little background are visible."
        )
    if result.start_problem == "partial":
        reasons.append(
            "For a cheque, hold the phone in landscape (sideways): a cheque is about twice as wide "
            "as it is tall, so in portrait it rarely fits."
        )
    elif not found:
        reasons.append(
            "No document outline was found. Use a background that contrasts with the paper."
        )
    if len(reasons) == 1:
        reasons.append("Most frames are blurred. Move the phone slowly and keep it steady.")
    return reasons


def _round(c: np.ndarray, inv_scale: float) -> list[list[float]]:
    return (c * inv_scale).round(2).tolist()


def track_stage(doc: Document) -> Document:
    if doc.kind is not InputKind.VIDEO or not doc.frames:
        return doc
    s = get_settings()
    inv = 1.0 / doc.video.get("scale", 1.0)
    t0 = time.perf_counter()
    result = track_document(doc.frames)
    ms = (time.perf_counter() - t0) * 1000 / len(doc.frames)
    sel = select_keyframes(
        result, min_view_change=s.keyframe_min_view_change, max_keyframes=s.keyframe_max
    )
    num = doc.frame_numbers
    doc.track = [
        TrackedFrame(
            frame=num[f.index],
            corners=None if f.corners is None else _round(f.corners, inv),
            tracks=f.n_tracks,
            motion_px=round(f.motion_px * inv, 2),
            sharpness=round(f.sharpness, 1),
            in_frame=f.in_frame,
        )
        for f in result.frames
    ]
    by_index = {f.index: f for f in result.frames}
    doc.keyframes = [
        Keyframe(
            frame=num[k.index],
            sharpness=round(k.sharpness, 1),
            view_change=round(k.view_change, 5),
            corners=_round(by_index[k.index].corners, inv),  # type: ignore[arg-type]
        )
        for k in sel.keyframes
    ]
    doc.tracking = {
        "ms_per_frame": round(ms, 2),
        "lost": result.lost,
        "redetections": result.redetections,
        "reacquisitions": result.reacquisitions,
        "snaps": result.snaps,
        "usable": sel.usable,
        "keyframes": len(sel.keyframes),
        "view_threshold": round(sel.min_view_change, 5),
        "mean_tracks": round(float(np.mean([f.n_tracks for f in result.frames])), 1),
    }
    if sel.best is None:
        doc.status = Status.REJECTED_QUALITY
        doc.messages.extend(rejection_reasons(result))
        return doc

    doc.best_frame = num[sel.best]
    wanted = sorted({doc.best_frame, *(k.frame for k in doc.keyframes)})
    full = read_frames(doc.source, wanted)
    doc.keyframe_images = [full[k.frame] for k in doc.keyframes if k.frame in full]
    doc.original = full.get(doc.best_frame)
    if doc.original is None:  # second pass failed; fall back to the tracking-size frame
        doc.original = doc.frames[sel.best]
        inv = 1.0
    doc.corners = _round(by_index[sel.best].corners, inv)  # type: ignore[arg-type]
    doc.detection_method = "tracked"
    doc.messages.append(
        f"Tracked {len(result.frames) - result.lost}/{len(result.frames)} frames "
        f"({ms:.1f} ms/frame); {len(doc.keyframes)} keyframes; best frame #{doc.best_frame}"
    )
    return doc
