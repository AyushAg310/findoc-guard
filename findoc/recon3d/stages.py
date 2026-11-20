"""Phase 5 pipeline stage: 3D reconstruction and flattening of video input.

Runs after tracking. On success the page is flattened from its 3D shape (``doc.flat``,
``doc.flattening = "3d"``) and the 2D rectification of Phase 1 is skipped; the cheque stages
then run on the 3D-flattened page. Otherwise (too few keyframes, SfM failure, or a page that is
flat anyway) the reason is recorded and the 2D path runs as before.
"""

from __future__ import annotations

import numpy as np

from findoc.config import get_settings
from findoc.recon3d.pipeline import flatten_document
from findoc.types import Document, InputKind

MAX_VIEWS = 10


def recon3d_stage(doc: Document) -> Document:
    if doc.kind is not InputKind.VIDEO or doc.original is None or doc.corners is None:
        return doc
    if get_settings().recon3d == "off":
        return doc
    # keyframes (at most MAX_VIEWS, evenly spread) plus the best frame as the reference view
    kf = list(zip(doc.keyframes, doc.keyframe_images, strict=False))
    kf = [p for p in kf if p[0].frame != doc.best_frame]
    if len(kf) > MAX_VIEWS - 1:
        pick = np.linspace(0, len(kf) - 1, MAX_VIEWS - 1).round().astype(int)
        kf = [kf[i] for i in pick]
    images = [doc.original] + [img for _, img in kf]
    if any(img.shape != doc.original.shape for img in images):
        doc.recon3d = {"ok": False, "reason": "keyframes differ in size from the best frame"}
        return doc
    quad = np.array(doc.corners, np.float32)
    result = flatten_document(images, 0, quad, focal_guess=get_settings().focal_px)
    frames = [doc.best_frame] + [k.frame for k, _ in kf]
    doc.recon3d = {"ok": result.ok, "reason": result.reason, "frames": frames, **result.stats}
    if result.ok and result.flat is not None:
        doc.flat = result.flat
        doc.flattening = "3d"
        doc.flat_map = (result.map_x, result.map_y)
        doc.recon = result
        doc.messages.append(
            f"3D: {result.stats['registered']}/{result.stats['views']} views, "
            f"{result.stats['doc_points']} page points, bend {result.stats['planarity']:.1%} of "
            f"the diagonal; page flattened from 3D (ARAP stretch {result.stats['stretch']:.2%})"
        )
    else:
        doc.recon = result if result.rec is not None else None
        doc.messages.append(f"3D: {result.reason}")
    return doc
