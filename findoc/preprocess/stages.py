"""Phase 1 pipeline stages: quality gate, detection, rectification, enhancement.

Each function has the stage signature ``(Document) -> Document`` and is registered in
:func:`findoc.pipeline.default_pipeline`.
"""

from __future__ import annotations

import cv2
import numpy as np

from findoc.config import get_settings
from findoc.preprocess.detect import find_document
from findoc.preprocess.illumination import normalize_illumination
from findoc.preprocess.quality import assess
from findoc.preprocess.rectify import warp_document
from findoc.types import Document, Status


def _require_image(doc: Document) -> bool:
    """Stages need an image: a photo, or the best video frame chosen by the track stage."""
    return doc.original is not None


def quality_stage(doc: Document) -> Document:
    if not _require_image(doc):
        return doc
    s = get_settings()
    report = assess(doc.original, blur_threshold=s.blur_threshold)
    doc.quality.update(report.as_dict())
    doc.messages.extend(report.warnings)
    if not report.acceptable:
        doc.status = Status.REJECTED_QUALITY
        doc.messages.extend(report.issues)
    return doc


def detect_stage(doc: Document) -> Document:
    if not _require_image(doc):
        return doc
    if doc.detection_method == "tracked" and doc.corners is not None:
        # Video: the tracker already located the document in the best frame.
        h, w = doc.original.shape[:2]
        area = cv2.contourArea(np.array(doc.corners, dtype=np.float32))
        doc.quality["coverage"] = round(area / (w * h), 4)
        return doc
    det = find_document(doc.original)
    doc.corners = det.corners.round(2).tolist()
    doc.detection_method = det.method
    doc.quality["coverage"] = round(det.coverage, 4)

    if det.method == "full_frame":
        doc.messages.append("No page boundary found; treating the whole image as the document.")
    elif det.coverage < get_settings().min_coverage:
        doc.status = Status.REJECTED_QUALITY
        doc.messages.append(
            f"Document fills only {det.coverage:.0%} of the frame. Move closer and retake."
        )
    return doc


def rectify_stage(doc: Document) -> Document:
    if not _require_image(doc) or doc.corners is None:
        return doc
    flat, homography = warp_document(doc.original, np.array(doc.corners, dtype=np.float32))
    doc.flat = flat
    doc.homography = homography.round(8).tolist()
    h, w = flat.shape[:2]
    doc.messages.append(f"Flattened page: {w}x{h}px ({doc.detection_method})")
    return doc


def enhance_stage(doc: Document) -> Document:
    if doc.flat is None:
        return doc
    doc.enhanced = normalize_illumination(doc.flat)
    return doc
