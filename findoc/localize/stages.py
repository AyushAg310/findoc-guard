"""Phase 2 pipeline stages: template identification/registration and field reading."""

from __future__ import annotations

from functools import partial

import numpy as np

from findoc.config import get_settings
from findoc.localize.align import identify, register, resize_to_width
from findoc.localize.templates import default_store
from findoc.recognize.fields import read_cheque_fields
from findoc.recognize.ocr import OcrUnavailableError, load_engine, note_engine_load
from findoc.types import DocType, Document

AMBIGUOUS_MARGIN = 1.3  # best template must beat the runner-up by 30% to be unambiguous


def align_stage(doc: Document) -> Document:
    """Is this a known cheque leaf? If so, warp it into the template's coordinate frame."""
    img = doc.enhanced if doc.enhanced is not None else doc.flat
    if img is None or doc.doc_type in (DocType.INVOICE, DocType.RECEIPT):  # classified (Phase 7)
        return doc
    s = get_settings()
    store = default_store(s.templates_dir)
    width = next(iter(store)).size[0]
    scaled, scale = resize_to_width(img, width)

    result = identify(scaled, store, method=s.feature_method)
    if result is None:
        doc.messages.append("No cheque template matched; treated as a general document.")
        return doc

    template = store.get(result.template_id)
    doc.doc_type = DocType.CHEQUE
    doc.template_id = template.id
    # flat-page pixels -> template pixels = (scaled -> template) @ (flat -> scaled)
    doc.template_homography = (result.homography @ scale).round(8).tolist()
    doc.registered = register(scaled, result.homography, template)

    runner_up = sorted(result.scores.values(), reverse=True)[1] if len(result.scores) > 1 else 0
    margin = result.inliers / max(runner_up, 1)
    doc.quality.update(
        {
            "template_inliers": result.inliers,
            "template_inlier_ratio": round(result.inlier_ratio, 4),
            "template_margin": round(margin, 2),
        }
    )
    doc.messages.append(
        f"Cheque template: {template.name} ({result.inliers} inliers, {margin:.1f}x runner-up)"
    )
    if margin < AMBIGUOUS_MARGIN:
        doc.messages.append("Template match is ambiguous; please confirm the bank.")
    return doc


def fields_stage(doc: Document, use_ocr: bool = True) -> Document:
    """Read every field of a registered cheque."""
    if doc.registered is None or doc.template_id is None:
        return doc
    template = default_store(get_settings().templates_dir).get(doc.template_id)
    engine = None
    if use_ocr:
        try:
            engine, load_ms = load_engine()
            note_engine_load(doc, engine, load_ms)
        except OcrUnavailableError as exc:
            doc.messages.append(f"Field OCR skipped: {exc}")
    doc.fields = read_cheque_fields(doc.registered, template, engine)
    doc.quality["fields_read_with_ocr"] = float(engine is not None)
    return doc


def make_fields_stage(use_ocr: bool) -> partial[Document]:
    return partial(fields_stage, use_ocr=use_ocr)


def registered_box_to_flat(doc: Document, box: tuple[int, int, int, int]) -> np.ndarray:
    """Map a template box to the flat page (for drawing fields on the user's image)."""
    import cv2

    assert doc.template_homography is not None
    inv = np.linalg.inv(np.array(doc.template_homography))
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, inv).reshape(-1, 2)
