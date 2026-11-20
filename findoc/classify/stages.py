"""Phase 7 pipeline stage: document type from the flat page (when a model is trained)."""

from __future__ import annotations

from findoc.classify.doctype import load_model
from findoc.config import get_settings
from findoc.types import DocType, Document

MIN_CONFIDENCE = 0.6
TYPES = {"cheque": DocType.CHEQUE, "invoice": DocType.INVOICE, "receipt": DocType.RECEIPT}


def classify_stage(doc: Document) -> Document:
    img = doc.enhanced if doc.enhanced is not None else doc.flat
    if img is None:
        return doc
    model = load_model(get_settings().doctype_model)
    if model is None:
        return doc
    label, conf, probs = model.predict(img)
    doc.quality["doc_type_confidence"] = round(conf, 4)
    if conf >= MIN_CONFIDENCE and label in TYPES:
        doc.doc_type = TYPES[label]
    doc.messages.append(f"Document type: {label} ({conf:.0%})")
    return doc
