"""Phase 7 pipeline stage: invoice fields, line items and invoice checks (after OCR).

**Second look at GSTINs.** A GSTIN that fails its check character after the page-wide OCR is
read again from its own crop, enlarged, with only 0-9 and A-Z allowed (no detection step, so
nothing else on the line interferes). If that reading is a valid GSTIN close to the first
one, it replaces it: the first reading was an OCR error. A printed GSTIN that really is
invalid stays invalid however often it is read, so this does not hide the fraud.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from findoc.config import get_settings
from findoc.fraud.invoice_rules import run_invoice_rules
from findoc.recognize import gstin as gst
from findoc.recognize.invoice import Invoice, RuleExtractor, amount_in_words_text
from findoc.types import BBox, DocType, Document, ExtractedField, OcrWord

INVOICE_CUES = re.compile(r"tax\s*invoice|gstin|\bhsn\b|\bcgst\b|\bigst\b")


@lru_cache(maxsize=1)
def _layout_extractor(path: Path):  # type: ignore[no-untyped-def]  # noqa: ANN202
    from findoc.recognize.layout import LayoutExtractor

    return LayoutExtractor(path)


def looks_like_invoice(text: str) -> bool:
    """At least two invoice cues ("Tax Invoice", GSTIN, HSN, CGST, IGST) in the OCR text."""
    return len(set(INVOICE_CUES.findall(text.lower()))) >= 2


def extract_invoice(doc: Document) -> Invoice:
    s = get_settings()
    use_layout = s.invoice_extractor == "layoutlmv3" or (
        s.invoice_extractor == "auto" and (s.invoice_model / "config.json").is_file()
    )
    if use_layout:
        img = doc.enhanced if doc.enhanced is not None else doc.flat
        try:
            return _layout_extractor(s.invoice_model).extract(doc.words, img)
        except ImportError as exc:  # transformers not installed: say so and use the rules
            doc.messages.append(f"LayoutLMv3 unavailable ({exc}); rule-based extraction used")
    return RuleExtractor().extract(doc.words)


GSTIN_FIELDS = ("vendor_gstin", "buyer_gstin")
ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _alnum(text: str) -> str:
    return re.sub(r"[^0-9A-Z]", "", text.upper())


def word_containing(words: list[OcrWord], value: str) -> OcrWord | None:
    """The OCR word (line) the extracted value came from: exact containment first, else the
    most similar word (the extractor may have repaired a few characters)."""
    target = _alnum(value)
    if not target:
        return None
    best, best_ratio = None, 0.75
    for w in words:
        t = _alnum(w.text)
        if target in t:
            return w
        ratio = SequenceMatcher(None, target, t[-len(target) :]).ratio()
        if ratio > best_ratio:
            best, best_ratio = w, ratio
    return best


def reread_gstin(engine: object, img: np.ndarray, box: BBox, first: str) -> str | None:
    """Read the GSTIN in ``box`` again at 2x and 3x; return a valid GSTIN that differs from
    the first reading in at most 4 places, or None."""
    h = box.y1 - box.y0
    y0, y1 = int(max(0, box.y0 - 0.15 * h)), int(min(img.shape[0], box.y1 + 0.15 * h))
    x0, x1 = int(max(0, box.x0 - 4)), int(min(img.shape[1], box.x1 + 4))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    first = gst.clean(_alnum(first))
    for scale in (2, 3):
        big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        text, _ = engine.read_line(big, allowlist=ALNUM)  # type: ignore[attr-defined]
        t = _alnum(text)
        if len(t) < 15:
            continue
        valid, _ = gst.validate(t[-15:])
        if valid and (
            len(first) != 15 or sum(a != b for a, b in zip(valid, first, strict=False)) <= 4
        ):
            return valid
    return None


def second_look_gstins(doc: Document, inv: Invoice) -> None:
    """Re-read each GSTIN that fails validation (see the module docstring)."""
    todo = [k for k in GSTIN_FIELDS if inv.fields.get(k) and not gst.validate(inv.fields[k])[0]]
    img = doc.enhanced if doc.enhanced is not None else doc.flat
    if not todo or img is None:
        return
    from findoc.recognize.ocr import OcrUnavailableError, load_engine

    try:
        engine, _ = load_engine()
    except OcrUnavailableError:
        return
    for k in todo:
        word = word_containing(doc.words, inv.fields[k])
        if word is None:
            continue
        new = reread_gstin(engine, img, word.bbox, inv.fields[k])
        if new:
            doc.messages.append(f"{k}: '{inv.fields[k]}' re-read as {new} (valid)")
            inv.fields[k] = new


def invoice_stage(doc: Document) -> Document:
    if doc.doc_type is DocType.CHEQUE or not doc.words:
        return doc
    if doc.doc_type is not DocType.INVOICE:
        if not looks_like_invoice(doc.full_text):
            return doc
        doc.doc_type = DocType.INVOICE
    inv = extract_invoice(doc)
    second_look_gstins(doc, inv)
    words_total = amount_in_words_text(doc.words)
    if words_total and "amount_words" not in inv.fields:
        inv.fields["amount_words"] = words_total
    norm = inv.normalized()
    conf = sum(w.confidence for w in doc.words) / len(doc.words)
    doc.fields = [
        ExtractedField(
            name=k,
            value=v,
            raw=v,
            normalized=norm["fields"].get(k),
            method=inv.method,
            confidence=round(conf, 4),
        )
        for k, v in inv.fields.items()
    ]
    doc.line_items = [
        {"row": i + 1, **{k: v for k, v in it.items()}, "normalized": norm["items"][i]}
        for i, it in enumerate(inv.items)
    ]
    checks = run_invoice_rules(norm)
    doc.checks.extend(checks)
    fired = [c.id for c in checks if c.fired]
    doc.messages.append(
        f"Invoice ({inv.method}): {len(inv.fields)} fields, {len(inv.items)} line items; "
        f"checks flagged {fired or 'nothing'}"
    )
    return doc
