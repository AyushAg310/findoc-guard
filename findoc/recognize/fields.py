"""Cheque field extraction with template subtraction (CO-1, CO-2, FR-EXT-03..08).

After registration, the photo and the blank template reference are pixel-aligned. Everything
pre-printed (labels, lines, boxes, the security pattern) appears in both; everything written
or printed per-cheque appears only in the photo. So:

    ink = pixels where the photo is clearly darker than the (slightly thickened) reference

**Template subtraction** gives each field crop as clean dark ink on white, with the guide
lines and background pattern gone. That helps OCR a lot on handwriting.

The reference is first darkened with a small **erosion** (minimum filter): its lines become a
little thicker, so a 1-2 px registration error doesn't leave thin "edge" ghosts of the
template lines in the difference.

Each field crop is then read with ``OcrEngine.read_line`` (recognition only, no text
detection, because we already know where the text is) and parsed into a validated value.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from findoc.localize.templates import ChequeTemplate
from findoc.preprocess.illumination import normalize_illumination
from findoc.recognize.parsers import (
    normalize_account,
    normalize_ifsc,
    parse_amount_figures,
    parse_cheque_date,
    parse_micr,
    words_to_amount,
)
from findoc.synth.cheque import FieldSpec
from findoc.types import BBox, ExtractedField

INK_DELTA = 45  # grey levels darker than the reference to count as ink
SIGNATURE_MIN_INK = 0.006  # fraction of the signature box that must be ink
OCR_HEIGHT = 64  # crops are scaled to this height for recognition


@dataclass
class FieldCrop:
    name: str
    image: np.ndarray  # BGR crop, ink on white
    ink_ratio: float


def ink_mask(registered: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """255 where the registered photo has ink that the blank template doesn't."""
    reg = cv2.cvtColor(normalize_illumination(registered), cv2.COLOR_BGR2GRAY)
    ref = cv2.cvtColor(normalize_illumination(reference), cv2.COLOR_BGR2GRAY)
    ref = cv2.erode(ref, np.ones((5, 5), np.uint8))  # thicken template strokes
    diff = ref.astype(np.int16) - reg.astype(np.int16)
    mask = (diff > INK_DELTA).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # drop speckle


def crop_fields(registered: np.ndarray, template: ChequeTemplate) -> dict[str, FieldCrop]:
    mask = ink_mask(registered, template.reference)
    crops: dict[str, FieldCrop] = {}
    for spec in template.fields:
        x0, y0, x1, y1 = spec.box
        m = mask[y0:y1, x0:x1]
        patch = registered[y0:y1, x0:x1]
        clean = np.full_like(patch, 255)
        clean[m > 0] = patch[m > 0]
        crops[spec.name] = FieldCrop(spec.name, clean, float(np.mean(m > 0)))
    return crops


def compact_cells(crop: np.ndarray, cells: int, gap_ratio: float = 0.15) -> np.ndarray:
    """Re-space a boxed field (one character per cell) like ordinary text.

    Date boxes put each digit in its own box, so after template subtraction the digits sit far
    apart. OCR models trained on normal text then drop or merge characters. Cutting the crop
    into its cells, trimming each character and joining them with a normal letter gap turns
    "1   8   0   9 ..." into "18092026". Empty cells are skipped.
    """
    h, w = crop.shape[:2]
    gap = np.full((h, max(2, round(h * gap_ratio)), 3), 255, np.uint8)
    parts: list[np.ndarray] = []
    for i in range(cells):
        cell = crop[:, round(i * w / cells) : round((i + 1) * w / cells)]
        xs = np.where((cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY) < 200).any(axis=0))[0]
        if len(xs) == 0:
            continue
        if parts:
            parts.append(gap)
        parts.append(cell[:, max(0, xs.min() - 2) : xs.max() + 3])
    return np.hstack(parts) if parts else crop


def _prepare(crop: np.ndarray, cells: int = 1) -> np.ndarray:
    """Trim empty margins and scale to a height the recogniser expects."""
    if cells > 1:
        crop = compact_cells(crop, cells)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray < 200)
    if len(xs):
        pad = 6
        crop = crop[
            max(0, ys.min() - pad) : ys.max() + pad, max(0, xs.min() - pad) : xs.max() + pad
        ]
    h, w = crop.shape[:2]
    scale = OCR_HEIGHT / max(h, 1)
    return cv2.resize(crop, (max(1, round(w * scale)), OCR_HEIGHT), interpolation=cv2.INTER_CUBIC)


def _bbox(box: tuple[int, int, int, int]) -> BBox:
    return BBox(x0=box[0], y0=box[1], x1=box[2], y1=box[3])


def per_cell_images(crop: np.ndarray, cells: int, pad: int = 10) -> list[np.ndarray]:
    """Each non-empty cell's character, trimmed and padded on white (one image per digit)."""
    h, w = crop.shape[:2]
    out = []
    for i in range(cells):
        cell = crop[:, round(i * w / cells) : round((i + 1) * w / cells)]
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(gray < 200)
        if len(xs) == 0:
            continue
        glyph = cell[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        out.append(cv2.copyMakeBorder(glyph, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                      value=(255, 255, 255)))  # fmt: skip
    return out


# How to tell a plausible reading of a boxed field from a garbled one.
BOXED_VALIDATORS = {"date": lambda text: parse_cheque_date(text) is not None}


def read_boxed(engine: object, spec: FieldSpec, crop: np.ndarray) -> tuple[str, float, str]:
    """Read a boxed field (one character per cell) with validated fallbacks.

    Tries, in order of measured reliability, and keeps the first reading that validates:

    1. ``spaced``: the crop as written (digits far apart);
    2. ``compact``: cells re-joined with normal letter spacing (:func:`compact_cells`);
    3. ``per_cell``: each character recognised on its own, then concatenated.

    Returns (text, confidence, variant). If nothing validates, the ``spaced`` reading is
    returned so the rule reports the field as unreadable.
    """
    validate = BOXED_VALIDATORS.get(spec.name, lambda text: bool(text))
    first: tuple[str, float, str] | None = None
    for variant in ("spaced", "compact"):
        img = _prepare(crop, spec.cells if variant == "compact" else 1)
        text, conf = _read(engine, spec, img)
        text = text.strip()
        first = first or (text, conf, variant)
        if validate(text):
            return text, conf, variant
    chars, confs = [], []
    for glyph in per_cell_images(crop, spec.cells):
        text, conf = _read(engine, spec, _prepare(glyph))
        chars.append(text.strip()[:1])
        confs.append(conf)
    text = "".join(chars)
    if validate(text):
        return text, min(confs, default=0.0), "per_cell"
    assert first is not None
    return first


def _read(engine: object, spec: FieldSpec, img: np.ndarray) -> tuple[str, float]:
    """Call the engine. Simulation/test engines may implement ``read_field`` to know which
    field they are reading; real engines only see the image."""
    read_field = getattr(engine, "read_field", None)
    if read_field is not None:
        return read_field(spec.name, img, spec.allowlist)
    return engine.read_line(img, allowlist=spec.allowlist)  # type: ignore[attr-defined]


def read_cheque_fields(
    registered: np.ndarray, template: ChequeTemplate, engine: object | None
) -> list[ExtractedField]:
    """Extract and normalise every field. ``engine`` None = no OCR (signature only)."""
    crops = crop_fields(registered, template)
    raw: dict[str, tuple[str, float]] = {}
    variants: dict[str, str] = {}
    if engine is not None:
        for spec in template.fields:
            if spec.kind == "signature":
                continue
            crop = crops[spec.name]
            if crop.ink_ratio < 0.001:  # nothing written there
                raw[spec.name] = ("", 0.0)
                continue
            if spec.cells > 1:
                text, conf, variant = read_boxed(engine, spec, crop.image)
                variants[spec.name] = variant
            else:
                text, conf = _read(engine, spec, _prepare(crop.image))
            raw[spec.name] = (text.strip(), conf)

    out: list[ExtractedField] = []

    def add(name: str, value: str, normalized: str | None, conf: float, box_of: str) -> None:
        out.append(
            ExtractedField(
                name=name,
                value=value,
                raw=value,
                normalized=normalized,
                method=variants.get(box_of),
                confidence=round(conf, 4),
                bbox=_bbox(template.field(box_of).box),
            )
        )

    if engine is not None:
        text, conf = raw.get("date", ("", 0.0))
        d = parse_cheque_date(text)
        add("date", text, d.isoformat() if d else None, conf, "date")

        text, conf = raw.get("payee", ("", 0.0))
        add("payee", text, text or None, conf, "payee")

        w1, c1 = raw.get("amount_words_1", ("", 0.0))
        w2, c2 = raw.get("amount_words_2", ("", 0.0))
        words = f"{w1} {w2}".strip()
        parsed = words_to_amount(words)
        conf = min(c for c in (c1, c2 or c1) if c is not None) * parsed.confidence
        normalized = str(parsed.value) if parsed.value is not None else None
        add("amount_words", words, normalized, conf, "amount_words_1")

        text, conf = raw.get("amount_figures", ("", 0.0))
        fig = parse_amount_figures(text)
        add("amount_figures", text, str(fig) if fig is not None else None, conf, "amount_figures")

        text, conf = raw.get("account_number", ("", 0.0))
        add("account_number", text, normalize_account(text), conf, "account_number")

        text, conf = raw.get("ifsc", ("", 0.0))
        add("ifsc", text, normalize_ifsc(text), conf, "ifsc")

        text, conf = raw.get("micr", ("", 0.0))
        micr = parse_micr(text)
        add("micr", text, str(micr) if micr else None, conf, "micr")

    sig = crops["signature"].ink_ratio
    present = sig >= SIGNATURE_MIN_INK
    add(
        "signature",
        "present" if present else "absent",
        f"{sig:.4f}",
        min(1.0, sig / (2 * SIGNATURE_MIN_INK)) if present else 1.0 - sig / SIGNATURE_MIN_INK,
        "signature",
    )
    return out
