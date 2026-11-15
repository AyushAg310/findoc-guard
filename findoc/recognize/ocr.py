"""Optical character recognition (CO-2, FR-EXT-05).

The OCR engine sits behind a small interface (:class:`OcrEngine`) so it can be swapped
(EasyOCR now, PaddleOCR or TrOCR later) without touching the pipeline.

EasyOCR is an optional dependency because it pulls in PyTorch (~2 GB). Install it with
``pip install -e ".[ocr]"``. Without it, the OCR stage is skipped with a message and the
rest of the pipeline still runs.

EasyOCR internally uses CRAFT (a CNN that finds text regions) followed by a CRNN
(convolutional + recurrent network trained with CTC loss) that reads each region.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import numpy as np

from findoc.config import get_settings
from findoc.logging import get_logger
from findoc.types import BBox, Document, OcrWord

log = get_logger(__name__)


class OcrUnavailableError(RuntimeError):
    """Raised when no OCR engine is installed."""


class OcrEngine(Protocol):
    def read(self, img: np.ndarray) -> list[OcrWord]: ...


def _use_gpu(device: str) -> bool:
    if device == "cpu":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


class EasyOcrEngine:
    def __init__(self, languages: tuple[str, ...] = ("en",), device: str = "auto") -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise OcrUnavailableError(
                'EasyOCR is not installed. Run: pip install -e ".[ocr]"'
            ) from exc
        gpu = _use_gpu(device)
        log.info("Loading EasyOCR (gpu=%s); first run downloads model weights", gpu)
        self._reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)

    def read(self, img: np.ndarray) -> list[OcrWord]:
        words: list[OcrWord] = []
        for points, text, conf in self._reader.readtext(img):
            pts = np.asarray(points, dtype=np.float32)
            words.append(
                OcrWord(
                    text=str(text),
                    confidence=round(float(conf), 4),
                    bbox=BBox(
                        x0=float(pts[:, 0].min()),
                        y0=float(pts[:, 1].min()),
                        x1=float(pts[:, 0].max()),
                        y1=float(pts[:, 1].max()),
                    ),
                )
            )
        return words


@lru_cache(maxsize=1)
def get_engine() -> OcrEngine:
    """Create the OCR engine once per process (model loading is slow)."""
    return EasyOcrEngine(device=get_settings().device)


def words_to_text(words: list[OcrWord], line_tol: float = 0.6) -> str:
    """Join words into reading-order text: group by line (similar y), then sort by x."""
    if not words:
        return ""
    items = sorted(words, key=lambda w: (w.bbox.y0 + w.bbox.y1) / 2)
    lines: list[list[OcrWord]] = []
    for w in items:
        cy = (w.bbox.y0 + w.bbox.y1) / 2
        height = max(1.0, w.bbox.y1 - w.bbox.y0)
        if lines:
            last = lines[-1][-1]
            last_cy = (last.bbox.y0 + last.bbox.y1) / 2
            if abs(cy - last_cy) < line_tol * height:
                lines[-1].append(w)
                continue
        lines.append([w])
    return "\n".join(" ".join(w.text for w in sorted(ln, key=lambda w: w.bbox.x0)) for ln in lines)


def ocr_stage(doc: Document) -> Document:
    img = doc.enhanced if doc.enhanced is not None else doc.flat
    if img is None:
        return doc
    try:
        engine = get_engine()
    except OcrUnavailableError as exc:
        doc.messages.append(f"OCR skipped: {exc}")
        return doc
    doc.words = engine.read(img)
    doc.full_text = words_to_text(doc.words)
    if doc.words:
        mean_conf = sum(w.confidence for w in doc.words) / len(doc.words)
        doc.quality["ocr_mean_confidence"] = round(mean_conf, 4)
    doc.messages.append(f"OCR found {len(doc.words)} text regions")
    return doc
