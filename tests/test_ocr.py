from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from findoc.pipeline import process
from findoc.recognize import ocr
from findoc.types import BBox, OcrWord


def word(text: str, x0: float, y0: float, x1: float, y1: float, conf: float = 0.9) -> OcrWord:
    return OcrWord(text=text, confidence=conf, bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1))


def test_words_to_text_reading_order() -> None:
    words = [
        word("TOTAL", 10, 200, 80, 220),
        word("world", 120, 12, 200, 30),
        word("Hello", 10, 10, 100, 30),
        word("RS 500", 100, 202, 180, 222),
    ]
    assert ocr.words_to_text(words) == "Hello world\nTOTAL RS 500"


def test_words_to_text_empty() -> None:
    assert ocr.words_to_text([]) == ""


def test_ocr_skipped_when_not_installed(synth_photo_path: Path) -> None:
    doc = process(synth_photo_path)  # conftest makes the engine unavailable
    assert doc.status.value == "ok"
    assert any("OCR skipped" in m for m in doc.messages)
    assert doc.words == []


def test_ocr_stage_with_fake_engine(
    synth_photo_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeEngine:
        def read(self, img: np.ndarray) -> list[OcrWord]:
            assert img.ndim == 3
            return [word("TOTAL", 10, 10, 80, 30, 0.8), word("RS 1,000", 90, 10, 200, 30, 0.6)]

    monkeypatch.setattr(ocr, "get_engine", lambda: FakeEngine())
    doc = process(synth_photo_path)
    assert doc.full_text == "TOTAL RS 1,000"
    assert doc.quality["ocr_mean_confidence"] == pytest.approx(0.7)
    assert doc.result()["words"][0]["text"] == "TOTAL"


def test_easyocr_missing_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "easyocr":
            raise ImportError("no easyocr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ocr.OcrUnavailableError, match="pip install"):
        ocr.EasyOcrEngine()


def test_use_gpu_cpu_forced() -> None:
    assert ocr._use_gpu("cpu") is False
