from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from findoc.synth.photo import photograph, render_page


@pytest.fixture(autouse=True)
def _no_real_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never load the real OCR model (slow, heavy); OCR tests inject a fake engine."""
    from findoc.recognize import ocr

    def unavailable() -> ocr.OcrEngine:
        raise ocr.OcrUnavailableError("disabled in tests")

    monkeypatch.setattr(ocr, "get_engine", unavailable)


@pytest.fixture(autouse=True)
def _classical_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests are deterministic whether or not a trained segmenter exists on this machine
    (``models/docseg.onnx``): the learned detector is off unless a test turns it on."""
    from findoc.config import get_settings

    monkeypatch.setenv("FINDOC_SEGMENTATION", "off")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """A simple 'document on a desk' photo: white page, dark text lines, grey background."""
    img = np.full((600, 800, 3), 90, dtype=np.uint8)
    page = np.array([[150, 80], [650, 100], [630, 520], [170, 500]], dtype=np.int32)
    cv2.fillPoly(img, [page], (245, 245, 245))
    for i in range(6):
        y = 160 + i * 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "PAY RS 25,000 ONLY", (210, y), font, 0.8, (20, 20, 20), 2)
    path = tmp_path / "doc.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def synth_photo() -> tuple[np.ndarray, np.ndarray]:
    """A realistic synthetic photo and its ground-truth page corners."""
    rng = np.random.default_rng(42)
    return photograph(render_page(rng), rng)


@pytest.fixture
def synth_photo_path(tmp_path: Path, synth_photo: tuple[np.ndarray, np.ndarray]) -> Path:
    path = tmp_path / "synth.jpg"
    cv2.imwrite(str(path), synth_photo[0])
    return path


@pytest.fixture
def blurry_photo_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(3)
    photo, _ = photograph(render_page(rng), rng, blur_sigma=5.0)
    path = tmp_path / "blurry.jpg"
    cv2.imwrite(str(path), photo)
    return path
