from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """A synthetic 'document on a desk' photo: white page, dark text lines, grey background."""
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
