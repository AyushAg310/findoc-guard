"""Synthetic cheque photo datasets with ground truth for evaluation and demos."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from findoc.synth.cheque import ChequeTruth, render_cheque
from findoc.synth.photo import photograph

FRAUD_KINDS = ("mismatch", "stale", "postdated", "unsigned")
CHEQUE_PHOTO_SIZE = (1600, 1200)  # landscape, like a phone held sideways
CHEQUE_FILL = (0.35, 0.6)


def cheque_photo(
    seed: int,
    fraud: str | None = None,
    today: date | None = None,
    blur_sigma: float = 0.0,
    bank: str | None = None,
) -> tuple[np.ndarray, ChequeTruth, np.ndarray]:
    """One synthetic cheque photo. Returns (photo, truth, page->photo homography)."""
    rng = np.random.default_rng(seed)
    flags = {k: fraud == k for k in FRAUD_KINDS}
    cheque, truth = render_cheque(rng, bank=bank, today=today, **flags)
    photo, quad = photograph(
        cheque, rng, size=CHEQUE_PHOTO_SIZE, fill=CHEQUE_FILL, blur_sigma=blur_sigma
    )
    h, w = cheque.shape[:2]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    page_to_photo = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    return photo, truth, page_to_photo


def generate_cheques(
    n: int,
    out_dir: Path,
    seed: int = 0,
    blur_sigma: float = 0.0,
    fraud_rate: float = 0.4,
    today: date | None = None,
) -> list[Path]:
    """Write ``n`` cheque photos and labels.json (truth + page->photo homography)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    picker = np.random.default_rng(seed + 99_991)
    labels: dict[str, dict] = {}
    paths = []
    for i in range(n):
        fraud = str(picker.choice(FRAUD_KINDS)) if picker.random() < fraud_rate else None
        photo, truth, homography = cheque_photo(seed + i, fraud, today, blur_sigma)
        name = f"cheque_{seed + i:04d}.jpg"
        cv2.imwrite(str(out_dir / name), photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
        labels[name] = asdict(truth) | {"page_to_photo": homography.round(6).tolist()}
        paths.append(out_dir / name)
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return paths
