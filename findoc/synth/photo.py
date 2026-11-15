"""Synthetic "phone photo of a document" generator with ground truth.

Renders a document page, then simulates taking a photo of it:

1. **Perspective**: choose four destination corners inside the frame (random tilt, scale,
   position) and warp the page with a homography. The corners are the ground truth for
   evaluating detection.
2. **Background**: a textured desk-like colour.
3. **Lighting**: multiply by a smooth gradient and an optional soft shadow, like a lamp on one
   side or a hand over the page.
4. **Sensor**: Gaussian noise, optional Gaussian blur (camera shake), JPEG compression.

Because every parameter is known, generated photos measure detection accuracy exactly
(see ``experiments/phase1_detection_eval.py``) and give tests realistic inputs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WORDS = (
    "INVOICE",
    "PAYMENT",
    "AMOUNT",
    "TOTAL",
    "DATE",
    "ACCOUNT",
    "BRANCH",
    "RUPEES",
    "ONLY",
    "BEARER",
    "ORDER",
    "GST",
    "CHEQUE",
    "BANK",
    "LIMITED",
    "SERVICES",
    "PRIVATE",
    "SUPPLY",
    "QUANTITY",
    "RATE",
    "TAX",
    "NUMBER",
)


@dataclass
class PhotoParams:
    corners: list[list[float]]  # ground-truth page corners in the photo (tl, tr, br, bl)
    blur_sigma: float
    shadow: bool
    jpeg_quality: int
    seed: int


def render_page(rng: np.random.Generator, width: int = 850, height: int = 1100) -> np.ndarray:
    """A white page with a header band, text lines and a table, like an invoice."""
    page = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.rectangle(page, (40, 40), (width - 40, 130), (60, 60, 60), 2)
    cv2.putText(page, "SYNTHETIC DOCUMENT", (60, 100), FONT, 1.3, (30, 30, 30), 3)
    y = 190
    for _ in range(int(rng.integers(5, 9))):
        text = " ".join(rng.choice(WORDS, size=int(rng.integers(3, 6))))
        cv2.putText(page, text, (60, y), FONT, 0.8, (25, 25, 25), 2)
        y += 48
    top = y + 20
    rows, cols = 6, 4
    cw, rh = (width - 120) // cols, 44
    for r in range(rows + 1):
        cv2.line(page, (60, top + r * rh), (60 + cols * cw, top + r * rh), (80, 80, 80), 1)
    for c in range(cols + 1):
        cv2.line(page, (60 + c * cw, top), (60 + c * cw, top + rows * rh), (80, 80, 80), 1)
    for r in range(rows):
        for c in range(cols):
            val = str(int(rng.integers(1, 99999))) if c else str(rng.choice(WORDS))
            cv2.putText(page, val, (70 + c * cw, top + r * rh + 30), FONT, 0.6, (30, 30, 30), 1)
    total_y = top + rows * rh + 60
    amount = int(rng.integers(1000, 999999))
    cv2.putText(page, f"TOTAL RS {amount:,}", (60, total_y), FONT, 1.0, (20, 20, 20), 2)
    return page


def _random_quad(
    rng: np.random.Generator, w: int, h: int, fill: tuple[float, float], aspect: float
) -> np.ndarray:
    """Four corners of a tilted page covering roughly ``fill`` of the frame.

    ``aspect`` is the page's width / height, preserved so the ground truth stays a true
    perspective view of the page.
    """
    frac = rng.uniform(*fill)
    area = frac * w * h
    qw, qh = np.sqrt(area * aspect), np.sqrt(area / aspect)
    shrink = min(1.0, 0.9 * w / qw, 0.9 * h / qh)  # keep the page inside the frame
    qw, qh = qw * shrink, qh * shrink
    cx = w / 2 + rng.uniform(-0.08, 0.08) * w
    cy = h / 2 + rng.uniform(-0.06, 0.06) * h
    base = np.array(
        [[-qw / 2, -qh / 2], [qw / 2, -qh / 2], [qw / 2, qh / 2], [-qw / 2, qh / 2]], np.float32
    )
    angle = np.deg2rad(rng.uniform(-12, 12))
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32)
    quad = base @ rot.T + np.array([cx, cy], np.float32)
    quad += rng.uniform(-0.04, 0.04, size=(4, 2)).astype(np.float32) * [qw, qh]  # perspective
    quad[:, 0] = np.clip(quad[:, 0], 5, w - 6)
    quad[:, 1] = np.clip(quad[:, 1], 5, h - 6)
    return quad


def photograph(
    page: np.ndarray,
    rng: np.random.Generator,
    size: tuple[int, int] = (1200, 1600),
    fill: tuple[float, float] = (0.45, 0.75),
    blur_sigma: float = 0.0,
    shadow: bool = True,
    jpeg_quality: int = 90,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a phone photo of ``page``. Returns (photo, ground-truth corners).

    ``size`` is (width, height) of the photo; ``fill`` the range of page area / frame area.
    """
    w, h = size
    bg_colour = rng.uniform(40, 120, size=3)
    photo = np.empty((h, w, 3), np.float32)
    photo[:] = bg_colour
    photo += rng.normal(0, 8, size=(h, w, 1)).astype(np.float32)  # desk texture

    ph, pw = page.shape[:2]
    quad = _random_quad(rng, w, h, fill, aspect=pw / ph)
    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
    homography = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(page, homography, (w, h)).astype(np.float32)
    mask = cv2.warpPerspective(np.ones((ph, pw), np.float32), homography, (w, h))[..., None]
    photo = photo * (1 - mask) + warped * mask

    # Lighting: linear gradient across the frame plus an optional soft shadow blob.
    gx, gy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    direction = rng.uniform(0, 2 * np.pi)
    grad = np.cos(direction) * gx + np.sin(direction) * gy
    grad = (grad - grad.min()) / (np.ptp(grad) + 1e-6)
    light = 0.7 + 0.3 * grad
    if shadow:
        sx, sy = rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.8) * h
        r = rng.uniform(0.15, 0.3) * max(w, h)
        light *= 1 - 0.35 * np.exp(-(((gx * w - sx) ** 2 + (gy * h - sy) ** 2) / (2 * r**2)))
    photo *= light[..., None]

    photo += rng.normal(0, 3, size=photo.shape).astype(np.float32)  # sensor noise
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    if blur_sigma > 0:
        photo = cv2.GaussianBlur(photo, (0, 0), blur_sigma)
    ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if ok:
        photo = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return photo, quad


def generate(
    n: int,
    out_dir: Path,
    seed: int = 0,
    blur_sigma: float = 0.0,
    size: tuple[int, int] = (1200, 1600),
) -> list[Path]:
    """Write ``n`` synthetic photos plus a ``labels.json`` with ground-truth corners."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels: dict[str, dict] = {}
    paths: list[Path] = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        page = render_page(rng)
        shadow = bool(rng.random() < 0.6)
        quality = int(rng.integers(75, 95))
        photo, quad = photograph(
            page, rng, size=size, blur_sigma=blur_sigma, shadow=shadow, jpeg_quality=quality
        )
        name = f"synth_{seed + i:04d}.jpg"
        path = out_dir / name
        cv2.imwrite(str(path), photo)
        labels[name] = asdict(
            PhotoParams(quad.round(2).tolist(), blur_sigma, shadow, quality, seed + i)
        )
        paths.append(path)
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return paths
