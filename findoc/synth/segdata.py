"""Synthetic training samples for document segmentation: photo + exact document mask (Phase 4).

Each sample is a random document (synthetic cheque, invoice-like page or plain printed sheet,
in white or tinted paper) seen in perspective on a hard background
(:mod:`findoc.synth.backgrounds`), with lighting gradients, shadows, an optional thumb, blur,
noise and JPEG compression. The label is the **whole** document quadrilateral (also under a
thumb), because what the pipeline needs downstream is the page outline, not just its visible
pixels.

Documents are rendered once into a small pool and reused with fresh geometry and
backgrounds, so generating a sample costs only a perspective warp and a few filters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import cv2
import numpy as np

from findoc.synth.backgrounds import background
from findoc.synth.cheque import render_cheque
from findoc.synth.photo import render_page

PAPER_TINTS = (  # BGR multipliers
    (1.0, 1.0, 1.0),
    (0.93, 0.98, 1.0),  # cream
    (0.97, 0.93, 1.0),  # pink
    (1.0, 0.97, 0.92),  # blue
    (0.93, 1.0, 0.95),  # green
)


@dataclass
class SegSample:
    image: np.ndarray  # BGR uint8
    mask: np.ndarray  # uint8, 255 inside the document
    corners: np.ndarray  # (4, 2) tl, tr, br, bl in image pixels
    background: str
    occluded: bool
    partial: bool = False  # the document runs off the frame (corners may lie outside it)


def plain_sheet(rng: np.random.Generator) -> np.ndarray:
    """A plain printed sheet: few lines of text, sometimes nearly blank."""
    h = int(rng.integers(500, 900))
    w = int(h * rng.choice([0.71, 1.41, 2.2, 1.0]))
    page = np.full((h, w, 3), rng.uniform(235, 252), np.float32)
    for i in range(int(rng.integers(0, 12))):
        y = int(40 + i * rng.uniform(30, 60))
        if y > h - 20:
            break
        x1 = int(rng.uniform(20, 0.6 * w))
        cv2.line(
            page,
            (20, y),
            (x1, y),
            tuple(float(v) for v in rng.uniform(20, 90, 3)),
            int(rng.integers(2, 6)),
        )
    return np.clip(page, 0, 255).astype(np.uint8)


def document_pool(seed: int = 0, n: int = 48) -> list[np.ndarray]:
    """Rendered documents to reuse across samples: cheques, pages and plain sheets."""
    rng = np.random.default_rng(seed)
    pool = []
    for i in range(n):
        kind = i % 3
        if kind == 0:
            doc, _ = render_cheque(rng, today=date(2026, 11, 17))
        elif kind == 1:
            doc = render_page(rng)
        else:
            doc = plain_sheet(rng)
        tint = np.array(PAPER_TINTS[int(rng.integers(0, len(PAPER_TINTS)))], np.float32)
        pool.append(np.clip(doc.astype(np.float32) * tint, 0, 255).astype(np.uint8))
    return pool


def _partial_quad(rng: np.random.Generator, w: int, h: int, aspect: float) -> np.ndarray:
    """A document that runs off the frame: larger than the frame and/or pushed past one edge."""
    quad = _quad(rng, w, h, aspect)
    centre = quad.mean(0)
    quad = centre + (quad - centre) * rng.uniform(1.0, 1.8)  # possibly bigger than the frame
    side = int(rng.integers(0, 4))  # push it past one edge
    shift = rng.uniform(0.15, 0.45)
    lo, hi = quad.min(0), quad.max(0)
    if side == 0:
        quad[:, 0] -= lo[0] + shift * (hi[0] - lo[0])
    elif side == 1:
        quad[:, 0] += (w - hi[0]) + shift * (hi[0] - lo[0])
    elif side == 2:
        quad[:, 1] -= lo[1] + shift * (hi[1] - lo[1])
    else:
        quad[:, 1] += (h - hi[1]) + shift * (hi[1] - lo[1])
    return quad.astype(np.float32)


def _quad(rng: np.random.Generator, w: int, h: int, aspect: float) -> np.ndarray:
    """A random perspective quadrilateral of the given aspect, fully inside a w x h frame."""
    frac = rng.uniform(0.15, 0.75)
    qw, qh = np.sqrt(frac * w * h * aspect), np.sqrt(frac * w * h / aspect)
    base = np.array([[-qw, -qh], [qw, -qh], [qw, qh], [-qw, qh]], np.float32) / 2
    a = np.deg2rad(rng.uniform(-25, 25) if rng.random() < 0.8 else rng.uniform(65, 115))
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    quad = base @ rot.T
    quad += rng.uniform(-0.08, 0.08, (4, 2)).astype(np.float32) * [qw, qh]  # perspective
    extent = quad.max(0) - quad.min(0)  # fit the *rotated* shape inside the frame
    quad *= min(1.0, 0.94 * w / extent[0], 0.94 * h / extent[1])
    lo, hi = quad.min(0), quad.max(0)
    cx = rng.uniform(-lo[0] + 2, w - hi[0] - 2)
    cy = rng.uniform(-lo[1] + 2, h - hi[1] - 2)
    return (quad + np.array([cx, cy], np.float32)).astype(np.float32)


def _order(quad: np.ndarray) -> np.ndarray:
    from findoc.preprocess.detect import order_corners

    return order_corners(quad)


def make_sample(
    rng: np.random.Generator,
    pool: list[np.ndarray],
    size: tuple[int, int] = (512, 384),
    kind: str | None = None,
    partial_rate: float = 0.2,
) -> SegSample:
    """One training sample. ``size`` = (width, height); aspect is randomised around it.

    With probability ``partial_rate`` the document runs off the frame; its mask then reaches the
    image border, which teaches the network to mark paper all the way to the edge so that the
    pipeline can tell a cut-off document from a whole one (found with real videos: a model
    trained only on whole documents outlined the visible part of a cut-off cheque as if it were
    complete).
    """
    w, h = size
    if rng.random() < 0.5:  # landscape or portrait frames
        w, h = h, w
    bg, bg_kind = background(rng, w, h, kind)
    doc = pool[int(rng.integers(0, len(pool)))]
    if rng.random() < 0.15:
        doc = cv2.rotate(doc, cv2.ROTATE_90_CLOCKWISE)
    dh, dw = doc.shape[:2]
    partial = bool(rng.random() < partial_rate)
    quad = _partial_quad(rng, w, h, dw / dh) if partial else _quad(rng, w, h, dw / dh)
    src = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], np.float32)
    hmat = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(doc, hmat, (w, h), flags=cv2.INTER_AREA).astype(np.float32)
    alpha = cv2.warpPerspective(np.ones((dh, dw), np.float32), hmat, (w, h))[..., None]
    img = bg.astype(np.float32) * (1 - alpha) + warped * alpha

    occluded = bool(rng.random() < 0.25)
    if occluded and not partial:  # a thumb holding the document by a corner or edge
        c = quad[int(rng.integers(0, 4))]
        d = quad.mean(0) - c
        centre = c + d * rng.uniform(0.0, 0.15)
        axes = (int(min(w, h) * rng.uniform(0.04, 0.08)), int(min(w, h) * rng.uniform(0.08, 0.16)))
        ang = float(np.degrees(np.arctan2(d[1], d[0])) + 90 + rng.uniform(-30, 30))
        tone = tuple(float(v) for v in np.array([120, 150, 200]) + rng.uniform(-30, 30, 3))
        cv2.ellipse(img, tuple(int(v) for v in centre), axes, ang, 0, 360, tone, -1)

    # lighting: gradient and soft shadow over everything
    gx, gy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    t = rng.uniform(0, 2 * np.pi)
    grad = np.cos(t) * gx + np.sin(t) * gy
    light = rng.uniform(0.65, 0.9) + rng.uniform(0.1, 0.35) * (grad - grad.min()) / (
        np.ptp(grad) + 1e-6
    )
    if rng.random() < 0.5:
        sx, sy, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0.15, 0.4) * max(w, h)
        light *= 1 - rng.uniform(0.1, 0.4) * np.exp(
            -(((gx * w - sx) ** 2 + (gy * h - sy) ** 2) / (2 * r**2))
        )
    img *= light[..., None]
    img *= rng.uniform(0.85, 1.15, 3)  # white balance
    img += rng.normal(0, rng.uniform(1, 6), img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.0))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(55, 95))])
    if ok:
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [quad.round().astype(np.int32)], 255)  # clipped to the frame if partial
    return SegSample(img, mask, _order(quad), bg_kind, occluded and not partial, partial)
