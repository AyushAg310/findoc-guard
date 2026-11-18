"""Varied, deliberately *hard* backgrounds for training a document segmenter (Phase 4).

The classical page detector (Phase 1) fails when the background is about as bright as paper or
has strong edges of its own. Real failures seen so far: a cheque on a pale wooden board (the
detector outlined the whole board) and a patterned red cloth. So besides ordinary desks this
module draws:

* ``pale_wood``: light wood close to paper brightness;
* ``light_table``: near-white plain surfaces (white desk, bed sheet);
* ``board``: a bright board or tray under the document on a darker surrounding, the exact case
  of the third real test video;
* ``cloth``: saturated patterned fabric (paisley-like ellipses and curls);
* ``poster``: dark graphic with large white letters and shapes, which merge with paper edges;
* ``clutter``: other rectangles (papers, phone, books) partly in view;
* ``noise``: multi-scale random texture.

Every function returns a BGR ``uint8`` image of the requested size and is deterministic for a
given ``rng`` state.
"""

from __future__ import annotations

import cv2
import numpy as np

KINDS = ("desk", "pale_wood", "light_table", "board", "cloth", "poster", "clutter", "noise")


def _u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def _wood(rng: np.random.Generator, w: int, h: int, base: np.ndarray) -> np.ndarray:
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    freq = rng.uniform(0.03, 0.12)
    if rng.random() < 0.5:
        ys, xs = xs, ys  # vertical grain
    grain = np.sin(
        ys * freq + rng.uniform(2, 8) * np.sin(xs * rng.uniform(0.002, 0.01)) + rng.uniform(0, 6)
    )
    img += (grain * rng.uniform(6, 18))[..., None] * np.array([0.8, 1.0, 1.2])
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    img += rng.normal(0, 4, size=(h, w, 1)).astype(np.float32)
    return _u8(img)


def desk(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    return _wood(rng, w, h, rng.uniform(40, 130, 3))


def pale_wood(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    # BGR: yellowish-beige, 150-215 bright (paper is ~200-250)
    v = rng.uniform(150, 215)
    return _wood(rng, w, h, np.array([v * rng.uniform(0.6, 0.8), v * rng.uniform(0.82, 0.95), v]))


def light_table(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    base = rng.uniform(175, 235) + rng.uniform(-12, 12, 3)
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    gx = np.linspace(-1, 1, w)[None, :]
    img += (gx * rng.uniform(-25, 25))[..., None]
    img += cv2.GaussianBlur(rng.normal(0, 10, size=(h, w)).astype(np.float32), (0, 0), 6)[..., None]
    return _u8(img)


def board(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """A bright board covering most of the frame, on a dark surround (it may run off-frame)."""
    img = np.empty((h, w, 3), np.float32)
    img[:] = noise(rng, w, h).astype(np.float32) * rng.uniform(0.2, 0.5)
    bw, bh = w * rng.uniform(0.7, 1.15), h * rng.uniform(0.55, 1.0)
    cx, cy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.35, 0.65)
    quad = np.array([[-bw, -bh], [bw, -bh], [bw, bh], [-bw, bh]], np.float32) / 2
    quad += rng.uniform(-0.05, 0.05, (4, 2)).astype(np.float32) * [bw, bh] + [cx, cy]
    surface = pale_wood(rng, w, h) if rng.random() < 0.6 else light_table(rng, w, h)
    m = np.zeros((h, w), np.float32)
    cv2.fillConvexPoly(m, quad.astype(np.int32), 1.0)
    m = cv2.GaussianBlur(m, (0, 0), 1.0)[..., None]
    return _u8(img * (1 - m) + surface.astype(np.float32) * m)


def cloth(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    base = np.array(
        rng.choice([[40, 30, 150], [60, 40, 120], [120, 60, 30], [40, 110, 60], [90, 30, 90]]),
        float,
    )
    img = np.empty((h, w, 3), np.float32)
    img[:] = base + rng.uniform(-20, 20, 3)
    step = int(rng.integers(40, 90))
    for y in range(-step, h + step, step):
        for x in range(-step, w + step, step):
            cx, cy = x + int(rng.integers(-10, 10)) + (step // 2 if (y // step) % 2 else 0), y
            col = tuple(
                float(c)
                for c in np.clip(base * rng.uniform(0.5, 1.9, 3) + rng.uniform(0, 60), 0, 255)
            )
            ax = (int(step * rng.uniform(0.15, 0.4)), int(step * rng.uniform(0.1, 0.3)))
            cv2.ellipse(img, (cx, cy), ax, float(rng.uniform(0, 180)), 0, 360, col, 2)
            cv2.circle(img, (cx, cy), max(2, ax[1] // 2), col, -1)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return _u8(img + rng.normal(0, 5, (h, w, 1)).astype(np.float32))


def poster(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    img = np.empty((h, w, 3), np.float32)
    img[:] = rng.uniform(10, 70, 3)
    for _ in range(int(rng.integers(3, 8))):  # white strokes and shapes
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(
            img, p1, p2, tuple(float(v) for v in rng.uniform(170, 255, 3)), int(rng.integers(4, 25))
        )
    img = _u8(img)  # putText needs 8-bit images (OpenCV 5)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for _ in range(int(rng.integers(2, 6))):
        text = "".join(rng.choice(list(letters), size=int(rng.integers(3, 8))))
        scale = rng.uniform(1.5, 5.0)
        org = (int(rng.integers(-50, w)), int(rng.integers(30, h + 50)))
        cv2.putText(
            img,
            text,
            org,
            cv2.FONT_HERSHEY_DUPLEX,
            scale,
            tuple(float(v) for v in rng.uniform(200, 255, 3)),
            int(scale * 3),
        )
    img = cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32)
    return _u8(img + rng.normal(0, 4, (h, w, 1)).astype(np.float32))


def clutter(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    img = desk(rng, w, h).astype(np.float32)
    for _ in range(int(rng.integers(2, 6))):
        cx, cy = rng.uniform(-0.2, 1.2) * w, rng.uniform(-0.2, 1.2) * h
        rw, rh = rng.uniform(0.15, 0.6) * w, rng.uniform(0.15, 0.6) * h
        box = cv2.boxPoints(((cx, cy), (rw, rh), rng.uniform(0, 90))).astype(np.int32)
        col = tuple(
            float(v)
            for v in (rng.uniform(170, 250, 3) if rng.random() < 0.5 else rng.uniform(10, 120, 3))
        )
        cv2.fillConvexPoly(img, box, col)
    return _u8(img + rng.normal(0, 4, (h, w, 1)).astype(np.float32))


def noise(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.float32)
    for scale, amp in ((64, 60), (16, 30), (4, 15)):
        small = rng.normal(0, 1, (max(2, h // scale), max(2, w // scale), 3)).astype(np.float32)
        img += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * amp
    return _u8(img + rng.uniform(40, 200, 3))


GENERATORS = {
    "desk": desk,
    "pale_wood": pale_wood,
    "light_table": light_table,
    "board": board,
    "cloth": cloth,
    "poster": poster,
    "clutter": clutter,
    "noise": noise,
}


def background(
    rng: np.random.Generator, w: int, h: int, kind: str | None = None
) -> tuple[np.ndarray, str]:
    """A background of the given kind (random if None). Returns (image, kind)."""
    kind = kind or str(rng.choice(KINDS))
    return GENERATORS[kind](rng, w, h), kind
