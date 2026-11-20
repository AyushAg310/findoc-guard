"""Synthetic signers: one person's signature with natural variation, and forgeries of it.

A real signature is a habitual movement: the same sequence of strokes each time, never exactly
the same shape. A ``Signer`` fixes that movement (control points of the main stroke, loops,
flourish, underline, pen width); ``draw`` renders one instance of it with small random changes
(position of each control point, overall size, slant, baseline).

Forgeries, as in the signature-verification literature:

* **random forgery**: someone signs with their own movement (another ``Signer``), not knowing the
  real signature;
* **skilled forgery** (``draw(..., skilled=True)``): the forger copies the real signature's
  shape, but slowly and less fluently: control points vary three times as much, the line has
  tremor, and pen width differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

INK = (110, 35, 20)  # BGR ballpoint


@dataclass(frozen=True)
class Signer:
    xs: tuple[float, ...]  # control points along the box, fraction of its width
    ys: tuple[float, ...]  # vertical offsets, fraction of its height
    loops: tuple[tuple[float, float, float], ...]  # (position, radius, height) of letter loops
    wiggle: tuple[float, float]  # frequency, amplitude of the stroke's wave
    underline: bool
    width: int

    @staticmethod
    def random(rng: np.random.Generator) -> Signer:
        n = int(rng.integers(6, 10))
        xs = np.sort(rng.uniform(0.08, 0.85, n))
        return Signer(
            xs=tuple(float(x) for x in xs),
            ys=tuple(float(y) for y in rng.normal(0, 0.18, n)),
            loops=tuple(
                (
                    float(rng.uniform(0.1, 0.8)),
                    float(rng.uniform(0.05, 0.12)),
                    float(rng.uniform(-0.2, 0.2)),
                )
                for _ in range(int(rng.integers(1, 4)))
            ),  # fmt: skip
            wiggle=(float(rng.uniform(2.0, 5.0)), float(rng.uniform(0.04, 0.12))),
            underline=bool(rng.random() < 0.6),
            width=int(rng.integers(2, 4)),
        )

    def draw(
        self,
        img: np.ndarray,
        box: tuple[int, int, int, int],
        rng: np.random.Generator,
        skilled: bool = False,
    ) -> None:
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        jit = 0.036 if skilled else 0.012  # control-point variation, fraction of the box
        scale = rng.uniform(0.93, 1.07)
        slant = rng.normal(0, 0.03)
        cx, cy = x0 + bw * rng.uniform(0.47, 0.53), y0 + bh * rng.uniform(0.47, 0.53)

        def place(u: np.ndarray, v: np.ndarray) -> np.ndarray:
            px = x0 + u * bw
            py = y0 + bh / 2 + v * bh
            px, py = cx + (px - cx) * scale + slant * (py - cy), cy + (py - cy) * scale
            return np.stack([px, py], 1)

        xs = np.array(self.xs) + rng.normal(0, jit, len(self.xs))
        ys = np.array(self.ys) + rng.normal(0, jit * 1.5, len(self.ys))
        t = np.linspace(0, len(xs) - 1, 260)
        u = np.interp(t, np.arange(len(xs)), xs)
        f, a = self.wiggle
        v = np.interp(t, np.arange(len(ys)), ys) + a * np.sin(t * f)
        if skilled:  # slow, hesitant copying: tremor along the line
            v = v + np.convolve(rng.normal(0, 0.012, len(v)), np.ones(5) / 5, mode="same")
        width = self.width + (int(rng.choice([-1, 1])) if skilled else 0)
        pts = place(u, v).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, INK, max(1, width), cv2.LINE_AA)
        for pos, r, h in self.loops:
            pos = pos + rng.normal(0, jit)
            ang = np.linspace(0, 2 * np.pi, 60)
            lu = pos + r * 0.5 * np.cos(ang) + 0.02 * ang / np.pi
            lv = h + r * 1.6 * np.sin(ang) + rng.normal(0, jit)
            lp = place(lu, lv).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [lp], False, INK, max(1, width - 1), cv2.LINE_AA)
        if self.underline:
            base = max(self.ys) + 0.28 + rng.normal(0, jit)
            p = place(np.array([xs[0], xs[-1] + 0.05]), np.array([base, base - 0.04]))
            cv2.line(img, tuple(p[0].astype(int)), tuple(p[1].astype(int)), INK, max(1, width - 1),
                     cv2.LINE_AA)  # fmt: skip
