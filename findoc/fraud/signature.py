"""Signature comparison against the account holder's reference signatures (Phase 6).

The SRS planned a Siamese network (two copies of one CNN mapping signatures to embeddings,
trained so genuine pairs land close). That needs thousands of real signatures from many
writers, which this project cannot use. This version is classical and explainable, and the
experiment states how it was evaluated (synthetic signers, :mod:`findoc.synth.signature`).

1. **Crop** (``signature_image``): the signature box of the registered cheque, template
   subtracted (Phase 2 ``ink_mask``), cut to the ink's bounding box and scaled into a 192x64
   canvas keeping its aspect ratio. Position, size and pen colour no longer matter.
2. **Describe** (``describe``), two complementary views of the shape:
   * a **histogram of oriented gradients** (HOG): for each 16x16 cell, how much of the stroke
     runs in each of 9 directions. It captures the strokes' directions and where they are;
   * a coarse **ink-density grid** (24x8 averages of the blurred crop): the overall layout.
3. **Compare** (``distance``): 1 - cosine similarity for each view, summed (0 = identical
   shape, about 2 = unrelated). A questioned signature is compared with every reference of the
   account, and the smallest distance counts: people sign with variation, and the closest
   reference is the fair one.

``SIGNATURE_MISMATCH`` fires above ``SIGNATURE_THRESHOLD``. References live in a private
folder (``FINDOC_SIGNATURES_DIR``, default ``data/signatures``), one subfolder per account:
``findoc signatures add ACCOUNT cheque1.jpg cheque2.jpg``.

Limits worth stating in a viva: shape descriptors judge the *picture* of a signature, not the
pen dynamics a skilled forger cannot copy (speed, pressure); they will miss a very careful
tracing, and a writer whose signature varies a lot needs more references.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

CANVAS = (192, 64)  # width, height
SIGNATURE_THRESHOLD = 0.39  # 95th percentile of genuine distances (calibration signers)
MIN_INK_PIXELS = 30


def signature_image(registered: np.ndarray, template) -> np.ndarray | None:  # noqa: ANN001
    """Normalised signature (float32 64x192, ink = 1) from a registered cheque, or None."""
    from findoc.recognize.fields import ink_mask

    x0, y0, x1, y1 = template.field("signature").box
    m = ink_mask(registered, template.reference)[y0:y1, x0:x1]
    return normalise((m > 0).astype(np.uint8))


def normalise(mask: np.ndarray) -> np.ndarray | None:
    """Binary ink mask (any size) -> ink cropped and centred in the canvas, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_INK_PIXELS:
        return None
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
    cw, ch = CANVAS
    h, w = crop.shape
    s = min((cw - 12) / w, (ch - 8) / h)
    crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((ch, cw), np.float32)
    oy, ox = (ch - crop.shape[0]) // 2, (cw - crop.shape[1]) // 2
    out[oy : oy + crop.shape[0], ox : ox + crop.shape[1]] = crop
    return out


def hog(img: np.ndarray, cell: int = 16, bins: int = 9) -> np.ndarray:
    """Histogram of oriented gradients: per cell, gradient magnitude per direction (unsigned,
    0-180 degrees in ``bins`` bins), each cell's histogram normalised to unit length."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    b = np.minimum(((np.arctan2(gy, gx) % np.pi) / np.pi * bins).astype(int), bins - 1)
    h, w = img.shape
    hc, wc = h // cell, w // cell
    out = np.zeros((hc, wc, bins), np.float32)
    for k in range(bins):
        out[..., k] = (
            (mag * (b == k))[: hc * cell, : wc * cell].reshape(hc, cell, wc, cell).sum((1, 3))
        )
    out /= np.linalg.norm(out, axis=-1, keepdims=True) + 1e-3
    return out.ravel()


def describe(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = hog(cv2.GaussianBlur(sig, (0, 0), 1.5))
    grid = cv2.resize(cv2.GaussianBlur(sig, (0, 0), 2), (24, 8), interpolation=cv2.INTER_AREA)
    grid = grid.ravel()
    return g / (np.linalg.norm(g) + 1e-6), grid / (np.linalg.norm(grid) + 1e-6)


def distance(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    return float((1 - a[0] @ b[0]) + (1 - a[1] @ b[1]))


class SignatureStore:
    """Reference signatures, one folder per account: ``<root>/<account>/<n>.png``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def accounts(self) -> dict[str, int]:
        if not self.root.is_dir():
            return {}
        return {
            d.name: len(list(d.glob("*.png"))) for d in sorted(self.root.iterdir()) if d.is_dir()
        }

    def add(self, account: str, sig: np.ndarray) -> Path:
        folder = self.root / account
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{len(list(folder.glob('*.png'))) + 1:03d}.png"
        cv2.imwrite(str(path), (sig * 255).astype(np.uint8))
        return path

    def references(self, account: str) -> list[np.ndarray]:
        folder = self.root / account
        if not folder.is_dir():
            return []
        out = []
        for p in sorted(folder.glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                out.append(img.astype(np.float32) / 255)
        return out

    def compare(self, account: str, sig: np.ndarray) -> tuple[float, int] | None:
        """(smallest distance to a reference, number of references), or None if none."""
        refs = self.references(account)
        if not refs:
            return None
        q = describe(sig)
        return min(distance(q, describe(r)) for r in refs), len(refs)
