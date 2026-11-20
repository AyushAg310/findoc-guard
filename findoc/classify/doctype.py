"""Document type from the flat page image: cheque, invoice, receipt or other (FR-EXT-01, CO-2).

Different document types look different before a single word is read: a cheque is a wide,
tinted leaf with a printed security pattern; an invoice is a tall white A4 page with a ruled
table; a till receipt is a very narrow strip of short lines; other documents (letters, forms)
are pages of text. ``page_features`` turns a flat page into 641 numbers describing exactly that:

* **shape**: aspect ratio (width / height);
* **layout**: a 24x24 grey thumbnail (where the ink is), and ink profiles down the page (32
  rows) and across it (16 columns);
* **colour**: mean saturation and an 8-bin hue histogram of coloured pixels (cheque tints);
* **structure**: the fraction of pixels on long horizontal and vertical lines (tables, boxes),
  found by morphological opening with long thin kernels, and edge density.

A gradient-boosted tree classifier learns the classes from synthetic pages of each type
(``findoc classify train``, about a minute on a CPU), photographed and flattened back so they
look like the pipeline's flat pages. Like the tamper model, the file stays out of git.

The pipeline trusts the cheque template match (Phase 2) over this model for cheques, and
falls back to text cues ("TAX INVOICE", "GSTIN", "HSN") after OCR when no model is trained.
"""

from __future__ import annotations

import pickle
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("cheque", "invoice", "receipt", "other")
MODEL_VERSION = 1


def page_features(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray,
        (256, int(256 * h / w)) if w > h else (int(256 * w / h), 256),
        interpolation=cv2.INTER_AREA,
    )
    ink = (small < np.median(small) - 40).astype(np.float32)
    thumb = cv2.resize(gray, (24, 24), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    rows = cv2.resize(ink.mean(axis=1, keepdims=True), (1, 32), interpolation=cv2.INTER_AREA)
    cols = cv2.resize(ink.mean(axis=0, keepdims=True), (16, 1), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255
    coloured = sat > 0.12
    hue = np.histogram(hsv[..., 0][coloured], bins=8, range=(0, 180))[0] / max(1, coloured.sum())
    bw = (small < np.median(small) - 30).astype(np.uint8)
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, max(8, small.shape[1] // 6)), np.uint8))
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((max(8, small.shape[0] // 6), 1), np.uint8))
    edges = cv2.Canny(small, 80, 160)
    return np.concatenate(
        [
            [
                w / h,
                np.log(w / h),
                sat.mean(),
                sat.std(),
                coloured.mean(),
                hl.mean(),
                vl.mean(),
                (edges > 0).mean(),
                ink.mean(),
            ],
            hue,
            thumb.ravel(),
            rows.ravel(),
            cols.ravel(),
        ]
    ).astype(np.float32)


def synthetic_page(cls: str, seed: int) -> np.ndarray:
    """A flat page of class ``cls`` as the pipeline would see it: rendered, photographed
    (perspective, light, noise, blur, JPEG) and flattened back with the true homography."""
    from findoc.synth.photo import photograph

    rng = np.random.default_rng((seed, CLASSES.index(cls)))
    if cls == "cheque":
        from findoc.synth.cheque import render_cheque

        page, _ = render_cheque(rng)
    elif cls == "invoice":
        from findoc.synth.invoice import render_invoice

        page, _, _ = render_invoice(rng)
    elif cls == "receipt":
        from findoc.synth.receipt import render_receipt

        page = render_receipt(rng)
    else:
        from findoc.synth.photo import render_page

        page = render_page(
            rng, width=int(rng.integers(700, 1000)), height=int(rng.integers(900, 1300))
        )
        if rng.random() < 0.5:  # a plain letter: lines of text, no table
            page[:] = 250
            y = 120
            while y < page.shape[0] - 80:
                x1 = int(rng.integers(page.shape[1] // 2, page.shape[1] - 60))
                for x in range(60, x1, int(rng.integers(30, 70))):
                    cv2.rectangle(
                        page, (x, y - 14), (x + int(rng.integers(15, 55)), y), (40, 40, 40), -1
                    )
                y += int(rng.integers(30, 48))
    ph, pw = page.shape[:2]
    size = (1600, 1200) if pw > ph else (1200, 1600)
    photo, quad = photograph(
        page, rng, size=size, fill=(0.35, 0.7), blur_sigma=float(rng.uniform(0, 1.2))
    )
    src = np.float32([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]])
    hmat = cv2.getPerspectiveTransform(quad.astype(np.float32), src)
    scale = min(1.0, 1400 / max(pw, ph))
    out = (int(pw * scale), int(ph * scale))
    return cv2.warpPerspective(photo, np.diag([scale, scale, 1]) @ hmat, out)


class DocTypeModel:
    def __init__(self, classifier: object, meta: dict | None = None) -> None:
        self.classifier = classifier
        self.meta = meta or {}

    def predict(self, img: np.ndarray) -> tuple[str, float, dict[str, float]]:
        p = self.classifier.predict_proba(page_features(img)[None])[0]
        probs = {
            CLASSES[int(c)]: float(v) for c, v in zip(self.classifier.classes_, p, strict=True)
        }
        best = max(probs, key=probs.get)
        return best, probs[best], probs

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            pickle.dumps(
                {
                    "version": MODEL_VERSION,
                    "classes": CLASSES,
                    "model": self.classifier,
                    "meta": self.meta,
                }
            )
        )
        return path

    @staticmethod
    def load(path: Path) -> DocTypeModel:
        data = pickle.loads(path.read_bytes())  # noqa: S301 - only load models you trained
        if data.get("version") != MODEL_VERSION or tuple(data.get("classes", ())) != CLASSES:
            raise ValueError(f"{path} was trained by another version; run `findoc classify train`")
        return DocTypeModel(data["model"], data["meta"])


@lru_cache(maxsize=2)
def load_model(path: Path) -> DocTypeModel | None:
    return DocTypeModel.load(path) if path.is_file() else None


def _sample(args: tuple[str, int]) -> np.ndarray:
    return page_features(synthetic_page(*args))


def train(
    out: Path = Path("models/doctype.pkl"),
    per_class: int = 150,
    val_per_class: int = 50,
    workers: int = -1,
    seed: int = 0,
    log=print,  # noqa: ANN001
) -> DocTypeModel:
    import os
    from concurrent.futures import ProcessPoolExecutor

    from sklearn.ensemble import HistGradientBoostingClassifier

    t0 = time.perf_counter()
    if workers < 0:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    def make(base: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        jobs = [(c, base + i) for c in CLASSES for i in range(n)]
        if workers > 1:
            with ProcessPoolExecutor(workers) as pool:
                x = list(pool.map(_sample, jobs, chunksize=4))
        else:
            x = [_sample(j) for j in jobs]
        return np.stack(x), np.array([CLASSES.index(c) for c, _ in jobs])

    log(f"Rendering {per_class} training pages per class ...")
    x, y = make(seed * 1_000_000, per_class)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    clf.fit(x, y)
    log(f"Validating on {val_per_class} other pages per class ...")
    xv, yv = make(seed * 1_000_000 + 500_000, val_per_class)
    pred = clf.predict(xv)
    confusion = {
        CLASSES[t]: {CLASSES[p]: int(((yv == t) & (pred == p)).sum()) for p in range(len(CLASSES))}
        for t in range(len(CLASSES))
    }
    meta = {
        "per_class": per_class,
        "val_per_class": val_per_class,
        "accuracy": round(float((pred == yv).mean()), 4),
        "confusion": confusion,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    model = DocTypeModel(clf, meta)
    model.save(out)
    return model
