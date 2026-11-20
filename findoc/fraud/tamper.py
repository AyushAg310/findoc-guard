"""Tamper detection on the original capture, mapped onto the flat page (Phase 6, SRS 10.3/10.5).

**Why on the original.** Forensic traces are tiny statistical regularities: the 8x8 block
quantisation left by JPEG compression, sensor noise, the exact pixel grid. Dewarping resamples
every pixel (interpolation between neighbours, a different scale in different places), which
smears exactly those traces. So detection runs on the **original, un-resampled** capture, and
the resulting heatmap (not the pixels) is carried to the flat page through the mapping that made
the flat page: the homography for a photo, the 3D correspondence map (Phase 5) for a video.
``experiments/phase6_tamper_eval.py`` measures how much this matters.

**What is measured, per 16x16 block** (``block_features``, 35 numbers):

* the **JPEG error curve**: re-compress the image at qualities 50, 55, ... 95 and record the
  log mean squared change of the block at each. A region compressed before at quality Q barely
  changes when compressed at Q again (a "JPEG ghost", Farid 2009); a region pasted in or drawn
  after the camera's compression has a different history, so its curve has a different shape;
* the same curve with its mean removed (shape only);
* texture energy (log mean squared Laplacian), brightness, fraction of dark (ink) pixels and
  high-pass noise energy: how much error a block *should* show depends on its content. A
  stroke of ink changes far more under compression than plain paper, edited or not;
* the error curve and noise **relative to the typical block of the same kind** (ink or paper)
  on this capture: every phone compresses and denoises differently, so "normal" is set per
  capture.

**Why learned.** A fixed rule on the error ("flag blocks with unusually high error") fires on
every line of handwriting, because ink compresses differently from paper; on clean synthetic
cheques it flagged a field on every one. Telling "unusual for this content" from "unusual"
needs a model of what each kind of content normally looks like. A gradient-boosted tree
classifier (``HistGradientBoostingClassifier``) learns it from synthetic clean and forged
cheques with exact masks (:mod:`findoc.synth.tamper`, ``findoc tamper train``, a few minutes on
a CPU). It outputs, per block, the probability that it was edited. Model files stay out of
git, like the Phase 4 weights.

Without a trained model, ``classical_heatmap`` (the JPEG-ghost anomaly against the page) is
still computed and saved for a reviewer to look at, but ``TAMPER_REGION`` does not fire on it.

**Where it matters.** For a cheque, the map is warped into the template frame (Phase 2) and
scored inside each field box (the 90th percentile of the block probabilities, so two edited
digits are enough). Amount, payee and date weigh fully; other fields less.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

BLOCK = 16  # analysis block: a multiple of the 8x8 JPEG grid
QUALITIES = tuple(range(50, 100, 5))
FEATURE_NAMES = (
    [f"log_err_q{q}" for q in QUALITIES]
    + [f"shape_q{q}" for q in QUALITIES]
    + ["texture", "brightness", "ink_fraction", "noise"]
    + [f"rel_err_q{q}" for q in QUALITIES]
    + ["rel_noise"]
)
TAMPER_THRESHOLD = 0.9  # field score above which TAMPER_REGION fires (see the Phase 6 guide)
FIELD_WEIGHTS = {  # how much an edit in this field matters
    "amount_figures": 1.0,
    "amount_words_1": 1.0,
    "amount_words_2": 1.0,
    "payee": 1.0,
    "date": 1.0,
    "signature": 0.8,
    "account_number": 0.7,
    "ifsc": 0.6,
    "micr": 0.6,
}
MODEL_VERSION = 2


def blocks(a: np.ndarray, b: int = BLOCK) -> np.ndarray:
    """Mean over non-overlapping b x b blocks (the image is cropped to a multiple of b)."""
    h, w = a.shape[:2]
    hb, wb = h // b, w // b
    return a[: hb * b, : wb * b].reshape(hb, b, wb, b, *a.shape[2:]).mean((1, 3))


def block_features(img: np.ndarray, page: np.ndarray | None = None) -> np.ndarray:
    """(H/16, W/16, 35) forensic features of every block (see the module docstring).

    ``page`` (bool, image size): the document region; the last 11 features compare each block
    with the typical block *of the same kind on this capture* (ink or paper), because what is
    normal depends on the phone's own compression and noise."""
    x = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    err = []
    for q in QUALITIES:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        rec = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32) if ok else x
        err.append(blocks(((x - rec) ** 2).mean(axis=2)))
    log_err = np.log(np.stack(err, axis=-1) + 1e-3)
    texture = np.log(blocks(cv2.Laplacian(gray, cv2.CV_32F) ** 2) + 1)
    brightness = blocks(gray) / 255
    ink = blocks((gray < np.median(gray) - 40).astype(np.float32))
    noise = np.log(blocks((gray - cv2.GaussianBlur(gray, (0, 0), 1.0)) ** 2) + 1e-3)
    pb = np.ones(ink.shape, bool) if page is None else blocks(page.astype(np.float32)) > 0.9
    rel_err = np.zeros_like(log_err)
    rel_noise = np.zeros_like(noise)
    for cls in (ink > 0.05, ink <= 0.05):  # ink blocks vs paper blocks
        ref = pb & cls
        if ref.sum() >= 5:
            rel_err[cls] = log_err[cls] - np.median(log_err[ref], axis=0)
            rel_noise[cls] = noise[cls] - np.median(noise[ref])
    return np.concatenate(
        [
            log_err,
            log_err - log_err.mean(axis=-1, keepdims=True),
            texture[..., None],
            brightness[..., None],
            ink[..., None],
            noise[..., None],
            rel_err,
            rel_noise[..., None],
        ],
        axis=-1,
    ).astype(np.float32)


@dataclass
class TamperModel:
    classifier: object  # sklearn HistGradientBoostingClassifier
    threshold: float = TAMPER_THRESHOLD
    meta: dict = field(default_factory=dict)  # training record: samples, scores, time

    def block_probability(self, img: np.ndarray, page: np.ndarray) -> np.ndarray:
        """P(edited) per block (block grid); 0 outside ``page``."""
        feats = block_features(img, page)
        pb = blocks(page.astype(np.float32)) > 0.9
        prob = np.zeros(pb.shape, np.float32)
        if pb.any():
            prob[pb] = self.classifier.predict_proba(feats[pb])[:, 1]
        return prob

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": MODEL_VERSION, "features": FEATURE_NAMES, "model": self.classifier,
                "threshold": self.threshold, "meta": self.meta}  # fmt: skip
        path.write_bytes(pickle.dumps(data))
        return path

    @staticmethod
    def load(path: Path) -> TamperModel:
        # pickle runs code when loading: only ever load a model file you trained yourself
        data = pickle.loads(path.read_bytes())  # noqa: S301
        if data.get("version") != MODEL_VERSION or list(data.get("features", [])) != list(
            FEATURE_NAMES
        ):
            raise ValueError(f"{path} was trained by another version; run `findoc tamper train`")
        return TamperModel(data["model"], data.get("threshold", TAMPER_THRESHOLD), data["meta"])


@lru_cache(maxsize=2)
def load_model(path: Path) -> TamperModel | None:
    return TamperModel.load(path) if path.is_file() else None


def classical_heatmap(img: np.ndarray, page: np.ndarray) -> np.ndarray:
    """Untrained fallback / baseline (block grid, [0, 1]): JPEG-ghost anomaly, i.e. how far the
    block's worst error ratio (error / page median error, max over qualities) lies above the
    page's typical value, in robust standard deviations (3 -> 0, 12 -> 1)."""
    feats = block_features(img)
    err = np.exp(feats[..., : len(QUALITIES)])
    pb = blocks(page.astype(np.float32)) > 0.9
    if pb.sum() < 10:
        return np.zeros(pb.shape, np.float32)
    ratio = (err / (np.median(err[pb], axis=0) + 1e-6)).max(axis=-1)
    med = float(np.median(ratio[pb]))
    mad = 1.4826 * float(np.median(np.abs(ratio[pb] - med))) + 1e-6
    heat = np.clip(((ratio - med) / mad - 3.0) / 9.0, 0, 1).astype(np.float32)
    heat[~pb] = 0
    return heat


def upsample(grid: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Block grid -> full image resolution (block centres interpolated linearly)."""
    h, w = shape
    full = cv2.resize(
        grid, (grid.shape[1] * BLOCK, grid.shape[0] * BLOCK), interpolation=cv2.INTER_LINEAR
    )
    return cv2.copyMakeBorder(
        full, 0, h - full.shape[0], 0, w - full.shape[1], cv2.BORDER_REPLICATE
    ).astype(np.float32)


def page_mask(shape: tuple[int, int], corners: np.ndarray, shrink: float = 0.01) -> np.ndarray:
    """The page region in the original image (slightly inside the outline, off the paper edge)."""
    c = corners.mean(axis=0)
    quad = c + (corners - c) * (1 - shrink)
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 1)
    return m.astype(bool)


def to_flat(
    heat: np.ndarray,
    flat_shape: tuple[int, int],
    homography: np.ndarray | list | None = None,
    flat_map: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Carry a full-resolution map from the original image to the flat page: through the 3D
    correspondence map (``flat_map[0][y, x], flat_map[1][y, x]`` = where flat pixel (x, y) came
    from) when there is one, otherwise through the original -> flat homography."""
    fh, fw = flat_shape
    if flat_map is not None:
        mx, my = flat_map
        return cv2.remap(heat, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)
    if homography is None:
        raise ValueError("need a homography or a flat map")
    hmat = np.asarray(homography, np.float64)
    return cv2.warpPerspective(heat, hmat, (fw, fh), flags=cv2.INTER_LINEAR)


def field_scores(
    heat_template: np.ndarray, boxes: dict[str, tuple[int, int, int, int]]
) -> dict[str, float]:
    """Score per field box: 90th percentile of the heat inside it (robust to a stray hot pixel,
    sensitive to a small edited area such as two changed digits)."""
    out = {}
    for name, (x0, y0, x1, y1) in boxes.items():
        v = heat_template[max(0, y0) : y1, max(0, x0) : x1]
        if v.size:
            out[name] = float(np.percentile(v, 90))
    return out
