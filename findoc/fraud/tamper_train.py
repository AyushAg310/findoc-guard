"""Train the tamper model on synthetic clean and forged cheques (``findoc tamper train``).

    findoc tamper train                     # ~3-5 min on a laptop CPU, writes models/tamper.pkl

Each training cheque is rendered, photographed (random camera JPEG quality 80-95) and either
left clean (half of those re-saved as JPEG, like an innocent re-upload) or forged with one of
the three manipulations of :mod:`findoc.synth.tamper` in a handwritten field, then re-saved.
Every 16x16 block on the page becomes one training example: its 35 forensic features
(:func:`findoc.fraud.tamper.block_features`) and whether the forger changed it.

Validation uses different seeds. Each validation cheque gets one score: the highest field
score (90th percentile of block probabilities inside each field box). The threshold is the
smallest value that keeps false alarms on clean validation cheques at or below
``target_fpr`` (never below 0.5); detection rate and AUC at that threshold are recorded in
the model file and printed.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from findoc.fraud.tamper import FIELD_WEIGHTS, TamperModel, block_features, blocks, upsample


def dewarp_sample(s: dict, size: tuple[int, int] = (1624, 736)) -> dict:
    """The same sample seen as a 2D-dewarped page (for the "detect on the flat page" baseline):
    image, mask and page warped by the true page -> image homography, inverted."""
    hinv = np.linalg.inv(s["page_to_image"])
    warp = lambda a, f: cv2.warpPerspective(a, hinv, size, flags=f)  # noqa: E731
    return s | {
        "image": warp(s["image"], cv2.INTER_CUBIC),
        "mask": warp(s["mask"].astype(np.uint8), cv2.INTER_NEAREST) > 0,
        "page": np.ones(size[::-1], bool),
        "page_to_image": np.eye(3),
    }


def _sample_blocks(args: tuple[int, bool, str]) -> tuple[np.ndarray, np.ndarray]:
    from findoc.synth.tamper import forged_sample

    seed, forged, view = args
    s = forged_sample(seed, forged)
    if view == "flat":
        s = dewarp_sample(s)
    pb = blocks(s["page"].astype(np.float32)) > 0.9
    feats = block_features(s["image"], s["page"])
    labels = blocks(s["mask"].astype(np.float32)) > 0.25
    return feats[pb], labels[pb]


def field_quads(sample: dict) -> dict[str, np.ndarray]:
    from findoc.synth.cheque import BANKS, field_specs
    from findoc.synth.tamper import box_quad

    return {
        f.name: box_quad(f.box, sample["page_to_image"])
        for f in field_specs(BANKS[sample["bank"]])
        if f.name in FIELD_WEIGHTS
    }


def capture_score(model: TamperModel, sample: dict) -> tuple[float, str, float]:
    """(weighted top field score, its name, pixel AUC or nan) for one synthetic capture,
    scored directly in the capture (ground-truth field positions)."""
    img = sample["image"]
    prob = upsample(model.block_probability(img, sample["page"]), img.shape[:2])
    best, name = -1.0, ""
    for fname, quad in field_quads(sample).items():
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 1)
        v = prob[m > 0]
        if v.size:
            s = float(np.percentile(v, 90)) * FIELD_WEIGHTS[fname]
            if s > best:
                best, name = s, fname
    auc = float("nan")
    if sample["mask"].any():
        from sklearn.metrics import roc_auc_score

        page = sample["page"]
        auc = float(roc_auc_score(sample["mask"][page][::7], prob[page][::7]))
    return best, name, auc


def _validate(
    model_bytes: bytes, seed: int, forged: bool, view: str
) -> tuple[float, str, float, str]:
    import pickle

    from findoc.synth.tamper import forged_sample

    model = pickle.loads(model_bytes)  # noqa: S301 - our own model, passed between processes
    s = forged_sample(seed, forged)
    if view == "flat":
        s = dewarp_sample(s)
    score, top, auc = capture_score(model, s)
    return score, top, auc, s["field"] or ""


def train(
    out: Path = Path("models/tamper.pkl"),
    samples: int = 80,
    val_samples: int = 40,
    seed: int = 0,
    workers: int = -1,
    target_fpr: float = 0.02,
    max_iter: int = 300,
    view: str = "original",
    log=print,  # noqa: ANN001
) -> TamperModel:
    """``view="flat"`` trains on 2D-dewarped pages instead (only for the Phase 6 experiment's
    baseline; the pipeline always uses a model trained on original captures)."""
    import pickle

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    t0 = time.perf_counter()
    if workers < 0:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    jobs = [(seed * 100_000 + 10_000 + i, f, view) for i in range(samples) for f in (False, True)]
    log(f"Rendering {len(jobs)} training captures with {workers} workers ...")
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            parts = list(pool.map(_sample_blocks, jobs, chunksize=2))
    else:
        parts = [_sample_blocks(j) for j in jobs]
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    log(f"{len(x):,} blocks, {y.mean():.1%} edited; fitting gradient-boosted trees ...")
    clf = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=31, class_weight="balanced",
        random_state=seed,
    )  # fmt: skip
    clf.fit(x, y)
    model = TamperModel(clf)

    log(f"Validating on {2 * val_samples} unseen captures ...")
    vjobs = [(seed * 100_000 + 50_000 + i, f) for i in range(val_samples) for f in (False, True)]
    blob = pickle.dumps(model)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            res = list(
                pool.map(
                    _validate, [blob] * len(vjobs), *zip(*vjobs, strict=True), [view] * len(vjobs)
                )
            )
    else:
        res = [_validate(blob, s, f, view) for s, f in vjobs]
    clean = np.array([r[0] for (_, f), r in zip(vjobs, res, strict=True) if not f])
    forged = np.array([r[0] for (_, f), r in zip(vjobs, res, strict=True) if f])
    top_ok = [r[1] == r[3] for (_, f), r in zip(vjobs, res, strict=True) if f]
    pixel_auc = [r[2] for r in res if np.isfinite(r[2])]
    k = int(np.floor(target_fpr * len(clean)))  # clean captures allowed above the threshold
    threshold = max(0.5, float(np.sort(clean)[::-1][k]) + 1e-3) if len(clean) else 0.9
    model.threshold = round(min(threshold, 0.99), 3)
    model.meta = {
        "view": view,
        "samples": samples,
        "val_samples": val_samples,
        "blocks": int(len(x)),
        "image_auc": round(
            float(roc_auc_score([0] * len(clean) + [1] * len(forged), [*clean, *forged])), 4
        ),  # fmt: skip
        "pixel_auc": round(float(np.mean(pixel_auc)), 4) if pixel_auc else None,
        "detection_rate": round(float((forged >= model.threshold).mean()), 4),
        "false_alarm_rate": round(float((clean >= model.threshold).mean()), 4),
        "top_field_correct": round(float(np.mean(top_ok)), 4),
        "threshold": model.threshold,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    model.save(out)
    return model
