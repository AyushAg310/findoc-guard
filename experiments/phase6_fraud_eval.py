"""Phase 6 experiment: duplicate detection and signature comparison on synthetic cheque photos.

    python experiments/phase6_fraud_eval.py --cheques 20 --signers 30

**Duplicates.** ``--cheques`` cheques are each photographed twice (different angle, light,
blur: two independent ``photograph`` calls of the same rendered leaf). The first photo of each
is registered in a fresh index; then every second photo is queried (should match its own
first photo) and every first photo of *another* cheque is compared too (should match nothing).
Reported: detection rate, false matches, and the inlier counts of both groups.

**Signatures.** ``--signers`` synthetic people (:mod:`findoc.synth.signature`). Each signs three
reference cheques and one more genuine cheque; a skilled forger copies the signature on one
cheque; a random forger signs their own way on another. All are photographed and run through
the pipeline up to registration, and the questioned signature's distance to the references is
recorded. The first half of the signers choose the threshold (the 95th percentile of genuine
distances, i.e. at most 5% of genuine signatures rejected); the second half measure it.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from findoc.fraud.duplicates import DuplicateIndex, ink_image
from findoc.fraud.signature import describe, distance, signature_image
from findoc.io import save_image
from findoc.localize.templates import default_store
from findoc.pipeline import default_pipeline
from findoc.synth.cheque import render_cheque
from findoc.synth.dataset import CHEQUE_FILL, CHEQUE_PHOTO_SIZE
from findoc.synth.photo import photograph
from findoc.synth.signature import Signer

STORE = default_store(None)


def register_pipeline():  # type: ignore[no-untyped-def]
    pipe = default_pipeline(ocr=False)
    keep = {"load", "video", "track", "quality", "detect", "recon3d", "rectify", "enhance", "align"}
    pipe.stages = [s for s in pipe.stages if s[0] in keep]
    return pipe


def shoot(page: np.ndarray, seed: tuple, tmp: Path, blur: float = 0.0):  # type: ignore[no-untyped-def]
    photo, _ = photograph(
        page, np.random.default_rng(seed), size=CHEQUE_PHOTO_SIZE, fill=CHEQUE_FILL, blur_sigma=blur
    )
    path = tmp / ("_".join(map(str, seed)) + ".png")
    save_image(path, photo)
    doc = register_pipeline().run(path)
    return doc if doc.registered is not None and doc.template_id else None


def duplicates(n: int, tmp: Path) -> dict:
    index = DuplicateIndex(tmp / "index")
    firsts, seconds = {}, {}
    for i in range(n):
        page, truth = render_cheque(np.random.default_rng(700_000 + i))
        a = shoot(page, (i, 0), tmp)
        b = shoot(page, (i, 1), tmp, blur=0.6)
        if a is None or b is None:
            continue
        ref = STORE.get(a.template_id).reference
        firsts[i] = (a.template_id, ink_image(a.registered, ref))
        seconds[i] = (b.template_id, ink_image(b.registered, STORE.get(b.template_id).reference))
        index.add(f"cheque_{i}", *firsts[i])
    same, other, found, false = [], [], 0, 0
    for i, (tid, ink) in seconds.items():
        matches = index.search(tid, ink)
        own = [m for m in matches if m.entry.source == f"cheque_{i}"]
        rest = [m for m in matches if m.entry.source != f"cheque_{i}"]
        if own:
            same.append(own[0].inliers)
            found += own[0].duplicate
        other += [m.inliers for m in rest]
        false += sum(m.duplicate for m in rest)
    return {
        "cheques": len(seconds),
        "detection_rate": round(found / max(1, len(seconds)), 4),
        "false_matches": int(false),
        "comparisons_with_other_cheques": len(other),
        "inliers_same_cheque": {"min": min(same, default=0), "median": float(np.median(same))},
        "inliers_other_cheques": {
            "max": max(other, default=0),
            "median": float(np.median(other)) if other else 0,
        },
    }


def leaf(p: int, k: int, who: Signer, tmp: Path, skilled: bool = False) -> np.ndarray | None:
    """Signer ``p``'s k-th cheque, signed by ``who``, photographed and registered."""
    rng = np.random.default_rng((p, k))
    page, _ = render_cheque(rng, signer=who, skilled_forgery=skilled)
    doc = shoot(page, (p, k, 9), tmp, blur=float(rng.uniform(0, 0.8)))
    if doc is None:
        return None
    return signature_image(doc.registered, STORE.get(doc.template_id))


def signatures(n: int, tmp: Path) -> dict:
    rows = []
    for p in range(n):
        signer = Signer.random(np.random.default_rng(800_000 + p))
        other = Signer.random(np.random.default_rng(900_000 + p))

        refs = [r for r in (leaf(p, k, signer, tmp) for k in range(3)) if r is not None]
        if not refs:
            continue
        ref_d = [describe(r) for r in refs]
        for kind, sig in (
            ("genuine", leaf(p, 3, signer, tmp)),
            ("skilled", leaf(p, 4, signer, tmp, True)),
            ("random", leaf(p, 5, other, tmp)),
        ):
            if sig is not None:
                d = min(distance(describe(sig), r) for r in ref_d)
                rows.append({"signer": p, "kind": kind, "distance": round(d, 4)})
        print(
            f"signer {p}: "
            + ", ".join(f"{r['kind']} {r['distance']:.3f}" for r in rows if r["signer"] == p),
            flush=True,
        )
    half = n // 2
    calib = [r for r in rows if r["signer"] < half]
    test = [r for r in rows if r["signer"] >= half]
    gen_c = [r["distance"] for r in calib if r["kind"] == "genuine"]
    threshold = float(np.percentile(gen_c, 95)) if gen_c else None
    out = {"signers": n, "threshold_from_calibration": round(threshold, 4) if threshold else None}
    for kind in ("skilled", "random"):
        g = [r["distance"] for r in test if r["kind"] == "genuine"]
        f = [r["distance"] for r in test if r["kind"] == kind]
        if g and f:
            out[f"auc_genuine_vs_{kind}"] = round(
                float(roc_auc_score([0] * len(g) + [1] * len(f), g + f)), 4
            )
            if threshold:
                out[f"{kind}_detected"] = round(float(np.mean(np.array(f) > threshold)), 4)
    g = [r["distance"] for r in test if r["kind"] == "genuine"]
    if g and threshold:
        out["genuine_rejected"] = round(float(np.mean(np.array(g) > threshold)), 4)
    out["rows"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cheques", type=int, default=20)
    ap.add_argument("--signers", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("experiments/results/phase6_fraud.json"))
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        result = {
            "duplicates": duplicates(a.cheques, Path(d)),
            "signatures": signatures(a.signers, Path(d)),
        }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in result.items()},
            indent=2,
        )
    )
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
