"""Phase 5 evaluation: 3D flattening vs. the 2D homography on synthetic bent cheques.

For each kind of bend (flat, wave, curl of one end, curl of both ends, fold) and several random
seeds, a hand-held video is rendered with exact ground truth (:mod:`findoc.synth.surface`), and
the full pipeline runs twice: with 3D reconstruction (``FINDOC_RECON3D=auto``) and without
(``off``, i.e. Phase 1-4 behaviour). Measured:

* **unwarping error**: SIFT features of the flattened page are matched to the true flat cheque;
  after the best homography (least median of squares; it removes position, scale and any
  residual perspective), the remaining displacement of *all* matched features is distortion the
  flattening did not undo. Median and 90th percentile, as % of the cheque width. (A first
  version kept only RANSAC inliers within 40 px, which silently dropped the worst-distorted
  part of the page and favoured 2D on folds.)
* **bank template identified** on the flattened page (Phase 2);
* **3D accuracy**: distance of reconstructed page points to the true surface after the best
  similarity transform, % of the cheque width;
* the **decision**: whether 3D was used, and why not when it wasn't (flat pages should keep 2D);
* time.

    python experiments/phase5_recon_eval.py --seeds 3

Results are printed as Markdown and saved to experiments/results/phase5.json.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from findoc.config import get_settings
from findoc.pipeline import process
from findoc.synth.camera import Pose, project
from findoc.synth.surface import Bend, DocSurface, random_bend
from findoc.synth.video import cheque_video, to_world, write_video

TODAY = date(2026, 11, 17)
KINDS = ("flat", "wave", "curl", "curl2", "fold")


def unwarp_error(flat: np.ndarray | None, texture: np.ndarray) -> tuple[float | None, float | None]:
    if flat is None:
        return None, None
    sift = cv2.SIFT_create(4000)
    w = texture.shape[1]
    f = cv2.resize(flat, (w, max(1, int(w * flat.shape[0] / flat.shape[1]))))
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = sift.detectAndCompute(cv2.cvtColor(texture, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None:
        return None, None
    m = [
        a
        for a, *b in cv2.BFMatcher().knnMatch(d1, d2, k=2)
        if b and a.distance < 0.75 * b[0].distance
    ]
    if len(m) < 20:
        return None, None
    p1 = np.float32([k1[x.queryIdx].pt for x in m])
    p2 = np.float32([k2[x.trainIdx].pt for x in m])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 40.0)
    if H is None:
        return None, None
    r = np.linalg.norm(
        cv2.perspectiveTransform(p1.reshape(-1, 1, 2), H).reshape(-1, 2) - p2, axis=1
    )
    r = r[inl.ravel() > 0]
    return float(np.median(r) / w * 100), float(np.percentile(r, 90) / w * 100)


def accuracy_3d(doc, truth, bend: Bend, texture: np.ndarray) -> float | None:  # type: ignore[no-untyped-def]
    """Median distance of reconstructed page points to the true surface, % of cheque width."""
    res = doc.recon
    if res is None or res.rec is None or res.doc_ids is None or not doc.recon3d.get("frames"):
        return None
    rec = res.rec
    frame = doc.recon3d["frames"][0]  # reference view = best frame
    K = np.array(truth.meta["K"])
    p = truth.meta["poses"][frame]
    pose = Pose(np.array(p["R"]), np.array(p["t"]))
    h, w = texture.shape[:2]
    surf = DocSurface(w, h, bend)
    st, _ = surf.grid(500, 250)
    world = to_world(surf, st)
    proj = project(K, pose, world)
    ref = rec.views[0]
    pts = np.array([rec.points[t] for t in res.doc_ids])
    obs = rec.project(ref, pts)
    _, nn = cKDTree(proj).query(obs)
    gt = world[nn]
    keep = np.ones(len(pts), bool)
    for _ in range(8):  # trimmed fit, see _similarity
        fit = _similarity(pts[keep], gt[keep])
        err = np.linalg.norm(fit(pts) - gt, axis=1)
        keep = err <= max(2.5 * np.median(err), 1e-9)
    return float(np.median(err) / w * 100)


def _similarity(src: np.ndarray, dst: np.ndarray):  # type: ignore[no-untyped-def]
    """Least-squares similarity (Umeyama) mapping ``src`` onto ``dst``. Used trimmed: a desk
    point that projects inside the page outline is matched to the page surface far above it,
    and a handful of those drag a plain least-squares fit by several percent (an earlier
    version of this metric reported 0.5-1% errors that were really ~0.2%)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s0, d0 = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    sc = (S * np.diag(D)).sum() / ((s0**2).sum() / len(src))
    return lambda x: sc * (x - mu_s) @ R.T + mu_d


def run(seeds: int, seconds: float, calibrated: bool = False) -> dict:
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for kind in KINDS:
            for s in range(seeds):
                seed = 81_000 + 100 * KINDS.index(kind) + s
                rng = np.random.default_rng(seed)
                bend = random_bend(kind, rng)
                frames, truth, cheque = cheque_video(
                    seed, None, TODAY, seconds=seconds, fps=15, subframes=2, jerks=1, bend=bend
                )
                from findoc.synth.cheque import render_cheque

                texture, _ = render_cheque(
                    np.random.default_rng(seed), bank=cheque.bank, today=TODAY
                )
                path = write_video(Path(d) / f"{kind}_{s}.mp4", frames, truth.fps)
                row = {
                    "kind": kind,
                    "seed": seed,
                    "lift": truth.meta.get("lift", 0.0),
                    "bank": cheque.bank,
                }
                if calibrated:  # the camera's focal length is known (calibrated phone)
                    os.environ["FINDOC_FOCAL_PX"] = str(np.array(truth.meta["K"])[0, 0])
                for mode in ("off", "auto"):
                    os.environ["FINDOC_RECON3D"] = mode
                    get_settings.cache_clear()
                    t0 = time.perf_counter()
                    doc = process(path, ocr=False)
                    key = "2d" if mode == "off" else "3d"
                    med, p90 = unwarp_error(doc.flat, texture)
                    row[f"{key}_err"] = med
                    row[f"{key}_err_p90"] = p90
                    row[f"{key}_template_ok"] = doc.template_id == cheque.bank
                    row[f"{key}_s"] = round(time.perf_counter() - t0, 1)
                    if mode == "auto":
                        row["used_3d"] = doc.flattening == "3d"
                        row["reason"] = doc.recon3d.get("reason")
                        row["plane_residual_px"] = doc.recon3d.get("plane_residual_px")
                        row["focal_px"] = doc.recon3d.get("focal_px")
                        row["true_focal_px"] = float(np.array(truth.meta["K"])[0, 0])
                        row["stretch"] = doc.recon3d.get("stretch")
                        row["acc3d"] = (
                            accuracy_3d(doc, truth, bend, texture)
                            if doc.recon is not None
                            else None
                        )
                rows.append(row)
                print(
                    f"{kind:6s} seed {seed} lift {row['lift']:6.1f}: 3D used {row['used_3d']!s:5s} "
                    f"err 2D {row['2d_err']} 3D {row['3d_err']}  ({row['reason']})",
                    flush=True,
                )
    os.environ.pop("FINDOC_RECON3D", None)
    os.environ.pop("FINDOC_FOCAL_PX", None)
    get_settings.cache_clear()

    def mean(key: str, sub: list[dict]) -> float | None:
        vals = [r[key] for r in sub if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {}
    for kind in KINDS:
        sub = [r for r in rows if r["kind"] == kind]
        summary[kind] = {
            "clips": len(sub),
            "used_3d": mean("used_3d", sub),
            "err_2d": mean("2d_err", sub),
            "err_3d": mean("3d_err", sub),
            "err_p90_2d": mean("2d_err_p90", sub),
            "err_p90_3d": mean("3d_err_p90", sub),
            "template_2d": mean("2d_template_ok", sub),
            "template_3d": mean("3d_template_ok", sub),
            "acc3d": mean("acc3d", sub),
            "s_2d": mean("2d_s", sub),
            "s_3d": mean("3d_s", sub),
        }
    return {
        "seeds": seeds,
        "seconds": seconds,
        "calibrated": calibrated,
        "summary": summary,
        "clips": rows,
    }


def to_markdown(r: dict) -> str:
    def pct(x: float | None, d: int = 3) -> str:
        return "n/a" if x is None else f"{x:.{d}f}%"

    lines = [
        f"### 3D flattening vs. 2D homography ({r['seeds']} synthetic hand-held videos per bend, "
        f"{r['seconds']:.0f} s, 1280x720, focal length "
        f"{'known (calibrated camera)' if r.get('calibrated') else 'estimated from the video'})",
        "",
        "Unwarping error = displacement left after the best homography between the flattened page "
        "and the true flat cheque (% of cheque width; median / 90th percentile).",
        "",
        "| Bend | 3D used | Error 2D | Error 3D | Template 2D | Template 3D | 3D point error "
        "| Time 2D / 3D |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for kind, s in r["summary"].items():
        lines.append(
            f"| {kind} | {s['used_3d']:.0%} | {pct(s['err_2d'])} / {pct(s['err_p90_2d'])} | "
            f"{pct(s['err_3d'])} / {pct(s['err_p90_3d'])} | {s['template_2d']:.0%} | "
            f"{s['template_3d']:.0%} | {pct(s['acc3d'], 2)} | {s['s_2d']} / {s['s_3d']} s |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument(
        "--calibrated", action="store_true", help="give the pipeline the true focal length"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    results = run(args.seeds, args.seconds, args.calibrated)
    args.out = args.out or Path(
        "experiments/results/phase5_calibrated.json"
        if args.calibrated
        else "experiments/results/phase5.json"
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(to_markdown(results))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
