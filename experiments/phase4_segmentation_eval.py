"""Phase 4 evaluation: learned vs. classical page detection on hard backgrounds.

Part A - synthetic test set (no training documents or seeds reused): ``--per-kind`` images for
each background type in :mod:`findoc.synth.backgrounds` (pale wood, light table, bright board,
patterned cloth, white-text poster, clutter, noise, ordinary desk), rendered at phone-video
size (1280x720 or 720x1280). For the classical detector, the learned detector, and the learned
detector without edge refinement:

* **success rate**: a quadrilateral was returned and its corners are within 2% of the image
  diagonal of the truth on average;
* **corner error** (% of diagonal) on successes;
* time per image;
* **false "whole document"** on a separate set where the document runs off the frame: how
  often the detector returns an outline clear of the image border, i.e. wrongly claims to see
  the whole page (the tracker would then follow a partial page, as in the real videos).

Part B - real videos (``--video``, optional): runs the full pipeline with the classical and the
learned detector and reports status, usable frames, keyframes and the flattened page's aspect
ratio. No ground truth: this is a qualitative check on real captures.

    python experiments/phase4_segmentation_eval.py --model models/docseg.onnx
    python experiments/phase4_segmentation_eval.py --model models/docseg.onnx --video myvideo3.mp4

Results are printed as Markdown and saved to experiments/results/phase4.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from findoc.preprocess.detect import find_document_classical, order_corners
from findoc.segment.infer import Segmenter
from findoc.synth.backgrounds import KINDS
from findoc.synth.segdata import document_pool, make_sample

SUCCESS = 0.02  # mean corner error below 2% of the diagonal


def corner_error(pred: np.ndarray, truth: np.ndarray, diag: float) -> float:
    return float(np.linalg.norm(order_corners(pred) - truth, axis=1).mean() / diag)


def evaluate_synthetic(model: Segmenter, per_kind: int, seed: int) -> dict:
    pool = document_pool(seed=4242, n=30)  # documents never used in training
    methods = ("classical", "learned", "learned_no_refine")
    rows: dict[str, dict[str, list]] = {m: {k: [] for k in KINDS} for m in methods}
    times: dict[str, list[float]] = {m: [] for m in methods}
    for k_i, kind in enumerate(KINDS):
        for i in range(per_kind):
            rng = np.random.default_rng((seed, k_i, i))
            s = make_sample(rng, pool, size=(1280, 720), kind=kind, partial_rate=0.0)
            diag = float(np.hypot(*s.image.shape[:2]))
            for m in methods:
                t0 = time.perf_counter()
                if m == "classical":
                    det = find_document_classical(s.image)
                    pred = None if det.method == "full_frame" else det.corners
                else:
                    seg = model.segment(s.image, refine=(m == "learned"))
                    pred = seg.corners if seg.agreement >= 0.9 else None
                times[m].append((time.perf_counter() - t0) * 1000)
                rows[m][kind].append(None if pred is None else corner_error(pred, s.corners, diag))

    # documents that run off the frame: a correct detector must not claim a whole page
    from findoc.track.klt import _clear_of_border

    false_whole: dict[str, list[bool]] = {m: [] for m in methods}
    for k_i, kind in enumerate(KINDS):
        for i in range(max(1, per_kind // 2)):
            rng = np.random.default_rng((seed + 1, k_i, i))
            s = make_sample(rng, pool, size=(1280, 720), kind=kind, partial_rate=1.0)
            shape = s.image.shape[:2]
            for m in methods:
                if m == "classical":
                    det = find_document_classical(s.image)
                    pred = None if det.method == "full_frame" else det.corners
                else:
                    seg = model.segment(s.image, refine=(m == "learned"))
                    pred = seg.corners if seg.agreement >= 0.9 else None
                false_whole[m].append(pred is not None and _clear_of_border(pred, shape))

    def summarise(errs: list) -> dict:
        ok = [e for e in errs if e is not None and e < SUCCESS]
        return {
            "n": len(errs),
            "success": round(len(ok) / max(len(errs), 1), 4),
            "corner_err_pct": round(100 * float(np.mean(ok)), 3) if ok else None,
        }

    out = {}
    for m in methods:
        every = [e for k in KINDS for e in rows[m][k]]
        out[m] = {
            "overall": summarise(every),
            "by_background": {k: summarise(rows[m][k]) for k in KINDS},
            "ms_per_image": round(float(np.median(times[m])), 1),
            "false_whole_on_cut_off": round(float(np.mean(false_whole[m])), 4),
            "cut_off_samples": len(false_whole[m]),
        }
    return out


def evaluate_videos(videos: list[Path]) -> dict:
    from findoc.config import get_settings
    from findoc.pipeline import process

    out: dict = {}
    for v in videos:
        out[v.name] = {}
        for mode in ("off", "auto"):
            os.environ["FINDOC_SEGMENTATION"] = mode
            get_settings.cache_clear()
            doc = process(v, ocr=False)
            aspect = None
            if doc.flat is not None:
                h, w = doc.flat.shape[:2]
                aspect = round(max(w, h) / min(w, h), 2)
            out[v.name]["classical" if mode == "off" else "learned"] = {
                "status": doc.status.value,
                "usable_frames": doc.tracking.get("usable"),
                "frames": len(doc.track),
                "keyframes": len(doc.keyframes),
                "flat_aspect": aspect,
                "template": doc.template_id,
                "messages": doc.messages[-2:],
            }
    os.environ.pop("FINDOC_SEGMENTATION", None)
    get_settings.cache_clear()
    return out


def to_markdown(r: dict) -> str:
    syn = r["synthetic"]
    names = {
        "classical": "Classical (Phase 1)",
        "learned": "Learned",
        "learned_no_refine": "Learned, no edge refinement",
    }
    lines = [
        f"### Page detection on hard backgrounds ({syn['learned']['overall']['n']} synthetic "
        "images, 1280x720, unseen documents)",
        "",
        "Success = corners within 2% of the image diagonal on average. "
        "Error is measured on successes.",
        "",
        "| Detector | Success | Corner error (% of diagonal) | Cut-off page called whole "
        "| Time per image |",
        "| --- | --- | --- | --- | --- |",
    ]
    for m, label in names.items():
        o = syn[m]["overall"]
        err = "n/a" if o["corner_err_pct"] is None else f"{o['corner_err_pct']:.3f}%"
        fw = syn[m]["false_whole_on_cut_off"]
        lines.append(
            f"| {label} | {o['success']:.1%} | {err} | {fw:.1%} | {syn[m]['ms_per_image']} ms |"
        )
    lines += ["", "| Background | Classical success | Learned success |", "| --- | --- | --- |"]
    for k in KINDS:
        c, lr = syn["classical"]["by_background"][k], syn["learned"]["by_background"][k]
        lines.append(f"| {k} | {c['success']:.0%} | {lr['success']:.0%} |")
    if r.get("videos"):
        lines += [
            "",
            "### Real videos (no ground truth)",
            "",
            "| Video | Detector | Status | Usable frames | Keyframes | Flat page aspect |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, res in r["videos"].items():
            for det, v in res.items():
                lines.append(
                    f"| {name} | {det} | {v['status']} | {v['usable_frames']}/{v['frames']} "
                    f"| {v['keyframes']} | {v['flat_aspect']} |"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/docseg.onnx"))
    parser.add_argument("--per-kind", type=int, default=40)
    parser.add_argument("--seed", type=int, default=900_000)
    parser.add_argument("--video", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, default=Path("experiments/results/phase4.json"))
    args = parser.parse_args()

    model = Segmenter(args.model)
    meta_path = args.model.with_suffix(".json")
    results = {
        "model": str(args.model),
        "training": json.loads(meta_path.read_text()) if meta_path.exists() else None,
        "synthetic": evaluate_synthetic(model, args.per_kind, args.seed),
    }
    if args.video:
        os.environ["FINDOC_SEGMENTER_MODEL"] = str(args.model)
        results["videos"] = evaluate_videos(args.video)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(to_markdown(results))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
