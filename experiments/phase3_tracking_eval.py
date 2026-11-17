"""Phase 3 evaluation on synthetic hand-held cheque videos with exact ground truth.

Part A - tracking (no OCR needed):
    corner error against ground truth (as % of the frame diagonal), share of in-view frames
    tracked, time per frame; KLT tracking vs. running the Phase 1 page detector on every frame.
    Three capture conditions, cycled over the clips: clean (document fully visible), thumb (a
    thumb holds the cheque by a corner, hiding it) and exit (the camera pans away mid-clip and
    the document leaves the view, then returns).
Part B - keyframes:
    motion blur at the chosen keyframes vs. the same number of evenly spaced frames, and the
    range of viewing angles they cover (3D reconstruction needs both sharpness and parallax).
Part C - best frame for the 2D pipeline:
    template identification and field-box IoU when the Phase 2 pipeline runs on the selected best
    frame vs. the first frame vs. a random frame; with --real-ocr also field accuracy (EasyOCR).
    (A frame that fails the Phase 1 quality gate, e.g. document too small, counts as not
    identified: in the app the user would have to retake.)

Run from the repository root:

    python experiments/phase3_tracking_eval.py --n 16
    python experiments/phase3_tracking_eval.py --n 16 --real-ocr      # on a machine with OCR

Results are printed as Markdown and saved to experiments/results/phase3*.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from phase2_cheque_eval import FIELDS, field_correct, geometry_pipeline, polygon_iou  # noqa: E402

from findoc.io import read_frames  # noqa: E402
from findoc.localize.templates import builtin_store  # noqa: E402
from findoc.pipeline import process  # noqa: E402
from findoc.preprocess.detect import find_document  # noqa: E402
from findoc.recognize.fields import read_cheque_fields  # noqa: E402
from findoc.synth.video import cheque_video, write_video  # noqa: E402

TODAY = date(2026, 11, 17)
CONDITIONS = ("clean", "thumb", "exit")


def view_angle(center: np.ndarray) -> np.ndarray:
    """Unit viewing direction from the document centre to the camera."""
    c = np.asarray(center, float)
    return c / np.linalg.norm(c)


def angle_span(centers: list) -> float:
    """Largest angle (degrees) between any two viewing directions."""
    dirs = np.array([view_angle(c) for c in centers])
    cos = np.clip(dirs @ dirs.T, -1, 1)
    return float(np.degrees(np.arccos(cos.min())))


def downstream(frame: np.ndarray, truth_corners: np.ndarray, tmp: Path, name: str, store) -> dict:  # type: ignore[no-untyped-def]
    """Phase 1-2 geometry on one frame (as a photo): template id and field-box IoU."""
    path = tmp / f"{name}.png"
    cv2.imwrite(str(path), frame)
    doc = geometry_pipeline().run(path)
    return _score_doc(doc, truth_corners, store)


def _score_doc(doc, truth_corners: np.ndarray, store) -> dict:  # type: ignore[no-untyped-def]
    out = {"template": doc.template_id, "iou": None, "doc": doc}
    if doc.template_id is None or doc.template_homography is None:
        return out
    template = store.get(doc.template_id)
    w, h = template.size
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    g = cv2.getPerspectiveTransform(src, truth_corners.astype(np.float32))
    tmpl_to_frame = np.linalg.inv(np.array(doc.template_homography) @ np.array(doc.homography))
    ious = []
    for spec in template.fields:
        x0, y0, x1, y1 = spec.box
        box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32).reshape(-1, 1, 2)
        est = cv2.perspectiveTransform(box, tmpl_to_frame).reshape(-1, 2)
        gt = cv2.perspectiveTransform(box, g).reshape(-1, 2)
        ious.append(polygon_iou(est, gt))
    out["iou"] = float(np.mean(ious))
    return out


def evaluate(n: int, seed: int, seconds: float, real_ocr: bool) -> dict:
    store = builtin_store()
    engine = None
    if real_ocr:
        from findoc.recognize.ocr import get_engine

        engine = get_engine()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for i in range(n):
            condition = CONDITIONS[i % len(CONDITIONS)]
            frames, truth, cheque = cheque_video(
                seed + i,
                None,
                TODAY,
                seconds=seconds,
                exit_frame=condition == "exit",
                thumb=condition == "thumb",
            )
            path = write_video(tmp / f"v{i}.mp4", frames, truth.fps)
            gt = np.array(truth.corners)
            blur = np.array(truth.blur_px)
            diag = float(np.hypot(*truth.size))
            in_view = np.array(truth.in_frame)

            # ---- full pipeline (tracking + keyframes + 2D stages on the best frame)
            t0 = time.perf_counter()
            doc = process(path, ocr=False)
            total_s = time.perf_counter() - t0
            tracked = {t.frame: t for t in doc.track}
            errs = [
                np.linalg.norm(np.array(t.corners) - gt[f], axis=1).mean() / diag * 100
                for f, t in tracked.items()
                if t.corners is not None and in_view[f]
            ]
            tracked_in_view = sum(
                tracked[f].corners is not None for f in tracked if in_view[f]
            ) / max(int(sum(in_view[f] for f in tracked)), 1)

            # ---- baseline: Phase 1 page detector on every frame (decoded from the mp4)
            decoded = read_frames(path, list(range(len(frames))))
            t0 = time.perf_counter()
            det_errs, det_found = [], 0
            for f, img in decoded.items():
                det = find_document(img)
                if det.method == "full_frame" or not in_view[f]:
                    continue
                det_found += 1
                det_errs.append(np.linalg.norm(det.corners - gt[f], axis=1).mean() / diag * 100)
            det_ms = (time.perf_counter() - t0) * 1000 / len(decoded)

            # ---- keyframes vs. evenly spaced frames
            keys = [k.frame for k in doc.keyframes]
            even = np.linspace(0, len(frames) - 1, max(len(keys), 1)).round().astype(int).tolist()

            # ---- best frame vs. first / random frame for the 2D pipeline
            best = _score_doc(doc, gt[doc.best_frame], store) if doc.best_frame is not None else {}
            first = downstream(decoded[0], gt[0], tmp, f"first{i}", store)
            r = int(rng.integers(0, len(frames)))
            rand = downstream(decoded[r], gt[r], tmp, f"rand{i}", store)

            row = {
                "condition": condition,
                "frames": len(frames),
                "track_err": float(np.mean(errs)) if errs else None,
                "track_err_p95": float(np.percentile(errs, 95)) if errs else None,
                "tracked_in_view": tracked_in_view,
                "track_ms": doc.tracking.get("ms_per_frame"),
                "reacquisitions": doc.tracking.get("reacquisitions"),
                "total_s": total_s,
                "det_err": float(np.mean(det_errs)) if det_errs else None,
                "det_found": det_found / max(int(in_view.sum()), 1),
                "det_ms": det_ms,
                "n_keys": len(keys),
                "key_blur_mean": float(blur[keys].mean()) if keys else None,
                "key_blur_max": float(blur[keys].max()) if keys else None,
                "key_blurry": float((blur[keys] > 3).mean()) if keys else None,
                "even_blur_mean": float(blur[even].mean()),
                "even_blur_max": float(blur[even].max()),
                "even_blurry": float((blur[even] > 3).mean()),
                "video_blur_max": float(blur.max()),
                "key_angle_span": angle_span([truth.camera_centers[k] for k in keys])
                if keys
                else 0.0,
                "video_angle_span": angle_span(truth.camera_centers),
                "best_blur": float(blur[doc.best_frame]) if doc.best_frame is not None else None,
                "first_blur": float(blur[0]),
                "rand_blur": float(blur[r]),
            }
            for name, res, frame_no in (
                ("best", best, doc.best_frame),
                ("first", first, 0),
                ("rand", rand, r),
            ):
                row[f"{name}_template_ok"] = res.get("template") == cheque.bank
                row[f"{name}_iou"] = res.get("iou")
                if engine is not None and res.get("template") == cheque.bank:
                    rdoc = res["doc"]
                    fields = read_cheque_fields(
                        rdoc.registered, store.get(rdoc.template_id), engine
                    )
                    by = {f.name: f for f in fields}
                    row[f"{name}_field_acc"] = float(
                        np.mean([field_correct(k, by[k], cheque) for k in FIELDS])
                    )
                elif engine is not None:
                    row[f"{name}_field_acc"] = 0.0
                _ = frame_no
            rows.append(row)
            print(
                f"clip {i + 1}/{n} ({condition}): err {row['track_err']:.3f}% keys {len(keys)} "
                f"best blur {row['best_blur']:.2f}px ({total_s:.1f}s)",
                file=sys.stderr,
            )

    def mean(key: str, subset: list[dict] | None = None) -> float | None:
        vals = [r[key] for r in (subset or rows) if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    exits = [r for r in rows if r["condition"] == "exit"]
    by_cond = {c: [r for r in rows if r["condition"] == c] for c in CONDITIONS}
    results = {
        "n": n,
        "seconds": seconds,
        "tracking": {
            "corner_err_pct_diag": mean("track_err"),
            "corner_err_p95_pct_diag": mean("track_err_p95"),
            "tracked_in_view": mean("tracked_in_view"),
            "tracked_in_view_exit_clips": mean("tracked_in_view", exits),
            "reacquisitions_exit_clips": mean("reacquisitions", exits),
            "ms_per_frame": mean("track_ms"),
            "pipeline_s_per_clip": mean("total_s"),
        },
        "by_condition": {
            c: {
                "clips": len(sub),
                "klt_err_pct_diag": mean("track_err", sub),
                "detector_err_pct_diag": mean("det_err", sub),
                "klt_in_view": mean("tracked_in_view", sub),
                "detector_in_view": mean("det_found", sub),
                "best_frame_template_accuracy": mean("best_template_ok", sub),
                "best_frame_field_iou": mean("best_iou", sub),
            }
            for c, sub in by_cond.items()
            if sub
        },
        "detector_baseline": {
            "corner_err_pct_diag": mean("det_err"),
            "found_in_view": mean("det_found"),
            "ms_per_frame": mean("det_ms"),
        },
        "keyframes": {
            "count": mean("n_keys"),
            "blur_mean_px": mean("key_blur_mean"),
            "blur_max_px": mean("key_blur_max"),
            "share_blurry_gt3px": mean("key_blurry"),
            "even_blur_mean_px": mean("even_blur_mean"),
            "even_blur_max_px": mean("even_blur_max"),
            "even_share_blurry_gt3px": mean("even_blurry"),
            "video_blur_max_px": mean("video_blur_max"),
            "angle_span_deg": mean("key_angle_span"),
            "video_angle_span_deg": mean("video_angle_span"),
        },
        "best_frame": {
            name: {
                "blur_px": mean(f"{name}_blur"),
                "template_accuracy": mean(f"{name}_template_ok"),
                "field_iou": mean(f"{name}_iou"),
                **({"field_accuracy": mean(f"{name}_field_acc")} if engine else {}),
            }
            for name in ("best", "first", "rand")
        },
        "clips": rows,
    }
    if engine is not None:
        results["ocr_device"] = getattr(engine, "device", "unknown")
    return results


def to_markdown(r: dict) -> str:
    t, d, k, b = r["tracking"], r["detector_baseline"], r["keyframes"], r["best_frame"]

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{x:.1%}"

    lines = [
        f"### Tracking (n = {r['n']} synthetic {r['seconds']:.0f} s clips, 1280x720 @ 30 fps)",
        "",
        "Corner error is the mean distance of the four estimated corners from ground truth, as a "
        "percentage of the frame diagonal (0.1% = 1.5 px).",
        "",
        "| Condition | Clips | KLT error | Detector error | KLT in-view frames | "
        "Detector in-view frames | Best-frame template id | Best-frame field IoU |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cond, c in r["by_condition"].items():
        iou = "n/a" if c["best_frame_field_iou"] is None else f"{c['best_frame_field_iou']:.3f}"
        lines.append(
            f"| {cond} | {c['clips']} | {c['klt_err_pct_diag']:.3f}% | "
            f"{c['detector_err_pct_diag']:.3f}% | {pct(c['klt_in_view'])} | "
            f"{pct(c['detector_in_view'])} | {pct(c['best_frame_template_accuracy'])} | {iou} |"
        )
    lines += [
        "",
        f"Overall: KLT {t['corner_err_pct_diag']:.3f}%, detector {d['corner_err_pct_diag']:.3f}%. "
        f"Time per frame: KLT {t['ms_per_frame']:.1f} ms (on 960 px frames), detector "
        f"{d['ms_per_frame']:.1f} ms (full 1280 px).",
        "",
        f"95th-percentile corner error: {t['corner_err_p95_pct_diag']:.3f}% of the diagonal. "
        f"Clips where the document leaves the view: {pct(t['tracked_in_view_exit_clips'])} of "
        f"in-view frames tracked, {t['reacquisitions_exit_clips']} re-localisations per clip. "
        f"Whole pipeline: {t['pipeline_s_per_clip']:.1f} s per clip.",
        "",
        "### Keyframes",
        "",
        "| Metric | Selected keyframes | Same number, evenly spaced |",
        "| --- | --- | --- |",
        f"| Mean motion blur | {k['blur_mean_px']:.2f} px | {k['even_blur_mean_px']:.2f} px |",
        f"| Worst motion blur | {k['blur_max_px']:.2f} px | {k['even_blur_max_px']:.2f} px |",
        f"| Frames blurred > 3 px | {pct(k['share_blurry_gt3px'])} "
        f"| {pct(k['even_share_blurry_gt3px'])} |",
        "",
        f"{k['count']:.1f} keyframes per clip on average; worst blur in the clips "
        f"{k['video_blur_max_px']:.1f} px. Viewing-angle range covered by the keyframes: "
        f"{k['angle_span_deg']:.1f} deg of the {k['video_angle_span_deg']:.1f} deg in the clip.",
        "",
        "### Which frame goes to the 2D pipeline",
        "",
        "| Frame | Blur | Template identified | Field-box IoU |"
        + (" Field accuracy (EasyOCR) |" if "field_accuracy" in b["best"] else ""),
        "| --- | --- | --- | --- |" + (" --- |" if "field_accuracy" in b["best"] else ""),
    ]
    for name, label in (
        ("best", "Selected best frame"),
        ("first", "First frame"),
        ("rand", "Random frame"),
    ):
        row = b[name]
        iou = "n/a" if row["field_iou"] is None else f"{row['field_iou']:.3f}"
        line = f"| {label} | {row['blur_px']:.2f} px | {pct(row['template_accuracy'])} | {iou} |"
        if "field_accuracy" in row:
            line += f" {pct(row['field_accuracy'])} |"
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--real-ocr", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = evaluate(args.n, args.seed, args.seconds, args.real_ocr)
    out = args.out or Path(
        "experiments/results/phase3_real_ocr.json"
        if args.real_ocr
        else "experiments/results/phase3.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(to_markdown(results))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
