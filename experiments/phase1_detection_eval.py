"""Phase 1 evaluation on synthetic photos with known ground truth.

Measures:
1. Page detection: success rate and corner error (as % of the image diagonal).
2. Quality gate: how often photos are accepted at increasing camera-shake blur.
3. Illumination: reduction in background unevenness after normalisation.
4. Latency: per-stage time (median, ms).

Run from the repository root:

    python experiments/phase1_detection_eval.py --n 200

Results are printed as Markdown tables and saved to experiments/results/phase1.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from findoc.preprocess.detect import find_document
from findoc.preprocess.illumination import background_unevenness, normalize_illumination
from findoc.preprocess.quality import assess
from findoc.preprocess.rectify import warp_document
from findoc.synth.photo import photograph, render_page

SUCCESS_TOL = 0.02  # corners within 2% of the image diagonal count as a correct detection


def evaluate(n: int, seed: int) -> dict:
    errors, methods = [], {}
    t_quality, t_detect, t_rectify, t_enhance = [], [], [], []
    uneven_before, uneven_after = [], []

    for i in range(n):
        rng = np.random.default_rng(seed + i)
        photo, gt = photograph(render_page(rng), rng)
        diag = float(np.hypot(*photo.shape[:2]))

        t0 = time.perf_counter()
        assess(photo)
        t_quality.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        det = find_document(photo)
        t_detect.append(time.perf_counter() - t0)
        errors.append(float(np.linalg.norm(det.corners - gt, axis=1).mean()) / diag)
        methods[det.method] = methods.get(det.method, 0) + 1

        t0 = time.perf_counter()
        flat, _ = warp_document(photo, det.corners)
        t_rectify.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        enhanced = normalize_illumination(flat)
        t_enhance.append(time.perf_counter() - t0)
        uneven_before.append(background_unevenness(flat))
        uneven_after.append(background_unevenness(enhanced))

    err = np.array(errors) * 100
    blur_levels = [0.0, 1.0, 1.5, 2.0, 3.0, 5.0]
    accept = {}
    for sigma in blur_levels:
        ok = 0
        for i in range(min(n, 50)):
            rng = np.random.default_rng(10_000 + seed + i)
            photo, _ = photograph(render_page(rng), rng, blur_sigma=sigma)
            ok += assess(photo).acceptable
        accept[str(sigma)] = ok / min(n, 50)

    ms = lambda xs: round(float(np.median(xs)) * 1000, 1)  # noqa: E731
    return {
        "n": n,
        "detection": {
            "success_rate": round(float(np.mean(err < SUCCESS_TOL * 100)), 4),
            "mean_corner_error_pct_diag": round(float(err.mean()), 3),
            "median_corner_error_pct_diag": round(float(np.median(err)), 3),
            "p95_corner_error_pct_diag": round(float(np.percentile(err, 95)), 3),
            "methods": methods,
        },
        "quality_gate_acceptance_by_blur_sigma": accept,
        "illumination": {
            "background_std_before": round(float(np.mean(uneven_before)), 2),
            "background_std_after": round(float(np.mean(uneven_after)), 2),
            "reduction_pct": round(
                100 * (1 - float(np.mean(uneven_after)) / float(np.mean(uneven_before))), 1
            ),
        },
        "median_latency_ms": {
            "quality": ms(t_quality),
            "detect": ms(t_detect),
            "rectify": ms(t_rectify),
            "enhance": ms(t_enhance),
        },
    }


def to_markdown(r: dict) -> str:
    d, il, lat = r["detection"], r["illumination"], r["median_latency_ms"]
    lines = [
        f"### Page detection (n = {r['n']} synthetic photos)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Success rate (error < {SUCCESS_TOL:.0%} of diagonal) | {d['success_rate']:.1%} |",
        f"| Mean corner error | {d['mean_corner_error_pct_diag']}% of diagonal |",
        f"| Median corner error | {d['median_corner_error_pct_diag']}% of diagonal |",
        f"| 95th percentile error | {d['p95_corner_error_pct_diag']}% of diagonal |",
        f"| Method used | {d['methods']} |",
        "",
        "### Quality gate: acceptance rate vs. simulated camera shake",
        "",
        "| Blur sigma (px) | " + " | ".join(r["quality_gate_acceptance_by_blur_sigma"]) + " |",
        "| --- |" + " --- |" * len(r["quality_gate_acceptance_by_blur_sigma"]),
        "| Accepted | "
        + " | ".join(f"{v:.0%}" for v in r["quality_gate_acceptance_by_blur_sigma"].values())
        + " |",
        "",
        "### Illumination normalisation",
        "",
        f"Background brightness std-dev: {il['background_std_before']} -> "
        f"{il['background_std_after']} ({il['reduction_pct']}% more even).",
        "",
        "### Median latency per stage (ms)",
        "",
        "| Quality | Detect | Rectify | Enhance |",
        "| --- | --- | --- | --- |",
        f"| {lat['quality']} | {lat['detect']} | {lat['rectify']} | {lat['enhance']} |",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/phase1.json"))
    args = parser.parse_args()

    results = evaluate(args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(to_markdown(results))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
