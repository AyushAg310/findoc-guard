"""Ablation: how should boxed date digits be presented to OCR?

For each synthetic cheque, the template-subtracted date crop is read four ways:

* ``spaced``   - the crop as written (digits far apart, one per box)
* ``compact``  - boxes cut apart, digits trimmed and re-joined with normal letter spacing
* ``per_cell`` - each digit recognised on its own, then concatenated
* ``validated`` - what the pipeline does: spaced, then compact, then per_cell, keeping the
  first reading that is a valid calendar date

and scored against the ground truth. Failures are saved as images with what OCR read, so you
can see *why* a variant fails.

Needs the [ocr] extra (real OCR). Run from the repository root:

    python experiments/date_reading_ablation.py --n 100
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from findoc.localize.stages import align_stage
from findoc.localize.templates import builtin_store
from findoc.pipeline import Pipeline
from findoc.preprocess.stages import detect_stage, enhance_stage, quality_stage, rectify_stage
from findoc.recognize.fields import _prepare, _read, crop_fields, per_cell_images, read_boxed
from findoc.recognize.ocr import get_engine
from findoc.recognize.parsers import parse_cheque_date
from findoc.synth.dataset import FRAUD_KINDS, cheque_photo

VARIANTS = ("spaced", "compact", "per_cell", "validated")


def read_variant(engine, spec, crop: np.ndarray, variant: str) -> str:  # type: ignore[no-untyped-def]
    if variant == "spaced":
        return _read(engine, spec, _prepare(crop))[0].strip()
    if variant == "compact":
        return _read(engine, spec, _prepare(crop, spec.cells))[0].strip()
    if variant == "per_cell":
        return "".join(
            _read(engine, spec, _prepare(g))[0].strip()[:1]
            for g in per_cell_images(crop, spec.cells)
        )
    text, _, used = read_boxed(engine, spec, crop)
    return f"{text}|{used}"


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = cv2.copyMakeBorder(img, 0, 30, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(img, text, (4, img.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
    return img


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--out", type=Path, default=Path("experiments/results/date_ablation"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    engine = get_engine()
    store = builtin_store()
    pipe = (
        Pipeline()
        .add("quality", quality_stage)
        .add("detect", detect_stage)
        .add("rectify", rectify_stage)
        .add("enhance", enhance_stage)
        .add("align", align_stage)
    )
    correct: Counter[str] = Counter()
    valid: Counter[str] = Counter()
    used: Counter[str] = Counter()
    rows, failures = [], {v: [] for v in VARIANTS}
    n_done = 0
    with tempfile.TemporaryDirectory() as d:
        for i in range(args.n):
            fraud = None if i % 2 == 0 else FRAUD_KINDS[(i // 2) % len(FRAUD_KINDS)]
            photo, truth, _ = cheque_photo(args.seed + i, fraud, today=date.today())
            path = Path(d) / f"c{i}.jpg"
            cv2.imwrite(str(path), photo)
            doc = pipe.run(path)
            if doc.registered is None:
                continue
            n_done += 1
            template = store.get(doc.template_id)
            spec = template.field("date")
            crop = crop_fields(doc.registered, template)["date"].image
            row = {"i": i, "truth": truth.fields["date"]}
            for v in VARIANTS:
                text = read_variant(engine, spec, crop, v)
                if v == "validated":
                    text, how = text.split("|")
                    used[how] += 1
                row[v] = text
                parsed = parse_cheque_date(text)
                valid[v] += parsed is not None
                ok = parsed is not None and parsed.isoformat() == truth.date
                correct[v] += ok
                if not ok and len(failures[v]) < 12:
                    img = _prepare(crop, spec.cells if v == "compact" else 1)
                    failures[v].append(label(img, f"truth {truth.fields['date']}  read '{text}'"))
            rows.append(row)

    for v, imgs in failures.items():
        if imgs:
            width = max(im.shape[1] for im in imgs)
            padded = [
                cv2.copyMakeBorder(
                    im, 4, 4, 0, width - im.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )  # fmt: skip
                for im in imgs
            ]
            cv2.imwrite(str(args.out / f"failures_{v}.png"), np.vstack(padded))

    summary = {
        "n": n_done,
        "device": getattr(engine, "device", "unknown"),
        "accuracy": {v: round(correct[v] / max(n_done, 1), 4) for v in VARIANTS},
        "valid_date_rate": {v: round(valid[v] / max(n_done, 1), 4) for v in VARIANTS},
        "validated_variant_used": dict(used),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out / "readings.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"### Date reading ablation (n = {n_done}, EasyOCR on {summary['device']})\n")
    print("| Variant | Correct date | Valid date |")
    print("| --- | --- | --- |")
    for v in VARIANTS:
        print(f"| {v} | {summary['accuracy'][v]:.1%} | {summary['valid_date_rate'][v]:.1%} |")
    print(f"\nValidated pipeline used: {dict(used)}")
    print(f"Saved {args.out}/summary.json, readings.json and failures_*.png")


if __name__ == "__main__":
    main()
