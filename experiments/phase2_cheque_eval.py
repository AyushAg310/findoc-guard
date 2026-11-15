"""Phase 2 evaluation on synthetic cheque photos with ground truth.

Part A - geometry (no OCR needed):
    template identification accuracy, field-box IoU in the photo, signature presence accuracy,
    alignment latency.
Part B - robustness to OCR errors (simulated OCR at increasing character error rates):
    field accuracy after parsing and rule detection rates for each fraud type.
Part C - real OCR (--real-ocr, needs the [ocr] extra; GPU recommended):
    the same metrics as Part B using EasyOCR on the template-subtracted field crops.

Run from the repository root:

    python experiments/phase2_cheque_eval.py --n 120
    python experiments/phase2_cheque_eval.py --n 100 --real-ocr      # on a machine with OCR

Results are printed as Markdown and saved to experiments/results/phase2*.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from findoc.fraud.rules import run_cheque_rules
from findoc.localize.stages import align_stage
from findoc.localize.templates import builtin_store
from findoc.pipeline import Pipeline
from findoc.preprocess.stages import detect_stage, enhance_stage, quality_stage, rectify_stage
from findoc.recognize.fields import read_cheque_fields
from findoc.synth.dataset import FRAUD_KINDS, cheque_photo
from findoc.synth.ocr_noise import OracleOcr

FIELDS = ["date", "payee", "amount_words", "amount_figures", "account_number", "ifsc", "micr"]
EXPECTED = {
    "mismatch": "AMOUNT_MISMATCH",
    "stale": "DATE_STALE",
    "postdated": "DATE_POSTDATED",
    "unsigned": "SIGNATURE_MISSING",
}


def geometry_pipeline() -> Pipeline:
    return (
        Pipeline()
        .add("quality", quality_stage)
        .add("detect", detect_stage)
        .add("rectify", rectify_stage)
        .add("enhance", enhance_stage)
        .add("align", align_stage)
    )


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    union = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - inter
    return float(inter / union) if union > 0 else 0.0


def build_set(n: int, seed: int, tmp: Path) -> list[dict]:
    """Photos with a balanced mix: half clean, half spread over the fraud kinds."""
    cases = []
    for i in range(n):
        fraud = None if i % 2 == 0 else FRAUD_KINDS[(i // 2) % len(FRAUD_KINDS)]
        photo, truth, page_to_photo = cheque_photo(seed + i, fraud, today=date.today())
        path = tmp / f"c{i:04d}.jpg"
        cv2.imwrite(str(path), photo, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cases.append(
            {"path": path, "truth": truth, "fraud": fraud, "g": page_to_photo, "seed": seed + i}
        )
    return cases


def field_correct(name: str, field, truth) -> bool:  # type: ignore[no-untyped-def]
    expected = {
        "date": truth.date,
        "payee": truth.payee.lower(),
        "amount_words": None,  # compared against what was *written*, see below
        "amount_figures": truth.amount,
        "account_number": truth.account_number,
        "ifsc": truth.ifsc,
        "micr": truth.micr,
    }[name]
    if name == "payee":
        return field.value.strip().lower() == expected
    if name == "amount_words":
        from findoc.recognize.parsers import words_to_amount

        return field.normalized == str(words_to_amount(truth.amount_words).value)
    return field.normalized == expected


def score_rules(cases: list[dict], fired_per_case: list[set[str]]) -> dict:
    out: dict[str, float] = {}
    for kind, check in EXPECTED.items():
        idx = [i for i, c in enumerate(cases) if c["fraud"] == kind]
        if idx:
            out[f"recall_{kind}"] = round(
                sum(check in fired_per_case[i] for i in idx) / len(idx), 4
            )
    clean = [i for i, c in enumerate(cases) if c["fraud"] is None]
    out["clean_with_any_flag"] = round(
        sum(bool(fired_per_case[i]) for i in clean) / max(len(clean), 1), 4
    )
    by_check: dict[str, int] = defaultdict(int)
    for i in clean:
        for check in fired_per_case[i]:
            by_check[check] += 1
    out["false_flags_on_clean"] = {k: round(v / max(len(clean), 1), 4) for k, v in by_check.items()}
    return out


def evaluate(n: int, seed: int, real_ocr: bool, cers: list[float]) -> dict:
    import tempfile

    store = builtin_store()
    results: dict = {"n": n}
    with tempfile.TemporaryDirectory() as d:
        cases = build_set(n, seed, Path(d))
        pipe = geometry_pipeline()

        # ---- Part A: geometry
        correct_id, ious, sig_ok, t_align = 0, defaultdict(list), 0, []
        docs = []
        for c in cases:
            doc = pipe.run(c["path"])
            docs.append(doc)
            t_align += [s.ms for s in doc.stages if s.stage == "align"]
            if doc.template_id != c["truth"].bank:
                continue
            correct_id += 1
            template = store.get(doc.template_id)
            tmpl_to_photo = np.linalg.inv(
                np.array(doc.template_homography) @ np.array(doc.homography)
            )
            for spec in template.fields:
                x0, y0, x1, y1 = spec.box
                box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)
                est = cv2.perspectiveTransform(box.reshape(-1, 1, 2), tmpl_to_photo)
                gt = cv2.perspectiveTransform(box.reshape(-1, 1, 2), c["g"])
                ious[spec.name].append(polygon_iou(est.reshape(-1, 2), gt.reshape(-1, 2)))
            sig = read_cheque_fields(doc.registered, template, engine=None)[0]
            sig_ok += (sig.value == "present") == c["truth"].signed

        results["geometry"] = {
            "template_accuracy": round(correct_id / n, 4),
            "field_iou_mean": {k: round(float(np.mean(v)), 4) for k, v in ious.items()},
            "field_iou_overall_mean": round(
                float(np.mean([x for v in ious.values() for x in v])), 4
            ),
            "field_iou_min": round(float(np.min([x for v in ious.values() for x in v])), 4),
            "signature_presence_accuracy": round(sig_ok / max(correct_id, 1), 4),
            "median_align_ms": round(float(np.median(t_align)), 1),
        }

        # ---- Part B / C: field accuracy and rules
        def run_reading(make_engine) -> dict:  # type: ignore[no-untyped-def]
            per_field = defaultdict(list)
            fired = []
            t_fields = []
            for c, doc in zip(cases, docs, strict=True):
                if doc.registered is None:
                    fired.append(set())
                    continue
                template = store.get(doc.template_id)
                engine = make_engine(c)
                t0 = time.perf_counter()
                fields = read_cheque_fields(doc.registered, template, engine)
                t_fields.append((time.perf_counter() - t0) * 1000)
                by_name = {f.name: f for f in fields}
                for name in FIELDS:
                    per_field[name].append(field_correct(name, by_name[name], c["truth"]))
                checks = run_cheque_rules(fields, today=date.today())
                fired.append({x.id for x in checks if x.fired})
            return {
                "field_accuracy": {k: round(float(np.mean(v)), 4) for k, v in per_field.items()},
                "rules": score_rules(cases, fired),
                "median_fields_ms": round(float(np.median(t_fields)), 1),
            }

        results["simulated_ocr"] = {
            str(cer): run_reading(
                lambda c, cer=cer: OracleOcr(c["truth"].fields, cer, seed=c["seed"])
            )
            for cer in cers
        }
        if real_ocr:
            from findoc.recognize.ocr import get_engine

            engine = get_engine()
            results["real_ocr"] = run_reading(lambda c: engine)
            results["real_ocr"]["device"] = getattr(engine, "device", "unknown")
    return results


def to_markdown(r: dict) -> str:
    g = r["geometry"]
    lines = [
        f"### Geometry (n = {r['n']} synthetic cheque photos, 4 fictional banks)",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Template identification accuracy | {g['template_accuracy']:.1%} |",
        f"| Mean field-box IoU (all fields) | {g['field_iou_overall_mean']:.3f} |",
        f"| Worst single field-box IoU | {g['field_iou_min']:.3f} |",
        f"| Signature presence accuracy | {g['signature_presence_accuracy']:.1%} |",
        f"| Median alignment time (SIFT + RANSAC, 4 templates) | {g['median_align_ms']} ms |",
        "",
        "Per-field IoU: " + ", ".join(f"{k} {v:.3f}" for k, v in g["field_iou_mean"].items()),
        "",
    ]

    def block(title: str, rows: dict[str, dict]) -> list[str]:
        cols = list(rows)
        out = [title, "", "| Metric | " + " | ".join(cols) + " |", "| --- |" + " --- |" * len(cols)]
        for name in FIELDS:
            vals = [rows[c]["field_accuracy"].get(name, float("nan")) for c in cols]
            out.append(f"| {name} accuracy | " + " | ".join(f"{v:.1%}" for v in vals) + " |")
        for key in rows[cols[0]]["rules"]:
            if key == "false_flags_on_clean":
                continue
            vals = [rows[c]["rules"][key] for c in cols]
            out.append(f"| {key} | " + " | ".join(f"{v:.1%}" for v in vals) + " |")
        checks = sorted({k for c in cols for k in rows[c]["rules"]["false_flags_on_clean"]})
        for check in checks:
            vals = [rows[c]["rules"]["false_flags_on_clean"].get(check, 0.0) for c in cols]
            out.append(
                f"| clean flagged by {check} | " + " | ".join(f"{v:.1%}" for v in vals) + " |"
            )
        return out + [""]

    sim = {f"CER {float(k):.0%}": v for k, v in r["simulated_ocr"].items()}
    lines += block("### Field accuracy and rule detection vs. simulated OCR error rate", sim)
    if "real_ocr" in r:
        lines += block(
            f"### Real OCR (EasyOCR on {r['real_ocr']['device']})", {"EasyOCR": r["real_ocr"]}
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=50_000)
    parser.add_argument("--real-ocr", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = evaluate(args.n, args.seed, args.real_ocr, cers=[0.0, 0.02, 0.05, 0.10])
    out = args.out or Path(
        "experiments/results/phase2_real_ocr.json"
        if args.real_ocr
        else "experiments/results/phase2.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(to_markdown(results))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
