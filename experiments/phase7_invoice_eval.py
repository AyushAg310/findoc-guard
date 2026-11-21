"""Phase 7 experiment: invoice extraction, invoice checks and document-type classification.

    python experiments/phase7_invoice_eval.py                          # rules, simulated OCR
    python experiments/phase7_invoice_eval.py --layout models/invoice_layoutlmv3   # + LayoutLMv3
    python experiments/phase7_invoice_eval.py --real-ocr --n 30        # EasyOCR on photos (GPU)

**Extraction.** ``--n`` synthetic invoices (the three layouts in turn). OCR is simulated from
the true words (:func:`findoc.synth.invoice.simulate_ocr`: phrases merged like EasyOCR, boxes
jittered) at character error rates 0%, 1% and 3%. Each extractor's normalised output is
compared with the truth: header fields (precision, recall, F1; a field counts only if its
normalised value is exactly right) and line-item cells (accuracy over every truth cell).
``--real-ocr`` instead photographs each invoice and runs the whole pipeline with EasyOCR.

**Checks.** Invoices with each planted error (wrong total, wrong line amount, wrong tax type,
invalid vendor GSTIN) and clean ones, extracted by the rules from simulated OCR: which checks
fire. OCR errors can make a clean invoice look wrong, so false alarms are measured at each
error rate.

**Classifier.** Accuracy and confusion matrix of the document-type model on fresh synthetic
pages (if ``--doctype`` points at a trained model).
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from findoc.fraud.invoice_rules import run_invoice_rules
from findoc.fraud.risk import risk_score
from findoc.recognize.invoice import (
    RuleExtractor,
    amount_in_words_text,
    field_counts,
    summarise,
)
from findoc.synth.invoice import (
    FRAUDS,
    LAYOUTS,
    invoice_photo,
    render_invoice,
    simulate_ocr,
    truth_normalized,
)

CERS = (0.0, 0.01, 0.03)
EXPECTED = {
    "total": "INVOICE_MATH",
    "lineitem": "INVOICE_MATH",
    "taxsplit": "TAX_TYPE_MISMATCH",
    "gstin": "GSTIN_INVALID",
}


def counts(pred: dict, truth: dict) -> np.ndarray:
    return np.array(field_counts(pred, truth), float)


def summary(c: np.ndarray) -> dict:
    return summarise(c)


def extraction(n: int, layout_dir: Path | None) -> dict:
    extractors = {"rules": RuleExtractor()}
    if layout_dir is not None:
        from findoc.recognize.layout import LayoutExtractor

        extractors["layoutlmv3"] = LayoutExtractor(layout_dir)
    out: dict = {}
    for cer in CERS:
        agg = {k: np.zeros(5) for k in extractors}
        per_field = {k: {} for k in extractors}
        times = {k: [] for k in extractors}
        for i in range(n):
            rng = np.random.default_rng(600_000 + i)
            page, truth, words = render_invoice(rng, LAYOUTS[i % len(LAYOUTS)])
            ocr = simulate_ocr(
                words, np.random.default_rng((600_000 + i, int(cer * 1000))), cer=cer
            )
            tn = truth_normalized(truth)
            for name, ex in extractors.items():
                t0 = time.perf_counter()
                inv = ex.extract(ocr) if name == "rules" else ex.extract(ocr, page)
                times[name].append(time.perf_counter() - t0)
                pred = inv.normalized()
                agg[name] += counts(pred, tn)
                for k, v in tn["fields"].items():
                    per_field[name].setdefault(k, []).append(pred["fields"].get(k) == v)
        out[f"cer_{cer}"] = {
            name: summary(agg[name])
            | {
                "per_field": {k: round(float(np.mean(v)), 3) for k, v in per_field[name].items()},
                "ms_per_invoice": round(1000 * float(np.median(times[name])), 1),
            }
            for name in extractors
        }
        print(
            f"CER {cer:.0%}: "
            + "; ".join(
                f"{k} F1 {v['f1']}, items {v['item_cell_accuracy']}"
                for k, v in out[f"cer_{cer}"].items()
            ),
            flush=True,
        )
    return out


def _check_one(args: tuple[float, str | None, int]) -> tuple[bool, bool, bool]:
    """(any check flagged, sent to review, the expected check fired) for one invoice."""
    cer, kind, i = args
    seed = 700_000 + 1000 * (FRAUDS.index(kind) + 1 if kind else 0) + i
    _, _, words = render_invoice(np.random.default_rng(seed), fraud=kind)
    ocr = simulate_ocr(words, np.random.default_rng((seed, 5)), cer=cer)
    inv = RuleExtractor().extract(ocr)
    text = amount_in_words_text(ocr)
    if text:
        inv.fields["amount_words"] = text
    results = [c for c in run_invoice_rules(inv.normalized()) if c.fired]
    fired = {c.id for c in results} - {"INVOICE_UNVERIFIED"}
    return (
        bool(fired),
        risk_score(results).band != "low",
        kind is not None and EXPECTED[kind] in fired,
    )


def checks(n_per_kind: int, workers: int = -1) -> dict:
    """Which checks fire, and whether the risk score sends the invoice to review (medium or
    high band), for clean invoices and each planted error, at each OCR error rate. Invoices
    are rendered and checked in parallel processes."""
    import os
    from concurrent.futures import ProcessPoolExecutor

    workers = workers if workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    jobs = [(cer, kind, i) for cer in CERS for kind in (None, *FRAUDS) for i in range(n_per_kind)]
    with ProcessPoolExecutor(workers) as pool:
        results = dict(zip(jobs, pool.map(_check_one, jobs, chunksize=4), strict=True))
    out: dict = {}
    for cer in CERS:
        res: dict = {}
        for kind in (None, *FRAUDS):
            rows = [results[(cer, kind, i)] for i in range(n_per_kind)]
            key = kind or "clean"
            res[key] = {
                "n": n_per_kind,
                "flagged": round(sum(r[0] for r in rows) / n_per_kind, 3),
                "sent_to_review": round(sum(r[1] for r in rows) / n_per_kind, 3),
            }
            if kind:
                res[key]["right_check"] = round(sum(r[2] for r in rows) / n_per_kind, 3)
        out[f"cer_{cer}"] = res
        print(
            f"CER {cer:.0%}: "
            + ", ".join(
                f"{k} flag {v['flagged']:.0%} review {v['sent_to_review']:.0%}"
                for k, v in res.items()
            ),
            flush=True,
        )
    return out


def classifier(path: Path, per_class: int) -> dict | None:
    if not path.is_file():
        return None
    from findoc.classify.doctype import CLASSES, DocTypeModel, synthetic_page

    m = DocTypeModel.load(path)
    confusion = {c: dict.fromkeys(CLASSES, 0) for c in CLASSES}
    for c in CLASSES:
        for i in range(per_class):
            confusion[c][m.predict(synthetic_page(c, 900_000 + i))[0]] += 1
    acc = sum(confusion[c][c] for c in CLASSES) / (len(CLASSES) * per_class)
    print(f"Classifier accuracy {acc:.1%}", flush=True)
    return {"accuracy": round(acc, 4), "per_class": per_class, "confusion": confusion}


def real_ocr(n: int) -> dict:
    """Photos through the whole pipeline with EasyOCR (needs the [ocr] extra; GPU advised).
    Saves per-invoice details (checks with confidence, risk band, GSTINs and total as read vs.
    true) so errors can be traced."""
    from findoc.io import save_image
    from findoc.pipeline import process
    from findoc.recognize import gstin as gst

    agg = np.zeros(5)
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for i in range(n):
            kind = None if i % 2 == 0 else FRAUDS[(i // 2) % len(FRAUDS)]
            photo, truth, _, _ = invoice_photo(800_000 + i, kind)
            path = Path(d) / f"inv{i}.jpg"
            save_image(path, photo)
            t0 = time.perf_counter()
            doc = process(path, ocr=True)
            fields = {f.name: f for f in doc.fields}
            pred = {
                "fields": {k: f.normalized for k, f in fields.items()},
                "items": [it.get("normalized", {}) for it in doc.line_items],
            }
            tn = truth_normalized(truth)
            agg += counts(pred, tn)
            fired = [c for c in doc.checks if c.fired and c.id != "INVOICE_UNVERIFIED"]
            row = {
                "invoice": i,
                "planted": kind,
                "seconds": round(time.perf_counter() - t0, 1),
                "method": next((f.method for f in doc.fields), None),
                "risk_band": doc.risk.get("band"),
                "risk_score": doc.risk_score,
                "checks": [
                    {"id": c.id, "confidence": c.confidence, "reason": c.reason} for c in fired
                ],
            }
            for g in ("vendor_gstin", "buyer_gstin"):
                read = fields[g].value if g in fields else None
                row[g] = {
                    "true": tn["fields"][g],
                    "read": read,
                    "normalized": pred["fields"].get(g),
                    "valid": bool(read and gst.validate(read)[0]),
                }
            row["total"] = {"true": tn["fields"]["total"], "read": pred["fields"].get("total")}
            rows.append(row)
            print(
                f"invoice {i} ({kind or 'clean'}): risk {row['risk_band']}, "
                f"{[(c['id'], c['confidence']) for c in row['checks']]}",
                flush=True,
            )

    def rate(sel, pred_fn) -> float:  # noqa: ANN001
        sub = [r for r in rows if sel(r)]
        return round(sum(pred_fn(r) for r in sub) / max(1, len(sub)), 3)

    review = lambda r: r["risk_band"] in ("medium", "high")  # noqa: E731
    gst_ok = [
        r[g]["normalized"] == r[g]["true"] for r in rows for g in ("vendor_gstin", "buyer_gstin")
    ]
    return summary(agg) | {
        "extractor": rows[0]["method"] if rows else None,
        "sent_to_review_clean": rate(lambda r: r["planted"] is None, review),
        "sent_to_review_planted": rate(lambda r: r["planted"] is not None, review),
        "right_check_planted": rate(
            lambda r: r["planted"] is not None,
            lambda r: EXPECTED[r["planted"]] in {c["id"] for c in r["checks"]},
        ),
        "gstin_read_exactly": round(float(np.mean(gst_ok)), 3),
        "median_seconds": float(np.median([r["seconds"] for r in rows])),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--checks-per-kind", type=int, default=40)
    ap.add_argument("--layout", type=Path, default=None, help="Fine-tuned LayoutLMv3 folder")
    ap.add_argument("--doctype", type=Path, default=Path("models/doctype.pkl"))
    ap.add_argument("--classifier-per-class", type=int, default=50)
    ap.add_argument("--real-ocr", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.real_ocr:
        result = {"real_ocr": real_ocr(a.n)}
        out = a.out or Path("experiments/results/phase7_real_ocr.json")
    else:
        result = {
            "n": a.n,
            "extraction": extraction(a.n, a.layout),
            "checks": checks(a.checks_per_kind),
            "classifier": classifier(a.doctype, a.classifier_per_class),
        }
        out = a.out or Path("experiments/results/phase7.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
