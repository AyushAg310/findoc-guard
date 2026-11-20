"""Phase 6 experiment: tamper detection on the original capture vs. on the dewarped page.

    python experiments/phase6_tamper_eval.py --model models/tamper.pkl \
        --flat-model models/tamper_flat.pkl --photos 40 --videos 6

The SRS (10.3) hypothesis: dewarping resamples every pixel and so weakens the forensic traces
a tamper detector relies on; running the detector on the **original** capture and mapping its
heatmap to the flat page through the correspondence map should preserve detection.

Part A, **mapping** (photos and 3D-flattened videos). Every capture is forged (or kept clean),
the pipeline produces its flat page, and each detector runs two ways:

* *on flat*: on the dewarped page itself;
* *original + map*: on the original capture, heatmap carried to the flat page (homography for
  photos, the Phase 5 correspondence map for videos).

Detectors: the untrained JPEG-ghost baseline, the model trained on original captures, and,
for a fair "on flat" comparison, a model trained on dewarped pages
(``findoc.fraud.tamper_train.train(view="flat")``). Metrics on the flat page: pixel AUC (edited
vs. untouched page pixels) and cheque-level AUC (score of the most suspicious handwritten
field, forged vs. clean cheques).

Part B, **end to end** (photos through ``findoc process``): how many forged cheques get
``TAMPER_REGION``, which field is named, how many clean cheques are flagged, and the risk band.

Forgeries are digital edits of the captured image (:mod:`findoc.synth.tamper`). For videos the
reference (best) frame, kept as a JPEG still, is edited: the frame the flat page is sampled
from. Forging every frame of a video and re-encoding it is not modelled here.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from findoc.config import get_settings
from findoc.fraud.tamper import (
    FIELD_WEIGHTS,
    TamperModel,
    classical_heatmap,
    page_mask,
    to_flat,
    upsample,
)
from findoc.io import save_image
from findoc.pipeline import process
from findoc.synth.camera import Pose, project
from findoc.synth.cheque import BANKS, field_specs
from findoc.synth.surface import Bend, DocSurface, random_bend
from findoc.synth.tamper import FIELDS, KINDS, box_quad, forged_sample, tamper
from findoc.synth.video import cheque_video, to_world, write_video

HANDWRITTEN = FIELDS


def detectors(model: TamperModel, flat_model: TamperModel | None) -> dict:
    d = {
        "ghost": lambda img, page: classical_heatmap(img, page),
        "model": lambda img, page: model.block_probability(img, page),
    }
    if flat_model is not None:
        d["model_flat"] = lambda img, page: flat_model.block_probability(img, page)
    return d


def quad_mask(shape: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, np.round(quad).astype(np.int32), 1)
    return m


def evaluate_capture(doc, original, mask, field_masks, dets) -> dict:  # type: ignore[no-untyped-def]
    """Heatmaps on flat vs. original+map for one capture whose flat page geometry is in ``doc``.
    ``mask`` and ``field_masks`` are in original-image pixels."""
    fh, fw = doc.flat.shape[:2]
    hom, fmap = doc.homography, doc.flat_map
    flat = (
        cv2.remap(original, *[m.astype(np.float32) for m in fmap], cv2.INTER_LINEAR)
        if fmap is not None
        else cv2.warpPerspective(original, np.array(hom), (fw, fh), flags=cv2.INTER_CUBIC)
    )
    nearest = lambda a: to_flat(a.astype(np.float32), (fh, fw), hom, fmap) > 0.5  # noqa: E731
    gt = nearest(mask)
    fields_flat = {k: nearest(v) for k, v in field_masks.items()}
    valid = nearest(np.ones(original.shape[:2], np.float32))
    page_o = page_mask(original.shape[:2], np.array(doc.corners, np.float32))
    out = {}
    for name, det in dets.items():
        views = {}
        if name != "model_flat":
            views["original+map"] = to_flat(
                upsample(det(original, page_o), original.shape[:2]), (fh, fw), hom, fmap
            )
        if name != "model":
            views["flat"] = upsample(det(flat, valid), (fh, fw))
        if name == "model":  # the original-trained model applied to the flat page, for reference
            views["flat"] = upsample(det(flat, valid), (fh, fw))
        for view, heat in views.items():
            r = {}
            if gt[valid].any():
                sub = valid[::3, ::3]
                r["pixel_auc"] = float(roc_auc_score(gt[::3, ::3][sub], heat[::3, ::3][sub]))
            scores = {
                k: float(np.percentile(heat[m], 90)) * FIELD_WEIGHTS[k]
                for k, m in fields_flat.items()
                if m.any()
            }
            r["score"] = max(scores.values()) if scores else 0.0
            r["top"] = max(scores, key=scores.get) if scores else None
            out[f"{name}|{view}"] = r
    return out


def photo_part(n: int, dets: dict, tmp: Path, seed0: int = 900_000) -> list[dict]:
    rows = []
    for i in range(n):
        for forged in (False, True):
            s = forged_sample(seed0 + i, forged)
            path = tmp / f"p{i}_{int(forged)}.png"
            save_image(path, s["image"])
            t0 = time.perf_counter()
            doc = process(path, ocr=False)
            row = {
                "capture": "photo",
                "seed": seed0 + i,
                "forged": forged,
                "kind": s["kind"],
                "field": s["field"],
                "process_s": round(time.perf_counter() - t0, 2),
            }
            tc = [c for c in doc.checks if c.id == "TAMPER_REGION"]
            row["tamper_fired"] = bool(tc and tc[0].fired)
            row["tamper_top"] = tc[0].evidence.get("top") if tc else None
            row["risk_band"] = doc.risk.get("band")
            row["risk_score"] = doc.risk_score
            if doc.flat is not None and doc.homography is not None:
                fm = {
                    f.name: quad_mask(s["image"].shape[:2], box_quad(f.box, s["page_to_image"]))
                    for f in field_specs(BANKS[s["bank"]])
                    if f.name in HANDWRITTEN
                }
                row["maps"] = evaluate_capture(doc, s["image"], s["mask"], fm, dets)
            rows.append(row)
            print(
                f"photo {i} forged={forged} {s['kind'] or ''} {s['field'] or ''}: "
                f"flag={row['tamper_fired']} top={row['tamper_top']} risk={row['risk_band']}",
                flush=True,
            )
    return rows


def video_part(n: int, dets: dict, tmp: Path, seed0: int = 950_000) -> list[dict]:
    rows = []
    for i in range(n):
        seed = seed0 + i
        kind_b = ("fold", "curl2", "curl")[i % 3]
        bend = random_bend(kind_b, np.random.default_rng(seed))
        frames, truth, cheque = cheque_video(
            seed, None, None, seconds=4, fps=15, subframes=2, jerks=1, bend=bend
        )
        path = write_video(tmp / f"v{i}.mp4", frames, truth.fps)
        os.environ["FINDOC_FOCAL_PX"] = str(np.array(truth.meta["K"])[0, 0])  # calibrated phone
        get_settings.cache_clear()
        doc = process(path, ocr=False)
        os.environ.pop("FINDOC_FOCAL_PX", None)
        get_settings.cache_clear()
        if doc.flat is None or doc.best_frame is None:
            continue
        p = truth.meta["poses"][doc.best_frame]
        pose = Pose(np.array(p["R"]), np.array(p["t"]))
        K = np.array(truth.meta["K"])  # noqa: N806
        tw, th = 1624, 736
        surf = DocSurface(tw, th, Bend(**truth.meta["bend"]))
        specs = {f.name: f for f in field_specs(BANKS[cheque.bank])}

        def quad_of(box: tuple[int, int, int, int]) -> np.ndarray:
            x0, y0, x1, y1 = box
            st = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
            return project(K, pose, to_world(surf, st))  # noqa: B023

        # The pipeline samples the flat page from the best frame. A capture app keeps that frame
        # as a still photo (JPEG, random quality 80-95) for evidence; that still is what a forger
        # would edit, and what the detectors look at. (Decoded MPEG-4 frames carry no JPEG
        # history, and the model was trained on JPEG photos: see the guide's limitations.)
        from findoc.synth.tamper import jpeg

        original = jpeg(doc.original, int(np.random.default_rng(seed + 3).integers(80, 96)))
        fm = {k: quad_mask(original.shape[:2], quad_of(specs[k].box)) for k in HANDWRITTEN}
        rng = np.random.default_rng(seed)
        for forged in (False, True):
            if forged:
                kind = KINDS[i % 3]
                field = HANDWRITTEN[i % len(HANDWRITTEN)]
                donor = (170, 276, 700, 342) if field == "payee" else (300, 200, 500, 268)
                # splice source: the same field from a photo of another cheque of this bank
                from findoc.synth.tamper import capture

                src, _, h2 = capture(seed + 7, bank=cheque.bank)
                img, mask = tamper(
                    original,
                    quad_of(specs[field].box),
                    kind,
                    rng,
                    donor_quad=quad_of(donor),
                    source=src,
                    source_quad=box_quad(specs[field].box, h2),
                )
            else:
                kind = field = None
                img, mask = original, np.zeros(original.shape[:2], bool)
            row = {
                "capture": "video",
                "seed": seed,
                "bend": kind_b,
                "forged": forged,
                "kind": kind,
                "field": field,
                "flattening": doc.flattening,
            }
            row["maps"] = evaluate_capture(doc, img, mask, fm, dets)
            rows.append(row)
            print(
                f"video {i} ({kind_b}, {doc.flattening}) forged={forged} {kind or ''}", flush=True
            )
    return rows


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for capture in ("photo", "video"):
        sub = [r for r in rows if r["capture"] == capture and "maps" in r]
        if not sub:
            continue
        keys = sorted({k for r in sub for k in r["maps"]})
        table = {}
        for k in keys:
            pix = [
                r["maps"][k]["pixel_auc"]
                for r in sub
                if r["forged"] and "pixel_auc" in r["maps"].get(k, {})
            ]
            clean = [r["maps"][k]["score"] for r in sub if not r["forged"] and k in r["maps"]]
            forged = [r["maps"][k]["score"] for r in sub if r["forged"] and k in r["maps"]]
            top = [r["maps"][k]["top"] == r["field"] for r in sub if r["forged"] and k in r["maps"]]
            table[k] = {
                "pixel_auc": round(float(np.mean(pix)), 4) if pix else None,
                "cheque_auc": round(
                    float(roc_auc_score([0] * len(clean) + [1] * len(forged), clean + forged)), 4
                )
                if clean and forged
                else None,
                "top_field_correct": round(float(np.mean(top)), 4) if top else None,
                "per_kind_pixel_auc": {
                    kind: round(
                        float(
                            np.mean(
                                [
                                    r["maps"][k]["pixel_auc"]
                                    for r in sub
                                    if r["kind"] == kind and "pixel_auc" in r["maps"].get(k, {})
                                ]
                            )
                        ),
                        4,
                    )
                    for kind in KINDS
                    if any(r["kind"] == kind and "pixel_auc" in r["maps"].get(k, {}) for r in sub)
                },
            }
        out[f"mapping_{capture}"] = table
    e2e = [r for r in rows if r["capture"] == "photo"]
    if e2e:
        forged = [r for r in e2e if r["forged"]]
        clean = [r for r in e2e if not r["forged"]]
        out["end_to_end_photo"] = {
            "forged": len(forged),
            "clean": len(clean),
            "detection_rate": round(float(np.mean([r["tamper_fired"] for r in forged])), 4),
            "false_alarm_rate": round(float(np.mean([r["tamper_fired"] for r in clean])), 4),
            "field_named_correctly": round(
                float(
                    np.mean(
                        [r["tamper_top"] == r["field"] for r in forged if r["tamper_fired"]] or [0]
                    )
                ),
                4,
            ),
            "detection_by_kind": {
                k: round(float(np.mean([r["tamper_fired"] for r in forged if r["kind"] == k])), 4)
                for k in KINDS
                if any(r["kind"] == k for r in forged)
            },
            "risk_bands_forged": {
                b: sum(r["risk_band"] == b for r in forged) for b in ("low", "medium", "high")
            },
            "risk_bands_clean": {
                b: sum(r["risk_band"] == b for r in clean) for b in ("low", "medium", "high")
            },
            "median_process_s": round(float(np.median([r["process_s"] for r in e2e])), 2),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/tamper.pkl"))
    ap.add_argument("--flat-model", type=Path, default=Path("models/tamper_flat.pkl"))
    ap.add_argument("--photos", type=int, default=40)
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("experiments/results/phase6.json"))
    a = ap.parse_args()
    model = TamperModel.load(a.model)
    flat_model = TamperModel.load(a.flat_model) if a.flat_model.is_file() else None
    os.environ["FINDOC_TAMPER_MODEL"] = str(a.model)
    get_settings.cache_clear()
    dets = detectors(model, flat_model)
    with tempfile.TemporaryDirectory() as d:
        rows = photo_part(a.photos, dets, Path(d)) + video_part(a.videos, dets, Path(d))
    result = {
        "photos": a.photos,
        "videos": a.videos,
        "model": model.meta,
        "flat_model": flat_model.meta if flat_model else None,
        "summary": summarise(rows),
        "rows": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
