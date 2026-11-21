"""Fraud pipeline stages: validation rules (Phase 2) and the risk score (Phase 6)."""

from __future__ import annotations

import cv2
import numpy as np

from findoc.config import get_settings
from findoc.fraud.duplicates import DuplicateIndex, Match, ink_image
from findoc.fraud.risk import risk_score
from findoc.fraud.rules import run_cheque_rules
from findoc.fraud.signature import SIGNATURE_THRESHOLD, SignatureStore, signature_image
from findoc.fraud.tamper import (
    FIELD_WEIGHTS,
    classical_heatmap,
    field_scores,
    load_model,
    page_mask,
    to_flat,
    upsample,
)
from findoc.types import CheckResult, DocType, Document, InputKind


def rules_stage(doc: Document) -> Document:
    if doc.doc_type is not DocType.CHEQUE or not doc.fields:
        return doc
    text_read = bool(doc.quality.get("fields_read_with_ocr"))
    doc.checks.extend(run_cheque_rules(doc.fields, text_read=text_read))
    fired = [c.id for c in doc.checks if c.fired]
    doc.messages.append(f"Rule checks: {len(doc.checks)} run, {len(fired)} flagged {fired or ''}")
    return doc


def risk_stage(doc: Document) -> Document:
    """Combine every check that ran into one explainable score (last fraud stage)."""
    if not doc.checks:
        return doc
    s = get_settings()
    risk = risk_score(doc.checks, s.risk_medium, s.risk_high, s.risk_points)
    doc.risk_score = risk.score
    doc.risk = risk.as_dict()
    top = ", ".join(f"{c['check']} +{c['contribution']:g}" for c in risk.contributions[:3])
    forced = f"; forced high by {risk.forced_by}" if risk.forced_by else ""
    doc.messages.append(f"Risk {risk.score}/100 ({risk.band}){': ' + top if top else ''}{forced}")
    return doc


def tamper_stage(doc: Document) -> Document:
    """Tamper heatmap on the original capture, carried to the flat page and the cheque fields.

    Sets ``doc.tamper_map`` (flat-page resolution, [0, 1]) and adds a ``TAMPER_REGION`` check.
    With a trained model (``findoc tamper train``) the map is the per-block probability of an
    edit; without one it is the classical JPEG-ghost anomaly, saved for review but not flagged.
    """
    s = get_settings()
    if s.tamper == "off":
        return doc
    if doc.original is None or doc.corners is None or doc.flat is None:
        return doc
    if doc.flat_map is None and doc.homography is None:
        return doc
    page = page_mask(doc.original.shape[:2], np.array(doc.corners, np.float32))
    model = load_model(s.tamper_model)
    grid = (
        model.block_probability(doc.original, page)
        if model is not None
        else classical_heatmap(doc.original, page)
    )
    heat = upsample(grid, doc.original.shape[:2])
    flat_heat = to_flat(heat, doc.flat.shape[:2], doc.homography, doc.flat_map)
    doc.tamper_map = flat_heat
    if doc.doc_type is DocType.CHEQUE and doc.template_homography is not None:
        from findoc.localize.templates import default_store

        template = default_store(s.templates_dir).get(doc.template_id)
        tw, th = template.size
        hmat = np.array(doc.template_homography, np.float64)
        heat_t = cv2.warpPerspective(flat_heat, hmat, (tw, th))
        scores = field_scores(heat_t, {f.name: f.box for f in template.fields})
        weighted = {k: v * FIELD_WEIGHTS.get(k, 0.5) for k, v in scores.items()}
    else:  # any other document: the page as one region (its hottest 1%)
        top_heat = float(np.percentile(flat_heat, 99))
        scores = weighted = {"page": top_heat}
    top = max(weighted, key=weighted.get)
    evidence = {"field_scores": {k: round(v, 3) for k, v in scores.items()}, "top": top}
    if model is None:
        doc.checks.append(
            CheckResult(
                id="TAMPER_REGION",
                fired=False,
                reason="Tamper model not trained (run `findoc tamper train`); classical JPEG "
                f"heatmap saved for review (hottest: '{top}', {scores[top]:.2f}).",
                evidence=evidence | {"model": None},
            )
        )
        return doc
    threshold = model.threshold
    if doc.kind is InputKind.VIDEO:
        # The model reads JPEG re-compression traces. A video frame went through H.264, which
        # leaves different traces across the whole frame, so the model calls every page edited
        # (a genuine cheque filmed on a phone scored 1.00). Keep the map for a reviewer to look
        # at; don't score it. Tamper evidence for videos needs a model trained on video frames.
        doc.checks.append(
            CheckResult(
                id="TAMPER_REGION",
                fired=False,
                reason="Not checked: the tamper model is trained on JPEG photos, and video frames "
                "are H.264-compressed. The heatmap is saved for review only; for a forensic "
                f"check, upload a photo (hottest region in the video: '{top}', {scores[top]:.2f}).",
                evidence=evidence | {"threshold": threshold, "skipped": "video"},
            )
        )
        return doc
    fired = weighted[top] >= threshold
    doc.checks.append(
        CheckResult(
            id="TAMPER_REGION",
            fired=fired,
            severity=round(weighted[top], 3) if fired else 0.0,
            # how far past (or short of) the threshold, as a confidence in [0.5, 1]
            confidence=round(min(1.0, 0.5 + abs(weighted[top] - threshold) * 5), 3),
            reason=(
                f"'{top}' looks edited (score {scores[top]:.2f}, threshold {threshold:.2f}): "
                "its compression history differs from the rest of the capture."
                if fired
                else f"No field looks edited (highest: '{top}', {scores[top]:.2f} < "
                f"{threshold:.2f})."
            ),
            evidence=evidence | {"threshold": threshold},
        )
    )
    return doc


def duplicate_stage(doc: Document, register: bool = False) -> Document:
    """Compare the cheque's handwriting with every indexed cheque of the same bank (DUPLICATE);
    with ``register`` (``findoc process --register``), add it to the index afterwards."""
    if doc.registered is None or doc.template_id is None:
        return doc
    from findoc.localize.templates import default_store

    s = get_settings()
    template = default_store(s.templates_dir).get(doc.template_id)
    ink = ink_image(doc.registered, template.reference)
    micr = next((f.normalized for f in doc.fields if f.name == "micr" and f.normalized), None)
    index = DuplicateIndex(s.duplicate_index)
    best = None
    if len(index):
        matches = index.search(doc.template_id, ink, micr)
        best = matches[0] if matches else None
        doc.checks.append(_duplicate_check(best, len(matches)))
    if register:
        if best is not None and best.duplicate:
            doc.messages.append(f"Not added to the duplicate index: matches entry {best.entry.id}")
        else:
            e = index.add(doc.source.name, doc.template_id, ink, micr)
            doc.messages.append(f"Added to the duplicate index as entry {e.id} ({len(index)})")
    return doc


def _duplicate_check(best: Match | None, compared: int) -> CheckResult:
    if best is not None and best.duplicate:
        how = (
            "identical MICR line"
            if best.same_micr
            else f"{best.inliers} handwriting features match ({best.inlier_ratio:.0%})"
        )
        return CheckResult(
            id="DUPLICATE",
            fired=True,
            severity=1.0,
            confidence=1.0 if best.same_micr else round(min(1.0, best.inlier_ratio / 0.4), 3),
            reason=f"Same cheque as '{best.entry.source}' (index entry {best.entry.id}): {how}.",
            evidence={
                "entry": best.entry.id,
                "source": best.entry.source,
                "inliers": best.inliers,
                "inlier_ratio": round(best.inlier_ratio, 3),
                "same_micr": best.same_micr,
            },
        )
    closest = f" (closest: {best.inliers} matching features)" if best else ""
    return CheckResult(
        id="DUPLICATE",
        fired=False,
        confidence=1.0,
        reason=f"Not seen before: compared with {compared} indexed cheques of this bank{closest}.",
        evidence={"compared": compared, "closest_inliers": best.inliers if best else 0},
    )


def signature_stage(doc: Document, account: str | None = None) -> Document:
    """Compare the signature with the account's reference signatures (SIGNATURE_MISMATCH).

    The account comes from OCR (``account_number``) or is given (``findoc process --account``).
    Nothing is reported when the account is unknown or has no references."""
    if doc.registered is None or doc.template_id is None:
        return doc
    if account is None:
        account = next(
            (f.normalized for f in doc.fields if f.name == "account_number" and f.normalized),
            None,
        )
    if account is None:
        return doc
    from findoc.localize.templates import default_store

    s = get_settings()
    sig = signature_image(doc.registered, default_store(s.templates_dir).get(doc.template_id))
    if sig is None:
        return doc  # no signature at all: SIGNATURE_MISSING (Phase 2) already reports it
    result = SignatureStore(s.signatures_dir).compare(account, sig)
    if result is None:
        doc.messages.append(f"No reference signatures for account {account}; not compared.")
        return doc
    dist, n = result
    fired = dist > SIGNATURE_THRESHOLD
    doc.checks.append(
        CheckResult(
            id="SIGNATURE_MISMATCH",
            fired=fired,
            severity=round(min(1.0, dist / (2 * SIGNATURE_THRESHOLD)), 3) if fired else 0.0,
            confidence=round(min(1.0, 0.5 + abs(dist - SIGNATURE_THRESHOLD) * 2), 3),
            reason=(
                f"Signature differs from the {n} reference(s) of account {account} "
                f"(shape distance {dist:.2f} > {SIGNATURE_THRESHOLD})."
                if fired
                else f"Signature matches account {account}'s references "
                f"(distance {dist:.2f}, {n} reference(s))."
            ),
            evidence={"account": account, "distance": round(dist, 3), "references": n},
        )
    )
    return doc
