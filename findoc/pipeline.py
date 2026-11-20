"""Pipeline orchestration.

A pipeline is an ordered list of stages. Each stage is a callable that takes a
:class:`~findoc.types.Document` and returns it (usually mutated). Stages are added
phase by phase.

Stage roadmap (see README):
    Phase 1  preprocess   quality gate, detection, rectification, illumination, OCR  (done)
    Phase 2  localize     cheque template alignment, field recognition, rules  (done)
    Phase 3  track        KLT tracking and keyframe selection (video)  (done)
    Phase 4  segment      learned page detection (used inside detect and track)  (done)
    Phase 5  recon3d      SfM, surface fit, ARAP flattening (video)  (done)
    Phase 6  fraud        tamper map, duplicates, signature, risk score  (done)
    Phase 7  invoices     LayoutLMv3 extraction, GST checks
"""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

from findoc.io import detect_kind, load_image
from findoc.logging import get_logger
from findoc.types import Document, InputKind, StageLog, Status

log = get_logger(__name__)

Stage = Callable[[Document], Document]


def load_stage(doc: Document) -> Document:
    if doc.kind is InputKind.IMAGE:
        doc.original = load_image(doc.source)
        h, w = doc.original.shape[:2]
        doc.messages.append(f"Loaded image {w}x{h}")
    return doc  # video frames are read by the Phase 3 video stage


class Pipeline:
    def __init__(self, stages: list[tuple[str, Stage]] | None = None) -> None:
        self.stages: list[tuple[str, Stage]] = stages or [("load", load_stage)]

    def add(self, name: str, stage: Stage) -> Pipeline:
        self.stages.append((name, stage))
        return self

    def run(self, source: Path) -> Document:
        doc = Document(source=source, kind=detect_kind(source))
        for name, stage in self.stages:
            if doc.status is not Status.OK:
                break
            t0 = time.perf_counter()
            try:
                doc = stage(doc)
                ok, note = True, ""
            except Exception as exc:  # noqa: BLE001 - record failure, don't crash
                log.exception("Stage %s failed", name)
                doc.status = Status.FAILED
                ok, note = False, str(exc)
            doc.stages.append(
                StageLog(stage=name, ms=(time.perf_counter() - t0) * 1000, ok=ok, note=note)
            )
        return doc


def default_pipeline(
    ocr: bool = True, register: bool = False, account: str | None = None
) -> Pipeline:
    """The full pipeline for the current phase.

    Phase 3: load -> video -> track (video only: best frame + keyframes)
    Phase 1: -> quality -> detect -> (Phase 5: recon3d, video only) -> rectify -> enhance
    Phase 2: -> align (cheque template) -> fields -> rules
    Phase 6: -> tamper -> duplicate -> signature
    then:    -> ocr (full-page text, for documents that aren't known cheques)
    Phase 6: -> risk (score from every check that ran)
    """
    # Imported here so `findoc --help` stays fast and import cycles are impossible.
    from findoc.fraud.stages import (
        duplicate_stage,
        risk_stage,
        rules_stage,
        signature_stage,
        tamper_stage,
    )
    from findoc.localize.stages import align_stage, make_fields_stage
    from findoc.preprocess.stages import detect_stage, enhance_stage, quality_stage, rectify_stage
    from findoc.recognize.ocr import ocr_stage
    from findoc.recon3d.stages import recon3d_stage
    from findoc.track.stages import track_stage, video_load_stage

    pipe = Pipeline()
    pipe.add("video", video_load_stage)
    pipe.add("track", track_stage)
    pipe.add("quality", quality_stage)
    pipe.add("detect", detect_stage)
    pipe.add("recon3d", recon3d_stage)
    pipe.add("rectify", rectify_stage)
    pipe.add("enhance", enhance_stage)
    pipe.add("align", align_stage)
    pipe.add("fields", make_fields_stage(use_ocr=ocr))
    pipe.add("rules", rules_stage)
    pipe.add("tamper", tamper_stage)
    pipe.add("duplicate", partial(duplicate_stage, register=register))
    pipe.add("signature", partial(signature_stage, account=account))
    if ocr:
        pipe.add("ocr", ocr_stage)
    pipe.add("risk", risk_stage)
    return pipe


def process(
    source: str | Path, ocr: bool = True, register: bool = False, account: str | None = None
) -> Document:
    """Run the default pipeline on one input file (Phase 6 options: ``register`` adds the cheque
    to the duplicate index after checking it; ``account`` names the account whose reference
    signatures to compare with when OCR doesn't read it)."""
    return default_pipeline(ocr=ocr, register=register, account=account).run(Path(source))
