"""Background processing (FR-APP-09, NFR-REL-02, NFR-SCL-01).

The ``documents`` table is the job queue: an upload is a row with status ``queued``. A worker
claims the oldest queued row, runs the pipeline, stores the result and images, and sets the
final status. Several workers (processes or machines) can share one PostgreSQL database:

* **Claiming** uses ``SELECT ... FOR UPDATE SKIP LOCKED`` on PostgreSQL, so two workers never
  take the same job and neither waits for the other. SQLite has no row locks; there the
  claim is a compare-and-set (``UPDATE ... WHERE status = 'queued'``) and whoever changes
  the row wins.
* **Retries**: a job that raises or whose pipeline fails goes back to ``queued`` until it has
  been tried ``max_attempts`` times, then ``failed`` with the error kept.
* **Crash recovery**: a job still ``processing`` after ``stale_minutes`` belonged to a worker
  that died; it is requeued (and counts as an attempt).
* **Progress**: the pipeline reports each stage, which is written to the row so the UI can
  show it.

Why not Celery + Redis (as the SRS planned)? The queue needs exactly these four things, and
PostgreSQL already provides them with one fewer service to run, secure and back up. Celery
also doesn't support Windows, where this project is developed. See the Phase 8 guide.
"""

from __future__ import annotations

import threading
import time
import traceback
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.db import get_sessionmaker
from app.deps import audit
from app.models import DocStatus, Document, now
from app.settings import get_api_settings
from app.storage import doc_dir, source_path
from findoc.io import save_image
from findoc.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def ocr_enabled() -> bool:
    mode = get_api_settings().ocr
    if mode in ("on", "true", "1"):
        return True
    if mode in ("off", "false", "0"):
        return False
    try:
        import easyocr  # noqa: F401

        return True
    except ImportError:
        return False


def requeue_stale(session: Session) -> int:
    s = get_api_settings()
    cutoff = now() - timedelta(minutes=s.stale_minutes)
    res = session.execute(
        update(Document)
        .where(Document.status == DocStatus.PROCESSING.value, Document.started_at < cutoff)
        .values(status=DocStatus.QUEUED.value, stage=None, error="worker stopped; requeued")
    )
    session.commit()
    return res.rowcount or 0


def claim_next(session: Session) -> str | None:
    """Claim the oldest queued document; its id, or None if the queue is empty."""
    q = (
        select(Document.id)
        .where(Document.status == DocStatus.QUEUED.value)
        .order_by(Document.created_at)
        .limit(1)
    )
    if session.get_bind().dialect.name == "postgresql":
        q = q.with_for_update(skip_locked=True)
    doc_id = session.scalar(q)
    if doc_id is None:
        session.rollback()
        return None
    res = session.execute(
        update(Document)
        .where(Document.id == doc_id, Document.status == DocStatus.QUEUED.value)
        .values(
            status=DocStatus.PROCESSING.value,
            attempts=Document.attempts + 1,
            started_at=now(),
            stage=None,
            progress=0.0,
        )
    )
    session.commit()
    return doc_id if res.rowcount == 1 else None


# Fields whose digits are private (NFR-PRV-01): the API sends only their last 4 digits, so
# the rendered images must not show them either. (field, share of the box from the left to cover)
REDACT = {"account_number": 0.72, "micr": 0.8}


def redaction_quads(doc: Any) -> list[np.ndarray]:
    """Template-coordinate quads to cover on a cheque, leaving the right end (last digits)."""
    from findoc.config import get_settings
    from findoc.localize.templates import default_store

    if not doc.template_id:
        return []
    template = default_store(get_settings().templates_dir).get(doc.template_id)
    quads = []
    for f in template.fields:
        if f.name in REDACT:
            x0, y0, x1, y1 = f.box
            xc = x0 + (x1 - x0) * REDACT[f.name]
            quads.append(np.array([[x0, y0], [xc, y0], [xc, y1], [x0, y1]], np.float32))
    return quads


def _cover(img: np.ndarray, quads: list[np.ndarray], hom: np.ndarray | None = None) -> np.ndarray:
    """Paint each quad (mapped by ``hom``, if given) with a hatched grey patch."""
    import cv2

    out = img.copy()
    for q in quads:
        pts = q.reshape(-1, 1, 2)
        if hom is not None:
            pts = cv2.perspectiveTransform(pts, hom)
        poly = np.round(pts.reshape(-1, 2)).astype(np.int32)
        x, y, w, h = cv2.boundingRect(poly)
        patch = np.full_like(out, 205)
        for k in range(-h, w, max(6, h // 3)):
            cv2.line(patch, (x + k, y + h), (x + k + h, y), (170, 170, 170), 1, cv2.LINE_AA)
        mask = np.zeros(out.shape[:2], np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        out[mask > 0] = patch[mask > 0]  # hatching clipped to the quad
        cv2.polylines(out, [poly], True, (120, 120, 120), 1, cv2.LINE_AA)
    return out


def save_images(doc: Any, out: Path) -> dict[str, str]:
    """Render the review images of a processed pipeline Document into ``out``. On cheques the
    account number and MICR line are covered in every image, mapped through the page geometry;
    when that geometry isn't a homography (3D flattening) the capture isn't saved at all."""
    from findoc.config import get_settings
    from findoc.fraud.viz import tamper_overlay
    from findoc.localize.templates import default_store, draw_fields
    from findoc.preprocess.detect import draw_detection

    images: dict[str, str] = {}
    quads = redaction_quads(doc)
    to_flat = None  # template -> flat page
    if quads and doc.template_homography is not None:
        to_flat = np.linalg.inv(np.array(doc.template_homography, dtype=np.float64))

    def put(variant: str, name: str, img: np.ndarray | None, hom: np.ndarray | None = None) -> None:
        if img is None:
            return
        if quads:
            img = _cover(img, quads, hom)
        save_image(out / name, img)
        images[variant] = name

    flat = doc.enhanced if doc.enhanced is not None else doc.flat
    put("flat", "flat.png", flat, to_flat)
    if doc.original is not None:
        orig = doc.original
        if doc.corners is not None:
            orig = draw_detection(orig, np.array(doc.corners, dtype=np.float32))
        if not quads:
            put("original", "original.jpg", orig)
        elif doc.flattening == "homography" and doc.homography is not None and to_flat is not None:
            # template -> flat -> original capture
            to_orig = np.linalg.inv(np.array(doc.homography, dtype=np.float64)) @ to_flat
            put("original", "original.jpg", orig, to_orig)
    if doc.tamper_map is not None and doc.flat is not None:
        put("heatmap", "heatmap.jpg", tamper_overlay(doc.flat, doc.tamper_map), to_flat)
    if doc.registered is not None and doc.template_id:
        put("page", "page.png", doc.registered)
        template = default_store(get_settings().templates_dir).get(doc.template_id)
        put("annotated", "annotated.png", draw_fields(doc.registered, template))
    else:
        put("page", "page.png", flat, to_flat)
    return images


def _set(doc_id: str, **values: object) -> None:
    with get_sessionmaker()() as s:
        s.execute(update(Document).where(Document.id == doc_id).values(**values))
        s.commit()


def process_document(doc_id: str) -> str:
    """Run the pipeline on a claimed document; returns its new status."""
    from findoc.pipeline import default_pipeline
    from findoc.types import Status

    s = get_api_settings()
    with get_sessionmaker()() as session:
        row = session.get(Document, doc_id)
        assert row is not None
        options = dict(row.options or {})
        attempts = row.attempts
    src = source_path(doc_id)

    def on_stage(name: str, i: int, total: int) -> None:
        _set(doc_id, stage=name, progress=round(i / total, 3))

    try:
        if src is None:
            raise FileNotFoundError("The uploaded file is missing (deleted after retention?)")
        pipe = default_pipeline(
            ocr=ocr_enabled(),
            register=bool(options.get("register")),
            account=options.get("account"),
        )
        doc = pipe.run(src, on_stage=on_stage)
        if doc.status is Status.FAILED:
            failed = next((st for st in doc.stages if not st.ok), None)
            raise RuntimeError(
                f"stage {failed.stage if failed else '?'}: {failed.note if failed else 'failed'}"
            )
        images = save_images(doc, doc_dir(doc_id))
        if doc.status is Status.REJECTED_QUALITY:
            status = DocStatus.REJECTED_QUALITY
        elif doc.risk.get("band") in ("medium", "high"):
            status = DocStatus.NEEDS_REVIEW
        else:
            status = DocStatus.PROCESSED
        _set(
            doc_id,
            status=status.value,
            stage=None,
            progress=1.0,
            error=None,
            result=doc.result(),
            images=images,
            doc_type=doc.doc_type.value,
            risk_score=doc.risk_score,
            risk_band=doc.risk.get("band"),
            finished_at=now(),
        )
        with get_sessionmaker()() as session:
            audit(
                session,
                None,
                "document.processed",
                "document",
                doc_id,
                status=status.value,
                risk_score=doc.risk_score,
                band=doc.risk.get("band"),
            )
            session.commit()
        return status.value
    except Exception as exc:  # noqa: BLE001 - a job must never take the worker down
        log.error("Document %s attempt %d failed: %s", doc_id, attempts, exc)
        final = attempts >= s.max_attempts
        status = DocStatus.FAILED if final else DocStatus.QUEUED
        _set(
            doc_id,
            status=status.value,
            stage=None,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"[-4000:],
            finished_at=now() if final else None,
        )
        if final:
            with get_sessionmaker()() as session:
                audit(session, None, "document.failed", "document", doc_id, error=str(exc))
                session.commit()
        return status.value


def run_once() -> str | None:
    """Process one queued document if there is one; returns its id."""
    with get_sessionmaker()() as session:
        doc_id = claim_next(session)
    if doc_id is not None:
        process_document(doc_id)
    return doc_id


def run_forever(stop: threading.Event | None = None, max_jobs: int | None = None) -> int:
    """Worker loop: process jobs until ``stop`` is set (or ``max_jobs`` are done)."""
    stop = stop or threading.Event()
    s = get_api_settings()
    done = 0
    last_sweep = 0.0
    log.info("Worker started (OCR %s)", "on" if ocr_enabled() else "off")
    while not stop.is_set() and (max_jobs is None or done < max_jobs):
        if time.monotonic() - last_sweep > 60:
            with get_sessionmaker()() as session:
                n = requeue_stale(session)
            if n:
                log.warning("Requeued %d stale job(s)", n)
            last_sweep = time.monotonic()
        try:
            doc_id = run_once()
        except Exception:  # noqa: BLE001 - database hiccup: wait and retry
            log.exception("Worker loop error")
            doc_id = None
        if doc_id is None:
            stop.wait(s.poll_seconds)
        else:
            done += 1
    return done


def start_inline_worker() -> tuple[threading.Thread, threading.Event]:
    stop = threading.Event()
    t = threading.Thread(target=run_forever, args=(stop,), name="findoc-worker", daemon=True)
    t.start()
    return t, stop
