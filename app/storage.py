"""Where a document's files live: ``<storage_dir>/documents/<id>/``.

    source.png | source.mp4 ...   the sanitised upload (deleted after the retention period)
    page.png                      base image for field boxes: the registered cheque
                                  (template coordinates) or the flat page
    flat.png                      flattened, enhanced page
    heatmap.jpg                   tamper heatmap over the flat page
    annotated.png                 cheque with its template's field boxes drawn
    original.jpg                  capture (best video frame) with the detected outline

A local folder (a Docker volume in Compose). S3 / MinIO would replace these few functions.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from app.settings import get_api_settings

VARIANTS = ("page", "flat", "heatmap", "annotated", "original")


def doc_dir(doc_id: str) -> Path:
    return get_api_settings().storage_dir / "documents" / doc_id


def exports_dir() -> Path:
    d = get_api_settings().storage_dir / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def source_path(doc_id: str) -> Path | None:
    hits = sorted(doc_dir(doc_id).glob("source.*"))
    return hits[0] if hits else None


def delete_source(doc_id: str) -> bool:
    p = source_path(doc_id)
    if p is None:
        return False
    p.unlink(missing_ok=True)
    return True


def delete_all(doc_id: str) -> None:
    shutil.rmtree(doc_dir(doc_id), ignore_errors=True)
