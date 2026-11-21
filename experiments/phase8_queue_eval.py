"""Phase 8: does the job queue scale with workers, and does it lose or repeat jobs?

    python experiments/phase8_queue_eval.py --db postgresql+psycopg://user:pw@host/db --n 500

For each worker count, the tables are recreated, ``--n`` synthetic cheque photos are queued
(a few different photos, uploaded many times), and that many worker processes drain the queue
(OCR off, so the timing is the CV pipeline and the queue, not the OCR model). Reported per
worker count: wall time, documents per minute, speed-up over one worker, and the integrity
checks: every document processed exactly once (one attempt, one "processed" audit entry), none
failed or left behind.

Needs a database that several processes can share (PostgreSQL; SQLite works but serialises
writes). Uses the [api] extra.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

PHOTOS = 6


def _env(db: str, storage: str) -> None:
    os.environ.update({
        "FINDOC_API_DATABASE_URL": db,
        "FINDOC_API_STORAGE_DIR": storage,
        "FINDOC_API_OCR": "off",
        "FINDOC_API_INLINE_WORKER": "false",
        "FINDOC_SEGMENTATION": "off",
        "FINDOC_TAMPER": "off",
    })  # fmt: skip


def _worker(db: str, storage: str, out: mp.Queue) -> None:
    _env(db, storage)
    from app.worker import run_once

    done, idle = 0, 0
    while idle < 3:  # three empty polls in a row: the queue is drained
        if run_once():
            done += 1
            idle = 0
        else:
            idle += 1
            time.sleep(0.2)
    out.put(done)


def run(db: str, n: int, workers: int, photos: list[bytes], storage: str) -> dict:
    _env(db, storage)
    from sqlalchemy import func, select

    from app import db as dbm
    from app.models import AuditLog, Document, User, new_id
    from app.storage import doc_dir
    from app.uploads import store_upload

    dbm.get_engine.cache_clear()
    dbm.get_sessionmaker.cache_clear()
    dbm.Base.metadata.drop_all(dbm.get_engine())
    dbm.init_db()
    with dbm.get_sessionmaker()() as s:
        user = User(id=new_id(), email="load@test", role="submitter", password_hash="x")
        s.add(user)
        for i in range(n):
            row = Document(id=new_id(), owner_id=user.id, filename=f"cheque_{i}.jpg",
                           media_type="", kind="", size_bytes=0, sha256="")  # fmt: skip
            st = store_upload(photos[i % len(photos)], doc_dir(row.id) / "source", 10**8)
            row.media_type, row.kind, row.size_bytes, row.sha256 = (
                st.media_type, st.kind, st.size_bytes, st.sha256)  # fmt: skip
            s.add(row)
        s.commit()
    dbm.get_engine().dispose()

    out: mp.Queue = mp.Queue()
    t0 = time.perf_counter()
    procs = [mp.Process(target=_worker, args=(db, storage, out)) for _ in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0
    per_worker = sorted(out.get() for _ in procs)

    with dbm.get_sessionmaker()() as s:
        statuses = dict(
            s.execute(select(Document.status, func.count()).group_by(Document.status)).all()
        )
        attempts = dict(
            s.execute(select(Document.attempts, func.count()).group_by(Document.attempts)).all()
        )
        audited = s.scalar(select(func.count()).where(AuditLog.action == "document.processed"))
    return {
        "workers": workers,
        "documents": n,
        "seconds": round(wall, 1),
        "per_minute": round(n / wall * 60, 1),
        "jobs_per_worker": per_worker,
        "statuses": statuses,
        "attempts": {str(k): v for k, v in attempts.items()},
        "exactly_once": attempts == {1: n} and audited == n and sum(per_worker) == n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="SQLAlchemy URL of a scratch database")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--out", type=Path, default=Path("experiments/results/phase8_queue.json"))
    a = ap.parse_args()

    import cv2

    from findoc.synth.dataset import cheque_photo

    photos = [cv2.imencode(".jpg", cheque_photo(100 + i)[0])[1].tobytes() for i in range(PHOTOS)]
    results = []
    for w in a.workers:
        with tempfile.TemporaryDirectory() as storage:
            r = run(a.db, a.n, w, photos, storage)
        if results:
            r["speedup"] = round(results[0]["seconds"] / r["seconds"], 2)
        results.append(r)
        print(json.dumps(r), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"cpu_cores": os.cpu_count(), "runs": results}, indent=2))
    print(f"Saved {a.out}")


if __name__ == "__main__":
    main()
