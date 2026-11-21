"""Documents: upload, list, read, images, field corrections, decisions (SRS 11.3).

Object-level access: a submitter sees only their own documents, and asking for someone
else's gives 404, not 403, so ids can't be probed for existence (OWASP API1, broken object
level authorisation). Reviewers and admins see all.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.db import get_db
from app.deps import ROLE_RANK, audit, current_user, require
from app.models import (
    OPEN_FOR_REVIEW,
    Decision,
    DocStatus,
    Document,
    FieldEdit,
    Role,
    User,
    new_id,
    now,
)
from app.schemas import (
    BulkDecisionIn,
    BulkDecisionOut,
    CheckView,
    DecisionIn,
    DecisionOut,
    DocumentDetail,
    DocumentList,
    DocumentSummary,
    FieldEditOut,
    FieldsPatch,
    FieldView,
    Stats,
    Submitted,
)
from app.settings import get_api_settings
from app.storage import VARIANTS, delete_all, doc_dir
from app.uploads import UploadError, store_upload

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

MASKED_FIELDS = {"account_number", "micr"}  # NFR-PRV-01: only the last 4 digits are shown


def mask_digits(text: str | None, keep: int = 4) -> str | None:
    """Replace every digit except the last ``keep`` with a bullet."""
    if text is None:
        return None
    total = sum(ch.isdigit() for ch in text)
    out, seen = [], 0
    for ch in text:
        if ch.isdigit():
            seen += 1
            out.append(ch if seen > total - keep else "•")
        else:
            out.append(ch)
    return "".join(out)


def _as_utc(stamp: datetime) -> datetime:
    """SQLite returns naive datetimes (stored as UTC); PostgreSQL returns aware ones."""
    return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)


def is_reviewer(user: User) -> bool:
    return ROLE_RANK[user.role] >= ROLE_RANK[Role.REVIEWER.value]


def get_visible(db: Session, doc_id: str, user: User) -> Document:
    row = db.get(Document, doc_id)
    if row is None or (not is_reviewer(user) and row.owner_id != user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return row


def summary(row: Document) -> DocumentSummary:
    return DocumentSummary(
        id=row.id,
        filename=row.filename,
        kind=row.kind,
        status=row.status,
        stage=row.stage,
        progress=row.progress,
        attempts=row.attempts,
        doc_type=row.doc_type,
        risk_score=row.risk_score,
        risk_band=row.risk_band,
        owner_email=row.owner.email,
        created_at=row.created_at,
        finished_at=row.finished_at,
    )


def allowed_decisions(row: Document, user: User) -> list[str]:
    if not is_reviewer(user):
        return []
    if row.status in {s.value for s in OPEN_FOR_REVIEW}:
        return ["approved", "rejected", "escalated"]
    if row.status == DocStatus.ESCALATED.value and user.role == Role.ADMIN.value:
        return ["approved", "rejected"]
    return []


def detail(row: Document, user: User) -> DocumentDetail:
    r: dict[str, Any] = row.result or {}
    contrib = {
        c["check"]: c["contribution"] for c in (r.get("risk") or {}).get("contributions", [])
    }
    fields = []
    for f in r.get("fields", []):
        masked = f["name"] in MASKED_FIELDS
        b = f.get("bbox")
        fields.append(
            FieldView(
                name=f["name"],
                value=mask_digits(f.get("value")) if masked else f.get("value"),
                normalized=mask_digits(f.get("normalized")) if masked else f.get("normalized"),
                corrected=(row.corrections or {}).get(f["name"]),
                confidence=f.get("confidence", 0.0),
                bbox=[b["x0"], b["y0"], b["x1"], b["y1"]] if b else None,
                method=f.get("method"),
            )
        )
    checks = [
        CheckView(
            id=c["id"],
            fired=c["fired"],
            severity=c.get("severity", 0.0),
            confidence=c.get("confidence", 0.0),
            reason=c.get("reason", ""),
            contribution=contrib.get(c["id"]),
            evidence={k: v for k, v in (c.get("evidence") or {}).items() if k != "account"},
        )
        for c in r.get("checks", [])
    ]
    base = summary(row).model_dump()
    return DocumentDetail(
        **base,
        media_type=row.media_type,
        size_bytes=row.size_bytes,
        error=row.error if is_reviewer(user) else (row.error or "").split("\n")[0] or None,
        images=[v for v in VARIANTS if v in (row.images or {})],
        page_frame="template" if r.get("template_id") else "flat",
        fields=fields,
        line_items=r.get("line_items", []),
        checks=checks,
        risk=r.get("risk", {}),
        quality=r.get("quality", {}),
        messages=r.get("messages", []),
        stages=r.get("stages", []),
        flattening=r.get("flattening"),
        decisions=[
            DecisionOut(
                decision=d.decision,
                comment=d.comment,
                user_email=d.user.email,
                created_at=d.created_at,
            )
            for d in row.decisions
        ],
        field_edits=[
            FieldEditOut(
                field=e.field,
                old_value=e.old_value,
                new_value=e.new_value,
                user_email=e.user.email,
                created_at=e.created_at,
            )
            for e in row.field_edits
        ],
        allowed_decisions=allowed_decisions(row, user),
    )


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=Submitted)
async def submit(
    file: UploadFile = File(..., description="Photo (JPEG/PNG/WebP/BMP/TIFF) or video"),
    add_to_index: bool = Form(
        False, alias="register", description="Add the cheque to the duplicate index"
    ),
    account: str | None = Form(None, max_length=34, description="Account for signature check"),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> Submitted:
    """Upload a document. Processing is asynchronous: poll ``GET /documents/{id}``."""
    s = get_api_settings()
    max_bytes = s.max_upload_mb * 2**20
    data = await file.read(max_bytes + 1)
    safe_name = re.sub(r"[^\w.\- ]", "_", file.filename or "upload")[:255]
    row = Document(
        id=new_id(),
        owner_id=user.id,
        filename=safe_name,
        media_type="",
        kind="",
        size_bytes=0,
        sha256="",
        options={"register": add_to_index, "account": account},
    )
    try:
        stored = store_upload(data, doc_dir(row.id) / "source", max_bytes)
    except UploadError as exc:
        delete_all(row.id)
        raise HTTPException(422, str(exc)) from exc
    row.media_type, row.kind = stored.media_type, stored.kind
    row.size_bytes, row.sha256 = stored.size_bytes, stored.sha256
    dup = db.scalar(
        select(Document.id).where(Document.sha256 == stored.sha256, Document.owner_id == user.id)
    )
    db.add(row)
    audit(
        db,
        user,
        "document.submitted",
        "document",
        row.id,
        filename=row.filename,
        media_type=row.media_type,
        size=row.size_bytes,
    )
    db.commit()
    return Submitted(id=row.id, status=row.status, duplicate_of=dup)


@router.get("", response_model=DocumentList)
def list_documents(
    status_: list[str] | None = Query(None, alias="status"),
    band: list[str] | None = Query(None),
    doc_type: str | None = None,
    q: str | None = Query(None, max_length=100, description="Filename contains"),
    date_from: datetime | None = Query(None, alias="from"),
    date_to: datetime | None = Query(None, alias="to"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> DocumentList:
    """List documents, newest first. Submitters see their own."""
    stmt = select(Document)
    if not is_reviewer(user):
        stmt = stmt.where(Document.owner_id == user.id)
    if status_:
        stmt = stmt.where(Document.status.in_(status_))
    if band:
        stmt = stmt.where(Document.risk_band.in_(band))
    if doc_type:
        stmt = stmt.where(Document.doc_type == doc_type)
    if q:
        stmt = stmt.where(Document.filename.ilike(f"%{q}%"))
    if date_from:
        stmt = stmt.where(Document.created_at >= date_from)
    if date_to:
        stmt = stmt.where(Document.created_at <= date_to)
    total = db.scalar(select(func.count()).select_from(stmt.subquery())) or 0
    rows = db.scalars(
        stmt.options(selectinload(Document.owner))
        .order_by(Document.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return DocumentList(items=[summary(r) for r in rows], total=total)


@router.get("/stats", response_model=Stats)
def stats(db: Session = Depends(get_db), _: User = Depends(require(Role.REVIEWER))) -> Stats:
    """Counts for the dashboard (FR-APP-03)."""

    def counts(col) -> dict[str, int]:  # noqa: ANN001
        return {
            (k or "none"): v for k, v in db.execute(select(col, func.count()).group_by(col)).all()
        }

    by_status = counts(Document.status)
    today = now().date()
    days = [today - timedelta(days=i) for i in range(13, -1, -1)]
    daily = {
        day: {"date": day.isoformat(), "received": 0, "decided": 0, "flagged": 0} for day in days
    }
    since = datetime.combine(days[0], datetime.min.time(), tzinfo=timezone.utc)
    for created, decided, band in db.execute(
        select(Document.created_at, Document.decided_at, Document.risk_band).where(
            (Document.created_at >= since) | (Document.decided_at >= since)
        )
    ):
        for stamp, key in ((created, "received"), (decided, "decided")):
            day = _as_utc(stamp).date() if stamp else None
            if day in daily:
                daily[day][key] += 1
        if band in ("medium", "high") and _as_utc(created).date() in daily:
            daily[_as_utc(created).date()]["flagged"] += 1
    by_check: dict[str, int] = {}
    seconds = []
    hist = [0] * 10
    recent = db.execute(
        select(Document.result, Document.started_at, Document.finished_at, Document.risk_score)
        .where(Document.result.is_not(None))
        .order_by(Document.created_at.desc())
        .limit(1000)
    ).all()
    for result, started, finished, score in recent:
        for c in (result or {}).get("checks", []):
            if c.get("fired"):
                by_check[c["id"]] = by_check.get(c["id"], 0) + 1
        if started and finished:
            seconds.append((_as_utc(finished) - _as_utc(started)).total_seconds())
        if score is not None:
            hist[min(9, int(score) // 10)] += 1
    return Stats(
        daily=list(daily.values()),
        by_check=dict(sorted(by_check.items(), key=lambda kv: -kv[1])),
        median_seconds=round(sorted(seconds)[len(seconds) // 2], 1) if seconds else None,
        risk_histogram=hist,
        total=sum(by_status.values()),
        by_status=by_status,
        by_band=counts(Document.risk_band),
        by_type=counts(Document.doc_type),
        average_risk=db.scalar(select(func.avg(Document.risk_score))),
        open_reviews=sum(by_status.get(s.value, 0) for s in OPEN_FOR_REVIEW)
        + by_status.get(DocStatus.ESCALATED.value, 0),
    )


@router.post("/bulk-decision", response_model=BulkDecisionOut)
def bulk_decide(
    body: BulkDecisionIn,
    db: Session = Depends(get_db),
    user: User = Depends(require(Role.REVIEWER)),
) -> BulkDecisionOut:
    """Decide several documents at once (for example, approve the low-risk ones). Each is
    checked like a single decision; the ones that can't take it are skipped, with the reason."""
    if body.decision in ("rejected", "escalated") and not body.comment.strip():
        raise HTTPException(422, "A comment is required to reject or escalate")
    done, skipped = [], {}
    for doc_id in dict.fromkeys(body.ids):
        row = db.get(Document, doc_id)
        if row is None:
            skipped[doc_id] = "not found"
            continue
        if body.decision not in allowed_decisions(row, user):
            skipped[doc_id] = f"is {row.status}"
            continue
        previous = row.status
        row.status, row.decided_at = body.decision, now()
        db.add(
            Decision(
                document_id=row.id,
                user_id=user.id,
                decision=body.decision,
                comment=body.comment.strip(),
            )
        )
        audit(
            db,
            user,
            f"document.{body.decision}",
            "document",
            row.id,
            previous=previous,
            comment=body.comment.strip(),
            bulk=True,
        )
        done.append(doc_id)
    db.commit()
    return BulkDecisionOut(done=done, skipped=skipped)


@router.get("/{doc_id}", response_model=DocumentDetail)
def get_document(
    doc_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> DocumentDetail:
    """Status, progress and (when processed) the full result."""
    return detail(get_visible(db, doc_id, user), user)


@router.get("/{doc_id}/image", response_class=FileResponse)
def get_image(
    doc_id: str,
    variant: str = Query("page", pattern="^(" + "|".join(VARIANTS) + ")$"),
    db: Session = Depends(get_db),
    user: User = Depends(current_user),
) -> FileResponse:
    """A rendered image: page (base for field boxes), flat, heatmap, annotated, original."""
    row = get_visible(db, doc_id, user)
    name = (row.images or {}).get(variant)
    path = doc_dir(row.id) / name if name else None
    if path is None or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No {variant} image for this document")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})


@router.patch("/{doc_id}/fields", response_model=DocumentDetail)
def correct_fields(
    doc_id: str,
    body: FieldsPatch,
    db: Session = Depends(get_db),
    user: User = Depends(require(Role.REVIEWER)),
) -> DocumentDetail:
    """Save the reviewer's corrected values (FR-APP-01). The pipeline's reading is kept;
    the correction is stored beside it with who made it and when."""
    row = get_visible(db, doc_id, user)
    if row.status in (DocStatus.QUEUED.value, DocStatus.PROCESSING.value):
        raise HTTPException(status.HTTP_409_CONFLICT, "The document is still being processed")
    known = {f["name"]: f.get("value") for f in (row.result or {}).get("fields", [])}
    unknown = set(body.fields) - set(known)
    if unknown:
        raise HTTPException(422, f"Unknown field(s): {', '.join(sorted(unknown))}")
    corrections = dict(row.corrections or {})
    for name, value in body.fields.items():
        old = corrections.get(name, known[name])
        if value == old:
            continue
        corrections[name] = value
        db.add(
            FieldEdit(
                document_id=row.id, user_id=user.id, field=name, old_value=old, new_value=value
            )
        )
        logged = (mask_digits(old), mask_digits(value)) if name in MASKED_FIELDS else (old, value)
        audit(
            db,
            user,
            "field.corrected",
            "document",
            row.id,
            field=name,
            old=logged[0],
            new=logged[1],
        )
    row.corrections = corrections
    db.commit()
    db.refresh(row)
    return detail(row, user)


@router.post("/{doc_id}/decision", response_model=DocumentDetail)
def decide(
    doc_id: str,
    body: DecisionIn,
    db: Session = Depends(get_db),
    user: User = Depends(require(Role.REVIEWER)),
) -> DocumentDetail:
    """Approve, reject or escalate (FR-APP-02). Escalated documents are decided by an admin."""
    row = get_visible(db, doc_id, user)
    allowed = allowed_decisions(row, user)
    if body.decision not in allowed:
        if row.status == DocStatus.ESCALATED.value:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Escalated documents need an admin")
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"Can't mark a {row.status} document {body.decision}"
        )
    if body.decision in ("rejected", "escalated") and not body.comment.strip():
        raise HTTPException(422, "A comment is required to reject or escalate")
    previous = row.status
    row.status = body.decision
    row.decided_at = now()
    db.add(
        Decision(
            document_id=row.id,
            user_id=user.id,
            decision=body.decision,
            comment=body.comment.strip(),
        )
    )
    audit(
        db,
        user,
        f"document.{body.decision}",
        "document",
        row.id,
        previous=previous,
        comment=body.comment.strip(),
    )
    db.commit()
    db.refresh(row)
    return detail(row, user)


@router.post("/{doc_id}/retry", response_model=DocumentSummary)
def retry(
    doc_id: str, db: Session = Depends(get_db), user: User = Depends(require(Role.REVIEWER))
) -> DocumentSummary:
    """Queue a failed document again (attempts start over)."""
    row = get_visible(db, doc_id, user)
    if row.status != DocStatus.FAILED.value:
        raise HTTPException(status.HTTP_409_CONFLICT, "Only failed documents can be retried")
    if row.media_deleted:
        raise HTTPException(status.HTTP_409_CONFLICT, "The upload was deleted (retention)")
    row.status, row.attempts, row.error = DocStatus.QUEUED.value, 0, None
    audit(db, user, "document.retried", "document", row.id)
    db.commit()
    return summary(row)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    doc_id: str, db: Session = Depends(get_db), user: User = Depends(require(Role.ADMIN))
) -> None:
    """Erase a document and its files (deletion on request, DPDP Act). The audit log keeps
    that it existed and who erased it, not its content."""
    row = get_visible(db, doc_id, user)
    for child in (*row.decisions, *row.field_edits):
        db.delete(child)
    db.delete(row)
    delete_all(doc_id)
    audit(db, user, "document.deleted", "document", doc_id, filename=row.filename)
    db.commit()
