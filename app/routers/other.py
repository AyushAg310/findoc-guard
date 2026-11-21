"""Auth, users and API keys, exports, rules, audit log, health and metrics."""

from __future__ import annotations

import copy
import json
import time
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app import __version__
from app.db import get_db
from app.deps import audit, current_user, require
from app.models import ApiKey, AuditLog, Document, Export, Role, User, new_id
from app.routers.documents import MASKED_FIELDS, mask_digits
from app.schemas import (
    ApiKeyIn,
    ApiKeyOut,
    AuditOut,
    ExportIn,
    ExportOut,
    TokenOut,
    UserIn,
    UserOut,
    UserPatch,
)
from app.security import create_token, hash_password, new_api_key, verify_password
from app.settings import get_api_settings
from app.storage import exports_dir

auth = APIRouter(prefix="/api/v1/auth", tags=["auth"])
admin = APIRouter(prefix="/api/v1", tags=["admin"])
exports = APIRouter(prefix="/api/v1/exports", tags=["exports"])
system = APIRouter(tags=["system"])

_DUMMY_HASH = hash_password("timing-equaliser")


# --------------------------------------------------------------------------- auth


@auth.post("/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> TokenOut:
    """Email (as ``username``) and password -> a JWT for ``Authorization: Bearer``."""
    s = get_api_settings()
    user = db.scalar(select(User).where(func.lower(User.email) == form.username.lower()))
    # verify against a dummy hash for unknown users too, so timing doesn't reveal which
    # emails exist
    ok = verify_password(form.password, user.password_hash if user else _DUMMY_HASH)
    if user is None or not ok or not user.active:
        audit(db, None, "auth.failed", "user", None, email=form.username[:254])
        db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong email or password")
    audit(db, user, "auth.login", "user", user.id)
    db.commit()
    token = create_token(user.id, user.role, s.jwt_secret, s.jwt_minutes)
    return TokenOut(access_token=token, expires_in=s.jwt_minutes * 60)


@auth.get("/me", response_model=UserOut)
def me(user: User = Depends(current_user)) -> User:
    return user


# --------------------------------------------------------------------------- users, keys


@admin.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db), _: User = Depends(require(Role.ADMIN))) -> Any:
    return db.scalars(select(User).order_by(User.created_at)).all()


@admin.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(
    body: UserIn, db: Session = Depends(get_db), actor: User = Depends(require(Role.ADMIN))
) -> User:
    if db.scalar(select(User).where(func.lower(User.email) == body.email.lower())):
        raise HTTPException(status.HTTP_409_CONFLICT, "A user with this email exists")
    user = User(
        id=new_id(),
        email=body.email.lower(),
        name=body.name,
        role=body.role.value,
        password_hash=hash_password(body.password),
    )
    db.add(user)
    audit(db, actor, "user.created", "user", user.id, email=user.email, role=user.role)
    db.commit()
    return user


@admin.patch("/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: str,
    body: UserPatch,
    db: Session = Depends(get_db),
    actor: User = Depends(require(Role.ADMIN)),
) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    if user.id == actor.id and (body.active is False or (body.role and body.role != Role.ADMIN)):
        raise HTTPException(status.HTTP_409_CONFLICT, "You can't demote or disable yourself")
    changed: dict[str, Any] = {}
    if body.name is not None:
        user.name = changed["name"] = body.name
    if body.role is not None:
        user.role = changed["role"] = body.role.value
    if body.active is not None:
        user.active = changed["active"] = body.active
    if body.password is not None:
        user.password_hash = hash_password(body.password)
        changed["password"] = "changed"
    audit(db, actor, "user.updated", "user", user.id, **changed)
    db.commit()
    return user


@admin.get("/api-keys", response_model=list[ApiKeyOut])
def list_keys(db: Session = Depends(get_db), user: User = Depends(current_user)) -> Any:
    """Your own API keys (the secret part is never shown again)."""
    return db.scalars(select(ApiKey).where(ApiKey.user_id == user.id)).all()


@admin.post("/api-keys", response_model=ApiKeyOut, status_code=status.HTTP_201_CREATED)
def create_key(
    body: ApiKeyIn, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> ApiKeyOut:
    """Create an API key acting as you. Copy ``key`` now: only its hash is stored."""
    key, prefix, digest = new_api_key()
    row = ApiKey(id=new_id(), user_id=user.id, name=body.name, prefix=prefix, key_hash=digest)
    db.add(row)
    audit(db, user, "apikey.created", "apikey", row.id, prefix=prefix)
    db.commit()
    return ApiKeyOut.model_validate(row).model_copy(update={"key": key})


@admin.delete("/api-keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
def revoke_key(
    key_id: str, db: Session = Depends(get_db), user: User = Depends(current_user)
) -> None:
    row = db.get(ApiKey, key_id)
    if row is None or (row.user_id != user.id and user.role != Role.ADMIN.value):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "API key not found")
    row.revoked = True
    audit(db, user, "apikey.revoked", "apikey", row.id)
    db.commit()


@admin.get("/audit", response_model=list[AuditOut])
def audit_log(
    target_id: str | None = None,
    action: str | None = None,
    limit: int = 100,
    db: Session = Depends(get_db),
    _: User = Depends(require(Role.ADMIN)),
) -> Any:
    stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(min(max(limit, 1), 500))
    if target_id:
        stmt = stmt.where(AuditLog.target_id == target_id)
    if action:
        stmt = stmt.where(AuditLog.action.like(f"{action}%"))
    return db.scalars(stmt).all()


@admin.get("/rules")
def rules(_: User = Depends(require(Role.REVIEWER))) -> dict[str, Any]:
    """Risk points per check and band thresholds in force (set with FINDOC_RISK_* variables)."""
    from findoc.config import get_settings
    from findoc.fraud.risk import CRITICAL, POINTS

    s = get_settings()
    return {
        "points": {**POINTS, **s.risk_points},
        "critical": CRITICAL,
        "bands": {"medium": s.risk_medium, "high": s.risk_high},
    }


# --------------------------------------------------------------------------- exports


def export_record(row: Document) -> dict[str, Any]:
    """The result as exported: reviewer corrections applied, account numbers masked."""
    r = copy.deepcopy(row.result or {})
    r["source"] = row.filename
    r["status"] = row.status
    for f in r.get("fields", []):
        if f["name"] in (row.corrections or {}):
            f["value"] = f["normalized"] = row.corrections[f["name"]]
            f["method"] = "reviewer"
        if f["name"] in MASKED_FIELDS:
            for k in ("value", "raw", "normalized"):
                f[k] = mask_digits(f.get(k))
    r.pop("words", None)
    r["document_id"] = row.id
    r["decisions"] = [
        {
            "decision": d.decision,
            "comment": d.comment,
            "by": d.user.email,
            "at": d.created_at.isoformat(),
        }
        for d in row.decisions
    ]
    return r


@exports.post("", response_model=ExportOut, status_code=status.HTTP_201_CREATED)
def create_export(
    body: ExportIn, db: Session = Depends(get_db), user: User = Depends(require(Role.REVIEWER))
) -> ExportOut:
    """Export processed documents matching the filter as JSON or Excel (FR-APP-04)."""
    stmt = select(Document).where(Document.result.is_not(None))
    if body.status:
        stmt = stmt.where(Document.status.in_(body.status))
    if body.band:
        stmt = stmt.where(Document.risk_band.in_(body.band))
    if body.doc_type:
        stmt = stmt.where(Document.doc_type == body.doc_type)
    rows = db.scalars(stmt.order_by(Document.created_at)).all()
    records = [export_record(r) for r in rows]
    ex = Export(
        id=new_id(),
        user_id=user.id,
        format=body.format,
        count=len(records),
        filters=body.model_dump(exclude={"format"}),
        filename="",
    )
    ex.filename = f"findoc-export-{ex.id[:8]}.{body.format}"
    path = exports_dir() / ex.filename
    if body.format == "json":
        path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")
    else:
        from findoc.export.excel import write_workbook

        write_workbook(records, path)
    db.add(ex)
    audit(db, user, "export.created", "export", ex.id, count=ex.count, format=ex.format)
    db.commit()
    return ExportOut(
        id=ex.id, format=ex.format, count=ex.count, download=f"/api/v1/exports/{ex.id}"
    )


@exports.get("/{export_id}", response_class=FileResponse)
def download_export(
    export_id: str, db: Session = Depends(get_db), user: User = Depends(require(Role.REVIEWER))
) -> FileResponse:
    ex = db.get(Export, export_id)
    if ex is None or not (exports_dir() / ex.filename).is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Export not found")
    audit(db, user, "export.downloaded", "export", ex.id)
    db.commit()
    return FileResponse(exports_dir() / ex.filename, filename=ex.filename)


# --------------------------------------------------------------------------- health, metrics


class Metrics:
    """Request counters for /metrics (Prometheus text format), kept in this process."""

    def __init__(self) -> None:
        self.count: dict[tuple[str, str, int], int] = defaultdict(int)
        self.seconds: dict[tuple[str, str], float] = defaultdict(float)
        self.started = time.time()

    def observe(self, method: str, route: str, code: int, seconds: float) -> None:
        self.count[(method, route, code)] += 1
        self.seconds[(method, route)] += seconds


METRICS = Metrics()


@system.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, Any]:
    """Liveness and readiness: the database answers."""
    db.execute(select(1))
    queued = db.scalar(select(func.count()).where(Document.status == "queued"))
    return {"status": "ok", "version": __version__, "queued": queued}


@system.get("/metrics", response_class=PlainTextResponse)
def metrics(request: Request, db: Session = Depends(get_db)) -> str:
    """Prometheus metrics: HTTP requests, documents by status, pipeline stage latency over
    the last 500 processed documents (NFR-OBS-01)."""
    lines = [
        "# HELP findoc_http_requests_total HTTP requests by route and status code",
        "# TYPE findoc_http_requests_total counter",
    ]
    for (m, route, code), n in sorted(METRICS.count.items()):
        lines.append(
            f'findoc_http_requests_total{{method="{m}",route="{route}",code="{code}"}} {n}'
        )
    lines += [
        "# HELP findoc_http_request_seconds_sum Time spent serving requests",
        "# TYPE findoc_http_request_seconds_sum counter",
    ]
    for (m, route), sec in sorted(METRICS.seconds.items()):
        lines.append(f'findoc_http_request_seconds_sum{{method="{m}",route="{route}"}} {sec:.4f}')
    lines += ["# HELP findoc_documents Documents by status", "# TYPE findoc_documents gauge"]
    for st, n in db.execute(select(Document.status, func.count()).group_by(Document.status)):
        lines.append(f'findoc_documents{{status="{st}"}} {n}')
    stage_ms: dict[str, list[float]] = defaultdict(list)
    recent = db.scalars(
        select(Document.result)
        .where(Document.result.is_not(None))
        .order_by(Document.finished_at.desc())
        .limit(500)
    ).all()
    for r in recent:
        for st in (r or {}).get("stages", []):
            stage_ms[st["stage"]].append(st["ms"])
    lines += [
        "# HELP findoc_stage_seconds Pipeline stage latency (last 500 documents)",
        "# TYPE findoc_stage_seconds summary",
    ]
    for stage, ms in sorted(stage_ms.items()):
        lines.append(f'findoc_stage_seconds_sum{{stage="{stage}"}} {sum(ms) / 1000:.3f}')
        lines.append(f'findoc_stage_seconds_count{{stage="{stage}"}} {len(ms)}')
    return "\n".join(lines) + "\n"
