"""Tables (SRS 11.1): users and their API keys, documents (which double as the job queue),
reviewer field corrections and decisions, exports, and an append-only audit log."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Role(str, Enum):
    SUBMITTER = "submitter"
    REVIEWER = "reviewer"
    ADMIN = "admin"


class DocStatus(str, Enum):
    """Document life cycle (SRS 9.3).

    queued -> processing -> needs_review | processed | rejected_quality | failed
    needs_review / processed -> approved | rejected | escalated (reviewer)
    escalated -> approved | rejected (admin)
    A failed attempt goes back to queued until ``max_attempts``.
    """

    QUEUED = "queued"
    PROCESSING = "processing"
    NEEDS_REVIEW = "needs_review"  # medium or high risk
    PROCESSED = "processed"  # low risk; still open to review
    REJECTED_QUALITY = "rejected_quality"  # retake needed (blur, no document found)
    FAILED = "failed"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


OPEN_FOR_REVIEW = {DocStatus.NEEDS_REVIEW, DocStatus.PROCESSED}
DECIDED = {DocStatus.APPROVED, DocStatus.REJECTED}


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    role: Mapped[str] = mapped_column(String(16), default=Role.SUBMITTER.value)
    password_hash: Mapped[str] = mapped_column(String(255))
    active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    api_keys: Mapped[list[ApiKey]] = relationship(back_populates="user")


class ApiKey(Base):
    """Only a SHA-256 of the key is stored; the key itself is shown once, at creation."""

    __tablename__ = "api_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    prefix: Mapped[str] = mapped_column(String(16), index=True)  # to find it, and to show
    key_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(default=False)

    user: Mapped[User] = relationship(back_populates="api_keys")


class Document(Base):
    """One upload, its processing state, and the pipeline's result."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    owner_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(64))  # sniffed, not the client's claim
    kind: Mapped[str] = mapped_column(String(8))  # image | video
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    options: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)  # register, account

    status: Mapped[str] = mapped_column(String(20), default=DocStatus.QUEUED.value, index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    stage: Mapped[str | None] = mapped_column(String(32))  # current pipeline stage
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0..1
    error: Mapped[str | None] = mapped_column(Text)

    doc_type: Mapped[str | None] = mapped_column(String(16), index=True)
    risk_score: Mapped[int | None] = mapped_column(Integer)
    risk_band: Mapped[str | None] = mapped_column(String(8), index=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    corrections: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)  # field -> value
    images: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)  # variant -> file name
    media_deleted: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    owner: Mapped[User] = relationship()
    decisions: Mapped[list[Decision]] = relationship(
        back_populates="document", order_by="Decision.created_at"
    )
    field_edits: Mapped[list[FieldEdit]] = relationship(
        back_populates="document", order_by="FieldEdit.created_at"
    )


class FieldEdit(Base):
    __tablename__ = "field_edits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    field: Mapped[str] = mapped_column(String(64))
    old_value: Mapped[str | None] = mapped_column(Text)
    new_value: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    document: Mapped[Document] = relationship(back_populates="field_edits")
    user: Mapped[User] = relationship()


class Decision(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    decision: Mapped[str] = mapped_column(String(16))  # approved | rejected | escalated
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)

    document: Mapped[Document] = relationship(back_populates="decisions")
    user: Mapped[User] = relationship()


class Export(Base):
    __tablename__ = "exports"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"))
    format: Mapped[str] = mapped_column(String(8))  # json | xlsx
    filters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    count: Mapped[int] = mapped_column(Integer, default=0)
    filename: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class AuditLog(Base):
    """Append-only: who did what to which object, when (FR-APP-02)."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    actor_id: Mapped[str | None] = mapped_column(String(32), index=True)  # None = system
    actor_email: Mapped[str | None] = mapped_column(String(254))
    action: Mapped[str] = mapped_column(String(48), index=True)
    target_type: Mapped[str] = mapped_column(String(24))
    target_id: Mapped[str | None] = mapped_column(String(32), index=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
