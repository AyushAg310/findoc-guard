"""Request and response bodies (Pydantic): these define the OpenAPI contract."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models import Role


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: str
    name: str
    role: str
    active: bool
    created_at: datetime


class UserIn(BaseModel):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^@\s]+@[^@\s]+$")
    name: str = ""
    role: Role = Role.SUBMITTER
    password: str = Field(min_length=8, max_length=128)


class UserPatch(BaseModel):
    name: str | None = None
    role: Role | None = None
    active: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=128)


class ApiKeyIn(BaseModel):
    name: str = Field(default="", max_length=120)


class ApiKeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked: bool
    key: str | None = None  # only in the response that creates it


class Submitted(BaseModel):
    id: str
    status: str
    duplicate_of: str | None = None  # same file uploaded before (by sha256)


class DocumentSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    filename: str
    kind: str
    status: str
    stage: str | None
    progress: float
    attempts: int
    doc_type: str | None
    risk_score: int | None
    risk_band: str | None
    owner_email: str
    created_at: datetime
    finished_at: datetime | None


class FieldView(BaseModel):
    name: str
    value: str | None  # as read (masked for account numbers)
    normalized: str | None
    corrected: str | None  # the reviewer's value, if edited
    confidence: float
    bbox: list[float] | None  # x0, y0, x1, y1 in the "page" image
    method: str | None = None


class CheckView(BaseModel):
    id: str
    fired: bool
    severity: float
    confidence: float
    reason: str
    contribution: float | None = None
    evidence: dict[str, Any] = {}


class DecisionOut(BaseModel):
    decision: str
    comment: str
    user_email: str
    created_at: datetime


class FieldEditOut(BaseModel):
    field: str
    old_value: str | None
    new_value: str
    user_email: str
    created_at: datetime


class DocumentDetail(DocumentSummary):
    media_type: str
    size_bytes: int
    error: str | None
    images: list[str]
    page_frame: Literal["template", "flat"]
    fields: list[FieldView]
    line_items: list[dict[str, Any]]
    checks: list[CheckView]
    risk: dict[str, Any]
    quality: dict[str, Any]
    messages: list[str]
    stages: list[dict[str, Any]]
    flattening: str | None
    decisions: list[DecisionOut]
    field_edits: list[FieldEditOut]
    allowed_decisions: list[str]


class DocumentList(BaseModel):
    items: list[DocumentSummary]
    total: int


class FieldsPatch(BaseModel):
    fields: dict[str, str] = Field(description="field name -> corrected value", min_length=1)


class DecisionIn(BaseModel):
    decision: Literal["approved", "rejected", "escalated"]
    comment: str = Field(default="", max_length=2000)


class BulkDecisionIn(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=200)
    decision: Literal["approved", "rejected", "escalated"]
    comment: str = Field(default="", max_length=2000)


class BulkDecisionOut(BaseModel):
    done: list[str]
    skipped: dict[str, str]  # id -> why


class ExportIn(BaseModel):
    format: Literal["json", "xlsx"] = "xlsx"
    status: list[str] | None = None
    band: list[str] | None = None
    doc_type: str | None = None


class ExportOut(BaseModel):
    id: str
    format: str
    count: int
    download: str


class Stats(BaseModel):
    total: int
    by_status: dict[str, int]
    by_band: dict[str, int]
    by_type: dict[str, int]
    average_risk: float | None
    open_reviews: int
    daily: list[dict[str, Any]] = []  # last 14 days: date, received, decided, flagged
    by_check: dict[str, int] = {}  # how often each check fired (last 1000 processed)
    median_seconds: float | None = None  # processing time, last 1000 processed
    risk_histogram: list[int] = []  # documents per 10-point risk bucket (0-9, ..., 90-100)


class AuditOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    at: datetime
    actor_email: str | None
    action: str
    target_type: str
    target_id: str | None
    detail: dict[str, Any]
