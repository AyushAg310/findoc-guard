"""Core data types shared across pipeline stages.

Every stage reads a :class:`Document` and adds its results to it, so stages can be
added phase by phase without changing earlier ones.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class InputKind(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class DocType(str, Enum):
    UNKNOWN = "unknown"
    CHEQUE = "cheque"
    INVOICE = "invoice"
    RECEIPT = "receipt"


class Status(str, Enum):
    OK = "ok"
    REJECTED_QUALITY = "rejected_quality"
    FAILED = "failed"


class BBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class ExtractedField(BaseModel):
    """A field extracted from a document (payee, amount, GSTIN, ...)."""

    name: str
    value: str
    confidence: float = 0.0
    bbox: BBox | None = None


class CheckResult(BaseModel):
    id: str
    fired: bool
    severity: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class StageLog(BaseModel):
    stage: str
    ms: float
    ok: bool = True
    note: str = ""


class Document(BaseModel):
    """Mutable state passed through the pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Path
    kind: InputKind
    doc_type: DocType = DocType.UNKNOWN
    status: Status = Status.OK

    # Images are kept out of the JSON result (exclude=True)
    original: np.ndarray | None = Field(default=None, exclude=True)
    flat: np.ndarray | None = Field(default=None, exclude=True)

    quality: dict[str, float] = Field(default_factory=dict)
    fields: list[ExtractedField] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    risk_score: int | None = None
    stages: list[StageLog] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)

    def result(self) -> dict[str, Any]:
        """JSON-serialisable result (images excluded)."""
        return self.model_dump(mode="json")
