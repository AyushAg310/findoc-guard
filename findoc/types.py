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
    value: str  # text as read (cleaned)
    raw: str = ""  # OCR output before cleaning
    normalized: str | None = None  # validated canonical value, None if invalid/unreadable
    method: str | None = None  # how it was read, e.g. "compact" for re-spaced date boxes
    confidence: float = 0.0
    bbox: BBox | None = None  # in template coordinates for cheques


class OcrWord(BaseModel):
    """One word or line recognised by OCR, in flat-page pixel coordinates."""

    text: str
    confidence: float
    bbox: BBox


class CheckResult(BaseModel):
    id: str
    fired: bool
    severity: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)


class TrackedFrame(BaseModel):
    """Per-frame tracking result (video input). Corners are in original-video pixels."""

    frame: int  # frame number in the source video
    corners: list[list[float]] | None  # None = document lost in this frame
    tracks: int
    motion_px: float
    sharpness: float
    in_frame: bool


class Keyframe(BaseModel):
    frame: int
    sharpness: float
    view_change: float  # projective shape change from the previous keyframe
    corners: list[list[float]]


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
    enhanced: np.ndarray | None = Field(default=None, exclude=True)

    # Geometry (Phase 1): page corners in the original image (tl, tr, br, bl) and the
    # 3x3 homography that maps original pixels to flat-page pixels.
    corners: list[list[float]] | None = None
    homography: list[list[float]] | None = None
    detection_method: str | None = None

    # Template matching (Phase 2): which cheque template, and flat-page -> template homography
    template_id: str | None = None
    template_homography: list[list[float]] | None = None
    registered: np.ndarray | None = Field(default=None, exclude=True)

    # Video (Phase 3): tracked frames (downscaled, for tracking), the per-frame track, the
    # keyframes chosen for 3D reconstruction (full-resolution images kept for Phase 4) and the
    # best frame, which becomes ``original`` for the 2D stages.
    frames: list[np.ndarray] | None = Field(default=None, exclude=True)
    frame_numbers: list[int] = Field(default_factory=list)
    video: dict[str, float] = Field(default_factory=dict)
    track: list[TrackedFrame] = Field(default_factory=list)
    tracking: dict[str, float] = Field(default_factory=dict)
    keyframes: list[Keyframe] = Field(default_factory=list)
    keyframe_images: list[np.ndarray] = Field(default_factory=list, exclude=True)
    best_frame: int | None = None

    quality: dict[str, float] = Field(default_factory=dict)
    words: list[OcrWord] = Field(default_factory=list)
    full_text: str = ""
    fields: list[ExtractedField] = Field(default_factory=list)
    checks: list[CheckResult] = Field(default_factory=list)
    risk_score: int | None = None
    stages: list[StageLog] = Field(default_factory=list)
    messages: list[str] = Field(default_factory=list)

    def result(self) -> dict[str, Any]:
        """JSON-serialisable result (images excluded)."""
        return self.model_dump(mode="json")
