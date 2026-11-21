"""Upload safety (NFR-SEC-05): size limit, type from the file's own bytes, re-encoding.

* The client's filename and Content-Type are not trusted: the type comes from the first
  bytes ("magic numbers").
* Images are decoded and saved again as PNG. That drops EXIF (location, camera serial) and
  anything appended to the file, and a file that doesn't decode is refused. PNG is lossless,
  so the pixels, including the JPEG block traces the tamper model looks for, are kept
  exactly. OpenCV applies the EXIF orientation while decoding, so phone photos stay upright.
* Videos can't be re-encoded without losing quality, so they are only type-checked and
  stored with a fixed extension; OpenCV (FFmpeg) reads them inside the worker.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_TYPES = {
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/bmp": (b"BM",),
    "image/tiff": (b"II*\x00", b"MM\x00*"),
}


class UploadError(ValueError):
    pass


@dataclass
class Sniffed:
    media_type: str
    kind: str  # image | video
    ext: str  # extension the stored file gets


def sniff(head: bytes) -> Sniffed:
    for mt, magics in IMAGE_TYPES.items():
        if any(head.startswith(m) for m in magics):
            return Sniffed(mt, "image", ".png")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return Sniffed("image/webp", "image", ".png")
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand.startswith(b"qt"):
            return Sniffed("video/quicktime", "video", ".mov")
        return Sniffed("video/mp4", "video", ".mp4")
    if head[:4] == b"RIFF" and head[8:12] == b"AVI ":
        return Sniffed("video/x-msvideo", "video", ".avi")
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return Sniffed("video/webm", "video", ".webm")
    raise UploadError(
        "Unsupported file type: upload a JPEG, PNG, WebP, BMP or TIFF photo, or an "
        "MP4, MOV, AVI or WebM video"
    )


@dataclass
class Stored:
    path: Path
    media_type: str
    kind: str
    size_bytes: int
    sha256: str


def store_upload(data: bytes, dest_stem: Path, max_bytes: int) -> Stored:
    """Check and store an upload as ``dest_stem`` + extension. Raises UploadError."""
    if not data:
        raise UploadError("The file is empty")
    if len(data) > max_bytes:
        raise UploadError(f"The file is larger than {max_bytes // 2**20} MB")
    s = sniff(data[:16])
    path = dest_stem.with_suffix(s.ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    if s.kind == "image":
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise UploadError("The image could not be decoded")
        h, w = img.shape[:2]
        if min(h, w) < 200 or max(h, w) > 12000:
            raise UploadError(f"Image size {w}x{h} is outside 200..12000 pixels")
        if not cv2.imwrite(str(path), img):
            raise UploadError("The image could not be stored")
    else:
        path.write_bytes(data)
    return Stored(path, s.media_type, s.kind, len(data), hashlib.sha256(data).hexdigest())
