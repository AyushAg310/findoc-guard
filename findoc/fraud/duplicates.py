"""Duplicate presentation: has this cheque been seen before? (Phase 6, SRS 10.5 DUPLICATE)

The classic fraud is presenting the *same* cheque twice: at two banks, by photo after the paper
was already deposited, or as a fresh photo of a copy. Two photos of one cheque differ in angle,
light and blur, so comparing whole images fails; comparing two *different* cheques of the same
bank is misleading too, because the printed template (logo, lines, pattern) is identical.

What is unique to one cheque is its **handwriting**. Phase 2 already registers the cheque to
its bank template and subtracts the template (``ink_mask``), leaving only the ink: payee,
amount, date and signature as written on this leaf. For each indexed cheque the index keeps the
**SIFT keypoints and descriptors of that ink image**. A new cheque is compared with every indexed
cheque of the same bank by matching descriptors and fitting one homography with RANSAC; inlier
count and inlier ratio decide. On synthetic cheques a second photo of the same leaf gave
232-305 inliers, two different leaves at most 90 (``experiments/phase6_fraud_eval.py``).

A perceptual hash of the ink image was tried as a fast pre-filter and dropped: every cheque has
its handwriting in the same boxes, so the hashes of different cheques were as close as those
of two photos of one cheque (Hamming distance 4-10 vs. 2-8 of 64 bits), and a filter that cannot
separate them could only hide true duplicates. Comparing against every same-bank entry costs a
few milliseconds each; a large deployment would add an approximate nearest-neighbour index
over the descriptors.

If OCR read the MICR line (cheque number + bank code + account), an identical MICR is also
reported, since it means the same leaf by definition.

The index is a folder (``FINDOC_DUPLICATE_INDEX``, default ``data/index``): ``index.json``
(one entry per cheque) and one ``.npz`` per cheque with its keypoints and descriptors. Cheques
are added explicitly (``findoc process ... --register``), never by default: a document should be
recorded only once it has been accepted. The folder holds data derived from real cheques when
used with them, so it is git-ignored and must stay private.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

MIN_INLIERS = 150  # same cheque: >= 232 in the experiment; different cheques: <= 90
MIN_INLIER_RATIO = 0.2
MAX_KEYPOINTS = 600


def ink_image(registered: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Dark ink on white, template removed (template coordinates)."""
    from findoc.recognize.fields import ink_mask

    m = ink_mask(registered, reference)
    return np.where(m > 0, 0, 255).astype(np.uint8)


def ink_features(ink: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Up to MAX_KEYPOINTS SIFT keypoints (N, 2) and descriptors (N, 128) on the ink."""
    sift = cv2.SIFT_create(MAX_KEYPOINTS)
    kp, des = sift.detectAndCompute(ink, None)
    if des is None:
        return np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32)
    return np.float32([k.pt for k in kp]), des.astype(np.float32)


def verify(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> tuple[int, float]:
    """(RANSAC inliers, inlier ratio) between two ink images' features."""
    (pa, da), (pb, db) = a, b
    if len(da) < 8 or len(db) < 8:
        return 0, 0.0
    good = [
        m
        for m, *rest in cv2.BFMatcher().knnMatch(da, db, k=2)
        if rest and m.distance < 0.8 * rest[0].distance
    ]
    if len(good) < 8:
        return 0, 0.0
    src = pa[[m.queryIdx for m in good]]
    dst = pb[[m.trainIdx for m in good]]
    h, inl = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    if h is None or inl is None:
        return 0, 0.0
    n = int(inl.sum())
    return n, n / min(len(da), len(db))


@dataclass
class Entry:
    id: str
    source: str
    template_id: str
    micr: str | None
    added: float


@dataclass
class Match:
    entry: Entry
    inliers: int
    inlier_ratio: float
    same_micr: bool

    @property
    def duplicate(self) -> bool:
        return self.same_micr or (
            self.inliers >= MIN_INLIERS and self.inlier_ratio >= MIN_INLIER_RATIO
        )


class DuplicateIndex:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: list[Entry] = []
        f = root / "index.json"
        if f.is_file():
            self.entries = [Entry(**e) for e in json.loads(f.read_text(encoding="utf-8"))]

    def __len__(self) -> int:
        return len(self.entries)

    def _features(self, entry_id: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(self.root / f"{entry_id}.npz") as z:
            return z["points"], z["descriptors"]

    def add(self, source: str, template_id: str, ink: np.ndarray, micr: str | None = None) -> Entry:
        self.root.mkdir(parents=True, exist_ok=True)
        entry = Entry(
            id=f"{len(self.entries) + 1:06d}",
            source=source,
            template_id=template_id,
            micr=micr,
            added=round(time.time(), 1),
        )
        pts, des = ink_features(ink)
        np.savez_compressed(self.root / f"{entry.id}.npz", points=pts, descriptors=des)
        self.entries.append(entry)
        (self.root / "index.json").write_text(
            json.dumps([asdict(e) for e in self.entries], indent=1), encoding="utf-8"
        )
        return entry

    def search(self, template_id: str, ink: np.ndarray, micr: str | None = None) -> list[Match]:
        """Every indexed cheque of the same bank, verified by feature matching; best first."""
        feats = ink_features(ink)
        out = []
        for e in self.entries:
            if e.template_id != template_id:
                continue
            n, r = verify(feats, self._features(e.id))
            out.append(Match(e, n, r, bool(micr and e.micr == micr)))
        return sorted(out, key=lambda m: (-m.duplicate, -m.inliers))
