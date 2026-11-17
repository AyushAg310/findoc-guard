"""Document tracking through a video with KLT optical flow (CO-4, FR-TRK-01/04).

**Optical flow** estimates how each pixel moves between two frames. The **Lucas-Kanade**
method assumes brightness constancy and small motion, so around a point the image changes
linearly with displacement d = (u, v):

    I_x u + I_y v + I_t = 0      (one equation per pixel)

Over a small window (21 x 21 px) that gives an over-determined linear system solved by least
squares; it is well conditioned only where the window has gradients in two directions, i.e. at
**corners**, which is why we track **Shi-Tomasi corners** (points whose structure tensor has
two large eigenvalues). The **pyramidal** version solves coarse-to-fine on downscaled copies,
so motions larger than the window are still recovered.

Robustness measures:

* **Forward-backward check**: track each point forward (t-1 to t) and back again (t to t-1); if
  it doesn't return within 1 px, the track is unreliable (occlusion, blur, drift) and dropped.
* **Reference-anchored homography**: every surviving point also remembers where it was in the
  *reference* frame (where it was detected). The document is flat, so reference and current
  positions are related by one homography, estimated with RANSAC; the document's corners in the
  reference frame are mapped through it. Drift therefore doesn't accumulate frame by frame,
  only at re-detections.
* **Re-detection with drift correction**: when fewer than ``MIN_TRACKS`` points survive, new
  corners are detected inside the current document outline and that frame becomes the new
  reference. Each such reset would freeze in whatever small error the estimate has (drift), so
  at that moment the Phase 1 page detector is also run: if it finds a quad clear of the image
  border that agrees with the track (within ``SNAP_MAX`` of the diagonal), the reference snaps
  to the detector's corners; if the detector can't confirm (page edge hidden or out of view),
  the track is checked against the stored anchor appearance instead (see below). Tracking gives
  continuity, detection and matching give accuracy.
* **Loss and re-localisation**: if too few points survive to fit a homography, the document is
  lost for that frame. Each later frame is matched against an **anchor** (the first frame where
  the document was found, with SIFT features inside the document) using the Phase 2 recipe:
  ratio test + RANSAC homography (CO-3). Matching against stored appearance relocates the whole
  document even when only part of it is back in view. The Phase 1 page detector is the last
  resort, and only trusted when the quad it finds is clear of the image border (a document
  cut off by the border would give a wrong, smaller quad).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from findoc.preprocess.detect import find_document
from findoc.preprocess.quality import sharpness

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
FEATURE_PARAMS = dict(maxCorners=600, qualityLevel=0.01, minDistance=7, blockSize=7)
FB_MAX_ERROR = 1.0  # px, forward-backward consistency
RANSAC_PX = 3.0
MIN_TRACKS = 150
MIN_FOR_HOMOGRAPHY = 20
RELOC_MIN_INLIERS = 25
RATIO = 0.75
BORDER_PX = 4
SNAP_MAX = 0.01  # max disagreement (fraction of the frame diagonal) to accept a detector snap


@dataclass
class FrameTrack:
    index: int
    corners: np.ndarray | None  # (4, 2) document corners in this frame, None if lost
    n_tracks: int  # surviving tracked points
    motion_px: float  # median point displacement from the previous frame
    sharpness: float  # Laplacian variance inside the document region
    coverage: float  # document area / frame area
    in_frame: bool  # all four corners inside the image
    redetected: bool = False
    reacquired: bool = False
    snapped: bool = False


@dataclass
class TrackResult:
    frames: list[FrameTrack] = field(default_factory=list)
    redetections: int = 0
    reacquisitions: int = 0
    snaps: int = 0

    @property
    def lost(self) -> int:
        return sum(f.corners is None for f in self.frames)


def quad_mask(shape: tuple[int, int], corners: np.ndarray, shrink: float = 0.03) -> np.ndarray:
    """Binary mask of the document quadrilateral, shrunk slightly away from its border."""
    centre = corners.mean(axis=0)
    inner = centre + (corners - centre) * (1 - shrink)
    mask = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(mask, inner.astype(np.int32), 255)
    return mask


def region_sharpness(gray: np.ndarray, corners: np.ndarray | None, width: int = 600) -> float:
    """Sharpness (Laplacian variance) of the document, measured on the **rectified** document.

    Warping the quad to a rectangle (at most ``width`` px wide, never upsampled) means every
    frame shows the same content at nearly the same scale, so the score changes with blur, not
    with viewpoint or how much textured desk is in view.
    """
    if corners is None:
        h, w = gray.shape[:2]
        s = width / max(h, w)
        return sharpness(
            cv2.resize(gray, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        )
    tl, tr, br, bl = corners.astype(np.float32)
    top_bottom = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    aspect = top_bottom / max((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2, 1e-6)
    width = int(min(width, max(64, top_bottom)))  # never upsample: that would hide blur
    height = max(8, round(width / max(aspect, 1e-3)))
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    flat = cv2.warpPerspective(
        gray, cv2.getPerspectiveTransform(corners.astype(np.float32), dst), (width, height)
    )
    m = int(0.03 * width)  # skip a thin border so the page edge / desk don't count
    return sharpness(flat[m:-m, m:-m])


def _coverage_and_inside(corners: np.ndarray, shape: tuple[int, int]) -> tuple[float, bool]:
    h, w = shape
    area = cv2.contourArea(corners.astype(np.float32))
    inside = bool(np.all((corners >= 0) & (corners <= [w - 1, h - 1])))
    return float(area / (w * h)), inside


def detect_features(gray: np.ndarray, corners: np.ndarray) -> np.ndarray:
    pts = cv2.goodFeaturesToTrack(gray, mask=quad_mask(gray.shape, corners), **FEATURE_PARAMS)
    return np.empty((0, 1, 2), np.float32) if pts is None else pts.astype(np.float32)


class Anchor:
    """Stored appearance of the document for re-localisation after the track is lost."""

    def __init__(self, gray: np.ndarray, corners: np.ndarray) -> None:
        self.sift = cv2.SIFT_create(nfeatures=1500)
        self.corners = corners.astype(np.float32)
        self.kp, self.des = self.sift.detectAndCompute(gray, quad_mask(gray.shape, corners, 0.0))

    def locate(self, gray: np.ndarray) -> np.ndarray | None:
        if self.des is None or len(self.kp) < RELOC_MIN_INLIERS:
            return None
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(kp) < RELOC_MIN_INLIERS:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(self.des, des, k=2)
        good = [m for m, *rest in pairs if rest and m.distance < RATIO * rest[0].distance]
        if len(good) < RELOC_MIN_INLIERS:
            return None
        src = np.float32([self.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)  # noqa: N806
        if H is None or int(inl.sum()) < RELOC_MIN_INLIERS:
            return None
        corners = cv2.perspectiveTransform(self.corners.reshape(-1, 1, 2), H).reshape(4, 2)
        return corners if cv2.isContourConvex(corners.astype(np.float32)) else None


def _clear_of_border(corners: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(np.all((corners > BORDER_PX) & (corners < [w - 1 - BORDER_PX, h - 1 - BORDER_PX])))


def snap_to_detection(frame: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    """Detector corners if they agree with ``corners`` and the page is clear of the border."""
    det = find_document(frame)
    if det.method == "full_frame" or not _clear_of_border(det.corners, frame.shape[:2]):
        return None
    diag = float(np.hypot(*frame.shape[:2]))
    if np.linalg.norm(det.corners - corners, axis=1).mean() > SNAP_MAX * diag:
        return None
    return det.corners.astype(np.float32)


def _initial_corners(
    gray_frames: list[np.ndarray], frames: list[np.ndarray]
) -> tuple[int, np.ndarray]:
    """First frame (among the first few) where the page detector finds the document."""
    for i in range(min(15, len(frames))):
        det = find_document(frames[i])
        if det.method != "full_frame":
            return i, det.corners
    h, w = gray_frames[0].shape
    return 0, np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)


def track_document(
    frames: list[np.ndarray], start_corners: np.ndarray | None = None
) -> TrackResult:
    """Track the document through ``frames`` (BGR). Returns per-frame corners and metrics."""
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    if start_corners is None:
        start, ref_corners = _initial_corners(grays, frames)
    else:
        start, ref_corners = 0, start_corners.astype(np.float32)
    result = TrackResult()
    shape = grays[0].shape

    # Frames before the document was first found are recorded as lost.
    for i in range(start):
        result.frames.append(
            FrameTrack(i, None, 0, 0.0, region_sharpness(grays[i], None), 0.0, False)
        )

    anchor = Anchor(grays[start], ref_corners)
    p_ref = detect_features(grays[start], ref_corners)
    p_prev = p_ref.copy()
    cov, inside = _coverage_and_inside(ref_corners, shape)
    result.frames.append(
        FrameTrack(
            start,
            ref_corners.copy(),
            len(p_ref),
            0.0,
            region_sharpness(grays[start], ref_corners),
            cov,
            inside,
        )
    )
    corners = ref_corners

    for t in range(start + 1, len(frames)):
        prev, cur = grays[t - 1], grays[t]
        n_good, motion = 0, 0.0
        if len(p_prev) >= MIN_FOR_HOMOGRAPHY:
            p_cur, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p_prev, None, **LK_PARAMS)
            p_back, st_back, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p_cur, None, **LK_PARAMS)
            fb_err = np.linalg.norm((p_back - p_prev).reshape(-1, 2), axis=1)
            good = (st.ravel() == 1) & (st_back.ravel() == 1) & (fb_err < FB_MAX_ERROR)
            p_ref, p_prev_g, p_cur = p_ref[good], p_prev[good], p_cur[good]
            n_good = int(good.sum())
            if n_good:
                motion = float(np.median(np.linalg.norm((p_cur - p_prev_g).reshape(-1, 2), axis=1)))
        else:
            p_cur = np.empty((0, 1, 2), np.float32)

        new_corners = None
        if n_good >= MIN_FOR_HOMOGRAPHY:
            H, inl = cv2.findHomography(p_ref, p_cur, cv2.RANSAC, RANSAC_PX)  # noqa: N806
            if H is not None:
                keep = inl.ravel().astype(bool)
                p_ref, p_cur = p_ref[keep], p_cur[keep]
                n_good = int(keep.sum())
                new_corners = cv2.perspectiveTransform(ref_corners.reshape(-1, 1, 2), H).reshape(
                    4, 2
                )
                if not cv2.isContourConvex(new_corners.astype(np.float32)):
                    new_corners = None

        frame = FrameTrack(t, None, n_good, motion, 0.0, 0.0, False)
        if new_corners is None:  # lost: re-localise from the anchor, else the page detector
            new_corners = anchor.locate(cur)
            if new_corners is None:
                det = find_document(frames[t])
                if det.method != "full_frame" and _clear_of_border(det.corners, shape):
                    new_corners = det.corners
            if new_corners is not None:
                frame.reacquired = True
                result.reacquisitions += 1
        if new_corners is not None:
            corners = new_corners
            frame.corners = corners.copy()
            frame.coverage, frame.in_frame = _coverage_and_inside(corners, shape)
            if n_good < MIN_TRACKS or frame.reacquired:  # refresh the point set
                # Correct drift before freezing a new reference: prefer the page detector
                # when it agrees, otherwise re-localise against the anchor's appearance.
                snapped = snap_to_detection(frames[t], corners)
                if snapped is None and not frame.reacquired:
                    snapped = anchor.locate(cur)
                if snapped is not None:
                    corners = snapped
                    frame.corners = corners.copy()
                    frame.coverage, frame.in_frame = _coverage_and_inside(corners, shape)
                    frame.snapped = True
                    result.snaps += 1
                ref_corners = corners.astype(np.float32)
                p_ref = detect_features(cur, ref_corners)
                p_cur = p_ref.copy()
                frame.redetected = not frame.reacquired
                result.redetections += int(frame.redetected)
        frame.sharpness = region_sharpness(cur, frame.corners)
        result.frames.append(frame)
        p_prev = p_cur
    return result


def draw_track(
    frame: np.ndarray, track: FrameTrack, colour: tuple[int, int, int] = (0, 220, 0)
) -> np.ndarray:
    """Overlay the tracked document outline and frame metrics."""
    out = frame.copy()
    if track.corners is not None:
        cv2.polylines(out, [track.corners.astype(np.int32).reshape(-1, 1, 2)], True, colour, 3)
    label = f"#{track.index} sharp {track.sharpness:.0f} tracks {track.n_tracks}"
    if track.corners is None:
        label += " LOST"
    cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(out, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out
