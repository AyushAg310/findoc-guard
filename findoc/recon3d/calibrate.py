"""Camera calibration from a printed checkerboard (Zhang's method, CO-5).

3D reconstruction from a phone video is much more accurate when the camera's **focal length** is
known: estimating it from the video itself is ill-conditioned for a nearly flat scene (see the
Phase 5 guide). A phone's focal length (in pixels, at a given video resolution) does not change,
so it only needs measuring once.

**Zhang's method.** A checkerboard is a plane with known geometry: its inner corners lie on a
regular grid. Each photo of it gives a homography from the board to the image, H = K [r1 r2 t]
(the same plane-induced homography used throughout). Because r1 and r2 are orthonormal, every
homography gives two equations on K; with several views at different angles, K follows, then
the poses, and finally everything (plus lens distortion) is refined by minimising the
reprojection error (``cv2.calibrateCamera``).

Use: print a checkerboard (``--board`` = inner corners, e.g. 9x6 for a 10x7-square board), film
it for ~10 s from different angles at the same resolution you use for documents, then
``findoc calibrate board.mp4 --save``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Calibration:
    K: np.ndarray
    dist: np.ndarray
    rms_px: float
    size: tuple[int, int]  # (width, height) of the images
    views: int

    @property
    def focal_px(self) -> float:
        return float((self.K[0, 0] + self.K[1, 1]) / 2)

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.size[0] / (2 * self.K[0, 0]))))


def board_points(board: tuple[int, int], square: float = 1.0) -> np.ndarray:
    cols, rows = board
    g = np.zeros((rows * cols, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    return g


def find_corners(gray: np.ndarray, board: tuple[int, int]) -> np.ndarray | None:
    ok, corners = cv2.findChessboardCorners(
        gray, board, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)


def calibrate(
    images: list[np.ndarray], board: tuple[int, int] = (9, 6), max_views: int = 30
) -> Calibration:
    """Calibrate from images (BGR or gray) showing the checkerboard. Raises ValueError if fewer
    than 4 views show the whole board."""
    obj, img = [], []
    size = None
    for im in images:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im
        size = (gray.shape[1], gray.shape[0])
        c = find_corners(gray, board)
        if c is not None:
            obj.append(board_points(board))
            img.append(c)
    if len(obj) < 4 or size is None:
        raise ValueError(
            f"checkerboard found in {len(obj)} images; need at least 4 (board {board})"
        )
    if len(obj) > max_views:  # evenly spaced subset keeps calibration fast
        pick = np.linspace(0, len(obj) - 1, max_views).round().astype(int)
        obj, img = [obj[i] for i in pick], [img[i] for i in pick]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return Calibration(K, dist.ravel(), float(rms), size, len(obj))


def load_images(source: Path, step: int = 10) -> list[np.ndarray]:
    """Every ``step``-th frame of a video, or every image in a folder."""
    from findoc.io import IMAGE_EXTS, load_image, read_frames, video_info

    if source.is_dir():
        return [load_image(p) for p in sorted(source.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    n = int(video_info(source)["frames"])
    frames = read_frames(source, list(range(0, max(n, 1), step)))
    return [frames[k] for k in sorted(frames)]


def save_focal(focal_px: float, env: Path = Path(".env")) -> Path:
    """Write ``FINDOC_FOCAL_PX`` into a .env file (replacing an existing value)."""
    lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    lines = [ln for ln in lines if not ln.startswith("FINDOC_FOCAL_PX=")]
    lines.append(f"FINDOC_FOCAL_PX={focal_px:.1f}")
    env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env
