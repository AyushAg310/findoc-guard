"""A minimal pinhole camera, used to render synthetic videos with exact ground truth.

World frame: the desk is the plane Z = 0, with the document centred on the origin and its
pixels used as world units (1 unit = 1 document pixel). The camera looks down at the desk from
negative Z.

Projection of a 3D point X (pinhole model):

    x ~ K (R X + t)          K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]

For points on the plane Z = 0, X = (x, y, 0), this collapses to a homography:

    x ~ K [r1 r2 t] (x, y, 1)^T        (r1, r2 = first two columns of R)

which is exactly why a flat page seen by a moving camera is related frame-to-frame by
homographies (Phase 1 and Phase 3), and why a *curved* page is not (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    f: float  # focal length in pixels
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:  # noqa: N802 - standard notation
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]], float)

    @classmethod
    def for_frame(cls, width: int, height: int, hfov_deg: float = 60.0) -> Intrinsics:
        """Intrinsics for a frame size and horizontal field of view (phones: ~60-70 deg)."""
        f = (width / 2) / np.tan(np.deg2rad(hfov_deg) / 2)
        return cls(f, width / 2, height / 2)


@dataclass(frozen=True)
class Pose:
    """World-to-camera transform: X_cam = R X_world + t."""

    R: np.ndarray  # noqa: N815
    t: np.ndarray

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates: C = -R^T t."""
        return -self.R.T @ self.t


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray, roll_deg: float = 0.0) -> Pose:
    """Camera at ``eye`` looking at ``target``. Camera axes: x right, y down, z forward."""
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(z, up)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])  # noqa: N806 - rows are camera axes in world coordinates
    if roll_deg:
        a = np.deg2rad(roll_deg)
        roll = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        R = roll @ R  # noqa: N806
    return Pose(R, -R @ eye)


def plane_homography(K: np.ndarray, pose: Pose) -> np.ndarray:  # noqa: N803
    """Homography mapping plane points (x, y, 1) with Z = 0 to image pixels."""
    H = K @ np.column_stack([pose.R[:, 0], pose.R[:, 1], pose.t])  # noqa: N806
    return H / H[2, 2]


def project(K: np.ndarray, pose: Pose, points: np.ndarray) -> np.ndarray:  # noqa: N803
    """Project (N, 3) world points to (N, 2) pixels."""
    cam = points @ pose.R.T + pose.t
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]
