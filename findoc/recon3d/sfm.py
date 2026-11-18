"""Incremental Structure-from-Motion on video keyframes (CO-5).

Given a few photos of a static scene from different viewpoints, SfM recovers **where each camera
was** (rotation R, translation t) and **the 3D positions of points** seen in several photos. It
uses only image correspondences and the pinhole model (:mod:`findoc.synth.camera`):

    x ~ K (R X + t)        K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]

Steps (the classic incremental pipeline, as in COLMAP or Bundler):

1. **Features** (``extract``): SIFT keypoints and descriptors in every keyframe.
2. **Matching** (``match_pair``): nearest neighbours with Lowe's ratio test, then geometric
   verification with the **essential matrix** (5-point algorithm + RANSAC). A cheque on a desk
   is an almost *planar* scene; the 8-point fundamental-matrix method is degenerate for planes,
   while the calibrated 5-point method is not. So we assume a focal length (a typical phone:
   ``FOCAL_GUESS`` x the longer image side) and let bundle adjustment refine it.
3. **Tracks** (``build_tracks``): pairwise matches are chained with union-find into tracks, one
   per physical point (a track that hits two keypoints in the same image is inconsistent and
   dropped).
4. **Initialisation**: the pair with many matches *and* enough parallax; ``recoverPose`` gives
   the relative pose (translation known only up to scale), and the shared tracks are
   **triangulated**.
5. **Registration**: repeatedly add the image that sees the most already-reconstructed points,
   finding its pose from 2D-3D correspondences with **PnP + RANSAC**; triangulate newly
   visible tracks.
6. **Bundle adjustment** (``bundle_adjust``): jointly refine all camera poses, all 3D points and
   the shared focal length to minimise the total reprojection error (robust Huber loss, sparse
   Jacobian, SciPy ``least_squares``). Observations that still reproject badly are removed.

The result is defined only up to a similarity transform (position, orientation and **scale**
of the whole scene are arbitrary): nothing in images alone fixes the size of the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

FOCAL_GUESS = 0.8  # x longer image side: a typical phone main camera (~65-70 deg wide)
RATIO = 0.75
EPIPOLAR_PX = 1.5
MIN_PAIR_INLIERS = 60
MIN_ANGLE_DEG = 1.0  # minimum triangulation angle for a new point
MAX_REPROJ_PX = 4.0


@dataclass
class View:
    index: int
    gray: np.ndarray
    kp: np.ndarray  # (N, 2) keypoint positions in full-resolution pixels
    des: np.ndarray
    colors: np.ndarray  # (N, 3) BGR at each keypoint
    R: np.ndarray | None = None
    t: np.ndarray | None = None

    @property
    def registered(self) -> bool:
        return self.R is not None


@dataclass
class Reconstruction:
    K: np.ndarray
    views: list[View]
    tracks: list[dict[int, int]]  # track -> {view index: keypoint index}
    points: dict[int, np.ndarray] = field(default_factory=dict)  # track -> 3D point
    reproj_px: float = float("nan")
    pairs: dict[tuple[int, int], int] = field(default_factory=dict)  # verified matches per pair
    focal_search: dict[float, float] = field(default_factory=dict)  # focal -> reprojection error

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    def registered(self) -> list[View]:
        return [v for v in self.views if v.registered]

    def point_array(self) -> tuple[np.ndarray, np.ndarray]:
        """(track ids, (M, 3) points)."""
        ids = np.array(sorted(self.points), int)
        pts = np.array([self.points[i] for i in ids]) if len(ids) else np.zeros((0, 3))
        return ids, pts

    def colors(self, ids: np.ndarray) -> np.ndarray:
        out = []
        for i in ids:
            v, k = next(iter(self.tracks[i].items()))
            out.append(self.views[v].colors[k])
        return np.array(out) if out else np.zeros((0, 3))

    def project(self, view: View, X: np.ndarray) -> np.ndarray:
        Xc = X @ view.R.T + view.t
        x = Xc @ self.K.T
        return x[:, :2] / x[:, 2:3]


# ---------------------------------------------------------------------------- features, matches


def extract(images: list[np.ndarray], max_side: int = 1600, n_features: int = 4000) -> list[View]:
    sift = cv2.SIFT_create(nfeatures=n_features)
    views = []
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        s = min(1.0, max_side / max(gray.shape))
        small = (
            gray if s == 1.0 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        )
        kps, des = sift.detectAndCompute(small, None)
        pts = np.array([k.pt for k in kps], np.float32).reshape(-1, 2) / s
        if des is None:
            des = np.zeros((0, 128), np.float32)
        xi = np.clip(pts[:, 0].round().astype(int), 0, gray.shape[1] - 1)
        yi = np.clip(pts[:, 1].round().astype(int), 0, gray.shape[0] - 1)
        cols = img[yi, xi] if img.ndim == 3 else np.repeat(gray[yi, xi][:, None], 3, 1)
        views.append(View(i, gray, pts, des, cols.astype(np.uint8)))
    return views


def match_pair(a: View, b: View, K: np.ndarray) -> np.ndarray:
    """Verified matches (M, 2) of keypoint indices a -> b."""
    if len(a.des) < 8 or len(b.des) < 8:
        return np.zeros((0, 2), int)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(a.des, b.des, k=2)
    good = [m for m, *r in pairs if r and m.distance < RATIO * r[0].distance]
    if len(good) < MIN_PAIR_INLIERS:
        return np.zeros((0, 2), int)
    idx = np.array([[m.queryIdx, m.trainIdx] for m in good], int)
    pa, pb = a.kp[idx[:, 0]], b.kp[idx[:, 1]]
    try:
        E, mask = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, EPIPOLAR_PX)
    except cv2.error:  # degenerate sample (OpenCV's polynomial solver can fail)
        return np.zeros((0, 2), int)
    if E is None or mask is None:
        return np.zeros((0, 2), int)
    return idx[mask.ravel().astype(bool)]


def build_tracks(n_views: int, matches: dict[tuple[int, int], np.ndarray]) -> list[dict[int, int]]:
    """Union-find over (view, keypoint) nodes; keep consistent tracks seen in >= 2 views."""
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (i, j), m in matches.items():
        for ka, kb in m:
            ra, rb = find((i, int(ka))), find((j, int(kb)))
            if ra != rb:
                parent[ra] = rb
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    tracks = []
    for nodes in groups.values():
        views = [v for v, _ in nodes]
        if len(nodes) >= 2 and len(set(views)) == len(views):
            tracks.append({v: k for v, k in nodes})
    return tracks


# ---------------------------------------------------------------------------- geometry helpers


def triangulate(Ps: list[np.ndarray], xs: list[np.ndarray]) -> np.ndarray:
    """Linear (DLT) triangulation of one point from several views."""
    rows = []
    for P, x in zip(Ps, xs, strict=True):
        rows.append(x[0] * P[2] - P[0])
        rows.append(x[1] * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.array(rows))
    X = vt[-1]
    return X[:3] / X[3]


def _centre(v: View) -> np.ndarray:
    return -v.R.T @ v.t


def _angle_deg(X: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> float:
    a, b = c1 - X, c2 - X
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _try_triangulate(rec: Reconstruction, tid: int) -> None:
    obs = [(v, k) for v, k in rec.tracks[tid].items() if rec.views[v].registered]
    if len(obs) < 2:
        return
    K = rec.K
    Ps = [K @ np.hstack([rec.views[v].R, rec.views[v].t[:, None]]) for v, _ in obs]
    xs = [rec.views[v].kp[k] for v, k in obs]
    X = triangulate(Ps, xs)
    centres = [_centre(rec.views[v]) for v, _ in obs]
    if (
        max(_angle_deg(X, a, b) for i, a in enumerate(centres) for b in centres[i + 1 :])
        < MIN_ANGLE_DEG
    ):
        return
    for (v, _k), x in zip(obs, xs, strict=True):
        view = rec.views[v]
        if (view.R @ X + view.t)[2] <= 0:  # behind the camera
            return
        if np.linalg.norm(rec.project(view, X[None])[0] - x) > MAX_REPROJ_PX:
            return
    rec.points[tid] = X


# ---------------------------------------------------------------------------- bundle adjustment


def bundle_adjust(rec: Reconstruction, refine_focal: bool = True, iters: int = 60) -> float:
    """Refine registered poses, points and the shared focal length. Returns RMS reprojection
    error (pixels). The first registered camera is held fixed (it defines the coordinate frame)."""
    cams = rec.registered()
    if len(cams) < 2 or not rec.points:
        return float("nan")
    cam_pos = {v.index: i for i, v in enumerate(cams)}
    tids = sorted(rec.points)
    pt_pos = {t: i for i, t in enumerate(tids)}
    obs_c, obs_p, obs_x = [], [], []
    for t in tids:
        for v, k in rec.tracks[t].items():
            if v in cam_pos:
                obs_c.append(cam_pos[v])
                obs_p.append(pt_pos[t])
                obs_x.append(rec.views[v].kp[k])
    obs_c, obs_p, obs_x = np.array(obs_c), np.array(obs_p), np.array(obs_x, float)
    nc, npt = len(cams), len(tids)
    cx, cy = rec.K[0, 2], rec.K[1, 2]
    f0 = rec.K[0, 0]

    x0 = [np.log(f0)] if refine_focal else []
    for v in cams[1:]:
        x0 += list(cv2.Rodrigues(v.R)[0].ravel()) + list(v.t)
    x0 = np.concatenate([np.array(x0), np.array([rec.points[t] for t in tids]).ravel()])
    nf = 1 if refine_focal else 0
    r0, t0 = cams[0].R, cams[0].t

    def unpack(x: np.ndarray):  # noqa: ANN202
        f = np.exp(x[0]) if refine_focal else f0
        Rs, ts = [r0], [t0]
        for i in range(nc - 1):
            p = x[nf + 6 * i : nf + 6 * i + 6]
            Rs.append(cv2.Rodrigues(p[:3])[0])
            ts.append(p[3:])
        pts = x[nf + 6 * (nc - 1) :].reshape(npt, 3)
        return f, np.array(Rs), np.array(ts), pts

    def residuals(x: np.ndarray) -> np.ndarray:
        f, Rs, ts, pts = unpack(x)
        Xc = np.einsum("nij,nj->ni", Rs[obs_c], pts[obs_p]) + ts[obs_c]
        z = np.where(np.abs(Xc[:, 2]) < 1e-9, 1e-9, Xc[:, 2])
        proj = np.column_stack([f * Xc[:, 0] / z + cx, f * Xc[:, 1] / z + cy])
        return (proj - obs_x).ravel()

    m = len(obs_c) * 2
    sparsity = lil_matrix((m, len(x0)), dtype=np.uint8)
    rows = np.arange(len(obs_c))
    if refine_focal:
        sparsity[:, 0] = 1
    for s in range(6):
        sel = obs_c > 0
        cols = nf + 6 * (obs_c[sel] - 1) + s
        sparsity[2 * rows[sel], cols] = 1
        sparsity[2 * rows[sel] + 1, cols] = 1
    for s in range(3):
        cols = nf + 6 * (nc - 1) + 3 * obs_p + s
        sparsity[2 * rows, cols] = 1
        sparsity[2 * rows + 1, cols] = 1
    res = least_squares(
        residuals,
        x0,
        jac_sparsity=sparsity,
        x_scale="jac",
        loss="huber",
        f_scale=1.5,
        method="trf",
        max_nfev=iters,
    )
    f, Rs, ts, pts = unpack(res.x)
    rec.K[0, 0] = rec.K[1, 1] = f
    for v, R, t in zip(cams, Rs, ts, strict=True):
        v.R, v.t = R, t
    for t, p in zip(tids, pts, strict=True):
        rec.points[t] = p
    r = residuals(res.x).reshape(-1, 2)
    err = np.linalg.norm(r, axis=1)
    # drop observations that still reproject badly; drop points left with < 2 observations
    for (c, p), e in zip(zip(obs_c, obs_p, strict=True), err, strict=True):
        if e > MAX_REPROJ_PX:
            rec.tracks[tids[p]].pop(cams[c].index, None)
    for t in tids:
        if sum(v in cam_pos for v in rec.tracks[t]) < 2:
            rec.points.pop(t, None)
    rec.reproj_px = (
        float(np.sqrt(np.mean(err[err <= MAX_REPROJ_PX] ** 2)))
        if (err <= MAX_REPROJ_PX).any()
        else float("nan")
    )
    return rec.reproj_px


# ---------------------------------------------------------------------------- the pipeline


def _parallax_score(rec: Reconstruction, i: int, j: int, m: np.ndarray) -> float:
    """Matches x median image displacement after removing the best homography: how much true
    parallax (not just camera rotation or a plane's projective change) the pair has."""
    pa, pb = rec.views[i].kp[m[:, 0]], rec.views[j].kp[m[:, 1]]
    H, _ = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is None:
        return 0.0
    res = np.linalg.norm(
        cv2.perspectiveTransform(pa.reshape(-1, 1, 2), H).reshape(-1, 2) - pb, axis=1
    )
    return float(len(m) * min(np.median(res), 20.0))


FOCAL_SCAN = (0.6, 0.75, 0.9, 1.05, 1.2)  # x longer image side (phones: ~0.7-1.0)
FOCAL_REFINE = (0.06,)  # then try best +- these fractions of the longer side
MAX_INIT_CANDIDATES = 2


def _init_candidates(
    rec: Reconstruction, i: int, j: int, m: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Possible relative poses (R, t) of view j w.r.t. view i.

    The essential matrix gives one pose (``recoverPose`` picks the solution that puts most
    points in front of both cameras). But a document on a desk is a nearly **planar** scene, and
    for a plane that choice is ambiguous: two different motions explain the matches equally
    well, so the essential matrix alone can pick the wrong one (synthetic flat page: 16% error).
    We therefore also decompose the **homography** between the two views
    (``decomposeHomographyMat``: up to 4 solutions) and keep every solution that puts most
    points in front of both cameras. The caller tries each candidate and keeps the one that best
    explains *all* the views (the ambiguity exists for two views, not for many).
    """
    pa, pb = rec.views[i].kp[m[:, 0]], rec.views[j].kp[m[:, 1]]
    K = rec.K
    cands: list[tuple[np.ndarray, np.ndarray, float]] = []
    try:
        E, _ = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, EPIPOLAR_PX)
        if E is not None and E.shape == (3, 3):
            _, R, t, _ = cv2.recoverPose(E, pa, pb, K)
            cands.append((R, t.ravel() / (np.linalg.norm(t) + 1e-12), 2.0))  # essential first
    except cv2.error:
        pass
    H, _ = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    if H is not None:
        _, Rs, ts, _ = cv2.decomposeHomographyMat(H, K)
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        for R, t in zip(Rs, ts, strict=True):
            nt = np.linalg.norm(t)
            if nt < 1e-9:
                continue
            t = t.ravel() / nt
            P2 = K @ np.hstack([R, t[:, None]])
            X = cv2.triangulatePoints(P1, P2, pa.T.astype(np.float64), pb.T.astype(np.float64))
            X = (X[:3] / X[3]).T
            front = (X[:, 2] > 0) & ((X @ R.T + t)[:, 2] > 0)
            if front.mean() > 0.8:
                cands.append((R, t, float(front.mean())))
    # drop near-duplicates (the essential solution often equals one homography solution)
    unique: list[tuple[np.ndarray, np.ndarray]] = []
    for R, t, _ in sorted(cands, key=lambda c: -c[2]):
        if all(np.linalg.norm(R - R2) > 1e-3 or np.linalg.norm(t - t2) > 1e-3 for R2, t2 in unique):
            unique.append((R, t))
    return unique[:MAX_INIT_CANDIDATES]


def _grazing(rec: Reconstruction, max_deg: float = 70.0) -> bool:
    """True if the cameras look along the scene's dominant plane instead of at it.

    A document capture always looks down at a page. A wrong initial pose for a near-planar scene
    can still fit the images (low reprojection error) by collapsing the points into a plane that
    contains the cameras; this rejects such reconstructions.
    """
    _, pts = rec.point_array()
    if len(pts) < 10:
        return True
    c = pts.mean(axis=0)
    normal = np.linalg.svd(pts - c, full_matrices=False)[2][2]
    angles = []
    for v in rec.registered():
        d = -v.R.T @ v.t - c
        angles.append(
            np.degrees(np.arccos(min(1.0, abs(d @ normal) / (np.linalg.norm(d) + 1e-12))))
        )
    return float(np.median(angles)) > max_deg


def _incremental(
    views: list[View], matches: dict[tuple[int, int], np.ndarray], K: np.ndarray, min_views: int
) -> Reconstruction | None:
    """Incremental SfM with a fixed focal length (the search over focal lengths calls this).
    Tries every initial-pose candidate and keeps the reconstruction that registers the most
    views with the lowest reprojection error."""
    probe = Reconstruction(K.copy(), views, build_tracks(len(views), matches))
    i, j = max(matches, key=lambda p: _parallax_score(probe, p[0], p[1], matches[p]))
    best: Reconstruction | None = None
    best_poses: list[tuple[np.ndarray | None, np.ndarray | None]] = []
    for R, t in _init_candidates(probe, i, j, matches[(i, j)]):
        rec = _grow(views, matches, K, i, j, R, t, min_views)
        if rec is None or _grazing(rec):
            continue
        key = (len(rec.registered()), -rec.reproj_px)
        if best is None or key > (len(best.registered()), -best.reproj_px):
            best = rec
            best_poses = [(v.R, v.t) for v in views]
    if best is not None:  # candidates share the View objects: restore the winner's poses
        for v, (R, t) in zip(views, best_poses, strict=True):
            v.R, v.t = R, t
    return best


def _grow(
    views: list[View],
    matches: dict[tuple[int, int], np.ndarray],
    K: np.ndarray,
    i: int,
    j: int,
    R: np.ndarray,
    t: np.ndarray,
    min_views: int,
) -> Reconstruction | None:
    """Triangulate the initial pair, register the other views by PnP, bundle-adjust."""
    for v in views:
        v.R, v.t = None, None
    rec = Reconstruction(K.copy(), views, build_tracks(len(views), matches))
    rec.pairs = {k: len(v) for k, v in matches.items()}
    views[i].R, views[i].t = np.eye(3), np.zeros(3)
    views[j].R, views[j].t = R, np.asarray(t, float).ravel()
    for tid in range(len(rec.tracks)):
        _try_triangulate(rec, tid)
    if len(rec.points) < 20:
        return None
    bundle_adjust(rec, refine_focal=False)

    # --- register the remaining views by PnP
    while True:
        best, best_corr = None, []
        for v in views:
            if v.registered:
                continue
            corr = [
                (rec.points[t], v.kp[tr[v.index]])
                for t, tr in enumerate(rec.tracks)
                if t in rec.points and v.index in tr
            ]
            if len(corr) > len(best_corr):
                best, best_corr = v, corr
        if best is None or len(best_corr) < 12:
            break
        obj = np.array([c[0] for c in best_corr], np.float64)
        img = np.array([c[1] for c in best_corr], np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, rec.K, None, iterationsCount=500, reprojectionError=MAX_REPROJ_PX
        )
        if not ok or inl is None or len(inl) < 12:
            break
        inl = inl.ravel()
        cv2.solvePnPRefineLM(obj[inl], img[inl], rec.K, None, rvec, tvec)
        best.R, best.t = cv2.Rodrigues(rvec)[0], tvec.ravel()
        for tid, tr in enumerate(rec.tracks):
            if tid not in rec.points and best.index in tr:
                _try_triangulate(rec, tid)
        if len(rec.registered()) % 3 == 0:
            bundle_adjust(rec, refine_focal=False)
    if len(rec.registered()) < min_views:
        return None
    for tid in range(len(rec.tracks)):  # points that became triangulable late
        if tid not in rec.points:
            _try_triangulate(rec, tid)
    bundle_adjust(rec, refine_focal=False, iters=100)
    return rec


def reconstruct(
    images: list[np.ndarray],
    focal_guess: float | None = None,
    min_views: int = 3,
    views: list[View] | None = None,
) -> Reconstruction | None:
    """Run incremental SfM on ``images`` (BGR, same size). None if it cannot initialise.

    **Focal length search.** A phone video has no reliable calibration, and a cheque on a desk is
    a nearly planar scene, where bundle adjustment started from a wrong focal length settles in a
    local minimum: the reconstruction bows into a dome and the reprojection error stays high
    (synthetic flat page: guess -> f = 1025 px, error 0.56 px, planarity 1.6%; true f = 1065 px,
    error 0.33 px, planarity 0.24%). The error is clearly lowest near the true focal length, so we
    scan focal lengths from 0.6 to 1.2 x the longer image side in 15% steps (each run with a
    fixed focal length and the same RANSAC seed, so runs are comparable), refine around the best
    with +-6% steps, keep the lowest error, and only then let bundle adjustment
    refine the focal length. Features and matches are computed once and reused. Synthetic fold:
    error 0.51 px at 1088 px (true 1065) against ~1 px elsewhere.
    """
    h, w = images[0].shape[:2]
    long = max(w, h)
    f0 = focal_guess or FOCAL_GUESS * long
    K0 = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]], float)
    views = views if views is not None else extract(images)
    matches = {}
    for i in range(len(views)):
        for j in range(i + 1, len(views)):
            m = match_pair(views[i], views[j], K0)
            if len(m) >= MIN_PAIR_INLIERS:
                matches[(i, j)] = m
    if not matches:
        return None
    search: dict[float, float] = {}

    def run(f: float) -> Reconstruction | None:
        cv2.setRNGSeed(0)  # deterministic RANSAC: runs at different focal lengths are comparable
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], float)
        rec = _incremental(views, matches, K, min_views)
        search[round(f, 1)] = round(rec.reproj_px, 4) if rec is not None else float("inf")
        return rec

    if focal_guess:
        best_f = focal_guess
        best = run(best_f)
    else:
        best, best_f = None, None
        for c in FOCAL_SCAN:
            rec = run(c * long)
            if rec is not None and (best is None or rec.reproj_px < best.reproj_px):
                best, best_f = rec, c * long
        for step in FOCAL_REFINE:
            if best_f is None:
                break
            for f in (best_f - step * long, best_f + step * long):
                rec = run(f)
                if rec is not None and rec.reproj_px < best.reproj_px:
                    best, best_f = rec, f
    if best is None:
        return None
    # runs share the View objects (later runs overwrite poses), so redo the best one, same seed
    scan = dict(search)
    rec = run(best_f)
    search.clear()
    search.update(scan)
    if rec is None:
        return None
    refine = focal_guess is None  # a calibrated focal length is kept fixed
    bundle_adjust(rec, refine_focal=refine, iters=100)
    bundle_adjust(rec, refine_focal=refine, iters=50)
    rec.focal_search = search
    return rec
