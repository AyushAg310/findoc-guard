"""From a sparse 3D reconstruction to a flat document image (CO-5).

1. **Document points**: SfM reconstructs the desk too. We keep the 3D points whose projection in
   the reference view falls inside the document outline (tracked corners or a segmentation
   mask).
2. **Surface fit** (``fit_surface``): a plane through the document points (PCA) gives a local
   frame (a, b along the page, height h above it). Height is modelled as a smooth function
   h(a, b), a **thin-plate spline** with smoothing: the surface with the least bending energy
   that passes close to the points. Refitted three times, each time dropping points more than 3
   robust standard deviations off the previous fit (mismatched features, desk points that
   project inside the outline).
   ``planarity`` = RMS height / page diagonal: below ``PLANAR_RMS`` the page is flat and the 2D
   homography of Phase 1 is already exact.
3. **Mesh by ray casting** (``grid_in_quad`` + ``cast_rays``): a grid of pixels over the
   document in the reference image; each pixel's viewing ray is intersected with the fitted
   surface, giving one 3D vertex per pixel, so every vertex has a known image position.
4. **ARAP flattening** (``arap_flatten``): find 2D positions (u, v) for the vertices so every
   triangle keeps its 3D shape as well as possible: "as rigid as possible" (Liu et al., 2008).
   Alternate a *local* step (for each triangle, the best rotation R_t between its flat isometric
   shape and its current 2D shape, by SVD) and a *global* step (solve one sparse linear system,
   the cotangent Laplacian, for all vertices given the rotations), until the vertices stop
   moving. Paper is developable, so the result is nearly distortion-free; ``stretch`` reports
   how close.
5. **Unwarp** (``unwarp``): rasterise the flat mesh at output resolution; for every output pixel,
   barycentric interpolation of its triangle's *image* positions gives where to sample the
   photo; one ``cv2.remap`` builds the flat page. The same two arrays (``map_x``, ``map_y``) are
   the **correspondence map** flat page -> original photo, which Phase 6 uses to show tamper
   evidence on the original photo.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.interpolate import RBFInterpolator
from scipy.sparse.linalg import factorized

PLANAR_RMS = 0.003  # RMS height / page diagonal below which the page counts as flat
MIN_DESK_POINTS = 60


class _HeightField:
    """Robust thin-plate-spline height field h(a, b) on normalised coordinates."""

    def __init__(self, ab: np.ndarray, h: np.ndarray, scale: float, smoothing: float) -> None:
        self.scale = scale
        keep = np.ones(len(h), bool)
        for _ in range(3):
            self._fit(ab[keep], h[keep], smoothing)
            res = h - self(ab)
            mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-12
            keep = np.abs(res) < max(3 * mad, 1e-4 * scale)
        self._fit(ab[keep], h[keep], smoothing)
        self.keep = keep

    def _fit(self, ab: np.ndarray, h: np.ndarray, smoothing: float) -> None:
        self.rbf = RBFInterpolator(
            ab / self.scale,
            h / self.scale,
            kernel="thin_plate_spline",
            smoothing=smoothing * len(h),
        )

    def __call__(self, ab: np.ndarray) -> np.ndarray:
        return self.rbf(ab / self.scale) * self.scale


@dataclass
class SurfaceFit:
    origin: np.ndarray  # plane origin (3,)
    axes: np.ndarray  # (3, 3) rows: a, b, normal
    doc: _HeightField  # document height above the plane
    ground: _HeightField | None  # desk height above the plane (reconstruction "doming")
    planarity: float  # RMS height of the page *above the desk* / page diagonal
    n_points: int
    n_inliers: int
    n_desk: int

    def to_local(self, X: np.ndarray) -> np.ndarray:
        return (X - self.origin) @ self.axes.T

    def height(self, ab: np.ndarray) -> np.ndarray:
        """Height of the reconstructed page above the plane (includes any doming)."""
        return self.doc(ab)

    def relief(self, ab: np.ndarray) -> np.ndarray:
        """Height of the page above the *desk*: the real bend, with doming removed."""
        return self.doc(ab) - (self.ground(ab) if self.ground is not None else 0.0)


def fit_surface(
    points: np.ndarray, desk: np.ndarray | None = None, smoothing: float = 1e-3
) -> SurfaceFit:
    """Robust smooth height field over (N, 3) document points.

    ``desk``: 3D points on the desk around the page. SfM of a nearly planar scene with an
    uncertain focal length bows the whole reconstruction into a dome; the desk is known to be
    flat, so a smooth fit to the desk points measures that distortion, and the page's real bend
    is its height *relative to the desk* (``relief``).
    """
    origin = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - origin, full_matrices=False)
    axes = vt.copy()
    local = (points - origin) @ axes.T
    ab, h = local[:, :2], local[:, 2]
    diag = float(np.linalg.norm(ab.max(0) - ab.min(0)))
    doc = _HeightField(ab, h, diag, smoothing)
    ground = None
    n_desk = 0
    if desk is not None and len(desk) >= MIN_DESK_POINTS:
        dl = (desk - origin) @ axes.T
        ground = _HeightField(dl[:, :2], dl[:, 2], diag, smoothing * 10)  # smoother: a trend
        n_desk = int(ground.keep.sum())
    fit = SurfaceFit(origin, axes, doc, ground, 0.0, len(h), int(doc.keep.sum()), n_desk)
    rel = fit.relief(ab[doc.keep])
    if ground is None:
        rel = rel - np.median(rel)
    fit.planarity = float(np.sqrt(np.mean(rel**2)) / diag)
    return fit


def cast_rays(
    fit: SurfaceFit, K: np.ndarray, R: np.ndarray, t: np.ndarray, pix: np.ndarray, iters: int = 30
) -> np.ndarray:
    """Intersect the viewing rays of image pixels (N, 2) with the fitted surface -> (N, 3).

    Along each ray X(lam) = C + lam d, find where g(lam) = height_above_plane(X) - h(a, b) = 0.
    The crossing is bracketed by sampling lam around the ray's intersection with the plane, then
    refined by bisection, which always converges. (A simple fixed-point iteration diverges where
    the surface is steep relative to the ray.) Rays with no crossing get NaN.
    """
    centre = -R.T @ t
    rays = np.column_stack([pix, np.ones(len(pix))]) @ np.linalg.inv(K).T @ R  # world directions
    c_loc = fit.to_local(centre[None])[0]
    d_loc = rays @ fit.axes.T

    def g(lam: np.ndarray) -> np.ndarray:
        p = c_loc + lam[:, None] * d_loc
        return p[:, 2] - fit.height(p[:, :2])

    lam0 = -c_loc[2] / d_loc[:, 2]  # on the plane h = 0
    fracs = np.linspace(0.5, 1.5, 41)
    vals = np.stack([g(lam0 * f) for f in fracs], axis=1)  # (N, S)
    sign = np.signbit(vals)
    change = sign[:, 1:] != sign[:, :-1]
    # crossing nearest the plane intersection (fraction 1.0, index 20)
    dist = np.abs(np.arange(len(fracs) - 1) + 0.5 - 20)
    k = np.argmin(np.where(change, dist, np.inf), axis=1)
    found = change[np.arange(len(pix)), k]
    lo, hi = lam0 * fracs[k], lam0 * fracs[k + 1]
    g_lo = vals[np.arange(len(pix)), k]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        g_mid = g(mid)
        left = np.signbit(g_mid) == np.signbit(g_lo)
        lo = np.where(left, mid, lo)
        g_lo = np.where(left, g_mid, g_lo)
        hi = np.where(left, hi, mid)
    lam = 0.5 * (lo + hi)
    p = c_loc + lam[:, None] * d_loc
    out = p @ fit.axes + fit.origin
    out[~found] = np.nan
    return out


def grid_in_quad(quad: np.ndarray, nx: int = 48, ny: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Image points of a bilinear nx x ny grid inside quad (tl, tr, br, bl) and triangles."""
    s, t = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    s, t = s.ravel()[:, None], t.ravel()[:, None]
    tl, tr, br, bl = quad
    pts = (1 - s) * (1 - t) * tl + s * (1 - t) * tr + s * t * br + (1 - s) * t * bl
    tris = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            tris += [(a, a + 1, a + nx), (a + 1, a + nx + 1, a + nx)]
    return pts.astype(np.float64), np.array(tris, np.int32)


# ------------------------------------------------------------------------------------- ARAP


def _local_frames(X: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Each triangle's vertices in its own 2D isometric frame: (T, 3, 2)."""
    p0, p1, p2 = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    e1 = p1 - p0
    l1 = np.linalg.norm(e1, axis=1)
    ex = e1 / l1[:, None]
    n = np.cross(e1, p2 - p0)
    ey = np.cross(n / np.linalg.norm(n, axis=1, keepdims=True), ex)
    out = np.zeros((len(tris), 3, 2))
    out[:, 1, 0] = l1
    out[:, 2, 0] = np.einsum("ij,ij->i", p2 - p0, ex)
    out[:, 2, 1] = np.einsum("ij,ij->i", p2 - p0, ey)
    return out


def _cotangents(x: np.ndarray) -> np.ndarray:
    """(T, 3) cotangent of the angle at each vertex of 2D triangles (T, 3, 2)."""
    cots = np.zeros((len(x), 3))
    for k in range(3):
        a, b, c = x[:, k], x[:, (k + 1) % 3], x[:, (k + 2) % 3]
        u, v = b - a, c - a
        cross = np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]) + 1e-12
        cots[:, k] = np.einsum("ij,ij->i", u, v) / cross
    return cots


def arap_flatten(
    X: np.ndarray, tris: np.ndarray, init_uv: np.ndarray, iters: int = 200, tol: float = 1e-5
) -> tuple[np.ndarray, float]:
    """ARAP parameterisation. Returns (uv (N, 2), stretch = RMS relative edge-length error).

    Local/global iterations run until the largest vertex move is below ``tol`` x the mesh size
    (at most ``iters``). Starting from the projection onto the page's best-fit plane, a strongly
    curled page needs ~100 iterations; stopping at 25 left curl2 pages visibly sheared.
    Degenerate input (non-finite vertices, e.g. rays that missed the surface) gives stretch inf.
    Zero-area triangles get zero weight.
    """
    n = len(X)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(init_uv)):
        return init_uv.astype(np.float64), float("inf")
    loc = _local_frames(X, tris)
    cot = _cotangents(loc)
    area2 = np.abs(loc[:, 1, 0] * loc[:, 2, 1] - loc[:, 1, 1] * loc[:, 2, 0])
    valid = np.isfinite(loc).all(axis=(1, 2)) & (area2 > 1e-9 * (np.median(area2) + 1e-12))
    loc[~valid] = 0.0
    cot[~valid] = 0.0
    # half-edge (i, j) opposite vertex k has weight cot at k
    rows, cols, vals = [], [], []
    edges = [(1, 2, 0), (2, 0, 1), (0, 1, 2)]
    for a, b, k in edges:
        i, j, w = tris[:, a], tris[:, b], np.clip(cot[:, k], 0, 1e3)  # clip keeps L well-posed
        rows += [i, j, i, j]
        cols += [j, i, i, j]
        vals += [-w, -w, w, w]
    L = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    L = L.tolil()
    L[0, :] = 0  # pin vertex 0 (removes the translation freedom)
    L[0, 0] = 1
    L = L + sp.identity(n, format="lil") * 1e-9  # keeps isolated vertices solvable
    try:
        solve = factorized(L.tocsc())
    except RuntimeError:  # singular system: a broken mesh
        return init_uv.astype(np.float64), float("inf")
    uv = init_uv.astype(np.float64).copy()
    size = float(np.linalg.norm(uv.max(0) - uv.min(0))) + 1e-12
    for _ in range(iters):
        # local step: best rotation per triangle (orthogonal Procrustes)
        cur = uv[tris]  # (T, 3, 2)
        S = np.zeros((len(tris), 2, 2))
        for a, b, k in edges:
            du = cur[:, a] - cur[:, b]
            dx = loc[:, a] - loc[:, b]
            S += np.clip(cot[:, k], 0, 1e3)[:, None, None] * np.einsum("ti,tj->tij", du, dx)
        U, _, Vt = np.linalg.svd(S)
        Rt = U @ Vt
        bad = np.linalg.det(Rt) < 0
        U[bad, :, 1] *= -1
        Rt = U @ Vt
        # global step: L uv = b
        rhs = np.zeros((n, 2))
        for a, b, k in edges:
            w = np.clip(cot[:, k], 0, 1e3)[:, None]
            d = np.einsum("tij,tj->ti", Rt, loc[:, a] - loc[:, b]) * w
            np.add.at(rhs, tris[:, a], d)
            np.add.at(rhs, tris[:, b], -d)
        rhs[0] = uv[0]
        new = np.column_stack([solve(rhs[:, 0]), solve(rhs[:, 1])])
        moved = float(np.abs(new - uv).max())
        uv = new
        if moved < tol * size:
            break
    e3 = np.concatenate(
        [np.linalg.norm(X[tris[:, a]] - X[tris[:, b]], axis=1) for a, b, _ in edges]
    )
    e2 = np.concatenate(
        [np.linalg.norm(uv[tris[:, a]] - uv[tris[:, b]], axis=1) for a, b, _ in edges]
    )
    stretch = float(np.sqrt(np.mean((e2 / np.maximum(e3, 1e-12) - 1) ** 2)))
    return uv, stretch


# ------------------------------------------------------------------------------------- unwarp


def unwarp(
    image: np.ndarray,
    img_pts: np.ndarray,
    uv: np.ndarray,
    tris: np.ndarray,
    quad_idx: tuple[int, int, int, int],
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterise the flat mesh and sample ``image``. Returns (flat image, map_x, map_y).

    ``quad_idx`` are the vertex indices of the page's tl, tr, br, bl corners: the output is
    rotated so tl -> tr is horizontal, mirrored if needed so it reads the right way round, and
    ``scale`` sets output pixels per uv unit.
    """
    tl, tr, br, bl = (uv[i] for i in quad_idx)
    ang = np.arctan2(tr[1] - tl[1], tr[0] - tl[0])
    rot = np.array([[np.cos(-ang), -np.sin(-ang)], [np.sin(-ang), np.cos(-ang)]])
    q = (uv - tl) @ rot.T
    if q[quad_idx[3], 1] < 0:  # bottom-left above top-left: mirrored parameterisation
        q[:, 1] *= -1
    q *= scale
    q -= q.min(axis=0)
    w, h = (np.ceil(q.max(axis=0)) + 1).astype(int)
    ids = np.full((h, w), -1, np.int32)
    for t in range(len(tris)):
        cv2.fillConvexPoly(ids, q[tris[t]].round().astype(np.int32), t)
    m = np.concatenate([q[tris], np.ones((len(tris), 3, 1))], axis=2)
    ok = np.abs(np.linalg.det(m)) > 1e-9
    aff = np.zeros((len(tris), 2, 3))
    aff[ok] = np.transpose(np.linalg.solve(m[ok], img_pts[tris[ok]]), (0, 2, 1))
    ys, xs = np.nonzero(ids >= 0)
    a = aff[ids[ys, xs]]
    map_x = np.full((h, w), -1, np.float32)
    map_y = np.full((h, w), -1, np.float32)
    map_x[ys, xs] = a[:, 0, 0] * xs + a[:, 0, 1] * ys + a[:, 0, 2]
    map_y[ys, xs] = a[:, 1, 0] * xs + a[:, 1, 1] * ys + a[:, 1, 2]
    flat = cv2.remap(
        image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return flat, map_x, map_y
