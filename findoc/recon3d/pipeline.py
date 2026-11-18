"""Reconstruct a document in 3D from video keyframes and flatten it (CO-5).

``flatten_document`` ties the pieces together:

    keyframes -> SfM (sfm.py) -> document points -> smooth surface -> mesh seen from the
    reference view -> ARAP flattening -> unwarped flat page + correspondence map

It reports why it did not produce a 3D result when that happens (too few views, failed
initialisation, too few document points, or a page that is flat anyway), so the caller can keep
the 2D homography result of Phase 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from findoc.recon3d.flatten import (
    arap_flatten,
    cast_rays,
    fit_surface,
    grid_in_quad,
    unwarp,
)
from findoc.recon3d.sfm import RATIO, Reconstruction, View, extract, reconstruct

MIN_DOC_POINTS = 40
MAX_STRETCH = 0.06  # ARAP stretch above which the 3D page is not trusted
MAX_VIEW_DEG = 70.0  # reference camera vs. page normal: steeper means the reconstruction collapsed
PLANE_P90_PX = 1.3  # at 1280 px: 90th-percentile homography residual of a flat page is ~1 px


def plane_residual(views: list[View], ref: int, quad: np.ndarray) -> float:
    """How far the document is from a plane, measured in 2D before any 3D reconstruction.

    For each other view: match the document's features, keep the epipolar-consistent matches
    (fundamental matrix + RANSAC), fit the best homography (RANSAC) and take the 90th percentile
    of the residuals. A flat page is one plane, so one homography explains (almost) every match;
    on a curved or folded page the points off the dominant plane are left over. The median over
    views is returned. (The median residual does not work: on a fold the RANSAC homography fits
    the larger flat half and that half's points dominate the median.)
    """
    v0 = views[ref]
    ins = _inside(quad, v0.kp, grow=0.0)
    kp0, d0 = v0.kp[ins], v0.des[ins]
    out = []
    if len(kp0) < 30:
        return float("nan")
    for k, v in enumerate(views):
        if k == ref or len(v.des) < 30:
            continue
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d0, v.des, k=2)
        good = [m for m, *r in pairs if r and m.distance < RATIO * r[0].distance]
        if len(good) < 30:
            continue
        pa = kp0[[m.queryIdx for m in good]]
        pb = v.kp[[m.trainIdx for m in good]]
        _, fm = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 1.0, 0.999)
        if fm is None or fm.sum() < 20:
            continue
        sel = fm.ravel() > 0
        pa, pb = pa[sel], pb[sel]
        H, _ = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
        if H is None:
            continue
        r = np.linalg.norm(
            cv2.perspectiveTransform(pa.reshape(-1, 1, 2), H).reshape(-1, 2) - pb, axis=1
        )
        out.append(float(np.percentile(r, 90)))
    return float(np.median(out)) if out else float("nan")


GRID = (56, 28)


@dataclass
class Flattening:
    ok: bool
    reason: str
    stats: dict[str, float] = field(default_factory=dict)
    flat: np.ndarray | None = None
    map_x: np.ndarray | None = None  # flat pixel -> reference-image x (correspondence map)
    map_y: np.ndarray | None = None
    planar: bool = False
    rec: Reconstruction | None = None
    doc_ids: np.ndarray | None = None  # track ids of document points
    mesh_xyz: np.ndarray | None = None
    mesh_uv: np.ndarray | None = None
    mesh_tris: np.ndarray | None = None


def _inside(quad: np.ndarray, pts: np.ndarray, grow: float = 0.02) -> np.ndarray:
    c = quad.mean(axis=0)
    q = (c + (quad - c) * (1 + grow)).astype(np.float32)
    return np.array([cv2.pointPolygonTest(q, (float(x), float(y)), False) >= 0 for x, y in pts])


def _near(quad: np.ndarray, pts: np.ndarray, frac: float) -> np.ndarray:
    """Points within ``frac`` x the quad's diagonal of its outline (the desk around the page)."""
    c = quad.mean(axis=0)
    diag = float(np.linalg.norm(quad[2] - quad[0]))
    return np.linalg.norm(pts - c, axis=1) < (0.5 + frac) * diag


def flatten_document(
    images: list[np.ndarray],
    ref_index: int,
    ref_quad: np.ndarray,
    focal_guess: float | None = None,
    force_3d: bool = False,
) -> Flattening:
    """``images``: keyframes (BGR, same size); ``ref_quad``: page corners (tl, tr, br, bl) in
    ``images[ref_index]``, the view the flat page is sampled from."""
    t0 = time.perf_counter()
    if len(images) < 3:
        return Flattening(False, "fewer than 3 keyframes")
    views = extract(images)
    h, w = images[0].shape[:2]
    p90 = plane_residual(views, ref_index, ref_quad)
    threshold = PLANE_P90_PX * max(1.0, max(w, h) / 1280)
    stats: dict[str, float] = {
        "plane_residual_px": round(p90, 3),
        "plane_threshold_px": round(threshold, 2),
    }
    if np.isfinite(p90) and p90 < threshold and not force_3d:
        stats["total_s"] = round(time.perf_counter() - t0, 2)
        return Flattening(
            False, "page is flat: one homography explains it (2D is exact)", stats, planar=True
        )
    rec = reconstruct(images, focal_guess=focal_guess, views=views)
    stats["sfm_s"] = round(time.perf_counter() - t0, 2)
    if rec is None:
        return Flattening(False, "structure-from-motion could not initialise", stats)
    ref = rec.views[ref_index]
    stats |= {
        "views": len(images),
        "registered": len(rec.registered()),
        "points": len(rec.points),
        "reproj_px": round(rec.reproj_px, 3),
        "focal_px": round(rec.focal, 1),
    }
    if not ref.registered:
        return Flattening(False, "reference view not registered", stats, rec=rec)
    ids, pts = rec.point_array()
    in_front = (pts @ ref.R.T + ref.t)[:, 2] > 0
    proj = rec.project(ref, pts)
    doc = in_front & _inside(ref_quad, proj)
    stats["doc_points"] = int(doc.sum())
    if doc.sum() < MIN_DOC_POINTS:
        return Flattening(False, f"only {int(doc.sum())} 3D points on the document", stats, rec=rec)
    desk = in_front & ~_inside(ref_quad, proj, grow=0.08) & _near(ref_quad, proj, 0.6)
    stats["desk_points"] = int(desk.sum())
    fit = fit_surface(pts[doc], pts[desk])
    stats["planarity"] = round(fit.planarity, 5)
    stats["surface_inliers"] = fit.n_inliers
    planar = False
    # sanity: the reference camera must look at the page from above, not along it. A collapsed
    # reconstruction (too little camera motion) can put the camera almost in the page's plane.
    c_loc = fit.to_local((-ref.R.T @ ref.t)[None])[0]
    view_deg = float(
        np.degrees(np.arccos(min(1.0, abs(c_loc[2]) / (np.linalg.norm(c_loc) + 1e-12))))
    )
    stats["view_angle_deg"] = round(view_deg, 1)
    if view_deg > MAX_VIEW_DEG and not force_3d:
        return Flattening(
            False,
            f"degenerate reconstruction (camera {view_deg:.0f} deg off the page normal)",
            stats,
            rec=rec,
        )

    img_pts, tris = grid_in_quad(ref_quad.astype(np.float64), *GRID)
    xyz = cast_rays(fit, rec.K, ref.R, ref.t, img_pts)  # on the reconstructed (domed) surface
    local = fit.to_local(xyz)
    # flatten the page's shape relative to the desk: same (a, b), height = relief above the desk
    shape = np.column_stack([local[:, :2], fit.relief(local[:, :2])])
    uv, stretch = arap_flatten(shape, tris, local[:, :2])
    stats["stretch"] = round(stretch, 5)
    if not stretch < MAX_STRETCH and not force_3d:
        # paper does not stretch: a surface that cannot be unrolled without it was reconstructed
        # wrongly (e.g. a bad focal length), and flattening it would distort the page
        return Flattening(
            False, f"3D surface not developable (stretch {stretch:.1%}): kept 2D", stats, rec=rec
        )
    e_img = np.linalg.norm(img_pts[tris[:, 0]] - img_pts[tris[:, 1]], axis=1).mean()
    e_uv = np.linalg.norm(uv[tris[:, 0]] - uv[tris[:, 1]], axis=1).mean()
    nx, ny = GRID
    corners = (0, nx - 1, nx * ny - 1, nx * (ny - 1))
    flat, mx, my = unwarp(images[ref_index], img_pts, uv, tris, corners, scale=1.1 * e_img / e_uv)
    stats["total_s"] = round(time.perf_counter() - t0, 2)
    return Flattening(
        True, "flattened from 3D", stats, flat, mx, my, planar, rec, ids[doc], xyz, uv, tris
    )
