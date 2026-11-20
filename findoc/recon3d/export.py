"""Save and visualise 3D results: PLY point clouds and meshes, a preview image.

PLY ("polygon file format") is a simple text format for points and meshes that MeshLab,
Blender, CloudCompare and online viewers open directly.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from findoc.recon3d.pipeline import Flattening


def write_ply_points(path: Path, xyz: np.ndarray, bgr: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(xyz)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue", "end_header",
    ]  # fmt: skip
    lines += [
        f"{x:.5f} {y:.5f} {z:.5f} {c[2]} {c[1]} {c[0]}"
        for (x, y, z), c in zip(xyz, bgr, strict=True)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def write_ply_mesh(path: Path, xyz: np.ndarray, tris: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(xyz)}",
        "property float x", "property float y", "property float z",
        f"element face {len(tris)}", "property list uchar int vertex_indices", "end_header",
    ]  # fmt: skip
    lines += [f"{x:.5f} {y:.5f} {z:.5f}" for x, y, z in xyz]
    lines += [f"3 {a} {b} {c}" for a, b, c in tris]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def preview(result: Flattening, size: tuple[int, int] = (1400, 520)) -> np.ndarray:
    """Left: oblique view of the reconstructed page (wireframe coloured by height above the
    page plane, plus the page's 3D points). Right: the flattened page."""
    w, h = size
    canvas = np.full((h, w, 3), 255, np.uint8)
    half = w // 2
    if result.mesh_xyz is not None and result.mesh_tris is not None:
        xyz = result.mesh_xyz
        c = xyz.mean(axis=0)
        _, _, vt = np.linalg.svd(xyz - c, full_matrices=False)
        loc = (xyz - c) @ vt.T  # a, b along the page, n = height
        if np.median(loc[:, 2]) > 0:
            loc[:, 2] *= -1
        tilt = np.deg2rad(55)
        y = loc[:, 1] * np.cos(tilt) - loc[:, 2] * np.sin(tilt) * 3.0  # exaggerate relief x3
        x = loc[:, 0]
        s = 0.9 * min(half / (np.ptp(x) + 1e-9), h / (np.ptp(y) + 1e-9))
        px = np.column_stack([(x - x.min()) * s + 0.05 * half, (y - y.min()) * s + 0.05 * h])
        z = loc[:, 2]
        zn = (z - z.min()) / (np.ptp(z) + 1e-9)
        cols = cv2.applyColorMap((zn * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO)[
            :, 0
        ]
        for a, b, cc in result.mesh_tris:
            for i, j in ((a, b), (b, cc)):
                col = tuple(int(v) for v in cols[i])
                cv2.line(
                    canvas, tuple(px[i].astype(int)), tuple(px[j].astype(int)), col, 1, cv2.LINE_AA
                )
        cv2.putText(
            canvas,
            "3D page (height x3, colour = height)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
    if result.flat is not None:
        f = result.flat
        s = min((w - half - 20) / f.shape[1], (h - 40) / f.shape[0])
        small = cv2.resize(
            f, (int(f.shape[1] * s), int(f.shape[0] * s)), interpolation=cv2.INTER_AREA
        )
        y0 = 30
        canvas[y0 : y0 + small.shape[0], half + 10 : half + 10 + small.shape[1]] = small
        cv2.putText(
            canvas,
            "flattened from 3D (ARAP)",
            (half + 10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
    return canvas


def save_recon(result: Flattening, out_dir: Path, stem: str) -> list[Path]:
    """Point cloud, page mesh, correspondence map and preview (whatever is available)."""
    written: list[Path] = []
    rec = result.rec
    if rec is not None and rec.points:
        ids, pts = rec.point_array()
        written.append(write_ply_points(out_dir / f"{stem}_points.ply", pts, rec.colors(ids)))
    if result.ok and result.mesh_xyz is not None and result.mesh_tris is not None:
        written.append(
            write_ply_mesh(out_dir / f"{stem}_mesh.ply", result.mesh_xyz, result.mesh_tris)
        )
        out = out_dir / f"{stem}_recon3d.jpg"
        cv2.imwrite(str(out), preview(result))
        written.append(out)
    if result.map_x is not None and result.map_y is not None:
        out = out_dir / f"{stem}_flatmap.npz"
        np.savez_compressed(out, map_x=result.map_x, map_y=result.map_y)
        written.append(out)
    return written
