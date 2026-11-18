"""Phase 5: developable surfaces, rendering, SfM, surface fitting, ARAP flattening, unwarping."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

from findoc.recon3d.export import preview, save_recon, write_ply_mesh, write_ply_points
from findoc.recon3d.flatten import arap_flatten, cast_rays, fit_surface, grid_in_quad, unwarp
from findoc.recon3d.pipeline import Flattening, flatten_document, plane_residual
from findoc.recon3d.sfm import build_tracks, extract, reconstruct, triangulate
from findoc.recon3d.stages import recon3d_stage
from findoc.synth.camera import Intrinsics, look_at, project
from findoc.synth.cheque import render_cheque
from findoc.synth.surface import Bend, DocSurface, render_surface
from findoc.synth.video import render_video
from findoc.types import Document, InputKind

TODAY = date(2026, 11, 17)


@pytest.fixture(scope="module")
def cheque() -> np.ndarray:
    return render_cheque(np.random.default_rng(4), bank="KVRY", today=TODAY)[0]


@pytest.fixture(scope="module")
def curl_clip(cheque):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(9)
    return render_video(
        cheque, rng, seconds=4, fps=3, size=(960, 540), subframes=1, shutter=0, jerks=0,
        bend=Bend("curl", radius=600, start=0.3), mesh_size=(40, 20),
    )  # fmt: skip


# ------------------------------------------------------------------ surfaces and rendering


@pytest.mark.parametrize(
    "bend",
    [Bend("flat"), Bend("curl", radius=500), Bend("curl2", radius=700, start=0.4),
     Bend("wave", amplitude=0.002, wavelength=500), Bend("curl", angle_deg=25, radius=600)],
)  # fmt: skip
def test_surfaces_are_isometric(bend: Bend) -> None:
    s = DocSurface(800, 400, bend)
    st, world, tris = s.mesh(30, 15)
    for a, b in ((0, 1), (1, 2), (0, 2)):
        d3 = np.linalg.norm(world[tris[:, a]] - world[tris[:, b]], axis=1)
        d2 = np.linalg.norm(st[tris[:, a]] - st[tris[:, b]], axis=1)
        assert np.abs(d3 / d2 - 1).max() < 2e-3  # paper does not stretch
    assert world[:, 2].max() == pytest.approx(0.0, abs=1e-6)  # resting on the desk
    assert (s.lift() > 1) == (bend.kind != "flat")


def test_fold_lifts_one_side() -> None:
    s = DocSurface(800, 400, Bend("fold", fold_deg=30, start=0.0))
    _, world, _ = s.mesh(30, 15)
    assert world[:, 2].min() < -100  # towards the camera (negative Z)


def test_render_surface(cheque) -> None:  # type: ignore[no-untyped-def]
    k = Intrinsics.for_frame(640, 360).K
    pose = look_at(np.array([0.0, 0.0, -2500.0]), np.zeros(3), np.array([0.0, -1.0, 0.0]))
    st, world, tris = DocSurface(cheque.shape[1], cheque.shape[0], Bend("curl", radius=600)).mesh(
        20, 10
    )
    world = world - [cheque.shape[1] / 2, cheque.shape[0] / 2, 0]
    img, mask = render_surface(cheque, st, world, tris, k, pose, np.zeros((360, 640, 3), np.uint8))
    assert img.shape == (360, 640, 3) and 0.1 < mask.mean() < 0.9
    assert img[mask].mean() > 100  # the (light) cheque was drawn


def test_curved_video_ground_truth(curl_clip) -> None:  # type: ignore[no-untyped-def]
    frames, truth = curl_clip
    assert len(truth.meta["poses"]) == len(frames) and truth.meta["lift"] > 50
    assert truth.meta["bend"]["kind"] == "curl" and len(truth.meta["K"]) == 3


# ------------------------------------------------------------------ SfM


def test_triangulate_and_tracks() -> None:
    k = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1.0]])
    x = np.array([0.1, -0.2, 5.0])
    p1 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = k @ np.hstack([np.eye(3), np.array([[-1.0], [0], [0]])])
    obs = [(p @ np.append(x, 1))[:2] / (p @ np.append(x, 1))[2] for p in (p1, p2)]
    assert np.allclose(triangulate([p1, p2], obs), x, atol=1e-6)
    tracks = build_tracks(3, {(0, 1): np.array([[0, 5], [1, 6]]), (1, 2): np.array([[5, 9]])})
    assert {0: 0, 1: 5, 2: 9} in tracks and len(tracks) == 2


def test_sfm_on_curved_clip(curl_clip) -> None:  # type: ignore[no-untyped-def]
    frames, truth = curl_clip
    imgs = frames[::2]
    rec = reconstruct(imgs)
    assert rec is not None and len(rec.registered()) >= len(imgs) - 1
    assert rec.reproj_px < 1.0 and len(rec.points) > 200
    true_f = np.array(truth.meta["K"])[0, 0]
    assert abs(rec.focal / true_f - 1) < 0.2  # focal length recovered from images alone


def test_plane_residual_separates_flat_and_curved(cheque) -> None:  # type: ignore[no-untyped-def]
    out = {}
    for kind, bend in (("flat", None), ("curl", Bend("curl", radius=500, start=0.2))):
        frames, truth = render_video(
            cheque, np.random.default_rng(2), seconds=2, fps=4, size=(960, 540), subframes=1,
            shutter=0, jerks=0, bend=bend, mesh_size=(40, 20),
        )  # fmt: skip
        views = extract(frames[::2])
        out[kind] = plane_residual(views, 1, np.array(truth.corners[2], np.float32))
    assert out["flat"] < 1.3 < out["curl"]


# ------------------------------------------------------------------ surface, ARAP, unwarp


def test_arap_recovers_true_flat_layout() -> None:
    s = DocSurface(800, 400, Bend("curl", radius=350, start=0.1))
    st, world, tris = s.mesh(32, 16)
    uv, stretch = arap_flatten(world, tris, world[:, :2])
    assert stretch < 1e-3
    a, b = uv - uv.mean(0), st - st.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)
    err = np.linalg.norm(a @ (u @ vt).T - b, axis=1)
    assert err.mean() < 0.5  # px: same layout as the original page up to a rotation
    naive = np.linalg.norm(world[:, :2] - world[:, :2].mean(0) - b, axis=1).mean()
    assert naive > 10  # just projecting onto the desk is far off


def test_arap_rejects_broken_input() -> None:
    world = np.zeros((4, 3))
    world[1, 0] = np.nan
    _, stretch = arap_flatten(world, np.array([[0, 1, 2], [1, 3, 2]]), np.zeros((4, 2)))
    assert stretch == float("inf")


def test_fit_surface_and_cast_rays() -> None:
    rng = np.random.default_rng(0)
    ab = rng.uniform(-1, 1, (400, 2))
    pts = np.column_stack([ab, 0.2 * ab[:, 0] ** 2])  # a cylinder-like bend
    fit = fit_surface(pts)
    assert fit.planarity > 3e-3
    flat = fit_surface(np.column_stack([ab, rng.normal(0, 1e-4, 400)]))
    assert flat.planarity < 1e-3
    k = np.array([[500, 0, 0], [0, 500, 0], [0, 0, 1.0]])
    rot, tvec = np.eye(3), np.array([0, 0, 5.0])  # camera 5 units in front of the surface
    hit = cast_rays(fit, k, rot, tvec, np.array([[0.0, 0.0], [100.0, 50.0]]))
    back = (hit @ rot.T + tvec) @ k.T
    assert np.allclose(back[:, :2] / back[:, 2:], [[0, 0], [100, 50]], atol=1e-6)


def test_unwarp_identity(cheque) -> None:  # type: ignore[no-untyped-def]
    h, w = cheque.shape[:2]
    quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)
    pts, tris = grid_in_quad(quad, 10, 6)
    flat, mx, my = unwarp(cheque, pts, pts.copy(), tris, (0, 9, 59, 50), scale=1.0)
    assert abs(flat.shape[1] - w) <= 2 and abs(flat.shape[0] - h) <= 2
    assert (
        np.abs(
            flat.astype(int)[5:-5, 5:-5] - cheque[5 : flat.shape[0] - 5, 5 : flat.shape[1] - 5]
        ).mean()
        < 3
    )
    assert mx.shape == flat.shape[:2]


# ------------------------------------------------------------------ pipeline, export, stage


def test_flatten_document_end_to_end(curl_clip, cheque, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    frames, truth = curl_clip
    imgs = frames[::2]
    res = flatten_document(imgs, 1, np.array(truth.corners[2], np.float32))
    assert res.ok, res.reason
    assert res.flat is not None and res.flat.shape[1] > res.flat.shape[0]  # cheque-shaped
    assert res.stats["stretch"] < 0.06 and res.stats["registered"] >= 4
    paths = save_recon(res, tmp_path, "c")
    names = {p.name for p in paths}
    assert {"c_points.ply", "c_mesh.ply", "c_recon3d.jpg", "c_flatmap.npz"} <= names
    assert preview(res).shape == (520, 1400, 3)


def test_flatten_document_keeps_flat_pages_2d(cheque) -> None:  # type: ignore[no-untyped-def]
    frames, truth = render_video(
        cheque, np.random.default_rng(3), seconds=2, fps=4, size=(960, 540), subframes=1,
        shutter=0, jerks=0,
    )  # fmt: skip
    res = flatten_document(frames[::2], 1, np.array(truth.corners[2], np.float32))
    assert not res.ok and res.planar and "flat" in res.reason
    assert flatten_document(frames[:2], 0, np.zeros((4, 2), np.float32)).reason.startswith("fewer")


def test_ply_writers(tmp_path: Path) -> None:
    p = write_ply_points(tmp_path / "p.ply", np.zeros((2, 3)), np.array([[0, 0, 255], [255, 0, 0]]))
    assert "element vertex 2" in p.read_text() and "255 0 0" in p.read_text()
    m = write_ply_mesh(tmp_path / "m.ply", np.zeros((3, 3)), np.array([[0, 1, 2]]))
    assert "element face 1" in m.read_text()


def test_recon3d_stage_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    from findoc.config import get_settings

    photo = Document(source=Path("a.jpg"), kind=InputKind.IMAGE)
    assert recon3d_stage(photo).recon3d == {}  # photos are not reconstructed
    video = Document(source=Path("a.mp4"), kind=InputKind.VIDEO)
    video.original = np.zeros((10, 10, 3), np.uint8)
    video.corners = [[0, 0], [9, 0], [9, 9], [0, 9]]
    out = recon3d_stage(video)
    assert out.recon3d["ok"] is False and "fewer than 3" in out.recon3d["reason"]
    monkeypatch.setenv("FINDOC_RECON3D", "off")
    get_settings.cache_clear()
    assert recon3d_stage(Document(source=Path("b.mp4"), kind=InputKind.VIDEO)).recon3d == {}


def test_flattening_result_defaults() -> None:
    f = Flattening(False, "x")
    assert f.flat is None and f.stats == {}
    _ = project  # imported helper used by other tests' fixtures
    _ = cv2


def _board_images(k: np.ndarray, n: int = 8) -> list[np.ndarray]:
    """Synthetic photos of a 9x6 checkerboard (10x7 squares) from different angles."""
    sq = 60
    board = np.full((7 * sq + 2 * sq, 10 * sq + 2 * sq), 255, np.uint8)
    for r in range(7):
        for c in range(10):
            if (r + c) % 2 == 0:
                board[sq + r * sq : sq + (r + 1) * sq, sq + c * sq : sq + (c + 1) * sq] = 0
    bh, bw = board.shape
    rng = np.random.default_rng(1)
    out = []
    for _ in range(n):
        eye = np.array([rng.uniform(-300, 300), rng.uniform(-200, 200), -rng.uniform(1000, 1400)])
        pose = look_at(
            eye, np.array([rng.uniform(-40, 40), rng.uniform(-40, 40), 0.0]), np.array([0, -1.0, 0])
        )
        from findoc.synth.camera import plane_homography

        hmat = plane_homography(k, pose) @ np.array([[1, 0, -bw / 2], [0, 1, -bh / 2], [0, 0, 1.0]])
        img = cv2.warpPerspective(board, hmat, (960, 540), borderValue=128)
        out.append(cv2.GaussianBlur(img, (3, 3), 0))
    return out


def test_calibration_recovers_focal_length(tmp_path: Path) -> None:
    from findoc.recon3d.calibrate import calibrate, save_focal

    k = Intrinsics.for_frame(960, 540, 65.0).K
    cal = calibrate(_board_images(k), (9, 6))
    assert abs(cal.focal_px / k[0, 0] - 1) < 0.03 and cal.rms_px < 1.0
    assert cal.hfov_deg == pytest.approx(65.0, abs=3.0)
    env = tmp_path / ".env"
    env.write_text("FINDOC_LOG_LEVEL=INFO\nFINDOC_FOCAL_PX=1\n")
    save_focal(cal.focal_px, env)
    text = env.read_text()
    assert text.count("FINDOC_FOCAL_PX") == 1 and "LOG_LEVEL" in text
    with pytest.raises(ValueError):
        calibrate([np.zeros((100, 100), np.uint8)] * 3)
