"""Synthetic phone videos of a document on a desk, with per-frame ground truth.

A virtual pinhole camera (:mod:`findoc.synth.camera`) sweeps over a desk with the document
lying on it, the way a person waves a phone over a cheque:

* **Trajectory**: the camera centre moves on a smooth arc above the document (lateral sweep plus
  a slow orbit, so the viewing angle changes by up to ~25 degrees), always looking near the
  document centre, plus low-frequency **hand shake**.
* **Rendering**: desk and document are coplanar (Z = 0), so each frame is the plane-induced
  homography applied to one large "desk canvas" with the document pasted in the middle.
* **Motion blur**: the shutter stays open for part of the frame interval; we render several
  sub-frames along the camera path during that time and average them. Blur is therefore
  physically tied to how fast the camera moves, as in a real phone.
* **Ground truth** per frame: document corners in the image, whether the document is fully in
  view, and the blur length (how far the document centre moves while the shutter is open).

Phase 5 reuses the camera model with a curved document instead of a flat one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import cv2
import numpy as np

from findoc.synth.camera import Intrinsics, look_at, plane_homography, project
from findoc.synth.cheque import ChequeTruth, render_cheque
from findoc.synth.surface import Bend, DocSurface, render_surface


@dataclass
class VideoTruth:
    fps: float
    size: tuple[int, int]  # (width, height)
    corners: list[list[list[float]]]  # per frame: 4 x (x, y), order tl, tr, br, bl
    in_frame: list[bool]  # document fully inside the frame
    blur_px: list[float]  # motion-blur length at the document centre
    camera_centers: list[list[float]]  # world coordinates
    meta: dict = field(default_factory=dict)


def desk_texture(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """A wood-like desk: base colour, grain stripes, blotches and fine noise (gives features)."""
    base = rng.uniform(60, 140, size=3)
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    grain = np.sin(ys * 0.05 + 6 * np.sin(xs * 0.004) + rng.uniform(0, 6)) * 14
    img += grain[..., None]
    for _ in range(60):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = rng.uniform(10, 60)
        cv2.circle(
            img,
            (int(cx), int(cy)),
            int(r),
            tuple(float(v) for v in base * rng.uniform(0.75, 1.2, 3)),
            -1,
        )
    img = cv2.GaussianBlur(img, (0, 0), 3)
    img += rng.normal(0, 6, size=(h, w, 1)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _canvas(
    document: np.ndarray, rng: np.random.Generator, margin: float = 1.2
) -> tuple[np.ndarray, np.ndarray]:
    """Desk canvas with the document in the middle; returns canvas and canvas->world transform."""
    dh, dw = document.shape[:2]
    cw, ch = int(dw * (1 + 2 * margin)), int(dh * (1 + 2 * margin))
    canvas = desk_texture(rng, cw, ch)
    x0, y0 = (cw - dw) // 2, (ch - dh) // 2
    canvas[y0 : y0 + dh, x0 : x0 + dw] = document
    # canvas pixel (u, v) -> world (x, y): centred on the document centre
    canvas_to_world = np.array([[1, 0, -(x0 + dw / 2)], [0, 1, -(y0 + dh / 2)], [0, 0, 1]], float)
    return canvas, canvas_to_world


def add_thumb(
    canvas: np.ndarray, doc_box: tuple[int, int, int, int], rng: np.random.Generator
) -> None:
    """Paint a thumb holding the document by its bottom-left corner (no fields there). It hides
    the corner itself, which breaks edge-based page detection but leaves most texture visible."""
    x0, y0, dw, dh = doc_box
    cx = x0 + int(dw * rng.uniform(0.02, 0.06))
    cy = y0 + dh  # on the bottom edge, overhanging the corner
    axes = (int(dw * 0.065), int(dh * 0.25))
    tone = np.array([150, 175, 215]) + rng.integers(-20, 20, 3)  # BGR skin tone
    cv2.ellipse(canvas, (cx, cy), axes, float(rng.uniform(-30, 0)), 0, 360, tone.tolist(), -1)
    nail = (cx, cy - int(axes[1] * 0.6))
    cv2.ellipse(
        canvas, nail, (int(axes[0] * 0.6), int(axes[1] * 0.25)), 0, 0, 360, (190, 205, 235), -1
    )
    canvas[:] = cv2.GaussianBlur(canvas, (3, 3), 0)


def _smooth_noise(rng: np.random.Generator, n: int, scale: float, smooth: int = 15) -> np.ndarray:
    """Low-frequency random wobble (hand shake) of length n."""
    raw = rng.normal(0, 1, n + 2 * smooth)
    kernel = np.hanning(2 * smooth + 1)
    out = np.convolve(raw, kernel / kernel.sum(), mode="same")[smooth : smooth + n]
    return out / (np.std(out) + 1e-9) * scale


def camera_path(
    rng: np.random.Generator,
    n: int,
    doc_size: tuple[int, int],
    fill: float,
    hfov_deg: float,
    sweep: float = 0.35,
    tilt_deg: float = 22.0,
    shake: float = 0.01,
) -> list[np.ndarray]:
    """Camera centres (world) for n frames: an arc above the document."""
    dw, dh = doc_size
    # Distance so the document's width covers `fill` of the frame width when seen head-on.
    dist = (dw / fill) / (2 * np.tan(np.deg2rad(hfov_deg) / 2))
    s = np.linspace(-1, 1, n)
    phase = rng.uniform(0, 2 * np.pi)
    lateral = sweep * dw * s  # left-to-right sweep
    depth = 0.15 * dh * np.sin(np.pi * s + phase)  # small forward/back drift
    tilt = np.tan(np.deg2rad(tilt_deg)) * dist  # max sideways offset that gives `tilt_deg`
    lateral = np.clip(lateral, -tilt, tilt)
    centers = []
    sx, sy, sz = (_smooth_noise(rng, n, shake * dist) for _ in range(3))
    for i in range(n):
        centers.append(
            np.array([lateral[i] + sx[i], depth[i] + sy[i], -dist * (1 + 0.08 * s[i] ** 2) + sz[i]])
        )
    return centers


def render_video(
    document: np.ndarray,
    rng: np.random.Generator,
    seconds: float = 4.0,
    fps: float = 30.0,
    size: tuple[int, int] = (1280, 720),
    fill: float = 0.6,
    hfov_deg: float = 62.0,
    shutter: float = 0.5,
    sweep: float = 0.35,
    shake: float = 0.01,
    exit_frame: bool = False,
    thumb: bool = False,
    jerks: int = 2,
    subframes: int = 5,
    bend: Bend | None = None,
    mesh_size: tuple[int, int] = (64, 30),
) -> tuple[list[np.ndarray], VideoTruth]:
    """Render frames of a camera sweep over ``document`` lying on a desk.

    ``shutter`` is the fraction of each frame interval the shutter is open (0 = no blur).
    ``exit_frame`` pans the camera away mid-clip so the document leaves the view; ``thumb``
    paints a thumb holding the document over one bottom corner (occludes edge and corner);
    ``jerks`` is the number of quick wrist flicks (short bursts of heavy motion blur).
    ``bend`` makes the document a curved, developable surface (:mod:`findoc.synth.surface`),
    rendered as a shaded mesh; ground truth then also holds the camera matrix, every frame's
    pose and the document's 3D mesh (``truth.meta``).
    """
    w, h = size
    n = int(round(seconds * fps))
    K = Intrinsics.for_frame(w, h, hfov_deg).K  # noqa: N806
    dh, dw = document.shape[:2]
    canvas, canvas_to_world = _canvas(document, rng, margin=1.8 if exit_frame else 1.2)
    surface = None
    if bend is not None and bend.kind != "flat":
        surface = DocSurface(dw, dh, bend)
        st, world, tris = surface.mesh(*mesh_size)
        x0, y0 = int(-canvas_to_world[0, 2] - dw / 2), int(-canvas_to_world[1, 2] - dh / 2)
        canvas[y0 : y0 + dh, x0 : x0 + dw] = desk_texture(rng, dw, dh)  # desk under the sheet
    if thumb:
        cx0 = int(-canvas_to_world[0, 2] - dw / 2)
        cy0 = int(-canvas_to_world[1, 2] - dh / 2)
        add_thumb(canvas, (cx0, cy0, dw, dh), rng)
    centers = camera_path(rng, n + 1, (dw, dh), fill, hfov_deg, sweep=sweep, shake=shake)
    look_jitter = [_smooth_noise(rng, n + 1, 0.02 * dw) for _ in range(2)]
    if exit_frame:  # pan away in the middle of the clip so the document leaves the view
        s = np.linspace(-1, 1, n + 1)
        look_jitter[0] = look_jitter[0] + 1.6 * dw * np.exp(-((s - 0.1) ** 2) / 0.02)
    # Hand jerks: a quick flick of the wrist turns the phone for ~2-3 frames, sweeping the
    # whole image sideways and blurring those frames heavily.
    idx = np.arange(n + 1, dtype=float)
    for _ in range(jerks):
        t0 = rng.uniform(0.1 * n, 0.9 * n)
        pulse = np.exp(-((idx - t0) ** 2) / (2 * 1.2**2))
        look_jitter[int(rng.integers(0, 2))] += 0.12 * dw * np.sign(rng.normal()) * pulse
    roll = _smooth_noise(rng, n + 1, 2.0)
    up = np.array([0.0, -1.0, 0.0])

    def pose_at(u: float):  # type: ignore[no-untyped-def]
        """Pose at fractional frame index u (linear interpolation between frame poses)."""
        i = min(int(u), n - 1)
        a = u - i
        c = (1 - a) * centers[i] + a * centers[i + 1]
        tgt = np.array(
            [(1 - a) * look_jitter[0][i] + a * look_jitter[0][i + 1],
             (1 - a) * look_jitter[1][i] + a * look_jitter[1][i + 1], 0.0]
        )  # fmt: skip
        return look_at(c, tgt, up, roll_deg=(1 - a) * roll[i] + a * roll[i + 1])

    doc_corners = np.array(
        [[-dw / 2, -dh / 2, 1], [dw / 2, -dh / 2, 1], [dw / 2, dh / 2, 1], [-dw / 2, dh / 2, 1]],
        float,
    )
    if surface is not None:  # texture corners tl, tr, br, bl -> world, centred like the canvas
        tex_corners = np.array([[0, 0], [dw - 1, 0], [dw - 1, dh - 1], [0, dh - 1]], float)
        surface_corners = to_world(surface, tex_corners)
        world = to_world(surface, st)
    frames: list[np.ndarray] = []
    truth = VideoTruth(fps, (w, h), [], [], [], [], meta={"hfov_deg": hfov_deg, "shutter": shutter})
    for i in range(n):
        times = (
            [i + shutter * k / max(subframes - 1, 1) for k in range(subframes)]
            if shutter > 0
            else [float(i)]
        )
        acc = np.zeros((h, w, 3), np.float32)
        for u in times:
            pose_u = pose_at(u)
            H = plane_homography(K, pose_u) @ canvas_to_world  # noqa: N806 - canvas -> image
            img = cv2.warpPerspective(
                canvas, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
            if surface is not None:
                img, _ = render_surface(document, st, world, tris, K, pose_u, img)
            acc += img
        frame = acc / len(times)
        frame += rng.normal(0, 2.5, size=frame.shape).astype(np.float32)
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))

        mid = pose_at(i + shutter / 2)
        if surface is not None:
            pts = project(K, mid, surface_corners)
        else:
            Hw = plane_homography(K, mid)  # noqa: N806 - world plane -> image
            pts = (Hw @ doc_corners.T).T
            pts = pts[:, :2] / pts[:, 2:3]
        truth.corners.append(pts.round(3).tolist())
        truth.in_frame.append(bool(np.all((pts >= 0) & (pts < [w - 1, h - 1]))))
        c0 = plane_homography(K, pose_at(i)) @ np.array([0, 0, 1.0])
        c1 = plane_homography(K, pose_at(i + shutter)) @ np.array([0, 0, 1.0])
        truth.blur_px.append(float(np.linalg.norm(c0[:2] / c0[2] - c1[:2] / c1[2])))
        truth.camera_centers.append(mid.center.round(3).tolist())
        truth.meta.setdefault("poses", []).append(
            {"R": mid.R.round(8).tolist(), "t": mid.t.round(5).tolist()}
        )
    truth.meta["K"] = K.round(6).tolist()
    if surface is not None:
        truth.meta["bend"] = asdict(bend)
        truth.meta["lift"] = round(surface.lift(), 2)
    return frames, truth


def to_world(surface: DocSurface, st: np.ndarray) -> np.ndarray:
    """Texture points of a bent document -> world points, centred on the origin like the canvas
    (the flat document spans [-w/2, w/2] x [-h/2, h/2] at Z = 0) and resting on the desk."""
    p = surface.points(st)
    grid_z = surface.points(surface.grid(48, 24)[0])[:, 2]
    p[:, 2] -= grid_z.max()
    return p


def write_video(path: Path, frames: list[np.ndarray], fps: float) -> Path:
    """Write frames as an MP4 (MPEG-4 Part 2, 'mp4v': plays in VLC and most desktop players)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {path}")
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def save_truth(path: Path, truth: VideoTruth) -> None:
    path.write_text(json.dumps(asdict(truth)), encoding="utf-8")


def cheque_video(
    seed: int,
    fraud: str | None = None,
    today: date | None = None,
    bank: str | None = None,
    **render_kw,
) -> tuple[list[np.ndarray], VideoTruth, ChequeTruth]:
    """One synthetic hand-held video of a cheque on a desk, with ground truth."""
    from findoc.synth.dataset import FRAUD_KINDS

    rng = np.random.default_rng(seed)
    flags = {k: fraud == k for k in FRAUD_KINDS}
    cheque, cheque_truth = render_cheque(rng, bank=bank, today=today, **flags)
    frames, truth = render_video(cheque, rng, **render_kw)
    truth.meta["seed"] = seed
    return frames, truth, cheque_truth


def generate_videos(
    n: int,
    out_dir: Path,
    seed: int = 0,
    fraud_rate: float = 0.4,
    today: date | None = None,
    **render_kw,
) -> list[Path]:
    """Write ``n`` cheque videos (``video_XXXX.mp4``), per-video ``*_truth.json`` (camera and
    corner ground truth) and ``labels.json`` (cheque field ground truth, as for photos)."""
    from findoc.synth.dataset import FRAUD_KINDS

    out_dir.mkdir(parents=True, exist_ok=True)
    picker = np.random.default_rng(seed + 99_991)
    labels: dict[str, dict] = {}
    paths = []
    for i in range(n):
        fraud = str(picker.choice(FRAUD_KINDS)) if picker.random() < fraud_rate else None
        frames, truth, cheque_truth = cheque_video(seed + i, fraud, today, **render_kw)
        name = f"video_{seed + i:04d}.mp4"
        write_video(out_dir / name, frames, truth.fps)
        save_truth(out_dir / f"video_{seed + i:04d}_truth.json", truth)
        labels[name] = asdict(cheque_truth)
        paths.append(out_dir / name)
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return paths
