"""Phase 3: camera model, synthetic video, KLT tracking, keyframes, video I/O and stages."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from findoc.cli import app
from findoc.io import read_frames, read_video, video_info
from findoc.pipeline import process
from findoc.synth.camera import Intrinsics, look_at, plane_homography, project
from findoc.synth.cheque import render_cheque
from findoc.synth.video import cheque_video, generate_videos, render_video, write_video
from findoc.track.keyframes import (
    best_frame,
    frontalness,
    select_keyframes,
    usable_frames,
    viewpoint_change,
)
from findoc.track.klt import FrameTrack, TrackResult, draw_track, quad_mask, track_document
from findoc.track.stages import track_stage
from findoc.track.viz import contact_sheet
from findoc.types import Document, InputKind, Status

SMALL = dict(seconds=1.5, fps=20.0, size=(640, 360), subframes=3)
TODAY = date(2026, 11, 17)


@pytest.fixture(scope="module")
def clip() -> tuple[list[np.ndarray], object]:
    rng = np.random.default_rng(3)
    cheque, _ = render_cheque(rng, bank="KVRY", today=TODAY)
    return render_video(cheque, rng, **SMALL)


@pytest.fixture(scope="module")
def clip_path(tmp_path_factory: pytest.TempPathFactory, clip) -> Path:  # type: ignore[no-untyped-def]
    frames, truth = clip
    return write_video(tmp_path_factory.mktemp("vid") / "clip.mp4", frames, truth.fps)


# ---------------------------------------------------------------- camera model


def test_intrinsics_from_fov() -> None:
    k = Intrinsics.for_frame(1280, 720, hfov_deg=90.0)
    assert k.f == pytest.approx(640.0)
    assert k.K[0, 2] == 640 and k.K[1, 2] == 360


def test_plane_homography_matches_projection() -> None:
    k = Intrinsics.for_frame(800, 600).K
    pose = look_at(np.array([0.1, -0.05, -0.6]), np.zeros(3), np.array([0, -1.0, 0]), roll_deg=5)
    pts = np.array([[0.0, 0, 0], [0.1, 0.05, 0], [-0.08, 0.03, 0]])
    h = plane_homography(k, pose)
    via_h = cv2.perspectiveTransform(pts[:, None, :2].astype(np.float64), h).reshape(-1, 2)
    np.testing.assert_allclose(via_h, project(k, pose, pts), atol=1e-6)
    assert pose.center == pytest.approx([0.1, -0.05, -0.6])


# ---------------------------------------------------------------- synthetic video


def test_render_video_ground_truth(clip) -> None:  # type: ignore[no-untyped-def]
    frames, truth = clip
    n = int(SMALL["seconds"] * SMALL["fps"])
    assert len(frames) == n == len(truth.corners) == len(truth.blur_px)
    assert frames[0].shape == (360, 640, 3)
    assert all(truth.in_frame)
    assert max(truth.blur_px) > 3 * np.median(truth.blur_px)  # jerks produce blur bursts


def test_exit_frame_takes_document_out_of_view() -> None:
    rng = np.random.default_rng(1)
    cheque, _ = render_cheque(rng, today=TODAY)
    _, truth = render_video(cheque, rng, exit_frame=True, jerks=0, **SMALL)
    assert not all(truth.in_frame)


def test_generate_videos_writes_truth(tmp_path: Path) -> None:
    paths = generate_videos(1, tmp_path, seed=5, today=TODAY, **SMALL)
    assert paths[0].exists()
    labels = json.loads((tmp_path / "labels.json").read_text())
    assert paths[0].name in labels and "bank" in labels[paths[0].name]
    truth = json.loads((tmp_path / "video_0005_truth.json").read_text())
    assert len(truth["corners"]) == 30


def test_cheque_video_fraud_flag() -> None:
    _, truth, cheque_truth = cheque_video(2, "stale", today=TODAY, **SMALL)
    assert truth.meta["seed"] == 2
    assert cheque_truth.fraud["stale"]


# ---------------------------------------------------------------- tracking


def test_tracking_follows_ground_truth(clip) -> None:  # type: ignore[no-untyped-def]
    frames, truth = clip
    result = track_document(frames)
    assert result.lost == 0
    gt = np.array(truth.corners)
    diag = np.hypot(640, 360)
    errs = [np.linalg.norm(f.corners - gt[f.index], axis=1).mean() / diag for f in result.frames]
    assert np.mean(errs) < 0.01  # within 1% of the frame diagonal on average


def test_tracking_from_given_corners(clip) -> None:  # type: ignore[no-untyped-def]
    frames, truth = clip
    result = track_document(frames[:10], start_corners=np.array(truth.corners[0]))
    assert len(result.frames) == 10 and result.frames[0].index == 0


def test_tracking_recovers_after_cut(clip) -> None:  # type: ignore[no-untyped-def]
    frames, _ = clip
    blank = np.full_like(frames[0], 90)
    result = track_document([*frames[:5], blank, blank, *frames[5:12]])
    assert result.frames[5].corners is None and result.frames[6].corners is None
    assert result.frames[-1].corners is not None
    assert result.reacquisitions >= 1


def test_quad_mask_and_draw() -> None:
    corners = np.array([[10, 10], [90, 10], [90, 50], [10, 50]], np.float32)
    mask = quad_mask((60, 100), corners)
    assert mask[30, 50] == 255 and mask[2, 2] == 0
    ft = FrameTrack(3, None, 0, 0.0, 10.0, 0.0, False)
    out = draw_track(np.zeros((60, 100, 3), np.uint8), ft)
    assert out.shape == (60, 100, 3)


# ---------------------------------------------------------------- keyframes


def test_viewpoint_change_ignores_similarity() -> None:
    a = np.array([[0, 0], [200, 0], [200, 90], [0, 90]], np.float64)
    rot = cv2.getRotationMatrix2D((100, 45), 20, 1.3)
    b = cv2.transform(a[:, None], rot).reshape(-1, 2) + [40, -15]
    assert viewpoint_change(a, b) < 1e-3
    keystone = np.array([[20, 0], [180, 0], [200, 90], [0, 90]], np.float64)
    assert viewpoint_change(a, keystone) > 0.02


def test_frontalness() -> None:
    rect = np.array([[0, 0], [200, 0], [200, 90], [0, 90]], np.float64)
    assert frontalness(rect) == pytest.approx(1.0)
    assert frontalness(np.array([[30, 0], [170, 0], [200, 90], [0, 90]], np.float64)) < 0.8


def test_keyframes_avoid_blur(clip) -> None:  # type: ignore[no-untyped-def]
    frames, truth = clip
    sel = select_keyframes(track_document(frames), min_keyframes=4)
    idx = [k.index for k in sel.keyframes]
    assert idx == sorted(idx) and len(idx) >= 4
    blur = np.array(truth.blur_px)
    assert blur[idx].max() < blur.max()
    assert sel.best is not None and blur[sel.best] <= np.median(blur) * 2


def test_selection_with_no_usable_frames() -> None:
    lost = TrackResult([FrameTrack(i, None, 0, 0.0, 0.0, 0.0, False) for i in range(5)])
    assert usable_frames(lost, 100.0) == []
    sel = select_keyframes(lost)
    assert sel.keyframes == [] and sel.best is None
    assert best_frame([]) is None


def test_track_stage_rejects_unusable_video(monkeypatch: pytest.MonkeyPatch) -> None:
    import findoc.track.stages as stages

    lost = TrackResult([FrameTrack(i, None, 0, 0.0, 0.0, 0.0, False) for i in range(3)])
    monkeypatch.setattr(stages, "track_document", lambda frames: lost)
    doc = Document(source=Path("x.mp4"), kind=InputKind.VIDEO)
    doc.frames = [np.zeros((10, 10, 3), np.uint8)] * 3
    doc.frame_numbers = [0, 1, 2]
    doc = track_stage(doc)
    assert doc.status is Status.REJECTED_QUALITY
    assert doc.tracking["lost"] == 3
    assert any("No document outline" in m for m in doc.messages)


def test_rejection_reasons_explain_the_cause() -> None:
    from findoc.track.stages import rejection_reasons

    c = np.array([[0, 0], [99, 0], [99, 59], [0, 59]], np.float32)  # touches the border
    blank = TrackResult([FrameTrack(i, c, 5, 0.0, 50.0, 1.0, False) for i in range(4)])
    msgs = " ".join(rejection_reasons(blank))
    assert "too little detail" in msgs and "never fits fully" in msgs
    blurred = TrackResult([FrameTrack(i, c, 300, 0.0, 5.0, 0.5, True) for i in range(4)])
    assert "blurred" in rejection_reasons(blurred)[-1]


def test_partial_document_is_not_tracked() -> None:
    """A page wider than the frame (only ever seen cut off) must not be tracked as a document."""
    from findoc.track.stages import rejection_reasons

    frames = []
    for shift in range(0, 60, 6):
        img = np.full((360, 640, 3), 60, np.uint8)
        cv2.rectangle(img, (200 - shift, 80), (900, 280), (235, 235, 235), -1)  # runs off the right
        cv2.putText(
            img, "PAY RUPEES", (260 - shift, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 3
        )
        frames.append(img)
    result = track_document(frames)
    assert result.start_problem == "partial" and result.lost == len(frames)
    assert "landscape" in " ".join(rejection_reasons(result))


def test_outline_on_the_border_is_not_in_view() -> None:
    from findoc.track.klt import _coverage_and_inside

    _, inside = _coverage_and_inside(
        np.array([[0, 0], [99, 0], [99, 59], [0, 59]], float), (60, 100)
    )
    assert not inside
    _, inside = _coverage_and_inside(
        np.array([[10, 10], [90, 10], [90, 50], [10, 50]], float), (60, 100)
    )
    assert inside


# ---------------------------------------------------------------- video I/O and pipeline


def test_video_io(clip_path: Path) -> None:
    info = video_info(clip_path)
    assert info["frames"] == 30 and info["width"] == 640
    frames, numbers, scale = read_video(clip_path, max_frames=10, max_side=320)
    assert len(frames) == 10 and numbers[:3] == [0, 3, 6]
    assert scale == pytest.approx(0.5) and frames[0].shape == (180, 320, 3)
    full = read_frames(clip_path, [4, 29])
    assert set(full) == {4, 29} and full[4].shape == (360, 640, 3)


def test_video_info_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(ValueError):
        video_info(bad)


def test_pipeline_on_video(clip_path: Path) -> None:
    doc = process(clip_path, ocr=False)
    assert doc.status is Status.OK, doc.messages
    assert doc.detection_method == "tracked"
    assert doc.best_frame is not None and doc.keyframes
    assert doc.flat is not None  # the 2D stages ran on the best frame
    assert len(doc.keyframe_images) == len(doc.keyframes)
    result = doc.result()
    assert "frames" not in result and "keyframe_images" not in result
    assert len(result["track"]) == 30 and result["tracking"]["lost"] == 0


def test_cli_video(tmp_path: Path) -> None:
    runner = CliRunner()
    res = runner.invoke(
        app, ["synth", "--kind", "video", "--n", "1", "--seconds", "1", "-o", str(tmp_path)]
    )
    assert res.exit_code == 0, res.output
    video = tmp_path / "video_0000.mp4"
    out = tmp_path / "out"
    res = runner.invoke(app, ["process", str(video), "--no-ocr", "-s", str(out)])
    assert res.exit_code == 0, res.output
    assert "Tracking:" in res.output
    assert "Cheque template:" in res.output  # 1280x720: large enough to match the template
    names = {p.name for p in out.iterdir()}
    assert {"video_0000_tracking.mp4", "video_0000_keyframes.jpg", "video_0000_best.jpg"} <= names


def test_contact_sheet() -> None:
    assert contact_sheet([], []).shape[1] == 360
    imgs = [np.zeros((50, 100, 3), np.uint8)] * 5
    sheet = contact_sheet(imgs, [str(i) for i in range(5)], cols=2, width=100)
    assert sheet.shape == (150, 200, 3)
